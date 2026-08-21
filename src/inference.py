import time

import numpy as np
import torch

from .data import get_feature_extractor
from .model import LightweightTurnModel, SmartTurnModel
from .utils import truncate_audio_to_last_n_seconds


class TurnDetector:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get("config", {})

        if (
            "conv_blocks" in checkpoint["model_state_dict"]
            or "conv_blocks.0.weight" in checkpoint["model_state_dict"]
        ):
            self.model = LightweightTurnModel()
        else:
            self.model = SmartTurnModel(
                base_model_name=config.get("model_name", "openai/whisper-tiny"),
                freeze_encoder=config.get("freeze_encoder", True),
                unfreeze_last_n=config.get("unfreeze_last_n", 0),
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.feature_extractor = get_feature_extractor()

    def predict(self, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
        start_time = time.time()

        audio = truncate_audio_to_last_n_seconds(audio_array, sample_rate, 8)
        features = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * sample_rate,
            truncation=True,
            do_normalize=True,
        )

        input_features = features.input_features.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_features)
            prob = outputs["probabilities"].item()

        inference_time_ms = (time.time() - start_time) * 1000

        return {
            "prediction": int(prob >= 0.5),
            "probability": prob,
            "inference_time_ms": inference_time_ms,
        }

    def predict_batch(self, audio_arrays: list[np.ndarray], sample_rate: int = 16000) -> list[dict]:
        return [self.predict(audio, sample_rate) for audio in audio_arrays]
