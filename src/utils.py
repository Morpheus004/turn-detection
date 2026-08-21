import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def truncate_audio_to_last_n_seconds(
    audio_array: np.ndarray, sample_rate: int = 16000, n_seconds: int = 8
) -> np.ndarray:
    target_length = sample_rate * n_seconds
    if len(audio_array) > target_length:
        return audio_array[-target_length:]
    elif len(audio_array) < target_length:
        return np.pad(audio_array, (target_length - len(audio_array), 0), "constant")
    return audio_array


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{seconds:.2f}s"
