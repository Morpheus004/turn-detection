import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor


def get_feature_extractor() -> WhisperFeatureExtractor:
    """Returns WhisperFeatureExtractor configured for 8-second chunks (800 frames)."""
    return WhisperFeatureExtractor(
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=8,
        n_fft=400,
        padding_value=0.0,
    )


def load_smart_turn_dataset(
    split: str = "train",
    train_dataset_name: str = "pipecat-ai/smart-turn-data-v3.2-train",
    test_dataset_name: str = "pipecat-ai/smart-turn-data-v3.2-test",
    val_split_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Loads smart-turn dataset.
    Supports 'train', 'validation', and 'test'.
    If 'validation' is requested, splits from the training dataset.
    """
    if split == "test":
        try:
            return load_dataset(test_dataset_name, split="train")
        except Exception:
            # Fallback to splitting training data
            full_train = load_dataset(train_dataset_name, split="train")
            splits = full_train.train_test_split(test_size=val_split_ratio, seed=seed)
            return splits["test"]

    elif split == "validation":
        try:
            return load_dataset(test_dataset_name, split="train")
        except Exception:
            full_train = load_dataset(train_dataset_name, split="train")
            splits = full_train.train_test_split(test_size=val_split_ratio, seed=seed)
            return splits["test"]

    else:  # 'train'
        try:
            full_train = load_dataset(train_dataset_name, split="train")
            splits = full_train.train_test_split(test_size=val_split_ratio, seed=seed)
            return splits["train"]
        except Exception:
            return load_dataset(train_dataset_name, split="train")


class AudioDataset(Dataset):
    def __init__(self, hf_dataset, feature_extractor=None):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor or get_feature_extractor()
        self.sample_rate = 16000
        self.n_seconds = 8

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]

        # Audio array extraction
        audio = np.array(item["audio"]["array"], dtype=np.float32)

        # Truncate to last 8 seconds, left-pad with zeros if shorter
        target_length = self.sample_rate * self.n_seconds
        if len(audio) > target_length:
            audio = audio[-target_length:]
        elif len(audio) < target_length:
            audio = np.pad(audio, (target_length - len(audio), 0), "constant")

        features = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.n_seconds * self.sample_rate,
            truncation=True,
            do_normalize=True,
        )

        input_features = features.input_features.squeeze(0)  # Shape: (80, 800)
        label = int(bool(item["endpoint_bool"]))

        return input_features, torch.tensor(label, dtype=torch.float32)


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    val_bs = eval_batch_size if eval_batch_size is not None else batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def analyze_dataset(dataset):
    print(f"Total samples: {len(dataset)}")
    df = dataset.to_pandas()
    for col in ["endpoint_bool", "language", "midfiller", "endfiller", "synthetic"]:
        if col in df.columns:
            print(f"\n{col} distribution:")
            print(df[col].value_counts(normalize=True))


def filter_by_language(dataset, languages: list[str]):
    return dataset.filter(lambda x: x["language"] in languages)
