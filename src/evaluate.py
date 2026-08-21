import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    preds_binary = (predictions >= threshold).astype(int)
    labels = labels.astype(int)

    return {
        "accuracy": accuracy_score(labels, preds_binary),
        "precision": precision_score(labels, preds_binary, zero_division="warn"),
        "recall": recall_score(labels, preds_binary, zero_division="warn"),
        "f1": f1_score(labels, preds_binary, zero_division="warn"),
        "confusion_matrix": confusion_matrix(labels, preds_binary).tolist(),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            outputs = model(features)
            probs = outputs["probabilities"]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return compute_metrics(np.array(all_preds), np.array(all_labels), threshold)


def evaluate_by_category(
    model: torch.nn.Module, dataset, feature_extractor, device: torch.device, batch_size: int = 32
) -> dict:
    """
    Evaluates model performance across subsets:
    - midfiller (user pausing in middle of thought)
    - endfiller (filler at end)
    - synthetic vs natural speech
    """
    model.eval()
    results = {}

    df = dataset.to_pandas()
    categories = {}
    if "midfiller" in df.columns:
        categories["midfiller_true"] = [i for i, v in enumerate(df["midfiller"]) if bool(v)]
        categories["midfiller_false"] = [i for i, v in enumerate(df["midfiller"]) if not bool(v)]
    if "endfiller" in df.columns:
        categories["endfiller_true"] = [i for i, v in enumerate(df["endfiller"]) if bool(v)]
        categories["endfiller_false"] = [i for i, v in enumerate(df["endfiller"]) if not bool(v)]
    if "synthetic" in df.columns:
        categories["synthetic_true"] = [i for i, v in enumerate(df["synthetic"]) if bool(v)]
        categories["synthetic_false"] = [i for i, v in enumerate(df["synthetic"]) if not bool(v)]

    for cat_name, indices in categories.items():
        if len(indices) == 0:
            continue
        subset_hf = dataset.select(indices)
        from .data import AudioDataset

        subset_ds = AudioDataset(subset_hf, feature_extractor)
        subset_loader = torch.utils.data.DataLoader(subset_ds, batch_size=batch_size, shuffle=False)
        metrics = evaluate_model(model, subset_loader, device)
        results[cat_name] = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        results[cat_name]["sample_count"] = len(indices)

    return results


def plot_results(metrics_dict: dict, save_path: str):
    metrics = {k: v for k, v in metrics_dict.items() if k != "confusion_matrix"}
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.title("Evaluation Metrics")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.savefig(save_path)
    plt.close()


def measure_inference_time(
    model: torch.nn.Module, device: torch.device, n_runs: int = 100
) -> float:
    model.eval()
    dummy_input = torch.randn(1, 80, 3000).to(device)

    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = ((end_time - start_time) / n_runs) * 1000
    return avg_time_ms
