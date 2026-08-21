import argparse
import os
import sys
import time
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import (
    AudioDataset,
    create_dataloaders,
    filter_by_language,
    get_feature_extractor,
    load_smart_turn_dataset,
)
from src.evaluate import evaluate_by_category, measure_inference_time, plot_results
from src.model import LightweightTurnModel, SmartTurnModel
from src.train import Trainer, TrainingConfig
from src.utils import count_parameters, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run Turn Detection Experiment")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--freeze", action=argparse.BooleanOptionalAction, default=True, help="Freeze encoder"
    )
    parser.add_argument(
        "--unfreeze-last-n", type=int, default=0, help="Unfreeze last n layers of encoder"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["whisper", "lightweight"],
        default="whisper",
        help="Model type",
    )
    parser.add_argument("--languages", type=str, default=None, help="Comma separated languages")
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False, help="Use wandb"
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument(
        "--eval-steps", type=int, default=500, help="Evaluation and checkpointing interval in steps"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for validation (defaults to --batch-size)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    feature_extractor = get_feature_extractor()

    print("Loading dataset...")
    train_hf = load_smart_turn_dataset("train", val_split_ratio=args.val_ratio)
    val_hf = load_smart_turn_dataset("validation", val_split_ratio=args.val_ratio)

    if args.languages:
        langs = [lang.strip() for lang in args.languages.split(",")]
        train_hf = filter_by_language(train_hf, langs)
        val_hf = filter_by_language(val_hf, langs)

    print(f"Train samples: {len(train_hf)}, Val samples: {len(val_hf)}")

    train_dataset = AudioDataset(train_hf, feature_extractor)
    val_dataset = AudioDataset(val_hf, feature_extractor)

    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    if args.model_type == "whisper":
        model = SmartTurnModel(freeze_encoder=args.freeze, unfreeze_last_n=args.unfreeze_last_n)
    else:
        model = LightweightTurnModel()

    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    config = TrainingConfig(
        experiment_name=args.experiment,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_steps=args.eval_steps,
        freeze_encoder=args.freeze,
        unfreeze_last_n=args.unfreeze_last_n,
        use_wandb=args.wandb,
    )

    trainer = Trainer(model, train_loader, val_loader, config, device)

    start_time = time.time()
    metrics = trainer.train()
    training_duration_s = time.time() - start_time

    # Latency benchmark
    inf_time_ms = measure_inference_time(model, device)
    print(f"Average inference latency: {inf_time_ms:.2f} ms")

    # Category evaluation
    category_metrics = evaluate_by_category(model, val_hf, feature_extractor, device)
    print("Category Breakdown Metrics:")
    for cat, res in category_metrics.items():
        print(
            f"  {cat}: Acc={res.get('accuracy', 0):.4f}, F1={res.get('f1', 0):.4f} (N={res.get('sample_count', 0)})"
        )

    # Plot metric summary
    os.makedirs("results_plots", exist_ok=True)
    plot_path = f"results_plots/{args.experiment}_metrics.png"
    plot_results(metrics, plot_path)

    # Format and save to RESULTS.md
    log_entry = f"""
### {args.experiment.upper()} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Model**: `{args.model_type}` (Freeze={args.freeze}, UnfreezeLastN={args.unfreeze_last_n})
- **Parameters**: Total={total_params:,} | Trainable={trainable_params:,}
- **Training Time**: {training_duration_s:.1f}s ({args.epochs} epochs, lr={args.lr})
- **GPU Inference Latency**: **{inf_time_ms:.2f} ms**
- **Overall Metrics**:
  - **Accuracy**: `{metrics["accuracy"]:.4f}`
  - **Precision**: `{metrics["precision"]:.4f}`
  - **Recall**: `{metrics["recall"]:.4f}`
  - **F1 Score**: `{metrics["f1"]:.4f}`
- **Confusion Matrix**: `{metrics["confusion_matrix"]}`
- **Category Breakdown**:
"""
    for cat, res in category_metrics.items():
        log_entry += f"  - `{cat}` (N={res['sample_count']}): Accuracy={res['accuracy']:.4f}, F1={res['f1']:.4f}\n"
    log_entry += "\n---\n"

    with open("RESULTS.md", "a") as f:
        f.write(log_entry)

    print(f"\nExperiment {args.experiment} completed successfully! Results logged in RESULTS.md")


if __name__ == "__main__":
    main()
