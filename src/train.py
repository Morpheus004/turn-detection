import os
from dataclasses import dataclass

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .evaluate import compute_metrics


@dataclass
class TrainingConfig:
    model_name: str = "openai/whisper-tiny"
    learning_rate: float = 5e-5
    num_epochs: int = 4
    batch_size: int = 32
    warmup_ratio: float = 0.2
    weight_decay: float = 0.01
    eval_steps: int = 500
    save_dir: str = "checkpoints"
    experiment_name: str = "exp"
    freeze_encoder: bool = True
    unfreeze_last_n: int = 0
    use_wandb: bool = False


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        os.makedirs(config.save_dir, exist_ok=True)

    def train(self):
        if self.config.use_wandb:
            wandb.init(
                project="shiprocket",
                name=self.config.experiment_name,
                config=self.config.__dict__,
            )

        global_step = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            for features, labels in pbar:
                features, labels = features.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features, labels=labels)
                loss = outputs["loss"]

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 100 == 0:
                    pbar.set_postfix({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})
                    if self.config.use_wandb:
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/lr": self.scheduler.get_last_lr()[0],
                                "step": global_step,
                            }
                        )

                if global_step % self.config.eval_steps == 0:
                    metrics = self.evaluate()
                    if self.config.use_wandb:
                        wandb.log(
                            {f"val/{k}": v for k, v in metrics.items() if k != "confusion_matrix"}
                            | {"step": global_step}
                        )

                    self.save_checkpoint(
                        os.path.join(
                            self.config.save_dir,
                            f"{self.config.experiment_name}_step_{global_step}.pt",
                        )
                    )
                    self.model.train()

        # Final evaluation and save
        metrics = self.evaluate()
        if self.config.use_wandb:
            wandb.log(
                {f"val/{k}": v for k, v in metrics.items() if k != "confusion_matrix"}
                | {"step": global_step}
            )
            wandb.finish()

        self.save_checkpoint(
            os.path.join(self.config.save_dir, f"{self.config.experiment_name}_final.pt")
        )
        return metrics

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Evaluating"):
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                probs = outputs["probabilities"]

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        print(f"Eval metrics: {metrics}")
        return metrics

    def save_checkpoint(self, path: str):
        torch.save(
            {"model_state_dict": self.model.state_dict(), "config": self.config.__dict__}, path
        )
        print(f"Saved checkpoint to {path}")
