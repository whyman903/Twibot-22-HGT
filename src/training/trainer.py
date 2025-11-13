"""Training loop and configuration for the TwiBot-22 model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
import torch.amp as amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import TwiBot22DataModule
from src.data.loaders import create_neighbor_loader
from src.models import TwiBotModel
from src.utils.metrics import compute_metrics


@dataclass
class TrainingConfig:
    raw_data_dir: Path
    processed_data_dir: Path
    batch_size: int = 8192
    num_neighbors: tuple[int, int] = (20, 20)
    num_layers: int = 3
    hidden_dim: int = 256
    graph_heads: int = 4
    dropout: float = 0.3
    max_tweets_per_user: int = 8
    text_model_name: str = "roberta-base"
    text_trainable: bool = False
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    epochs: int = 20
    patience: int = 5
    lr_backbone: float = 3e-4
    lr_text: float = 5e-5
    weight_decay: float = 1e-2
    loss_type: str = "focal"
    gamma: float = 1.5
    mixed_precision: bool = True
    log_dir: Optional[Path] = None
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    text_max_length: int = 512


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: TwiBotModel,
        data_module: TwiBot22DataModule,
        criterion: nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> None:
        self.cfg = config
        self.model = model
        self.data_module = data_module
        self.criterion = criterion
        self.optimizers = optimizers

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Use torch.amp GradScaler device-aware API
        self.scaler = amp.GradScaler(device="cuda" if self.device.type == "cuda" else "cpu", enabled=config.mixed_precision and self.device.type == "cuda")

        self.writer: Optional[SummaryWriter] = None
        if config.log_dir is not None:
            self.writer = SummaryWriter(log_dir=str(config.log_dir))

    def train(self) -> None:
        print("Starting training...", flush=True)
        best_metric = -math.inf
        best_epoch = -1
        for epoch in range(1, self.cfg.epochs + 1):
            train_stats = self._run_epoch(epoch)
            val_stats = self._evaluate(split="valid")
            val_auprc = val_stats.get("auprc", float("nan"))

            if self.writer is not None:
                for key, value in train_stats.items():
                    self.writer.add_scalar(f"train/{key}", value, epoch)
                for key, value in val_stats.items():
                    self.writer.add_scalar(f"val/{key}", value, epoch)

            print(f"Epoch {epoch}: train_loss={train_stats['loss']:.4f} val_auprc={val_auprc:.4f}")

            if val_auprc > best_metric:
                best_metric = val_auprc
                best_epoch = epoch
                self._save_checkpoint(epoch)
            elif epoch - best_epoch >= self.cfg.patience:
                print("Early stopping triggered.")
                break


    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """Public evaluation entry point."""

        return self._evaluate(split=split)

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        train_indices = self.data_module.get_split_indices("train")
        loader = create_neighbor_loader(
            self.data_module.graph,
            input_indices=train_indices,
            batch_size=self.cfg.batch_size,
            num_neighbors=self.cfg.num_neighbors,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        total_loss = 0.0
        total_examples = 0

        print(f"Starting epoch {epoch}...", flush=True)
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            batch = batch.to(self.device)
            user_nodes = batch["user"]
            batch_size = user_nodes.batch_size
            input_nodes = user_nodes.n_id[:batch_size]

            labels_full = self.data_module.labels[input_nodes.cpu()].to(self.device)
            mask = labels_full >= 0
            if mask.sum() == 0:
                continue

            if self.data_module.text_store is None or self.data_module.profile_store is None:
                raise RuntimeError("Data module stores are not initialized. Call setup() first.")

            text_inputs = self.data_module.text_store.fetch(input_nodes.cpu().tolist())
            profile_features = self.data_module.profile_store.fetch(input_nodes.cpu().tolist()).to(self.device)

            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            with amp.autocast("cuda", enabled=self.scaler.is_enabled() and self.device.type == "cuda"):
                logits = self.model(batch, text_inputs, profile_features)
                logits = logits[mask]
                labels = labels_full[mask]
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            for opt in self.optimizers.values():
                self.scaler.step(opt)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

        avg_loss = total_loss / max(total_examples, 1)
        return {"loss": avg_loss}

    @torch.no_grad()
    def _evaluate(self, split: str) -> Dict[str, float]:
        self.model.eval()
        indices = self.data_module.get_split_indices(split)
        loader = create_neighbor_loader(
            self.data_module.graph,
            input_indices=indices,
            batch_size=self.cfg.batch_size,
            num_neighbors=self.cfg.num_neighbors,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        all_labels = []
        all_logits = []

        for batch in tqdm(loader, desc=f"Evaluating {split}"):
            batch = batch.to(self.device)
            user_nodes = batch["user"]
            batch_size = user_nodes.batch_size
            input_nodes = user_nodes.n_id[:batch_size]

            labels_full = self.data_module.labels[input_nodes.cpu()].to(self.device)
            mask = labels_full >= 0
            if mask.sum() == 0:
                continue

            if self.data_module.text_store is None or self.data_module.profile_store is None:
                raise RuntimeError("Data module stores are not initialized. Call setup() first.")

            text_inputs = self.data_module.text_store.fetch(input_nodes.cpu().tolist())
            profile_features = self.data_module.profile_store.fetch(input_nodes.cpu().tolist()).to(self.device)

            logits = self.model(batch, text_inputs, profile_features)
            logits = logits[mask]
            labels = labels_full[mask]

            all_labels.append(labels)
            all_logits.append(logits)

        if not all_labels:
            return {"loss": float("nan"), "auprc": float("nan"), "f1": float("nan")}

        labels_tensor = torch.cat(all_labels)
        logits_tensor = torch.cat(all_logits)
        probs = torch.softmax(logits_tensor, dim=-1)
        preds = probs.argmax(dim=-1)

        metrics = compute_metrics(labels_tensor, preds, probs)
        loss = self.criterion(logits_tensor, labels_tensor)
        metrics["loss"] = float(loss.item())
        return metrics

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_path = Path(self.cfg.processed_data_dir) / "best_model.pt"
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
