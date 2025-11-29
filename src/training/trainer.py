"""Training loop and configuration for the TwiBot-22 model."""
from __future__ import annotations

import json
import math
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR
import torch.amp as amp
import numpy as np
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
    batch_size: int = 64
    num_neighbors: Tuple[int, int] = (15, 10)
    num_layers: int = 3
    hidden_dim: int = 384
    graph_heads: int = 4
    dropout: float = 0.4
    max_tweets_per_user: int = -1
    text_model_name: str = "xlm-roberta-base"
    text_trainable: bool = False
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    tweet_text_model_name: Optional[str] = None
    tweet_text_max_length: int = 96
    tweet_text_batch_size: int = 512
    epochs: int = 15
    patience: int = 5
    lr_backbone: float = 1e-4
    lr_text: float = 2e-5
    weight_decay: float = 0.05
    loss_type: str = "focal"
    gamma: float = 1.5
    mixed_precision: bool = True
    log_dir: Optional[Path] = None
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    text_max_length: int = 512
    max_grad_norm: float = 1.0
    resume: bool = False
    seed: int = 42
    deterministic: bool = True
    target_pos_ratio: Optional[float] = None
    bot_weight_mult: float = 1.0
    label_smoothing: float = 0.0
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    warmup_epochs: int = 1
    val_every_n_batches: int = 0


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for reproducibility across common libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

        set_seed(config.seed, deterministic=config.deterministic)

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion.to(self.device)

        self.scaler = amp.GradScaler(
            device="cuda" if self.device.type == "cuda" else "cpu",
            enabled=config.mixed_precision and self.device.type == "cuda"
        )

        self.writer: Optional[SummaryWriter] = None
        if config.log_dir is not None:
            self.writer = SummaryWriter(log_dir=str(config.log_dir))

        self.history: List[Dict] = []

        self.start_epoch = 1
        self.best_metric = -math.inf
        self.best_epoch = -1

        self.schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        if config.use_scheduler and optimizers:
            self._init_schedulers()

        if self.cfg.resume:
            self._resume_from_checkpoint()

    def _init_schedulers(self) -> None:
        """Initialize learning rate schedulers for each optimizer."""
        for name, opt in self.optimizers.items():
            if self.cfg.scheduler_type == "cosine":
                base_scheduler = CosineAnnealingWarmRestarts(
                    opt,
                    T_0=max(1, self.cfg.epochs // 3),
                    T_mult=2,
                    eta_min=1e-7,
                )
                if self.cfg.warmup_epochs > 0:
                    warmup_epochs = self.cfg.warmup_epochs
                    def lr_lambda(epoch: int, warmup_epochs=warmup_epochs) -> float:
                        if epoch < warmup_epochs:
                            return float(epoch + 1) / float(warmup_epochs)
                        return 1.0
                    warmup_scheduler = LambdaLR(opt, lr_lambda)
                    self.schedulers[name] = (warmup_scheduler, base_scheduler, warmup_epochs)
                else:
                    self.schedulers[name] = base_scheduler
            else:
                scheduler = ReduceLROnPlateau(
                    opt,
                    mode='max',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-7,
                    verbose=True,
                )
                self.schedulers[name] = scheduler

    def train(self) -> None:
        print("Starting training...", flush=True)
        print(f"  Device: {self.device}", flush=True)
        print(f"  Batch size: {self.cfg.batch_size}", flush=True)
        print(f"  Epochs: {self.cfg.epochs}", flush=True)
        print(f"  Learning rates: backbone={self.cfg.lr_backbone}, text={self.cfg.lr_text}", flush=True)

        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            train_stats = self._run_epoch(epoch)
            val_stats = self._evaluate(split="valid")
            val_auprc = val_stats.get("auprc", float("nan"))

            if self.writer is not None:
                for key, value in train_stats.items():
                    self.writer.add_scalar(f"train/{key}", value, epoch)
                for key, value in val_stats.items():
                    self.writer.add_scalar(f"val/{key}", value, epoch)
                for name, opt in self.optimizers.items():
                    lr = opt.param_groups[0]['lr']
                    self.writer.add_scalar(f"lr/{name}", lr, epoch)

            self.history.append({
                "epoch": epoch,
                "train": train_stats,
                "val": val_stats,
                "lr": {name: opt.param_groups[0]['lr'] for name, opt in self.optimizers.items()}
            })

            lr_str = ", ".join([f"{name}={opt.param_groups[0]['lr']:.2e}" for name, opt in self.optimizers.items()])
            print(f"Epoch {epoch}: train_loss={train_stats['loss']:.4f} "
                  f"train_f1={train_stats.get('f1', float('nan')):.4f} | "
                  f"val_loss={val_stats.get('loss', float('nan')):.4f} "
                  f"val_auprc={val_auprc:.4f} "
                  f"val_f1={val_stats.get('f1', float('nan')):.4f} | "
                  f"lr: {lr_str}")

            for name, sched in self.schedulers.items():
                if isinstance(sched, tuple):
                    warmup_sched, base_sched, warmup_epochs = sched
                    if epoch <= warmup_epochs:
                        warmup_sched.step()
                    else:
                        base_sched.step()
                elif isinstance(sched, ReduceLROnPlateau):
                    sched.step(val_auprc)
                else:
                    sched.step(epoch)

            if val_auprc > self.best_metric:
                self.best_metric = val_auprc
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)
            elif epoch - self.best_epoch >= self.cfg.patience:
                print("Early stopping triggered.")
                break

            self._save_checkpoint(epoch, is_best=False)

        self._save_metrics_json()

        best_f1 = float("nan")
        for entry in self.history:
            if entry["epoch"] == self.best_epoch:
                best_f1 = entry["val"].get("f1", float("nan"))
                break

        print(f"Training complete. Best val AUPRC: {self.best_metric:.4f} at epoch {self.best_epoch} (F1: {best_f1:.4f})")

    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """Public evaluation entry point."""
        return self._evaluate(split=split)

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        train_indices = self._sample_train_indices(epoch)
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
        finite_batches = 0

        running_correct = 0
        running_total = 0
        running_tp = 0
        running_fp = 0
        running_fn = 0

        num_batches = len(loader)
        print(f"Starting epoch {epoch} ({num_batches} batches)...", flush=True)

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", file=sys.stdout, miniters=100, mininterval=30)):
            batch = batch.to(self.device)
            user_nodes = batch["user"]
            batch_size = user_nodes.batch_size
            input_nodes = user_nodes.n_id[:batch_size]
            input_nodes_cpu = input_nodes.cpu()

            labels_full = self.data_module.labels[input_nodes_cpu].to(self.device)
            mask = labels_full >= 0
            if mask.sum() == 0:
                continue

            if self.data_module.text_store is None or self.data_module.profile_store is None:
                raise RuntimeError("Data module stores are not initialized. Call setup() first.")

            text_inputs = self.data_module.text_store.fetch(input_nodes_cpu)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            profile_features = self.data_module.profile_store.fetch(input_nodes_cpu)
            profile_features = profile_features.to(self.device)

            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            node_features = self.data_module.fetch_node_features(batch, self.device)

            with amp.autocast("cuda", enabled=self.scaler.is_enabled() and self.device.type == "cuda"):
                logits = self.model(batch, text_inputs, profile_features, node_features=node_features)
                logits = logits[mask]
                labels = labels_full[mask]
                loss = self.criterion(logits, labels)

            if not torch.isfinite(loss):
                print(f"Non-finite loss detected at epoch {epoch}, batch {batch_idx}: {loss.item()}", flush=True)
                continue

            self.scaler.scale(loss).backward()

            for name, opt in self.optimizers.items():
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in opt.param_groups for p in group["params"] if p.requires_grad],
                    max_norm=self.cfg.max_grad_norm,
                )

            for opt in self.optimizers.values():
                self.scaler.step(opt)
            self.scaler.update()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                running_correct += (preds == labels).sum().item()
                running_total += labels.size(0)
                running_tp += ((preds == 1) & (labels == 1)).sum().item()
                running_fp += ((preds == 1) & (labels == 0)).sum().item()
                running_fn += ((preds == 0) & (labels == 1)).sum().item()

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            finite_batches += 1

        if finite_batches == 0:
            return {"loss": float("nan")}

        avg_loss = total_loss / max(total_examples, 1)

        accuracy = running_correct / max(running_total, 1)
        precision = running_tp / max(running_tp + running_fp, 1)
        recall = running_tp / max(running_tp + running_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision_1": precision,
            "recall_1": recall,
            "f1": f1,
        }

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
        total_loss = 0.0
        total_examples = 0

        for batch in tqdm(loader, desc=f"Evaluating {split}", file=sys.stdout, miniters=50, mininterval=30):
            batch = batch.to(self.device)
            user_nodes = batch["user"]
            batch_size = user_nodes.batch_size
            input_nodes = user_nodes.n_id[:batch_size]
            input_nodes_cpu = input_nodes.cpu()

            labels_full = self.data_module.labels[input_nodes_cpu].to(self.device)
            mask = labels_full >= 0
            if mask.sum() == 0:
                continue

            if self.data_module.text_store is None or self.data_module.profile_store is None:
                raise RuntimeError("Data module stores are not initialized. Call setup() first.")

            text_inputs = self.data_module.text_store.fetch(input_nodes_cpu)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            profile_features = self.data_module.profile_store.fetch(input_nodes_cpu)
            profile_features = profile_features.to(self.device)

            node_features = self.data_module.fetch_node_features(batch, self.device)
            logits = self.model(batch, text_inputs, profile_features, node_features=node_features)
            logits = logits[mask]
            labels = labels_full[mask]

            batch_loss = self.criterion(logits, labels)
            total_loss += batch_loss.item() * labels.size(0)
            total_examples += labels.size(0)

            all_labels.append(labels.detach().cpu())
            all_logits.append(logits.detach().cpu())

        if not all_labels:
            return {"loss": float("nan"), "auprc": float("nan"), "f1": float("nan")}

        labels_tensor = torch.cat(all_labels)
        logits_tensor = torch.cat(all_logits)
        probs = torch.softmax(logits_tensor, dim=-1)
        preds = probs.argmax(dim=-1)

        metrics = compute_metrics(labels_tensor, preds, probs)
        if total_examples == 0:
            metrics["loss"] = float("nan")
        else:
            metrics["loss"] = float(total_loss / total_examples)
        return metrics

    def _sample_train_indices(self, epoch: int) -> torch.Tensor:
        """Optionally rebalance training indices toward a target positive ratio."""
        base_indices = self.data_module.get_split_indices("train")
        target_ratio = self.cfg.target_pos_ratio
        if target_ratio is None or target_ratio <= 0 or target_ratio >= 1:
            return base_indices
        labels = self.data_module.labels
        if labels is None:
            return base_indices
        pos = base_indices[(labels[base_indices] == 1).cpu()]
        neg = base_indices[(labels[base_indices] == 0).cpu()]
        if pos.numel() == 0 or neg.numel() == 0:
            return base_indices

        total = base_indices.numel()
        target_pos = max(1, int(total * target_ratio))
        target_neg = max(1, total - target_pos)

        rng = torch.Generator()
        rng.manual_seed(self.cfg.seed + epoch)
        pos_sel = pos[torch.randint(low=0, high=pos.numel(), size=(target_pos,), generator=rng)]
        neg_sel = neg[torch.randint(low=0, high=neg.numel(), size=(target_neg,), generator=rng)]
        mixed = torch.cat([pos_sel, neg_sel], dim=0)
        perm = torch.randperm(mixed.numel(), generator=rng)
        return mixed[perm]

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_states": {k: v.state_dict() for k, v in self.optimizers.items()},
            "scheduler_states": {
                k: (v[0].state_dict(), v[1].state_dict(), v[2]) if isinstance(v, tuple) else v.state_dict()
                for k, v in self.schedulers.items()
            },
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "config": {
                "text_trainable": self.cfg.text_trainable,
                "use_lora": self.cfg.use_lora,
                "text_model_name": self.cfg.text_model_name,
                "hidden_dim": self.cfg.hidden_dim,
                "num_layers": self.cfg.num_layers,
                "graph_heads": self.cfg.graph_heads,
            },
        }

        if self.scaler.is_enabled():
            checkpoint["scaler_state"] = self.scaler.state_dict()

        last_path = Path(self.cfg.processed_data_dir) / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = Path(self.cfg.processed_data_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")

    def _resume_from_checkpoint(self) -> None:
        last_path = Path(self.cfg.processed_data_dir) / "last_checkpoint.pt"
        if not last_path.exists():
            print(f"No checkpoint found at {last_path}. Starting from scratch.")
            return

        print(f"Loading checkpoint from {last_path}...")
        checkpoint = torch.load(last_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        for name, opt in self.optimizers.items():
            if name in checkpoint["optimizer_states"]:
                opt.load_state_dict(checkpoint["optimizer_states"][name])

        if "scheduler_states" in checkpoint:
            for name, sched in self.schedulers.items():
                if name in checkpoint["scheduler_states"]:
                    saved_state = checkpoint["scheduler_states"][name]
                    if isinstance(sched, tuple) and isinstance(saved_state, tuple):
                        sched[0].load_state_dict(saved_state[0])
                        sched[1].load_state_dict(saved_state[1])
                    elif not isinstance(sched, tuple) and not isinstance(saved_state, tuple):
                        sched.load_state_dict(saved_state)

        if "scaler_state" in checkpoint and self.scaler.is_enabled():
            self.scaler.load_state_dict(checkpoint["scaler_state"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint.get("best_metric", -math.inf)
        self.best_epoch = checkpoint.get("best_epoch", -1)
        self.history = checkpoint.get("history", [])

        print(f"Resumed from epoch {checkpoint['epoch']} (Next epoch: {self.start_epoch})")

    def _save_metrics_json(self) -> None:
        """Save training history to JSON for easy plotting."""
        if self.cfg.log_dir is None:
            return

        metrics_file = Path(self.cfg.log_dir) / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Metrics saved to {metrics_file}")
