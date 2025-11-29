"""Loss functions used in training."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional label smoothing."""

    def __init__(
        self,
        gamma: float = 1.5,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        eps: float = 1e-8,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)
        self.reduction = reduction
        self.eps = eps
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        targets = targets.long()

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        if self.label_smoothing > 0:
            focal_factor = (1 - probs).pow(self.gamma)
            loss = -focal_factor * log_probs * smooth_targets
            loss = loss.sum(dim=-1)
            if self.alpha is not None:
                alpha = self.alpha.gather(dim=0, index=targets)
                loss = loss * alpha
        else:
            log_probs_target = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            probs_target = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            focal_factor = (1 - probs_target).pow(self.gamma)
            loss = -focal_factor * log_probs_target
            if self.alpha is not None:
                alpha = self.alpha.gather(dim=0, index=targets)
                loss = loss * alpha

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropy(nn.Module):
    """Convenience wrapper for class-weighted cross entropy with optional label smoothing."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )
