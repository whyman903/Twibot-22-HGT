"""Loss functions used in training."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss."""

    def __init__(
        self,
        gamma: float = 1.5,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        targets = targets.long()
        log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        focal_factor = (1 - probs).pow(self.gamma)
        loss = -focal_factor * log_probs
        if self.alpha is not None:
            alpha = self.alpha.gather(dim=0, index=targets)
            loss = loss * alpha
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropy(nn.Module):
    """Convenience wrapper for class-weighted cross entropy."""

    def __init__(self, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets.long(), weight=self.weight)
