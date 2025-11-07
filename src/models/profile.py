"""Profile feature encoder."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ProfileFeatureEncoder(nn.Module):
    """Two-layer MLP for normalized profile features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self._output_dim = dims[-1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)

    @property
    def output_dim(self) -> int:
        return self._output_dim
