"""Optimizer and scheduler builders."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW


def build_optimizers(
    model: nn.Module,
    lr_backbone: float = 3e-4,
    lr_text: float = 5e-5,
    weight_decay: float = 1e-2,
    text_module_names: Tuple[str, ...] = ("text_encoder",),
) -> Dict[str, torch.optim.Optimizer]:
    """Constructs optimizers for graph/head parameters and optional text adapters."""

    text_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(module_name) for module_name in text_module_names):
            text_params.append(param)
        else:
            other_params.append(param)

    optimizers: Dict[str, torch.optim.Optimizer] = {}
    if other_params:
        optimizers["main"] = AdamW(
            other_params,
            lr=lr_backbone,
            weight_decay=weight_decay,
        )
    if text_params:
        optimizers["text"] = AdamW(
            text_params,
            lr=lr_text,
            weight_decay=weight_decay,
        )
    return optimizers
