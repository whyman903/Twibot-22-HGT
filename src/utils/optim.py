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
    """Constructs optimizers with proper weight decay handling (no decay for bias/Norm)."""

    def _group_params(modules: Iterable[nn.Module], lr: float, wd: float) -> List[Dict]:
        decay = []
        no_decay = []
        for module in modules:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if "bias" in name or "Norm" in name or "norm" in name:
                    no_decay.append(param)
                else:
                    decay.append(param)
        return [
            {"params": decay, "lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]

    text_modules = []
    other_modules = []

    # Split top-level modules to avoid traversing same params twice if possible,
    # but since we don't know the structure perfectly, let's iterate named_children
    # to assign them to text or other buckets.
    
    # Helper to classify a top-level module
    def is_text_module(name: str) -> bool:
        return any(name.startswith(tm) for tm in text_module_names)

    # We need to be careful not to miss parameters. 
    # Let's iterate all parameters and assign them to groups.
    # But the standard way is to group by param pointers.
    
    text_params_decay = []
    text_params_no_decay = []
    other_params_decay = []
    other_params_no_decay = []
    
    seen_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        seen_params.add(param)
        
        # Determine if text or other
        is_text = any(name.startswith(tm) for tm in text_module_names)
        
        # Determine if decay or no decay
        # Typical rule: bias, LayerNorm.weight, LayerNorm.bias -> no decay
        # Also embeddings sometimes? But usually we want decay on embeddings.
        # Simplest heuristic: 'bias' in name or 'norm' in name (case insensitive).
        no_decay = "bias" in name or "norm" in name.lower()
        
        if is_text:
            if no_decay:
                text_params_no_decay.append(param)
            else:
                text_params_decay.append(param)
        else:
            if no_decay:
                other_params_no_decay.append(param)
            else:
                other_params_decay.append(param)

    optimizers: Dict[str, torch.optim.Optimizer] = {}
    
    other_groups = []
    if other_params_decay:
        other_groups.append({"params": other_params_decay, "weight_decay": weight_decay})
    if other_params_no_decay:
        other_groups.append({"params": other_params_no_decay, "weight_decay": 0.0})
        
    if other_groups:
        optimizers["main"] = AdamW(other_groups, lr=lr_backbone)

    text_groups = []
    if text_params_decay:
        text_groups.append({"params": text_params_decay, "weight_decay": weight_decay})
    if text_params_no_decay:
        text_groups.append({"params": text_params_no_decay, "weight_decay": 0.0})

    if text_groups:
        optimizers["text"] = AdamW(text_groups, lr=lr_text)

    return optimizers
