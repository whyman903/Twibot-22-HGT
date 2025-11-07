"""RoBERTa-based text encoder for user content."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoConfig


class RobertaTextEncoder(nn.Module):
    """Encodes user bios and recent tweets with a frozen RoBERTa backbone."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        trainable: bool = False,
        dropout: float = 0.1,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        if hasattr(cfg, "add_pooling_layer"):
            cfg.add_pooling_layer = False
        self.model = AutoModel.from_pretrained(model_name, config=cfg)

        # Track whether the backbone is frozen (no trainable params in this module)
        self._frozen = (not trainable) and (not use_lora)
        if self._frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            # Ensure deterministic behavior (disable dropout etc.) when frozen
            self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:  
                raise RuntimeError(
                    "LoRA adapters requested but the 'peft' package is not installed."
                ) from exc
            if target_modules is None:
                target_modules = ["query", "value"]
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.model = get_peft_model(self.model, lora_cfg)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._frozen:
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden * mask
        token_counts = mask.sum(dim=1).clamp(min=1)
        pooled = masked_hidden.sum(dim=1) / token_counts
        pooled = self.dropout(pooled)
        return pooled

    def output_dim(self) -> int:
        return self.hidden_size
