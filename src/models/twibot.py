"""Full TwiBot-22 multi-branch model."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn
from torch_geometric.data import HeteroData

from .graph import RelationalGraphBackbone
from .profile import ProfileFeatureEncoder
from .text import RobertaTextEncoder


class TwiBotModel(nn.Module):
    """Joint model that fuses graph, text, and profile representations."""

    def __init__(
        self,
        graph_backbone: RelationalGraphBackbone,
        text_encoder: RobertaTextEncoder,
        profile_encoder: ProfileFeatureEncoder,
        fusion_hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.graph_backbone = graph_backbone
        self.text_encoder = text_encoder
        self.profile_encoder = profile_encoder
        fusion_input_dim = (
            self.graph_backbone.hidden_dim
            + self.text_encoder.output_dim()
            + self.profile_encoder.output_dim
        )
        dims = [fusion_input_dim, *fusion_hidden_dims, num_classes]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != num_classes:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.classifier = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        batch_data: HeteroData,
        text_inputs: Dict[str, torch.Tensor],
        profile_features: torch.Tensor,
    ) -> torch.Tensor:
        x_dict = self.graph_backbone(batch_data)
        user_store = batch_data["user"]
        if not hasattr(user_store, "batch_size"):
            raise ValueError("NeighborLoader mini-batch expected to provide batch_size for user nodes.")
        batch_size = user_store.batch_size
        graph_emb = x_dict["user"][:batch_size]
        text_emb = self.text_encoder(text_inputs)
        profile_emb = self.profile_encoder(profile_features)
        fused = torch.cat([graph_emb, text_emb, profile_emb], dim=-1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

    def inference(
        self,
        batch_data: HeteroData,
        text_inputs: Dict[str, torch.Tensor],
        profile_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass that returns softmax probabilities."""

        logits = self.forward(batch_data, text_inputs, profile_features)
        return torch.softmax(logits, dim=-1)
