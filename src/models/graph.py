"""Graph backbone built on top of a relational graph transformer."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple, Sequence

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, LayerNorm, TransformerConv


class RelationalGraphLayer(nn.Module):
    """Single heterogenous transformer layer with relation-specific attention."""

    def __init__(
        self,
        metadata: Tuple[Iterable[str], Iterable[Tuple[str, str, str]]],
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata

        convs = {}
        for edge_type in edge_types:
            convs[edge_type] = TransformerConv(
                (-1, -1),
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
                beta=False,
                edge_dim=None,
                concat=False,
            )

        self.convs = HeteroConv(convs, aggr="sum")
        self.norms = nn.ModuleDict({ntype: LayerNorm(hidden_dim) for ntype in node_types})
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        out_dict = self.convs(x_dict, edge_index_dict)
        for node_type, out in out_dict.items():
            residual = x_dict[node_type]
            out = self.norms[node_type](out)
            out_dict[node_type] = self.dropout(out + residual)
        return out_dict


class RelationalGraphBackbone(nn.Module):
    """Multi-layer relational graph transformer backbone."""

    def __init__(
        self,
        metadata: Tuple[Iterable[str], Iterable[Tuple[str, str, str]]],
        num_nodes_dict: Dict[str, int],
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        embed_node_types: Sequence[str] = ("user",),
        use_type_vectors_for_others: bool = True,
    ) -> None:
        super().__init__()
        node_types, _ = metadata
        self.hidden_dim = hidden_dim
        # Only allocate full embeddings for selected node types (default: 'user').
        self.node_embeddings = nn.ModuleDict()
        self.type_vectors = nn.ParameterDict()
        embed_set = set(embed_node_types)
        for ntype in node_types:
            if ntype in embed_set:
                self.node_embeddings[ntype] = nn.Embedding(num_nodes_dict[ntype], hidden_dim)
            elif use_type_vectors_for_others:
                # A single learnable vector shared by all nodes of this type.
                self.type_vectors[ntype] = nn.Parameter(torch.empty(hidden_dim))
                nn.init.xavier_uniform_(self.type_vectors[ntype].unsqueeze(0))
        self.layers = nn.ModuleList(
            [RelationalGraphLayer(metadata, hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        for emb in self.node_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for tv in self.type_vectors.values():
            nn.init.xavier_uniform_(tv.unsqueeze(0))
        for layer in self.layers:
            for module in layer.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict: Dict[str, torch.Tensor] = {}
        # Produce per-node inputs for all node types, using either a lookup or a shared type vector
        for node_type in data.node_types:
            if node_type in self.node_embeddings:
                embedding = self.node_embeddings[node_type]
                if hasattr(data[node_type], "n_id"):
                    node_idx = data[node_type].n_id
                else:
                    node_idx = torch.arange(
                        data[node_type].num_nodes,
                        device=embedding.weight.device,
                        dtype=torch.long,
                    )
                x = embedding(node_idx.to(embedding.weight.device))
            else:
                # Use a shared type vector expanded to the node count in the mini-batch
                if hasattr(data[node_type], "n_id"):
                    node_count = data[node_type].n_id.numel()
                else:
                    node_count = data[node_type].num_nodes
                # Place the type vector on the same device as available embeddings or edge indices
                device = next(self.parameters()).device
                if node_type in self.type_vectors:
                    tv = self.type_vectors[node_type].to(device)
                else:
                    # Fallback: non-embedded type without a vector; create a zeros tensor
                    tv = torch.zeros(self.hidden_dim, device=device)
                x = tv.unsqueeze(0).expand(node_count, -1)
            x_dict[node_type] = x

        for layer in self.layers:
            x_dict = layer(x_dict, data.edge_index_dict)

        return x_dict
