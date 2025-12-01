"""Graph backbone built on relational graph transformer architecture."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, LayerNorm, TransformerConv, HGTConv


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

        self.convs = HeteroConv(convs, aggr="mean")
        self.norms = nn.ModuleDict({ntype: LayerNorm(hidden_dim) for ntype in node_types})
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict_norm = {k: self.norms[k](v) for k, v in x_dict.items()}
        out_dict = self.convs(x_dict_norm, edge_index_dict)

        new_x_dict = {}
        for k, x in x_dict.items():
            if k in out_dict:
                out = self.dropout(out_dict[k])
                new_x_dict[k] = x + out
            else:
                new_x_dict[k] = x
        return new_x_dict


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
        node_feature_dims: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__()
        node_types, _ = metadata
        self.hidden_dim = hidden_dim
        self.node_embeddings = nn.ModuleDict()
        self.type_vectors = nn.ParameterDict()
        embed_set = set(embed_node_types)
        feature_dims = dict(node_feature_dims or {})
        for ntype in node_types:
            if ntype in feature_dims:
                continue
            if ntype in embed_set:
                self.node_embeddings[ntype] = nn.Embedding(num_nodes_dict[ntype], hidden_dim)
            elif use_type_vectors_for_others:
                self.type_vectors[ntype] = nn.Parameter(torch.empty(hidden_dim))
                nn.init.xavier_uniform_(self.type_vectors[ntype].unsqueeze(0))
        self.layers = nn.ModuleList(
            [RelationalGraphLayer(metadata, hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.feature_projectors = nn.ModuleDict(
            {ntype: nn.Linear(dim, hidden_dim) for ntype, dim in feature_dims.items()}
        )

    def reset_parameters(self) -> None:
        for emb in self.node_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for tv in self.type_vectors.values():
            nn.init.xavier_uniform_(tv.unsqueeze(0))
        for layer in self.layers:
            for module in layer.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    def forward(
        self,
        data: HeteroData,
        node_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        x_dict: Dict[str, torch.Tensor] = {}
        for node_type in data.node_types:
            if node_features is not None and node_type in node_features:
                if node_type not in self.feature_projectors:
                    raise ValueError(f"No projector registered for features of node type '{node_type}'")
                projector = self.feature_projectors[node_type]
                feats = node_features[node_type]
                x = projector(feats)
            elif node_type in self.node_embeddings:
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
                if hasattr(data[node_type], "n_id"):
                    node_count = data[node_type].n_id.numel()
                else:
                    node_count = data[node_type].num_nodes
                device = next(self.parameters()).device
                if node_type in self.type_vectors:
                    tv = self.type_vectors[node_type].to(device)
                else:
                    tv = torch.zeros(self.hidden_dim, device=device)
                x = tv.unsqueeze(0).expand(node_count, -1)
            x_dict[node_type] = x

        for layer in self.layers:
            x_dict = layer(x_dict, data.edge_index_dict)

        return x_dict


class HGTTransformerLayer(nn.Module):
    """Single HGT layer with proper Transformer architecture (Pre-Norm -> Attn -> Add -> Pre-Norm -> FFN -> Add)."""

    def __init__(
        self,
        metadata: Tuple[Iterable[str], Iterable[Tuple[str, str, str]]],
        hidden_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        node_types, _ = metadata
        
        self.attn = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=heads,
        )
        self.norm1 = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
        
        self.norm2 = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
        self.ffn = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ) for nt in node_types
        })
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_norm = {k: self.norm1[k](v) for k, v in x_dict.items()}
        out_dict = self.attn(x_norm, edge_index_dict)
        
        x_dict_attn = {}
        for k, v in x_dict.items():
            if k in out_dict:
                x_dict_attn[k] = v + self.dropout(out_dict[k])
            else:
                x_dict_attn[k] = v
                
        x_out = {}
        for k, v in x_dict_attn.items():
            if k in self.ffn:
                norm_v = self.norm2[k](v)
                ffn_out = self.ffn[k](norm_v)
                x_out[k] = v + ffn_out
            else:
                x_out[k] = v
                
        return x_out


class HGTBackbone(nn.Module):
    """Heterogeneous Graph Transformer backbone using full Transformer blocks."""

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
        node_feature_dims: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__()
        node_types, _ = metadata
        self.hidden_dim = hidden_dim

        self.node_embeddings = nn.ModuleDict()
        self.type_vectors = nn.ParameterDict()
        embed_set = set(embed_node_types)
        feature_dims = dict(node_feature_dims or {})
        for ntype in node_types:
            if ntype in feature_dims:
                continue
            if ntype in embed_set:
                self.node_embeddings[ntype] = nn.Embedding(num_nodes_dict[ntype], hidden_dim)
            elif use_type_vectors_for_others:
                self.type_vectors[ntype] = nn.Parameter(torch.empty(hidden_dim))
                nn.init.xavier_uniform_(self.type_vectors[ntype].unsqueeze(0))

        self.layers = nn.ModuleList(
            [
                HGTTransformerLayer(
                    metadata=metadata,
                    hidden_dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.feature_projectors = nn.ModuleDict(
            {ntype: nn.Linear(dim, hidden_dim) for ntype, dim in feature_dims.items()}
        )

    def reset_parameters(self) -> None:
        for emb in self.node_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for tv in self.type_vectors.values():
            nn.init.xavier_uniform_(tv.unsqueeze(0))
        pass

    def _initial_x(
        self,
        data: HeteroData,
        node_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        x_dict: Dict[str, torch.Tensor] = {}
        for node_type in data.node_types:
            if node_features is not None and node_type in node_features:
                if node_type not in self.feature_projectors:
                    raise ValueError(f"No projector registered for features of node type '{node_type}'")
                projector = self.feature_projectors[node_type]
                feats = node_features[node_type]
                x = projector(feats)
            elif node_type in self.node_embeddings:
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
                if hasattr(data[node_type], "n_id"):
                    node_count = data[node_type].n_id.numel()
                else:
                    node_count = data[node_type].num_nodes
                device = next(self.parameters()).device
                if node_type in self.type_vectors:
                    tv = self.type_vectors[node_type].to(device)
                else:
                    tv = torch.zeros(self.hidden_dim, device=device)
                x = tv.unsqueeze(0).expand(node_count, -1)
            x_dict[node_type] = x
        return x_dict

    def forward(
        self,
        data: HeteroData,
        node_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        x_dict = self._initial_x(data, node_features=node_features)
        for layer in self.layers:
            x_dict = layer(x_dict, data.edge_index_dict)
        return x_dict
