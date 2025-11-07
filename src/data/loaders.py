"""DataLoader helpers for the TwiBot-22 project."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import torch
from torch_geometric.loader import NeighborLoader


def create_neighbor_loader(
    data,
    input_indices: torch.Tensor,
    batch_size: int,
    num_neighbors: Sequence[int] = (20, 20),
    shuffle: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = True,
):
    """Constructs a NeighborLoader for user node mini-batches."""

    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("user", input_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=pin_memory,
    )
