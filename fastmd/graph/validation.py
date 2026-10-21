"""Graph validity checks shared by eager and CUDA Graph execution paths."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from fastmd.graph.radiusgraph import RadiusGraph


def raise_if_isolated_atoms(count: int | Tensor) -> None:
    """Raise the historical MatRIS error when one or more atoms have no edges."""
    if torch.is_tensor(count):
        count = int(count.item() if count.numel() == 1 else count.sum().item())
    else:
        count = int(count)
    if count:
        raise ValueError(
            f"Error: Detected {count} isolated atom. Calculation stopped"
        )


def raise_if_graph_has_isolated_atoms(
    graphs: "RadiusGraph | Sequence[RadiusGraph]",
) -> None:
    """Synchronously validate metadata, or derive counts for legacy graphs.

    This is an explicit strict-mode check, not part of tolerant graph replay.
    Graphs loaded from older files may not carry isolated-atom metadata.
    """
    if not isinstance(graphs, Sequence):
        graphs = [graphs]
    counts = []
    for graph in graphs:
        isolated_count = getattr(graph, "isolated_atom_count", None)
        if isolated_count is None:
            target_index = graph.atom_graph[:, 0].long()
            degrees = torch.zeros(
                graph.atomic_number.shape[0], dtype=torch.long,
                device=graph.atom_graph.device,
            )
            degrees.index_add_(0, target_index, torch.ones_like(target_index))
            isolated_count = degrees.eq(0).sum().reshape(1)
        counts.append(isolated_count.reshape(-1))
    if len(counts) == 1:
        raise_if_isolated_atoms(counts[0])
    elif counts:
        raise_if_isolated_atoms(torch.cat(counts))
