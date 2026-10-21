from __future__ import annotations

from fastmd.config import env_value


import numpy as np
import torch
from torch import Tensor

from fastmd._nvtx import nvtx_range
from fastmd.adapters.protocol import GraphBatch
from fastmd.graph.geometry import small_matmul
from fastmd.graph.matris_topology import matris_builder_evidence
from fastmd.graph.radiusgraph import RadiusGraph
from fastmd.graph.validation import raise_if_isolated_atoms

from fastmd.graph.neighbor_list_backend import available as op_available
from fastmd.graph.neighbor_list_backend import cell_list_for_fixed_cell
from fastmd.graph.neighbor_list_backend import estimate_max_neighbors
from fastmd.graph.neighbor_list_backend import neighbor_list as _nvidia_nl

try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:
    triton = None
    tl = None
    _triton_available = False

# Static upper bound on periodic-image shift magnitude, used to size sort keys
# without a .item() D2H sync. |shift| <= MAX_IMAGE holds whenever the smallest
# cell dimension >= cutoff / MAX_IMAGE, i.e. always for physical cells.
MAX_IMAGE = 10

_GPU_BUILDER_EVIDENCE = matris_builder_evidence(
    "fastmd.gpu_radius_graph",
    atom_target_sorted=True,
    line_owner_sorted=True,
)


def _env_flag(name: str, default: bool) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "off", "no"}


_LINE_GRAPH_TRITON = _env_flag("MATRIS_LINE_GRAPH_TRITON", True)
_LINE_GRAPH_TRITON_BLOCK = int(env_value("MATRIS_LINE_GRAPH_TRITON_BLOCK", "256"))
# Elide the redundant _build_d2u key sort and _build_line_graph stable sort
# when the edge stream is already ordered by construction (single-system path;
# neighbor_list_nvidia argsorts by (batch, center, neighbor, image), and
# batch is a monotone function of the graph-contiguous center index, so both
# downstream key orders are postconditions of that argsort).
_PRESORTED_BUILD = _env_flag("MATRIS_PRESORTED_BUILD", True)


def _cuda_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(device)
    return device


# Grown neighbor capacities are rounded up to this many slots, matching
# estimate_max_neighbors' alignment.
_NEIGHBOR_SLOT_QUANTUM = 16


def _initial_max_neighbors(cutoff: float, max_neighbors: int | None) -> int:
    if max_neighbors is None:
        return int(estimate_max_neighbors(cutoff=cutoff, safety_factor=2.0))
    capacity = int(max_neighbors)
    if capacity <= 0:
        raise ValueError(f"max_neighbors must be positive, got {max_neighbors}")
    return capacity


def _grown_max_neighbors(capacity: int, max_count: int) -> int:
    quantum = _NEIGHBOR_SLOT_QUANTUM
    return max(2 * capacity, -(-max_count // quantum) * quantum)


class _NeighborOverflow(Exception):
    """A ragged neighbor row outgrew its slots; the build must be redone."""

    def __init__(self, max_count: int) -> None:
        super().__init__(max_count)
        self.max_count = max_count


def _check_neighbor_capacity(max_count: int, capacity: int) -> None:
    # nvalchemiops increments num_neighbors for every candidate but stores only
    # the first ``capacity`` per row, so this detects truncation exactly.
    if max_count > capacity:
        raise _NeighborOverflow(max_count)


def _max_neighbor_count(num_neighbors: Tensor) -> int:
    return int(num_neighbors.max()) if num_neighbors.numel() else 0


@torch.no_grad()
def neighbor_list_nvidia(
    positions: Tensor,
    cell: Tensor,
    cutoff: float,
    batch_idx: Tensor,
    pbc: Tensor,
    device: torch.device | str = "cuda",
    return_num_neighbors: bool = False,
    *,
    max_neighbors: int | None = None,
    cell_list: tuple | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    """Full neighbor list sorted by (batch, center, neighbor, image).

    ``max_neighbors`` is the initial per-atom slot count (None: the volume
    estimate). With ``return_num_neighbors`` the caller must reject
    ``num_neighbors.max() > max_neighbors`` (a truncated list), which lets it
    fold the check into a transfer it already makes; otherwise it is checked
    here and the query reruns with more slots. ``cell_list``: the caller's
    ``cell_list_for_fixed_cell`` (None: sized per call).
    """
    with nvtx_range("neighbor_list_nvidia"):
        device = _cuda_device(device)
        positions = positions.to(device)
        cell = cell.to(device)
        batch_idx = batch_idx.to(device=device, dtype=torch.int32)
        pbc = pbc.to(device=device, dtype=torch.bool)

        n_atoms = positions.shape[0]
        buffer_max = _initial_max_neighbors(cutoff, max_neighbors)
        fixed_cell = {} if cell_list is None else {"cell_list": cell_list}
        while True:
            neighbor_matrix = torch.full(
                (n_atoms, buffer_max),
                n_atoms,
                dtype=torch.int32,
                device=device,
            )
            neighbor_shifts = torch.zeros(
                (n_atoms, buffer_max, 3),
                dtype=torch.int32,
                device=device,
            )
            num_neighbors = torch.zeros(n_atoms, dtype=torch.int32, device=device)

            _nvidia_nl(
                positions=positions,
                cutoff=cutoff + 1e-6,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                method="batch_cell_list",
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_shifts,
                num_neighbors=num_neighbors,
                half_fill=False,
                **fixed_cell,
            )
            if return_num_neighbors:
                break
            max_count = _max_neighbor_count(num_neighbors)
            if max_count <= buffer_max:
                break
            buffer_max = _grown_max_neighbors(buffer_max, max_count)

        atom_idx = torch.arange(n_atoms, device=device).unsqueeze(1)
        neigh_idx = torch.arange(buffer_max, device=device).unsqueeze(0)
        valid = neigh_idx < num_neighbors.unsqueeze(1)
        center = atom_idx.expand(-1, buffer_max)[valid].long()
        neighbor = neighbor_matrix[valid].long()
        images = neighbor_shifts[valid]

        edge_batch = batch_idx[center].long()
        diff = positions[neighbor] - positions[center]
        diff = diff + small_matmul(
            images.float().unsqueeze(-2), cell[edge_batch]
        ).squeeze(-2)
        dist = diff.norm(dim=1)
        mask = dist < cutoff
        center = center[mask]
        neighbor = neighbor[mask]
        images = images[mask]
        dist = dist[mask]

        img = images.long()
        # Static image bound avoids a .item() D2H sync. |shift| <= MAX_IMAGE holds for
        # any cell whose smallest dimension >= cutoff/MAX_IMAGE (always true physically).
        max_img = MAX_IMAGE
        radix = 2 * max_img + 1
        order = (
            batch_idx[center].long() * n_atoms * n_atoms * radix**3
            + center * n_atoms * radix**3
            + neighbor * radix**3
            + (img[:, 0] + max_img) * radix * radix
            + (img[:, 1] + max_img) * radix
            + (img[:, 2] + max_img)
        ).argsort()
        result = center[order], neighbor[order], images[order], dist[order]
        if return_num_neighbors:
            return (*result, num_neighbors)
        return result


def _build_d2u(
    center: Tensor,
    neighbor: Tensor,
    images: Tensor,
    n_atoms: int,
    presorted: bool = False,
) -> tuple[Tensor, Tensor]:
    with nvtx_range("_build_d2u"):
        device = center.device
        n_edges = center.shape[0]
        if n_edges == 0:
            # A valid all-isolated graph has no reverse-edge pairs. Shape-based
            # dispatch is host-known and does not synchronize the device.
            empty = torch.empty(0, dtype=torch.int32, device=device)
            return empty, empty.clone()
        img = images.long()
        max_img = MAX_IMAGE  # static bound -> no .item() D2H sync
        radix = 2 * max_img + 1
        volume = radix * radix * radix
        forward_key = (
            (center * n_atoms + neighbor) * volume
            + (img[:, 0] + max_img) * radix * radix
            + (img[:, 1] + max_img) * radix
            + (img[:, 2] + max_img)
        )
        reverse_key = (
            (neighbor * n_atoms + center) * volume
            + (-img[:, 0] + max_img) * radix * radix
            + (-img[:, 1] + max_img) * radix
            + (-img[:, 2] + max_img)
        )
        if presorted:
            # forward_key order is a postcondition of the neighbor-list
            # argsort (same lexicographic components; the omitted batch term
            # is monotone in center), so the sort is the identity.
            pair_idx = torch.searchsorted(forward_key, reverse_key).clamp(
                max=n_edges - 1
            )
            sorted_key = forward_key
        else:
            sorted_key, sort_idx = forward_key.sort()
            pair_idx = sort_idx[
                torch.searchsorted(sorted_key, reverse_key).clamp(
                    max=n_edges - 1
                )
            ]
        edge_ids = torch.arange(n_edges, device=device)
        uvals, directed2undirected = torch.unique(
            torch.minimum(edge_ids, pair_idx),
            return_inverse=True,
        )
        undirected2directed = torch.full(
            (uvals.shape[0],),  # = num unique undirected edges; shape read, no .item() D2H sync
            n_edges,
            dtype=torch.int64,
            device=device,
        )
        undirected2directed.scatter_reduce_(
            0,
            directed2undirected.long(),
            edge_ids,
            reduce="amin",
            include_self=False,
        )
        return directed2undirected.int(), undirected2directed.int()


if _triton_available:

    # Edge/pair counts change every MD step: they are runtime arguments excluded
    # from Triton's value specialization (==1, %16), so a new count never
    # compiles. The atom count stays a compile-time key (fixed per system).
    @triton.jit(do_not_specialize=["n_edges"])
    def _count_isolated_sorted_kernel(
        center,
        batch_idx,
        isolated_atom_counts,
        n_edges,
        SEARCH_STEPS: tl.constexpr,
    ):
        atom = tl.program_id(0).to(tl.int64)
        lo = atom * 0
        hi = lo + n_edges
        for _ in tl.static_range(0, SEARCH_STEPS):
            mid = (lo + hi) // 2
            in_bounds = mid < n_edges
            value = tl.load(center + mid, mask=in_bounds, other=-1)
            move_right = in_bounds & (value < atom)
            lo = tl.where(move_right, mid + 1, lo)
            hi = tl.where(move_right, hi, mid)

        value = tl.load(center + lo, mask=lo < n_edges, other=-1)
        graph_idx = tl.load(batch_idx + atom)
        tl.atomic_add(
            isolated_atom_counts + graph_idx,
            1,
            mask=value != atom,
        )


    @triton.jit
    def _line_graph_status_kernel(
        pair_counts,
        atom_neighbor_counts,
        status,
        N_ATOMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        atoms = tl.arange(0, BLOCK)
        mask = atoms < N_ATOMS
        pairs = tl.load(pair_counts + atoms, mask=mask, other=0)
        neighbors = tl.load(atom_neighbor_counts + atoms, mask=mask, other=0)
        tl.store(status, tl.sum(pairs, axis=0))
        tl.store(
            status + 1,
            tl.sum((mask & (neighbors == 0)).to(tl.int64), axis=0),
        )
        tl.store(status + 2, tl.sum(neighbors.to(tl.int64), axis=0))
        tl.store(status + 3, tl.max(neighbors.to(tl.int64), axis=0))

    @triton.jit(do_not_specialize=["TOTAL_PAIRS"])
    def _line_graph_materialize_kernel(
        counts,
        offsets,
        pair_offsets,
        short_de,
        short_ude,
        out,
        TOTAL_PAIRS,
        N_ATOMS: tl.constexpr,
        LOG_N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = rows < TOTAL_PAIRS

        lo = tl.full((BLOCK,), 0, dtype=tl.int64)
        hi = tl.full((BLOCK,), N_ATOMS - 1, dtype=tl.int64)
        for _ in tl.static_range(0, LOG_N):
            mid = (lo + hi + 1) // 2
            mid_offsets = tl.load(pair_offsets + mid)
            take_upper = mid_offsets <= rows
            lo = tl.where(take_upper, mid, lo)
            hi = tl.where(take_upper, hi, mid - 1)

        atom = lo
        count = tl.load(counts + atom)
        atom_pair_offset = tl.load(pair_offsets + atom)
        local_idx = rows - atom_pair_offset
        denom = tl.maximum(count - 1, 1)
        first_in_group = local_idx // denom
        second_raw = local_idx - first_in_group * denom
        second_in_group = second_raw + (second_raw >= first_in_group)
        atom_offset = tl.load(offsets + atom)
        first_pos = atom_offset + first_in_group
        second_pos = atom_offset + second_in_group

        first_ude = tl.load(short_ude + first_pos, mask=mask, other=0)
        first_de = tl.load(short_de + first_pos, mask=mask, other=0)
        second_ude = tl.load(short_ude + second_pos, mask=mask, other=0)
        second_de = tl.load(short_de + second_pos, mask=mask, other=0)
        base = rows * 5
        tl.store(out + base + 0, atom, mask=mask)
        tl.store(out + base + 1, first_ude, mask=mask)
        tl.store(out + base + 2, first_de, mask=mask)
        tl.store(out + base + 3, second_ude, mask=mask)
        tl.store(out + base + 4, second_de, mask=mask)

else:
    _count_isolated_sorted_kernel = None
    _line_graph_status_kernel = None
    _line_graph_materialize_kernel = None


def _count_isolated_atoms(
    center: Tensor,
    batch_idx: Tensor,
    *,
    n_atoms: int,
    n_graphs: int,
) -> Tensor:
    isolated_atom_counts = torch.zeros(
        n_graphs,
        dtype=torch.int32,
        device=center.device,
    )
    if n_atoms == 0:
        return isolated_atom_counts
    if _triton_available and center.is_cuda:
        # ``neighbor_list_nvidia`` sorts by batch and global center atom. A
        # fixed-width lower bound avoids bincount + compare + reduction kernels.
        _count_isolated_sorted_kernel[(n_atoms,)](
            center,
            batch_idx,
            isolated_atom_counts,
            center.shape[0],
            SEARCH_STEPS=32,
            num_warps=1,
        )
        return isolated_atom_counts

    neighbor_counts = torch.bincount(center, minlength=n_atoms)
    for graph_idx in range(n_graphs):
        atom_ids = torch.where(batch_idx == graph_idx)[0]
        isolated_atom_counts[graph_idx] = (neighbor_counts[atom_ids] == 0).sum()
    return isolated_atom_counts


def _line_graph_status(
    pair_counts: Tensor,
    atom_neighbor_counts: Tensor,
) -> Tensor:
    """[total pairs, zero-candidate atoms, candidates, max candidates per atom].

    One vector so the single-system build reads all of it in one D2H; the last
    entry is the neighbor-capacity overflow check.
    """
    n_atoms = int(pair_counts.shape[0])
    if (
        _triton_available
        and pair_counts.is_cuda
        and n_atoms <= 65_536
    ):
        status = torch.empty(4, dtype=torch.int64, device=pair_counts.device)
        block = triton.next_power_of_2(max(n_atoms, 1))
        _line_graph_status_kernel[(1,)](
            pair_counts,
            atom_neighbor_counts,
            status,
            N_ATOMS=n_atoms,
            BLOCK=block,
            num_warps=8,
        )
        return status
    return torch.stack(
        [
            pair_counts.sum(),
            (atom_neighbor_counts == 0).sum(),
            atom_neighbor_counts.sum(),
            atom_neighbor_counts.max().long(),
        ]
    )


def _exact_isolated_count(
    center: Tensor,
    n_atoms: int,
    atom_neighbor_counts: Tensor | None = None,
    neighbor_capacity: int | None = None,
) -> tuple[Tensor, int]:
    count = (
        torch.bincount(center, minlength=n_atoms) == 0
    ).sum().reshape(1)
    if atom_neighbor_counts is None:
        return count, int(count.item())
    # No status vector on this branch: fold the capacity check into its read.
    isolated_count, max_count = torch.cat(
        [count, atom_neighbor_counts.max().reshape(1).long()]
    ).tolist()
    _check_neighbor_capacity(max_count, neighbor_capacity)
    return count, isolated_count


def _materialize_line_graph_triton(
    counts: Tensor,
    offsets: Tensor,
    pair_offsets: Tensor,
    short_de: Tensor,
    short_ude: Tensor,
    total_pairs: int,
    n_atoms: int,
) -> Tensor | None:
    if not (_LINE_GRAPH_TRITON and _triton_available and short_de.is_cuda):
        return None
    with nvtx_range("_build_line_graph_materialize"):
        out = torch.empty((total_pairs, 5), dtype=torch.int32, device=short_de.device)
        block = _LINE_GRAPH_TRITON_BLOCK
        grid = (triton.cdiv(total_pairs, block),)
        _line_graph_materialize_kernel[grid](
            counts,
            offsets,
            pair_offsets,
            short_de,
            short_ude,
            out,
            TOTAL_PAIRS=total_pairs,
            N_ATOMS=n_atoms,
            LOG_N=max(1, (n_atoms - 1).bit_length()),
            BLOCK=block,
            num_warps=4,
        )
        return out


def _build_line_graph(
    center: Tensor,
    directed2undirected: Tensor,
    dist: Tensor,
    n_atoms: int,
    cutoff: float,
    atom_neighbor_counts: Tensor | None = None,
    presorted: bool = False,
    neighbor_capacity: int | None = None,
) -> Tensor | tuple[Tensor, Tensor, int]:
    """Ordered triplets; with ``atom_neighbor_counts`` also isolated counts.

    ``atom_neighbor_counts`` (raw nvalchemiops counts, single-system path)
    requires ``neighbor_capacity``: the overflow check rides on the status read
    and raises ``_NeighborOverflow`` before any isolated count is derived.
    """
    with nvtx_range("_build_line_graph"):
        device = center.device
        short_de = torch.where(dist < cutoff)[0]
        if short_de.numel() == 0:
            empty = torch.empty((0, 5), dtype=torch.int32, device=device)
            if atom_neighbor_counts is None:
                return empty
            if center.shape[0] == 0:
                # Capacity already checked by _build_graphs_from_payload.
                isolated = torch.full((1,), n_atoms, dtype=torch.int64, device=device)
                return empty, isolated, n_atoms
            isolated, isolated_count = _exact_isolated_count(
                center,
                n_atoms,
                atom_neighbor_counts,
                neighbor_capacity,
            )
            return empty, isolated, isolated_count

        if presorted:
            # center is non-decreasing (neighbor-list argsort postcondition)
            # and short_de is an ascending index subset, so the stable sort
            # is the identity permutation.
            sorted_center = center[short_de]
        else:
            sorted_center, order = center[short_de].sort(stable=True)
            short_de = short_de[order]
        short_ude = directed2undirected[short_de].long()
        counts = torch.bincount(sorted_center.int(), minlength=n_atoms).long()
        pair_counts = counts * (counts - 1)
        isolated_atom_counts = None
        isolated_count = 0
        if atom_neighbor_counts is None:
            total_pairs = int(pair_counts.sum().item())
        else:
            graph_status = _line_graph_status(pair_counts, atom_neighbor_counts)
            total_pairs, candidate_isolated, candidate_edges, max_count = (
                int(value) for value in graph_status.tolist()
            )
            _check_neighbor_capacity(max_count, neighbor_capacity)
            if candidate_edges == center.shape[0]:
                isolated_atom_counts = graph_status[1:2]
                isolated_count = candidate_isolated
            else:
                isolated_atom_counts, isolated_count = _exact_isolated_count(
                    center,
                    n_atoms,
                )
        if total_pairs == 0:
            empty = torch.empty((0, 5), dtype=torch.int32, device=device)
            if isolated_atom_counts is None:
                return empty
            return empty, isolated_atom_counts, isolated_count

        offsets = torch.zeros(n_atoms, dtype=torch.long, device=device)
        offsets[1:] = counts[:-1].cumsum(0)
        pair_offsets = torch.zeros(n_atoms, dtype=torch.long, device=device)
        pair_offsets[1:] = pair_counts[:-1].cumsum(0)
        triton_line_graph = _materialize_line_graph_triton(
            counts,
            offsets,
            pair_offsets,
            short_de,
            short_ude,
            total_pairs,
            n_atoms,
        )
        if triton_line_graph is not None:
            if isolated_atom_counts is None:
                return triton_line_graph
            return triton_line_graph, isolated_atom_counts, isolated_count

        pair_atom = torch.repeat_interleave(torch.arange(n_atoms, device=device), pair_counts)
        local_idx = torch.arange(total_pairs, device=device) - pair_offsets[pair_atom]
        group_size = counts[pair_atom]
        first_in_group = local_idx // (group_size - 1)
        second_raw = local_idx % (group_size - 1)
        second_in_group = second_raw + (second_raw >= first_in_group).long()
        first_pos = offsets[pair_atom] + first_in_group
        second_pos = offsets[pair_atom] + second_in_group

        line_graph = torch.stack(
            [
                pair_atom.int(),
                short_ude[first_pos].int(),
                short_de[first_pos].int(),
                short_ude[second_pos].int(),
                short_de[second_pos].int(),
            ],
            dim=1,
        )
        if isolated_atom_counts is None:
            return line_graph
        return line_graph, isolated_atom_counts, isolated_count


def _extract_atoms(atoms_list, device: torch.device):
    atomic_numbers = []
    frac_coords = []
    positions = []
    cells = []
    pbc = []
    batch_idx = []
    atom_offsets = [0]
    compositions = []

    # Pack one or more ASE Atoms objects into a single batched CUDA payload.
    for graph_idx, atoms in enumerate(atoms_list):
        cell = torch.from_numpy(atoms.get_cell().array.astype(np.float32)).to(device)
        pos = torch.from_numpy(atoms.get_positions().astype(np.float32)).to(device)
        frac = small_matmul(pos, torch.linalg.inv(cell)) % 1.0
        n_atoms = len(atoms)
        atomic_numbers.append(
            torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=device)
        )
        frac_coords.append(frac)
        positions.append(small_matmul(frac, cell))
        cells.append(cell)
        pbc.append(torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=device))
        batch_idx.append(torch.full((n_atoms,), graph_idx, dtype=torch.int32, device=device))
        atom_offsets.append(atom_offsets[-1] + n_atoms)
        compositions.append(atoms.get_chemical_formula())

    return {
        "atomic_numbers": torch.cat(atomic_numbers),
        "frac_coords": torch.cat(frac_coords),
        "positions": torch.cat(positions),
        "cells": torch.stack(cells),
        "pbc": torch.stack(pbc),
        "batch_idx": torch.cat(batch_idx),
        "atom_offsets": atom_offsets,
        "compositions": compositions,
    }


def _extract_tensor_system(
    positions: Tensor,
    cell: Tensor,
    atomic_numbers: Tensor,
    pbc: Tensor | None,
    device: torch.device,
    composition: str | None = None,
):
    positions = positions.to(device=device, dtype=torch.float32)
    cell = cell.to(device=device, dtype=torch.float32)
    atomic_numbers = atomic_numbers.to(device=device, dtype=torch.int32)
    if pbc is None:
        pbc = torch.ones(3, dtype=torch.bool, device=device)
    else:
        pbc = pbc.to(device=device, dtype=torch.bool)

    if cell.dim() != 2 or tuple(cell.shape) != (3, 3):
        raise ValueError(f"cell must have shape (3, 3), got {tuple(cell.shape)}")
    if positions.dim() != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}")
    if atomic_numbers.dim() != 1 or atomic_numbers.shape[0] != positions.shape[0]:
        raise ValueError(
            "atomic_numbers must be a 1D tensor with the same length as positions"
        )
    if pbc.dim() != 1 or pbc.shape[0] != 3:
        raise ValueError(f"pbc must have shape (3,), got {tuple(pbc.shape)}")

    frac = small_matmul(positions, torch.linalg.inv(cell)) % 1.0
    n_atoms = int(positions.shape[0])
    return {
        "atomic_numbers": atomic_numbers,
        "frac_coords": frac,
        "positions": small_matmul(frac, cell),
        "cells": cell.unsqueeze(0),
        "pbc": pbc.unsqueeze(0),
        "batch_idx": torch.zeros(n_atoms, dtype=torch.int32, device=device),
        "atom_offsets": [0, n_atoms],
        "compositions": [composition or ""],
    }


def _split_graphs(
    payload: dict[str, Tensor | list],
    center: Tensor,
    neighbor: Tensor,
    images: Tensor,
    directed2undirected: Tensor,
    undirected2directed: Tensor,
    line_graph: Tensor,
    isolated_atom_counts: Tensor,
    atom_graph_cutoff: float,
    line_graph_cutoff: float,
) -> list[RadiusGraph]:
    # Single-system fast path (the MD case): atom_start==0 and all edges/triplets
    # belong to graph 0, so global indices ARE local — skip the per-graph
    # where/unique/searchsorted/boolean-mask remapping (~7 D2H syncs/step).
    if len(payload["atom_offsets"]) - 1 == 1:
        return [
            RadiusGraph(
                atomic_number=payload["atomic_numbers"],
                atom_frac_coord=payload["frac_coords"],
                atom_graph=torch.stack([center, neighbor], dim=1).int(),
                neighbor_image=images.float(),
                directed2undirected=directed2undirected.int(),
                undirected2directed=undirected2directed.int(),
                line_graph=line_graph.int(),
                lattice=payload["cells"][0],
                graph_id=None,
                mp_id=None,
                composition=payload["compositions"][0],
                atom_graph_cutoff=atom_graph_cutoff,
                line_graph_cutoff=line_graph_cutoff,
                atom_target_sorted=True,
                line_atom_sorted=True,
                isolated_atom_count=isolated_atom_counts[0:1],
                topology_evidence=_GPU_BUILDER_EVIDENCE,
            )
        ]

    graphs = []
    for graph_idx in range(len(payload["atom_offsets"]) - 1):
        atom_start = payload["atom_offsets"][graph_idx]
        atom_end = payload["atom_offsets"][graph_idx + 1]
        edge_ids = torch.where(payload["batch_idx"][center] == graph_idx)[0].long()
        local_d2u_global = directed2undirected[edge_ids].long()
        unique_ude_global, local_d2u = torch.unique(
            local_d2u_global,
            sorted=True,
            return_inverse=True,
        )
        local_line_graph = line_graph[
            (line_graph[:, 0] >= atom_start) & (line_graph[:, 0] < atom_end)
        ].clone().long()
        if local_line_graph.numel() != 0:
            local_line_graph[:, 0] -= atom_start
            local_line_graph[:, 1] = torch.searchsorted(
                unique_ude_global,
                local_line_graph[:, 1].contiguous(),
            )
            local_line_graph[:, 2] = torch.searchsorted(
                edge_ids,
                local_line_graph[:, 2].contiguous(),
            )
            local_line_graph[:, 3] = torch.searchsorted(
                unique_ude_global,
                local_line_graph[:, 3].contiguous(),
            )
            local_line_graph[:, 4] = torch.searchsorted(
                edge_ids,
                local_line_graph[:, 4].contiguous(),
            )

        graphs.append(
            RadiusGraph(
                atomic_number=payload["atomic_numbers"][atom_start:atom_end],
                atom_frac_coord=payload["frac_coords"][atom_start:atom_end],
                atom_graph=torch.stack(
                    [center[edge_ids] - atom_start, neighbor[edge_ids] - atom_start],
                    dim=1,
                ).int(),
                neighbor_image=images[edge_ids].float(),
                directed2undirected=local_d2u.int(),
                undirected2directed=torch.searchsorted(
                    edge_ids,
                    undirected2directed[unique_ude_global].long(),
                ).int(),
                line_graph=local_line_graph.int(),
                lattice=payload["cells"][graph_idx],
                graph_id=None,
                mp_id=None,
                composition=payload["compositions"][graph_idx],
                atom_graph_cutoff=atom_graph_cutoff,
                line_graph_cutoff=line_graph_cutoff,
                atom_target_sorted=True,
                line_atom_sorted=True,
                isolated_atom_count=isolated_atom_counts[
                    graph_idx : graph_idx + 1
                ],
                topology_evidence=_GPU_BUILDER_EVIDENCE,
            )
        )
    return graphs


def _build_graphs_from_payload(
    payload: dict[str, Tensor | list],
    atom_graph_cutoff: float,
    line_graph_cutoff: float,
    device: torch.device,
    *,
    check_isolated_atoms: bool = True,
    neighbor_capacity: int,
) -> list[RadiusGraph]:
    """One build with ``neighbor_capacity`` slots per atom.

    Raises ``_NeighborOverflow`` if a neighbor row was truncated; always before
    any isolated-atom error, which a truncated list could fake.
    """
    n_atoms = int(payload["atomic_numbers"].shape[0])
    n_graphs = len(payload["atom_offsets"]) - 1
    atom_neighbor_counts = None
    if n_graphs == 1:
        center, neighbor, images, dist, atom_neighbor_counts = neighbor_list_nvidia(
            payload["positions"],
            payload["cells"],
            atom_graph_cutoff,
            payload["batch_idx"],
            payload["pbc"],
            device=device,
            return_num_neighbors=True,
            max_neighbors=neighbor_capacity,
            cell_list=payload.get("cell_list"),
        )
        if center.numel() == 0:
            # No edges means no status read in _build_line_graph to fold the
            # check into; this rare branch pays one small read of its own.
            _check_neighbor_capacity(
                _max_neighbor_count(atom_neighbor_counts),
                neighbor_capacity,
            )
            if check_isolated_atoms:
                raise_if_isolated_atoms(n_atoms)
        isolated_atom_counts = None
    else:
        # Retries internally (one small read); multi-graph builds are not
        # the per-step MD path.
        center, neighbor, images, dist = neighbor_list_nvidia(
            payload["positions"],
            payload["cells"],
            atom_graph_cutoff,
            payload["batch_idx"],
            payload["pbc"],
            device=device,
            max_neighbors=neighbor_capacity,
        )
        isolated_atom_counts = _count_isolated_atoms(
            center,
            payload["batch_idx"],
            n_atoms=n_atoms,
            n_graphs=n_graphs,
        )
        if check_isolated_atoms:
            raise_if_isolated_atoms(isolated_atom_counts)
    presorted = _PRESORTED_BUILD and n_graphs == 1
    directed2undirected, undirected2directed = _build_d2u(
        center,
        neighbor,
        images,
        n_atoms,
        presorted=presorted,
    )
    line_graph_result = _build_line_graph(
        center,
        directed2undirected,
        dist,
        n_atoms,
        line_graph_cutoff,
        atom_neighbor_counts=atom_neighbor_counts,
        presorted=presorted,
        neighbor_capacity=neighbor_capacity,
    )
    if n_graphs != 1:
        line_graph = line_graph_result
    else:
        line_graph, isolated_atom_counts, isolated_count = line_graph_result
        if check_isolated_atoms:
            raise_if_isolated_atoms(isolated_count)
    return _split_graphs(
        payload,
        center,
        neighbor,
        images,
        directed2undirected,
        undirected2directed,
        line_graph,
        isolated_atom_counts,
        atom_graph_cutoff,
        line_graph_cutoff,
    )


def _build_graphs(
    payload: dict[str, Tensor | list],
    atom_graph_cutoff: float,
    line_graph_cutoff: float,
    device: torch.device,
    *,
    check_isolated_atoms: bool,
    max_neighbors: int | None,
) -> tuple[list[RadiusGraph], int]:
    """Build, rebuilding from scratch with more slots until nothing overflows.

    Returns the graphs and the neighbor capacity that sufficed. The edge set
    and its argsort key are capacity-independent, so a rebuilt graph is
    bitwise the graph a large enough first capacity would have produced.
    """
    capacity = _initial_max_neighbors(atom_graph_cutoff, max_neighbors)
    while True:
        try:
            graphs = _build_graphs_from_payload(
                payload,
                atom_graph_cutoff,
                line_graph_cutoff,
                device,
                check_isolated_atoms=check_isolated_atoms,
                neighbor_capacity=capacity,
            )
        except _NeighborOverflow as overflow:
            capacity = _grown_max_neighbors(capacity, overflow.max_count)
        else:
            return graphs, capacity


class TensorGraphBuilder:
    """Cached single-system tensor graph builder for fixed-cell GPU MD.

    ``max_neighbors`` is the initial neighbor slot count per atom (None: the
    volume estimate). Overflow grows it and the builder keeps the grown value,
    so a dense trajectory pays the rebuild once rather than every step.
    ``line_graph_cutoff=None``: edges only; ``build`` then returns a ragged
    :class:`~fastmd.adapters.protocol.GraphBatch` (no padding rows) instead
    of MatRIS's ``RadiusGraph``.
    """

    def __init__(
        self,
        *,
        cell: Tensor,
        atomic_numbers: Tensor,
        pbc: Tensor | None = None,
        atom_graph_cutoff: float = 6.0,
        line_graph_cutoff: float | None = 4.5,
        device: torch.device | str = "cuda",
        composition: str | None = None,
        check_isolated_atoms: bool = True,
        max_neighbors: int | None = None,
        positions_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = _cuda_device(device)
        self.cell = cell.to(device=self.device, dtype=torch.float32)
        if positions_dtype not in (torch.float32, torch.float64):
            raise ValueError(f"positions_dtype must be torch.float32 or torch.float64, got {positions_dtype}")
        if positions_dtype != torch.float32 and line_graph_cutoff is not None:
            raise ValueError("float64 positions need the edges-only mode (line_graph_cutoff=None)")
        # Edges only: the dtype of the batch's positions and cell (float64 from the caller's cell).
        self.positions_dtype = positions_dtype
        self.model_cell = self.cell if positions_dtype == torch.float32 else cell.to(device=self.device,
                                                                                     dtype=positions_dtype)
        self.inv_cell = torch.linalg.inv(self.cell)
        self.cells = self.cell.unsqueeze(0)
        self.atomic_numbers = atomic_numbers.to(device=self.device, dtype=torch.int32)
        if pbc is None:
            pbc = torch.ones(3, dtype=torch.bool, device=self.device)
        else:
            pbc = pbc.to(device=self.device, dtype=torch.bool)
        self.pbc = pbc.unsqueeze(0)
        self.n_atoms = int(self.atomic_numbers.shape[0])
        self.batch_idx = torch.zeros(self.n_atoms, dtype=torch.int32, device=self.device)
        self.atom_offsets = [0, self.n_atoms]
        self.compositions = [composition or ""]
        self.atom_graph_cutoff = float(atom_graph_cutoff)
        self.line_graph_cutoff = None if line_graph_cutoff is None else float(line_graph_cutoff)
        self.check_isolated_atoms = bool(check_isolated_atoms)
        self.max_neighbors = _initial_max_neighbors(
            self.atom_graph_cutoff, max_neighbors
        )

        if self.cell.dim() != 2 or tuple(self.cell.shape) != (3, 3):
            raise ValueError(f"cell must have shape (3, 3), got {tuple(self.cell.shape)}")
        if self.pbc.dim() != 2 or tuple(self.pbc.shape) != (1, 3):
            raise ValueError(f"pbc must have shape (3,), got {tuple(self.pbc.squeeze(0).shape)}")
        if self.atomic_numbers.dim() != 1:
            raise ValueError("atomic_numbers must be a 1D tensor")
        # Sized once at neighbor_list_nvidia's candidate cutoff (None with the 0.2 API).
        self.cell_list = cell_list_for_fixed_cell(
            self.cells, self.pbc, self.atom_graph_cutoff + 1e-6, self.n_atoms, self.device
        )

    @torch.no_grad()
    def build(self, positions: Tensor) -> RadiusGraph | GraphBatch:
        with nvtx_range("tensors_to_graph_gpu"):
            given = positions
            positions = positions.to(device=self.device, dtype=torch.float32)
            if positions.dim() != 2 or positions.shape[1] != 3:
                raise ValueError(
                    f"positions must have shape (N, 3), got {tuple(positions.shape)}"
                )
            if positions.shape[0] != self.n_atoms:
                raise ValueError(
                    "positions must have the same length as atomic_numbers "
                    f"({positions.shape[0]} != {self.n_atoms})"
                )
            frac = small_matmul(positions, self.inv_cell) % 1.0
            payload = {
                "atomic_numbers": self.atomic_numbers,
                "frac_coords": frac,
                "positions": small_matmul(frac, self.cell),
                "cells": self.cells,
                "pbc": self.pbc,
                "batch_idx": self.batch_idx,
                "atom_offsets": self.atom_offsets,
                "compositions": self.compositions,
                "cell_list": self.cell_list,
            }
            if self.line_graph_cutoff is None:
                if self.positions_dtype != torch.float32:
                    # The caller's positions minus the whole lattice vectors the float32 wrap removed.
                    shift = torch.floor(small_matmul(positions, self.inv_cell)).to(self.positions_dtype)
                    cell = self.model_cell
                    payload["model_positions"] = (
                        given.to(device=self.device, dtype=self.positions_dtype)
                        - (shift[:, 0:1] * cell[0] + shift[:, 1:2] * cell[1] + shift[:, 2:3] * cell[2]))
                return self._build_edges(payload)
            graphs, self.max_neighbors = _build_graphs(
                payload,
                self.atom_graph_cutoff,
                self.line_graph_cutoff,
                self.device,
                check_isolated_atoms=self.check_isolated_atoms,
                max_neighbors=self.max_neighbors,
            )
            return graphs[0]

    __call__ = build

    def _build_edges(self, payload: dict) -> GraphBatch:
        """Edges only: the sorted neighbour list and one status read (truncated rows, isolated atoms)."""
        while True:
            center, neighbor, images, _, counts = neighbor_list_nvidia(
                payload["positions"],
                self.cells,
                self.atom_graph_cutoff,
                self.batch_idx,
                self.pbc,
                device=self.device,
                return_num_neighbors=True,
                max_neighbors=self.max_neighbors,
                cell_list=self.cell_list,
            )
            # Edges per atom (an atom with no edge is isolated); index_add sizes nothing from the data.
            edges_per_atom = torch.zeros(self.n_atoms, dtype=torch.int64, device=self.device).index_add_(
                0, center, torch.ones_like(center))
            max_count, isolated = torch.stack(
                [counts.max().long(), (edges_per_atom == 0).sum()]
            ).tolist()
            if max_count <= self.max_neighbors:
                break
            self.max_neighbors = _grown_max_neighbors(self.max_neighbors, max_count)
        if self.check_isolated_atoms:
            raise_if_isolated_atoms(isolated)
        return GraphBatch(
            positions=payload.get("model_positions", payload["positions"]),
            frac_coords=payload["frac_coords"],
            cell=self.model_cell,
            species=self.atomic_numbers,
            n_real=self.n_atoms,
            edge_index=torch.stack([center, neighbor], dim=1).int(),
            edge_image=images.float(),
            n_edges=torch.full((1,), center.shape[0], dtype=torch.int64, device=self.device),
            cutoff=self.atom_graph_cutoff,
            padding=None,
            composition=self.compositions[0],
            evidence=_GPU_BUILDER_EVIDENCE,
        )


@torch.no_grad()
def atoms_to_edges_gpu(atoms, atom_graph_cutoff: float, device: torch.device | str = "cuda", *,
                       check_isolated_atoms: bool = True, positions_dtype: torch.dtype = torch.float32) -> GraphBatch:
    """The edges-only ``GraphBatch`` of ASE ``atoms`` (a model without triplets; ``atoms_to_graph_gpu``
    is MatRIS's graph): a ragged build of ``TensorGraphBuilder(line_graph_cutoff=None)`` from the
    float32 cell and positions as ``atoms_to_graph_gpu`` packs them (the same fractional coordinates).
    ``positions_dtype=torch.float64``: the batch's positions and cell are float64 (from the float64
    arrays; the neighbour search stays float32)."""
    device = _cuda_device(device)
    float64 = positions_dtype == torch.float64
    with torch.cuda.device(device):
        builder = TensorGraphBuilder(
            cell=torch.from_numpy(atoms.get_cell().array.astype(np.float64 if float64 else np.float32)).to(device),
            atomic_numbers=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=device),
            pbc=torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=device),
            atom_graph_cutoff=atom_graph_cutoff,
            line_graph_cutoff=None,
            device=device,
            composition=atoms.get_chemical_formula(),
            check_isolated_atoms=check_isolated_atoms,
            positions_dtype=positions_dtype,
        )
        return builder.build(torch.from_numpy(atoms.get_positions().astype(np.float64 if float64 else
                                                                         np.float32)).to(device))


@torch.no_grad()
def tensors_to_graph_gpu(
    positions: Tensor,
    cell: Tensor,
    atomic_numbers: Tensor,
    pbc: Tensor | None = None,
    atom_graph_cutoff: float = 6.0,
    line_graph_cutoff: float = 4.5,
    device: torch.device | str = "cuda",
    composition: str | None = None,
    check_isolated_atoms: bool = True,
    *,
    max_neighbors: int | None = None,
) -> RadiusGraph:
    """Build a single-system RadiusGraph directly from CUDA/CPU tensors.

    This is the GPU-resident MD entry point: callers can keep positions, cell,
    and atomic numbers as tensors and avoid the per-step ASE -> NumPy -> CUDA
    packing performed by :func:`atoms_to_graph_gpu`. Neighbor overflow of the
    initial ``max_neighbors`` slots is rebuilt per call (see
    :class:`TensorGraphBuilder` to keep the grown capacity).
    """
    with nvtx_range("tensors_to_graph_gpu"):
        device = _cuda_device(device)
        payload = _extract_tensor_system(
            positions=positions,
            cell=cell,
            atomic_numbers=atomic_numbers,
            pbc=pbc,
            device=device,
            composition=composition,
        )
        graphs, _ = _build_graphs(
            payload,
            atom_graph_cutoff,
            line_graph_cutoff,
            device,
            check_isolated_atoms=check_isolated_atoms,
            max_neighbors=max_neighbors,
        )
        return graphs[0]


@torch.no_grad()
def atoms_to_graph_gpu(
    atoms,
    atom_graph_cutoff: float = 6.0,
    line_graph_cutoff: float = 4.5,
    device: torch.device | str = "cuda",
    check_isolated_atoms: bool = True,
    *,
    max_neighbors: int | None = None,
) -> RadiusGraph | list[RadiusGraph]:
    device = _cuda_device(device)
    batched = not hasattr(atoms, "get_positions")
    atoms_list = list(atoms) if batched else [atoms]
    payload = _extract_atoms(atoms_list, device)

    graphs, _ = _build_graphs(
        payload,
        atom_graph_cutoff,
        line_graph_cutoff,
        device,
        check_isolated_atoms=check_isolated_atoms,
        max_neighbors=max_neighbors,
    )
    return graphs if batched else graphs[0]
