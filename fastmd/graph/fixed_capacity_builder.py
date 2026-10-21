"""Capture-safe fixed-capacity graph construction for GPU-resident MD.

The regular GPU graph builder returns ragged edge and triplet tensors.  This
module keeps all allocations static and fills unused rows with padding (the
adapter's ``PaddingRule``: MatRIS routes them to dummy sink atoms), so graph
construction can live in the same CUDA Graph as the model and integrator.
The builder publishes its buffers as a ``GraphBatch`` (``builder.batch``);
without a line graph cutoff it builds the edges only. With ``GhostSpec`` its
rows are one rank's P>1 local domain (owned slots, then ghost slots).
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fastmd._nvtx import nvtx_range
from fastmd.adapters.protocol import GraphBatch
from fastmd.adapters.protocol import PaddingRule
from fastmd.graph.geometry import small_matmul
from fastmd.graph.gpu_graph_builder import MAX_IMAGE
from fastmd.graph.gpu_graph_builder import _cuda_device
from fastmd.graph.gpu_graph_builder import _env_flag as _env_flag_fixed
from fastmd.graph.matris_topology import matris_builder_evidence
from fastmd.graph.radiusgraph import RadiusGraph

try:
    import triton
    import triton.language as tl

    from fastmd.graph.neighbor_list_backend import NeighborOverflowError
    from fastmd.graph.neighbor_list_backend import allocate_cell_list
    from fastmd.graph.neighbor_list_backend import batch_build_cell_list
    from fastmd.graph.neighbor_list_backend import batch_query_cell_list
    from fastmd.graph.neighbor_list_backend import estimate_batch_cell_list_sizes
    from fastmd.graph.neighbor_list_backend import estimate_max_neighbors
    from fastmd.graph.neighbor_list_backend import require

    require()
    _fixed_builder_available = True
    _fixed_builder_unavailable = None
except ImportError as error:
    triton = None
    tl = None
    _fixed_builder_available = False
    _fixed_builder_unavailable = str(error)


_FIXED_BUILDER_EVIDENCE = matris_builder_evidence(
    "fastmd.fixed_capacity_radius_graph",
    atom_target_sorted=True,
    line_owner_sorted=True,
    fixed_capacity=True,
    incidence_consistent=False,
)


if _fixed_builder_available:

    @triton.jit
    def _ghost_edge_predicates(atom, neighbor, slots, valid_slot, edge_mask, short_mask,
                               n_owned_slots, BMAX: tl.constexpr):
        """HAS_GHOSTS: the caller's per-candidate edge and short decisions.

        The caller decides every candidate slot with its own arithmetic (the
        P>1 epoch: exactly the rebuild's torch expressions on the model
        coordinates, so both agree to the last bit at a cutoff tie). An edge
        also needs an owned end (rows below ``n_owned_slots``), a short row an
        owned center; ghost-ghost pairs are dropped.
        """
        base = atom * BMAX + slots
        edge_valid = valid_slot & (tl.load(edge_mask + base, mask=valid_slot, other=0) != 0)
        short_valid = valid_slot & (tl.load(short_mask + base, mask=valid_slot, other=0) != 0)
        edge_valid = edge_valid & ((atom < n_owned_slots) | (neighbor < n_owned_slots))
        short_valid = short_valid & edge_valid & (atom < n_owned_slots)
        return edge_valid, short_valid


    @triton.jit
    def _ghost_edge_rule(edge_valid, atom, neighbor, n_owned_slots, GHOST_EDGES: tl.constexpr):
        """Ghost rows with the builder's own cutoff decision, then the edge rule
        (``GhostSpec.edges``): 1, ``"receiver"``, keeps the edges of an owned centre; 2,
        ``"either"``, the edges with at least one owned end. Ghost-ghost pairs are dropped
        (with a line graph the caller also keeps the short edges of owned centres only)."""
        if GHOST_EDGES == 1:
            edge_valid = edge_valid & (atom < n_owned_slots)
        else:
            edge_valid = edge_valid & ((atom < n_owned_slots) | (neighbor < n_owned_slots))
        return edge_valid


    @triton.jit
    def _count_neighbors_kernel(
        positions,
        cell,
        neighbor_matrix,
        neighbor_shifts,
        num_neighbors,
        wrap_delta,
        edge_counts,
        short_counts,
        isolated_atom_count,
        owned_count,
        n_owned_slots,
        edge_mask,
        short_mask,
        system_of_row,
        N: tl.constexpr,
        BMAX: tl.constexpr,
        BLOCK: tl.constexpr,
        EDGE_CUTOFF2: tl.constexpr,
        LINE_CUTOFF2: tl.constexpr,
        HAS_GHOSTS: tl.constexpr = False,
        HAS_SHORT: tl.constexpr = True,
        GHOST_EDGES: tl.constexpr = 0,
        SYSTEM_CELLS: tl.constexpr = False,
    ):
        atom = tl.program_id(0).to(tl.int64)
        slots = tl.arange(0, BLOCK)
        slot_mask = slots < BMAX
        count = tl.load(num_neighbors + atom)
        valid_slot = slot_mask & (slots < count)
        neighbor = tl.load(
            neighbor_matrix + atom * BMAX + slots,
            mask=valid_slot,
            other=0,
        )
        neighbor = tl.minimum(neighbor, N - 1)
        if SYSTEM_CELLS:
            # Systems with cells of their own ([systems, 3, 3]): the cell of the atom's system (no
            # candidate leaves its system).
            cell = cell + tl.load(system_of_row + atom).to(tl.int64) * 9

        px = tl.load(positions + atom * 3)
        py = tl.load(positions + atom * 3 + 1)
        pz = tl.load(positions + atom * 3 + 2)
        nx = tl.load(positions + neighbor * 3, mask=valid_slot, other=0.0)
        ny = tl.load(positions + neighbor * 3 + 1, mask=valid_slot, other=0.0)
        nz = tl.load(positions + neighbor * 3 + 2, mask=valid_slot, other=0.0)

        shift_base = (atom * BMAX + slots) * 3
        sx = tl.load(neighbor_shifts + shift_base, mask=valid_slot, other=0).to(tl.float32)
        sy = tl.load(neighbor_shifts + shift_base + 1, mask=valid_slot, other=0).to(tl.float32)
        sz = tl.load(neighbor_shifts + shift_base + 2, mask=valid_slot, other=0).to(tl.float32)
        sx += tl.load(wrap_delta + neighbor * 3, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3
        )
        sy += tl.load(wrap_delta + neighbor * 3 + 1, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3 + 1
        )
        sz += tl.load(wrap_delta + neighbor * 3 + 2, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3 + 2
        )
        dx = nx - px + sx * tl.load(cell) + sy * tl.load(cell + 3) + sz * tl.load(cell + 6)
        dy = ny - py + sx * tl.load(cell + 1) + sy * tl.load(cell + 4) + sz * tl.load(cell + 7)
        dz = nz - pz + sx * tl.load(cell + 2) + sy * tl.load(cell + 5) + sz * tl.load(cell + 8)
        dist2 = dx * dx + dy * dy + dz * dz
        edge_valid = valid_slot & (dist2 < EDGE_CUTOFF2)
        short_valid = valid_slot & (dist2 < LINE_CUTOFF2)
        if HAS_GHOSTS:
            edge_valid, short_valid = _ghost_edge_predicates(
                atom, neighbor, slots, valid_slot, edge_mask, short_mask, n_owned_slots, BMAX,
            )
        if GHOST_EDGES != 0:
            edge_valid = _ghost_edge_rule(edge_valid, atom, neighbor, n_owned_slots, GHOST_EDGES)
            if HAS_SHORT:
                # A line graph's short edge (so a triplet) needs an owned centre.
                short_valid = short_valid & edge_valid & (atom < n_owned_slots)
        edge_count = tl.sum(edge_valid.to(tl.int32), axis=0)
        tl.store(edge_counts + atom, edge_count)
        if HAS_SHORT:
            tl.store(short_counts + atom, tl.sum(short_valid.to(tl.int32), axis=0))
        if HAS_GHOSTS or GHOST_EDGES != 0:
            # Only real owned rows count as isolated (inactive slots have no edges; a ghost's owner counts it).
            tl.atomic_add(
                isolated_atom_count, 1,
                mask=(edge_count == 0) & (atom < tl.load(owned_count)),
            )
        else:
            tl.atomic_add(isolated_atom_count, 1, mask=edge_count == 0)


    @triton.jit
    def _materialize_neighbors_kernel(
        positions,
        cell,
        neighbor_matrix,
        neighbor_shifts,
        num_neighbors,
        wrap_delta,
        edge_offsets,
        center_out,
        neighbor_out,
        image_out,
        distance_out,
        short_flag_out,
        short_edge_matrix,
        n_owned_slots,
        edge_mask,
        short_mask,
        system_of_row,
        N: tl.constexpr,
        BMAX: tl.constexpr,
        E_CAP: tl.constexpr,
        BLOCK: tl.constexpr,
        EDGE_CUTOFF2: tl.constexpr,
        LINE_CUTOFF2: tl.constexpr,
        HAS_GHOSTS: tl.constexpr = False,
        HAS_SHORT: tl.constexpr = True,
        GHOST_EDGES: tl.constexpr = 0,
        SYSTEM_CELLS: tl.constexpr = False,
    ):
        atom = tl.program_id(0).to(tl.int64)
        slots = tl.arange(0, BLOCK)
        slot_mask = slots < BMAX
        count = tl.load(num_neighbors + atom)
        valid_slot = slot_mask & (slots < count)
        neighbor = tl.load(
            neighbor_matrix + atom * BMAX + slots,
            mask=valid_slot,
            other=0,
        )
        neighbor = tl.minimum(neighbor, N - 1)
        if SYSTEM_CELLS:
            cell = cell + tl.load(system_of_row + atom).to(tl.int64) * 9

        px = tl.load(positions + atom * 3)
        py = tl.load(positions + atom * 3 + 1)
        pz = tl.load(positions + atom * 3 + 2)
        nx = tl.load(positions + neighbor * 3, mask=valid_slot, other=0.0)
        ny = tl.load(positions + neighbor * 3 + 1, mask=valid_slot, other=0.0)
        nz = tl.load(positions + neighbor * 3 + 2, mask=valid_slot, other=0.0)

        shift_base = (atom * BMAX + slots) * 3
        sx_i = tl.load(neighbor_shifts + shift_base, mask=valid_slot, other=0)
        sy_i = tl.load(neighbor_shifts + shift_base + 1, mask=valid_slot, other=0)
        sz_i = tl.load(neighbor_shifts + shift_base + 2, mask=valid_slot, other=0)
        sx_i += tl.load(wrap_delta + neighbor * 3, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3
        )
        sy_i += tl.load(wrap_delta + neighbor * 3 + 1, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3 + 1
        )
        sz_i += tl.load(wrap_delta + neighbor * 3 + 2, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3 + 2
        )
        sx = sx_i.to(tl.float32)
        sy = sy_i.to(tl.float32)
        sz = sz_i.to(tl.float32)
        dx = nx - px + sx * tl.load(cell) + sy * tl.load(cell + 3) + sz * tl.load(cell + 6)
        dy = ny - py + sx * tl.load(cell + 1) + sy * tl.load(cell + 4) + sz * tl.load(cell + 7)
        dz = nz - pz + sx * tl.load(cell + 2) + sy * tl.load(cell + 5) + sz * tl.load(cell + 8)
        dist2 = dx * dx + dy * dy + dz * dz
        edge_valid = valid_slot & (dist2 < EDGE_CUTOFF2)
        short_valid = valid_slot & (dist2 < LINE_CUTOFF2)
        if HAS_GHOSTS:
            edge_valid, short_valid = _ghost_edge_predicates(
                atom, neighbor, slots, valid_slot, edge_mask, short_mask, n_owned_slots, BMAX,
            )
        if GHOST_EDGES != 0:
            edge_valid = _ghost_edge_rule(edge_valid, atom, neighbor, n_owned_slots, GHOST_EDGES)
            if HAS_SHORT:
                short_valid = short_valid & edge_valid & (atom < n_owned_slots)

        edge_rank = tl.cumsum(edge_valid.to(tl.int32), axis=0) - 1
        edge_index = tl.load(edge_offsets + atom) + edge_rank
        edge_store = edge_valid & (edge_index < E_CAP)
        tl.store(center_out + edge_index, atom, mask=edge_store)
        tl.store(neighbor_out + edge_index, neighbor, mask=edge_store)
        tl.store(image_out + edge_index * 3, sx_i, mask=edge_store)
        tl.store(image_out + edge_index * 3 + 1, sy_i, mask=edge_store)
        tl.store(image_out + edge_index * 3 + 2, sz_i, mask=edge_store)
        tl.store(distance_out + edge_index, tl.sqrt(dist2), mask=edge_store)
        if HAS_SHORT:
            tl.store(short_flag_out + edge_index, short_valid, mask=edge_store)

            short_rank = tl.cumsum(short_valid.to(tl.int32), axis=0) - 1
            short_store = short_valid & (short_rank < BMAX) & (edge_index < E_CAP)
            tl.store(
                short_edge_matrix + atom * BMAX + short_rank,
                edge_index,
                mask=short_store,
            )


    @triton.jit
    def _pad_edges_kernel(
        total_edges,
        center,
        neighbor,
        images,
        distances,
        short_flags,
        N: tl.constexpr,
        N_DUMMY: tl.constexpr,
        E_CAP: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_SHORT: tl.constexpr = True,
    ):
        rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        total = tl.load(total_edges)
        total_clamped = tl.minimum(total, E_CAP)
        mask = (rows < E_CAP) & (rows >= total_clamped)
        real_u = total_clamped // 2
        n_pad_u = E_CAP // 2 - real_u
        pad_pair = (rows - total_clamped) // 2
        sink = N + (pad_pair * N_DUMMY) // tl.maximum(n_pad_u, 1)
        tl.store(center + rows, sink, mask=mask)
        tl.store(neighbor + rows, sink, mask=mask)
        tl.store(images + rows * 3, 3, mask=mask)
        tl.store(images + rows * 3 + 1, 3, mask=mask)
        tl.store(images + rows * 3 + 2, 3, mask=mask)
        tl.store(distances + rows, 1.0e6, mask=mask)
        if HAS_SHORT:
            tl.store(short_flags + rows, 0, mask=mask)


    @triton.jit(do_not_specialize=["image_x", "image_y", "image_z"])
    def _pad_self_loops_kernel(
        total_edges,
        center,
        neighbor,
        images,
        vec_override,
        image_x,
        image_y,
        image_z,
        pad_x,
        pad_y,
        pad_z,
        owned_count,
        N: tl.constexpr,
        E_CAP: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_OVERRIDE: tl.constexpr,
        OWNED_LOOPS: tl.constexpr = False,
    ):
        """Self-loop padding: padded row k is a loop of real atom k mod N
        (round robin) with a constant image; with an override buffer, its
        rows past the real edges get the pad vector and the real rows 0.
        The image and the vector are runtime values (no compile per cell).
        ``OWNED_LOOPS`` (ghost rows): the loops go round robin over the real
        owned rows ``[0, owned_count)`` only (row 0 if there are none)."""
        rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        total_clamped = tl.minimum(tl.load(total_edges), E_CAP)
        in_range = rows < E_CAP
        mask = in_range & (rows >= total_clamped)
        if OWNED_LOOPS:
            atom = (rows - total_clamped) % tl.maximum(tl.load(owned_count).to(tl.int64), 1)
        else:
            atom = (rows - total_clamped) % N
        tl.store(center + rows, atom, mask=mask)
        tl.store(neighbor + rows, atom, mask=mask)
        tl.store(images + rows * 3, image_x, mask=mask)
        tl.store(images + rows * 3 + 1, image_y, mask=mask)
        tl.store(images + rows * 3 + 2, image_z, mask=mask)
        if HAS_OVERRIDE:
            pad = rows >= total_clamped
            tl.store(vec_override + rows * 3, tl.where(pad, pad_x, 0.0), mask=in_range)
            tl.store(vec_override + rows * 3 + 1, tl.where(pad, pad_y, 0.0), mask=in_range)
            tl.store(vec_override + rows * 3 + 2, tl.where(pad, pad_z, 0.0), mask=in_range)


    @triton.jit
    def _pad_system_self_loops_kernel(
        total_edges,
        center,
        neighbor,
        images,
        system_of_row,
        loop_images,
        N: tl.constexpr,
        E_CAP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``_pad_self_loops_kernel`` (far self-loops) for systems with cells of their own: padded row k is
        a loop of row k mod N (round robin) with the far image of that row's system (``loop_images``,
        ``[systems, 3]`` int32)."""
        rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        total_clamped = tl.minimum(tl.load(total_edges), E_CAP)
        mask = (rows < E_CAP) & (rows >= total_clamped)
        atom = (rows - total_clamped) % N
        system = tl.load(system_of_row + atom, mask=mask, other=0).to(tl.int64)
        tl.store(center + rows, atom, mask=mask)
        tl.store(neighbor + rows, atom, mask=mask)
        tl.store(images + rows * 3, tl.load(loop_images + system * 3, mask=mask, other=0), mask=mask)
        tl.store(images + rows * 3 + 1, tl.load(loop_images + system * 3 + 1, mask=mask, other=0), mask=mask)
        tl.store(images + rows * 3 + 2, tl.load(loop_images + system * 3 + 2, mask=mask, other=0), mask=mask)


    @triton.jit(do_not_specialize=["image_x", "image_y", "image_z", "length"])
    def _pad_loop_pairs_kernel(
        total_edges,
        center,
        neighbor,
        images,
        distances,
        short_flags,
        image_x,
        image_y,
        image_z,
        length,
        N: tl.constexpr,
        E_CAP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Far self-loop padding with a line graph: padded pair p (rows 2p, 2p + 1 past the real
        edges) is the two opposite self-loops, images +image and -image, of real atom p mod N (round
        robin), never a short edge. The image and length are runtime values (no compile per cell)."""
        rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        total_clamped = tl.minimum(tl.load(total_edges), E_CAP)
        mask = (rows < E_CAP) & (rows >= total_clamped)
        pad = rows - total_clamped
        atom = (pad // 2) % N
        sign = 1 - 2 * (pad % 2).to(tl.int32)
        tl.store(center + rows, atom, mask=mask)
        tl.store(neighbor + rows, atom, mask=mask)
        tl.store(images + rows * 3, sign * image_x, mask=mask)
        tl.store(images + rows * 3 + 1, sign * image_y, mask=mask)
        tl.store(images + rows * 3 + 2, sign * image_z, mask=mask)
        tl.store(distances + rows, length, mask=mask)
        tl.store(short_flags + rows, 0, mask=mask)


    @triton.jit
    def _window_flags_kernel(
        edge_count,
        invalid_pairs,
        overflow,
        isolated_count,
        window_overflow,
        window_capacity_overflow,
        window_max_edges,
        window_max_isolated,
        E_CAP: tl.constexpr,
        CHECK_ISOLATED: tl.constexpr,
    ):
        """One edges-only build's contribution to a whole-step window's device flags, in one program:
        the window overflow (|= the build's overflow, and isolated atoms when checked), the capacity
        overflow (|= edges > E_CAP or invalid pairs) and the maxima of the edge and isolated-atom counts."""
        edges = tl.load(edge_count)
        isolated = tl.load(isolated_count)
        flag = tl.load(overflow) != 0
        if CHECK_ISOLATED:
            flag = flag | (isolated > 0)
        tl.store(window_overflow, (tl.load(window_overflow) != 0) | flag)
        capacity = (edges > E_CAP) | (tl.load(invalid_pairs) != 0)
        tl.store(window_capacity_overflow, (tl.load(window_capacity_overflow) != 0) | capacity)
        tl.store(window_max_edges, tl.maximum(tl.load(window_max_edges), edges))
        tl.store(window_max_isolated, tl.maximum(tl.load(window_max_isolated), isolated))


    @triton.jit
    def _max_nan(a, b):
        return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


    @triton.jit(do_not_specialize=["skin_limit"])
    def _float64_prologue_kernel(
        positions,
        inv_cell,
        cell,
        cell64,
        reference_positions,
        reference_wraps,
        capacity_valid,
        frac_coords,
        wrapped,
        wrap_delta,
        model_positions,
        max_displacement,
        candidate_valid,
        isolated_atom_count,
        skin_limit,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One program over all N atoms: what a candidate-reusing build computes before its edges,
        from float64 coordinates. The float32 wrap (fractional coordinates, wrapped positions, the
        wrap delta) feeds the neighbour kernels; the model's float64 positions are the coordinates
        minus the same whole lattice vectors; the largest displacement since the refresh decides
        whether the candidates are still valid; the isolated-atom count starts at zero."""
        rows = tl.arange(0, BLOCK)
        mask = rows < N
        p0 = tl.load(positions + rows * 3, mask=mask, other=0.0)
        p1 = tl.load(positions + rows * 3 + 1, mask=mask, other=0.0)
        p2 = tl.load(positions + rows * 3 + 2, mask=mask, other=0.0)
        q0 = p0.to(tl.float32)
        q1 = p1.to(tl.float32)
        q2 = p2.to(tl.float32)
        f0 = q0 * tl.load(inv_cell) + q1 * tl.load(inv_cell + 3) + q2 * tl.load(inv_cell + 6)
        f1 = q0 * tl.load(inv_cell + 1) + q1 * tl.load(inv_cell + 4) + q2 * tl.load(inv_cell + 7)
        f2 = q0 * tl.load(inv_cell + 2) + q1 * tl.load(inv_cell + 5) + q2 * tl.load(inv_cell + 8)
        w0 = tl.floor(f0)
        w1 = tl.floor(f1)
        w2 = tl.floor(f2)
        r0 = f0 - w0
        r1 = f1 - w1
        r2 = f2 - w2
        tl.store(frac_coords + rows * 3, r0, mask=mask)
        tl.store(frac_coords + rows * 3 + 1, r1, mask=mask)
        tl.store(frac_coords + rows * 3 + 2, r2, mask=mask)
        tl.store(wrapped + rows * 3, r0 * tl.load(cell) + r1 * tl.load(cell + 3) + r2 * tl.load(cell + 6), mask=mask)
        tl.store(wrapped + rows * 3 + 1, r0 * tl.load(cell + 1) + r1 * tl.load(cell + 4) + r2 * tl.load(cell + 7),
                 mask=mask)
        tl.store(wrapped + rows * 3 + 2, r0 * tl.load(cell + 2) + r1 * tl.load(cell + 5) + r2 * tl.load(cell + 8),
                 mask=mask)
        tl.store(wrap_delta + rows * 3, w0.to(tl.int32) - tl.load(reference_wraps + rows * 3, mask=mask, other=0),
                 mask=mask)
        tl.store(wrap_delta + rows * 3 + 1,
                 w1.to(tl.int32) - tl.load(reference_wraps + rows * 3 + 1, mask=mask, other=0), mask=mask)
        tl.store(wrap_delta + rows * 3 + 2,
                 w2.to(tl.int32) - tl.load(reference_wraps + rows * 3 + 2, mask=mask, other=0), mask=mask)
        v0 = w0.to(tl.float64)
        v1 = w1.to(tl.float64)
        v2 = w2.to(tl.float64)
        tl.store(model_positions + rows * 3,
                 p0 - (v0 * tl.load(cell64) + v1 * tl.load(cell64 + 3) + v2 * tl.load(cell64 + 6)), mask=mask)
        tl.store(model_positions + rows * 3 + 1,
                 p1 - (v0 * tl.load(cell64 + 1) + v1 * tl.load(cell64 + 4) + v2 * tl.load(cell64 + 7)), mask=mask)
        tl.store(model_positions + rows * 3 + 2,
                 p2 - (v0 * tl.load(cell64 + 2) + v1 * tl.load(cell64 + 5) + v2 * tl.load(cell64 + 8)), mask=mask)
        d0 = q0 - tl.load(reference_positions + rows * 3, mask=mask, other=0.0)
        d1 = q1 - tl.load(reference_positions + rows * 3 + 1, mask=mask, other=0.0)
        d2 = q2 - tl.load(reference_positions + rows * 3 + 2, mask=mask, other=0.0)
        # NaN-propagating (a non-finite state must fail the skin test and reach the replay's check).
        largest = tl.sqrt(tl.reduce(tl.where(mask, d0 * d0 + d1 * d1 + d2 * d2, 0.0), 0, _max_nan))
        tl.store(max_displacement, largest)
        tl.store(candidate_valid, (largest * 2.0 <= skin_limit) & (tl.load(capacity_valid) != 0))
        tl.store(isolated_atom_count, 0)  # the count kernel that follows adds the atoms without edges


    @triton.jit
    def _materialize_unsorted_edges_kernel(
        positions,
        cell,
        neighbor_matrix,
        neighbor_shifts,
        num_neighbors,
        wrap_delta,
        edge_counts,
        edge_offsets,
        center_out,
        neighbor_out,
        image_out,
        distance_out,
        edge_index_out,
        edge_image_out,
        N: tl.constexpr,
        BMAX: tl.constexpr,
        E_CAP: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EDGE_CUTOFF2: tl.constexpr,
    ):
        """Edges only, unsorted, one program per atom: the rows of ``_materialize_neighbors_kernel``
        (the same arithmetic) at the offset this program sums from the counts of
        ``_count_neighbors_kernel`` (it writes its own ``edge_offsets`` entry: no cumsum), written to the
        builder's buffers and to the model's (``edge_index`` [E, 2] int32, ``edge_image`` [E, 3] float32)."""
        atom = tl.program_id(0).to(tl.int64)
        atoms = tl.arange(0, BLOCK_N)
        offset = tl.sum(tl.load(edge_counts + atoms, mask=atoms < atom, other=0), axis=0)
        tl.store(edge_offsets + atom + 1, offset + tl.load(edge_counts + atom))
        tl.store(edge_offsets + atom * 0, offset * 0, mask=atom == 0)
        slots = tl.arange(0, BLOCK)
        slot_mask = slots < BMAX
        count = tl.load(num_neighbors + atom)
        valid_slot = slot_mask & (slots < count)
        neighbor = tl.load(
            neighbor_matrix + atom * BMAX + slots,
            mask=valid_slot,
            other=0,
        )
        neighbor = tl.minimum(neighbor, N - 1)

        px = tl.load(positions + atom * 3)
        py = tl.load(positions + atom * 3 + 1)
        pz = tl.load(positions + atom * 3 + 2)
        nx = tl.load(positions + neighbor * 3, mask=valid_slot, other=0.0)
        ny = tl.load(positions + neighbor * 3 + 1, mask=valid_slot, other=0.0)
        nz = tl.load(positions + neighbor * 3 + 2, mask=valid_slot, other=0.0)

        shift_base = (atom * BMAX + slots) * 3
        sx_i = tl.load(neighbor_shifts + shift_base, mask=valid_slot, other=0)
        sy_i = tl.load(neighbor_shifts + shift_base + 1, mask=valid_slot, other=0)
        sz_i = tl.load(neighbor_shifts + shift_base + 2, mask=valid_slot, other=0)
        sx_i += tl.load(wrap_delta + neighbor * 3, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3
        )
        sy_i += tl.load(wrap_delta + neighbor * 3 + 1, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3 + 1
        )
        sz_i += tl.load(wrap_delta + neighbor * 3 + 2, mask=valid_slot, other=0) - tl.load(
            wrap_delta + atom * 3 + 2
        )
        sx = sx_i.to(tl.float32)
        sy = sy_i.to(tl.float32)
        sz = sz_i.to(tl.float32)
        dx = nx - px + sx * tl.load(cell) + sy * tl.load(cell + 3) + sz * tl.load(cell + 6)
        dy = ny - py + sx * tl.load(cell + 1) + sy * tl.load(cell + 4) + sz * tl.load(cell + 7)
        dz = nz - pz + sx * tl.load(cell + 2) + sy * tl.load(cell + 5) + sz * tl.load(cell + 8)
        dist2 = dx * dx + dy * dy + dz * dz
        edge_valid = valid_slot & (dist2 < EDGE_CUTOFF2)

        edge_rank = tl.cumsum(edge_valid.to(tl.int32), axis=0) - 1
        edge_index = offset + edge_rank
        edge_store = edge_valid & (edge_index < E_CAP)
        tl.store(center_out + edge_index, atom, mask=edge_store)
        tl.store(neighbor_out + edge_index, neighbor, mask=edge_store)
        tl.store(image_out + edge_index * 3, sx_i, mask=edge_store)
        tl.store(image_out + edge_index * 3 + 1, sy_i, mask=edge_store)
        tl.store(image_out + edge_index * 3 + 2, sz_i, mask=edge_store)
        tl.store(distance_out + edge_index, tl.sqrt(dist2), mask=edge_store)
        tl.store(edge_index_out + edge_index * 2, atom, mask=edge_store)
        tl.store(edge_index_out + edge_index * 2 + 1, neighbor, mask=edge_store)
        tl.store(edge_image_out + edge_index * 3, sx, mask=edge_store)
        tl.store(edge_image_out + edge_index * 3 + 1, sy, mask=edge_store)
        tl.store(edge_image_out + edge_index * 3 + 2, sz, mask=edge_store)


    @triton.jit(do_not_specialize=["image_x", "image_y", "image_z"])
    def _finish_unsorted_edges_kernel(
        edge_offsets,
        center,
        neighbor,
        images,
        edge_index_out,
        edge_image_out,
        candidate_valid,
        overflow,
        image_x,
        image_y,
        image_z,
        N: tl.constexpr,
        E_CAP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """After ``_materialize_unsorted_edges_kernel``: the self-loop padding rows of
        ``_pad_self_loops_kernel`` (no override vector) in the builder's and the model's edge buffers,
        and the build's overflow flag (more edges than rows, or candidates the skin no longer covers)."""
        pid = tl.program_id(0).to(tl.int64)
        rows = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        total = tl.load(edge_offsets + N)
        total_clamped = tl.minimum(total, E_CAP)
        mask = (rows < E_CAP) & (rows >= total_clamped)
        atom = (rows - total_clamped) % N
        tl.store(center + rows, atom, mask=mask)
        tl.store(neighbor + rows, atom, mask=mask)
        tl.store(images + rows * 3, image_x, mask=mask)
        tl.store(images + rows * 3 + 1, image_y, mask=mask)
        tl.store(images + rows * 3 + 2, image_z, mask=mask)
        tl.store(edge_index_out + rows * 2, atom, mask=mask)
        tl.store(edge_index_out + rows * 2 + 1, atom, mask=mask)
        tl.store(edge_image_out + rows * 3, image_x.to(tl.float32), mask=mask)
        tl.store(edge_image_out + rows * 3 + 1, image_y.to(tl.float32), mask=mask)
        tl.store(edge_image_out + rows * 3 + 2, image_z.to(tl.float32), mask=mask)
        tl.store(overflow, (total > E_CAP) | (tl.load(candidate_valid) == 0), mask=pid == 0)


# Candidate-list state a ``candidates_from`` builder shares with its source.
_CANDIDATE_STATE = (
    "cell_list_cache",
    "neighbor_matrix",
    "neighbor_shifts",
    "num_neighbors",
    "reference_positions",
    "reference_wraps",
    "wrap_delta",
    "_candidate_capacity_valid",
)


if _fixed_builder_available:

    @triton.jit(do_not_specialize=["e_capacity", "u_capacity", "t_capacity"])
    def _line_graph_kernel(
        pair_offsets,
        short_counts,
        short_edge_matrix,
        directed2undirected,
        edge_total,
        overflow,
        overflow_line_graph,
        line_graph,
        e_capacity,
        u_capacity,
        t_capacity,
        N: tl.constexpr,
        BMAX: tl.constexpr,
        N_DUMMY: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
        BLOCK: tl.constexpr,
        LOOP_PADDING: tl.constexpr = False,
    ):
        """One row per triplet slot: the exact triplets of the short-edge
        groups, then the padding (sink or loop), or the all-padding topology on overflow
        (the arithmetic of the torch expression it replaced, row by row).
        ``LOOP_PADDING`` (far self-loops): a padded triplet is centred on the
        real atom of its padded pair, not on a sink atom; on overflow the
        substitute is the all-loop topology."""
        rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        in_range = rows < t_capacity
        out = line_graph + rows * 5
        if tl.load(overflow) != 0:
            source = overflow_line_graph + rows * 5
            for col in tl.static_range(5):
                tl.store(out + col, tl.load(source + col, mask=in_range, other=0), mask=in_range)
        else:
            total_triplets = tl.load(pair_offsets + N)
            t_clamped = tl.minimum(total_triplets, t_capacity)
            real_edges = tl.minimum(tl.load(edge_total), e_capacity)
            valid = rows < t_clamped
            # The last atom whose pair offset is <= row (searchsorted right - 1).
            lo = tl.zeros([BLOCK], dtype=tl.int64)
            hi = lo + N
            for _ in tl.static_range(SEARCH_STEPS):
                mid = (lo + hi + 1) // 2
                take = tl.load(pair_offsets + mid) <= rows
                lo = tl.where(take, mid, lo)
                hi = tl.where(take, hi, mid - 1)
            atom = tl.minimum(lo, N - 1)
            local = rows - tl.load(pair_offsets + atom)
            denom = tl.maximum(tl.load(short_counts + atom) - 1, 1)
            first = local // denom
            second_raw = local - first * denom
            second = second_raw + (second_raw >= first).to(tl.int64)
            first_de = tl.load(short_edge_matrix + atom * BMAX + tl.minimum(first, BMAX - 1),
                               mask=in_range, other=0).to(tl.int64)
            second_de = tl.load(short_edge_matrix + atom * BMAX + tl.minimum(second, BMAX - 1),
                                mask=in_range, other=0).to(tl.int64)
            first_u = tl.load(directed2undirected + tl.minimum(tl.maximum(first_de, 0), e_capacity - 1),
                              mask=in_range, other=0).to(tl.int64)
            second_u = tl.load(directed2undirected + tl.minimum(tl.maximum(second_de, 0), e_capacity - 1),
                               mask=in_range, other=0).to(tl.int64)
            real_u = real_edges // 2
            pad_id = tl.maximum(rows - t_clamped, 0)
            pad_pair = pad_id % tl.maximum(u_capacity - real_u, 1)
            pad_u = real_u + pad_pair
            pad_de = real_edges + 2 * pad_pair
            if LOOP_PADDING:
                pad_atom = pad_pair % N
            else:
                pad_atom = N + pad_id * N_DUMMY // tl.maximum(t_capacity - t_clamped, 1)
            tl.store(out, tl.where(valid, atom, pad_atom).to(tl.int32), mask=in_range)
            tl.store(out + 1, tl.where(valid, first_u, pad_u).to(tl.int32), mask=in_range)
            tl.store(out + 2, tl.where(valid, first_de, pad_de).to(tl.int32), mask=in_range)
            tl.store(out + 3, tl.where(valid, second_u, pad_u).to(tl.int32), mask=in_range)
            tl.store(out + 4, tl.where(valid, second_de, pad_de).to(tl.int32), mask=in_range)


@dataclass
class FixedCapacityGraphStatus:
    edge_count: Tensor
    undirected_count: Tensor
    triplet_count: Tensor
    overflow: Tensor
    invalid_pairs: Tensor
    candidate_valid: Tensor
    max_displacement: Tensor
    isolated_atom_count: Tensor
    # No candidate row was truncated at the last refresh (part of candidate_valid).
    candidate_capacity_valid: Tensor | None = None


@dataclass(frozen=True)
class GhostSpec:
    """Rows of a P>1 local domain (unified step 3.4, ``HAS_GHOSTS``).

    Rows ``[0, n_owned)`` are owned slots and the remaining rows ghost slots;
    ``set_atoms`` installs the global IDs and the count of real owned rows of
    an epoch, and the candidate refresh leaves unused slots (rows not in
    ``set_atoms``' active rows) without candidates. A directed edge needs an owned end and a short edge
    (so a triplet) an owned center (``_ghost_edge_predicates``); ghost-ghost
    pairs are dropped. ``atom_ids`` (global IDs) are kept for the caller.

    With a line graph and ``edges=None`` (MatRIS's topology epoch) the caller
    decides every candidate slot (``build(predicate=...)``). Otherwise
    ``edges`` names the edge rule and the builder decides each pair itself,
    with the arithmetic of a single-system build: ``"receiver"`` (edges only)
    keeps the edges of an owned centre (each edge on one rank: the centre's
    owner), ``"either"`` the edges with at least one owned end (a boundary
    edge on both ranks); with a line graph (``"either"`` only, sink padding)
    the short edges, so the triplets, are those of owned centres.
    """

    n_owned: int
    # The builder's edge rule, "receiver" (edges only) or "either"; None with a line graph: the caller's predicate.
    edges: str | None = None


def fixed_capacity_exceeded(status, builder) -> Tensor:
    """Device predicate: the real graph in ``status`` does not fit ``builder``.

    A line graph's padding (sink atoms or far self-loops) needs at least one
    padding undirected pair, because ``_materialize_line_graph`` routes every
    padding triplet to one.
    A real directed-edge count equal to ``e_capacity`` leaves no such pair,
    so it is a capacity overflow just like ``edges > e_capacity`` (the same
    rule as the bucketed model graph, which requires ``u_capacity > u``).
    An edges-only builder routes nothing to a padding row, so its edges fit
    up to ``e_capacity`` inclusive.
    """
    if getattr(builder, "edges_only", False):
        return (status.edge_count[0] > builder.e_capacity) | status.invalid_pairs[0]
    return (
        (status.edge_count[0] >= builder.e_capacity)
        | (status.triplet_count[0] > builder.t_capacity)
        | status.invalid_pairs[0]
    )


@dataclass(frozen=True)
class StaticizationInvariantReport:
    edge_count: int
    undirected_count: int
    triplet_count: int
    real_edges_isolated: bool
    padding_edges_isolated: bool
    real_triplets_isolated: bool
    padding_triplets_isolated: bool
    reverse_pairs_valid: bool
    sorted_segments_valid: bool
    reference_exact: bool | None


@torch.no_grad()
def validate_staticization_invariants(
    graph: RadiusGraph,
    status: FixedCapacityGraphStatus,
    *,
    n_real: int,
    reference: RadiusGraph | None = None,
) -> StaticizationInvariantReport:
    """Synchronously validate the sink-padding contract for tests/artifacts (sink padding only)."""
    edge_count = int(status.edge_count.item())
    undirected_count = int(status.undirected_count.item())
    triplet_count = int(status.triplet_count.item())
    if bool(status.overflow.item()):
        raise AssertionError("cannot validate an overflowed fixed-capacity graph")

    atom_graph = graph.atom_graph.long()
    line_graph = graph.line_graph.long()
    d2u = graph.directed2undirected.long()
    u2d = graph.undirected2directed.long()
    real_edges_isolated = bool((atom_graph[:edge_count] < n_real).all().item())
    padding_edges_isolated = bool((atom_graph[edge_count:] >= n_real).all().item())
    real_triplets_isolated = bool(
        (
            (line_graph[:triplet_count, 0] < n_real)
            & (line_graph[:triplet_count, 1] < undirected_count)
            & (line_graph[:triplet_count, 2] < edge_count)
            & (line_graph[:triplet_count, 3] < undirected_count)
            & (line_graph[:triplet_count, 4] < edge_count)
        ).all().item()
    )
    padding_triplets_isolated = bool(
        (
            (line_graph[triplet_count:, 0] >= n_real)
            & (line_graph[triplet_count:, 1] >= undirected_count)
            & (line_graph[triplet_count:, 2] >= edge_count)
            & (line_graph[triplet_count:, 3] >= undirected_count)
            & (line_graph[triplet_count:, 4] >= edge_count)
        ).all().item()
    )

    real_pair_rows = torch.argsort(d2u[:edge_count]).reshape(undirected_count, 2)
    first = atom_graph[real_pair_rows[:, 0]]
    second = atom_graph[real_pair_rows[:, 1]]
    first_image = graph.neighbor_image[real_pair_rows[:, 0]].long()
    second_image = graph.neighbor_image[real_pair_rows[:, 1]].long()
    reverse_pairs_valid = bool(
        (
            (first[:, 0] == second[:, 1])
            & (first[:, 1] == second[:, 0])
            & (first_image == -second_image).all(dim=1)
            & (d2u[u2d[:undirected_count]] == torch.arange(
                undirected_count, device=d2u.device
            )).all()
        ).all().item()
    )
    sorted_segments_valid = bool(
        (atom_graph[1:edge_count, 0] >= atom_graph[: edge_count - 1, 0]).all().item()
        and (
            line_graph[1:triplet_count, 0]
            >= line_graph[: triplet_count - 1, 0]
        ).all().item()
    )

    reference_exact = None
    if reference is not None:
        fields = (
            "atomic_number",
            "atom_frac_coord",
            "atom_graph",
            "neighbor_image",
            "directed2undirected",
            "undirected2directed",
            "line_graph",
            "lattice",
        )
        reference_exact = all(
            torch.equal(getattr(graph, field), getattr(reference, field))
            for field in fields
        )

    report = StaticizationInvariantReport(
        edge_count=edge_count,
        undirected_count=undirected_count,
        triplet_count=triplet_count,
        real_edges_isolated=real_edges_isolated,
        padding_edges_isolated=padding_edges_isolated,
        real_triplets_isolated=real_triplets_isolated,
        padding_triplets_isolated=padding_triplets_isolated,
        reverse_pairs_valid=reverse_pairs_valid,
        sorted_segments_valid=sorted_segments_valid,
        reference_exact=reference_exact,
    )
    checks = (
        real_edges_isolated,
        padding_edges_isolated,
        real_triplets_isolated,
        padding_triplets_isolated,
        reverse_pairs_valid,
        sorted_segments_valid,
        reference_exact is not False,
    )
    if not all(checks):
        raise AssertionError(f"staticization invariant failure: {report}")
    return report


_SINK_PADDING = PaddingRule("sink_atoms")
# GhostSpec.edges -> the kernels' GHOST_EDGES (0: no edges-only ghost rows).
_GHOST_EDGE_RULES = {"receiver": 1, "either": 2}
# The most atoms whose float64 build prologue runs as one Triton program (FixedCapacityGraphBuilder).
_FLOAT64_PROLOGUE_ATOMS = 4096
# The most atoms whose unsorted float64 edges are written in three kernels after the prologue (each
# program sums the counts before its atom).
_UNSORTED_EDGES_ATOMS = 1024


def radius_graph_view(batch: GraphBatch) -> RadiusGraph:
    """MatRIS's ``RadiusGraph`` over a fixed-capacity builder's ``batch``: the same tensors, no copy."""
    if batch.triplets is None:
        raise ValueError("a RadiusGraph needs the line graph: this graph batch has edges only")
    return RadiusGraph(
        graph_id=None,
        mp_id=None,
        composition=batch.composition,
        atomic_number=batch.species,
        atom_frac_coord=batch.frac_coords,
        lattice=batch.cell,
        neighbor_image=batch.edge_image,
        atom_graph=batch.edge_index,
        atom_graph_cutoff=batch.cutoff,
        line_graph=batch.triplets,
        line_graph_cutoff=batch.triplet_cutoff,
        directed2undirected=batch.pair_d2u,
        undirected2directed=batch.pair_u2d,
        atom_target_sorted=True,
        line_atom_sorted=True,
        topology_evidence=batch.evidence,
    )


def line_graph_batch(graph: RadiusGraph, *, positions: Tensor | None = None,
                     n_real: int | None = None) -> GraphBatch:
    """A ``GraphBatch`` over MatRIS's ``RadiusGraph`` (edges, pairs and triplets: the same tensors, no
    copy), for a model with a line graph that the engine differentiates (its position leaf is
    ``positions``). ``positions``: the wrapped Cartesian coordinates, by default ``frac @ cell`` as the
    builders compute them; ``n_real``: the real rows of a padded graph (default: every row)."""
    if positions is None:
        positions = small_matmul(graph.atom_frac_coord, graph.lattice)
    rows = graph.atomic_number.shape[0]
    return GraphBatch(
        positions=positions,
        frac_coords=graph.atom_frac_coord,
        cell=graph.lattice,
        species=graph.atomic_number,
        n_real=rows if n_real is None else int(n_real),
        edge_index=graph.atom_graph,
        edge_image=graph.neighbor_image,
        n_edges=None,
        cutoff=graph.atom_graph_cutoff,
        padding=None if n_real is None else _SINK_PADDING,
        triplet_cutoff=graph.line_graph_cutoff,
        pair_d2u=graph.directed2undirected,
        pair_u2d=graph.undirected2directed,
        triplets=graph.line_graph,
        composition=graph.composition or "",
        evidence=getattr(graph, "topology_evidence", None),
    )


def _batch_itself(batch: GraphBatch) -> GraphBatch:
    return batch


def self_loop_padding(padding: PaddingRule, cell: Tensor, cutoff: float) -> tuple[tuple[int, int, int], float,
                                                                                  tuple[float, float, float] | None]:
    """The padded self-loops of ``padding`` (far_self_loops, exact_rc_loops) in ``cell``: their
    integer image, their length and the override vector (exact_rc_loops). Host only."""
    if padding.kind == "exact_rc_loops":
        vector = tuple(map(float, padding.pad_vector or (cutoff, 0.0, 0.0)))
        # The FP32 edge a model computes from it must not be inside the cutoff (nor of length 0).
        length = float(torch.linalg.vector_norm(torch.tensor(vector, dtype=torch.float32)))
        if not length >= float(torch.tensor(cutoff, dtype=torch.float32)) or length == 0.0:
            raise ValueError(f"exact_rc_loops: pad_vector {vector} is shorter than the cutoff {cutoff}")
        return (0, 0, 0), length, vector
    if padding.kind != "far_self_loops":
        raise ValueError(f"{padding.kind} padding has no self-loops")
    # A whole number of the longest lattice vector, at least min_length and
    # twice the cutoff (far beyond it in FP32 too).
    lengths = cell.detach().cpu().double().norm(dim=1)
    axis = int(torch.argmax(lengths))
    longest = float(lengths[axis])
    if not longest > 0.0:
        raise ValueError("far self-loop padding needs a cell with a nonzero lattice vector")
    multiple = max(1, math.ceil(max(padding.min_length, 2.0 * cutoff) / longest))
    image = [0, 0, 0]
    image[axis] = multiple
    return (image[0], image[1], image[2]), multiple * longest, None


class FixedCapacityGraphBuilder:
    """Build a single-system padded graph with static shapes.

    With a line graph cutoff the builder builds MatRIS's topology (edges,
    undirected pairs, triplets) padded with sink atoms, or with far
    self-loops where ``padding`` asks for them; without one
    (``line_graph_cutoff=None``, ``t_capacity=0``) it builds the edges only,
    padded by ``padding``. Its buffers are published as ``batch``
    (:class:`~fastmd.adapters.protocol.GraphBatch`), and ``build`` returns
    ``prepare(batch)``, computed once here: the model's own input over the
    same tensors (default: MatRIS's ``RadiusGraph`` with triplets, the batch
    itself without). With a line graph, ``batch.n_edges`` and
    ``batch.status`` are set by each build (None before the first).
    ``ghosts`` (``GhostSpec``): the rows of one rank's P>1 local domain
    instead of a single system (owned slots, then ghost slots).
    """

    def __init__(
        self,
        *,
        cell: Tensor,
        atomic_numbers: Tensor,
        u_capacity: int,
        t_capacity: int,
        pbc: Tensor | None = None,
        atom_graph_cutoff: float = 6.0,
        line_graph_cutoff: float | None = 4.5,
        n_dummy: int = 64,
        device: torch.device | str = "cuda",
        composition: str | None = None,
        candidate_cutoff: float | None = None,
        ghosts: GhostSpec | None = None,
        minimum_image: bool = False,
        candidates_from: FixedCapacityGraphBuilder | None = None,
        padding: PaddingRule | None = None,
        prepare: Callable[[GraphBatch], Any] | None = None,
        positions_dtype: torch.dtype = torch.float32,
        sorted_edges: bool = True,
        max_neighbor_count: Tensor | None = None,
        publish_positions: bool = False,
    ) -> None:
        """``minimum_image`` (single system): the caller may hand in coordinates
        wrapped by any lattice vector since the candidate refresh (ASE
        ``atoms.wrap()``); the skin test uses the minimum-image displacement and
        the wrap bookkeeping removes the lattice jump, so the edge images stay
        exact. ``candidates_from``: share another builder's candidate list (same
        system, cell and candidate cutoff; different capacities). It is
        refreshed through either builder and read by both.
        ``line_graph_cutoff=None``: edges only (no pairs, triplets or sink
        atoms unless ``padding`` asks for sinks); ``t_capacity`` must be 0.
        ``padding``: the form of the padded edge rows (default sink atoms;
        with a line graph also far self-loops, single-system: padded pairs of
        two opposite loops, padded triplets on them). ``prepare``: maps ``batch`` to what
        ``build`` returns (default above). ``positions_dtype`` (edges only, or
        a line graph with ``publish_positions``):
        the dtype of the published positions and cell; ``torch.float64`` for
        a float64 model (pass the float64 cell): each build wraps the caller's
        positions in float64 with the same integer wraps, so the edge vectors
        are exact in float64; the neighbour search stays float32.
        ``sorted_edges=False`` (edges only): the real rows stay grouped by
        centre in the candidates' order, without the per-build sort (a
        reused candidate list then orders a centre's edges differently from a
        fresh one). ``max_neighbor_count`` (edges only): a ``[1]`` int64 device
        tensor, owned by the caller, into which every build folds the largest
        number of edges of one atom (a running maximum, on the device), for a
        model that keeps at most a number of neighbours per atom.
        ``publish_positions`` (with a line graph): also publish the wrapped
        Cartesian coordinates as ``batch.positions`` (sink rows, if any, at the origin),
        for a model the engine differentiates (its position leaf); the
        edges-only mode always publishes them. ``ghosts``: with a line graph,
        sink padding and the caller's per-candidate predicate (MatRIS's
        topology epoch); edges only, ``GhostSpec.edges`` names the edge rule,
        the coordinates are wrapped with the single-system expression, a
        lattice jump since the refresh is removed as with ``minimum_image``,
        and padded self-loops go round robin over the real owned rows (float32
        positions only).
        """
        if not _fixed_builder_available:
            raise RuntimeError(f"fixed-capacity graph builder requires Triton and nvalchemiops: {_fixed_builder_unavailable}")
        padding = _SINK_PADDING if padding is None else padding
        self.edges_only = line_graph_cutoff is None
        self.padding = padding
        self.device = _cuda_device(device)
        self.cell = cell.to(device=self.device, dtype=torch.float32)
        self.cells = self.cell.unsqueeze(0)
        if positions_dtype not in (torch.float32, torch.float64):
            raise ValueError(f"positions_dtype must be torch.float32 or torch.float64, got {positions_dtype}")
        if positions_dtype != torch.float32 and line_graph_cutoff is not None and not publish_positions:
            raise ValueError("float64 positions need the edges-only mode (line_graph_cutoff=None) or, with a line "
                             "graph, published positions")
        if positions_dtype != torch.float32 and ghosts is not None:
            raise ValueError("float64 positions need a single-system builder (no ghost rows)")
        if not sorted_edges and line_graph_cutoff is not None:
            raise ValueError("unsorted edges need the edges-only mode (line_graph_cutoff=None)")
        self.sorted_edges = bool(sorted_edges)
        if max_neighbor_count is not None and line_graph_cutoff is not None:
            raise ValueError("a neighbour count maximum needs the edges-only mode (line_graph_cutoff=None)")
        self.max_neighbor_count = max_neighbor_count
        # The model's cell (float64 for float64 positions, from the caller's cell).
        model_cell = self.cell if positions_dtype == torch.float32 else cell.to(device=self.device,
                                                                                dtype=positions_dtype)
        # Ghost rows with a line graph and no edge rule: the caller's predicate decides every
        # pair (MatRIS's topology epoch); otherwise the builder's edge rule.
        self._caller_predicate = ghosts is not None and line_graph_cutoff is not None and ghosts.edges is None
        if not self._caller_predicate:
            # Ghost rows with the builder's edge rule wrap with the single-system expression: an
            # owned or ghost row holds the bits a single-system build gives its atom.
            self.inv_cell = torch.linalg.inv(self.cell)
        else:
            # Every rank wraps the rows it holds; an FP64 host inverse gives
            # the same bits on every rank (a device FP32 inverse need not).
            self.inv_cell = torch.linalg.inv(self.cell.double().cpu()).float().to(self.device)
        self.atomic_numbers_real = atomic_numbers.to(device=self.device, dtype=torch.int32)
        self.n_real = int(self.atomic_numbers_real.shape[0])
        # Sink rows exist only for sink padding.
        self.n_dummy = int(n_dummy) if padding.kind == "sink_atoms" else 0
        self.u_capacity = int(u_capacity)
        self.e_capacity = 2 * self.u_capacity
        self.t_capacity = int(t_capacity)
        self.atom_graph_cutoff = float(atom_graph_cutoff)
        self.line_graph_cutoff = None if self.edges_only else float(line_graph_cutoff)
        self.candidate_cutoff = float(candidate_cutoff or (self.atom_graph_cutoff + 1.0e-6))
        self.skin = self.candidate_cutoff - self.atom_graph_cutoff
        self.composition = composition or ""
        if self.edges_only:
            if self.u_capacity <= 0 or self.t_capacity != 0:
                raise ValueError("an edges-only builder needs u_capacity > 0 and t_capacity == 0")
            if padding.kind == "sink_atoms" and self.n_dummy <= 0:
                raise ValueError("sink padding needs n_dummy > 0")
            if ghosts is not None and ghosts.edges not in _GHOST_EDGE_RULES:
                raise ValueError(f"an edges-only builder with ghost rows needs GhostSpec.edges in "
                                 f"{tuple(_GHOST_EDGE_RULES)}, got {ghosts.edges!r}")
        elif ghosts is not None and ghosts.edges not in (None, "either"):
            raise ValueError("with a line graph GhostSpec.edges is None (the caller's predicate decides the ghost "
                             f"rows' edges) or 'either', got {ghosts.edges!r}")
        else:
            if self.u_capacity <= 0 or self.t_capacity <= 0 or (padding.kind == "sink_atoms" and self.n_dummy <= 0):
                raise ValueError("capacities and n_dummy must be positive")
            if self.line_graph_cutoff > self.atom_graph_cutoff:
                raise ValueError("line graph cutoff must not exceed atom graph cutoff")
            if padding.kind not in ("sink_atoms", "far_self_loops") or (padding.kind != "sink_atoms"
                                                                        and ghosts is not None):
                raise ValueError("a builder with a line graph pads with sink atoms, or a single-system builder "
                                 "with far self-loops (padded triplets need padded pairs)")

        if pbc is None:
            pbc = torch.ones(3, dtype=torch.bool, device=self.device)
        self.pbc = pbc.to(device=self.device, dtype=torch.bool).reshape(1, 3)
        self.batch_idx = torch.zeros(self.n_real, dtype=torch.int32, device=self.device)
        self.buffer_max = int(
            estimate_max_neighbors(cutoff=self.candidate_cutoff, safety_factor=2.0)
        )
        self._slot_block = triton.next_power_of_2(self.buffer_max)

        # Certified pair-sort key bound. For a fixed cell, the per-step edge
        # image is candidate_shift + wrap[neighbor] - wrap[atom]. The
        # candidate shift is bounded by ceil(candidate_cutoff / h_min) for
        # the smallest perpendicular cell height h_min, and each wrap term is
        # bounded by 1 whenever displacement stays within the skin validity
        # predicate (a violating step is rolled back before its state is
        # visible), so the +2 margin is transactionally protected. When the
        # resulting key range plus the padding-sentinel range fits int32, the
        # pair sort and reverse-key search run on 4-byte keys.
        cell_volume = torch.abs(torch.linalg.det(self.cell.double()))
        face_areas = torch.stack(
            [
                torch.linalg.cross(
                    self.cell.double()[(axis + 1) % 3],
                    self.cell.double()[(axis + 2) % 3],
                ).norm()
                for axis in range(3)
            ]
        )
        min_height = float((cell_volume / face_areas).min())
        certified_shift = math.ceil(self.candidate_cutoff / max(min_height, 1e-6))
        self.certified_max_image = int(certified_shift + 2)
        narrow_radix = 2 * self.certified_max_image + 1
        narrow_volume = narrow_radix**3
        narrow_key_max = self.n_real * self.n_real * narrow_volume + narrow_volume
        use_narrow = (
            _env_flag_fixed("MATRIS_NARROW_SORT_KEYS", True)
            and narrow_key_max + self.e_capacity
            < torch.iinfo(torch.int32).max
        )
        if use_narrow:
            self.sort_key_dtype = torch.int32
            self.sort_key_max_image = self.certified_max_image
        else:
            self.sort_key_dtype = torch.int64
            self.sort_key_max_image = MAX_IMAGE

        self.minimum_image = bool(minimum_image)
        if self.minimum_image and ghosts is not None:
            raise ValueError("minimum_image applies to single-system builders (the ghost path has its own)")
        # The builder whose candidate state (cell list, candidates, reference
        # positions/wraps, wrap delta, flags) this one reads (_candidate_owner);
        # None: its own. No reference to itself, so a dropped builder's buffers
        # are freed by reference counting.
        self._candidate_source = None
        if candidates_from is not None:
            source = candidates_from._candidate_owner
            if (ghosts is not None or source.ghosts is not None or source.device != self.device
                    or source.n_real != self.n_real or source.candidate_cutoff != self.candidate_cutoff
                    or source.minimum_image != self.minimum_image
                    or not torch.equal(source.cell, self.cell) or not torch.equal(source.pbc, self.pbc)):
                raise ValueError("candidates_from must be a single-system builder of the same system, "
                                 "cell, pbc, candidate cutoff and minimum_image setting")
            self._candidate_source = source
            for name in _CANDIDATE_STATE:
                setattr(self, name, getattr(source, name))
        else:
            max_total_cells, neighbor_radius = estimate_batch_cell_list_sizes(
                self.cells,
                self.pbc,
                self.candidate_cutoff,
            )
            self.cell_list_cache = allocate_cell_list(
                self.n_real,
                max_total_cells,
                neighbor_radius,
                self.device,
            )
            self.neighbor_matrix = torch.full(
                (self.n_real, self.buffer_max),
                self.n_real,
                dtype=torch.int32,
                device=self.device,
            )
            self.neighbor_shifts = torch.zeros(
                (self.n_real, self.buffer_max, 3),
                dtype=torch.int32,
                device=self.device,
            )
            self.num_neighbors = torch.zeros(
                self.n_real, dtype=torch.int32, device=self.device
            )
            self.reference_positions = torch.zeros(
                (self.n_real, 3), dtype=torch.float32, device=self.device
            )
            self.reference_wraps = torch.zeros(
                (self.n_real, 3), dtype=torch.int32, device=self.device
            )
            self.wrap_delta = torch.zeros_like(self.reference_wraps)
            self._candidate_capacity_valid = torch.zeros(
                1, dtype=torch.bool, device=self.device
            )
        self._candidate_initialized = False

        self.edge_counts = torch.zeros(self.n_real, dtype=torch.int64, device=self.device)
        if self.edges_only:
            # No short edges, pairs or triplets: the triplet count stays 0.
            self.short_counts = self.short_offsets = self.short_edge_matrix = None
            self.edge_offsets = torch.zeros(self.n_real + 1, dtype=torch.int64, device=self.device)
            self.pair_offsets = torch.zeros(1, dtype=torch.int64, device=self.device)
        else:
            self.short_counts = torch.zeros_like(self.edge_counts)
            self.edge_offsets = torch.zeros(self.n_real + 1, dtype=torch.int64, device=self.device)
            self.short_offsets = torch.zeros_like(self.edge_offsets)
            self.pair_offsets = torch.zeros_like(self.edge_offsets)
            self.short_edge_matrix = torch.zeros(
                (self.n_real, self.buffer_max), dtype=torch.int32, device=self.device
            )

        self.center = torch.empty(self.e_capacity, dtype=torch.int32, device=self.device)
        self.neighbor = torch.empty_like(self.center)
        self.images = torch.empty(
            (self.e_capacity, 3), dtype=torch.int32, device=self.device
        )
        self.distances = torch.empty(
            self.e_capacity, dtype=torch.float32, device=self.device
        )
        if self.edges_only:
            self.short_flags = self.directed2undirected = self.undirected2directed = self.line_graph = None
        else:
            self.short_flags = torch.empty(
                self.e_capacity, dtype=torch.bool, device=self.device
            )
            self.directed2undirected = torch.empty_like(self.center)
            self.undirected2directed = torch.empty(
                self.u_capacity, dtype=torch.int32, device=self.device
            )
            self.line_graph = torch.empty(
                (self.t_capacity, 5), dtype=torch.int32, device=self.device
            )

        atomic_numbers_all = torch.cat(
            [
                self.atomic_numbers_real,
                torch.ones(self.n_dummy, dtype=torch.int32, device=self.device),
            ]
        )
        self.frac_coords = torch.zeros(
            (self.n_real + self.n_dummy, 3), dtype=torch.float32, device=self.device
        )
        # The model's edge buffers (build copies the sorted rows in).
        edge_image = self.images.float()
        edge_index = torch.stack([self.center, self.neighbor], dim=1)
        # Wrapped Cartesian coordinates, for models that read them (MatRIS
        # reads the fractional ones); the override of exact-cutoff padding.
        self.positions = self.edge_vec_override = None
        if self.edges_only or publish_positions:
            self.positions = torch.zeros_like(self.frac_coords, **({} if positions_dtype == torch.float32 else
                                                                  {"dtype": positions_dtype}))
            if padding.kind == "exact_rc_loops":
                self.edge_vec_override = torch.zeros(
                    (self.e_capacity, 3), dtype=torch.float32, device=self.device
                )
        # A float64 edges-only build reusing its candidates runs its prologue in one kernel (up to
        # _FLOAT64_PROLOGUE_ATOMS atoms, one program); its float32 wrapped positions go here.
        self._float64_prologue = (self.edges_only and positions_dtype == torch.float64 and ghosts is None
                                  and not minimum_image and self.n_real <= _FLOAT64_PROLOGUE_ATOMS)
        self._wrapped32 = (torch.zeros((self.n_real, 3), dtype=torch.float32, device=self.device)
                           if self._float64_prologue else None)
        # Row-major copies of the matrices the kernel reads by row (torch.linalg.inv returns a
        # column-major inverse).
        self._prologue_matrices = ((self.inv_cell.contiguous(), self.cell.contiguous(), model_cell.contiguous())
                                   if self._float64_prologue else None)
        # Self-loop padding: (image, length, override vector), from the cell once.
        self._pad_loop = (None if padding.kind == "sink_atoms"
                          else self_loop_padding(padding, self.cell, self.atom_graph_cutoff))
        # Unsorted self-loop-padded edges after that prologue: three kernels write the rows, the
        # padding and the overflow flag into the builder's and the model's buffers (no cumsum, no copies).
        self._unsorted_edges = (self._float64_prologue and not self.sorted_edges and self._pad_loop is not None
                                and self.edge_vec_override is None and self.n_real <= _UNSORTED_EDGES_ATOMS)

        self.edge_ids = torch.arange(self.e_capacity, device=self.device, dtype=torch.int64)
        if not self.edges_only:
            self.u_ids = torch.arange(self.u_capacity, device=self.device, dtype=torch.int64)
            self.triplet_ids = torch.arange(
                self.t_capacity, device=self.device, dtype=torch.int64
            )
            # Scatter targets, then one discard slot per edge: an edge that does
            # not contribute writes its own slot, not a shared sentinel (hundreds
            # of thousands of atomics on one address serialized the scatter).
            self._u2d_scatter = torch.empty(
                self.u_capacity + self.e_capacity, dtype=torch.int64, device=self.device
            )
            self._short_scatter = torch.empty(
                self.n_real * self.buffer_max + self.e_capacity,
                dtype=torch.int64,
                device=self.device,
            )
        self._overflow = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._invalid_pairs = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._candidate_valid = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._max_displacement = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._isolated_atom_count = torch.zeros(
            1, dtype=torch.int32, device=self.device
        )
        self.ghosts = ghosts
        # The kernels' GHOST_EDGES (0: no ghost rows, or the caller's predicate).
        self._ghost_edges = _GHOST_EDGE_RULES[ghosts.edges] if ghosts is not None and ghosts.edges is not None else 0
        # Ghost rows with the builder's edge rule: [rows] bool device masks of the real owned rows and of the real
        # (owned or ghost) rows, rewritten by ``set_atoms`` (with a line graph also over its sink rows, in neither);
        # None otherwise (the single-system batch and MatRIS's topology have no such rows).
        self.owned_mask = self.real_mask = None
        if ghosts is None:
            # Unused kernel arguments of the HAS_GHOSTS=False programs.
            self._n_owned_slots = self.n_real
            self._owned_count = self._isolated_atom_count
            self.edge_mask = self.short_mask = self.num_neighbors
        elif not self._caller_predicate:
            if not 0 < ghosts.n_owned <= self.n_real:
                raise ValueError("GhostSpec.n_owned must be in [1, number of rows]")
            self._n_owned_slots = int(ghosts.n_owned)
            self.atom_ids = torch.full((self.n_real,), -1, dtype=torch.int64, device=self.device)
            self._owned_count = torch.zeros(1, dtype=torch.int32, device=self.device)
            self._active_rows = torch.arange(self.n_real, device=self.device)
            self.owned_mask = torch.zeros(self.n_real + (0 if self.edges_only else self.n_dummy), dtype=torch.bool,
                                          device=self.device)
            self.real_mask = torch.zeros_like(self.owned_mask)
            # Unused kernel arguments (the builder decides every pair itself).
            self.edge_mask = self.short_mask = self.num_neighbors
            if not self.edges_only:
                # The sorted pair keys and each edge's reverse row (``edge_rows``).
                self.sorted_key = torch.empty(self.e_capacity, dtype=self.sort_key_dtype, device=self.device)
                self.pair_row = torch.empty(self.e_capacity, dtype=torch.int64, device=self.device)
        else:
            if not 0 < ghosts.n_owned <= self.n_real:
                raise ValueError("GhostSpec.n_owned must be in [1, number of rows]")
            self._n_owned_slots = int(ghosts.n_owned)
            self.atom_ids = torch.full((self.n_real,), -1, dtype=torch.int64, device=self.device)
            self._owned_count = torch.zeros(1, dtype=torch.int32, device=self.device)
            # Per-step outputs the distributed layer reads (sorted pair keys,
            # the reverse row of each edge, wraps and wrapped coordinates).
            self.sorted_key = torch.empty(self.e_capacity, dtype=self.sort_key_dtype, device=self.device)
            self.pair_row = torch.empty(self.e_capacity, dtype=torch.int64, device=self.device)
            self.step_wraps = torch.zeros((self.n_real, 3), dtype=torch.int32, device=self.device)
            self.wrapped = torch.zeros((self.n_real, 3), dtype=torch.float32, device=self.device)
            self._active_rows = torch.arange(self.n_real, device=self.device)
            # Per candidate slot: edge and short decisions (``build(predicate=...)``).
            self.edge_mask = torch.zeros((self.n_real, self.buffer_max), dtype=torch.int8, device=self.device)
            self.short_mask = torch.zeros_like(self.edge_mask)

        # Systems with cells of their own (fastmd.execution.system_batch sets these): the re-filter kernels' row ->
        # system table and SYSTEM_CELLS, and each system's far self-loop image ([systems, 3] int32); unused here.
        self._row_systems = self.num_neighbors
        self._system_cells = {}
        self._loop_images = None
        if not self.edges_only:
            self._allocate_overflow_topology()
        # The count and the device flags (views of the persistent buffers):
        # an edges-only builder publishes them now; MatRIS's topology takes
        # each build's own views (a view made here would be one more recorded
        # tensor op per builder than the operator snapshots expect).
        n_edges = status = None
        if self.edges_only:
            n_edges = self.edge_offsets[-1:]
            status = FixedCapacityGraphStatus(
                edge_count=n_edges,
                undirected_count=None,  # no tensor of its own: edge_count // 2 when needed
                triplet_count=self.pair_offsets[-1:],
                overflow=self._overflow,
                invalid_pairs=self._invalid_pairs,
                candidate_valid=self._candidate_valid,
                max_displacement=self._max_displacement,
                isolated_atom_count=self._isolated_atom_count,
                candidate_capacity_valid=self._candidate_capacity_valid,
            )
        self.batch = GraphBatch(
            positions=self.positions,
            frac_coords=self.frac_coords,
            cell=model_cell,
            species=atomic_numbers_all,
            n_real=self.n_real,
            edge_index=edge_index,
            edge_image=edge_image,
            n_edges=n_edges,
            cutoff=self.atom_graph_cutoff,
            padding=padding,
            edge_vec_override=self.edge_vec_override,
            triplet_cutoff=self.line_graph_cutoff,
            pair_d2u=self.directed2undirected,
            pair_u2d=self.undirected2directed,
            triplets=self.line_graph,
            status=status,
            composition=self.composition,
            evidence=_FIXED_BUILDER_EVIDENCE,
        )
        if prepare is None:
            prepare = _batch_itself if self.edges_only else radius_graph_view
        # What build returns: the model's own input over these buffers (no copy).
        self.graph = prepare(self.batch)

    def _allocate_overflow_topology(self) -> None:
        # An overflow is observed only after the captured model evaluation. Keep
        # that evaluation memory-safe by substituting a valid all-padding topology
        # on device (all-sink; with far self-loop padding all-loop: pair p on real
        # atom p mod n_real); the transaction runner then rolls back and grows capacity.
        safe_u = self.edge_ids // 2
        if self._pad_loop is None:
            safe_dummy = self.n_real + safe_u * self.n_dummy // self.u_capacity
            self._overflow_center = safe_dummy.int()
        else:
            self._overflow_center = (safe_u % self.n_real).int()
        self._overflow_neighbor = self._overflow_center.clone()
        self._overflow_images = torch.zeros(
            (self.e_capacity, 3), dtype=torch.int32, device=self.device
        )
        image_sign = torch.where(
            self.edge_ids.remainder(2) == 0,
            torch.ones_like(self.edge_ids),
            -torch.ones_like(self.edge_ids),
        ).int()
        if self._pad_loop is None:
            self._overflow_images.copy_(image_sign[:, None].expand(-1, 3) * 3)
        else:
            image = torch.tensor(self._pad_loop[0], dtype=torch.int32, device=self.device)
            self._overflow_images.copy_(image_sign[:, None] * image)
        self._overflow_d2u = safe_u.int()
        self._overflow_u2d = (2 * self.u_ids).int()
        if self._pad_loop is None:
            line_dummy = self.triplet_ids * self.n_dummy // self.t_capacity
            line_u = (
                line_dummy * self.u_capacity + self.n_dummy - 1
            ) // self.n_dummy
        else:
            line_u = self.triplet_ids * self.u_capacity // self.t_capacity
        line_u = line_u.clamp(max=self.u_capacity - 1)
        line_de = 2 * line_u
        line_atom = self._overflow_center[line_de]
        self._overflow_line_graph = torch.stack(
            [line_atom, line_u, line_de, line_u, line_de], dim=1
        ).int()

    @property
    def _candidate_owner(self) -> FixedCapacityGraphBuilder:
        return self._candidate_source or self

    def _build_candidates(self, wrapped_positions: Tensor) -> None:
        if self.ghosts is not None:
            self._build_active_candidates(wrapped_positions)
            return
        self.neighbor_matrix.fill_(self.n_real)
        self.neighbor_shifts.zero_()
        self.num_neighbors.zero_()
        # nvalchemiops launches Warp kernels on Warp's own stream: run them on
        # torch's current stream so they are ordered after the wrapped
        # positions and before the filter (see _build_active_candidates).
        import warp as wp

        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            batch_build_cell_list(
                wrapped_positions,
                self.candidate_cutoff,
                self.cells,
                self.pbc,
                self.batch_idx,
                *self.cell_list_cache,
            )
            batch_query_cell_list(
                wrapped_positions,
                self.cells,
                self.pbc,
                self.candidate_cutoff,
                self.batch_idx,
                *self.cell_list_cache,
                self.neighbor_matrix,
                self.neighbor_shifts,
                self.num_neighbors,
                False,
            )
        self._candidate_capacity_valid.copy_(
            (self.num_neighbors <= self.buffer_max).all().reshape(1)
        )

    def _build_active_candidates(self, wrapped_positions: Tensor) -> None:
        """HAS_GHOSTS: the cell list of the active rows only (``set_atoms``).

        Unused owned and ghost slots hold no atom: they get no candidates and
        are nobody's candidate. The query runs on the compacted active rows
        (prefix views of the cell-list buffers) and is scattered back.
        """
        rows = self._active_rows
        count = int(rows.numel())
        self.neighbor_matrix.fill_(self.n_real)
        self.neighbor_shifts.zero_()
        self.num_neighbors.zero_()
        if count == 0:
            self._candidate_capacity_valid.fill_(True)
            return
        cells_per_dimension, radius, shifts, mapping, per_cell, starts, cell_atoms = self.cell_list_cache
        cache = (cells_per_dimension, radius, shifts[:count], mapping[:count], per_cell, starts,
                 cell_atoms[:count])
        compact = wrapped_positions.index_select(0, rows)
        matrix = torch.full((count, self.buffer_max), count, dtype=torch.int32, device=self.device)
        images = torch.zeros((count, self.buffer_max, 3), dtype=torch.int32, device=self.device)
        counts = torch.zeros(count, dtype=torch.int32, device=self.device)
        batch = self.batch_idx[:count]
        # nvalchemiops launches Warp kernels on Warp's own stream for the
        # device, not on torch's current stream: run them on the torch stream
        # so the compaction before and the scatter after are ordered with them.
        import warp as wp

        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            batch_build_cell_list(compact, self.candidate_cutoff, self.cells, self.pbc, batch, *cache)
            batch_query_cell_list(compact, self.cells, self.pbc, self.candidate_cutoff, batch, *cache,
                                  matrix, images, counts, False)
        filled = torch.arange(self.buffer_max, device=self.device)[None, :] < counts[:, None]
        mapped = torch.where(filled, rows[matrix.long().clamp(max=count - 1)],
                             torch.full_like(rows[:1], self.n_real)).int()
        self.neighbor_matrix.index_copy_(0, rows, mapped)
        self.neighbor_shifts.index_copy_(0, rows, images)
        self.num_neighbors.index_copy_(0, rows, counts)
        if self._ghost_edges == _GHOST_EDGE_RULES["receiver"]:
            # A ghost row is never a receiver: no candidates of its own (it stays the owned rows' candidate).
            self.num_neighbors[self._n_owned_slots:] = 0
        self._candidate_capacity_valid.copy_((counts <= self.buffer_max).all().reshape(1))

    def set_atomic_numbers(self, atomic_numbers: Tensor) -> None:
        """Single system: new atomic numbers of the same atoms, in place.

        The candidates depend on the geometry only, so they stay valid, and a
        captured graph reads the new numbers.
        """
        if self.ghosts is not None:
            raise RuntimeError("a HAS_GHOSTS builder takes its numbers with set_atoms")
        if atomic_numbers.shape != (self.n_real,):
            raise ValueError(f"atomic_numbers must have shape {(self.n_real,)}")
        self.atomic_numbers_real.copy_(atomic_numbers)
        self.batch.species[: self.n_real].copy_(atomic_numbers)

    def set_atoms(self, atomic_numbers: Tensor, atom_ids: Tensor, active_rows: Tensor,
                  owned_count: int) -> None:
        """HAS_GHOSTS epoch contents: per-row Z and global ID, the active rows, real owned count.

        Rows ``[0, owned_count)`` must be the real owned atoms; ``active_rows``
        (sorted row indices) lists them and the real ghosts. Unused rows keep
        any Z; they never enter an edge. Edges only, the rows' masks follow
        (``owned_mask``: rows ``[0, owned_count)``, ``real_mask``: the active rows).
        """
        if self.ghosts is None:
            raise RuntimeError("set_atoms requires a HAS_GHOSTS builder")
        if atomic_numbers.shape != (self.n_real,) or atom_ids.shape != (self.n_real,):
            raise ValueError(f"atomic_numbers and atom_ids must have shape {(self.n_real,)}")
        if not 0 <= owned_count <= self._n_owned_slots:
            raise ValueError("owned_count exceeds the owned slots")
        self.atomic_numbers_real.copy_(atomic_numbers)
        self.batch.species[: self.n_real].copy_(atomic_numbers)
        self.atom_ids.copy_(atom_ids)
        self._owned_count.fill_(owned_count)
        self._active_rows = active_rows.to(device=self.device, dtype=torch.int64)
        if self.owned_mask is not None:
            # In place: a captured graph may read the masks (sink rows, if any, in neither).
            self.owned_mask.copy_(torch.arange(self.owned_mask.shape[0], device=self.device) < owned_count)
            self.real_mask.zero_()
            self.real_mask.index_fill_(0, self._active_rows, True)
        self._candidate_initialized = False

    def edge_rows(self, center: Tensor, neighbor: Tensor, reference_images: Tensor) -> Tensor:
        """HAS_GHOSTS: the last build's directed edge row of each (center, neighbor, image), else -1.

        ``reference_images`` are relative to the wrapped coordinates of the
        last candidate refresh; the lookup applies this step's wrap changes
        (the edge image rule of the materialization kernels) and searches the
        sorted pair keys of ``_materialize_pairs``. Device only, no sync.
        """
        if self.ghosts is None or self.edges_only:
            raise RuntimeError("edge_rows requires a HAS_GHOSTS builder with a line graph")
        center, neighbor = center.long(), neighbor.long()
        delta = self.wrap_delta.long()
        images = reference_images.long() + delta[neighbor] - delta[center]
        max_image = self.sort_key_max_image
        radix = 2 * max_image + 1
        volume = radix**3
        key = ((center * self.n_real + neighbor) * volume
               + (images[:, 0] + max_image) * radix * radix
               + (images[:, 1] + max_image) * radix + images[:, 2] + max_image)
        in_range = (images.abs() <= max_image).all(dim=1)
        key = torch.where(in_range, key, torch.zeros_like(key)).to(self.sort_key_dtype)
        position = torch.searchsorted(self.sorted_key, key).clamp(max=self.e_capacity - 1)
        edges = self.edge_offsets[-1].clamp(max=self.e_capacity)
        found = in_range & (self.sorted_key[position] == key) & (position < edges)
        return torch.where(found, position, torch.full_like(position, -1))

    def assert_candidate_capacity(self) -> None:
        """Raise on the slow path if nvalchemi truncated a neighbor row."""
        if bool(self._candidate_capacity_valid.item()):
            return
        raise NeighborOverflowError(
            self.buffer_max,
            int(self.num_neighbors.max().item()),
        )

    @torch.no_grad()
    def refresh_candidates(self, positions: Tensor) -> None:
        """Refresh the slow-path cell list used by subsequent exact re-filter steps."""
        positions = positions.to(device=self.device, dtype=torch.float32)
        frac_unwrapped = small_matmul(positions, self.inv_cell)
        wraps = torch.floor(frac_unwrapped).to(torch.int32)
        wrapped = small_matmul(frac_unwrapped - wraps.float(), self.cell)
        self.reference_positions.copy_(positions)
        self.reference_wraps.copy_(wraps)
        self.wrap_delta.zero_()
        self._build_candidates(wrapped)
        self._candidate_owner._candidate_initialized = True

    def _materialize_edges(self, wrapped_positions: Tensor) -> None:
        if self.edges_only:
            self._materialize_edges_only(wrapped_positions)
            return
        self._isolated_atom_count.zero_()
        # Ghost rows with the builder's edge rule (else the caller's predicate or a single system).
        ghost_edges = {"GHOST_EDGES": self._ghost_edges} if self._ghost_edges else {}
        _count_neighbors_kernel[(self.n_real,)](
            wrapped_positions,
            self.cell,
            self.neighbor_matrix,
            self.neighbor_shifts,
            self.num_neighbors,
            self.wrap_delta,
            self.edge_counts,
            self.short_counts,
            self._isolated_atom_count,
            self._owned_count,
            self._n_owned_slots,
            self.edge_mask,
            self.short_mask,
            self._row_systems,
            N=self.n_real,
            BMAX=self.buffer_max,
            BLOCK=self._slot_block,
            EDGE_CUTOFF2=self.atom_graph_cutoff * self.atom_graph_cutoff,
            LINE_CUTOFF2=self.line_graph_cutoff * self.line_graph_cutoff,
            HAS_GHOSTS=self._caller_predicate,
            **ghost_edges,
            **self._system_cells,
            num_warps=8,
        )
        self.edge_offsets[0].zero_()
        torch.cumsum(self.edge_counts, dim=0, out=self.edge_offsets[1:])
        self.short_edge_matrix.zero_()
        _materialize_neighbors_kernel[(self.n_real,)](
            wrapped_positions,
            self.cell,
            self.neighbor_matrix,
            self.neighbor_shifts,
            self.num_neighbors,
            self.wrap_delta,
            self.edge_offsets,
            self.center,
            self.neighbor,
            self.images,
            self.distances,
            self.short_flags,
            self.short_edge_matrix,
            self._n_owned_slots,
            self.edge_mask,
            self.short_mask,
            self._row_systems,
            N=self.n_real,
            BMAX=self.buffer_max,
            E_CAP=self.e_capacity,
            BLOCK=self._slot_block,
            EDGE_CUTOFF2=self.atom_graph_cutoff * self.atom_graph_cutoff,
            LINE_CUTOFF2=self.line_graph_cutoff * self.line_graph_cutoff,
            HAS_GHOSTS=self._caller_predicate,
            **ghost_edges,
            **self._system_cells,
            num_warps=8,
        )
        if self._pad_loop is not None:
            image, length, _ = self._pad_loop
            _pad_loop_pairs_kernel[(triton.cdiv(self.e_capacity, 256),)](
                self.edge_offsets[-1:],
                self.center,
                self.neighbor,
                self.images,
                self.distances,
                self.short_flags,
                *image,
                length,
                N=self.n_real,
                E_CAP=self.e_capacity,
                BLOCK=256,
                num_warps=4,
            )
            return
        _pad_edges_kernel[(triton.cdiv(self.e_capacity, 256),)](
            self.edge_offsets[-1:],
            self.center,
            self.neighbor,
            self.images,
            self.distances,
            self.short_flags,
            N=self.n_real,
            N_DUMMY=self.n_dummy,
            E_CAP=self.e_capacity,
            BLOCK=256,
            num_warps=4,
        )

    def _materialize_edges_only(self, wrapped_positions: Tensor) -> None:
        """Edges only: the exact edges of the candidates (no short edges), then the padding rows."""
        self._isolated_atom_count.zero_()
        # The short-edge outputs are not written (HAS_SHORT=False): any tensor fills their slots.
        unused = self.edge_counts
        # Ghost rows: the edge rule, and padded self-loops on the real owned rows only.
        ghost_edges = {"GHOST_EDGES": self._ghost_edges} if self._ghost_edges else {}
        _count_neighbors_kernel[(self.n_real,)](
            wrapped_positions,
            self.cell,
            self.neighbor_matrix,
            self.neighbor_shifts,
            self.num_neighbors,
            self.wrap_delta,
            self.edge_counts,
            unused,
            self._isolated_atom_count,
            self._owned_count,
            self._n_owned_slots,
            self.edge_mask,
            self.short_mask,
            self._row_systems,
            N=self.n_real,
            BMAX=self.buffer_max,
            BLOCK=self._slot_block,
            EDGE_CUTOFF2=self.atom_graph_cutoff * self.atom_graph_cutoff,
            LINE_CUTOFF2=0.0,
            HAS_SHORT=False,
            **ghost_edges,
            **self._system_cells,
            num_warps=8,
        )
        self.edge_offsets[0].zero_()
        torch.cumsum(self.edge_counts, dim=0, out=self.edge_offsets[1:])
        if self.max_neighbor_count is not None:
            torch.maximum(self.max_neighbor_count, self.edge_counts.max().reshape(1), out=self.max_neighbor_count)
        _materialize_neighbors_kernel[(self.n_real,)](
            wrapped_positions,
            self.cell,
            self.neighbor_matrix,
            self.neighbor_shifts,
            self.num_neighbors,
            self.wrap_delta,
            self.edge_offsets,
            self.center,
            self.neighbor,
            self.images,
            self.distances,
            unused,
            unused,
            self._n_owned_slots,
            self.edge_mask,
            self.short_mask,
            self._row_systems,
            N=self.n_real,
            BMAX=self.buffer_max,
            E_CAP=self.e_capacity,
            BLOCK=self._slot_block,
            EDGE_CUTOFF2=self.atom_graph_cutoff * self.atom_graph_cutoff,
            LINE_CUTOFF2=0.0,
            HAS_SHORT=False,
            **ghost_edges,
            **self._system_cells,
            num_warps=8,
        )
        if self._pad_loop is None:
            _pad_edges_kernel[(triton.cdiv(self.e_capacity, 256),)](
                self.edge_offsets[-1:],
                self.center,
                self.neighbor,
                self.images,
                self.distances,
                unused,
                N=self.n_real,
                N_DUMMY=self.n_dummy,
                E_CAP=self.e_capacity,
                BLOCK=256,
                HAS_SHORT=False,
                num_warps=4,
            )
            return
        if self._loop_images is not None:
            _pad_system_self_loops_kernel[(triton.cdiv(self.e_capacity, 256),)](
                self.edge_offsets[-1:],
                self.center,
                self.neighbor,
                self.images,
                self._row_systems,
                self._loop_images,
                N=self.n_real,
                E_CAP=self.e_capacity,
                BLOCK=256,
                num_warps=4,
            )
            return
        image, _, vector = self._pad_loop
        vector = (0.0, 0.0, 0.0) if vector is None else vector
        _pad_self_loops_kernel[(triton.cdiv(self.e_capacity, 256),)](
            self.edge_offsets[-1:],
            self.center,
            self.neighbor,
            self.images,
            self.images if self.edge_vec_override is None else self.edge_vec_override,
            *image,
            *vector,
            self._owned_count,
            N=self.n_real,
            E_CAP=self.e_capacity,
            BLOCK=256,
            HAS_OVERRIDE=self.edge_vec_override is not None,
            **({"OWNED_LOOPS": True} if self._ghost_edges else {}),
            num_warps=4,
        )

    def _materialize_unsorted_edges(self) -> None:
        """A float64 build's ``_materialize_edges_only`` + ``_sort_edges`` (unsorted) + the copies into
        the model's edge buffers, in three kernels: the counts, the rows (each program sums the counts
        before its atom), the self-loop padding with the overflow flag. Same rows and flags (and the same
        neighbour count maximum); the prologue has zeroed the isolated-atom count."""
        unused = self.edge_counts
        _count_neighbors_kernel[(self.n_real,)](
            self._wrapped32,
            self.cell,
            self.neighbor_matrix,
            self.neighbor_shifts,
            self.num_neighbors,
            self.wrap_delta,
            self.edge_counts,
            unused,
            self._isolated_atom_count,
            self._owned_count,
            self._n_owned_slots,
            self.edge_mask,
            self.short_mask,
            self._row_systems,
            N=self.n_real,
            BMAX=self.buffer_max,
            BLOCK=self._slot_block,
            EDGE_CUTOFF2=self.atom_graph_cutoff * self.atom_graph_cutoff,
            LINE_CUTOFF2=0.0,
            HAS_SHORT=False,
            **self._system_cells,
            num_warps=8,
        )
        if self.max_neighbor_count is not None:
            torch.maximum(self.max_neighbor_count, self.edge_counts.max().reshape(1), out=self.max_neighbor_count)
        _materialize_unsorted_edges_kernel[(self.n_real,)](
            self._wrapped32,
            self.cell,
            self.neighbor_matrix,
            self.neighbor_shifts,
            self.num_neighbors,
            self.wrap_delta,
            self.edge_counts,
            self.edge_offsets,
            self.center,
            self.neighbor,
            self.images,
            self.distances,
            self.batch.edge_index,
            self.batch.edge_image,
            N=self.n_real,
            BMAX=self.buffer_max,
            E_CAP=self.e_capacity,
            BLOCK=self._slot_block,
            BLOCK_N=triton.next_power_of_2(self.n_real),
            EDGE_CUTOFF2=self.atom_graph_cutoff * self.atom_graph_cutoff,
            num_warps=8,
        )
        image = self._pad_loop[0]
        _finish_unsorted_edges_kernel[(triton.cdiv(self.e_capacity, 256),)](
            self.edge_offsets,
            self.center,
            self.neighbor,
            self.images,
            self.batch.edge_index,
            self.batch.edge_image,
            self._candidate_valid,
            self._overflow,
            *image,
            N=self.n_real,
            E_CAP=self.e_capacity,
            BLOCK=256,
            num_warps=4,
        )

    def _sort_edges(self) -> None:
        """Edges only: the real rows in the canonical (centre, neighbour,
        image) order of ``_materialize_pairs``, padding rows in place. The
        order of the candidates does not show: a reused candidate list and a
        fresh one give the same rows. (``distances``, which nothing reads
        without a line graph, stays in candidate order.)"""
        if self.sorted_edges:
            total_clamped = self.edge_offsets[-1].clamp(max=self.e_capacity)
            is_real = self.edge_ids < total_clamped
            key_dtype = self.sort_key_dtype
            max_image = self.sort_key_max_image
            center = self.center.to(key_dtype)
            neighbor = self.neighbor.to(key_dtype)
            images = self.images.to(key_dtype)
            radix = 2 * max_image + 1
            volume = radix**3
            forward_key = (
                (center * self.n_real + neighbor) * volume
                + (images[:, 0] + max_image) * radix * radix
                + (images[:, 1] + max_image) * radix
                + images[:, 2]
                + max_image
            )
            invalid_key = (
                torch.iinfo(key_dtype).max - self.e_capacity + self.edge_ids
            ).to(key_dtype)
            _, sort_idx = torch.sort(torch.where(is_real, forward_key, invalid_key))
            self.center.copy_(self.center[sort_idx])
            self.neighbor.copy_(self.neighbor[sort_idx])
            self.images.copy_(self.images[sort_idx])
        overflow = (self.edge_offsets[-1] > self.e_capacity) | ~self._candidate_valid[0]
        self._overflow.copy_(overflow.reshape(1))

    def _materialize_pairs(self) -> None:
        total_edges = self.edge_offsets[-1]
        total_clamped = total_edges.clamp(max=self.e_capacity)
        is_real = self.edge_ids < total_clamped
        key_dtype = self.sort_key_dtype
        max_image = self.sort_key_max_image
        center = self.center.to(key_dtype)
        neighbor = self.neighbor.to(key_dtype)
        images = self.images.to(key_dtype)
        radix = 2 * max_image + 1
        volume = radix**3
        forward_key = (
            (center * self.n_real + neighbor) * volume
            + (images[:, 0] + max_image) * radix * radix
            + (images[:, 1] + max_image) * radix
            + images[:, 2]
            + max_image
        )
        reverse_key = (
            (neighbor * self.n_real + center) * volume
            + (-images[:, 0] + max_image) * radix * radix
            + (-images[:, 1] + max_image) * radix
            - images[:, 2]
            + max_image
        )
        invalid_key = (
            torch.iinfo(key_dtype).max - self.e_capacity + self.edge_ids
        ).to(key_dtype)
        forward_key = torch.where(is_real, forward_key, invalid_key)
        sorted_key, sort_idx = torch.sort(forward_key)
        self.center.copy_(self.center[sort_idx])
        self.neighbor.copy_(self.neighbor[sort_idx])
        self.images.copy_(self.images[sort_idx])
        self.distances.copy_(self.distances[sort_idx])
        self.short_flags.copy_(self.short_flags[sort_idx])

        center = self.center.to(key_dtype)
        neighbor = self.neighbor.to(key_dtype)
        images = self.images.to(key_dtype)
        reverse_key = (
            (neighbor * self.n_real + center) * volume
            + (-images[:, 0] + max_image) * radix * radix
            + (-images[:, 1] + max_image) * radix
            - images[:, 2]
            + max_image
        )
        reverse_key = torch.where(is_real, reverse_key, invalid_key)
        reverse_pos = torch.searchsorted(sorted_key, reverse_key).clamp(
            max=self.e_capacity - 1
        )
        pair_idx = reverse_pos
        reverse_matches = sorted_key[reverse_pos] == reverse_key
        if self.ghosts is not None:
            self.sorted_key.copy_(sorted_key)
            self.pair_row.copy_(torch.where(is_real & reverse_matches, pair_idx, self.edge_ids))
        if self._ghost_edges:
            # Ghost rows with the builder's edge rule: a pair's first edge is the one a single-system build puts
            # first (its rows in global ID order: the lower ID's centre), so the model reads each pair from the
            # same direction as there.
            ids, last = self.atom_ids, self.n_real - 1  # (padded rows: sink rows, not real, clamped)
            centre_id = ids[self.center.long().clamp(max=last)]
            neighbour_id = ids[self.neighbor.long().clamp(max=last)]
            first = (centre_id < neighbour_id) | ((centre_id == neighbour_id) & (self.edge_ids <= pair_idx))
            canonical_row = torch.where(first, self.edge_ids, pair_idx)
            pair_leader = is_real & reverse_matches & first
        else:
            canonical_row = torch.minimum(self.edge_ids, pair_idx)
            pair_leader = is_real & reverse_matches & (self.edge_ids <= pair_idx)
        pair_prefix = torch.cumsum(pair_leader.long(), dim=0)
        real_pair_id = pair_prefix[canonical_row] - 1
        real_u = total_clamped // 2
        pad_pair_id = real_u + (self.edge_ids - total_clamped).clamp_min(0) // 2
        d2u = torch.where(is_real, real_pair_id, pad_pair_id).clamp(
            min=0, max=self.u_capacity - 1
        )
        self.directed2undirected.copy_(d2u.int())

        scatter_index = torch.where(
            pair_leader, real_pair_id.clamp(min=0, max=self.u_capacity), self.u_capacity + self.edge_ids
        )
        self._u2d_scatter.fill_(self.e_capacity)
        self._u2d_scatter.scatter_reduce_(
            0,
            scatter_index,
            self.edge_ids,
            reduce="amin",
            include_self=True,
        )
        pad_u2d = total_clamped + 2 * (self.u_ids - real_u).clamp_min(0)
        u2d = torch.where(self.u_ids < real_u, self._u2d_scatter[: self.u_capacity], pad_u2d)
        self.undirected2directed.copy_(u2d.clamp(max=self.e_capacity - 1).int())

        pair_count = pair_leader.long().sum()
        invalid = (
            (total_edges.remainder(2) != 0)
            | (pair_count != total_edges // 2)
            | (is_real & ~reverse_matches).any()
        )
        self._invalid_pairs.copy_(invalid.reshape(1))

    def _materialize_line_graph(self) -> None:
        total_edges = self.edge_offsets[-1].clamp(max=self.e_capacity)
        # Preserve the exact dist2 cutoff decision made by the materialization
        # kernel. Recomputing it from sqrt(dist2) can disagree at the boundary.
        is_short = (self.edge_ids < total_edges) & self.short_flags
        self.short_offsets[0].zero_()
        torch.cumsum(self.short_counts, dim=0, out=self.short_offsets[1:])
        short_global_rank = torch.cumsum(is_short.long(), dim=0) - 1
        safe_center = self.center.long().clamp(min=0, max=self.n_real - 1)
        short_local_rank = short_global_rank - self.short_offsets[safe_center]
        short_target = safe_center * self.buffer_max + short_local_rank
        slots = self.n_real * self.buffer_max
        short_target = torch.where(
            is_short, short_target.clamp(min=0, max=slots), slots + self.edge_ids
        )
        self._short_scatter.fill_(self.e_capacity)
        self._short_scatter.scatter_reduce_(
            0,
            short_target,
            self.edge_ids,
            reduce="amin",
            include_self=True,
        )
        self.short_edge_matrix.copy_(
            self._short_scatter[:slots].reshape(self.n_real, self.buffer_max).int()
        )

        pair_counts = self.short_counts * (self.short_counts - 1)
        self.pair_offsets[0].zero_()
        torch.cumsum(pair_counts, dim=0, out=self.pair_offsets[1:])
        # ``>=``: at edges == e_capacity no padding pair is left and the
        # padding triplets would index u == u_capacity (one past the end),
        # so the safe all-padding topology must be substituted as well.
        overflow = (
            (self.edge_offsets[-1] >= self.e_capacity)
            | (self.pair_offsets[-1] > self.t_capacity)
            | self._invalid_pairs[0]
            | ~self._candidate_valid[0]
        )
        self._overflow.copy_(overflow.reshape(1))
        block = 512
        _line_graph_kernel[(triton.cdiv(self.t_capacity, block),)](
            self.pair_offsets,
            self.short_counts,
            self.short_edge_matrix,
            self.directed2undirected,
            self.edge_offsets[-1:],
            self._overflow,
            self._overflow_line_graph,
            self.line_graph,
            self.e_capacity,
            self.u_capacity,
            self.t_capacity,
            N=self.n_real,
            BMAX=self.buffer_max,
            N_DUMMY=self.n_dummy,
            SEARCH_STEPS=self.n_real.bit_length() + 1,
            BLOCK=block,
            **({} if self._pad_loop is None else {"LOOP_PADDING": True}),
            num_warps=4,
        )

    def _substitute_overflow_topology(self) -> None:
        overflow = self._overflow[0]
        self.center.copy_(torch.where(overflow, self._overflow_center, self.center))
        self.neighbor.copy_(
            torch.where(overflow, self._overflow_neighbor, self.neighbor)
        )
        self.images.copy_(
            torch.where(overflow, self._overflow_images, self.images)
        )
        self.directed2undirected.copy_(
            torch.where(overflow, self._overflow_d2u, self.directed2undirected)
        )
        self.undirected2directed.copy_(
            torch.where(overflow, self._overflow_u2d, self.undirected2directed)
        )
        # The line graph's substitute is written by _line_graph_kernel.

    @torch.no_grad()
    def build(
        self,
        positions: Tensor,
        *,
        rebuild_candidates: bool = True,
        predicate=None,
    ) -> tuple[Any, FixedCapacityGraphStatus]:
        """Build into persistent buffers and return ``graph`` (``prepare(batch)``) plus device-side status.

        HAS_GHOSTS with a line graph: ``predicate(builder)`` (required) returns this step's
        (edge, short) decision per candidate slot, [rows, buffer_max] bools; it
        runs after the step's wraps (``step_wraps``, ``wrap_delta``,
        ``wrapped``) are set. Edges-only ghost rows: ``positions`` holds every
        row, each ghost row with the coordinates its owner hands its own builder
        (the same bits give the owner's wrapped coordinates, wraps and edge
        images, which are a single-system build's; a ``minimum_image`` one's
        when the caller wrapped atoms since the refresh).
        """
        if self._float64_prologue and not rebuild_candidates:
            return self._build_float64(positions)
        with nvtx_range("fixed_capacity_graph_build"):
            given = positions
            positions = positions.to(device=self.device, dtype=torch.float32)
            if positions.shape != (self.n_real, 3):
                raise ValueError(
                    f"positions must have shape {(self.n_real, 3)}, got {tuple(positions.shape)}"
                )
            frac_unwrapped = small_matmul(positions, self.inv_cell)
            wraps = torch.floor(frac_unwrapped).to(torch.int32)
            frac = frac_unwrapped - wraps.float()
            wrapped = small_matmul(frac, self.cell)
            self.frac_coords[: self.n_real].copy_(frac)
            if self._caller_predicate:
                self.step_wraps.copy_(wraps)
                self.wrapped.copy_(wrapped)
            if rebuild_candidates:
                self.reference_positions.copy_(positions)
                self.reference_wraps.copy_(wraps)
                self.wrap_delta.zero_()
                self._build_candidates(wrapped)
                self._candidate_owner._candidate_initialized = True
            elif not self._candidate_owner._candidate_initialized:
                raise RuntimeError("refresh_candidates must run before candidate reuse")
            else:
                self.wrap_delta.copy_(wraps - self.reference_wraps)
            displacement = positions - self.reference_positions
            if self._caller_predicate:
                # Minimum image: a P>1 caller may hand in cell-wrapped
                # coordinates (a lattice jump is no motion; wrap_delta keeps
                # the images exact).
                displacement = displacement - small_matmul(
                    torch.round(small_matmul(displacement, self.inv_cell)), self.cell)
            elif self.minimum_image or self.ghosts is not None:
                # A lattice jump L since the refresh (the caller wrapped the
                # atom) is no motion: the skin test takes the minimum-image
                # displacement d, and with the jump removed from the wrap
                # delta the wrapped row is the refresh's row + d, so every
                # candidate image stays exact (any integer L would; the
                # rounded one gives the smallest |d|). Edges-only ghost rows
                # always: a ghost row jumps with its owner's coordinates, and
                # every rank holding the atom removes the same jump.
                jumps = torch.round(small_matmul(displacement, self.inv_cell))
                displacement = displacement - small_matmul(jumps, self.cell)
                if not rebuild_candidates:
                    self.wrap_delta.sub_(jumps.to(torch.int32))
            max_displacement = torch.linalg.vector_norm(displacement, dim=1).max()
            candidate_valid = (
                (max_displacement * 2.0 <= self.skin + 1.0e-6)
                & self._candidate_capacity_valid[0]
            )
            self._max_displacement.copy_(max_displacement.reshape(1))
            self._candidate_valid.copy_(candidate_valid.reshape(1))
            if self._caller_predicate:
                if predicate is None:
                    raise ValueError("a HAS_GHOSTS build needs the caller's edge predicate")
                edge, short = predicate(self)
                self.edge_mask.copy_(edge)
                self.short_mask.copy_(short)
            self._materialize_edges(wrapped)
            if self.edges_only:
                self._sort_edges()  # also sets the overflow flag
                if self.positions.dtype == torch.float32:
                    self.positions[: self.n_real].copy_(wrapped)
                else:
                    # The caller's positions minus the same whole lattice vectors (an FP64 GEMM: no TF32).
                    torch.addmm(given.to(device=self.device, dtype=self.positions.dtype),
                                wraps.to(self.positions.dtype), self.batch.cell, alpha=-1.0,
                                out=self.positions[: self.n_real])
            else:
                self._materialize_pairs()
                self._materialize_line_graph()  # also sets the overflow flag
                self._substitute_overflow_topology()
                if self.positions is not None and self.positions.dtype == torch.float32:
                    self.positions[: self.n_real].copy_(wrapped)
                elif self.positions is not None:
                    # Float64 published positions: as the edges-only build's.
                    torch.addmm(given.to(device=self.device, dtype=self.positions.dtype),
                                wraps.to(self.positions.dtype), self.batch.cell, alpha=-1.0,
                                out=self.positions[: self.n_real])
            self.batch.edge_index[:, 0].copy_(self.center)
            self.batch.edge_index[:, 1].copy_(self.neighbor)
            self.batch.edge_image.copy_(self.images)
            status = FixedCapacityGraphStatus(
                edge_count=self.edge_offsets[-1:],
                undirected_count=self.edge_offsets[-1:] // 2,
                triplet_count=self.pair_offsets[-1:],
                overflow=self._overflow,
                invalid_pairs=self._invalid_pairs,
                candidate_valid=self._candidate_valid,
                max_displacement=self._max_displacement,
                isolated_atom_count=self._isolated_atom_count,
                candidate_capacity_valid=self._candidate_capacity_valid,
            )
            if not self.edges_only:
                # MatRIS's topology: this build's views (no tensor op, nothing kept alive but views).
                self.batch.n_edges = status.edge_count
                self.batch.status = dataclasses.replace(status, undirected_count=None)
            return self.graph, status

    def accumulate_window(self, status: FixedCapacityGraphStatus, *, window_overflow: Tensor,
                          window_capacity_overflow: Tensor, window_max_edges: Tensor,
                          window_max_isolated: Tensor, check_isolated: bool) -> None:
        """Edges only: fold one build's ``status`` into a whole-step window's flags in one kernel (the
        overflow, the capacity overflow of ``fixed_capacity_exceeded`` and the edge and isolated-atom
        maxima; there are no triplets)."""
        if not self.edges_only:
            raise ValueError("accumulate_window is for edges-only builders")
        _window_flags_kernel[(1,)](
            status.edge_count, status.invalid_pairs, status.overflow, status.isolated_atom_count,
            window_overflow, window_capacity_overflow, window_max_edges, window_max_isolated,
            E_CAP=self.e_capacity, CHECK_ISOLATED=bool(check_isolated), num_warps=1,
        )

    @torch.no_grad()
    def _build_float64(self, positions: Tensor) -> tuple[Any, FixedCapacityGraphStatus]:
        """``build(positions, rebuild_candidates=False)`` of a float64 edges-only builder: the same
        edges and buffers, with the per-build prologue (wrap, wrap delta, float64 positions, skin test)
        in one kernel instead of a chain of small tensor ops."""
        with nvtx_range("fixed_capacity_graph_build"):
            if positions.shape != (self.n_real, 3):
                raise ValueError(
                    f"positions must have shape {(self.n_real, 3)}, got {tuple(positions.shape)}"
                )
            if not self._candidate_owner._candidate_initialized:
                raise RuntimeError("refresh_candidates must run before candidate reuse")
            inv_cell, cell, cell64 = self._prologue_matrices
            _float64_prologue_kernel[(1,)](
                positions.to(device=self.device, dtype=torch.float64).contiguous(),
                inv_cell,
                cell,
                cell64,
                self.reference_positions,
                self.reference_wraps,
                self._candidate_capacity_valid,
                self.frac_coords,
                self._wrapped32,
                self.wrap_delta,
                self.positions,
                self._max_displacement,
                self._candidate_valid,
                self._isolated_atom_count,
                self.skin + 1.0e-6,
                N=self.n_real,
                BLOCK=max(triton.next_power_of_2(self.n_real), 16),
                num_warps=4,
            )
            if self._unsorted_edges:
                self._materialize_unsorted_edges()
            else:
                self._materialize_edges(self._wrapped32)
                self._sort_edges()  # also sets the overflow flag
                self.batch.edge_index[:, 0].copy_(self.center)
                self.batch.edge_index[:, 1].copy_(self.neighbor)
                self.batch.edge_image.copy_(self.images)
            status = FixedCapacityGraphStatus(
                edge_count=self.edge_offsets[-1:],
                undirected_count=self.edge_offsets[-1:] // 2,
                triplet_count=self.pair_offsets[-1:],
                overflow=self._overflow,
                invalid_pairs=self._invalid_pairs,
                candidate_valid=self._candidate_valid,
                max_displacement=self._max_displacement,
                isolated_atom_count=self._isolated_atom_count,
                candidate_capacity_valid=self._candidate_capacity_valid,
            )
            return self.graph, status


class BatchedCandidateRefresher:
    """Refresh homogeneous fixed-capacity builders with one cell-list call."""

    def __init__(
        self,
        builders: Sequence[FixedCapacityGraphBuilder],
    ) -> None:
        if not builders:
            raise ValueError("builders must not be empty")
        self.builders = list(builders)
        first = self.builders[0]
        self.device = first.device
        self.batch_size = len(self.builders)
        self.n_real = first.n_real
        self.buffer_max = first.buffer_max
        self.candidate_cutoff = first.candidate_cutoff
        for builder in self.builders:
            if builder.device != self.device:
                raise ValueError("all builders must use the same device")
            if builder.n_real != self.n_real:
                raise ValueError("all builders must have equal atom counts")
            if builder.buffer_max != self.buffer_max:
                raise ValueError("all builders must have equal neighbor capacity")
            if builder.candidate_cutoff != self.candidate_cutoff:
                raise ValueError("all builders must have equal candidate cutoffs")

        self.total_atoms = self.batch_size * self.n_real
        self.cells = torch.stack([builder.cell for builder in self.builders])
        self.inv_cells = torch.stack(
            [builder.inv_cell for builder in self.builders]
        )
        self.pbc = torch.cat([builder.pbc for builder in self.builders], dim=0)
        self.batch_idx = torch.arange(
            self.batch_size, dtype=torch.int32, device=self.device
        ).repeat_interleave(self.n_real)
        max_total_cells, neighbor_radius = estimate_batch_cell_list_sizes(
            self.cells,
            self.pbc,
            self.candidate_cutoff,
        )
        self.cell_list_cache = allocate_cell_list(
            self.total_atoms,
            max_total_cells,
            neighbor_radius,
            self.device,
        )
        self.neighbor_matrix = torch.full(
            (self.total_atoms, self.buffer_max),
            self.total_atoms,
            dtype=torch.int32,
            device=self.device,
        )
        self.neighbor_shifts = torch.zeros(
            (self.total_atoms, self.buffer_max, 3),
            dtype=torch.int32,
            device=self.device,
        )
        self.num_neighbors = torch.zeros(
            self.total_atoms, dtype=torch.int32, device=self.device
        )

    @torch.no_grad()
    def refresh(self, positions: Tensor) -> None:
        expected_shape = (self.batch_size, self.n_real, 3)
        if tuple(positions.shape) != expected_shape:
            raise ValueError(
                f"positions must have shape {expected_shape}, got {tuple(positions.shape)}"
            )
        positions = positions.to(self.device, torch.float32)
        frac_unwrapped = small_matmul(positions, self.inv_cells)
        wraps = torch.floor(frac_unwrapped).to(torch.int32)
        wrapped = small_matmul(frac_unwrapped - wraps.float(), self.cells)
        wrapped_flat = wrapped.reshape(self.total_atoms, 3)

        self.neighbor_matrix.fill_(self.total_atoms)
        self.neighbor_shifts.zero_()
        self.num_neighbors.zero_()
        batch_build_cell_list(
            wrapped_flat,
            self.candidate_cutoff,
            self.cells,
            self.pbc,
            self.batch_idx,
            *self.cell_list_cache,
        )
        batch_query_cell_list(
            wrapped_flat,
            self.cells,
            self.pbc,
            self.candidate_cutoff,
            self.batch_idx,
            *self.cell_list_cache,
            self.neighbor_matrix,
            self.neighbor_shifts,
            self.num_neighbors,
            False,
        )

        neighbor_matrix = self.neighbor_matrix.view(
            self.batch_size, self.n_real, self.buffer_max
        )
        neighbor_shifts = self.neighbor_shifts.view(
            self.batch_size, self.n_real, self.buffer_max, 3
        )
        num_neighbors = self.num_neighbors.view(self.batch_size, self.n_real)
        for replica, builder in enumerate(self.builders):
            builder.reference_positions.copy_(positions[replica])
            builder.reference_wraps.copy_(wraps[replica])
            builder.wrap_delta.zero_()
            builder.neighbor_matrix.copy_(
                neighbor_matrix[replica] - replica * self.n_real
            )
            builder.neighbor_shifts.copy_(neighbor_shifts[replica])
            builder.num_neighbors.copy_(num_neighbors[replica])
            builder._candidate_capacity_valid.copy_(
                (num_neighbors[replica] <= self.buffer_max).all().reshape(1)
            )
            builder._candidate_initialized = True
