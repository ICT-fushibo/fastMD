"""Different systems of a model the engine differentiates, in slots of one whole-step CUDA graph.

The heterogeneous counterpart of :mod:`fastmd.execution.replica_batch`: S
systems, each with its own atoms, cell, positions, momenta and thermostat, run
as one disjoint union in **slots**. Slot ``s`` owns the fixed rows
``[start_s, start_s + ceiling_s)`` of the union (:class:`SlotLayout`); its
system is the first ``n_s <= ceiling_s`` of them and the slot's other rows are
vacant (no candidate, no real edge: only padding self-loops, which add nothing;
no energy counted, never moved). A system of at most the
slot's ceiling can be written into a slot in place (positions, species, cell
and the ``systems`` buffers; the union's candidates refreshed), so a captured
graph serves another system without a recapture. Batch-aware parts let the
single-system :class:`~fastmd.execution.whole_step_cuda_graph.WholeStepGraphRunner`
run the union unchanged, as for replicas:

* :class:`SlotUnionGraphBuilder`: the edges-only fixed-capacity builder over
  the slots. One batched cell list with one batch index per system (vacant
  rows left out) gives the candidates, so no edge joins two systems; each
  build wraps every row with its own system's inverse cell, the single-system
  re-filter kernels read the cell of each atom's system (their
  ``SYSTEM_CELLS`` path), and far self-loop padding takes the far image of its
  row's system. It publishes the systems with their cells and atom counts
  (:class:`~fastmd.adapters.protocol.SystemSegments`).
* :func:`system_energy_derivative`: one FP64 energy per system (its occupied
  rows' sum) and one backward pass for every system's forces, the model's
  edge vectors from the engine's per-edge shifts (``image @ cells[system]``,
  row by row: the adapters' own arithmetic,
  :class:`~fastmd.adapters.protocol.EdgeShiftInput`).
* the integrators, one thermostat per system: :class:`SystemNoseHooverChainNVT`
  (ASE's Nose-Hoover chain per system: the single-system Triton kernels, one
  launch per size class over the slots, each system bit for bit a
  single-system integrator of its atom count fed the same forces) and
  :class:`SystemVelocityVerlet` (NVE and Berendsen NVT with each system's
  kinetic energy, scale, centre-of-mass correction and ``3 n_s`` degrees of
  freedom).
* :class:`SlotWholeStepRunner`: the replica runner over the slots (one FP64
  energy and one health flag per system).

``fastmd.interfaces.multi_system_md`` drives these parts.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np
import torch
from torch import Tensor

from fastmd.adapters.protocol import SystemSegments
from fastmd.engine.energy_derivative import engine_derivative
from fastmd.engine.entry_config import scoped_call
from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder, self_loop_padding
from fastmd.graph.geometry import small_matmul
from fastmd.graph.gpu_graph_builder import MAX_IMAGE
from fastmd.execution.gpu_integrator import GPUIntegrator, StatefulIntegrator
from fastmd.execution.gpu_nhc import _C_Q, _HAS_TRITON, _TRITON_MAX_ELEMS, nhc_constants
from fastmd.execution.replica_batch import UNION_GROWTH_HEADROOM, ReplicaWholeStepRunner, graph_builder_options

if _HAS_TRITON:
    import triton
    import triton.language as tl

    from fastmd.graph.fixed_capacity_builder import allocate_cell_list, estimate_batch_cell_list_sizes
    from fastmd.graph.neighbor_list_backend import batch_build_cell_list, batch_query_cell_list
    from fastmd.execution.gpu_nhc import _nhc_post_force_kernel, _nhc_pre_force_kernel

    @triton.jit
    def _nhc_pre_force_systems_kernel(q_ptr, p_ptr, f_ptr, m_ptr, eta_ptr, pe_ptr, c_ptr, offsets, mass_offsets,
                                      n_elems, blocks, BLOCK: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
                                      TLOOP: tl.constexpr, F32_KICK: tl.constexpr, FAST: tl.constexpr):
        # The single-system pre-force half on system program_id(0) if its size class is this launch's BLOCK:
        # its rows (an offset that is a multiple of 16 elements, as a single system's buffers are aligned), its
        # masses, its chain and its constants (its atom count's DOF and Q).
        system = tl.program_id(0).to(tl.int64)
        if tl.load(blocks + system) == BLOCK:
            rows = tl.multiple_of(tl.load(offsets + system), 16)
            masses = tl.multiple_of(tl.load(mass_offsets + system), 16)
            chain = system * M
            _nhc_pre_force_kernel(q_ptr + rows, p_ptr + rows, f_ptr + rows, m_ptr + masses, eta_ptr + chain,
                                  pe_ptr + chain, c_ptr + system * C, tl.load(n_elems + system), BLOCK, M, TLOOP,
                                  F32_KICK, FAST)

    @triton.jit
    def _nhc_post_force_systems_kernel(p_ptr, fin_ptr, fcache_ptr, m_ptr, eta_ptr, pe_ptr, c_ptr, ein_ptr, eout_ptr,
                                       offsets, mass_offsets, n_elems, blocks, BLOCK: tl.constexpr, M: tl.constexpr,
                                       C: tl.constexpr, TLOOP: tl.constexpr, F32_KICK: tl.constexpr,
                                       HAS_E: tl.constexpr, FAST: tl.constexpr):
        # The single-system post-force half on system program_id(0) of this launch's size class, with its energy.
        system = tl.program_id(0).to(tl.int64)
        if tl.load(blocks + system) == BLOCK:
            rows = tl.multiple_of(tl.load(offsets + system), 16)
            masses = tl.multiple_of(tl.load(mass_offsets + system), 16)
            chain = system * M
            _nhc_post_force_kernel(p_ptr + rows, fin_ptr + rows, fcache_ptr + rows, m_ptr + masses, eta_ptr + chain,
                                   pe_ptr + chain, c_ptr + system * C, ein_ptr + system, eout_ptr + system,
                                   tl.load(n_elems + system), BLOCK, M, TLOOP, F32_KICK, HAS_E, FAST)


def _aligned(count: int) -> int:
    """``count`` rounded up to a multiple of 16."""
    return -(-int(count) // 16) * 16


def nhc_block(n_atoms: int) -> int:
    """The Triton block of a single-system Nose-Hoover chain of ``n_atoms`` atoms (its size class)."""
    return max(16, 1 << (3 * int(n_atoms) - 1).bit_length())


def nhc_num_warps(block: int) -> int:
    """The warps of a single-system Nose-Hoover chain launch of ``block`` (``GPUNoseHooverChainNVT``'s)."""
    return 1 if block <= 64 else 2 if block <= 128 else 4 if block <= 2048 else 8 if block <= 8192 else 16


@dataclass(frozen=True)
class SlotLayout:
    """The slots of one union: slot ``s`` is the rows ``[starts[s], starts[s] + ceilings[s])``; the system in it
    occupies its first rows (at most ``ceilings[s]``). Fixed for the life of the union's captured graphs."""

    ceilings: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.ceilings or any(int(c) < 1 for c in self.ceilings):
            raise ValueError("a slot layout needs one or more slots of at least one atom")

    @property
    def count(self) -> int:
        return len(self.ceilings)

    @cached_property
    def starts(self) -> tuple[int, ...]:
        return tuple(int(v) for v in np.concatenate(([0], np.cumsum(self.ceilings)[:-1])))

    @property
    def rows(self) -> int:
        return int(sum(self.ceilings))

    def system_of_row(self, device) -> Tensor:
        """``[rows]`` int32: the slot of each row."""
        return torch.repeat_interleave(torch.arange(self.count, dtype=torch.int32),
                                       torch.tensor(self.ceilings)).to(device)

    def occupied_rows(self, atoms) -> np.ndarray:
        """The occupied rows (int64, ascending) when slot ``s`` holds ``atoms[s]`` atoms."""
        return np.concatenate([np.arange(start, start + int(n)) for start, n in zip(self.starts, atoms)])



def segment_sums(values: Tensor, ceilings: Tensor, atoms: Tensor) -> Tensor:
    """``(S, ...)`` float64 sums of ``values`` (``(rows, ...)``, every row of the slots in order) over each
    system's rows, the first ``atoms[s]`` of its slot of ``ceilings[s]`` rows: one ``torch.segment_reduce`` over
    the segments (system 0, vacant rows of slot 0, system 1, ...), O(rows), deterministic, and a system's sum
    reads its own rows only (a non-finite row stays in its system). The lengths are device values (``atoms``
    rewritten in place: no capture holds them), unchecked against the rows (no host synchronization). Values with
    more dimensions are reduced column by column, as 1-D segments in one call (the 1-D reduction is the fast
    one for long segments)."""
    lengths = torch.stack((atoms, ceilings - atoms), dim=1).reshape(-1)
    rows, trailing = values.shape[0], tuple(values.shape[1:])
    columns = values.to(torch.float64).reshape(rows, -1).t().reshape(-1)
    sums = torch.segment_reduce(columns, "sum", lengths=lengths.repeat(columns.shape[0] // max(rows, 1)), unsafe=True)
    return sums.view(-1, lengths.shape[0])[:, ::2].t().reshape((atoms.shape[0],) + trailing)


def validate_cell(cell, name: str = "cell") -> np.ndarray:
    """A system's cell as a float64 ``(3, 3)`` array: finite, with a positive volume (rows are the lattice
    vectors, right- or left-handed). Raises ValueError otherwise (host only)."""
    cell = np.asarray(cell.detach().cpu() if torch.is_tensor(cell) else cell, dtype=np.float64)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    volume = abs(float(np.linalg.det(cell)))
    if not volume > 1e-6 * max(float(np.linalg.norm(cell, axis=1).prod()), 1e-300):
        raise ValueError(f"{name} is singular (volume {volume:.3g} Å^3)")
    return cell


class SlotUnionGraphBuilder(FixedCapacityGraphBuilder):
    """The edges-only fixed-capacity builder over the slots of :class:`SlotLayout` (see the module docstring).

    ``cells``: ``[count, 3, 3]`` (float64 for exactness: the builder takes its float32 search cells from them
    and gives a float64 model its exact cells); ``atomic_numbers``: ``[rows]`` (a vacant row: any element of
    the model, e.g. its slot's first); ``atoms``: the atoms of each slot's system. The candidate list is a
    batched cell list over the occupied rows with one batch index per system; the edge capacity, the overflow,
    the skin test and the candidate refresh are the union's. :meth:`write_system` puts another system into a
    slot in place. The batch's ``systems`` is set before the adapter's ``prepare`` sees the batch.
    ``admit_cells``: ``[k, 3, 3]`` cells of systems written later; the sort keys are sized for them too.
    """

    def __init__(self, *, layout: SlotLayout, cells: Tensor, atomic_numbers: Tensor, atoms, pbc: Tensor | None = None,
                 admit_cells: Tensor | None = None, **kwargs) -> None:
        for name in ("ghosts", "candidates_from"):
            if kwargs.get(name) is not None:
                raise ValueError(f"a slot union builder has no {name}")
        if kwargs.get("minimum_image"):
            raise ValueError("a slot union builder has no minimum_image")
        if kwargs.get("line_graph_cutoff") is not None:
            raise ValueError("a slot union builder builds the edges only")
        padding = kwargs.get("padding")
        if padding is None or padding.kind == "sink_atoms":
            raise ValueError("a slot union builder pads with self-loops (far_self_loops or exact_rc_loops)")
        device = torch.device(kwargs.get("device", "cuda"))
        self.layout = layout
        count, rows = layout.count, layout.rows
        cells64 = torch.as_tensor(cells, dtype=torch.float64).reshape(-1, 3, 3).to(device)
        atoms = [int(n) for n in atoms]
        if cells64.shape[0] != count or len(atoms) != count:
            raise ValueError(f"a slot union of {count} slots needs {count} cells and atom counts")
        if any(not 1 <= n <= c for n, c in zip(atoms, layout.ceilings)):
            raise ValueError("every slot holds a system of at least one atom and at most its ceiling")
        if pbc is not None and not bool(torch.as_tensor(pbc).all()):
            raise ValueError("a slot union builder needs systems periodic in all three directions")
        positions_dtype = kwargs.get("positions_dtype", torch.float32)
        system_of_row = layout.system_of_row(device)
        self.systems = SystemSegments(
            count=count,
            row_ptr=torch.tensor(layout.starts + (rows,), dtype=torch.int64, device=device),
            system_of_row=system_of_row,
            cells=cells64.to(positions_dtype).contiguous(),
            atoms=torch.tensor(atoms, dtype=torch.int64, device=device))
        prepare = kwargs.pop("prepare", None)
        systems = self.systems

        def prepare_systems(batch, prepare=prepare):
            batch.systems = systems
            return batch if prepare is None else prepare(batch)

        kwargs["prepare"] = prepare_systems
        super().__init__(cell=cells64[0], atomic_numbers=atomic_numbers, pbc=None, **kwargs)
        if self.n_real != rows:
            raise ValueError(f"atomic_numbers must hold the layout's {rows} rows")
        # The re-filter kernels read the cell of each atom's system from [count, 3, 3] (SYSTEM_CELLS); the base's
        # single-system cell state (cell 0's) is replaced or dropped.
        self.atoms = list(atoms)
        self._system_cells = {"SYSTEM_CELLS": True}
        self._row_systems = self.batch_idx = system_of_row
        self.cell = torch.empty((count, 3, 3), dtype=torch.float32, device=device)
        self.cells = self.cell
        self.inv_cells = torch.empty_like(self.cell)
        self.inv_cell = None
        self.pbc = torch.ones((count, 3), dtype=torch.bool, device=device)
        # Per row: its system's cell and inverse (float32; the model's dtype for float64 positions). The inverses
        # column-major as torch.linalg.inv returns them: the wrap's sums then take the single-system builder's
        # order (bit for bit its wrap of the same coordinates).
        self.row_cells = torch.empty((rows, 3, 3), dtype=torch.float32, device=device)
        self.row_inv_cells = torch.empty_like(self.row_cells).transpose(1, 2)
        self.row_model_cells = (None if positions_dtype == torch.float32 else
                                torch.empty((rows, 3, 3), dtype=positions_dtype, device=device))
        self._loop_images = (torch.zeros((count, 3), dtype=torch.int32, device=device)
                             if self._pad_loop is not None and self.padding.kind == "far_self_loops" else None)
        # The occupied rows: the cell list's rows and batch indices, and the isolated-atom count.
        self.occupied = torch.zeros(rows, dtype=torch.bool, device=device)
        # No float64 single-kernel prologue (one cell): every build wraps row by row.
        self._float64_prologue = self._unsorted_edges = False
        self._wrapped32 = self._prologue_matrices = None
        self._cell_list_cells = 0
        for slot in range(count):
            self._set_cell(slot, cells64[slot])
        self._set_occupancy()
        # The pair sort's keys (``sorted`` edge order), from the thinnest cell (the base chose them by the first
        # system's): fixed from here on, since a captured graph keeps them (:meth:`check_cell`).
        self.certified_max_image = self._image_bound(self.cell if admit_cells is None else torch.cat(
            (self.cell, torch.as_tensor(admit_cells, device=device).reshape(-1, 3, 3).float())))
        if self.certified_max_image > self.sort_key_max_image:
            self.sort_key_dtype, self.sort_key_max_image = torch.int64, MAX_IMAGE

    # ------------------------------------------------------------------ the systems' tables
    def _set_cell(self, slot: int, cell64: Tensor) -> None:
        """Slot ``slot``'s cell in every table (float32 search cell and its inverse, the model's cell, the rows'
        copies, the far self-loop image), in place."""
        cell32 = cell64.to(torch.float32)
        self.cell[slot].copy_(cell32)
        # The single-system builder's inverse of its float32 cell (bitwise its wrap).
        self.inv_cells[slot].copy_(torch.linalg.inv(cell32))
        self.systems.cells[slot].copy_(cell64)
        start, stop = self.layout.starts[slot], self.layout.starts[slot] + self.layout.ceilings[slot]
        self.row_cells[start:stop].copy_(cell32.expand(stop - start, 3, 3))
        self.row_inv_cells[start:stop].copy_(self.inv_cells[slot].expand(stop - start, 3, 3))
        if self.row_model_cells is not None:
            self.row_model_cells[start:stop].copy_(self.systems.cells[slot].expand(stop - start, 3, 3))
        if self._loop_images is not None:
            image, _, _ = self_loop_padding(self.padding, cell32, self.atom_graph_cutoff)
            self._loop_images[slot].copy_(torch.tensor(image, dtype=torch.int32))
        if slot == 0:
            self.batch.cell.copy_(self.systems.cells[0])

    def _set_occupancy(self) -> None:
        """The occupied rows, the per-system row table and the cell list (reallocated where the cells need more
        bins), from ``self.atoms``; the sort keys' image bound of the current cells."""
        layout, device = self.layout, self.device
        occupied = layout.occupied_rows(self.atoms)
        self.systems.atoms.copy_(torch.tensor(self.atoms, dtype=torch.int64))
        self.occupied.zero_()
        self._active_rows = torch.as_tensor(occupied, device=device)
        self.occupied.index_fill_(0, self._active_rows, True)
        self._vacancies = int(occupied.shape[0]) < layout.rows
        self._active_batch = self.systems.system_of_row.index_select(0, self._active_rows).contiguous()
        cells, radius = estimate_batch_cell_list_sizes(self.cells, self.pbc, self.candidate_cutoff)
        if cells > self._cell_list_cells:  # the cell list is eager only: a new one is no recapture
            self._cell_list_cells = int(cells)
            self.cell_list_cache = allocate_cell_list(self.n_real, int(cells), radius, device)
        else:
            self.cell_list_cache[1].copy_(radius)  # each system's search radius in cells

    def _image_bound(self, cells: Tensor) -> int:
        """The largest candidate image plus the wrap margin in any of ``cells`` (``[k, 3, 3]``;
        ``FixedCapacityGraphBuilder``'s certified bound, per cell)."""
        cells = cells.double()
        volume = torch.linalg.det(cells).abs()
        heights = torch.stack([volume / torch.linalg.cross(cells[:, (axis + 1) % 3],
                                                           cells[:, (axis + 2) % 3]).norm(dim=1) for axis in range(3)])
        return math.ceil(self.candidate_cutoff / max(float(heights.min()), 1e-6)) + 2

    def check_cell(self, cell) -> Tensor:
        """``cell`` (``(3, 3)``) as a float64 device tensor if this union can hold a system in it: finite and
        non-singular, and, for sorted edges, no thinner than the union's sort keys allow (they are fixed at
        construction). Raises ValueError otherwise; changes nothing."""
        cell64 = torch.as_tensor(validate_cell(cell), dtype=torch.float64, device=self.device)
        if self.sorted_edges and self._image_bound(cell64[None].float()) > self.sort_key_max_image:
            raise ValueError(f"a cell thinner than the union's sort keys allow (image bound "
                             f"{self._image_bound(cell64[None].float())} > {self.sort_key_max_image}): build a new "
                             "union for it")
        return cell64

    def write_system(self, slot: int, *, atomic_numbers: Tensor, cell, fill_number: int) -> None:
        """Put a system of ``len(atomic_numbers)`` atoms (at most the slot's ceiling) with ``cell`` (``(3, 3)``)
        into ``slot``, in place: its species (vacant rows: ``fill_number``), cell tables and atom count; the
        checks (the ceiling, :meth:`check_cell`) run before anything changes. The caller writes the positions and
        refreshes the candidates (``refresh_candidates``)."""
        numbers = torch.as_tensor(atomic_numbers, dtype=torch.int32).reshape(-1)
        ceiling = self.layout.ceilings[slot]
        if not 1 <= numbers.shape[0] <= ceiling:
            raise ValueError(f"slot {slot} holds 1 to {ceiling} atoms, got {numbers.shape[0]}")
        cell64 = self.check_cell(cell)
        start = self.layout.starts[slot]
        species = torch.full((ceiling,), int(fill_number), dtype=torch.int32)
        species[:numbers.shape[0]] = numbers
        species = species.to(self.device)
        self.atomic_numbers_real[start:start + ceiling].copy_(species)
        self.batch.species[start:start + ceiling].copy_(species)
        self.atoms[slot] = int(numbers.shape[0])
        self._set_cell(slot, cell64)
        self._set_occupancy()

    # ------------------------------------------------------------------ candidates and builds
    def _wrap(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(fractional coordinates in [0, 1), integer wraps, wrapped Cartesian coordinates) of float32
        ``positions``, each row with its own system's cell: the single-system arithmetic row by row."""
        frac_unwrapped = small_matmul(positions.unsqueeze(1), self.row_inv_cells).squeeze(1)
        wraps = torch.floor(frac_unwrapped).to(torch.int32)
        frac = frac_unwrapped - wraps.float()
        return frac, wraps, small_matmul(frac.unsqueeze(1), self.row_cells).squeeze(1)

    def _build_candidates(self, wrapped_positions: Tensor) -> None:
        self.neighbor_matrix.fill_(self.n_real)
        self.neighbor_shifts.zero_()
        self.num_neighbors.zero_()
        import warp as wp

        # nvalchemiops runs its Warp kernels on torch's current stream (ordered with the wrap and the filter).
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            if not self._vacancies:
                batch_build_cell_list(wrapped_positions, self.candidate_cutoff, self.cells, self.pbc,
                                      self._row_systems, *self.cell_list_cache)
                batch_query_cell_list(wrapped_positions, self.cells, self.pbc, self.candidate_cutoff,
                                      self._row_systems, *self.cell_list_cache, self.neighbor_matrix,
                                      self.neighbor_shifts, self.num_neighbors, False)
                self._candidate_capacity_valid.copy_((self.num_neighbors <= self.buffer_max).all().reshape(1))
                return
            # Vacant rows: the occupied rows only, compacted, then scattered back (vacant rows get no
            # candidates and are nobody's).
            rows = self._active_rows
            count = int(rows.numel())
            per_dimension, radius, shifts, mapping, per_cell, starts, cell_atoms = self.cell_list_cache
            cache = (per_dimension, radius, shifts[:count], mapping[:count], per_cell, starts, cell_atoms[:count])
            compact = wrapped_positions.index_select(0, rows)
            matrix = torch.full((count, self.buffer_max), count, dtype=torch.int32, device=self.device)
            images = torch.zeros((count, self.buffer_max, 3), dtype=torch.int32, device=self.device)
            counts = torch.zeros(count, dtype=torch.int32, device=self.device)
            batch_build_cell_list(compact, self.candidate_cutoff, self.cells, self.pbc, self._active_batch, *cache)
            batch_query_cell_list(compact, self.cells, self.pbc, self.candidate_cutoff, self._active_batch, *cache,
                                  matrix, images, counts, False)
        filled = torch.arange(self.buffer_max, device=self.device)[None, :] < counts[:, None]
        mapped = torch.where(filled, rows[matrix.long().clamp(max=count - 1)],
                             torch.full_like(rows[:1], self.n_real)).int()
        self.neighbor_matrix.index_copy_(0, rows, mapped)
        self.neighbor_shifts.index_copy_(0, rows, images)
        self.num_neighbors.index_copy_(0, rows, counts)
        self._candidate_capacity_valid.copy_((counts <= self.buffer_max).all().reshape(1))

    def _displacement(self, positions: Tensor) -> Tensor:
        """The skin test's largest move since the candidate refresh: of a row, in Å (device scalar)."""
        return torch.linalg.vector_norm(positions - self.reference_positions, dim=1).max()

    @torch.no_grad()
    def refresh_candidates(self, positions: Tensor) -> None:
        positions = positions.to(device=self.device, dtype=torch.float32)
        _, wraps, wrapped = self._wrap(positions)
        self.reference_positions.copy_(positions)
        self.reference_wraps.copy_(wraps)
        self.wrap_delta.zero_()
        self._build_candidates(wrapped)
        self._candidate_initialized = True

    @torch.no_grad()
    def build(self, positions: Tensor, *, rebuild_candidates: bool = True, predicate=None):
        """:meth:`FixedCapacityGraphBuilder.build` (edges only) with each row wrapped in its own system's cell,
        the re-filter in each atom's system's cell and the isolated atoms counted over the occupied rows."""
        given = positions
        positions = positions.to(device=self.device, dtype=torch.float32)
        if positions.shape != (self.n_real, 3):
            raise ValueError(f"positions must have shape {(self.n_real, 3)}, got {tuple(positions.shape)}")
        frac, wraps, wrapped = self._wrap(positions)
        self.frac_coords[: self.n_real].copy_(frac)
        if rebuild_candidates:
            self.reference_positions.copy_(positions)
            self.reference_wraps.copy_(wraps)
            self.wrap_delta.zero_()
            self._build_candidates(wrapped)
            self._candidate_initialized = True
        elif not self._candidate_initialized:
            raise RuntimeError("refresh_candidates must run before candidate reuse")
        else:
            self.wrap_delta.copy_(wraps - self.reference_wraps)
        max_displacement = self._displacement(positions)
        candidate_valid = (max_displacement * 2.0 <= self.skin + 1.0e-6) & self._candidate_capacity_valid[0]
        self._max_displacement.copy_(max_displacement.reshape(1))
        self._candidate_valid.copy_(candidate_valid.reshape(1))
        self._materialize_edges(wrapped)
        # A vacant row has no edge and is not an isolated atom (in every build: a captured graph serves the systems
        # written into its slots later).
        self._isolated_atom_count.copy_(((self.edge_counts == 0) & self.occupied).sum().to(torch.int32).reshape(1))
        self._sort_edges()  # also sets the overflow flag
        if self.positions.dtype == torch.float32:
            self.positions[: self.n_real].copy_(wrapped)
        else:
            # The caller's positions minus the same whole lattice vectors of each row's system (no GEMM).
            shift = small_matmul(wraps.to(self.positions.dtype).unsqueeze(1), self.row_model_cells).squeeze(1)
            torch.sub(given.to(device=self.device, dtype=self.positions.dtype), shift,
                      out=self.positions[: self.n_real])
        self.batch.edge_index[:, 0].copy_(self.center)
        self.batch.edge_index[:, 1].copy_(self.neighbor)
        self.batch.edge_image.copy_(self.images)
        return self.graph, self.batch.status


def slot_union_builder(adapter, *, layout: SlotLayout, cells, atomic_numbers: Tensor, atoms, u_capacity: int,
                       candidate_skin: float, device) -> SlotUnionGraphBuilder:
    """The slot union builder of ``adapter`` (edges only) at ``u_capacity`` undirected pairs: its padding rule,
    ``prepare`` and builder options; candidates within cutoff + ``candidate_skin``."""
    cutoff = float(adapter.topology_spec().cutoff)
    return SlotUnionGraphBuilder(
        layout=layout, cells=cells, atomic_numbers=atomic_numbers, atoms=atoms, u_capacity=int(u_capacity),
        atom_graph_cutoff=cutoff, candidate_cutoff=cutoff + candidate_skin, device=device,
        padding=adapter.padding_rule(), prepare=adapter.prepare, **graph_builder_options(adapter))


def slot_union_demand(adapter, positions: Tensor, *, layout: SlotLayout, cells, atomic_numbers: Tensor, atoms,
                      candidate_skin: float, device) -> int:
    """The union's undirected pair count at ``positions`` (``(rows, 3)``): one eager build with a capacity
    estimated from the density, again with the count found if that was too small."""
    rows = int(sum(atoms))
    cutoff = float(adapter.topology_spec().cutoff)
    volume = float(np.abs(np.linalg.det(np.asarray(torch.as_tensor(cells, dtype=torch.float64).cpu()))).sum())
    guess = int(4.0 / 3.0 * math.pi * cutoff ** 3 * rows / max(volume, 1e-9) * rows / 2 * 1.5) + 1024
    while True:
        builder = slot_union_builder(adapter, layout=layout, cells=cells, atomic_numbers=atomic_numbers, atoms=atoms,
                                     u_capacity=guess, candidate_skin=candidate_skin, device=device)
        builder.refresh_candidates(positions)
        builder.build(positions, rebuild_candidates=False)
        edges = int(builder.edge_offsets[-1].item())
        del builder
        if edges <= 2 * guess:
            return -(-edges // 2)
        guess = -(-edges // 2)


def system_edge_shifts(batch, dtype: torch.dtype, cells: Tensor | None = None) -> Tensor:
    """``[edge rows, 3]``: each edge's Cartesian shift ``edge_image @ cells[s]`` in its own system's cell, row by
    row (``image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]``, an adapter's own
    arithmetic for one cell: the same bits per edge), in ``dtype``. ``cells``: ``[S, 3, 3]`` in place of the
    systems' cells (strained ones, for a stress)."""
    systems = batch.systems
    image = batch.edge_image.to(dtype)
    cells = (systems.cells if cells is None else cells).index_select(
        0, systems.system_of_row.index_select(0, batch.edge_index[:, 0])).to(dtype)
    return image[:, 0:1] * cells[:, 0] + image[:, 1:2] * cells[:, 1] + image[:, 2:3] * cells[:, 2]


def system_volumes(cells: Tensor) -> Tensor:
    """``(S,)``: |det| of each ``[S, 3, 3]`` cell as the triple product (elementwise: no solver, no host
    synchronization)."""
    return (cells[:, 0] * torch.linalg.cross(cells[:, 1], cells[:, 2], dim=1)).sum(dim=1).abs()


def system_energy_derivative(adapter, batch, inputs, *, cell: Tensor | None = None, stress: bool = False):
    """``(energies (S,) float64, forces (rows, 3))`` of the systems of a slot union: each system's energy is the
    FP64 sum of its occupied rows' energies (``adapter.node_energy`` with the engine's per-edge shifts,
    :func:`system_edge_shifts`; :func:`segment_sums` over ``batch.systems``); one backward pass of their sum
    gives every system's forces (no term couples two systems; a vacant row's force is 0). ``cell``: the model's
    ``cell`` argument in place of the batch's (a check that the model reads the shifts, not the cell).

    ``stress``: also each system's stress, ``(energies, forces, stresses (S, 3, 3))`` in eV/Å³ in the dtype of
    the positions: a zero strain leaf per system deforms the rows of its system by its symmetric part,
    ``x (1 + e_s)`` row by row (no GEMM: a row's arithmetic does not depend on the row count), the shifts come
    from the strained cells ``cells_s (1 + e_s)``, and the same backward pass gives every system's dE/de,
    divided by its volume (the single-system engine-owned derivative's stress, per system). The strain and the
    strained cells are float64 and cast to the model's dtype per row and per edge: the model's inputs are the
    same values (the strain is zero), and each system's dE/de accumulates its rows' and edges' terms in float64
    (in the model's dtype the scattered sums of one system's many cancelling terms would round by more than the
    single-system path's reduction)."""
    rows, systems = batch.species.shape[0], batch.systems
    with torch.enable_grad():
        leaf = batch.positions.detach().requires_grad_(True)
        if not stress:
            node = adapter.node_energy(inputs, leaf, batch.cell if cell is None else cell,
                                       edge_shifts=system_edge_shifts(batch, leaf.dtype))
            energies = segment_sums(node[:rows], systems.row_ptr[1:] - systems.row_ptr[:-1], systems.atoms)
            (grad,) = torch.autograd.grad(energies.sum(), (leaf,))
            return energies.detach(), -grad[:rows]
        strain = torch.zeros((systems.count, 3, 3), dtype=torch.float64, device=leaf.device, requires_grad=True)
        symmetric = 0.5 * (strain + strain.transpose(1, 2))
        per_row = symmetric.index_select(0, systems.system_of_row)
        exact = leaf.to(torch.float64)
        positions = leaf + (exact[:, 0:1] * per_row[:, 0] + exact[:, 1:2] * per_row[:, 1]
                            + exact[:, 2:3] * per_row[:, 2]).to(leaf.dtype)
        cells = systems.cells.to(torch.float64)
        strained = cells + (cells[:, :, 0:1] * symmetric[:, 0:1] + cells[:, :, 1:2] * symmetric[:, 1:2]
                            + cells[:, :, 2:3] * symmetric[:, 2:3])
        node = adapter.node_energy(inputs, positions, batch.cell if cell is None else cell,
                                   edge_shifts=system_edge_shifts(batch, leaf.dtype, cells=strained))
        energies = segment_sums(node[:rows], systems.row_ptr[1:] - systems.row_ptr[:-1], systems.atoms)
        grad, strain_grad = torch.autograd.grad(energies.sum(), (leaf, strain))
    volumes = system_volumes(systems.cells.detach()).to(torch.float64)
    return energies.detach(), -grad[:rows], (strain_grad / volumes[:, None, None]).to(leaf.dtype)


@dataclass
class SystemNHCState:
    """Views into one flat float64 buffer ``[q | p | F | eta | p_eta | E]``: q, p and F hold slot ``s``'s
    ``3 n_s`` coordinates from element ``offsets[s]`` (a multiple of 16; room for its ceiling), eta and p_eta
    ``(S, tchain)``, E ``(S,)``."""

    flat: Tensor
    q: Tensor
    p: Tensor
    f: Tensor
    eta: Tensor
    p_eta: Tensor
    potential_energy: Tensor

    @staticmethod
    def allocate(elements: int, systems: int, tchain: int, device) -> "SystemNHCState":
        flat = torch.zeros(3 * elements + 2 * systems * tchain + systems, dtype=torch.float64, device=device)
        q, p, f = (flat[k * elements: (k + 1) * elements] for k in range(3))
        offset = 3 * elements
        eta = flat[offset: offset + systems * tchain].view(systems, tchain)
        offset += systems * tchain
        p_eta = flat[offset: offset + systems * tchain].view(systems, tchain)
        offset += systems * tchain
        return SystemNHCState(flat, q, p, f, eta, p_eta, flat[offset: offset + systems])


class SystemNoseHooverChainNVT(StatefulIntegrator):
    """ASE's ``NoseHooverChainNVT`` for the systems of a slot union, each with its own chain, degrees of freedom
    (``3 n_s``) and ``Q`` (one target temperature for all), and each slot at most ``_TRITON_MAX_ELEMS / 3``
    (5,461) atoms.

    Each half step is the single-system integrator's Triton kernel, one launch per size class (the block of
    ``3 n``) over all slots, each system on the launch of its own class and at an offset that is a multiple of
    16 elements: bit for bit a single-system
    :class:`~fastmd.execution.gpu_nhc.GPUNoseHooverChainNVT` of ``n_s`` atoms fed the same forces (an
    empirical property of this Triton, tested). ``positions``, ``momenta`` and ``forces`` are ``(rows, 3)``
    copies in the union's row layout (a vacant row: its parked position, zero momentum and force);
    :meth:`step_pre_force` returns the new positions (``(rows, 3)`` float64) and :meth:`step_post_force` takes
    ``(rows, 3)`` forces and one energy per system. :meth:`write_system` puts a new system into a slot (a fresh chain).
    No restart files.
    """

    ensemble = "nvt_nhc"
    comm = None

    def __init__(self, layout: SlotLayout, masses, positions, momenta, *, temperature_K: float, timestep: float,
                 tdamp: float, tchain: int = 3, tloop: int = 1, device="cuda",
                 force_dtype: torch.dtype = torch.float64, numpy_kick_semantics: bool = True) -> None:
        if not _HAS_TRITON:
            raise RuntimeError("the per-system Nose-Hoover chain needs Triton")
        if tchain < 1 or tloop < 1 or tdamp <= 0 or timestep < 0:
            raise ValueError("tchain and tloop must be >= 1, tdamp positive and timestep non-negative")
        if any(3 * c > _TRITON_MAX_ELEMS for c in layout.ceilings):
            raise ValueError(f"the per-system Nose-Hoover chain supports 3n <= {_TRITON_MAX_ELEMS} per slot")
        self.device = torch.device(device)
        self.layout = layout
        self.systems = S = layout.count
        self.tchain, self.tloop = int(tchain), int(tloop)
        self.temperature_K = float(temperature_K)
        self.dt, self.tdamp = float(timestep), float(tdamp)
        self.force_dtype = force_dtype
        self.f32_kick = bool(numpy_kick_semantics and force_dtype == torch.float32)
        self.generation = 0
        # Slot s: elements [offsets[s], offsets[s] + 3 ceiling) of q, p, F; masses from mass_offsets[s].
        sizes = [_aligned(3 * c) for c in layout.ceilings]
        mass_sizes = [_aligned(c) for c in layout.ceilings]
        self._offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
        self._mass_offsets = np.concatenate(([0], np.cumsum(mass_sizes)[:-1])).astype(np.int64)
        self.elements = int(sum(sizes))
        self.offsets = torch.as_tensor(self._offsets, device=self.device)
        self.mass_offsets = torch.as_tensor(self._mass_offsets, device=self.device)
        self.masses = torch.ones(int(sum(mass_sizes)), dtype=torch.float64, device=self.device)
        self.n_elems = torch.zeros(S, dtype=torch.int32, device=self.device)
        self.blocks = torch.zeros(S, dtype=torch.int32, device=self.device)
        self._consts_stride = _aligned(_C_Q + 2 * self.tchain)
        self.consts = torch.zeros((S, self._consts_stride), dtype=torch.float64, device=self.device)
        self.atoms = [0] * S
        self.state = SystemNHCState.allocate(self.elements, S, self.tchain, self.device)
        self._backup = SystemNHCState.allocate(self.elements, S, self.tchain, self.device)
        # The model's forces, copied into the state's layout before the post-force half.
        self._forces_in = torch.zeros(self.elements, dtype=force_dtype, device=self.device)
        # Each union row's three elements in q, p, F (a vacant row: its slot's elements past the system).
        row_elements = np.concatenate([self._offsets[s] + np.arange(3 * c) for s, c in enumerate(layout.ceilings)])
        self.row_elements = torch.as_tensor(row_elements, device=self.device)
        # One launch per size class a slot can hold (a block of 16 up to that of its ceiling).
        largest = max(nhc_block(c) for c in layout.ceilings)
        self._classes = [1 << k for k in range(4, largest.bit_length())]
        for s in range(S):
            self.write_system(s, masses=masses[s], positions=positions[s], momenta=momenta[s])

    # ------------------------------------------------------------------ systems
    @torch.no_grad()
    def write_system(self, slot: int, *, masses, positions, momenta, parked=None) -> None:
        """A system of ``n`` atoms (``masses`` ``(n,)`` amu, ``positions`` and ``momenta`` ``(n, 3)``) in
        ``slot``, with a fresh chain (``eta = p_eta = 0``), zero forces and energy (the caller sets F(q)
        before stepping: ``set_forces``); the slot's vacant rows at ``parked`` (``(3,)``; default the
        origin). Not in a capture."""
        masses = np.asarray(masses.detach().cpu() if torch.is_tensor(masses) else masses, dtype=np.float64).reshape(-1)
        n, ceiling = masses.shape[0], self.layout.ceilings[slot]
        if not 1 <= n <= ceiling:
            raise ValueError(f"slot {slot} holds 1 to {ceiling} atoms, got {n}")
        st, offset, moff = self.state, int(self._offsets[slot]), int(self._mass_offsets[slot])
        elements = slice(offset, offset + 3 * ceiling)
        for field in (st.q, st.p, st.f, self._forces_in):
            field[elements].zero_()
        if parked is not None:
            st.q[elements].view(ceiling, 3)[n:].copy_(torch.as_tensor(np.asarray(parked, dtype=np.float64)))
        st.q[offset: offset + 3 * n].copy_(self._as_f64(positions).reshape(-1))
        st.p[offset: offset + 3 * n].copy_(self._as_f64(momenta).reshape(-1))
        self.masses[moff: moff + _aligned(ceiling)].fill_(1.0)
        self.masses[moff: moff + n].copy_(torch.as_tensor(masses, device=self.device))
        st.eta[slot].zero_()
        st.p_eta[slot].zero_()
        st.potential_energy[slot].zero_()
        constants = nhc_constants(n, temperature_K=self.temperature_K, timestep=self.dt, tdamp=self.tdamp,
                                  tchain=self.tchain, tloop=self.tloop)
        self.consts[slot].zero_()
        self.consts[slot, : len(constants.values)].copy_(torch.tensor(constants.values, dtype=torch.float64))
        self.n_elems[slot] = 3 * n
        self.blocks[slot] = nhc_block(n)
        self.atoms[slot] = n

    def _as_f64(self, x) -> Tensor:
        if torch.is_tensor(x):
            return x.detach().to(device=self.device, dtype=torch.float64)
        return torch.as_tensor(np.asarray(x, dtype=np.float64), device=self.device)

    def _rows(self, field: Tensor) -> Tensor:
        return field.index_select(0, self.row_elements).view(-1, 3)

    @property
    def n_owned(self) -> int:
        return self.layout.rows

    @property
    def positions(self) -> Tensor:
        return self._rows(self.state.q)

    @property
    def momenta(self) -> Tensor:
        return self._rows(self.state.p)

    @property
    def forces(self) -> Tensor:
        return self._rows(self.state.f)

    @property
    def potential_energy(self) -> Tensor:
        return self.state.potential_energy

    @property
    def eta(self) -> Tensor:
        return self.state.eta

    @property
    def p_eta(self) -> Tensor:
        return self.state.p_eta

    def system_view(self, field: Tensor, slot: int) -> Tensor:
        """``(n_s, 3)`` view of slot ``slot``'s system in ``q``, ``p`` or ``F``."""
        offset = int(self._offsets[slot])
        return field[offset: offset + 3 * self.atoms[slot]].view(-1, 3)

    @torch.no_grad()
    def set_forces(self, forces, energy=None, state: SystemNHCState | None = None) -> None:
        """F(q) ``(rows, 3)`` (and one E(q) per system) for the current positions (capture-safe for device
        tensors)."""
        st = self.state if state is None else state
        forces = forces if torch.is_tensor(forces) else self._as_f64(forces)
        st.f.index_copy_(0, self.row_elements, forces.reshape(-1).to(torch.float64))
        if energy is not None:
            st.potential_energy.copy_(energy.reshape(self.systems) if torch.is_tensor(energy)
                                      else self._as_f64(energy).reshape(self.systems))

    # -------------------------------------------------------------- step halves
    def step_pre_force(self, state: SystemNHCState | None = None) -> Tensor:
        """Every system's ASE ``step()`` first half; returns a float64 ``(rows, 3)`` copy of the new positions."""
        st = self.state if state is None else state
        for block in self._classes:
            _nhc_pre_force_systems_kernel[(self.systems,)](
                st.q, st.p, st.f, self.masses, st.eta, st.p_eta, self.consts, self.offsets, self.mass_offsets,
                self.n_elems, self.blocks, BLOCK=block, M=self.tchain, C=self._consts_stride, TLOOP=self.tloop,
                F32_KICK=self.f32_kick, FAST=False, num_warps=nhc_num_warps(block), enable_fp_fusion=False,
            )
        return self._rows(st.q)

    def step_post_force(self, forces: Tensor, energy: Tensor | None = None,
                        state: SystemNHCState | None = None) -> None:
        """Every system's ASE ``step()`` second half with F(q_{n+1}) ``(rows, 3)`` (the model's dtype) and,
        optionally, one energy per system stored as its potential energy."""
        st = self.state if state is None else state
        forces_in = self._forces_in if forces.dtype == self._forces_in.dtype else torch.zeros(
            self._forces_in.shape, dtype=forces.dtype, device=self.device)
        forces_in.index_copy_(0, self.row_elements, forces.reshape(-1))
        has_e = energy is not None
        e = energy.reshape(self.systems).contiguous() if has_e else st.potential_energy  # the kernel reads it flat
        for block in self._classes:
            _nhc_post_force_systems_kernel[(self.systems,)](
                st.p, forces_in, st.f, self.masses, st.eta, st.p_eta, self.consts, e, st.potential_energy,
                self.offsets, self.mass_offsets, self.n_elems, self.blocks, BLOCK=block, M=self.tchain,
                C=self._consts_stride, TLOOP=self.tloop, F32_KICK=self.f32_kick, HAS_E=has_e, FAST=False,
                num_warps=nhc_num_warps(block), enable_fp_fusion=False,
            )

    # -------------------------------------------------------------- observables
    def kinetic_energy(self, state: SystemNHCState | None = None) -> Tensor:
        """``(S,)`` float64: each system's ``0.5 * vdot(p, p / m)``."""
        st = self.state if state is None else state
        return torch.stack([0.5 * (p * (p / m)).sum() for p, m in self._per_system(st.p)])

    def _per_system(self, field: Tensor):
        for slot in range(self.systems):
            offset, moff, n = int(self._offsets[slot]), int(self._mass_offsets[slot]), self.atoms[slot]
            yield field[offset: offset + 3 * n].view(n, 3), self.masses[moff: moff + n].view(n, 1)

    def temperature(self, state: SystemNHCState | None = None) -> Tensor:
        """``(S,)``: each system's ``Atoms.get_temperature()``."""
        from ase import units

        atoms = torch.tensor(self.atoms, dtype=torch.float64, device=self.device)
        return 2 * self.kinetic_energy(state) / (3 * atoms * units.kB)

    def thermostat_energy(self, state: SystemNHCState | None = None) -> Tensor:
        st = self.state if state is None else state
        ndof_kT, kT = self.consts[:, 3], self.consts[:, 2]
        Q = self.consts[:, _C_Q: _C_Q + self.tchain]
        return ndof_kT * st.eta[:, 0] + kT * st.eta[:, 1:].sum(dim=1) + (0.5 * (st.p_eta * st.p_eta) / Q).sum(dim=1)

    def conserved_energy(self, state: SystemNHCState | None = None) -> Tensor:
        """``(S,)``: each system's ``NoseHooverChainNVT.get_conserved_energy``."""
        st = self.state if state is None else state
        return st.potential_energy + self.kinetic_energy(st) + self.thermostat_energy(st)

    # ------------------------------------------------------------ transactions
    def backup(self, state: SystemNHCState | None = None) -> None:
        self._backup.flat.copy_((self.state if state is None else state).flat)

    def restore(self, state: SystemNHCState | None = None) -> None:
        (self.state if state is None else state).flat.copy_(self._backup.flat)

    def snapshot(self) -> tuple:
        return self.state.flat.clone(), self._backup.flat.clone()

    def load_snapshot(self, snapshot: tuple) -> None:
        state, backup = snapshot
        self.state.flat.copy_(state)
        self._backup.flat.copy_(backup)

    def new_state(self) -> SystemNHCState:
        return SystemNHCState.allocate(self.elements, self.systems, self.tchain, self.device)

    def state_dict(self) -> dict:
        raise NotImplementedError("restart files of per-system integrators are not implemented")

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError("restart files of per-system integrators are not implemented")


class SystemVelocityVerlet(GPUIntegrator):
    """:class:`~fastmd.execution.gpu_integrator.GPUIntegrator` (NVE, or Berendsen NVT with ``ensemble="nvt"``)
    for the systems of a slot union, on the union's ``(rows, 3)`` buffers.

    Each system's kinetic energy (its occupied rows), Berendsen scale and degrees of freedom (``3 n_s``) are its
    own, and ``fix_com`` removes each system's own mean momentum; a vacant row has no momentum or force and does
    not move. ``masses``: ``(rows,)`` (a vacant row: any positive mass); ``ceilings``: each slot's rows (int64);
    ``atoms``: ``(S,)`` int64, the systems' atom counts (rewritten in place when a system changes, as ``masses``
    and ``occupied``); per-system sums are :func:`segment_sums` in float64, rounded to the momenta's dtype.
    ``temperature`` and ``kinetic_energy`` return one value per system.
    """

    def __init__(self, masses: Tensor, dt_fs: float, *, ceilings: Tensor, atoms: Tensor, system_of_row: Tensor,
                 occupied: Tensor, ensemble: str = "nvt", temperature: float = 300.0, taut_fs: float = 100.0,
                 fix_com: bool | None = None) -> None:
        super().__init__(masses.reshape(-1), dt_fs, ensemble=ensemble, temperature=temperature, taut_fs=taut_fs,
                         fix_com=fix_com)
        self.ceilings = ceilings
        self.atoms = atoms
        self.system_of_row = system_of_row
        self.occupied = occupied

    def _per_system(self, rows: Tensor) -> Tensor:
        """``(S, ...)`` sums over each system's rows of ``rows`` (``(rows, ...)``), in ``rows``' dtype."""
        return segment_sums(rows, self.ceilings, self.atoms).to(rows.dtype)

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        return self._per_system(0.5 * momenta * momenta / self.masses).sum(dim=-1)

    def temperature(self, momenta: Tensor) -> Tensor:
        from ase import units

        return 2.0 * self.kinetic_energy(momenta) / (3.0 * self.atoms.to(momenta.dtype) * units.kB)

    def step_pre_force(self, positions: Tensor, momenta: Tensor, forces: Tensor,
                       masses: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Each system's Berendsen scaling, half kick, centre-of-mass correction and drift: ``(q_{n+1},
        p_{n+1/2})``."""
        masses = self.masses if masses is None else masses.reshape(-1, 1)
        rows = self.system_of_row.long()
        if self.ensemble == "nvt":
            scale = self._berendsen_scale(self.temperature(momenta).clamp_min(1e-12))
            momenta = momenta * scale.index_select(0, rows)[:, None]
        momenta = momenta + 0.5 * self.dt * forces
        if self.fix_com:
            mean = self._per_system(momenta) / self.atoms.to(momenta.dtype)[:, None]
            momenta = momenta - mean.index_select(0, rows) * self.occupied.to(momenta.dtype)[:, None]
        return positions + self.dt * momenta / masses, momenta


class SlotWholeStepRunner(ReplicaWholeStepRunner):
    """:class:`~fastmd.execution.replica_batch.ReplicaWholeStepRunner` over a :class:`SlotUnionGraphBuilder`
    with a per-system integrator (:class:`SystemNoseHooverChainNVT` or :class:`SystemVelocityVerlet`): one FP64
    potential energy per system (``(S,)``), the model's edge vectors from the engine's per-edge shifts, and one
    health flag per system (its occupied rows' positions and momenta and its energy finite)."""

    @staticmethod
    def _union_shape(builder, integrator) -> tuple[int, int]:
        if not isinstance(builder, SlotUnionGraphBuilder) or not isinstance(
                integrator, (SystemNoseHooverChainNVT, SystemVelocityVerlet)):
            raise TypeError("SlotWholeStepRunner needs a SlotUnionGraphBuilder and a per-system integrator "
                            "(SystemNoseHooverChainNVT or SystemVelocityVerlet)")
        if isinstance(integrator, SystemNoseHooverChainNVT) and integrator.layout != builder.layout:
            raise ValueError("builder and integrator must hold the same slots")
        return builder.layout.count, 0

    def _health(self) -> Tensor:
        """``(S,)`` float64: 1 where a system's positions, momenta or energy are not finite (device)."""
        finite = torch.isfinite(self.positions).all(dim=1) & torch.isfinite(self.momenta).all(dim=1)
        systems = self.builder.systems
        bad = segment_sums((~finite).to(torch.float64), systems.row_ptr[1:] - systems.row_ptr[:-1], systems.atoms) > 0
        return (bad | ~torch.isfinite(self.potential_energy)).to(torch.float64)

    @scoped_call
    def _evaluate(self, positions: Tensor, *, rebuild_candidates: bool, stress: bool = False):
        if stress:
            raise ValueError("SlotWholeStepRunner records no stress")
        graph, status = self.builder.build(positions, rebuild_candidates=rebuild_candidates)
        energies, forces = system_energy_derivative(self.adapter, self.builder.batch, graph)
        return forces, energies, status


def systems_self_check(adapter, systems, *, candidate_skin: float, device, repeats: int = 3, rattle: float = 0.1,
                       tol_multiplier: float = 3.0, absolute_floor: float = 1e-5, energy_floor: float = 1e-6) -> dict:
    """:func:`~fastmd.execution.replica_batch.union_self_check` for two different systems with cells of their
    own (``systems``: two ``(positions (n, 3), cell (3, 3), atomic_numbers (n,))``): both rattled by up to
    ``rattle`` Å, evaluated ``repeats`` times as one slot union (per-edge shifts) and ``repeats`` times each alone
    (the single-system path: a fixed-capacity build without systems and the engine-owned derivative). The same
    statistics and tolerances; and once more as the union with twice the cell as the model's ``cell`` argument,
    which a model that takes its edge vectors from the shifts does not read (``failed`` "cell" otherwise, within
    the same force tolerance)."""
    from fastmd.execution.replica_batch import _rattled, union_capacity, union_demand

    prepared = []
    for seed, (positions, cell, numbers) in enumerate(systems, start=1):
        cell = torch.as_tensor(validate_cell(cell), dtype=torch.float64, device=device)
        base = torch.as_tensor(positions, dtype=torch.float32, device=device)
        prepared.append((_rattled(base, rattle, seed), cell, torch.as_tensor(numbers, dtype=torch.int32,
                                                                             device=device).reshape(-1)))
    atoms = [int(p.shape[0]) for p, _, _ in prepared]
    layout = SlotLayout(tuple(atoms))
    cutoff = float(adapter.topology_spec().cutoff)
    pbc = torch.ones(3, dtype=torch.bool, device=device)

    def union(wrong_cell: bool = False) -> tuple[Tensor, list]:
        positions = torch.cat([p for p, _, _ in prepared])
        cells = torch.stack([c for _, c, _ in prepared])
        numbers = torch.cat([z for _, _, z in prepared])
        pairs = slot_union_demand(adapter, positions, layout=layout, cells=cells, atomic_numbers=numbers, atoms=atoms,
                                  candidate_skin=candidate_skin, device=device)
        builder = slot_union_builder(adapter, layout=layout, cells=cells, atomic_numbers=numbers, atoms=atoms,
                                     u_capacity=union_capacity(pairs), candidate_skin=candidate_skin, device=device)
        builder.refresh_candidates(positions)
        graph, _ = builder.build(positions, rebuild_candidates=False)
        energies, forces = system_energy_derivative(adapter, builder.batch, graph,
                                                    cell=2.0 * builder.batch.cell if wrong_cell else None)
        forces = forces.detach().to(torch.float64).cpu()
        return energies.cpu(), [forces[start: start + n] for start, n in zip(layout.starts, atoms)]

    def alone(index: int) -> tuple[Tensor, Tensor]:
        positions, cell, numbers = prepared[index]
        pairs, _ = union_demand(adapter, positions, replicas=1, cell=cell, atomic_numbers=numbers, pbc=pbc,
                                candidate_skin=candidate_skin, device=device)
        builder = FixedCapacityGraphBuilder(
            cell=cell, atomic_numbers=numbers, pbc=pbc, u_capacity=union_capacity(pairs), atom_graph_cutoff=cutoff,
            candidate_cutoff=cutoff + candidate_skin, device=device, padding=adapter.padding_rule(),
            prepare=adapter.prepare, **graph_builder_options(adapter))
        builder.refresh_candidates(positions)
        graph, _ = builder.build(positions, rebuild_candidates=False)
        n = int(positions.shape[0])
        out = engine_derivative(adapter)(builder.batch, graph, task="ef", n_real=n)
        return out["e"].cpu(), out["f"][0][:n].detach().to(torch.float64).cpu()

    unions = [union() for _ in range(repeats)]
    singles = [[alone(index) for _ in range(repeats)] for index in range(2)]
    wrong_cell = union(wrong_cell=True)

    def largest(pairs) -> float:
        return max((float((a - b).abs().max()) for a, b in pairs), default=0.0)

    single_floor = largest((singles[s][i][1], singles[s][j][1]) for s in range(2)
                           for i in range(repeats) for j in range(i + 1, repeats))
    union_floor = largest((unions[i][1][s], unions[j][1][s]) for s in range(2)
                          for i in range(repeats) for j in range(i + 1, repeats))
    cross = largest((unions[i][1][s], singles[s][j][1]) for s in range(2) for i in range(repeats)
                    for j in range(repeats))
    energy_cross = max(abs(float(unions[i][0][s] - singles[s][j][0][0])) / atoms[s] for s in range(2)
                       for i in range(repeats) for j in range(repeats))
    energy_single = max((abs(float(singles[s][i][0][0] - singles[s][j][0][0])) / atoms[s] for s in range(2)
                         for i in range(repeats) for j in range(i + 1, repeats)), default=0.0)
    energy_union = max((abs(float(unions[i][0][s] - unions[j][0][s])) / atoms[s] for s in range(2)
                        for i in range(repeats) for j in range(i + 1, repeats)), default=0.0)
    epsilon = float(np.finfo(np.float64 if adapter.topology_spec().model_dtype == "float64" else np.float32).eps)
    resolution = max(4.0 * epsilon * abs(float(singles[s][i][0][0])) / atoms[s] for s in range(2)
                     for i in range(repeats))
    tolerance = max(absolute_floor, tol_multiplier * max(single_floor, union_floor))
    energy_tolerance = max(energy_floor, resolution, tol_multiplier * max(energy_single, energy_union))
    cell_forces = largest((wrong_cell[1][s], unions[0][1][s]) for s in range(2))
    cell_energies = max(abs(float(wrong_cell[0][s] - unions[0][0][s])) / atoms[s] for s in range(2))
    failed = ("forces" if cross > tolerance else "energies" if energy_cross > energy_tolerance
              else "cell" if cell_forces > tolerance or cell_energies > energy_tolerance else None)
    return {"passed": failed is None, "failed": failed, "max_force_difference": cross, "tolerance": tolerance,
            "single_floor": single_floor, "union_floor": union_floor, "max_energy_difference_per_atom": energy_cross,
            "energy_tolerance_per_atom": energy_tolerance, "energy_single_floor": energy_single,
            "energy_union_floor": energy_union, "energy_resolution_per_atom": resolution,
            "cell_argument_force_difference": cell_forces, "cell_argument_energy_difference_per_atom": cell_energies,
            "systems": 2, "atoms": atoms, "repeats": int(repeats),
            "rattle": float(rattle), "evaluated": "two of its systems"}


def slot_union_memory_runs(adapter, systems, *, candidate_skin: float, device, step=None,
                           headroom: float = UNION_GROWTH_HEADROOM, ceilings=None):
    """``({kind: run(k)}, pairs)`` for ``MeasuredMemoryModel.measure`` on sub-unions of ``systems`` (one
    ``(positions (n, 3), cell (3, 3), atomic_numbers (n,))`` each, in slot order): ``run(k)`` evaluates the slot
    union of the first ``k`` systems (each slot at its ceiling of ``ceilings``, default its atom count, vacant rows
    included) at the capacity one growth would give it
    (:func:`~fastmd.execution.replica_batch.union_capacity` with ``headroom``), eager ("eager") or captured in a
    CUDA graph after a warmup on the capture stream ("capture"), and returns the (edges, 0) capacity it held;
    ``pairs(k)`` is that union's pair demand. Atoms and edges grow together over the sub-unions, so a fit over
    edges holds the per-atom costs too. Each run makes its own builder; a captured graph is released before
    the run returns."""
    from fastmd.engine._graph_capture import capture_graph
    from fastmd.execution.replica_batch import union_capacity

    stream = torch.cuda.Stream(device)
    demands: dict[int, int] = {}

    sizes = [int(np.asarray(numbers).shape[0]) for _, _, numbers in systems]
    ceilings = sizes if ceilings is None else [int(c) for c in ceilings]

    def tables(k: int):
        layout, atoms = SlotLayout(tuple(ceilings[:k])), sizes[:k]
        positions = np.zeros((layout.rows, 3), dtype=np.float64)
        numbers = np.zeros(layout.rows, dtype=np.int32)
        for start, ceiling, (coordinates, _, z) in zip(layout.starts, layout.ceilings, systems[:k]):
            z = np.asarray(z)
            numbers[start:start + ceiling] = z[0]  # a vacant row: its system's first element, at the origin
            numbers[start:start + len(z)] = z
            positions[start:start + len(z)] = np.asarray(coordinates, dtype=np.float64)
        cells = torch.as_tensor(np.stack([np.asarray(c, dtype=np.float64) for _, c, _ in systems[:k]]),
                                dtype=torch.float64, device=device)
        return (layout, atoms, torch.as_tensor(positions, dtype=torch.float32, device=device), cells,
                torch.as_tensor(numbers, device=device))

    def pairs(k: int) -> int:
        if k not in demands:
            layout, atoms, positions, cells, numbers = tables(k)
            demands[k] = slot_union_demand(adapter, positions, layout=layout, cells=cells, atomic_numbers=numbers,
                                           atoms=atoms, candidate_skin=candidate_skin, device=device)
        return demands[k]

    def evaluator(k: int):
        layout, atoms, positions, cells, numbers = tables(k)
        builder = slot_union_builder(adapter, layout=layout, cells=cells, atomic_numbers=numbers, atoms=atoms,
                                     u_capacity=union_capacity(pairs(k), step, headroom=headroom),
                                     candidate_skin=candidate_skin, device=device)
        builder.refresh_candidates(positions)

        def evaluate() -> None:
            graph, _ = builder.build(positions, rebuild_candidates=False)
            system_energy_derivative(adapter, builder.batch, graph)

        return builder, evaluate

    def eager(k: int) -> tuple[int, int]:
        builder, evaluate = evaluator(k)
        evaluate()
        return builder.e_capacity, 0

    def capture(k: int) -> tuple[int, int]:
        builder, evaluate = evaluator(k)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            evaluate()
        torch.cuda.current_stream(device).wait_stream(stream)
        # As the runner's capture: the warmup's blocks go back before the capture allocates its pool.
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        graph = torch.cuda.CUDAGraph()
        with capture_graph(graph, stream=stream):
            evaluate()
        del graph  # every point starts from the same state
        return builder.e_capacity, 0

    return {"eager": eager, "capture": capture}, pairs


def slot_union_persistent_bytes(builder: SlotUnionGraphBuilder) -> int:
    """The device bytes per row a slot union keeps outside its graph pool, estimated: ``builder``'s buffers (the
    candidate list above all) per row, plus 32 float64 per row for the runner's and the integrator's per-atom
    state (positions, momenta, forces, their backups and mirrors)."""
    seen, total = set(), 0
    for value in vars(builder).values():
        if torch.is_tensor(value) and value.device.type == "cuda" and value.data_ptr() not in seen:
            seen.add(value.data_ptr())
            total += value.untyped_storage().nbytes()
    return -(-total // builder.n_real) + 32 * 8
