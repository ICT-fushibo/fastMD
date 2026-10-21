"""Homogeneous MD replicas of a model the engine differentiates, in one whole-step CUDA graph.

N replicas of one system (the same atoms and cell; each with its own
positions, momenta and thermostat) run as one disjoint union: rows
``[s * n, (s + 1) * n)`` are replica ``s``. Three batch-aware parts let the
single-system :class:`~fastmd.execution.whole_step_cuda_graph.WholeStepGraphRunner`
run the union with its window transaction, rollback, fallback and candidate
refresh unchanged:

* :class:`ReplicaUnionGraphBuilder`: the fixed-capacity builder over the
  union, edges only or with the line graph of a model with triplets. Its
  candidate list comes from a batched cell list with one batch index per
  replica, so no edge (and no triplet) joins two replicas; one edge capacity
  holds the union's edges (the sum over the replicas), and one triplet
  capacity its triplets, padded by the model's rule (a line graph: far
  self-loops, padded triplets on padded pairs). It publishes the union's
  systems on its batch (:class:`~fastmd.adapters.protocol.SystemSegments`).
* the integrators, one thermostat per replica:
  :class:`ReplicaNoseHooverChainNVT` (ASE's Nose-Hoover chain, its own
  ``eta``, ``p_eta`` and kinetic energy per replica: the single-system Triton
  kernels of :mod:`~fastmd.execution.gpu_nhc`, one program per replica, bit
  for bit a single-system integrator fed the same forces) and
  :class:`ReplicaVelocityVerlet` (NVE and Berendsen NVT: the single-system
  :class:`~fastmd.execution.gpu_integrator.GPUIntegrator` on ``(N, n, 3)``
  views, so the kinetic energy, the Berendsen scale, the centre-of-mass
  correction and the degrees of freedom are each replica's own).
* :class:`ReplicaWholeStepRunner`: the runner with one FP64 energy per
  replica (the sum of its rows' energies, or the replica's entry of a
  whole-system tap that returns one energy per system) and one backward pass
  for every replica's forces. The model sees one graph; its per-atom energies
  must depend on the atom's own graph component only (``"union"``), or the
  model keeps the systems apart itself (``"segments"``: it pools over each
  system of ``GraphBatch.systems``), which a model declares
  (:class:`~fastmd.adapters.protocol.SystemBatching`) and
  :func:`union_self_check` tests. Each window also leaves a health flag per
  replica, read with the window's other flags.

The union's first capture holds the model's own capacity tiers; a growth adds a
relative headroom over the overflowing demand (:func:`union_capacity`), so a
union whose pair count drifts is captured again once, not at every step of the
drift. A line graph's triplets (:func:`union_triplet_capacity`) also hold at
least the minimum triplet padding of each replica alone: a triplet count can
grow from almost nothing (bonds that come within the short triplet cutoff as
the system heats), which no relative headroom follows. Its memory is measured
on unions of the same system (:func:`union_memory_runs`).
``fastmd.interfaces.replica_md`` drives these parts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fastmd.adapters import as_adapter
from fastmd.adapters.protocol import EnergyTap, NegotiationError, SystemEnergyTap, SystemSegments
from fastmd.engine.energy_derivative import edges_only_builder_options, engine_derivative
from fastmd.engine._graph_capture import capture_graph
from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder
from fastmd.execution.gpu_integrator import GPUIntegrator, StatefulIntegrator
from fastmd.execution.gpu_nhc import _HAS_TRITON, _TRITON_MAX_ELEMS, _TargetTemperature
from fastmd.execution.whole_step_cuda_graph import WholeStepGraphRunner
from fastmd.engine.entry_config import scoped_call

if _HAS_TRITON:
    import triton
    import triton.language as tl

    from fastmd.graph.fixed_capacity_builder import allocate_cell_list, estimate_batch_cell_list_sizes
    from fastmd.execution.gpu_nhc import _nhc_post_force_kernel, _nhc_pre_force_kernel

    @triton.jit
    def _nhc_pre_force_replicas_kernel(q_ptr, p_ptr, f_ptr, m_ptr, eta_ptr, pe_ptr, c_ptr, n_elem, stride,
                                       BLOCK: tl.constexpr, M: tl.constexpr, TLOOP: tl.constexpr,
                                       F32_KICK: tl.constexpr, FAST: tl.constexpr):
        # The single-system pre-force half on replica program_id(0): its rows and its chain (the masses are
        # shared). The rows are ``stride`` apart, a multiple of 16 elements; rows at an odd offset (3n odd) gave
        # 1-ulp differences (presumably another reduction order of sum(p**2/m)), these are bit for bit the single
        # system fed its own buffers (an empirical property of this Triton, for the sizes the tests cover).
        replica = tl.program_id(0).to(tl.int64)
        rows = replica * stride
        chain = replica * M
        _nhc_pre_force_kernel(q_ptr + rows, p_ptr + rows, f_ptr + rows, m_ptr, eta_ptr + chain, pe_ptr + chain,
                              c_ptr, n_elem, BLOCK, M, TLOOP, F32_KICK, FAST)

    @triton.jit
    def _nhc_post_force_replicas_kernel(p_ptr, fin_ptr, fcache_ptr, m_ptr, eta_ptr, pe_ptr, c_ptr, ein_ptr,
                                        eout_ptr, n_elem, stride, BLOCK: tl.constexpr, M: tl.constexpr,
                                        TLOOP: tl.constexpr, F32_KICK: tl.constexpr, HAS_E: tl.constexpr,
                                        FAST: tl.constexpr):
        # The single-system post-force half on replica program_id(0), with its energy.
        replica = tl.program_id(0).to(tl.int64)
        rows = replica * stride
        chain = replica * M
        _nhc_post_force_kernel(p_ptr + rows, fin_ptr + rows, fcache_ptr + rows, m_ptr, eta_ptr + chain,
                               pe_ptr + chain, c_ptr, ein_ptr + replica, eout_ptr + replica, n_elem, BLOCK, M,
                               TLOOP, F32_KICK, HAS_E, FAST)


# The relative pair headroom a capacity growth adds over the overflowing demand: a heating system's pair count
# drifts up by a few percent (135-atom CsSnI3 at 500 K: +4 to +8 % within 1,000 steps of 0.25 fs with a 6 Å
# cutoff, under 1 % with 5 Å), the same in every replica, so the absolute padding of the model's tiers does not
# cover a union's drift, and growing by the tiers alone captures the whole union again several times as the
# count drifts (MACE-MPA-0 at 500 K: twice in 1,000 steps against once with this headroom; dozens of times for a
# model without a thermal factor). The first capture keeps the model's tiers: a union whose pair count does not
# drift pays no padding for it.
UNION_GROWTH_HEADROOM = 0.08
# The relative triplet headroom of a growth (a line graph): a triplet count drifts far more than the pair count
# (the triplets of a centre grow with the square of its bonds within the shorter triplet cutoff; 135-atom CsSnI3
# under CHGNet at 500 K, 3 Å: from 0 in the start frame to 52-76 per replica within 1,000 steps of 0.25 fs, a
# mean of 22-38), and every padded triplet runs the model's angle networks. A model declares its own growth
# headrooms (CapacityStep.u_union_headroom, t_union_headroom) where its unions drift by more.
UNION_TRIPLET_GROWTH_HEADROOM = 0.25
# The default union: replicas up to this many rows. A union of MACE-MPA-0 on a 135-atom system saturated a GB200
# at about 256 replicas (34,560 rows; 2,048 replicas ran 1-3 % faster); more replicas run as several unions of
# balanced sizes that share one graph pool.
DEFAULT_UNION_ROWS = 1 << 16
# The union sizes the memory model measures, as fractions of the union it predicts (atoms and edges grow
# together): predicted at the union's own capacity, as measured, a union of 4 or more replicas stays within
# twice the largest measured one.
UNION_PROBE_FRACTIONS = (0.25, 0.5, 0.75)


class ReplicaUnionGraphBuilder(FixedCapacityGraphBuilder):
    """The fixed-capacity builder over ``replicas`` copies of one system (atoms and cell): edges only, or with
    the line graph of a model with triplets (padded with far self-loops).

    Rows ``[s * n, (s + 1) * n)`` are replica ``s``. The candidate list is a
    batched cell list with one batch index per replica, so the exact
    re-filter of each build (the single-system kernels, over all rows) never
    joins two replicas, and a triplet (two bonds of one centre) never either;
    the edge and triplet capacities, the overflow, the skin test and the
    candidate refresh are the union's. ``atomic_numbers`` are one replica's.
    The batch's ``systems`` (:class:`~fastmd.adapters.protocol.SystemSegments`)
    is set before the adapter's ``prepare`` sees the batch.
    """

    def __init__(self, *, replicas: int, cell: Tensor, atomic_numbers: Tensor, pbc: Tensor | None = None,
                 **kwargs) -> None:
        if int(replicas) < 1:
            raise ValueError("replicas must be >= 1")
        for name in ("ghosts", "candidates_from"):
            if kwargs.get(name) is not None:
                raise ValueError(f"a replica union builder has no {name}")
        if kwargs.get("minimum_image"):
            raise ValueError("a replica union builder has no minimum_image")
        padding = kwargs.get("padding")
        if padding is None or padding.kind == "sink_atoms":
            raise ValueError("a replica union builder pads with self-loops (far_self_loops or exact_rc_loops): sink "
                             "rows would sit after the last replica's rows")
        self.replicas = int(replicas)
        numbers = atomic_numbers.reshape(-1)
        self.replica_atoms = int(numbers.shape[0])
        device = torch.device(kwargs.get("device", "cuda"))
        rows = self.replicas * self.replica_atoms
        system_of_row = torch.arange(self.replicas, dtype=torch.int32, device=device).repeat_interleave(
            self.replica_atoms)
        self.systems = SystemSegments(
            count=self.replicas,
            row_ptr=torch.arange(0, rows + 1, self.replica_atoms, dtype=torch.int64, device=device),
            system_of_row=system_of_row)
        prepare = kwargs.pop("prepare", None)
        systems = self.systems

        def prepare_systems(batch, prepare=prepare):
            batch.systems = systems
            return batch if prepare is None else prepare(batch)

        kwargs["prepare"] = prepare_systems
        super().__init__(cell=cell, atomic_numbers=numbers.repeat(self.replicas), pbc=pbc, **kwargs)
        # One cell-list batch per replica (_build_candidates reads these): no candidate crosses replicas.
        self.cells = self.cell.unsqueeze(0).expand(self.replicas, 3, 3).contiguous()
        self.pbc = self.pbc.expand(self.replicas, 3).contiguous()
        self.batch_idx = system_of_row
        max_total_cells, neighbor_radius = estimate_batch_cell_list_sizes(self.cells, self.pbc,
                                                                          self.candidate_cutoff)
        self.cell_list_cache = allocate_cell_list(self.n_real, max_total_cells, neighbor_radius, self.device)


def graph_builder_options(adapter, t_capacity: int = 0) -> dict:
    """The fixed-capacity builder options of ``adapter``'s graph beyond its capacities and cutoffs: edges only
    (its edges-only options), or the line graph of a model with triplets at ``t_capacity`` triplets, with the
    positions (the engine's leaf) published, in float64 for a float64 model."""
    topology = adapter.topology_spec()
    if topology.triplet_cutoff is None:
        return dict(t_capacity=0, line_graph_cutoff=None, **edges_only_builder_options(adapter))
    return dict(t_capacity=int(t_capacity), line_graph_cutoff=float(topology.triplet_cutoff), publish_positions=True,
                **({"positions_dtype": torch.float64} if topology.model_dtype == "float64" else {}))


def union_builder(adapter, *, replicas: int, cell: Tensor, atomic_numbers: Tensor, pbc: Tensor, u_capacity: int,
                  candidate_skin: float, device, t_capacity: int = 0) -> ReplicaUnionGraphBuilder:
    """The union builder of ``replicas`` copies for ``adapter`` at ``u_capacity`` undirected pairs (and
    ``t_capacity`` triplets for a model with a line graph): its padding rule, ``prepare`` and builder options
    (:func:`graph_builder_options`); candidates within cutoff + ``candidate_skin``."""
    cutoff = float(adapter.topology_spec().cutoff)
    return ReplicaUnionGraphBuilder(
        replicas=replicas, cell=cell, atomic_numbers=atomic_numbers, pbc=pbc, u_capacity=int(u_capacity),
        atom_graph_cutoff=cutoff, candidate_cutoff=cutoff + candidate_skin, device=device,
        padding=adapter.padding_rule(), prepare=adapter.prepare, **graph_builder_options(adapter, t_capacity))


def union_demand(adapter, positions: Tensor, *, replicas: int, cell: Tensor, atomic_numbers: Tensor, pbc: Tensor,
                 candidate_skin: float, device) -> tuple[int, int]:
    """The union's undirected pair and triplet counts at ``positions`` (``(N * n, 3)``; triplets 0 without a
    line graph): one eager build with capacities estimated from the density, again with the counts found if
    those were too small."""
    rows = int(positions.shape[0])
    topology = adapter.topology_spec()
    cutoff = float(topology.cutoff)
    density = (rows / replicas) / max(float(torch.linalg.det(cell.double()).abs()), 1e-9)
    guess = int(4.0 / 3.0 * math.pi * cutoff ** 3 * density * rows / 2 * 1.5) + 1024
    t_guess = 0 if topology.triplet_cutoff is None else guess
    while True:
        builder = union_builder(adapter, replicas=replicas, cell=cell, atomic_numbers=atomic_numbers, pbc=pbc,
                                u_capacity=guess, t_capacity=t_guess, candidate_skin=candidate_skin, device=device)
        builder.refresh_candidates(positions)
        builder.build(positions, rebuild_candidates=False)
        edges, triplets = int(builder.edge_offsets[-1].item()), int(builder.pair_offsets[-1].item())
        del builder
        if edges <= 2 * guess and triplets <= t_guess:
            return -(-edges // 2), triplets
        guess, t_guess = max(guess, -(-edges // 2)), max(t_guess, triplets) if t_guess else 0


def union_pair_demand(adapter, positions: Tensor, *, replicas: int, cell: Tensor, atomic_numbers: Tensor,
                      pbc: Tensor, candidate_skin: float, device) -> int:
    """The union's undirected pair count at ``positions`` (``(N * n, 3)``): :func:`union_demand`'s pairs."""
    return union_demand(adapter, positions, replicas=replicas, cell=cell, atomic_numbers=atomic_numbers, pbc=pbc,
                        candidate_skin=candidate_skin, device=device)[0]


def union_capacity(pairs: int, step=None, *, headroom: float = 0.0, line_graph: bool = False) -> int:
    """The union's pair capacity for a demand of ``pairs``: the larger of ``1 + headroom`` and the model's
    thermal factor times the demand, plus the model's minimum padding, rounded up to its pair step
    (:class:`~fastmd.adapters.protocol.CapacityStep`; the engine's 512 and 256 without one); with a
    ``line_graph`` one step more where that holds no pair above the demand (its padded triplets need a padded
    pair, as the single-system tiers keep). A first capture takes ``headroom`` 0 (the model's tiers), a growth
    the union's growth headroom (:func:`union_headrooms`)."""
    if not 0.0 <= float(headroom) <= 1.0:
        raise ValueError("the union's pair headroom must be in [0, 1]")
    u_step, min_pad, factor = (step.u_step, step.min_pad_u, step.u_thermal_factor) if step is not None else (512, 256,
                                                                                                           1.0)
    factor = max(float(factor), 1.0 + float(headroom))
    capacity = -(-(math.ceil(int(pairs) * factor) + int(min_pad)) // int(u_step)) * int(u_step)
    return capacity + int(u_step) if line_graph and capacity <= int(pairs) else capacity


def union_headrooms(step=None, pairs: float | None = None, triplets: float | None = None) -> tuple[float, float]:
    """The (pair, triplet) headrooms of a union's capacity growths: ``pairs`` and ``triplets`` where given, else
    the model's declaration (:class:`~fastmd.adapters.protocol.CapacityStep` ``u_union_headroom`` and
    ``t_union_headroom``), else ``UNION_GROWTH_HEADROOM`` and ``UNION_TRIPLET_GROWTH_HEADROOM``."""
    declared = (None, None) if step is None else (step.u_union_headroom, step.t_union_headroom)
    return tuple(float(next(value for value in (given, model, default) if value is not None))
                 for given, model, default in zip((pairs, triplets), declared,
                                                  (UNION_GROWTH_HEADROOM, UNION_TRIPLET_GROWTH_HEADROOM)))


def union_triplet_capacity(triplets: int, step=None, *, headroom: float = 0.0, replicas: int = 1) -> int:
    """:func:`union_capacity` of a line graph's triplets: the larger of ``1 + headroom`` and the model's
    triplet thermal factor times the demand, plus its minimum triplet padding, rounded up to its triplet step
    (the engine's 8,192 and 4,096 where the model declares no triplet tiers), and at least that minimum
    padding for each of the union's ``replicas`` (each replica alone would hold it) and one step."""
    if not 0.0 <= float(headroom) <= 1.0:
        raise ValueError("the union's triplet headroom must be in [0, 1]")
    declared = step is not None and step.t_step is not None
    t_step, min_pad, factor = (step.t_step, step.min_pad_t, step.t_thermal_factor) if declared else (8192, 4096, 1.0)
    factor = max(float(factor), 1.0 + float(headroom))
    capacity = -(-(math.ceil(int(triplets) * factor) + int(min_pad)) // int(t_step)) * int(t_step)
    return max(capacity, -(-int(replicas) * int(min_pad) // int(t_step)) * int(t_step), int(t_step))


def replica_stride(n_atoms: int) -> int:
    """Elements between two replicas' rows in the integrator's per-atom buffers: ``3 n`` rounded up to a
    multiple of 16."""
    return -(-3 * int(n_atoms) // 16) * 16


@dataclass
class ReplicaNHCState:
    """Views into one flat float64 buffer: ``[q | p | F | eta | p_eta | E]``. q, p and F are
    ``(N, stride)`` rows (:func:`replica_stride`; replica ``s``'s ``3 n`` coordinates first), eta and
    p_eta ``(N, tchain)``, E ``(N,)``."""

    flat: Tensor
    q: Tensor
    p: Tensor
    f: Tensor
    eta: Tensor
    p_eta: Tensor
    potential_energy: Tensor
    n_atoms: int

    def rows(self, field: Tensor) -> Tensor:
        """``(N, n, 3)`` view of ``q``, ``p`` or ``f`` (the padding columns left out)."""
        return field[:, :3 * self.n_atoms].view(field.shape[0], self.n_atoms, 3)

    @staticmethod
    def allocate(replicas: int, n_atoms: int, tchain: int, device) -> "ReplicaNHCState":
        stride = replica_stride(n_atoms)
        size = replicas * stride
        flat = torch.zeros(3 * size + 2 * replicas * tchain + replicas, dtype=torch.float64, device=device)
        q, p, f = (flat[k * size: (k + 1) * size].view(replicas, stride) for k in range(3))
        offset = 3 * size
        eta = flat[offset: offset + replicas * tchain].view(replicas, tchain)
        offset += replicas * tchain
        p_eta = flat[offset: offset + replicas * tchain].view(replicas, tchain)
        offset += replicas * tchain
        return ReplicaNHCState(flat, q, p, f, eta, p_eta, flat[offset: offset + replicas], int(n_atoms))


class ReplicaNoseHooverChainNVT(_TargetTemperature, StatefulIntegrator):
    """ASE's ``NoseHooverChainNVT`` for ``replicas`` copies of one system, each with its own chain.

    ``masses``: ``(n,)`` amu of one replica; ``positions`` and ``momenta``:
    ``(N, n, 3)`` (or ``(N * n, 3)``). The degrees of freedom, ``Q`` and the
    temperature are per replica (``3 n``), as for a single-system
    :class:`~fastmd.execution.gpu_nhc.GPUNoseHooverChainNVT` of ``n`` atoms;
    each half step is that integrator's Triton kernel, one program per replica
    on rows padded to multiples of 16 elements (:func:`replica_stride`); each
    replica integrates bit for bit like a single-system integrator fed the same
    forces (an empirical property of this Triton, tested). ``positions``,
    ``momenta`` and ``forces`` are ``(N * n, 3)`` tensors: copies, or views of
    the state when ``3 n`` is a multiple of 16 (never write into them);
    :meth:`step_pre_force` returns such a tensor of the new positions and
    :meth:`step_post_force` takes ``(N * n, 3)`` forces and one energy per
    replica. The step halves, :meth:`backup`
    and :meth:`restore` are capture-safe; :meth:`temperature`,
    :meth:`kinetic_energy` and :meth:`conserved_energy` return one value per
    replica. No restart files (:meth:`state_dict` raises).
    """

    ensemble = "nvt_nhc"
    comm = None

    def __init__(self, masses, positions, momenta, *, replicas: int, temperature_K: float, timestep: float,
                 tdamp: float, tchain: int = 3, tloop: int = 1, forces=None, potential_energy=None,
                 device="cuda", force_dtype: torch.dtype = torch.float64, numpy_kick_semantics: bool = True) -> None:
        if not _HAS_TRITON:
            raise RuntimeError("the replica Nose-Hoover chain needs Triton")
        if tchain < 1 or tloop < 1 or tdamp <= 0 or timestep < 0:
            raise ValueError("tchain and tloop must be >= 1, tdamp positive and timestep non-negative")
        self.device = torch.device(device)
        masses_np = np.asarray(masses.detach().cpu().numpy() if torch.is_tensor(masses) else masses,
                               dtype=np.float64).reshape(-1)
        self.replicas = int(replicas)
        self.n_atoms = n = int(masses_np.shape[0])
        if 3 * n > _TRITON_MAX_ELEMS:
            raise ValueError(f"the replica Nose-Hoover chain supports 3n <= {_TRITON_MAX_ELEMS} per replica")
        self.stride = replica_stride(n)
        self.tchain, self.tloop = int(tchain), int(tloop)
        self.temperature_K = float(temperature_K)
        self.dt, self.tdamp = float(timestep), float(tdamp)
        self.force_dtype = force_dtype
        self.f32_kick = bool(numpy_kick_semantics and force_dtype == torch.float32)
        self.generation = 0
        self._init_constants(n)  # one replica's DOF and Q
        self.masses = torch.tensor(masses_np, dtype=torch.float64, device=self.device).reshape(n, 1).contiguous()
        self.state = ReplicaNHCState.allocate(self.replicas, n, self.tchain, self.device)
        self._backup = ReplicaNHCState.allocate(self.replicas, n, self.tchain, self.device)
        # The model's forces, copied into rows of the state's stride before the post-force half.
        self._forces_in = torch.zeros((self.replicas, self.stride), dtype=force_dtype, device=self.device)
        self.set_state(positions=positions, momenta=momenta, forces=forces, potential_energy=potential_energy)
        # The launch parameters of GPUNoseHooverChainNVT for n atoms.
        self._block = max(16, triton.next_power_of_2(3 * n))
        self._num_warps = (1 if self._block <= 64 else 2 if self._block <= 128 else 4 if self._block <= 2048
                           else 8 if self._block <= 8192 else 16)

    # ------------------------------------------------------------------ state
    @property
    def n_owned(self) -> int:
        return self.replicas * self.n_atoms

    def _flat_rows(self, field: Tensor) -> Tensor:
        return self.state.rows(field).reshape(self.replicas * self.n_atoms, 3)

    @property
    def positions(self) -> Tensor:
        return self._flat_rows(self.state.q)

    @property
    def momenta(self) -> Tensor:
        return self._flat_rows(self.state.p)

    @property
    def forces(self) -> Tensor:
        return self._flat_rows(self.state.f)

    @property
    def potential_energy(self) -> Tensor:
        return self.state.potential_energy

    @property
    def eta(self) -> Tensor:
        return self.state.eta

    @property
    def p_eta(self) -> Tensor:
        return self.state.p_eta

    def _as_f64(self, x) -> Tensor:
        if torch.is_tensor(x):
            return x.detach().to(device=self.device, dtype=torch.float64)
        return torch.as_tensor(np.asarray(x, dtype=np.float64), device=self.device)

    @torch.no_grad()
    def set_state(self, *, positions=None, momenta=None, forces=None, eta=None, p_eta=None,
                  potential_energy=None, replica: int | None = None) -> None:
        """(Re)load the state of every replica (``(N, n, 3)`` or ``(N * n, 3)``; eta/p_eta ``(N, tchain)``,
        energies ``(N,)``), or of one ``replica`` (``(n, 3)``, ``(tchain,)``, a scalar). Not in a capture."""
        st = self.state
        which = slice(None) if replica is None else replica
        for field, value in ((st.q, positions), (st.p, momenta), (st.f, forces)):
            if value is not None:
                target = st.rows(field)[which]
                target.copy_(self._as_f64(value).reshape(target.shape))
        for name, value in (("eta", eta), ("p_eta", p_eta), ("potential_energy", potential_energy)):
            if value is not None:
                target = getattr(st, name)[which]
                target.copy_(self._as_f64(value).reshape(target.shape))

    @torch.no_grad()
    def set_forces(self, forces, energy=None, state: ReplicaNHCState | None = None) -> None:
        """F(q) ``(N * n, 3)`` (and one E(q) per replica) for the current positions (capture-safe for device
        tensors)."""
        st = self.state if state is None else state
        target = st.rows(st.f)
        target.copy_(forces.reshape(target.shape) if torch.is_tensor(forces)
                     else self._as_f64(forces).reshape(target.shape))
        if energy is not None:
            st.potential_energy.copy_(energy.reshape(self.replicas) if torch.is_tensor(energy)
                                      else self._as_f64(energy).reshape(self.replicas))

    # -------------------------------------------------------------- step halves
    def step_pre_force(self, state: ReplicaNHCState | None = None) -> Tensor:
        """Every replica's ASE ``step()`` first half; returns a float64 ``(N * n, 3)`` copy of the new
        positions."""
        st = self.state if state is None else state
        _nhc_pre_force_replicas_kernel[(self.replicas,)](
            st.q, st.p, st.f, self.masses, st.eta, st.p_eta, self.consts, 3 * self.n_atoms, self.stride,
            BLOCK=self._block, M=self.tchain, TLOOP=self.tloop, F32_KICK=self.f32_kick, FAST=False,
            num_warps=self._num_warps, enable_fp_fusion=False,
        )
        return st.rows(st.q).reshape(self.replicas * self.n_atoms, 3)

    def step_post_force(self, forces: Tensor, energy: Tensor | None = None,
                        state: ReplicaNHCState | None = None) -> None:
        """Every replica's ASE ``step()`` second half with F(q_{n+1}) ``(N * n, 3)`` (the model's dtype)
        and, optionally, one energy per replica (``(N,)``) stored as its potential energy."""
        st = self.state if state is None else state
        forces_in = self._forces_in if forces.dtype == self._forces_in.dtype else torch.zeros(
            self._forces_in.shape, dtype=forces.dtype, device=self.device)
        forces_in[:, :3 * self.n_atoms].copy_(forces.reshape(self.replicas, 3 * self.n_atoms))
        has_e = energy is not None
        e = energy.reshape(self.replicas) if has_e else st.potential_energy
        _nhc_post_force_replicas_kernel[(self.replicas,)](
            st.p, forces_in, st.f, self.masses, st.eta, st.p_eta, self.consts, e, st.potential_energy,
            3 * self.n_atoms, self.stride, BLOCK=self._block, M=self.tchain, TLOOP=self.tloop,
            F32_KICK=self.f32_kick, HAS_E=has_e, FAST=False, num_warps=self._num_warps, enable_fp_fusion=False,
        )

    # -------------------------------------------------------------- observables
    def kinetic_energy(self, state: ReplicaNHCState | None = None) -> Tensor:
        """``(N,)`` float64: each replica's ``0.5 * vdot(p, p / m)``."""
        st = self.state if state is None else state
        p = st.rows(st.p)
        return 0.5 * (p * (p / self.masses)).sum(dim=(1, 2))

    def temperature(self, state: ReplicaNHCState | None = None) -> Tensor:
        """``(N,)``: each replica's ``Atoms.get_temperature()``."""
        from ase import units

        return 2 * self.kinetic_energy(state) / (3 * self.n_atoms * units.kB)

    def thermostat_energy(self, state: ReplicaNHCState | None = None) -> Tensor:
        st = self.state if state is None else state
        return (self.ndof_kT * st.eta[:, 0] + self.kT * st.eta[:, 1:].sum(dim=1)
                + (0.5 * (st.p_eta * st.p_eta) / self.Q).sum(dim=1))

    def conserved_energy(self, state: ReplicaNHCState | None = None) -> Tensor:
        """``(N,)``: each replica's ``NoseHooverChainNVT.get_conserved_energy``."""
        st = self.state if state is None else state
        return st.potential_energy + self.kinetic_energy(st) + self.thermostat_energy(st)

    # ------------------------------------------------------------ transactions
    def backup(self, state: ReplicaNHCState | None = None) -> None:
        self._backup.flat.copy_((self.state if state is None else state).flat)

    def restore(self, state: ReplicaNHCState | None = None) -> None:
        (self.state if state is None else state).flat.copy_(self._backup.flat)

    def snapshot(self) -> tuple:
        return self.state.flat.clone(), self._backup.flat.clone()

    def load_snapshot(self, snapshot: tuple) -> None:
        state, backup = snapshot
        self.state.flat.copy_(state)
        self._backup.flat.copy_(backup)

    def new_state(self) -> ReplicaNHCState:
        return ReplicaNHCState.allocate(self.replicas, self.n_atoms, self.tchain, self.device)

    def state_dict(self) -> dict:
        raise NotImplementedError("restart files of replica integrators are not implemented")

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError("restart files of replica integrators are not implemented")


class ReplicaVelocityVerlet(GPUIntegrator):
    """:class:`~fastmd.execution.gpu_integrator.GPUIntegrator` (NVE, or Berendsen NVT with
    ``ensemble="nvt"``) for ``replicas`` copies of one system of ``masses`` (one replica's).

    The whole-step runner hands it the union's flat ``(N * n, 3)`` buffers; the
    step halves see them as ``(N, n, 3)``, so the kinetic energy and the
    Berendsen scale are each replica's (``3 n`` degrees of freedom, as a
    single system of ``n`` atoms), and ``fix_com`` removes each replica's own
    centre-of-mass momentum. ``temperature`` and ``kinetic_energy`` take
    ``(N, n, 3)`` or ``(N * n, 3)`` momenta and return one value per replica.
    """

    def __init__(self, masses: Tensor, dt_fs: float, *, replicas: int, ensemble: str = "nvt",
                 temperature: float = 300.0, taut_fs: float = 100.0, fix_com: bool | None = None) -> None:
        masses = masses.reshape(-1)
        super().__init__(masses, dt_fs, ensemble=ensemble, temperature=temperature, taut_fs=taut_fs,
                         fix_com=fix_com, degrees_of_freedom=3 * int(masses.numel()))
        self.replicas = int(replicas)
        self.n_atoms = int(masses.numel())

    def _replica_view(self, tensor: Tensor) -> Tensor:
        return tensor.view(self.replicas, self.n_atoms, 3)

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        return super().kinetic_energy(momenta if momenta.ndim == 3 else self._replica_view(momenta))

    def step_pre_force(self, positions: Tensor, momenta: Tensor, forces: Tensor,
                       masses: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Each replica's Berendsen scaling, half kick, centre-of-mass correction and drift; flat
        ``(q_{n+1}, p_{n+1/2})``."""
        positions, momenta = super().step_pre_force(self._replica_view(positions), self._replica_view(momenta),
                                                    self._replica_view(forces), masses)
        return positions.view(-1, 3), momenta.view(-1, 3)

    def step_post_force(self, momenta: Tensor, forces: Tensor) -> Tensor:
        return super().step_post_force(self._replica_view(momenta), self._replica_view(forces)).view(-1, 3)


def replica_energy_derivative(adapter, batch, inputs, *, replicas: int, replica_atoms: int) -> tuple[Tensor, Tensor]:
    """``(energies (N,) float64, forces (N * n, 3))`` of a disjoint union of replicas: each replica's
    energy is the FP64 sum of its rows' energies (``adapter.node_energy``), or, for a whole-system tap over the
    batch's systems (``"segments"``), its entry of ``adapter.system_energy`` in FP64; one backward pass of their
    sum gives every replica's forces (no term couples two replicas). The engine-owned derivative of
    :class:`~fastmd.engine.energy_derivative.EnergyDerivative`, one energy per replica. A whole-system tap that
    does not return one energy per replica raises :class:`~fastmd.adapters.protocol.NegotiationError`."""
    rows = replicas * replica_atoms
    with torch.enable_grad():
        leaf = batch.positions.detach().requires_grad_(True)
        if isinstance(adapter, EnergyTap):
            node = adapter.node_energy(inputs, leaf, batch.cell)
            energies = node[:rows].to(torch.float64).reshape(replicas, replica_atoms).sum(dim=1)
        else:
            energies = adapter.system_energy(inputs, leaf, batch.cell).to(torch.float64).reshape(-1)
            if energies.shape[0] != replicas:
                name = getattr(adapter, "name", type(adapter).__name__)
                raise NegotiationError(f"model {name!r}: its whole-system energy tap returned {energies.shape[0]} "
                                       f"value(s) for a batch of {replicas} systems; a batch of several systems takes "
                                       "one energy per system (GraphBatch.systems, 'segments')")
        (grad,) = torch.autograd.grad(energies.sum(), (leaf,))
    return energies.detach(), -grad[:rows]


class ReplicaHealthError(ValueError):
    """Replicas whose state turned non-finite (``replicas``: their indices in the union when a runner raises it,
    among all replicas when ``ReplicaMolecularDynamics`` does; the systems of ``MultiSystemMolecularDynamics``,
    ``member="system"``)."""

    def __init__(self, replicas: list[int], window: int, member: str = "replica") -> None:
        shown = ", ".join(map(str, replicas[:16])) + (", ..." if len(replicas) > 16 else "")
        super().__init__(f"non-finite MD state in {len(replicas)} {member}(s) [{shown}] after whole-step window "
                         f"{window}")
        self.replicas = list(replicas)
        self.window = int(window)


class ReplicaWholeStepRunner(WholeStepGraphRunner):
    """:class:`WholeStepGraphRunner` over a :class:`ReplicaUnionGraphBuilder` with a replica integrator
    (:class:`ReplicaNoseHooverChainNVT` or :class:`ReplicaVelocityVerlet`): the same windows, rollback,
    fallback and candidate refresh, with one FP64 potential energy per replica (``potential_energy``,
    ``(N,)``). Models the engine differentiates only (an energy tap per atom, or of each system over the
    batch's systems); no stress record.

    Each window's last step also writes one health flag per replica (1 where its positions, momenta or
    energy are not finite), read with the window's flags in one copy; :meth:`settle` raises
    :class:`ReplicaHealthError` naming those replicas (the window is not rolled back).
    """

    def __init__(self, *, builder: ReplicaUnionGraphBuilder, integrator, **kwargs) -> None:
        replicas, replica_atoms = self._union_shape(builder, integrator)
        if kwargs.get("record_stress") or kwargs.get("step_targets") is not None:
            raise ValueError("ReplicaWholeStepRunner records no stress and takes no per-step targets")
        adapter = as_adapter(kwargs["model"])
        if not isinstance(adapter, (EnergyTap, SystemEnergyTap)) or engine_derivative(adapter) is None:
            raise ValueError(f"ReplicaWholeStepRunner differentiates an energy tap (per atom, or one energy per "
                             f"system); model {adapter.name!r} has none")
        self.replicas = replicas
        self.replica_atoms = replica_atoms
        self._replica_energy = None
        self._window_record = None
        super().__init__(builder=builder, integrator=integrator, **kwargs)

    @staticmethod
    def _union_shape(builder, integrator) -> tuple[int, int]:
        """(replicas, atoms per replica) of the union builder and replica integrator, which must hold the same
        replicas."""
        if not isinstance(builder, ReplicaUnionGraphBuilder) or not isinstance(
                integrator, (ReplicaNoseHooverChainNVT, ReplicaVelocityVerlet)):
            raise TypeError("ReplicaWholeStepRunner needs a ReplicaUnionGraphBuilder and a replica integrator "
                            "(ReplicaNoseHooverChainNVT or ReplicaVelocityVerlet)")
        if builder.replicas != integrator.replicas or builder.replica_atoms != integrator.n_atoms:
            raise ValueError("builder and integrator must hold the same replicas")
        return builder.replicas, builder.replica_atoms

    # The runner's energy buffer: one FP64 energy per replica in place of the single system's [1].
    @property
    def potential_energy(self) -> Tensor:
        return self._replica_energy

    @potential_energy.setter
    def potential_energy(self, value: Tensor) -> None:
        if self._replica_energy is not None:
            raise AttributeError("the replica energies are a persistent buffer: copy_ into it")
        self._replica_energy = torch.zeros(self.replicas, dtype=torch.float64, device=value.device)

    # The window's flags (overflow, edges, triplets, largest displacement) are the first four entries of one
    # buffer whose other entries are the replicas' health flags: one copy reads both.
    @property
    def _window_stats(self) -> Tensor | None:
        return None if self._window_record is None else self._window_record[:4]

    @_window_stats.setter
    def _window_stats(self, value: Tensor | None) -> None:
        if self._window_record is not None:
            raise AttributeError("the window record is a persistent buffer: copy_ into it")
        if value is not None:
            self._window_record = torch.zeros(4 + self.replicas, dtype=torch.float64, device=value.device)

    def _flag_sources(self) -> list[Tensor]:
        return [self._window_record]

    def _health(self) -> Tensor:
        """``(N,)`` float64: 1 where a replica's positions, momenta or energy are not finite (device)."""
        shape = (self.replicas, self.replica_atoms * 3)
        finite = (torch.isfinite(self.positions.view(shape)).all(dim=1)
                  & torch.isfinite(self.momenta.view(shape)).all(dim=1) & torch.isfinite(self.potential_energy))
        return (~finite).to(torch.float64)

    def _window_body(self, record: bool = False) -> None:
        super()._window_body(record)
        self._window_record[4:].copy_(self._health())

    @scoped_call
    def _evaluate(self, positions: Tensor, *, rebuild_candidates: bool, stress: bool = False):
        if stress:
            raise ValueError("ReplicaWholeStepRunner records no stress")
        graph, status = self.builder.build(positions, rebuild_candidates=rebuild_candidates)
        energies, forces = replica_energy_derivative(self.adapter, self.builder.batch, graph,
                                                     replicas=self.replicas, replica_atoms=self.replica_atoms)
        return forces, energies, status

    def settle(self, *, record: bool = False, flags: tuple | None = None) -> None:
        """:meth:`WholeStepGraphRunner.settle`, after the replicas' health flags (read in the same copy as the
        window's flags; read again after a fallback window, whose eager steps write none)."""
        packed = tuple(self._window_record.tolist()) if flags is None else tuple(flags)
        self._raise_unhealthy(packed[4:])
        fallbacks = self.fallback_windows
        try:
            super().settle(record=record, flags=packed[:4])
        except ValueError as error:
            if not isinstance(error, ReplicaHealthError) and "non-finite" in str(error):
                self._raise_unhealthy(self._health().tolist(), cause=error)
            raise
        if self.fallback_windows != fallbacks:
            self._raise_unhealthy(self._health().tolist())

    def _raise_unhealthy(self, health, cause: BaseException | None = None) -> None:
        bad = [replica for replica, flag in enumerate(health) if flag != 0.0]
        if bad:
            raise ReplicaHealthError(bad, self.launches) from cause

    def health(self) -> Tensor:
        """``(N,)`` bool on the host: the replicas whose current state is finite."""
        return self._health().cpu() == 0.0

    @torch.no_grad()
    def reset(self, *, positions: Tensor, momenta: Tensor, forces: Tensor, potential_energy: Tensor,
              reset_counters: bool = True) -> None:
        """:meth:`WholeStepGraphRunner.reset` with ``(N,)`` energies (a stateful integrator's state is the
        caller's)."""
        self.positions.copy_(positions.reshape(self.positions.shape))
        self.momenta.copy_(momenta.reshape(self.momenta.shape))
        self.forces.copy_(forces.reshape(self.forces.shape))
        self.potential_energy.copy_(potential_energy.reshape(self.replicas))
        self.builder.refresh_candidates(self.positions)
        self._settled_displacement = 0.0
        self._displacement_growth = [None, None]
        if reset_counters:
            self.fallback_windows = 0
            self.candidate_refreshes = 1
            self.recovery_replays = 0
            self.launches = 0
            self.max_edge_count = 0
            self.max_triplet_count = 0
            self.max_displacement_seen = 0.0
            self.last_edge_count = 0
            self.last_triplet_count = 0


def _rattled(positions: Tensor, amplitude: float, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = (torch.rand(positions.shape, generator=generator, dtype=torch.float64) * 2.0 - 1.0) * amplitude
    return positions + noise.to(device=positions.device, dtype=positions.dtype)


def union_self_check(adapter, *, positions: Tensor, cell: Tensor, atomic_numbers: Tensor, pbc: Tensor,
                     candidate_skin: float, device, repeats: int = 3, rattle: float = 0.1,
                     tol_multiplier: float = 3.0, absolute_floor: float = 1e-5,
                     energy_floor: float = 1e-6) -> dict:
    """Whether ``adapter`` computes the same energies and forces for systems in a union as alone: two copies
    of the system (``positions``, ``(n, 3)``) rattled differently by up to ``rattle`` Å, evaluated
    ``repeats`` times as a union of the two and ``repeats`` times each alone (fixed-capacity builds, with the
    line graph of a model with triplets, eager; alone, the single-system path: a batch without systems and the
    engine-owned derivative).
    The largest force difference between a union and a single evaluation of the same copy (``cross``) passes
    within ``tol_multiplier`` times the larger of the two paths' own run-to-run differences, and at least
    ``absolute_floor`` eV/Å (a deterministic model has a floor of 0; the caller passes one for its dtype).
    The maximum over more pairs on the union-vs-single side is why the multiplier is above the accuracy
    gates' 1.5: this check guards the declaration, not the accuracy. The energies per atom pass the same way
    (their own floors, at least ``energy_floor`` eV/atom and four rounding steps of the energy in the model's
    dtype, where a model rounds its system's energy once): a per-system term without a gradient changes no
    force. A model whose atoms see another system's atoms (a pooled readout, a per-graph embedding) differs
    far beyond that. Returns the numbers, ``passed`` and what ``failed`` ("forces", "energies" or None)."""
    n = int(positions.shape[0])
    base = positions.to(device=device, dtype=torch.float32)
    copies = [_rattled(base, rattle, seed) for seed in (1, 2)]
    numbers = atomic_numbers.to(device=device, dtype=torch.int32)
    cutoff = float(adapter.topology_spec().cutoff)
    line_graph = adapter.topology_spec().triplet_cutoff is not None
    counts = []  # the triplets of each evaluation's graph (a line graph)

    def evaluate(replicas: int, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        pairs, triplets = union_demand(adapter, coordinates, replicas=replicas, cell=cell, atomic_numbers=numbers,
                                       pbc=pbc, candidate_skin=candidate_skin, device=device)
        u_capacity = union_capacity(pairs, None, line_graph=line_graph)
        t_capacity = union_triplet_capacity(triplets, None, replicas=replicas)
        counts.append(triplets)
        if replicas > 1:
            builder = union_builder(adapter, replicas=replicas, cell=cell, atomic_numbers=numbers, pbc=pbc,
                                    u_capacity=u_capacity, t_capacity=t_capacity, candidate_skin=candidate_skin,
                                    device=device)
        else:  # alone: no systems, so a model that maps them is held to its single-system input
            builder = FixedCapacityGraphBuilder(
                cell=cell, atomic_numbers=numbers, pbc=pbc, u_capacity=u_capacity, atom_graph_cutoff=cutoff,
                candidate_cutoff=cutoff + candidate_skin, device=device, padding=adapter.padding_rule(),
                prepare=adapter.prepare, **graph_builder_options(adapter, t_capacity))
        builder.refresh_candidates(coordinates)
        graph, _ = builder.build(coordinates, rebuild_candidates=False)
        if replicas > 1:
            energies, forces = replica_energy_derivative(adapter, builder.batch, graph, replicas=replicas,
                                                         replica_atoms=n)
        else:
            out = engine_derivative(adapter)(builder.batch, graph, task="ef", n_real=n)
            energies, forces = out["e"], out["f"][0][:n]
        return energies.cpu(), forces.detach().to(torch.float64).cpu().reshape(replicas, n, 3)

    union = [evaluate(2, torch.cat(copies)) for _ in range(repeats)]
    single = [[evaluate(1, copy) for _ in range(repeats)] for copy in copies]

    def largest(pairs) -> float:
        return max((float((a - b).abs().max()) for a, b in pairs), default=0.0)

    single_floor = largest((single[s][i][1][0], single[s][j][1][0]) for s in range(2)
                           for i in range(repeats) for j in range(i + 1, repeats))
    union_floor = largest((union[i][1][s], union[j][1][s]) for s in range(2)
                          for i in range(repeats) for j in range(i + 1, repeats))
    cross = largest((union[i][1][s], single[s][j][1][0]) for s in range(2)
                    for i in range(repeats) for j in range(repeats))
    energy_cross = max(abs(float(union[i][0][s] - single[s][j][0][0])) / n for s in range(2)
                       for i in range(repeats) for j in range(repeats))
    energy_single = max((abs(float(single[s][i][0][0] - single[s][j][0][0])) / n for s in range(2)
                         for i in range(repeats) for j in range(i + 1, repeats)), default=0.0)
    energy_union = max((abs(float(union[i][0][s] - union[j][0][s])) / n for s in range(2)
                        for i in range(repeats) for j in range(i + 1, repeats)), default=0.0)
    # Four rounding steps of the largest energy in the model's dtype, per atom: a model that rounds its system's
    # energy once (a whole-system tap in float32) differs by a step or two between two arithmetics of one sum.
    epsilon = float(np.finfo(np.float64 if adapter.topology_spec().model_dtype == "float64" else np.float32).eps)
    resolution = 4.0 * epsilon * max(abs(float(single[s][i][0][0])) for s in range(2) for i in range(repeats)) / n
    tolerance = max(absolute_floor, tol_multiplier * max(single_floor, union_floor))
    energy_tolerance = max(energy_floor, resolution, tol_multiplier * max(energy_single, energy_union))
    failed = "forces" if cross > tolerance else "energies" if energy_cross > energy_tolerance else None
    return {"passed": failed is None, "failed": failed, "max_force_difference": cross, "tolerance": tolerance,
            "single_floor": single_floor, "union_floor": union_floor, "max_energy_difference_per_atom": energy_cross,
            "energy_tolerance_per_atom": energy_tolerance, "energy_single_floor": energy_single,
            "energy_union_floor": energy_union, "energy_resolution_per_atom": resolution, "replicas": 2,
            "repeats": int(repeats), "rattle": float(rattle),
            **({"triplets_union_alone": [counts[0], counts[repeats], counts[2 * repeats]]} if line_graph else {})}


def union_memory_runs(adapter, *, positions: Tensor, cell: Tensor, atomic_numbers: Tensor, pbc: Tensor,
                      candidate_skin: float, device, step=None, headroom: float = UNION_GROWTH_HEADROOM,
                      triplet_headroom: float = UNION_TRIPLET_GROWTH_HEADROOM):
    """``({kind: run(capacity)}, edges(replicas), triplets(replicas))`` for ``MeasuredMemoryModel.measure`` on
    unions of this system (``positions``: one replica's ``(n, 3)``): ``adapter``'s evaluation of a union of
    that many copies at the capacity one growth would give it (:func:`union_capacity` with ``headroom``, and
    :func:`union_triplet_capacity` with ``triplet_headroom`` and its padding per replica for a line graph: the
    bound covers a union grown once over the counts of ``positions``), eager ("eager") or captured in a CUDA
    graph after a warmup on the capture stream ("capture"). A
    run's capacity is a union size, or for a line graph a (union size, triplet capacity) pair
    (:func:`union_probe_capacities`: the triplets vary apart from the union, so that the fit tells the
    per-triplet cost from the per-edge one); each run returns the (edges, triplets) capacity it held, which
    ``edges`` and ``triplets`` give for a union of any size (predict there; ``triplets`` 0 without a line
    graph). Atoms and edges grow together over the unions measured, so a fit over edges holds the per-atom
    costs too. Each run makes its own builder, and a captured graph is released before the run returns."""
    numbers = atomic_numbers.to(device=device, dtype=torch.int32)
    base = positions.to(device=device, dtype=torch.float32)
    single, single_triplets = union_demand(adapter, base, replicas=1, cell=cell, atomic_numbers=numbers, pbc=pbc,
                                           candidate_skin=candidate_skin, device=device)
    line_graph = adapter.topology_spec().triplet_cutoff is not None
    stream = torch.cuda.Stream(device)

    def edges(replicas: int) -> int:
        return 2 * union_capacity(single * int(replicas), step, headroom=headroom, line_graph=line_graph)

    def triplets(replicas: int) -> int:
        return (union_triplet_capacity(single_triplets * int(replicas), step, headroom=triplet_headroom,
                                       replicas=replicas) if line_graph else 0)

    def evaluator(capacity):
        replicas, t_capacity = capacity if isinstance(capacity, tuple) else (capacity, triplets(capacity))
        coordinates = base.repeat(replicas, 1)
        builder = union_builder(adapter, replicas=replicas, cell=cell, atomic_numbers=numbers, pbc=pbc,
                                u_capacity=edges(replicas) // 2, t_capacity=t_capacity, candidate_skin=candidate_skin,
                                device=device)
        builder.refresh_candidates(coordinates)

        def evaluate() -> None:
            graph, _ = builder.build(coordinates, rebuild_candidates=False)
            replica_energy_derivative(adapter, builder.batch, graph, replicas=builder.replicas,
                                      replica_atoms=base.shape[0])

        return builder, evaluate

    def eager(capacity) -> tuple[int, int]:
        builder, evaluate = evaluator(capacity)
        evaluate()
        return builder.e_capacity, builder.t_capacity

    def capture(capacity) -> tuple[int, int]:
        builder, evaluate = evaluator(capacity)
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
        return builder.e_capacity, builder.t_capacity

    return {"eager": eager, "capture": capture}, edges, triplets


def union_persistent_bytes(adapter, *, positions: Tensor, cell: Tensor, atomic_numbers: Tensor, pbc: Tensor,
                           candidate_skin: float, device, step=None, headroom: float = UNION_GROWTH_HEADROOM,
                           triplet_headroom: float = UNION_TRIPLET_GROWTH_HEADROOM) -> int:
    """The device bytes one more replica adds to a union outside its graph pool, estimated: the builder's buffers
    (the candidate list above all) of two replicas less those of one, plus 32 float64 per atom for the runner's
    and the integrator's per-atom state (positions, momenta, forces, their backups and mirrors: about 30 for a
    float32 model, 34 for a float64 one, more for very small systems)."""
    numbers = atomic_numbers.to(device=device, dtype=torch.int32)
    base = positions.to(device=device, dtype=torch.float32)
    single, single_triplets = union_demand(adapter, base, replicas=1, cell=cell, atomic_numbers=numbers, pbc=pbc,
                                           candidate_skin=candidate_skin, device=device)
    line_graph = adapter.topology_spec().triplet_cutoff is not None

    def held(replicas: int) -> int:
        builder = union_builder(adapter, replicas=replicas, cell=cell, atomic_numbers=numbers, pbc=pbc,
                                u_capacity=union_capacity(single * replicas, step, headroom=headroom,
                                                          line_graph=line_graph),
                                t_capacity=union_triplet_capacity(single_triplets * replicas, step,
                                                                  headroom=triplet_headroom,
                                                                  replicas=replicas) if line_graph else 0,
                                candidate_skin=candidate_skin, device=device)
        seen, total = set(), 0
        for value in vars(builder).values():
            if torch.is_tensor(value) and value.device.type == "cuda" and value.data_ptr() not in seen:
                seen.add(value.data_ptr())
                total += value.untyped_storage().nbytes()
        return total

    return max(held(2) - held(1), 0) + 32 * 8 * int(base.shape[0])


def union_probe_sizes(replicas: int) -> list[int]:
    """The union sizes the memory of a union of ``replicas`` is measured at (``UNION_PROBE_FRACTIONS``)."""
    return sorted({max(1, int(round(fraction * replicas))) for fraction in UNION_PROBE_FRACTIONS})


def union_probe_capacities(replicas: int, edges, triplets=None) -> list:
    """The capacities the memory of a union of ``replicas`` is measured at (the runs of
    :func:`union_memory_runs`): the union sizes of :func:`union_probe_sizes`; for a line graph (``edges`` and
    ``triplets``: the union's capacities by size) four (union size, triplet capacity) pairs: the smallest and
    the largest of those sizes at their own triplet capacity, and the middle one twice, at a triplet/edge ratio
    no larger than that of any union the sizing may choose (``replicas`` halved down to one, whose padding
    shifts the ratio) and at one no smaller than any of theirs, with at least an eighth of its edge rows more
    triplets: the second tells the per-triplet cost apart from the per-edge one even where the triplets are
    few (a small union's reserved bytes vary by a few MiB), and the fit predicts within its measured ratios at
    each union the sizing may choose."""
    sizes = union_probe_sizes(replicas)
    if triplets is None:
        return sizes
    chosen, size = [], int(replicas)
    while True:
        chosen.append(size)
        if size == 1:
            break
        size = max(1, size // 2)
    ratios = [triplets(size) / edges(size) for size in chosen]
    small, middle, large = sizes[0], sizes[len(sizes) // 2], sizes[-1]
    low = min(triplets(middle), int(min(ratios) * edges(middle)))
    high = max(triplets(middle) + max(triplets(middle), edges(middle) // 8), math.ceil(max(ratios) * edges(middle)))
    return [(small, triplets(small)), (middle, max(1, low)), (middle, high), (large, triplets(large))]
