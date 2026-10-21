"""Molecular dynamics of many replicas of one system, advanced together in whole-step CUDA graphs.

:class:`ReplicaMolecularDynamics` runs N independent trajectories of one
system (the same atoms and cell; each replica its own positions, momenta and
thermostat) for a model the engine differentiates (an energy tap) that
declares its systems may be evaluated as one disjoint union
(:class:`~fastmd.adapters.protocol.SystemBatching`: ``"union"``, MACE's
``ScaleShiftMACE``, SevenNet and CHGNet, whose union carries the line graph;
``"segments"``, ORB v3, which pools over each system of the union). The
replicas form unions
(:mod:`fastmd.execution.replica_batch`): one graph build, one model call
and one backward pass per step for all replicas of a union, the integrator,
neighbour list, model and derivative of ``steps_per_launch`` steps captured as
one CUDA graph. Small systems that leave a GPU idle one at a time fill it this
way. ``GPUResidentMolecularDynamics`` stays the entry point for one system.
"""
from __future__ import annotations

from dataclasses import replace
import gc
import math
import time

import numpy as np
import torch
from ase import Atoms, units

from fastmd.adapters import load_adapter
from fastmd.adapters.memory import MeasuredMemoryModel
from fastmd.adapters.negotiate import batch_systems, check_system_batching, negotiate, negotiated, size_unions
from fastmd.adapters.protocol import NegotiationError
from fastmd.config import InferenceConfig, resolve_config
from fastmd.engine.entry_config import release_adapter, use_adapter
from fastmd.execution.gpu_integrator import default_fix_com, kinetic_energy_fp64
from fastmd.execution.gpu_nhc import maxwell_boltzmann_momenta
from fastmd.execution.replica_batch import (
    DEFAULT_UNION_ROWS,
    ReplicaHealthError,
    ReplicaNoseHooverChainNVT,
    ReplicaVelocityVerlet,
    ReplicaWholeStepRunner,
    union_builder,
    union_capacity,
    union_demand,
    union_headrooms,
    union_memory_runs,
    union_persistent_bytes,
    union_probe_capacities,
    union_self_check,
    union_triplet_capacity,
)
from fastmd.execution.whole_step_cuda_graph import FixedCapacityOverflow
from fastmd.interfaces.calculator import require_finite_geometry

ENSEMBLES = ("nvt_nhc", "nvt", "nve")
# The smallest union whose memory is measured (three distinct probe sizes); smaller ones are not bounded.
_MEASURED_UNION = 4
_COUNTERS = ("launches", "candidate_refreshes", "fallback_windows", "recovery_replays")


class _Union:
    """One union of replicas ``[start, stop)``: its integrator, runner and pair (and triplet) capacity."""

    def __init__(self, start: int, stop: int) -> None:
        self.start, self.stop = int(start), int(stop)
        self.integrator = None
        self.runner = None
        self.u_capacity = 0
        self.t_capacity = 0
        self.capture_seconds: list[float] = []
        self.accumulated = dict.fromkeys(_COUNTERS, 0)
        self.max_edge_count = 0
        self.max_triplet_count = 0
        self.recaptures = 0

    @property
    def replicas(self) -> int:
        return self.stop - self.start


class _UnionMolecularDynamics:
    """The union loop of the batched MD entry points: every union runs each window (an overflow rolls it back,
    grows its capacity and captures it again), health, statistics and the adapter's release. A subclass sets
    ``_unions``, ``_line_graph``, ``_capacity_step``, the headrooms, ``_config``, ``ensemble``, ``nsteps``,
    ``steps_per_launch``, ``loginterval``, ``_log`` and ``_plan``, and provides ``_capture`` and ``_write_log``."""

    _adapter_finalizer = None
    _members = "replicas"  # what the batch holds, in messages

    def run(self, steps: int) -> None:
        """Advance every member of the batch by ``steps`` MD steps (whole windows), logging every
        ``loginterval``."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0 or steps % self.steps_per_launch:
            raise ValueError(f"steps must be a nonnegative multiple of steps_per_launch={self.steps_per_launch}")
        if self.loginterval and not self._logged_initial:
            self._write_log()
            self._logged_initial = True
        with self._config.scope():
            for _ in range(steps // self.steps_per_launch):
                unhealthy = []
                for union in self._unions:  # every union runs the window, so that they stay at one step
                    try:
                        self._window(union)
                    except ReplicaHealthError as error:
                        unhealthy.append((union, error.with_traceback(None)))
                self.nsteps += self.steps_per_launch
                if unhealthy:
                    raise ReplicaHealthError([union.start + replica for union, error in unhealthy
                                              for replica in error.replicas],
                                             self.nsteps // self.steps_per_launch,
                                             member=self._members[:-1]) from unhealthy[0][1]
                if self.loginterval and self.nsteps % self.loginterval == 0:
                    self._write_log()

    def _window(self, union: _Union) -> None:
        """One window of ``union``: an overflow rolls it back, grows the capacity, captures the union again and
        runs the window again."""
        while True:
            counts = None
            try:
                union.runner.replay(strict=True)
            except FixedCapacityOverflow as overflow:
                counts = overflow.edge_count, overflow.triplet_count
            if counts is None:
                union.max_edge_count = max(union.max_edge_count, union.runner.max_edge_count)
                union.max_triplet_count = max(union.max_triplet_count, union.runner.max_triplet_count)
                return
            # The window was rolled back to its start: grow and capture again, outside the except block (its
            # traceback would keep the old runner's graph alive).
            self._grow(union, *counts)

    def _grow(self, union: _Union, edges: int, triplets: int = 0) -> None:
        """Capture ``union`` again with the capacity that overflowed grown over ``edges`` or ``triplets`` (its
        window rolled back): the triplets' where they outgrew theirs (a line graph), else the pairs'. A failed
        capture (e.g. out of memory) leaves the union captured at its previous capacities, its state kept, and
        raises."""
        old = union.runner
        previous = union.u_capacity, union.t_capacity
        triplets_over = self._line_graph and int(triplets) > union.t_capacity
        if not triplets_over or int(edges) >= 2 * union.u_capacity:  # a line graph keeps a padding pair
            union.u_capacity = union_capacity(max(-(-int(edges) // 2), previous[0]), self._capacity_step,
                                              headroom=self.capacity_headroom, line_graph=self._line_graph)
        if triplets_over:
            union.t_capacity = union_triplet_capacity(int(triplets), self._capacity_step,
                                                      headroom=self.triplet_headroom, replicas=union.replicas)
        stats = old.stats()
        for name in _COUNTERS:
            union.accumulated[name] += stats[name]
        union.max_edge_count = max(union.max_edge_count, old.max_edge_count)
        union.max_triplet_count = max(union.max_triplet_count, old.max_triplet_count)
        positions, momenta = old.positions.clone(), old.momenta.clone()
        forces = (union.integrator.forces if self.ensemble == "nvt_nhc" else old.forces).clone()
        union.runner = old = None
        torch.cuda.empty_cache()
        try:
            self._capture(union, positions, momenta, forces=forces)
        except Exception as error:  # noqa: BLE001 - raised again below, after the previous capacity is back
            failure = error.with_traceback(None)  # no frame of the failed capture stays referenced
        else:
            union.recaptures += 1
            return
        grown = union.u_capacity, union.t_capacity
        union.u_capacity, union.t_capacity = previous
        gc.collect()
        torch.cuda.empty_cache()
        self._capture(union, positions, momenta, forces=forces)
        what, old, new, held = (("(pair, triplet) capacity", previous, grown, previous) if self._line_graph
                                else ("pair capacity", previous[0], grown[0], f"{previous[0]} pairs"))
        raise RuntimeError(f"{self._members} [{union.start}, {union.stop}): growing the {what} from {old} to {new} "
                           f"failed ({type(failure).__name__}); the union stays captured at {held} (its state is "
                           "kept; its next window overflows again)") from failure

    def _per_union(self, read) -> torch.Tensor:
        return torch.cat([read(union) for union in self._unions])

    def _union_stats(self, **members) -> dict:
        """The counters of :meth:`stats` over every union and every capture of it (after them ``members``, e.g.
        the replica count)."""
        total = dict.fromkeys(_COUNTERS, 0)
        for union in self._unions:
            stats = union.runner.stats()
            for name in _COUNTERS:
                total[name] += union.accumulated[name] + stats[name]
        max_edges = [max(union.max_edge_count, union.runner.max_edge_count) for union in self._unions]
        stats = {**total, **members, "unions": [union.replicas for union in self._unions],
                 "steps_per_launch": self.steps_per_launch, "recaptures": sum(u.recaptures for u in self._unions),
                 "u_capacity": [union.u_capacity for union in self._unions], "max_edge_count": max_edges,
                 "capacity_fill": [edges / (2 * union.u_capacity) for edges, union in zip(max_edges, self._unions)],
                 "capture_seconds": [list(union.capture_seconds) for union in self._unions]}
        if self._line_graph:
            max_triplets = [max(union.max_triplet_count, union.runner.max_triplet_count) for union in self._unions]
            stats.update(t_capacity=[union.t_capacity for union in self._unions], max_triplet_count=max_triplets,
                         triplet_capacity_fill=[triplets / union.t_capacity for triplets, union in
                                                zip(max_triplets, self._unions)])
        return stats

    def plan_record(self):
        """The negotiated plan (:class:`~fastmd.adapters.negotiate.EnginePlanRecord`) with the system batching
        and union size decisions."""
        return self._plan

    @property
    def adapter(self):
        return self._adapter

    def close(self) -> None:
        """Close the log, then release the model adapter (one loaded by name is closed with its last user). The
        object is not used afterwards; context exit closes too."""
        try:
            if self._log is not None:
                self._log.close()
        finally:
            self._log = None
            release_adapter(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class ReplicaMolecularDynamics(_UnionMolecularDynamics):
    """N replicas of ``atoms`` advanced together (see the module docstring).

    Parameters
    ----------
    model:
        An adapter, or a name or checkpoint path that :func:`~fastmd.adapters.load_adapter` loads
        (``model_path``, ``model_dtype`` as there). MatRIS is not batched here (replicas of MatRIS:
        :class:`~fastmd.execution.whole_step_cuda_graph.BatchedWholeStepGraphRunner`); a model that does not
        declare ``"union"`` or ``"segments"`` (or declares ``"union"`` with a whole-system tap, or pads with sink
        atoms, or a line graph otherwise than with far self-loops: ``check_system_batching``), or fails its check
        on two copies of the system, raises :class:`~fastmd.adapters.protocol.NegotiationError`.
    atoms:
        The system (periodic in all three directions).
    replicas:
        N; default: the length of ``momenta`` or ``positions``.
    momenta, positions:
        ``(N, n, 3)`` float64 in ASE units. Default positions: ``atoms``' for every replica; default momenta:
        Maxwell-Boltzmann at ``temperature`` for replica ``r`` with ``seed + r`` (ASE's
        ``MaxwellBoltzmannDistribution`` with ``np.random.default_rng(seed + r)``).
    ensemble:
        ``"nvt_nhc"`` (ASE's ``NoseHooverChainNVT`` per replica, the default), ``"nvt"`` (Berendsen) or
        ``"nve"``; ``fix_com`` as in ``GPUResidentMolecularDynamics`` (each replica's own centre of mass).
        A float64 model needs ``"nvt_nhc"`` (the integrator with a float64 state).
    temperature, timestep, taut, tdamp, tchain, tloop:
        K, fs, the Berendsen time constant (fs), the chain's damping (fs, default ``100 * timestep``), chain
        length and loops, as in ``GPUResidentMolecularDynamics``.
    steps_per_launch:
        MD steps per captured window; ``run``, ``loginterval`` and the logged steps are whole windows.
    union_size:
        Replicas per union; default: as many as fit ``DEFAULT_UNION_ROWS`` atom rows (past the union size where
        MACE-MPA-0's throughput on a 135-atom system stopped growing), fewer where the measured memory of the
        unions requires. More replicas
        run as several unions of balanced sizes that share one CUDA graph memory pool, one after another.
    capacity_headroom, triplet_headroom:
        The relative pair (triplet) headroom a capacity growth adds over the overflowing demand; default: the
        model's (``CapacityStep.u_union_headroom``, ``t_union_headroom``), else ``UNION_GROWTH_HEADROOM`` and
        ``UNION_TRIPLET_GROWTH_HEADROOM`` (triplets: a model with a line graph). A union's first capture holds
        the model's own capacity tiers; when the pair or triplet count outgrows its capacity, the window is
        rolled back, that capacity grows and the union is captured again.
    config:
        :class:`~fastmd.config.InferenceConfig`: precision (IEEE FP32 unless TF32 is asked for), profile,
        capacity warmup. Execution is whole-step (``"auto"`` or ``"whole_step"``).
    trajectory, logfile, loginterval:
        ``trajectory``: a file name with ``{replica}`` (e.g. ``"md_{replica:04d}.traj"``): one ASE trajectory
        per replica, a frame (positions, momenta, energy, forces) every ``loginterval`` steps and at the
        start, appended with one file open at a time; ``logfile``: one line per replica and logged step (time,
        energies, temperature).
    self_check:
        Check the model's union on two copies of the system at construction (default on).
    """

    def __init__(self, model, atoms: Atoms, *, replicas: int | None = None, momenta=None, positions=None,
                 ensemble: str = "nvt_nhc", temperature: float = 300.0, timestep: float = 1.0, taut: float = 100.0,
                 tdamp: float | None = None, tchain: int = 3, tloop: int = 1, fix_com: bool | None = None,
                 seed: int = 0, steps_per_launch: int = 10, union_size: int | None = None,
                 capacity_headroom: float | None = None, triplet_headroom: float | None = None,
                 config: InferenceConfig | None = None,
                 model_path: str | None = None, model_dtype: str | None = None, device=None,
                 trajectory: str | None = None, logfile: str | None = None, loginterval: int = 0,
                 self_check: bool = True) -> None:
        ensemble = str(ensemble).lower()
        if ensemble not in ENSEMBLES:
            raise ValueError(f"ReplicaMolecularDynamics: ensemble must be one of {ENSEMBLES}, got {ensemble!r}")
        if ensemble == "nvt_nhc" and fix_com:
            raise ValueError("ReplicaMolecularDynamics: ensemble='nvt_nhc' has no centre-of-mass correction (ASE "
                             "NoseHooverChainNVT); fix_com must be None or False")
        for name, value, low in (("steps_per_launch", steps_per_launch, 1), ("loginterval", loginterval, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"ReplicaMolecularDynamics: {name} must be an integer >= {low}")
        if loginterval % steps_per_launch:
            raise ValueError(f"ReplicaMolecularDynamics: loginterval={loginterval} must be a multiple of "
                             f"steps_per_launch={steps_per_launch} (a window's inner steps are not visible)")
        if trajectory is not None and "{replica" not in str(trajectory):
            raise ValueError("ReplicaMolecularDynamics: trajectory must name each replica's file with '{replica}'")
        if (trajectory is not None or logfile is not None) and not loginterval:
            raise ValueError("ReplicaMolecularDynamics: trajectory and logfile write every loginterval steps: give "
                             "loginterval > 0")
        finite = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if not finite(timestep) or timestep <= 0:
            raise ValueError("ReplicaMolecularDynamics: timestep must be finite and positive")
        if ensemble != "nve" and (not finite(temperature) or temperature <= 0):
            raise ValueError(f"ReplicaMolecularDynamics: ensemble={ensemble!r} needs a finite positive temperature")
        for name, value in (("taut", taut), ("tdamp", tdamp)):
            if value is not None and (not finite(value) or value <= 0):
                raise ValueError(f"ReplicaMolecularDynamics: {name} must be finite and positive")
        for name, value in (("capacity_headroom", capacity_headroom), ("triplet_headroom", triplet_headroom)):
            if value is not None and (not finite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"ReplicaMolecularDynamics: {name} must be None or in [0, 1]")
        if union_size is not None and (isinstance(union_size, bool) or not isinstance(union_size, int)
                                       or union_size < 1):
            raise ValueError("ReplicaMolecularDynamics: union_size must be a positive integer")
        if not bool(np.all(atoms.get_pbc())):
            raise ValueError("ReplicaMolecularDynamics: the cell must be periodic in all three directions")
        require_finite_geometry(atoms.positions, atoms.cell.array, "ReplicaMolecularDynamics")
        self.n_atoms = n = len(atoms)
        momenta = None if momenta is None else np.asarray(momenta, dtype=np.float64)
        positions = None if positions is None else np.asarray(positions, dtype=np.float64)
        for name, value in (("momenta", momenta), ("positions", positions)):
            if value is not None and (value.ndim != 3 or value.shape[1:] != (n, 3) or not np.isfinite(value).all()):
                raise ValueError(f"ReplicaMolecularDynamics: {name} must be finite with shape (N, {n}, 3)")
        counts = {int(value.shape[0]) for value in (momenta, positions) if value is not None}
        if replicas is not None:
            counts.add(int(replicas))
        if len(counts) != 1 or min(counts) < 1:
            raise ValueError("ReplicaMolecularDynamics: give replicas (N >= 1), or momenta / positions of N replicas, "
                             "consistently")
        self.replicas = counts.pop()
        self.ensemble = ensemble
        self.timestep = float(timestep)
        self.steps_per_launch = steps_per_launch
        self.loginterval = loginterval
        self._headrooms = capacity_headroom, triplet_headroom  # resolved with the model's capacity step
        self.device = torch.device(device if device is not None else "cuda")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("ReplicaMolecularDynamics runs on a CUDA device")
        if self.device.index is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.set_device(self.device)
        resolved = resolve_config(config, default_execution="whole_step")
        if resolved.execution not in ("auto", "whole_step"):
            raise ValueError(f"ReplicaMolecularDynamics runs whole-step CUDA graphs: execution must be 'auto' or "
                             f"'whole_step', got {resolved.execution!r}")
        self._adapter = load_adapter(model, model_path=model_path, device=str(self.device), model_dtype=model_dtype,
                                     optimization_profile=resolved.optimization_profile)
        use_adapter(self, model)  # an adapter loaded by name is closed with its last user (also on a failure below)
        if self._adapter is model or self._adapter.module is model:  # supplied, not loaded
            self._adapter.module.to(self.device)
        self._trajectories = False  # whether the replicas' trajectory files exist (the first logged step makes them)
        self._log = None
        try:
            self._setup(atoms, momenta, positions, resolved, temperature=temperature, taut=taut, tdamp=tdamp,
                        tchain=tchain, tloop=tloop, fix_com=fix_com, seed=seed, union_size=union_size,
                        self_check=self_check)
            self._trajectory_name = trajectory
            if logfile is not None:
                self._log = open(logfile, "w")
                self._log.write(f"{'Time[ps]':>12} {'Replica':>8} {'Etot[eV]':>16} {'Epot[eV]':>16} "
                                f"{'Ekin[eV]':>14} {'T[K]':>10}\n")
        except BaseException:
            release_adapter(self)  # a failed constructor gives its adapter back now, not at its collection
            raise
        self.nsteps = 0
        self._logged_initial = False

    # ------------------------------------------------------------------ construction
    def _setup(self, atoms, momenta, positions, resolved, *, temperature, taut, tdamp, tchain, tloop, fix_com, seed,
               union_size, self_check) -> None:
        adapter = self._adapter
        check_system_batching(adapter)  # before anything is negotiated or measured
        if not negotiated(adapter):
            raise NegotiationError(f"model {adapter.name!r} is planned by the runtime's own rules, not batched here")
        plan = negotiate(adapter, replace(resolved, execution="whole_step"), entry="gpu_md", device=self.device)
        execution = plan.decision("execution")
        if execution.chosen != "whole_step":
            raise NegotiationError(f"ReplicaMolecularDynamics needs a whole-step CUDA graph for model "
                                   f"{adapter.name!r}: {execution.reason}")
        # The plan's precision applies to every model call from here on (IEEE FP32 unless TF32 is asked for).
        self._config = replace(resolved, execution="whole_step", matmul_precision=plan.matmul_precision)
        topology = adapter.topology_spec()
        self._float64 = topology.model_dtype == "float64"
        if self._float64 and self.ensemble != "nvt_nhc":
            raise ValueError("ReplicaMolecularDynamics: a float64 model needs ensemble='nvt_nhc' (the integrator with "
                             "a float64 state); nve and nvt keep a float32 state")
        self.candidate_skin = float(self._config.capacity.candidate_skin)
        self._capacity_step = adapter.capacity_step() if callable(getattr(adapter, "capacity_step", None)) else None
        self.capacity_headroom, self.triplet_headroom = union_headrooms(self._capacity_step, *self._headrooms)
        self._line_graph = topology.triplet_cutoff is not None  # the union's line graph: triplet capacities too
        n, N = self.n_atoms, self.replicas
        self.atoms_template = atoms.copy()
        self.numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        # float64: the builder takes its float32 search cell from it and gives a float64 model the exact cell
        self.cell = torch.as_tensor(np.asarray(atoms.get_cell()), dtype=torch.float64, device=self.device)
        self.pbc = torch.ones(3, dtype=torch.bool, device=self.device)
        self.masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        self._masses_t = torch.as_tensor(self.masses, dtype=torch.float64, device=self.device)
        if positions is None:
            positions = np.repeat(atoms.get_positions()[None], N, axis=0)
        if momenta is None:
            momenta = np.stack([maxwell_boltzmann_momenta(self.masses, temperature, seed=int(seed) + r)
                                for r in range(N)])
        if union_size is None:
            union_size = max(1, min(N, DEFAULT_UNION_ROWS // n))
        union_size = min(int(union_size), N)
        system = torch.as_tensor(atoms.get_positions(), dtype=torch.float32, device=self.device)
        geometry = dict(cell=self.cell, atomic_numbers=self.numbers, pbc=self.pbc, candidate_skin=self.candidate_skin,
                        device=self.device)
        with self._config.scope():
            if self_check:
                floor, energy_floor = (1e-10, 1e-12) if self._float64 else (1e-5, 1e-6)
                plan = batch_systems(plan, adapter, systems=N, self_check=lambda: union_self_check(
                    adapter, positions=system, absolute_floor=floor, energy_floor=energy_floor, **geometry))
            else:
                plan = batch_systems(plan, adapter, systems=N)
            while True:  # a probe that runs out of memory: unions half the size
                memory, union_edges, union_triplets, out_of_memory = self._union_memory(adapter, system, union_size,
                                                                                       geometry)
                if not out_of_memory or union_size == 1:
                    break
                union_size = max(1, union_size // 2)
            # The other replicas' buffers outside the shared pool, wherever the sizing may leave more than one union.
            persistent = (union_persistent_bytes(adapter, positions=system, step=self._capacity_step,
                                                 headroom=self.capacity_headroom,
                                                 triplet_headroom=self.triplet_headroom, **geometry)
                          if N > 1 and memory is not None else 0)
            union_size, plan = size_unions(plan, requested=union_size, memory=memory, union_edges=union_edges,
                                           device=self.device, replicas=N, persistent_per_replica=persistent,
                                           union_triplets=union_triplets if self._line_graph else None)
            count = -(-N // union_size)
            bounds = np.cumsum([0] + [N // count + (1 if k < N % count else 0) for k in range(count)])
            self._unions = [_Union(start, stop) for start, stop in zip(bounds[:-1], bounds[1:])]
            # The decision records the balanced partition the replicas run in.
            plan.decisions = [replace(decision, evidence={**decision.evidence, "unions": [u.replicas for u in
                                                                                        self._unions]})
                              if decision.id == "union_size" else decision for decision in plan.decisions]
            self._plan = plan
            # One CUDA graph memory pool for every union's captures: they replay one after another.
            self._graph_pool = torch.cuda.graph_pool_handle()
            self._kinds = dict(ensemble=self.ensemble, temperature=float(temperature), taut=float(taut),
                               tdamp=float(100.0 * self.timestep if tdamp is None else tdamp), tchain=int(tchain),
                               tloop=int(tloop), fix_com=default_fix_com(self.ensemble) if fix_com is None
                               else bool(fix_com))
            for union in self._unions:
                self._start_union(union, positions[union.start:union.stop], momenta[union.start:union.stop])

    def _union_memory(self, adapter, system, union_size: int, geometry: dict):
        """The measured memory model of unions of this system up to ``union_size`` replicas (peak reserved
        bytes, what runs out; with a per-triplet cost for a line graph; None for a union too small to measure),
        the edge and triplet capacities of a union of a given size, and whether the measurement ran out of CUDA
        memory. A failed measurement gives an invalid model, which bounds nothing."""
        runs, union_edges, union_triplets = union_memory_runs(
            adapter, positions=system, step=self._capacity_step, headroom=self.capacity_headroom,
            triplet_headroom=self.triplet_headroom, **geometry)
        if union_size < _MEASURED_UNION:
            return None, union_edges, union_triplets, False
        capacities = union_probe_capacities(union_size, union_edges, union_triplets if self._line_graph else None)
        try:
            memory = MeasuredMemoryModel.measure(runs, capacities, self.device, triplets=self._line_graph,
                                                 metric="reserved")
        except torch.OutOfMemoryError as error:
            why, out_of_memory = f"ran out of CUDA memory ({str(error).splitlines()[0][:160]})", True
        except RuntimeError as error:
            if "capture" not in str(error).lower():
                raise
            why, out_of_memory = f"failed in a capture ({str(error).splitlines()[0][:160]})", False
        else:
            gc.collect()
            torch.cuda.empty_cache()  # the probes' blocks go back before the unions are captured
            return memory, union_edges, union_triplets, False
        # Outside the except blocks: a traceback would pin the failed run's graph and buffers.
        gc.collect()
        torch.cuda.empty_cache()
        failed = MeasuredMemoryModel.failed(f"the union memory measurement {why}", triplets=self._line_graph)
        return failed, union_edges, union_triplets, out_of_memory

    def _start_union(self, union: _Union, positions: np.ndarray, momenta: np.ndarray) -> None:
        """The integrator of ``union`` at ``positions`` and ``momenta`` (``(k, n, 3)``), and its first runner."""
        kinds, k = self._kinds, union.replicas
        if self.ensemble == "nvt_nhc":
            union.integrator = ReplicaNoseHooverChainNVT(
                self.masses, positions, momenta, replicas=k, temperature_K=kinds["temperature"],
                timestep=self.timestep * units.fs, tdamp=kinds["tdamp"] * units.fs, tchain=kinds["tchain"],
                tloop=kinds["tloop"], device=self.device, force_dtype=torch.float64 if self._float64 else torch.float32)
            state = union.integrator.positions.to(torch.float32), union.integrator.momenta.to(torch.float32)
        else:
            union.integrator = ReplicaVelocityVerlet(
                torch.as_tensor(self.masses, dtype=torch.float32, device=self.device), self.timestep, replicas=k,
                ensemble=self.ensemble, temperature=kinds["temperature"], taut_fs=kinds["taut"], fix_com=kinds["fix_com"])
            state = (torch.as_tensor(positions.reshape(-1, 3), dtype=torch.float32, device=self.device),
                     torch.as_tensor(momenta.reshape(-1, 3), dtype=torch.float32, device=self.device))
        pairs, triplets = union_demand(self._adapter, state[0], replicas=k, cell=self.cell,
                                       atomic_numbers=self.numbers, pbc=self.pbc, candidate_skin=self.candidate_skin,
                                       device=self.device)
        union.u_capacity = union_capacity(pairs, self._capacity_step, line_graph=self._line_graph)  # the model's tiers
        union.t_capacity = (union_triplet_capacity(triplets, self._capacity_step, replicas=k) if self._line_graph
                            else 0)
        self._capture(union, *state, forces=None)

    def _capture(self, union: _Union, positions, momenta, *, forces) -> None:
        """A runner of ``union`` at its capacity (a recapture when ``forces``, the cached F of the state, are
        given)."""
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        builder = union_builder(self._adapter, replicas=union.replicas, cell=self.cell, atomic_numbers=self.numbers,
                                pbc=self.pbc, u_capacity=union.u_capacity, t_capacity=union.t_capacity,
                                candidate_skin=self.candidate_skin, device=self.device)
        union.runner = ReplicaWholeStepRunner(
            model=self._adapter, builder=builder, integrator=union.integrator, positions=positions, momenta=momenta,
            forces=forces, steps_per_launch=self.steps_per_launch, warmup=self._config.capacity.warmup,
            graph_pool=self._graph_pool, config=self._config, recapture=forces is not None,
            validate_isolated_atoms=not self._config.handle_isolated_atoms)
        torch.cuda.synchronize(self.device)
        union.capture_seconds.append(time.perf_counter() - started)

    # ------------------------------------------------------------------ state and observables
    def _momenta(self, union: _Union) -> torch.Tensor:
        source = union.integrator.momenta if self.ensemble == "nvt_nhc" else union.runner.momenta
        return source.reshape(union.replicas, self.n_atoms, 3)

    def get_positions(self) -> np.ndarray:
        """``(N, n, 3)`` float64 positions, unwrapped (the MD state: the Nose-Hoover chain's float64 positions,
        or the runner's float32 ones for nve and nvt)."""
        return self._per_union(lambda union: (union.integrator.positions if self.ensemble == "nvt_nhc"
                                              else union.runner.positions).reshape(union.replicas, self.n_atoms, 3)
                               .to(torch.float64)).cpu().numpy()

    def get_momenta(self) -> np.ndarray:
        """``(N, n, 3)`` float64 momenta."""
        return self._per_union(lambda union: self._momenta(union).to(torch.float64)).cpu().numpy()

    def get_forces(self) -> np.ndarray:
        """``(N, n, 3)`` float64 forces at the current positions."""
        return self._per_union(lambda union: (union.integrator.forces if self.ensemble == "nvt_nhc"
                                              else union.runner.forces).reshape(union.replicas, self.n_atoms, 3)
                               .to(torch.float64)).cpu().numpy()

    def kinetic_energies(self) -> torch.Tensor:
        """``(N,)`` float64 kinetic energies (eV)."""
        return self._per_union(lambda union: kinetic_energy_fp64(self._momenta(union), self._masses_t))

    def temperatures(self) -> torch.Tensor:
        """``(N,)`` temperatures (K): each replica's ``Atoms.get_temperature()``."""
        return 2.0 * self.kinetic_energies() / (3 * self.n_atoms * units.kB)

    def potential_energies(self) -> torch.Tensor:
        """``(N,)`` float64 potential energies (eV)."""
        return self._per_union(lambda union: (union.integrator.potential_energy if self.ensemble == "nvt_nhc"
                                              else union.runner.potential_energy).to(torch.float64))

    def conserved_energies(self) -> torch.Tensor:
        """``(N,)``: the Nose-Hoover chain's conserved energy (``nvt_nhc``), else potential plus kinetic."""
        if self.ensemble == "nvt_nhc":
            return self._per_union(lambda union: union.integrator.conserved_energy())
        return self.potential_energies() + self.kinetic_energies()

    def health(self) -> np.ndarray:
        """``(N,)`` bool: the replicas whose state is finite (a window that leaves one non-finite raises
        :class:`~fastmd.execution.replica_batch.ReplicaHealthError` naming them)."""
        return torch.cat([union.runner.health() for union in self._unions]).numpy()

    def get_atoms(self, replica: int) -> Atoms:
        """Replica ``replica`` as ASE atoms with its positions and momenta."""
        atoms = self.atoms_template.copy()
        atoms.set_positions(self.get_positions()[replica])
        atoms.set_momenta(self.get_momenta()[replica])
        return atoms

    def snapshot(self) -> dict:
        """The replicas' MD state (device copies), for :meth:`restore`."""
        unions = []
        for union in self._unions:
            runner = union.runner
            unions.append({"integrator": union.integrator.snapshot() if self.ensemble == "nvt_nhc" else None,
                           "state": [tensor.clone() for tensor in (runner.positions, runner.momenta, runner.forces,
                                                                   runner.potential_energy)]})
        return {"nsteps": self.nsteps, "layout": [(union.start, union.stop) for union in self._unions],
                "n_atoms": self.n_atoms, "ensemble": self.ensemble, "unions": unions}

    @torch.no_grad()
    def restore(self, snapshot: dict, *, momenta=None) -> None:
        """Back to ``snapshot`` (:meth:`snapshot` of this object), with new ``(N, n, 3)`` momenta if given
        (positions, forces and energies stay the snapshot's): the captured graphs and the grown capacities are
        kept; the window counters of :meth:`stats` start again."""
        if (snapshot.get("layout") != [(union.start, union.stop) for union in self._unions]
                or snapshot.get("n_atoms") != self.n_atoms or snapshot.get("ensemble") != self.ensemble):
            raise ValueError("ReplicaMolecularDynamics.restore: the snapshot is not one of this object's unions")
        momenta = None if momenta is None else np.asarray(momenta, dtype=np.float64).reshape(self.replicas,
                                                                                              self.n_atoms, 3)
        with self._config.scope():
            for union, saved in zip(self._unions, snapshot["unions"]):
                positions, moment, forces, energies = saved["state"]
                if self.ensemble == "nvt_nhc":
                    union.integrator.load_snapshot(saved["integrator"])
                    if momenta is not None:
                        union.integrator.set_state(momenta=momenta[union.start:union.stop])
                    union.integrator.backup()
                    integrator = union.integrator
                    positions, moment = integrator.positions.to(torch.float32), integrator.momenta.to(torch.float32)
                    forces, energies = integrator.forces.to(union.runner.forces.dtype), integrator.potential_energy
                elif momenta is not None:
                    moment = torch.as_tensor(momenta[union.start:union.stop].reshape(-1, 3), dtype=torch.float32,
                                             device=self.device)
                union.runner.reset(positions=positions, momenta=moment, forces=forces, potential_energy=energies)
                union.accumulated = dict.fromkeys(_COUNTERS, 0)
                union.max_edge_count = union.max_triplet_count = 0
        self.nsteps = snapshot["nsteps"]

    def stats(self) -> dict:
        """Counters over every union and every capture of it since construction or the last :meth:`restore`:
        windows launched, candidate refreshes, fallback windows, recovery replays and the largest edge count
        (with the capacities and their fill; and the triplets' for a line graph); since construction:
        recaptures (capacity growth) and capture times (s)."""
        return self._union_stats(replicas=self.replicas)

    # ------------------------------------------------------------------ output
    def _write_log(self) -> None:
        if self._log is None and self._trajectory_name is None:
            return
        positions, momenta, forces = self.get_positions(), self.get_momenta(), self.get_forces()
        potential = self.potential_energies().cpu().numpy()
        kinetic = self.kinetic_energies().cpu().numpy()
        if self._trajectory_name is not None:
            from ase.calculators.singlepoint import SinglePointCalculator
            from ase.io import Trajectory

            # One file open at a time (a file per replica would pass the open-file limit at a few thousand
            # replicas): the first logged step creates every replica's file, each later one appends to it (a
            # missing file is created; one written before a failure is not truncated).
            mode = "a" if self._trajectories else "w"
            self._trajectories = True
            for r in range(self.replicas):
                atoms = self.atoms_template.copy()
                atoms.set_positions(positions[r])
                atoms.set_momenta(momenta[r])
                atoms.calc = SinglePointCalculator(atoms, energy=float(potential[r]), forces=forces[r])
                with Trajectory(self._trajectory_name.format(replica=r), mode) as writer:
                    writer.write(atoms)
        if self._log is not None:
            time_ps = self.nsteps * self.timestep / 1000.0
            temperatures = 2.0 * kinetic / (3 * self.n_atoms * units.kB)
            for r in range(self.replicas):
                self._log.write(f"{time_ps:12.4f} {r:8d} {potential[r] + kinetic[r]:16.6f} {potential[r]:16.6f} "
                                f"{kinetic[r]:14.6f} {temperatures[r]:10.2f}\n")
            self._log.flush()
