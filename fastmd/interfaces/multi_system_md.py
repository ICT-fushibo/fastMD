"""Molecular dynamics of a set of different systems, advanced together in whole-step CUDA graphs.

:class:`MultiSystemMolecularDynamics` runs one trajectory for each of S
systems (each with its own atoms, atom count, cell, positions, momenta and
thermostat) for a model the engine differentiates that declares its systems
may be evaluated as one disjoint union (``SystemBatching`` ``"union"``) and
takes the engine's per-edge shifts (``EdgeShiftInput``: MACE's
``ScaleShiftMACE`` and SevenNet). The systems sit in the slots of unions
(:mod:`fastmd.execution.system_batch`): one graph build, one model call and one
backward pass per step for all systems of a union, ``steps_per_launch`` steps
of integrator, neighbour list, model and derivative captured as one CUDA graph.
A slot holds at most its atom ceiling; :meth:`write_system` puts another system
into a slot in place, without a recapture.
:class:`~fastmd.interfaces.replica_md.ReplicaMolecularDynamics` stays the entry
point for replicas of one system, ``GPUResidentMolecularDynamics`` for one
system.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import gc
import math
import numbers
import time

import numpy as np
import torch
from ase import Atoms, units

from fastmd.adapters import load_adapter
from fastmd.adapters.memory import MeasuredMemoryModel
from fastmd.adapters.negotiate import batch_systems, check_system_cells, negotiate, negotiated, size_unions
from fastmd.adapters.protocol import NegotiationError
from fastmd.config import InferenceConfig, resolve_config
from fastmd.engine.entry_config import release_adapter, use_adapter
from fastmd.execution.gpu_integrator import default_fix_com, kinetic_energy_fp64
from fastmd.execution.gpu_nhc import maxwell_boltzmann_momenta
from fastmd.engine.energy_derivative import engine_derivative
from fastmd.execution.replica_batch import (
    DEFAULT_UNION_ROWS,
    UNION_PROBE_FRACTIONS,
    graph_builder_options,
    union_capacity,
    union_demand,
    union_headrooms,
    union_self_check,
)
from fastmd.execution.system_batch import (
    SlotLayout,
    SlotWholeStepRunner,
    SystemNoseHooverChainNVT,
    SystemVelocityVerlet,
    slot_union_builder,
    slot_union_demand,
    slot_union_memory_runs,
    slot_union_persistent_bytes,
    systems_self_check,
    validate_cell,
)
from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder
from fastmd.interfaces.calculator import require_finite_geometry
from fastmd.interfaces.replica_md import _COUNTERS, _MEASURED_UNION, ENSEMBLES, _Union, _UnionMolecularDynamics


class _SlotUnion(_Union):
    """One union of systems ``[start, stop)`` in slots: its layout, the current systems' cells, species and
    atom counts, and its integrator, runner and pair capacity."""

    def __init__(self, start: int, stop: int, ceilings) -> None:
        super().__init__(start, stop)
        self.layout = SlotLayout(tuple(int(c) for c in ceilings))
        self.cells = None  # [systems, 3, 3] float64 on the device
        self.numbers = None  # [rows] int32 on the device (a vacant row: its system's first element)
        self.atoms: list[int] = []


class MultiSystemMolecularDynamics(_UnionMolecularDynamics):
    """S different systems advanced together (see the module docstring).

    Parameters
    ----------
    model:
        An adapter, or a name or checkpoint path that :func:`~fastmd.adapters.load_adapter` loads
        (``model_path``, ``model_dtype`` as there). A model that does not declare ``"union"``, does not take the
        engine's per-edge shifts or has a line graph (``check_system_cells``: MatRIS, ORB, CHGNet), or fails its
        check on two of the systems, raises :class:`~fastmd.adapters.protocol.NegotiationError`.
    systems:
        The systems (ASE atoms, periodic in all three directions, each with its own cell).
    momenta:
        One ``(n_s, 3)`` float64 array per system in ASE units; default: Maxwell-Boltzmann at ``temperature`` for
        system ``s`` with ``seed + s`` (ASE's ``MaxwellBoltzmannDistribution``).
    ensemble, temperature, timestep, taut, tdamp, tchain, tloop, fix_com:
        As in :class:`~fastmd.interfaces.replica_md.ReplicaMolecularDynamics`, per system (its own chain, degrees
        of freedom ``3 n_s`` and centre of mass).
    steps_per_launch:
        MD steps per captured window; ``run``, ``loginterval`` and the logged steps are whole windows.
    slot_atoms:
        The slots' atom ceilings: None (each system's own atom count), one int (at least that many for every
        slot) or one per system (at least its atom count). A slot's rows past its system are vacant: room for a
        larger system written in later (:meth:`write_system`).
    union_rows:
        The most rows (slot ceilings) per union; default ``DEFAULT_UNION_ROWS``. More systems run as several
        unions of consecutive systems with balanced rows, sharing one CUDA graph memory pool.
    capacity_headroom:
        The relative pair headroom a capacity growth adds over the overflowing demand (default: the model's, else
        ``UNION_GROWTH_HEADROOM``); a union's first capture holds the model's own capacity tiers.
    config, model_path, model_dtype, device, self_check:
        As in ``ReplicaMolecularDynamics`` (the self-check evaluates the smallest system and the smallest one in
        another cell, else the second smallest, as a union and alone, and the union with twice the cell as the
        model's ``cell`` argument; a set of one system checks two copies of it, without that probe).
    trajectory, logfile, loginterval:
        ``trajectory``: a file name with ``{system}`` (e.g. ``"md_{system:04d}.traj"``): one ASE trajectory per
        system, a frame every ``loginterval`` steps and at the start; ``logfile``: one line per system and
        logged step (time, energies, temperature).
    """

    _members = "systems"

    def __init__(self, model, systems: Sequence[Atoms], *, momenta=None, ensemble: str = "nvt_nhc",
                 temperature: float = 300.0, timestep: float = 1.0, taut: float = 100.0, tdamp: float | None = None,
                 tchain: int = 3, tloop: int = 1, fix_com: bool | None = None, seed: int = 0,
                 steps_per_launch: int = 10, slot_atoms=None, union_rows: int | None = None,
                 capacity_headroom: float | None = None, config: InferenceConfig | None = None,
                 model_path: str | None = None, model_dtype: str | None = None, device=None,
                 trajectory: str | None = None, logfile: str | None = None, loginterval: int = 0,
                 self_check: bool = True) -> None:
        where = "MultiSystemMolecularDynamics"
        ensemble = str(ensemble).lower()
        if ensemble not in ENSEMBLES:
            raise ValueError(f"{where}: ensemble must be one of {ENSEMBLES}, got {ensemble!r}")
        if ensemble == "nvt_nhc" and fix_com:
            raise ValueError(f"{where}: ensemble='nvt_nhc' has no centre-of-mass correction (ASE "
                             "NoseHooverChainNVT); fix_com must be None or False")
        for name, value, low in (("steps_per_launch", steps_per_launch, 1), ("loginterval", loginterval, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{where}: {name} must be an integer >= {low}")
        if loginterval % steps_per_launch:
            raise ValueError(f"{where}: loginterval={loginterval} must be a multiple of "
                             f"steps_per_launch={steps_per_launch} (a window's inner steps are not visible)")
        if trajectory is not None and "{system" not in str(trajectory):
            raise ValueError(f"{where}: trajectory must name each system's file with '{{system}}'")
        if (trajectory is not None or logfile is not None) and not loginterval:
            raise ValueError(f"{where}: trajectory and logfile write every loginterval steps: give loginterval > 0")
        finite = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if not finite(timestep) or timestep <= 0:
            raise ValueError(f"{where}: timestep must be finite and positive")
        if ensemble != "nve" and (not finite(temperature) or temperature <= 0):
            raise ValueError(f"{where}: ensemble={ensemble!r} needs a finite positive temperature")
        for name, value in (("taut", taut), ("tdamp", tdamp)):
            if value is not None and (not finite(value) or value <= 0):
                raise ValueError(f"{where}: {name} must be finite and positive")
        if capacity_headroom is not None and (not finite(capacity_headroom) or not 0.0 <= capacity_headroom <= 1.0):
            raise ValueError(f"{where}: capacity_headroom must be None or in [0, 1]")
        if union_rows is not None and (isinstance(union_rows, bool) or not isinstance(union_rows, int)
                                       or union_rows < 1):
            raise ValueError(f"{where}: union_rows must be a positive integer")
        systems = list(systems)
        if not systems or not all(isinstance(atoms, Atoms) and len(atoms) > 0 for atoms in systems):
            raise ValueError(f"{where}: systems must be one or more ASE Atoms with atoms")
        for index, atoms in enumerate(systems):
            self._check_system(atoms, f"{where}: system {index}")
        counts = [len(atoms) for atoms in systems]
        if momenta is None:
            momenta = [None] * len(systems)
        elif len(momenta) != len(systems):
            raise ValueError(f"{where}: momenta must hold one (n_s, 3) array per system")
        momenta = [None if value is None else np.asarray(value, dtype=np.float64) for value in momenta]
        for index, (value, n) in enumerate(zip(momenta, counts)):
            if value is not None and (value.shape != (n, 3) or not np.isfinite(value).all()):
                raise ValueError(f"{where}: momenta of system {index} must be finite with shape ({n}, 3)")
        self.ceilings = self._ceilings(slot_atoms, counts, where)
        self.systems = len(systems)
        self.ensemble = ensemble
        self.timestep = float(timestep)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.steps_per_launch = steps_per_launch
        self.loginterval = loginterval
        self._headroom = capacity_headroom
        self.device = torch.device(device if device is not None else "cuda")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError(f"{where} runs on a CUDA device")
        if self.device.index is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.set_device(self.device)
        resolved = resolve_config(config, default_execution="whole_step")
        if resolved.execution not in ("auto", "whole_step"):
            raise ValueError(f"{where} runs whole-step CUDA graphs: execution must be 'auto' or 'whole_step', got "
                             f"{resolved.execution!r}")
        self._adapter = load_adapter(model, model_path=model_path, device=str(self.device), model_dtype=model_dtype,
                                     optimization_profile=resolved.optimization_profile)
        use_adapter(self, model)  # an adapter loaded by name is closed with its last user (also on a failure below)
        if self._adapter is model or self._adapter.module is model:  # supplied, not loaded
            self._adapter.module.to(self.device)
        self._trajectories = False
        self._log = None
        try:
            self._setup(systems, momenta, resolved, taut=taut, tdamp=tdamp, tchain=tchain, tloop=tloop,
                        fix_com=fix_com, union_rows=union_rows, self_check=self_check)
            self._trajectory_name = trajectory
            if logfile is not None:
                self._log = open(logfile, "w")
                self._log.write(f"{'Time[ps]':>12} {'System':>8} {'Etot[eV]':>16} {'Epot[eV]':>16} "
                                f"{'Ekin[eV]':>14} {'T[K]':>10}\n")
        except BaseException:
            release_adapter(self)  # a failed constructor gives its adapter back now, not at its collection
            raise
        self.nsteps = 0
        self._logged_initial = False

    @staticmethod
    def _check_system(atoms: Atoms, where: str) -> np.ndarray:
        """Host checks of a system: periodic in all three directions, finite coordinates, a non-singular cell;
        its cell as float64."""
        if not bool(np.all(atoms.get_pbc())):
            raise ValueError(f"{where}: the cell must be periodic in all three directions")
        require_finite_geometry(atoms.positions, atoms.cell.array, where)
        return validate_cell(atoms.cell.array, f"{where}: the cell")

    @staticmethod
    def _ceilings(slot_atoms, counts: list[int], where: str) -> list[int]:
        if slot_atoms is None:
            return list(counts)
        if isinstance(slot_atoms, numbers.Integral) and not isinstance(slot_atoms, bool):
            return [max(n, int(slot_atoms)) for n in counts]
        ceilings = [int(c) for c in slot_atoms]
        if len(ceilings) != len(counts) or any(c < n for c, n in zip(ceilings, counts)):
            raise ValueError(f"{where}: slot_atoms must hold one ceiling per system, at least its atom count")
        return ceilings

    # ------------------------------------------------------------------ construction
    def _setup(self, systems, momenta, resolved, *, taut, tdamp, tchain, tloop, fix_com, union_rows,
               self_check) -> None:
        adapter = self._adapter
        check_system_cells(adapter)  # before anything is negotiated or measured
        if not negotiated(adapter):
            raise NegotiationError(f"model {adapter.name!r} is planned by the runtime's own rules, not batched here")
        plan = negotiate(adapter, replace(resolved, execution="whole_step"), entry="gpu_md", device=self.device)
        execution = plan.decision("execution")
        if execution.chosen != "whole_step":
            raise NegotiationError(f"MultiSystemMolecularDynamics needs a whole-step CUDA graph for model "
                                   f"{adapter.name!r}: {execution.reason}")
        # The plan's precision applies to every model call from here on (IEEE FP32 unless TF32 is asked for).
        self._config = replace(resolved, execution="whole_step", matmul_precision=plan.matmul_precision)
        self._float64 = adapter.topology_spec().model_dtype == "float64"
        if self._float64 and self.ensemble != "nvt_nhc":
            raise ValueError("MultiSystemMolecularDynamics: a float64 model needs ensemble='nvt_nhc' (the integrator "
                             "with a float64 state); nve and nvt keep a float32 state")
        self.candidate_skin = float(self._config.capacity.candidate_skin)
        self._capacity_step = adapter.capacity_step() if callable(getattr(adapter, "capacity_step", None)) else None
        self.capacity_headroom, self.triplet_headroom = union_headrooms(self._capacity_step, self._headroom)
        self._line_graph = False  # systems with cells of their own: edges only (check_system_cells)
        self.templates = [atoms.copy() for atoms in systems]
        self._writes = [0] * self.systems  # the systems written into each slot since construction
        self.masses = [np.asarray(atoms.get_masses(), dtype=np.float64) for atoms in systems]
        self._numbers = [np.asarray(atoms.get_atomic_numbers(), dtype=np.int32) for atoms in systems]
        self._cells = [np.asarray(atoms.cell.array, dtype=np.float64) for atoms in systems]
        positions = [np.asarray(atoms.get_positions(), dtype=np.float64) for atoms in systems]
        momenta = [maxwell_boltzmann_momenta(masses, self.temperature, seed=self.seed + index) if value is None
                   else value for index, (masses, value) in enumerate(zip(self.masses, momenta))]
        self._kinds = dict(taut=float(taut), tdamp=float(100.0 * self.timestep if tdamp is None else tdamp),
                           tchain=int(tchain), tloop=int(tloop),
                           fix_com=default_fix_com(self.ensemble) if fix_com is None else bool(fix_com))
        with self._config.scope():
            if self_check:
                floor, energy_floor = (1e-10, 1e-12) if self._float64 else (1e-5, 1e-6)
                check = self._self_check(floor, energy_floor)
                plan = batch_systems(plan, adapter, systems=self.systems, self_check=check, own_cells=True)
            else:
                plan = batch_systems(plan, adapter, systems=self.systems, own_cells=True)
            union_rows = DEFAULT_UNION_ROWS if union_rows is None else union_rows
            union_rows, plan = self._size_unions(plan, union_rows)
            bounds = self._partition(union_rows)
            self._unions = [_SlotUnion(start, stop, self.ceilings[start:stop])
                            for start, stop in zip(bounds[:-1], bounds[1:])]
            # The decision records the partition the systems run in (rows per union).
            plan.decisions = [replace(decision, evidence={**decision.evidence, "unions": [
                union.layout.rows for union in self._unions]}) if decision.id == "union_size" else decision
                for decision in plan.decisions]
            self._plan = plan
            # One CUDA graph memory pool for every union's captures: they replay one after another.
            self._graph_pool = torch.cuda.graph_pool_handle()
            for union in self._unions:
                self._start_union(union, positions[union.start:union.stop], momenta[union.start:union.stop])

    def _self_check(self, floor: float, energy_floor: float):
        """The negotiation's check of the declaration: the smallest system and the smallest one in another cell
        (else the second smallest) as one union and alone (two copies of the system for a set of one)."""
        geometry = dict(candidate_skin=self.candidate_skin, device=self.device)
        if self.systems == 1:
            numbers = torch.as_tensor(self._numbers[0], device=self.device)
            cell = torch.as_tensor(self._cells[0], dtype=torch.float64, device=self.device)
            pbc = torch.ones(3, dtype=torch.bool, device=self.device)
            positions = torch.as_tensor(self.templates[0].get_positions(), dtype=torch.float32, device=self.device)
            return lambda: union_self_check(self._adapter, positions=positions, cell=cell, atomic_numbers=numbers,
                                            pbc=pbc, absolute_floor=floor, energy_floor=energy_floor, **geometry)
        order = sorted(range(self.systems), key=lambda index: (len(self.templates[index]), index))
        other = next((index for index in order[1:] if not np.array_equal(self._cells[index], self._cells[order[0]])),
                     order[1])
        smallest = [order[0], other]
        pair = [(self.templates[index].get_positions(), self._cells[index], self._numbers[index])
                for index in smallest]
        return lambda: {**systems_self_check(self._adapter, pair, absolute_floor=floor, energy_floor=energy_floor,
                                             **geometry), "system_indices": smallest}

    def _size_unions(self, plan, union_rows: int):
        """The rows per union the measured memory admits, and the plan with that decision ("union_size"):
        ``union_rows``, or less where the largest union of that partition would not fit (``size_unions``, in rows).
        The memory of a union is measured on sub-unions of the largest one (its first systems, at
        ``UNION_PROBE_FRACTIONS`` of them, eager and captured, peak reserved bytes), fitted over their edge
        capacities and predicted at the union's capacity after one growth, plus the other unions' buffers outside
        the shared pool; a union of fewer than ``_MEASURED_UNION`` systems is not measured."""
        bounds = self._partition(union_rows)
        rows = [sum(self.ceilings[start:stop]) for start, stop in zip(bounds[:-1], bounds[1:])]
        largest = int(np.argmax(rows))
        start, stop = int(bounds[largest]), int(bounds[largest + 1])
        requested, total = int(rows[largest]), int(sum(self.ceilings))
        systems = [(self.templates[index].get_positions(), self._cells[index], self._numbers[index])
                   for index in range(start, stop)]
        runs, pairs = slot_union_memory_runs(self._adapter, systems, candidate_skin=self.candidate_skin,
                                             device=self.device, step=self._capacity_step,
                                             headroom=self.capacity_headroom, ceilings=self.ceilings[start:stop])
        occupied = sum(len(numbers) for _, _, numbers in systems)

        def union_edges(size: int) -> int:  # a union of `size` rows, at the largest union's pairs per row
            return 2 * union_capacity(math.ceil(pairs(stop - start) * size / requested), self._capacity_step,
                                      headroom=self.capacity_headroom)

        memory, persistent = None, 0
        if stop - start >= _MEASURED_UNION:
            probes = sorted({max(1, int(round(fraction * (stop - start)))) for fraction in UNION_PROBE_FRACTIONS})
            try:
                memory = MeasuredMemoryModel.measure(runs, probes, self.device, triplets=False, metric="reserved")
            except torch.OutOfMemoryError as error:
                why = f"ran out of CUDA memory ({str(error).splitlines()[0][:160]})"
            except RuntimeError as error:
                if "capture" not in str(error).lower():
                    raise
                why = f"failed in a capture ({str(error).splitlines()[0][:160]})"
            # Outside the except blocks: a traceback would pin the failed run's graph and buffers.
            gc.collect()
            torch.cuda.empty_cache()
            if memory is None:
                memory = MeasuredMemoryModel.failed(f"the union memory measurement {why}", triplets=False)
            else:  # the other unions' buffers, should the sizing leave more than one
                layout = SlotLayout(tuple(self.ceilings[start:start + probes[0]]))
                numbers = np.concatenate([np.full(ceiling, z[0], dtype=np.int32) for ceiling, (_, _, z) in
                                          zip(layout.ceilings, systems)])
                builder = slot_union_builder(
                    self._adapter, layout=layout,
                    cells=torch.as_tensor(np.stack([cell for _, cell, _ in systems[:probes[0]]])),
                    atomic_numbers=torch.as_tensor(numbers), atoms=[len(z) for _, _, z in systems[:probes[0]]],
                    u_capacity=union_capacity(pairs(probes[0])), candidate_skin=self.candidate_skin,
                    device=self.device)
                persistent = slot_union_persistent_bytes(builder)
                del builder
                torch.cuda.empty_cache()
        chosen, plan = size_unions(plan, requested=requested, memory=memory, union_edges=union_edges,
                                   device=self.device, replicas=total, persistent_per_replica=persistent, unit="rows")
        plan.decisions[-1].evidence.update(occupied_rows=occupied, largest_union_systems=stop - start,
                                           largest_slot_rows=max(self.ceilings))
        if chosen >= requested:
            return union_rows, plan
        # A slot is the smallest union there is (its system alone): a slot beyond the admitted rows runs alone where
        # the measured memory admits a union of its own size.
        index = int(np.argmax(self.ceilings))
        if self.ceilings[index] > chosen:
            alone, _ = size_unions(plan, requested=self.ceilings[index], memory=memory, union_edges=union_edges,
                                   device=self.device, replicas=total, persistent_per_replica=persistent, unit="rows")
            if alone < self.ceilings[index]:
                raise NegotiationError(f"system {index}'s slot of {self.ceilings[index]} rows does not fit the "
                                       f"measured memory (unions of {chosen} rows do): run it with fewer systems or a "
                                       "smaller slot_atoms")
        return chosen, plan

    def _partition(self, union_rows: int) -> np.ndarray:
        """The union bounds: consecutive systems in as few unions of at most ``union_rows`` rows as the systems in
        order allow (a slot larger than that alone in its union), with balanced rows (the smallest limit that needs
        no more unions)."""
        def pack(limit: int) -> list[int]:
            bounds, rows = [0], 0
            for index, ceiling in enumerate(self.ceilings):
                if rows and rows + ceiling > limit:
                    bounds.append(index)
                    rows = 0
                rows += ceiling
            return bounds + [self.systems]

        count = len(pack(int(union_rows))) - 1
        low, high = 1, int(union_rows)
        while low < high:  # the number of unions only falls as the limit grows
            middle = (low + high) // 2
            low, high = (low, middle) if len(pack(middle)) - 1 <= count else (middle + 1, high)
        return np.asarray(pack(low))

    def _union_tables(self, union: _SlotUnion, positions: list) -> tuple[torch.Tensor, torch.Tensor]:
        """The union's row tables at ``positions`` (one ``(n, 3)`` per system): its species (a vacant row: its
        system's first element) and float32 coordinates (a vacant row: the origin)."""
        layout = union.layout
        numbers = np.zeros(layout.rows, dtype=np.int32)
        coordinates = np.zeros((layout.rows, 3), dtype=np.float64)
        for slot, index in enumerate(range(union.start, union.stop)):
            start, n = layout.starts[slot], len(self._numbers[index])
            numbers[start:start + layout.ceilings[slot]] = self._numbers[index][0]
            numbers[start:start + n] = self._numbers[index]
            coordinates[start:start + n] = positions[slot]
        return (torch.as_tensor(numbers, device=self.device),
                torch.as_tensor(coordinates, dtype=torch.float32, device=self.device))

    def _start_union(self, union: _SlotUnion, positions: list, momenta: list) -> None:
        """The integrator of ``union`` at ``positions`` and ``momenta`` (one ``(n, 3)`` per system), and its first
        runner."""
        kinds = self._kinds
        union.atoms = [len(self._numbers[index]) for index in range(union.start, union.stop)]
        union.cells = torch.as_tensor(np.stack(self._cells[union.start:union.stop]), dtype=torch.float64,
                                      device=self.device)
        union.numbers, coordinates = self._union_tables(union, positions)
        layout = union.layout
        if self.ensemble == "nvt_nhc":
            union.integrator = SystemNoseHooverChainNVT(
                layout, self.masses[union.start:union.stop], positions, momenta,
                temperature_K=self.temperature, timestep=self.timestep * units.fs, tdamp=kinds["tdamp"] * units.fs,
                tchain=kinds["tchain"], tloop=kinds["tloop"], device=self.device,
                force_dtype=torch.float64 if self._float64 else torch.float32)
            state = union.integrator.positions.to(torch.float32), union.integrator.momenta.to(torch.float32)
        else:
            masses = np.ones(layout.rows, dtype=np.float64)
            moment = np.zeros((layout.rows, 3), dtype=np.float64)
            for slot, index in enumerate(range(union.start, union.stop)):
                start, n = layout.starts[slot], union.atoms[slot]
                masses[start:start + n] = self.masses[index]
                moment[start:start + n] = momenta[slot]
            occupied = torch.zeros(layout.rows, dtype=torch.bool, device=self.device)
            occupied.index_fill_(0, torch.as_tensor(layout.occupied_rows(union.atoms), device=self.device), True)
            union.integrator = SystemVelocityVerlet(
                torch.as_tensor(masses, dtype=torch.float32, device=self.device), self.timestep,
                ceilings=torch.tensor(layout.ceilings, dtype=torch.int64, device=self.device),
                atoms=torch.tensor(union.atoms, dtype=torch.int64, device=self.device),
                system_of_row=layout.system_of_row(self.device), occupied=occupied, ensemble=self.ensemble,
                temperature=self.temperature, taut_fs=kinds["taut"], fix_com=kinds["fix_com"])
            state = coordinates, torch.as_tensor(moment, dtype=torch.float32, device=self.device)
        pairs = slot_union_demand(self._adapter, state[0], layout=layout, cells=union.cells,
                                  atomic_numbers=union.numbers, atoms=union.atoms, candidate_skin=self.candidate_skin,
                                  device=self.device)
        union.u_capacity = union_capacity(pairs, self._capacity_step)  # the model's tiers
        self._capture(union, *state, forces=None)

    def _capture(self, union: _SlotUnion, positions, momenta, *, forces) -> None:
        """A runner of ``union`` at its capacity (a recapture when ``forces``, the cached F of the state, are
        given)."""
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        builder = slot_union_builder(self._adapter, layout=union.layout, cells=union.cells,
                                     atomic_numbers=union.numbers, atoms=union.atoms, u_capacity=union.u_capacity,
                                     candidate_skin=self.candidate_skin, device=self.device)
        union.runner = SlotWholeStepRunner(
            model=self._adapter, builder=builder, integrator=union.integrator, positions=positions, momenta=momenta,
            forces=forces, steps_per_launch=self.steps_per_launch, warmup=self._config.capacity.warmup,
            graph_pool=self._graph_pool, config=self._config, recapture=forces is not None,
            validate_isolated_atoms=not self._config.handle_isolated_atoms)
        torch.cuda.synchronize(self.device)
        union.capture_seconds.append(time.perf_counter() - started)

    # ------------------------------------------------------------------ a system into a slot
    def _locate(self, index: int) -> tuple[_SlotUnion, int]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < self.systems:
            raise IndexError(f"system index {index!r} out of range [0, {self.systems})")
        union = next(union for union in self._unions if union.start <= index < union.stop)
        return union, int(index) - union.start

    @torch.no_grad()
    def write_system(self, index: int, atoms: Atoms, *, momenta=None) -> None:
        """Put ``atoms`` into system ``index``'s slot in place: its positions, species, cell and atom count (at
        most the slot's ceiling), momenta (default: Maxwell-Boltzmann at the temperature with ``seed + index``), a
        fresh thermostat, and its forces and energy evaluated as a fresh single-system run starts (the system
        alone). Checks first, before anything changes: periodic in all three directions, finite coordinates, a
        non-singular cell the union can hold, at most the ceiling's atoms, and the model's own input check (its
        elements, in its ``prepare``) and the candidate capacity of the system alone. No recapture unless the
        union's edges then outgrow its capacity (an overflow at the next window grows it, as in ``run``). Its next
        windows run as a fresh single-system run would from this state; the other systems keep theirs. Between
        ``run`` calls only."""
        union, slot = self._locate(index)
        where = f"MultiSystemMolecularDynamics.write_system({index})"
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            raise ValueError(f"{where}: atoms must be ASE Atoms with atoms")
        cell = self._check_system(atoms, where)
        n, ceiling = len(atoms), union.layout.ceilings[slot]
        if n > ceiling:
            raise ValueError(f"{where}: {n} atoms exceed the slot's ceiling of {ceiling} (slot_atoms)")
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
        masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if momenta is None:
            momenta = maxwell_boltzmann_momenta(masses, self.temperature, seed=self.seed + int(index))
        momenta = np.asarray(momenta, dtype=np.float64)
        if momenta.shape != (n, 3) or not np.isfinite(momenta).all():
            raise ValueError(f"{where}: momenta must be finite with shape ({n}, 3)")
        runner, builder = union.runner, union.runner.builder
        try:
            cell64 = builder.check_cell(cell)
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from error
        with self._config.scope():
            forces, energy = self._evaluate_alone(numbers, positions, cell, runner.validate_isolated_atoms, where)
        # Commit: nothing below rejects the system.
        builder.write_system(slot, atomic_numbers=torch.as_tensor(numbers), cell=cell64, fill_number=int(numbers[0]))
        self._numbers[index], self._cells[index], self.masses[index] = numbers, cell, masses
        self.templates[index] = atoms.copy()
        self._writes[index] += 1
        union.atoms[slot] = n
        union.cells[slot].copy_(cell64)
        union.numbers.copy_(builder.atomic_numbers_real)
        start = union.layout.starts[slot]
        rows = slice(start, start + ceiling)
        integrator = union.integrator
        if self.ensemble == "nvt_nhc":
            integrator.write_system(slot, masses=masses, positions=positions, momenta=momenta)
            runner.positions.copy_(integrator.positions)
            runner.momenta.copy_(integrator.momenta)
        else:
            integrator.masses[rows].fill_(1.0)
            integrator.masses[start:start + n].copy_(torch.as_tensor(masses[:, None]))
            integrator.atoms[slot] = n
            integrator.occupied.copy_(builder.occupied)
            for buffer, values in ((runner.positions, positions), (runner.momenta, momenta)):
                buffer[rows].zero_()
                buffer[start:start + n].copy_(torch.as_tensor(values))
        runner.forces[rows].zero_()
        runner.forces[start:start + n].copy_(forces)
        runner.potential_energy[slot].copy_(energy)
        if self.ensemble == "nvt_nhc":
            integrator.set_forces(runner.forces, energy=runner.potential_energy)
            integrator.backup()
        runner.reset(positions=runner.positions.clone(), momenta=runner.momenta.clone(), forces=runner.forces.clone(),
                     potential_energy=runner.potential_energy.clone(), reset_counters=False)

    def _evaluate_alone(self, numbers: np.ndarray, positions: np.ndarray, cell: np.ndarray, isolated: bool,
                        where: str) -> tuple[torch.Tensor, torch.Tensor]:
        """F(q0) ``(n, 3)`` (the model's dtype) and E(q0) (FP64) of one system alone, as a fresh single-system run
        starts: a fixed-capacity build at its own demand (the adapter's ``prepare`` checks its input) and the
        engine-owned derivative; ValueError where the model rejects the system, its candidates overflow, or it has
        isolated atoms that ``isolated`` (the union's check) rejects."""
        adapter, device = self._adapter, self.device
        cutoff = float(adapter.topology_spec().cutoff)
        geometry = dict(cell=torch.as_tensor(cell, dtype=torch.float64, device=device),
                        atomic_numbers=torch.as_tensor(numbers, device=device),
                        pbc=torch.ones(3, dtype=torch.bool, device=device), device=device)
        coordinates = torch.as_tensor(positions, dtype=torch.float64 if self._float64 else torch.float32,
                                      device=device)
        try:
            pairs, _ = union_demand(adapter, coordinates.float(), replicas=1, candidate_skin=self.candidate_skin,
                                    **geometry)
            builder = FixedCapacityGraphBuilder(
                u_capacity=union_capacity(pairs), atom_graph_cutoff=cutoff,
                candidate_cutoff=cutoff + self.candidate_skin, padding=adapter.padding_rule(),
                prepare=adapter.prepare, **graph_builder_options(adapter), **geometry)
        except ValueError as error:
            raise ValueError(f"{where}: the model rejects the system: {error}") from error
        builder.refresh_candidates(coordinates)
        try:
            builder.assert_candidate_capacity()
        except Exception as error:  # nvalchemiops' neighbour overflow, as a rejected system
            raise ValueError(f"{where}: {error}") from error
        graph, status = builder.build(coordinates, rebuild_candidates=False)
        if isolated and int(status.isolated_atom_count.item()):
            raise ValueError(f"{where}: {int(status.isolated_atom_count.item())} atoms without a neighbour within "
                             f"the cutoff {cutoff} Å")
        out = engine_derivative(adapter)(builder.batch, graph, task="ef", n_real=len(numbers))
        return out["f"][0][:len(numbers)].detach(), out["e"].reshape(-1)[0]

    # ------------------------------------------------------------------ state and observables
    def _system_rows(self, read) -> list[np.ndarray]:
        """One ``(n_s, 3)`` float64 array per system of the per-row tensor ``read(union)`` (``(rows, 3)``)."""
        out = []
        for union in self._unions:
            values = read(union).to(torch.float64).cpu().numpy()
            out.extend(values[start:start + n] for start, n in zip(union.layout.starts, union.atoms))
        return out

    def get_positions(self) -> list[np.ndarray]:
        """One ``(n_s, 3)`` float64 array of unwrapped positions per system (the MD state: the Nose-Hoover chain's
        float64 positions, or the runner's float32 ones for nve and nvt)."""
        return self._system_rows(lambda union: union.integrator.positions if self.ensemble == "nvt_nhc"
                                 else union.runner.positions)

    def get_momenta(self) -> list[np.ndarray]:
        """One ``(n_s, 3)`` float64 array of momenta per system."""
        return self._system_rows(lambda union: union.integrator.momenta if self.ensemble == "nvt_nhc"
                                 else union.runner.momenta)

    def get_forces(self) -> list[np.ndarray]:
        """One ``(n_s, 3)`` float64 array of forces per system at the current positions."""
        return self._system_rows(lambda union: union.integrator.forces if self.ensemble == "nvt_nhc"
                                 else union.runner.forces)

    def kinetic_energies(self) -> torch.Tensor:
        """``(S,)`` float64 kinetic energies (eV)."""
        momenta, masses = self.get_momenta(), self.masses
        return torch.tensor([float(kinetic_energy_fp64(torch.as_tensor(p), torch.as_tensor(m)))
                             for p, m in zip(momenta, masses)], dtype=torch.float64)

    def temperatures(self) -> torch.Tensor:
        """``(S,)`` temperatures (K): each system's ``Atoms.get_temperature()``."""
        atoms = torch.tensor([len(m) for m in self.masses], dtype=torch.float64)
        return 2.0 * self.kinetic_energies() / (3 * atoms * units.kB)

    def potential_energies(self) -> torch.Tensor:
        """``(S,)`` float64 potential energies (eV)."""
        return self._per_union(lambda union: (union.integrator.potential_energy if self.ensemble == "nvt_nhc"
                                              else union.runner.potential_energy).to(torch.float64)).cpu()

    def conserved_energies(self) -> torch.Tensor:
        """``(S,)``: the Nose-Hoover chain's conserved energy (``nvt_nhc``), else potential plus kinetic."""
        if self.ensemble == "nvt_nhc":
            return self._per_union(lambda union: union.integrator.conserved_energy()).cpu()
        return self.potential_energies() + self.kinetic_energies()

    def health(self) -> np.ndarray:
        """``(S,)`` bool: the systems whose state is finite (a window that leaves one non-finite raises
        :class:`~fastmd.execution.replica_batch.ReplicaHealthError` naming them)."""
        return torch.cat([union.runner.health() for union in self._unions]).numpy()

    def get_atoms(self, index: int) -> Atoms:
        """System ``index`` as ASE atoms with its positions and momenta."""
        atoms = self.templates[index].copy()
        atoms.set_positions(self.get_positions()[index])
        atoms.set_momenta(self.get_momenta()[index])
        return atoms

    def snapshot(self) -> dict:
        """The systems' MD state (device copies), for :meth:`restore` (the systems in the slots stay)."""
        unions = []
        for union in self._unions:
            runner = union.runner
            unions.append({"integrator": union.integrator.snapshot() if self.ensemble == "nvt_nhc" else None,
                           "atoms": list(union.atoms),
                           "state": [tensor.clone() for tensor in (runner.positions, runner.momenta, runner.forces,
                                                                   runner.potential_energy)]})
        return {"nsteps": self.nsteps, "layout": [(union.start, union.stop) for union in self._unions],
                "ensemble": self.ensemble, "writes": list(self._writes), "unions": unions}

    @torch.no_grad()
    def restore(self, snapshot: dict, *, momenta=None) -> None:
        """Back to ``snapshot`` (:meth:`snapshot` of this object, taken with the same systems in the slots: none
        written since), with new momenta if given (one ``(n_s, 3)`` per system; positions, forces and energies stay
        the snapshot's): the captured graphs and the grown capacities are kept; the window counters of
        :meth:`stats` start again."""
        if (snapshot.get("layout") != [(union.start, union.stop) for union in self._unions]
                or snapshot.get("ensemble") != self.ensemble or snapshot.get("writes") != self._writes):
            raise ValueError("MultiSystemMolecularDynamics.restore: the snapshot is not one of this object's states "
                             "with the systems it holds now")
        if momenta is not None:
            if len(momenta) != self.systems:
                raise ValueError("MultiSystemMolecularDynamics.restore: momenta must hold one (n_s, 3) array per "
                                 "system")
            momenta = [np.asarray(value, dtype=np.float64) for value in momenta]
            for index, (value, masses) in enumerate(zip(momenta, self.masses)):
                if value.shape != (len(masses), 3) or not np.isfinite(value).all():
                    raise ValueError(f"MultiSystemMolecularDynamics.restore: momenta of system {index} must be "
                                     f"finite with shape ({len(masses)}, 3)")
        with self._config.scope():
            for union, saved in zip(self._unions, snapshot["unions"]):
                positions, moment, forces, energies = saved["state"]
                if self.ensemble == "nvt_nhc":
                    integrator = union.integrator
                    integrator.load_snapshot(saved["integrator"])
                    if momenta is not None:
                        for slot, index in enumerate(range(union.start, union.stop)):
                            integrator.system_view(integrator.state.p, slot).copy_(
                                torch.as_tensor(np.asarray(momenta[index], dtype=np.float64)))
                    integrator.backup()
                    positions, moment = integrator.positions.to(torch.float32), integrator.momenta.to(torch.float32)
                    forces, energies = integrator.forces.to(union.runner.forces.dtype), integrator.potential_energy
                elif momenta is not None:
                    moment = moment.clone()
                    for slot, index in enumerate(range(union.start, union.stop)):
                        start, n = union.layout.starts[slot], union.atoms[slot]
                        moment[start:start + n].copy_(torch.as_tensor(np.asarray(momenta[index])))
                union.runner.reset(positions=positions, momenta=moment, forces=forces, potential_energy=energies)
                union.accumulated = dict.fromkeys(_COUNTERS, 0)
                union.max_edge_count = union.max_triplet_count = 0
        self.nsteps = snapshot["nsteps"]

    def stats(self) -> dict:
        """Counters over every union and every capture of it since construction or the last :meth:`restore`:
        windows launched, candidate refreshes, fallback windows, recovery replays and the largest edge count (with
        the capacities and their fill); since construction: recaptures (capacity growth) and capture times (s);
        the slots' ceilings and atoms."""
        return {**self._union_stats(systems=self.systems),
                "slot_atoms": [list(union.layout.ceilings) for union in self._unions],
                "atoms": [list(union.atoms) for union in self._unions], "writes": list(self._writes)}

    # ------------------------------------------------------------------ output
    def _write_log(self) -> None:
        if self._log is None and self._trajectory_name is None:
            return
        positions, momenta, forces = self.get_positions(), self.get_momenta(), self.get_forces()
        potential = self.potential_energies().numpy()
        kinetic = self.kinetic_energies().numpy()
        if self._trajectory_name is not None:
            from ase.calculators.singlepoint import SinglePointCalculator
            from ase.io import Trajectory

            # One file open at a time; the first logged step creates every system's file, later ones append.
            mode = "a" if self._trajectories else "w"
            self._trajectories = True
            for s in range(self.systems):
                atoms = self.templates[s].copy()
                atoms.set_positions(positions[s])
                atoms.set_momenta(momenta[s])
                atoms.info["system_writes"] = self._writes[s]  # a write into the slot starts another system
                atoms.calc = SinglePointCalculator(atoms, energy=float(potential[s]), forces=forces[s])
                with Trajectory(self._trajectory_name.format(system=s), mode) as writer:
                    writer.write(atoms)
        if self._log is not None:
            time_ps = self.nsteps * self.timestep / 1000.0
            temperatures = self.temperatures().numpy()
            for s in range(self.systems):
                self._log.write(f"{time_ps:12.4f} {s:8d} {potential[s] + kinetic[s]:16.6f} {potential[s]:16.6f} "
                                f"{kinetic[s]:14.6f} {temperatures[s]:10.2f}\n")
            self._log.flush()
