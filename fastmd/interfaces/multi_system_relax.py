"""Structure relaxation of many systems in the slots of whole-step CUDA graphs: ASE's FIRE per system, with a
fixed cell or ASE's FrechetCellFilter, the structures streaming through the slots.

:class:`MultiSystemRelaxation` relaxes each of a list of structures as ASE would one at a time
(``FIRE(atoms).run(fmax, steps)``, or ``FIRE(FrechetCellFilter(atoms))`` for a variable cell; ASE's default
parameters), for a model the engine differentiates that declares its systems may be evaluated as one disjoint union
and takes the engine's per-edge shifts (MACE's ``ScaleShiftMACE`` and SevenNet, as
:class:`~fastmd.interfaces.multi_system_md.MultiSystemMolecularDynamics`). Up to ``slots`` structures relax at a time
in the slots of unions (:mod:`fastmd.execution.system_relax`): one graph build, one model call and one backward pass
per FIRE step for all the structures of a union, ``steps_per_launch`` steps captured as one CUDA graph, the
optimizer's state and the convergence of every system on the device. Each window ends with one read of every system's
phase; a structure that converged, reached the step limit or failed leaves its slot, and the next structure that fits
is written into the slot in place, without a recapture (in-flight batching). The results come back per input
structure.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import gc
import math
import time

import numpy as np
import torch
from ase import Atoms

from fastmd.execution.replica_batch import DEFAULT_UNION_ROWS, union_capacity
from fastmd.execution.system_batch import slot_union_demand
from fastmd.execution.system_relax import (
    ACTIVE,
    FINISHED,
    IDLE,
    PENDING,
    PHASES,
    FIREParameters,
    SlotRelaxationRunner,
    SystemFIRE,
    relaxation_builder,
)
from fastmd.interfaces.multi_system_md import MultiSystemMolecularDynamics, _SlotUnion

# The ratio of consecutive slot size classes: a structure sits in the smallest class that holds it, so at most a fifth
# of a slot's rows are vacant while its own class has structures left (with fewer slots than classes on this grid, the
# ratio doubles until each class has a slot, and so does the vacancy bound).
SLOT_CLASS_RATIO = 1.25
# Draining: with no structure left to write, a union whose relaxing systems hold at most this fraction of its rows may
# be captured again with only them (finished systems are evaluated with the others at every step until then): the
# slowest structures of a set take several times the median's steps (WBM, MACE-MPA-0, a variable cell: median 37,
# largest 914), and a step of a small union costs far less than one of the full union. It is captured again when that
# pays back DRAIN_PAYBACK times the estimated time of the capture: the relaxing systems taking as many more steps as
# the median of them has taken (their remaining steps are unknown), each step saving the measured step time times the
# fraction of the rows that the finished systems hold; a short tail (a fixed cell: largest 133 steps of 256
# structures) is seldom worth a capture. The decision reads measured times, so whether and when a union drains can
# differ from run to run (and with it the rounding of the systems that move: ``drain=False`` keeps the unions).
DRAIN_FRACTION = 0.25
DRAIN_PAYBACK = 2.0
# ... unless it has no more rows than this: a capture again costs one warmup window and the capture (MACE-MPA-0, 5-step
# windows: about 1.1 s), which a union of fewer rows, its steps near their launch cost, seldom saves back.
DRAIN_MIN_ROWS = 2048
# A variable cell may shrink while it relaxes: the union's sort keys hold the queued cells compressed by this factor.
CELL_FILTER_COMPRESSION = 0.8


@dataclass
class RelaxationResult:
    """One structure's relaxation: ``atoms``, the final structure (positions, and its cell with a cell filter) with
    a ``SinglePointCalculator`` of its energy, forces (and stress); ``fmax`` the optimizer's largest force row (with
    a cell filter the filter's, its cell rows included: ASE's convergence criterion); ``steps`` its FIRE steps
    (ASE's ``nsteps``); ``status`` "converged", "max_steps" or "failed" (a non-finite energy or force: the positions
    of the step that gave it, the non-finite values, ``fmax`` inf)."""

    index: int
    atoms: Atoms
    energy: float
    forces: np.ndarray
    stress: np.ndarray | None
    fmax: float
    steps: int
    status: str

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def plan_slots(sizes: Sequence[int], slots: int) -> tuple[list[int], list[int], list[list[int]]]:
    """``(ceilings, first, queues)`` for ``slots`` concurrent structures of atom counts ``sizes``: every structure
    when there are no more than ``slots`` (each slot its own structure's size); otherwise size classes on a
    geometric grid of ratio ``SLOT_CLASS_RATIO`` (each structure in the smallest class that holds it, each class's
    ceiling its largest structure; the ratio doubled while there are more classes than slots), the slots shared
    among the classes in proportion to their atoms (one each first, then each next slot to the class with the most
    atoms per slot, never more slots than structures). ``first``: the structure first in each slot; ``queues``: the
    rest of each class's structures in input order, one queue per class, smallest class first."""
    sizes = [int(n) for n in sizes]
    if len(sizes) <= slots:
        return list(sizes), list(range(len(sizes))), []
    ratio = SLOT_CLASS_RATIO
    while True:
        grid, value = [], min(sizes)
        while value < max(sizes) and ratio * min(sizes) < max(sizes):
            grid.append(value)
            value = max(value + 1, math.ceil(value * ratio))
        grid.append(max(value, max(sizes)))  # one class holds them all once the ratio spans the sizes
        members: dict[int, list[int]] = {}
        for index, n in enumerate(sizes):
            members.setdefault(next(c for c in grid if c >= n), []).append(index)
        if len(members) <= slots:
            break
        ratio *= 2.0
    classes = sorted(members)
    work = [sum(sizes[i] for i in members[c]) for c in classes]
    share = [1] * len(classes)
    for _ in range(slots - len(classes)):  # each next slot to the class with the most atoms per slot
        open_classes = [k for k, c in enumerate(classes) if share[k] < len(members[c])]
        if not open_classes:
            break
        share[max(open_classes, key=lambda k: (work[k] / share[k], -k))] += 1
    ceilings, first, queues = [], [], []
    for k, c in enumerate(classes):
        queue = list(members[c])
        count = share[k]
        ceiling = max(sizes[i] for i in queue)
        ceilings += [ceiling] * count
        first += queue[:count]
        queues.append(queue[count:])
    return ceilings, first, queues


class _SlotRelaxationEngine(MultiSystemMolecularDynamics):
    """The slot unions of :class:`~fastmd.interfaces.multi_system_md.MultiSystemMolecularDynamics` (negotiation,
    self-check, measured memory, unions sharing a graph pool, captures and capacity growth) with
    :class:`~fastmd.execution.system_relax.SystemFIRE` as each union's integrator and
    :class:`~fastmd.execution.system_relax.SlotRelaxationRunner` as its runner. Private: the relaxation's driver."""

    def __init__(self, model, systems, *, relaxation: dict, admit_cells, **options) -> None:
        self._relaxation = relaxation
        self._admit_cells = admit_cells
        # The MD constructor's thermostat arguments are unused: the integrator is FIRE (its float64 state, as the
        # Nose-Hoover chain's, takes float64 models too).
        super().__init__(model, systems, ensemble="nvt_nhc", **options)

    def _start_union(self, union, positions: list, momenta: list) -> None:
        """The FIRE integrator of ``union`` with its systems pending at ``positions``, and its first runner."""
        union.atoms = [len(self._numbers[index]) for index in range(union.start, union.stop)]
        union.cells = torch.as_tensor(np.stack(self._cells[union.start:union.stop]), dtype=torch.float64,
                                      device=self.device)
        union.numbers, coordinates = self._union_tables(union, positions)
        integrator = SystemFIRE(union.layout, device=self.device, **self._relaxation)
        for slot, index in enumerate(range(union.start, union.stop)):
            integrator.write_system(slot, positions=positions[slot], cell=self._cells[index])
        integrator.sync()
        union.integrator = integrator
        pairs = slot_union_demand(self._adapter, coordinates, layout=union.layout, cells=union.cells,
                                  atomic_numbers=union.numbers, atoms=union.atoms, candidate_skin=self.candidate_skin,
                                  device=self.device)
        union.u_capacity = union_capacity(pairs, self._capacity_step)  # the model's tiers
        self._capture(union, coordinates, integrator.momenta.to(torch.float32), forces=None)

    def _capture(self, union, positions, momenta, *, forces) -> None:
        """A runner of ``union`` at its capacity (a recapture when ``forces``, the cached forces of the state, are
        given), its builder in the integrator's current cells."""
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        integrator = union.integrator
        builder = relaxation_builder(
            self._adapter, cell_filter=integrator.cell_filter, layout=union.layout,
            cells=integrator.cells if integrator.cell_filter else union.cells, atomic_numbers=union.numbers,
            atoms=union.atoms, u_capacity=union.u_capacity, candidate_skin=self.candidate_skin, device=self.device,
            admit_cells=self._admit_cells)
        union.runner = SlotRelaxationRunner(
            model=self._adapter, builder=builder, integrator=integrator, positions=positions, momenta=momenta,
            forces=forces, steps_per_launch=self.steps_per_launch, warmup=self._config.capacity.warmup,
            graph_pool=self._graph_pool, config=self._config, recapture=forces is not None,
            validate_isolated_atoms=not self._config.handle_isolated_atoms)
        torch.cuda.synchronize(self.device)
        union.capture_seconds.append(time.perf_counter() - started)

    @torch.no_grad()
    def write_structure(self, union, slot: int, atoms: Atoms) -> None:
        """``atoms`` (checked at construction) into ``slot`` of ``union`` in place: its species, cell and atom count
        in the builder, a pending system with a fresh optimizer in the integrator. The caller then calls
        :meth:`commit_writes`."""
        builder = union.runner.builder
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        cell64 = builder.check_cell(cell)
        builder.write_system(slot, atomic_numbers=torch.as_tensor(numbers), cell=cell64, fill_number=int(numbers[0]))
        union.integrator.write_system(slot, positions=atoms.get_positions(), cell=cell)
        index = union.start + slot
        self._numbers[index], self._cells[index] = numbers, cell
        self.masses[index] = np.asarray(atoms.get_masses(), dtype=np.float64)
        self.templates[index] = atoms.copy()
        self._writes[index] += 1
        union.atoms[slot] = len(numbers)
        union.cells[slot].copy_(cell64)
        union.numbers.copy_(builder.atomic_numbers_real)

    @torch.no_grad()
    def compact(self, position: int, keep: list[int]) -> None:
        """The union at ``position`` in ``_unions`` replaced by one that holds only the systems of its slots
        ``keep`` (still relaxing), each in a slot of its own atom count, their optimizer state carried over
        (:meth:`SystemFIRE.take_systems`), captured again; its other slots' systems are finished and collected.
        A failed capture raises and changes nothing: the old union, its runner and the per-system tables stay."""
        old = self._unions[position]
        integrator = old.integrator
        atoms = [integrator.host_atoms[slot] for slot in keep]
        start = len(self._numbers)
        union = _SlotUnion(start, start + len(keep), atoms)
        cells = integrator.cells if integrator.cell_filter else old.cells
        union.atoms = list(atoms)
        union.cells = cells.index_select(0, torch.as_tensor(keep, device=self.device)).clone()
        numbers = old.runner.builder.atomic_numbers_real
        union.numbers = torch.cat([numbers[old.layout.starts[slot]:old.layout.starts[slot] + n]
                                   for slot, n in zip(keep, atoms)])
        union.integrator = SystemFIRE(union.layout, device=self.device, **self._relaxation)
        union.integrator.take_systems(integrator, keep)
        union.integrator.sync()
        positions = union.integrator.positions.to(torch.float32)
        pairs = slot_union_demand(self._adapter, positions, layout=union.layout, cells=union.cells,
                                  atomic_numbers=union.numbers, atoms=union.atoms, candidate_skin=self.candidate_skin,
                                  device=self.device)
        union.u_capacity = union_capacity(pairs, self._capacity_step, headroom=self.capacity_headroom)
        self._capture(union, positions, union.integrator.momenta.to(torch.float32),
                      forces=union.integrator.forces.to(old.runner.forces.dtype))
        for slot in keep:  # the per-system tables of the new union's systems (rows start..stop)
            index = old.start + slot
            for table in (self._numbers, self._cells, self.templates, self.masses, self._writes):
                table.append(table[index])
        self.ceilings += atoms
        self.systems += len(keep)
        for name in ("launches", "candidate_refreshes", "fallback_windows", "recovery_replays"):
            union.accumulated[name] = old.accumulated[name] + old.runner.stats()[name]
        union.recaptures = old.recaptures + 1
        union.capture_seconds = old.capture_seconds + union.capture_seconds
        self._unions[position] = union
        old.runner = old.integrator = None

    @torch.no_grad()
    def commit_writes(self, union) -> None:
        """After writes into ``union``: the integrator's derived state, the runner's mirrors and the union's
        candidates (one refresh for all the writes)."""
        integrator, runner = union.integrator, union.runner
        integrator.sync()
        runner.reset(positions=integrator.positions.to(torch.float32), momenta=integrator.momenta.to(torch.float32),
                     forces=integrator.forces.to(runner.forces.dtype), potential_energy=integrator.potential_energy,
                     reset_counters=False)


class MultiSystemRelaxation:
    """Relax ``structures`` (see the module docstring).

    Parameters
    ----------
    model:
        An adapter, or a name or checkpoint path that :func:`~fastmd.adapters.load_adapter` loads (``model_path``,
        ``model_dtype`` as there); the models of ``MultiSystemMolecularDynamics`` (a model that does not declare
        ``"union"`` and the engine's per-edge shifts raises :class:`~fastmd.adapters.protocol.NegotiationError`).
    structures:
        ASE atoms, periodic in all three directions, each with its own cell, without constraints; checked at
        construction (with the model's elements, on one structure holding the elements of those that do not start in
        a slot).
    fmax, max_steps:
        ASE's ``run(fmax, steps)``: a structure converges when its largest force row is below ``fmax`` (eV/Å; with
        a cell filter the filter's rows, its cell's included), and stops unconverged after ``max_steps`` steps.
    cell_filter:
        None (fixed cell) or ``"frechet"``: ASE's ``FrechetCellFilter`` with its default arguments (the atom count as
        the cell factor, no pressure, every strain component).
    fire:
        :class:`~fastmd.execution.system_relax.FIREParameters`; default ASE's ``FIRE`` defaults.
    slots:
        The structures relaxed at a time; default: every structure if their atoms fit ``union_rows``, else as many
        as the planned slots' ceilings fit in it. With more structures than slots, the slots come in size classes
        (``plan_slots``) and a freed slot takes the next structure of its class, else of the largest smaller class.
    steps_per_launch:
        FIRE steps per captured window: a finished structure leaves its slot at the end of its window. Default 5:
        early in a relaxation the atoms (and cells) move fast, and a window that outruns the candidates' skin is
        replayed (10-step windows did so in the first windows of variable-cell relaxations), and a capture again
        (draining) runs the window in its warmup; a window costs the host one read.
    drain:
        With no structure left to write, capture a union again with only its relaxing structures where that pays
        back (``DRAIN_FRACTION``); it reads measured times, so whether and when it happens can differ between runs.
    union_rows, capacity_headroom, config, device, self_check:
        As in :class:`~fastmd.interfaces.multi_system_md.MultiSystemMolecularDynamics` (``union_rows`` caps the
        slots' rows per union; the self-check runs on the structures that start in a slot).

    The structures that do not start in a slot get the slots' construction checks on the host (the model's elements,
    the candidate buffer, isolated atoms where the configuration rejects them). What a relaxation brings about later
    (a variable cell that compresses beyond them, or thinner than the union's sort keys hold) raises from :meth:`run`;
    the results so far stay in ``results``.
    """

    def __init__(self, model, structures: Sequence[Atoms], *, fmax: float = 0.05, max_steps: int = 1000,
                 cell_filter: str | None = None, fire: FIREParameters | None = None, slots: int | None = None,
                 steps_per_launch: int = 5, union_rows: int | None = None, capacity_headroom: float | None = None,
                 drain: bool = True, config=None, model_path: str | None = None, model_dtype: str | None = None,
                 device=None, self_check: bool = True) -> None:
        where = "MultiSystemRelaxation"
        if isinstance(fmax, bool) or not isinstance(fmax, (int, float)) or not math.isfinite(fmax) or fmax <= 0:
            raise ValueError(f"{where}: fmax must be finite and positive")
        for name, value, low in (("max_steps", max_steps, 0), ("steps_per_launch", steps_per_launch, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{where}: {name} must be an integer >= {low}")
        if cell_filter not in (None, "frechet"):
            raise ValueError(f"{where}: cell_filter must be None or 'frechet', got {cell_filter!r}")
        if fire is not None and not isinstance(fire, FIREParameters):
            raise ValueError(f"{where}: fire must be FIREParameters")
        if slots is not None and (isinstance(slots, bool) or not isinstance(slots, int) or slots < 1):
            raise ValueError(f"{where}: slots must be a positive integer")
        structures = list(structures)
        if not structures or not all(isinstance(atoms, Atoms) and len(atoms) > 0 for atoms in structures):
            raise ValueError(f"{where}: structures must be one or more ASE Atoms with atoms")
        for index, atoms in enumerate(structures):
            MultiSystemMolecularDynamics._check_system(atoms, f"{where}: structure {index}")
            if atoms.constraints:  # ASE's optimizers apply them to every step; the batch does not
                raise ValueError(f"{where}: structure {index} has constraints "
                                 f"({', '.join(type(c).__name__ for c in atoms.constraints)}), which it does not apply")
        self._structures = [atoms.copy() for atoms in structures]
        for atoms in self._structures:
            atoms.calc = None
        sizes = [len(atoms) for atoms in structures]
        rows = DEFAULT_UNION_ROWS if union_rows is None else int(union_rows)
        if slots is None:  # every structure, else as many slots as the planned ceilings fit in one union's rows
            slots = len(sizes)
            while slots > 1 and sum(plan_slots(sizes, slots)[0]) > rows:
                slots = max(1, min(slots - 1, int(slots * rows / sum(plan_slots(sizes, slots)[0]))))
        ceilings, first, queues = plan_slots(sizes, slots)
        self.cell_filter = cell_filter
        self.fmax, self.max_steps = float(fmax), int(max_steps)
        cells = np.stack([np.asarray(atoms.cell.array, dtype=np.float64) for atoms in structures])
        admit = cells * CELL_FILTER_COMPRESSION if cell_filter else cells
        self._engine = _SlotRelaxationEngine(
            model, [self._structures[index] for index in first], slot_atoms=ceilings, union_rows=union_rows,
            steps_per_launch=steps_per_launch, capacity_headroom=capacity_headroom, config=config,
            model_path=model_path, model_dtype=model_dtype, device=device, self_check=self_check,
            relaxation=dict(fmax=float(fmax), max_steps=int(max_steps), parameters=fire,
                            cell_filter=cell_filter is not None), admit_cells=torch.as_tensor(admit))
        try:
            self._occupant = {}  # (union, slot) -> structure index
            for system, index in enumerate(first):
                union = next(u for u, union in enumerate(self._engine._unions) if union.start <= system < union.stop)
                self._occupant[(union, system - self._engine._unions[union].start)] = index
            # The queues follow the classes, smallest first; a class's ceiling (its largest structure) is its slots'.
            self._queues = [deque(queue) for queue in queues]
            self._class_ceilings = sorted(set(ceilings)) if queues else []
            self._check_elements(set(range(len(structures))) - set(first))
            self._check_neighbours(set(range(len(structures))) - set(first))
        except BaseException:
            self._engine.close()
            raise
        self.results: list[RelaxationResult | None] = [None] * len(structures)
        self.windows = 0
        self.writes = 0
        self.compactions = 0
        self.drain = bool(drain)
        self.drain_failures: list[str] = []  # captures again that failed (the union kept running as it was)
        self._kept: set = set()  # unions whose capture again failed: not tried again
        self._window_seconds: dict = {}  # union -> the wall seconds of its last windows
        self.turnover_seconds = 0.0

    def _check_elements(self, queued: set[int]) -> None:
        """The model's own input check (its ``prepare``) on one structure holding every element of the queued
        structures that no structure in a slot has: ValueError naming the structures with an element it rejects."""
        engine = self._engine
        known = set(int(z) for numbers in engine._numbers for z in numbers)
        missing = sorted({int(z) for index in queued for z in self._structures[index].get_atomic_numbers()} - known)
        if not missing:
            return
        side = math.ceil(len(missing) ** (1.0 / 3.0))
        spacing = 2.5
        grid = np.array([(i, j, k) for i in range(side) for j in range(side) for k in range(side)][:len(missing)])
        cell = np.eye(3) * spacing * side
        try:
            with engine._config.scope():
                engine._evaluate_alone(np.asarray(missing, dtype=np.int32), grid * spacing, cell, False,
                                       "MultiSystemRelaxation")
        except ValueError as error:
            rejected = set()
            for z in missing:  # which elements: each alone, two atoms 2.5 A apart
                try:
                    with engine._config.scope():
                        engine._evaluate_alone(np.asarray([z, z], dtype=np.int32), np.array([[0.0] * 3, [2.5, 0, 0]]),
                                               np.eye(3) * 5.0, False, "MultiSystemRelaxation")
                except ValueError:
                    rejected.add(z)
            bad = sorted(index for index in queued
                         if set(map(int, self._structures[index].get_atomic_numbers())) & (rejected or set(missing)))
            raise ValueError(f"MultiSystemRelaxation: the model rejects elements {sorted(rejected) or missing} of "
                             f"structures {bad[:16]}{' ...' if len(bad) > 16 else ''}: {error}") from error

    def _check_neighbours(self, queued: set[int]) -> None:
        """The checks a slot's construction runs, on the host, for the structures that do not start in a slot (written
        into a captured union later): every atom's neighbours within the candidate cutoff fit the union's candidate
        buffer and, where the union rejects isolated atoms, every atom has a neighbour within the cutoff (ASE's
        neighbour list). ValueError naming the structures that fail."""
        if not queued:
            return
        from ase.neighborlist import neighbor_list

        runner = self._engine._unions[0].runner
        builder = runner.builder
        dense, lone = [], []
        for index in sorted(queued):
            atoms = self._structures[index]
            if np.bincount(neighbor_list("i", atoms, builder.candidate_cutoff), minlength=len(atoms)).max() > \
                    builder.buffer_max:
                dense.append(index)
            if runner.validate_isolated_atoms and not np.bincount(
                    neighbor_list("i", atoms, builder.atom_graph_cutoff), minlength=len(atoms)).all():
                lone.append(index)
        for bad, why in ((dense, f"more than {builder.buffer_max} neighbours within {builder.candidate_cutoff} A "
                                 "(the candidate buffer)"),
                         (lone, f"an atom without a neighbour within {builder.atom_graph_cutoff} A")):
            if bad:
                raise ValueError(f"MultiSystemRelaxation: structures {bad[:16]}{' ...' if len(bad) > 16 else ''} have "
                                 f"{why}")

    # ------------------------------------------------------------------ the run
    def run(self) -> list[RelaxationResult]:
        """Relax every structure; returns their results in input order (also kept as ``results``)."""
        engine = self._engine
        with engine._config.scope():
            while True:
                live = [union for u, union in enumerate(engine._unions) if self._live(u)]
                if not live:
                    break
                for union in live:  # each live union runs one window; an overflow grows and captures it again
                    started = time.perf_counter()
                    engine._window(union)
                    # the last few windows' wall times (their minimum: a window that grew or refreshed costs more)
                    self._window_seconds.setdefault(union, []).append(time.perf_counter() - started)
                    del self._window_seconds[union][:-4]
                self.windows += 1
                started = time.perf_counter()
                for u, union in enumerate(engine._unions):
                    if union in live:
                        self._turnover(u, union)
                if self.drain and not any(self._queues):
                    for u in range(len(engine._unions)):
                        self._drain(u)
                self.turnover_seconds += time.perf_counter() - started
        missing = [index for index, result in enumerate(self.results) if result is None]
        if missing:
            raise RuntimeError(f"MultiSystemRelaxation: structures {missing[:16]} were never relaxed")
        return list(self.results)

    def _live(self, u: int) -> bool:
        union = self._engine._unions[u]
        phases = union.runner.phases or tuple(union.integrator.phase.tolist())
        return any(int(phase) not in FINISHED + (IDLE,) for phase in phases)

    def _next(self, ceiling: int) -> int | None:
        """The next queued structure for a freed slot of ``ceiling`` rows: of the largest class whose structures it
        holds and whose queue is not empty (its own class first)."""
        for size, queue in zip(reversed(self._class_ceilings), reversed(self._queues)):
            if size <= ceiling and queue:
                return queue.popleft()
        return None

    def _turnover(self, u: int, union) -> None:
        """The finished structures of ``union`` leave their slots (their results), the next ones come in."""
        finished = [slot for slot, phase in enumerate(union.runner.phases) if int(phase) in FINISHED]
        if not finished:
            return
        self._collect(u, union, finished)
        engine, wrote = self._engine, False
        for slot in finished:
            index = self._next(union.layout.ceilings[slot])
            if index is None:
                union.integrator.release(slot)
                continue
            engine.write_structure(union, slot, self._structures[index])
            self._occupant[(u, slot)] = index
            self.writes += 1
            wrote = True
        if wrote:
            engine.commit_writes(union)
        union.runner.phases = tuple(union.integrator.phase.tolist())  # the written and released slots' phases

    def _drain(self, u: int) -> None:
        """With no structure left to write: the union ``u`` captured again with only its relaxing systems once they
        hold at most ``DRAIN_FRACTION`` of its rows and that pays back (``DRAIN_FRACTION``'s comment;
        ``_SlotRelaxationEngine.compact``)."""
        union = self._engine._unions[u]
        live = [slot for slot, phase in enumerate(union.runner.phases) if int(phase) in (PENDING, ACTIVE)]
        rows = sum(union.layout.ceilings[slot] for slot in live)
        if not live or union.layout.rows <= DRAIN_MIN_ROWS or rows > DRAIN_FRACTION * union.layout.rows:
            return
        seconds = self._window_seconds.get(union)
        if not seconds or not union.capture_seconds:
            return
        steps = float(np.median(union.integrator.state.steps[live].cpu().numpy()))
        step = min(seconds) / self._engine.steps_per_launch
        if union in self._kept or (steps * step * (1.0 - rows / union.layout.rows)
                                   < DRAIN_PAYBACK * self._recapture_seconds(union)):
            return
        failure = None
        try:
            self._engine.compact(u, live)
        except Exception as error:  # noqa: BLE001 - an optimisation: the union runs on as it was
            failure = f"{type(error).__name__}: {str(error).splitlines()[0][:200] if str(error) else ''}"
        if failure is not None:  # outside the except block: its traceback would pin the failed capture's memory
            gc.collect()
            torch.cuda.empty_cache()
            self.drain_failures.append(failure)
            self._kept.add(union)
            return
        occupants = [self._occupant.pop((u, slot)) for slot in live]
        for slot, index in enumerate(occupants):
            self._occupant[(u, slot)] = index
        self._window_seconds.pop(union, None)
        union = self._engine._unions[u]
        union.runner.phases = tuple(union.integrator.phase.tolist())
        self.compactions += 1

    def _recapture_seconds(self, union) -> float:
        """The time a capture again of ``union`` takes, estimated: its last capture's after a capture again, else
        its first capture's scaled from that capture's warmup windows to a capture again's (each warmup and the
        capture itself run the window once)."""
        from fastmd.engine.capacity_policy import CapturePolicy

        if union.recaptures:
            return union.capture_seconds[-1]
        warmups = self._engine._config.capacity.warmup
        return union.capture_seconds[-1] * (CapturePolicy.from_config().recapture_warmup + 1) / (warmups + 1)

    def _collect(self, u: int, union, slots: list[int]) -> None:
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.stress import full_3x3_to_voigt_6_stress

        integrator = union.integrator
        state = integrator.state
        positions = integrator.positions.cpu().numpy()
        forces = integrator.forces.cpu().numpy()
        energy, fmax, steps, phase = torch.stack((state.energy, state.fmax, state.steps, state.phase)).cpu().numpy()
        cells = integrator.cells.cpu().numpy() if integrator.cell_filter else None
        stress = state.stress.cpu().numpy() if integrator.cell_filter else None
        for slot in slots:
            index = self._occupant.pop((u, slot))
            start, n = union.layout.starts[slot], integrator.host_atoms[slot]
            atoms = self._structures[index].copy()
            if cells is not None:
                atoms.set_cell(cells[slot], scale_atoms=False)
            atoms.set_positions(positions[start:start + n])
            properties = dict(energy=float(energy[slot]), forces=forces[start:start + n].copy())
            if stress is not None:
                properties["stress"] = full_3x3_to_voigt_6_stress(stress[slot])
            atoms.calc = SinglePointCalculator(atoms, **properties)
            self.results[index] = RelaxationResult(
                index=index, atoms=atoms, energy=float(energy[slot]), forces=properties["forces"],
                stress=None if stress is None else stress[slot].copy(), fmax=float(fmax[slot]), steps=int(steps[slot]),
                status=PHASES[int(phase[slot])])

    # ------------------------------------------------------------------ records
    def stats(self) -> dict:
        """The unions' counters (``MultiSystemMolecularDynamics.stats``: windows launched, candidate refreshes,
        fallback windows, recaptures, capacities and their fill, capture times), the relaxation's windows, writes
        (structures written into freed slots) and host time between windows (s), and the results so far by
        status."""
        done = [result for result in self.results if result is not None]
        statuses = {name: sum(result.status == name for result in done) for name in ("converged", "max_steps",
                                                                                     "failed")}
        unions = {key: value for key, value in self._engine.stats().items() if key not in ("systems", "writes")}
        return {**unions, "relaxation_windows": self.windows, "structure_writes": self.writes,
                "compactions": self.compactions, "drain_failures": list(self.drain_failures),
                "turnover_seconds": self.turnover_seconds, "finished": statuses, "structures": len(self.results)}

    def plan_record(self):
        """The negotiated plan (:class:`~fastmd.adapters.negotiate.EnginePlanRecord`)."""
        return self._engine.plan_record()

    @property
    def adapter(self):
        return self._engine.adapter

    def close(self) -> None:
        """Release the model adapter (one loaded by name is closed with its last user); context exit closes too."""
        self._engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
