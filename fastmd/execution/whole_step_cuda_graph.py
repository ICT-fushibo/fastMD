"""Whole-step CUDA Graph execution for fixed-cell MatRIS molecular dynamics."""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
from torch import Tensor

from fastmd.graph.fixed_capacity_builder import BatchedCandidateRefresher
from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder
from fastmd.graph.fixed_capacity_builder import fixed_capacity_exceeded
from fastmd.graph.validation import raise_if_isolated_atoms
from fastmd.engine.policy import fusion_scope
from fastmd.adapters import as_adapter
from fastmd.engine.energy_derivative import engine_derivative
from fastmd.execution.gpu_integrator import GPUIntegrator
from fastmd.execution.gpu_integrator import StatefulIntegrator
from fastmd.engine.entry_config import scoped_call
from fastmd.engine.entry_config import scoped_init
from fastmd.engine._graph_capture import capture_graph


def _require_finite_displacement(displacement: float, launch: int, replica: int | None = None) -> None:
    """Stop on non-finite positions, from the displacement each replay reads anyway.

    NaN positions make every atom isolated: the reference energy stays finite
    and forces are zero, so the run would otherwise continue silently. The max
    displacement is NaN as soon as one position is; no extra synchronization.
    """
    if not math.isfinite(displacement):
        where = "" if replica is None else f" (replica {replica})"
        raise ValueError(f"non-finite MD state after whole-step launch {launch}{where}: "
                         f"max displacement {displacement}")


class FixedCapacityOverflow(RuntimeError):
    def __init__(self, edge_count: int, triplet_count: int) -> None:
        super().__init__(
            "fixed graph capacity exceeded: "
            f"edges={edge_count}, triplets={triplet_count}"
        )
        self.edge_count = edge_count
        self.triplet_count = triplet_count


@dataclass
class WholeStepState:
    positions: Tensor
    momenta: Tensor
    forces: Tensor
    potential_energy: Tensor


class WholeStepGraphRunner:
    """Capture integrator + exact graph re-filter + model force evaluation.

    Candidate cell-list refresh remains an eager rare path. A captured window
    checkpoints its input state and ORs all device-side overflow flags. In
    strict mode an invalid window is rolled back and replayed eagerly with a
    fresh candidate list before its state becomes visible to the caller.
    Isolated atoms are permitted when the model has reference energies unless
    ``validate_isolated_atoms=True``; capacity and candidate guards remain active.

    ``integrator`` is either a single-GPU :class:`GPUIntegrator` (NVE /
    Berendsen NVT, its ``step_pre_force`` / ``step_post_force``; the runner's
    float32 buffers are the MD state; the Berendsen target is a device value
    read at every replay, so ``set_temperature`` between replays needs no
    recapture) or a
    :class:`StatefulIntegrator` such as the float64 Nose-Hoover chain, which
    owns the MD state: each step runs ``step_pre_force`` -> model ->
    ``step_post_force``, the runner's buffers become float32 mirrors, and the
    integrator's backup/restore joins the window transaction and the capture
    warmup snapshot (also restored when the warmup or capture fails).
    Supplied ``forces`` become its cached F(positions). Runners of different
    capacity tiers may share one stateful integrator; its state is the
    committed MD state.

    A window of ``steps_per_launch > 1`` steps reads one thermostat target,
    unless ``step_targets`` (an object with ``apply(k)``, e.g. the GPU-MD
    driver's per-step target table) loads step k's target inside the graph.

    ``record_stress``: a second captured window, replayed by
    ``replay(record=True)``, whose last step evaluates ``task`` plus stress
    and copies the 3x3 stress (the model's GPa) into the persistent
    ``stress`` buffer; the initial evaluation computes it as well. Both
    graphs share the persistent state and one memory pool (they never replay
    concurrently). ``record_every_window`` (a driver that logs every window):
    the record window is the only captured window and every replay records.
    """

    @scoped_init
    def __init__(
        self,
        *,
        model,
        builder: FixedCapacityGraphBuilder,
        integrator: GPUIntegrator | StatefulIntegrator,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor | None = None,
        task: str = "ef",
        steps_per_launch: int = 1,
        warmup: int = 3,
        proactive_refresh_fraction: float = 0.25,
        enable_model_fusions: bool = True,
        graph_pool=None,
        validate_isolated_atoms: bool = False,
        config=None,
        recapture: bool = False,
        step_targets=None,
        record_stress: bool = False,
        record_every_window: bool = False,
    ) -> None:
        if steps_per_launch <= 0:
            raise ValueError("steps_per_launch must be positive")
        if record_every_window and not record_stress:
            raise ValueError("record_every_window needs record_stress=True")
        if getattr(integrator, "comm", None) is not None:
            raise ValueError("a whole-step runner integrates one single-GPU system (integrator comm=None)")
        self.adapter = as_adapter(model)
        self.model = self.adapter.module
        # None: the model differentiates itself (adapter.evaluate); else the engine-owned derivative.
        self._derivative = engine_derivative(self.adapter)
        self.builder = builder
        self.integrator = integrator
        # Per-step thermostat targets of a multi-step window: ``apply(k)`` loads
        # step k's target from a device table before the step, inside the graph.
        self.step_targets = step_targets
        # A later capture of the same MD (capacity growth, tier switch, pool
        # reset) warms up once (CapturePolicy.warmups): its program is warm.
        self.recapture = bool(recapture)
        self.warmup_runs = 0
        from fastmd.engine.capacity_policy import CapturePolicy
        self._recapture_warmup = CapturePolicy.from_config().recapture_warmup
        self.stateful_integrator = (
            integrator if isinstance(integrator, StatefulIntegrator) else None
        )
        self.task = task
        self.record_stress = bool(record_stress)
        self.record_every_window = bool(record_every_window)
        self.record_task = task if "s" in task else task + "s"
        self.record_graph = None
        self.steps_per_launch = int(steps_per_launch)
        self.warmup = int(warmup)
        self.proactive_refresh_fraction = float(proactive_refresh_fraction)
        self.enable_model_fusions = bool(enable_model_fusions)
        if config is not None and config.optimization_profile != "legacy":
            self.enable_model_fusions = config.optimization_profile != "generic"
        self.validate_isolated_atoms = (
            bool(validate_isolated_atoms)
            or self.adapter.isolated_atom_policy() == "error"
        )
        # Optional shared CUDA graph memory pool. Tier-cached runners replay
        # mutually exclusively, so their capture-time intermediates may share
        # one pool (same discipline as the bucketed model-graph cache).
        self.graph_pool = graph_pool
        if not 0.0 < self.proactive_refresh_fraction < 0.5:
            raise ValueError("proactive_refresh_fraction must be in (0, 0.5)")
        self.n_real = builder.n_real
        self.device = builder.device

        # Forces, energy and stress: float64 for a float64 model the engine differentiates on a
        # stateful (float64) integrator, which then also evaluates its own float64 positions; else
        # float32. Positions and momenta are float32 either way (a stateful integrator's mirrors).
        value_dtype = torch.float32
        if self._derivative is not None and self.stateful_integrator is not None:
            value_dtype = getattr(torch, self.adapter.topology_spec().model_dtype)
        positions = positions.to(device=self.device, dtype=torch.float32)
        momenta = momenta.to(device=self.device, dtype=torch.float32)
        self.positions = positions.clone()
        self.momenta = momenta.clone()
        self.forces = torch.empty_like(self.positions, **({} if value_dtype == torch.float32 else
                                                          {"dtype": value_dtype}))
        self.potential_energy = torch.zeros(1, dtype=value_dtype, device=self.device)
        self.backup_positions = torch.empty_like(self.positions)
        self.backup_momenta = torch.empty_like(self.momenta)
        self.backup_forces = torch.empty_like(self.forces)
        self.backup_energy = torch.empty_like(self.potential_energy)
        # The stress of the last record window's final state (or the initial state's).
        self.stress = (torch.zeros((3, 3), dtype=value_dtype, device=self.device)
                       if self.record_stress else None)
        self.window_overflow = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.window_capacity_overflow = torch.zeros(
            1, dtype=torch.bool, device=self.device
        )
        self.window_max_edges = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.window_max_triplets = torch.zeros(
            1, dtype=torch.int64, device=self.device
        )
        self.window_max_isolated = torch.zeros(
            1, dtype=torch.int32, device=self.device
        )
        # A model the engine differentiates: the window's last step leaves (overflow, edges, triplets,
        # largest displacement) here, which the host reads in one copy after the window.
        self._window_stats = (torch.zeros(4, dtype=torch.float64, device=self.device)
                              if self._derivative is not None else None)
        # Windows launched before the previous one is settled (launch(behind=True)): host copies of the flags
        # a window's settling reads (two slots, allocated at the first launch()), the copy of the previous
        # window's start state and of the neighbour-count maximum kept for a rollback, and the largest
        # displacement of the last settled window with its growth per window (current and previous candidate
        # list), which tell whether a window will likely refresh the candidates.
        self._flag_slots = None
        self._next_slot = 0
        self._held = None
        self._settled_displacement = 0.0
        self._settled_fallback = False
        self._displacement_growth = [None, None]
        self.fallback_windows = 0
        self.candidate_refreshes = 1
        self.recovery_replays = 0
        self.launches = 0
        self.max_edge_count = 0
        self.max_triplet_count = 0
        self.max_displacement_seen = 0.0
        self.last_edge_count = 0
        self.last_triplet_count = 0

        self.builder.refresh_candidates(self.positions)
        self.builder.assert_candidate_capacity()
        with fusion_scope(self.enable_model_fusions):
            initial_forces, initial_energy, initial_status = self._evaluate(
                self.positions if value_dtype == torch.float32 else integrator.positions,
                rebuild_candidates=False,
                **({"stress": True} if self.record_stress else {}),
            )
            # The builder substitutes the all-sink topology when the initial
            # graph does not fit, so its forces/energy would be silently
            # wrong: reject the capacity like a replay window does.
            if bool(fixed_capacity_exceeded(initial_status, self.builder).item()):
                raise FixedCapacityOverflow(
                    int(initial_status.edge_count.item()),
                    int(initial_status.triplet_count.item()),
                )
            if self.validate_isolated_atoms:
                raise_if_isolated_atoms(initial_status.isolated_atom_count)
            if forces is None:
                self.forces.copy_(initial_forces)
                if self.stateful_integrator is not None:
                    self.stateful_integrator.set_forces(
                        initial_forces.detach(), energy=initial_energy.detach()
                    )
            else:
                self.forces.copy_(forces.to(device=self.device, dtype=self.forces.dtype))
                if self.stateful_integrator is not None:
                    # the first step kicks with the integrator's cached F(q):
                    # a no-op for the MD driver, whose mirrors equal it; the
                    # energy stays the integrator's own
                    self.stateful_integrator.set_forces(self.forces)
            self.potential_energy.copy_(initial_energy)

        self._capture()

    @scoped_call
    def _evaluate(
        self,
        positions: Tensor,
        *,
        rebuild_candidates: bool,
        stress: bool = False,
    ) -> tuple[Tensor, Tensor, object]:
        graph, status = self.builder.build(
            positions,
            rebuild_candidates=rebuild_candidates,
        )
        if self._derivative is None:
            out = self.adapter.evaluate(
                [graph],
                task=self.record_task if stress else self.task,
                is_training=False,
                n_real=self.n_real,
            )
        else:
            out = self._derivative(self.builder.batch, graph, task=self.record_task if stress else self.task,
                                   n_real=self.n_real)
        forces = out["f"][0] if isinstance(out["f"], (list, tuple)) else out["f"]
        forces = forces[: self.n_real]
        energy = (self.adapter if self._derivative is None else self._derivative).extensive_energy(
            out["e"].reshape(-1)[:1], self.n_real)
        if stress:
            self.stress.copy_(out["s"][0].detach())
        return forces, energy, status

    def _step_body(self, *, rebuild_candidates: bool, stress: bool = False, mirror: bool = True) -> object:
        stress_kw = {"stress": True} if stress else {}  # calls without stress are unchanged
        if self.stateful_integrator is not None:
            return self._stateful_step_body(rebuild_candidates=rebuild_candidates, **stress_kw,
                                            **({} if mirror else {"mirror": False}))
        positions_new, momenta = self.integrator.step_pre_force(
            self.positions, self.momenta, self.forces
        )
        forces, energy, status = self._evaluate(
            positions_new,
            rebuild_candidates=rebuild_candidates,
            **stress_kw,
        )
        momenta = self.integrator.step_post_force(momenta, forces)
        self.positions.copy_(positions_new)
        self.momenta.copy_(momenta)
        self.forces.copy_(forces.detach())
        self.potential_energy.copy_(energy.detach())
        return status

    def _stateful_step_body(self, *, rebuild_candidates: bool, stress: bool = False, mirror: bool = True) -> object:
        integrator = self.stateful_integrator
        positions = integrator.step_pre_force()
        forces, energy, status = self._evaluate(
            positions,
            rebuild_candidates=rebuild_candidates,
            **({"stress": True} if stress else {}),
        )
        with torch.no_grad():
            forces = forces.detach()
            energy = energy.detach()
            integrator.step_post_force(forces, energy=energy)
            if not mirror:  # an inner step of a window whose last step mirrors the state
                return status
            # float32 mirrors read by replay / refresh / fallback / stats
            self.positions.copy_(positions)
            self.momenta.copy_(integrator.momenta)
            self.forces.copy_(forces)
            self.potential_energy.copy_(energy)
        return status

    def _window_body(self, record: bool = False) -> None:
        if self.stateful_integrator is not None:
            self.stateful_integrator.backup()
        self.backup_positions.copy_(self.positions)
        self.backup_momenta.copy_(self.momenta)
        self.backup_forces.copy_(self.forces)
        self.backup_energy.copy_(self.potential_energy)
        self.window_overflow.zero_()
        self.window_capacity_overflow.zero_()
        self.window_max_edges.zero_()
        self.window_max_triplets.zero_()
        self.window_max_isolated.zero_()
        for k in range(self.steps_per_launch):
            if self.step_targets is not None:
                self.step_targets.apply(k)
            last = k == self.steps_per_launch - 1
            if self._window_stats is not None and self.builder.edges_only:
                # An edges-only model the engine differentiates: the stateful integrator's mirrors at the
                # window's end only, the same flags in one kernel per step, the window's statistics in one
                # buffer.
                status = self._step_body(rebuild_candidates=False, **({"stress": True} if record and last else {}),
                                         **({} if last else {"mirror": False}))
                self.builder.accumulate_window(
                    status, window_overflow=self.window_overflow,
                    window_capacity_overflow=self.window_capacity_overflow,
                    window_max_edges=self.window_max_edges, window_max_isolated=self.window_max_isolated,
                    check_isolated=self.validate_isolated_atoms)
                if last:
                    self._window_stats.copy_(torch.cat([
                        self.window_overflow.to(torch.float64), status.edge_count.to(torch.float64),
                        status.triplet_count.to(torch.float64), status.max_displacement.to(torch.float64)]))
                continue
            # A record window's last step also computes the stress. A line-graph model the engine
            # differentiates: the stateful integrator's mirrors at the window's end only.
            status = self._step_body(rebuild_candidates=False,
                                     **({"stress": True} if record and k == self.steps_per_launch - 1 else {}),
                                     **({} if last or self._window_stats is None else {"mirror": False}))
            capacity_overflow = fixed_capacity_exceeded(status, self.builder)
            overflow = self.window_overflow | status.overflow
            if self.validate_isolated_atoms:
                overflow = overflow | (status.isolated_atom_count > 0)
            self.window_overflow.copy_(overflow)
            self.window_capacity_overflow.copy_(
                self.window_capacity_overflow | capacity_overflow.reshape(1)
            )
            self.window_max_edges.copy_(
                torch.maximum(self.window_max_edges, status.edge_count)
            )
            self.window_max_triplets.copy_(
                torch.maximum(self.window_max_triplets, status.triplet_count)
            )
            self.window_max_isolated.copy_(
                torch.maximum(
                    self.window_max_isolated,
                    status.isolated_atom_count,
                )
            )
            if last and self._window_stats is not None:
                self._window_stats.copy_(torch.cat([
                    self.window_overflow.to(torch.float64), status.edge_count.to(torch.float64),
                    status.triplet_count.to(torch.float64), status.max_displacement.to(torch.float64)]))

    def _restore_backup(self) -> None:
        self.positions.copy_(self.backup_positions)
        self.momenta.copy_(self.backup_momenta)
        self.forces.copy_(self.backup_forces)
        self.potential_energy.copy_(self.backup_energy)
        if self.stateful_integrator is not None:
            self.stateful_integrator.restore()

    def _warmups(self) -> int:
        """Side-stream warmup windows: ``warmup`` for the first capture of an MD,
        CapturePolicy's later-capture count for a recapture (``recapture``), plus
        one where recompute_policy "fast" still has to check an uncaptured run."""
        from fastmd.engine.capacity_policy import CapturePolicy
        from fastmd.config import current_config
        from fastmd.engine.saved_recompute import capture_check_pending
        count = CapturePolicy(warmup=self.warmup, recapture_warmup=self._recapture_warmup).warmups(not self.recapture)
        config = current_config()
        if count < self.warmup and config is not None and capture_check_pending(self.model, config):
            count += 1
        return count

    @scoped_call
    def _capture(self) -> None:
        saved_positions = self.positions.clone()
        saved_momenta = self.momenta.clone()
        saved_forces = self.forces.clone()
        saved_energy = self.potential_energy.clone()
        saved_stress = None if self.stress is None else self.stress.clone()
        # warmup windows advance a stateful integrator's own state as well
        integrator = self.stateful_integrator
        saved_integrator = integrator.snapshot() if integrator is not None else None

        def restore_saved() -> None:
            self.positions.copy_(saved_positions)
            self.momenta.copy_(saved_momenta)
            self.forces.copy_(saved_forces)
            self.potential_energy.copy_(saved_energy)
            if saved_stress is not None:
                self.stress.copy_(saved_stress)
            if integrator is not None:
                integrator.load_snapshot(saved_integrator)

        # Freed blocks go back to the device before the side-stream warmup and
        # before the capture: the caching allocator keeps blocks per stream and
        # the capture allocates from its own pool, so the initial evaluation's,
        # the warmup's and the graph's working sets would otherwise all stay
        # reserved (164 GiB for an 82 GiB working set at 10,752 RDX atoms).
        torch.cuda.empty_cache()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        try:
            with fusion_scope(self.enable_model_fusions):
                with torch.cuda.stream(side):
                    for _ in range(0 if self.record_every_window else self._warmups()):
                        self._window_body()
                        restore_saved()
                        self.warmup_runs += 1
                    # Before either capture, so that no warmup runs next to a captured pool.
                    for _ in range(self._warmups() if self.record_stress else 0):
                        self._window_body(record=True)
                        restore_saved()
                        self.warmup_runs += 1
                torch.cuda.current_stream().wait_stream(side)
                restore_saved()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if not self.record_every_window:
                    self.graph = torch.cuda.CUDAGraph()
                    with capture_graph(self.graph, pool=self.graph_pool):
                        self._window_body()
                if self.record_stress:
                    self.record_graph = torch.cuda.CUDAGraph()
                    pool = (self.graph.pool() if self.graph_pool is None and not self.record_every_window
                            else self.graph_pool)
                    with capture_graph(self.record_graph, pool=pool):
                        self._window_body(record=True)
                    if self.record_every_window:
                        self.graph = self.record_graph
        except BaseException:
            # A failed warmup window or capture (e.g. out of memory in the
            # model) must not leave a shared stateful integrator advanced:
            # the active runner of another tier continues from its state.
            torch.cuda.current_stream().wait_stream(side)
            restore_saved()
            raise
        restore_saved()
        torch.cuda.synchronize()

    @scoped_call
    def _fallback(self, record: bool = False) -> None:
        self._restore_backup()
        if bool(self.window_capacity_overflow.item()):
            edge_count = int(self.window_max_edges.item())
            triplet_count = int(self.window_max_triplets.item())
            raise FixedCapacityOverflow(edge_count, triplet_count)
        self.builder.refresh_candidates(self.positions)
        self.builder.assert_candidate_capacity()
        self.candidate_refreshes += 1
        (self.record_graph if record else self.graph).replay()
        self.recovery_replays += 1
        if bool(self.window_capacity_overflow.item()):
            self._restore_backup()
            raise FixedCapacityOverflow(
                int(self.window_max_edges.item()),
                int(self.window_max_triplets.item()),
            )
        if self.validate_isolated_atoms:
            isolated_count = int(self.window_max_isolated.item())
            if isolated_count:
                self._restore_backup()
                raise_if_isolated_atoms(isolated_count)
        if bool(self.window_overflow.item()):
            self._restore_backup()
            with fusion_scope(self.enable_model_fusions):
                for k in range(self.steps_per_launch):
                    if self.step_targets is not None:
                        self.step_targets.apply(k)
                    status = self._step_body(
                        rebuild_candidates=True,
                        **({"stress": True} if record and k == self.steps_per_launch - 1 else {}))
                    capacity_overflow = bool(
                        fixed_capacity_exceeded(status, self.builder).item()
                    )
                    if capacity_overflow:
                        self._restore_backup()
                        raise FixedCapacityOverflow(
                            int(status.edge_count.item()),
                            int(status.triplet_count.item()),
                        )
                    if self.validate_isolated_atoms:
                        isolated_count = int(status.isolated_atom_count.item())
                        if isolated_count:
                            self._restore_backup()
                            raise_if_isolated_atoms(isolated_count)
        self.fallback_windows += 1

    @scoped_call
    def replay(self, *, strict: bool = True, record: bool = False, launched=None) -> WholeStepState:
        """One window; ``record`` (``record_stress``): the window that also leaves its last state's stress in
        ``stress``. ``launched``: called once the window is launched, before its flags are read (host
        work that overlaps the window)."""
        record = record or self.record_every_window
        if record and self.record_graph is None:
            raise ValueError("replay(record=True) needs a runner with record_stress=True")
        (self.record_graph if record else self.graph).replay()
        self.launches += 1
        if launched is not None:
            launched()
        if strict:
            self.settle(record=record)
        return self.state

    def settle(self, *, record: bool = False, flags: tuple | None = None) -> None:
        """The checks after a window (``replay(strict=True)``): an overflow's fallback, the counts, the
        displacement and the proactive candidate refresh. ``flags``: the window's (overflow, edges, triplets,
        largest displacement) on the host (:meth:`launched_flags`), read instead of the device's; with a
        window launched behind it, an overflowing one is never settled here (its fallback reads the device
        flags: :meth:`rewind` and replay it)."""
        fallbacks = self.fallback_windows
        packed = flags
        if packed is None and self._window_stats is not None:
            # One read: (overflow, edges, triplets, largest displacement) of the window's last step.
            packed = self._window_stats.tolist()
        if packed is not None:
            if packed[0]:
                packed = None  # the fallback rebuilds the window: its own counts are read below
                self._fallback(**({"record": True} if record else {}))
        elif bool(self.window_overflow.item()):
            self._fallback(**({"record": True} if record else {}))
        if packed is not None:
            self.last_edge_count, self.last_triplet_count = int(packed[1]), int(packed[2])
        else:
            self.last_edge_count = int(self.builder.edge_offsets[-1].item())
            self.last_triplet_count = int(self.builder.pair_offsets[-1].item())
        self.max_edge_count = max(self.max_edge_count, self.last_edge_count)
        self.max_triplet_count = max(
            self.max_triplet_count, self.last_triplet_count
        )
        displacement = packed[3] if packed is not None else float(self.builder._max_displacement.item())
        _require_finite_displacement(displacement, self.launches)
        self.max_displacement_seen = max(
            self.max_displacement_seen,
            displacement,
        )
        # The displacement's growth in this window; none after a fallback (whose eager steps may rebuild
        # the candidates at every step).
        self._settled_fallback = self.fallback_windows != fallbacks
        if not self._settled_fallback:
            growth = displacement - self._settled_displacement
            self._displacement_growth[0] = max(growth, self._displacement_growth[0] or 0.0)
        self._settled_displacement = displacement
        refresh_threshold = self.builder.skin * self.proactive_refresh_fraction
        if displacement >= refresh_threshold:
            self.builder.refresh_candidates(self.positions)
            self.candidate_refreshes += 1
            self._settled_displacement = 0.0
            self._displacement_growth = [None, self._displacement_growth[0]]

    def refresh_unlikely(self) -> bool:
        """Whether the window in flight will likely neither refresh the candidates nor fall back: the last
        settled window did not fall back, and its largest displacement plus twice the largest growth per
        window seen with this or the previous candidate list stays below the refresh threshold. A window
        launched behind one that does is rolled back (one window of GPU time), so the MD driver launches
        none behind a likely one."""
        growth = [value for value in self._displacement_growth if value is not None]
        if not growth or self._settled_fallback:
            return False
        threshold = self.builder.skin * self.proactive_refresh_fraction
        return self._settled_displacement + 2.0 * max(growth) < threshold

    def launch(self, *, record: bool = False, behind: bool = False) -> int:
        """Replay one window without settling it, and copy the flags :meth:`settle` reads to host memory
        (no host synchronization); returns their slot for :meth:`launched_flags`. ``behind``: the previous
        window is not settled yet; its start state, which this window's backup overwrites, and the
        neighbour-count maximum after it are kept first, for :meth:`rewind` and :meth:`drop`."""
        record = record or self.record_every_window
        if record and self.record_graph is None:
            raise ValueError("replay(record=True) needs a runner with record_stress=True")
        if behind:
            held, sources = self._held_buffers()
            torch._foreach_copy_(held, sources)
            if self.stateful_integrator is not None:
                # its backup (the previous window's start) into the held state
                self.stateful_integrator.restore(self._held_state)
        (self.record_graph if record else self.graph).replay()
        self.launches += 1
        if self._flag_slots is None:
            sources = self._flag_sources()
            self._flag_slots = [([torch.empty(source.shape, dtype=source.dtype, pin_memory=True)
                                  for source in sources], torch.cuda.Event()) for _ in range(2)]
        slot = self._next_slot
        self._next_slot ^= 1
        copies, event = self._flag_slots[slot]
        for copy, source in zip(copies, self._flag_sources()):
            copy.copy_(source, non_blocking=True)
        event.record()
        return slot

    def _flag_sources(self) -> list[Tensor]:
        if self._window_stats is not None:
            return [self._window_stats]
        return [self.window_overflow, self.builder.edge_offsets[-1:], self.builder.pair_offsets[-1:],
                self.builder._max_displacement]

    def launched_flags(self, slot: int) -> tuple:
        """The (overflow, edges, triplets, largest displacement) of the window :meth:`launch` copied into
        ``slot``, once the copy is done (it waits for that window only)."""
        copies, event = self._flag_slots[slot]
        event.synchronize()
        if self._window_stats is not None:
            return tuple(copies[0].tolist())
        return tuple(copy.item() for copy in copies)

    def settling_work(self, flags: tuple) -> str | None:
        """What settling a window with these host ``flags`` changes besides host bookkeeping: "overflow"
        (the fallback replays the window), "refresh" (new candidates; also a non-finite state, which
        raises), else None."""
        if flags[0]:
            return "overflow"
        displacement = flags[3]
        if not math.isfinite(displacement):
            return "refresh"
        return "refresh" if displacement >= self.builder.skin * self.proactive_refresh_fraction else None

    def _held_buffers(self) -> tuple[list[Tensor], list[Tensor]]:
        sources = [self.backup_positions, self.backup_momenta, self.backup_forces, self.backup_energy]
        if self.builder.max_neighbor_count is not None:
            sources.append(self.builder.max_neighbor_count)
        if self._held is None:
            self._held = [torch.empty_like(source) for source in sources]
            if self.stateful_integrator is not None:
                self._held_state = self.stateful_integrator.new_state()
        return self._held, sources

    def drop(self) -> None:
        """Roll back the window launched last (``launch(behind=True)``) to its start, the previous
        window's end: the state, the stateful integrator's and the neighbour-count maximum."""
        self._restore_backup()
        if self.builder.max_neighbor_count is not None:
            self.builder.max_neighbor_count.copy_(self._held[-1])
        self.launches -= 1

    def rewind(self) -> None:
        """Roll back the unsettled window and the one launched behind it (``launch(behind=True)``) to the
        unsettled window's start, kept by that launch: both then run as if never launched."""
        held = self._held
        torch._foreach_copy_([self.positions, self.momenta, self.forces, self.potential_energy], held[:4])
        if self.stateful_integrator is not None:
            # the held state into the integrator's backup, then its state
            self.stateful_integrator.backup(self._held_state)
            self.stateful_integrator.restore()
        if self.builder.max_neighbor_count is not None:
            self.builder.max_neighbor_count.copy_(held[-1])
        self.launches -= 2

    @property
    def state(self) -> WholeStepState:
        return WholeStepState(
            positions=self.positions,
            momenta=self.momenta,
            forces=self.forces,
            potential_energy=self.potential_energy,
        )

    @torch.no_grad()
    def reset(
        self,
        *,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor,
        potential_energy: Tensor,
        reset_counters: bool = True,
    ) -> None:
        """Reset persistent state without changing captured tensor addresses.

        With a stateful integrator the arguments are the float32 mirrors of
        its (already current) state; the integrator state is not changed.
        """
        self.positions.copy_(positions)
        self.momenta.copy_(momenta)
        self.forces.copy_(forces)
        self.potential_energy.copy_(potential_energy.reshape(1))
        self.builder.refresh_candidates(self.positions)
        self._settled_displacement = 0.0
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

    def stats(self) -> dict[str, int | float]:
        return {
            "steps_per_launch": self.steps_per_launch,
            "launches": self.launches,
            "candidate_refreshes": self.candidate_refreshes,
            "fallback_windows": self.fallback_windows,
            "recovery_replays": self.recovery_replays,
            "max_edge_count": self.max_edge_count,
            "max_triplet_count": self.max_triplet_count,
            "max_displacement": self.max_displacement_seen,
            "u_capacity": self.builder.u_capacity,
            "t_capacity": self.builder.t_capacity,
        }


class BatchedWholeStepGraphRunner:
    """Capture multiple homogeneous MD replicas in one whole-step graph.

    Each replica retains an independent candidate cache, capacity status,
    thermostat, and state. The model sees one graph batch so large sparse
    motifs and GEMMs amortize launch latency across replicas.
    Isolated-atom validation follows :class:`WholeStepGraphRunner`.
    """

    @scoped_init
    def __init__(
        self,
        *,
        model,
        builders: Sequence[FixedCapacityGraphBuilder],
        integrator: GPUIntegrator,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor | None = None,
        task: str = "ef",
        steps_per_launch: int = 1,
        warmup: int = 3,
        proactive_refresh_fraction: float = 0.25,
        parallel_graph_build: bool = True,
        validate_isolated_atoms: bool = False,
        enable_model_fusions: bool = True,
        config=None,
    ) -> None:
        if not builders:
            raise ValueError("builders must not be empty")
        if steps_per_launch <= 0:
            raise ValueError("steps_per_launch must be positive")
        if isinstance(integrator, StatefulIntegrator):
            raise ValueError(
                "BatchedWholeStepGraphRunner supports GPUIntegrator (nve/nvt) only"
            )
        if getattr(integrator, "comm", None) is not None:
            raise ValueError("a whole-step runner integrates one single-GPU system (integrator comm=None)")
        self.adapter = as_adapter(model)
        if engine_derivative(self.adapter) is not None:
            raise ValueError(f"BatchedWholeStepGraphRunner calls the model's own evaluate (a model-owned derivative); "
                             f"model {self.adapter.name!r} has an energy the engine differentiates: run each replica "
                             "with GPUResidentMolecularDynamics (or WholeStepGraphRunner)")
        self.model = self.adapter.module
        self.builders = list(builders)
        self.integrator = integrator
        self.task = task
        self.batch_size = len(self.builders)
        self.steps_per_launch = int(steps_per_launch)
        self.warmup = int(warmup)
        self.parallel_graph_build = bool(parallel_graph_build)
        self.enable_model_fusions = bool(enable_model_fusions)
        if config is not None and config.optimization_profile != "legacy":
            self.enable_model_fusions = config.optimization_profile != "generic"
        self.validate_isolated_atoms = (
            bool(validate_isolated_atoms)
            or self.adapter.isolated_atom_policy() == "error"
        )
        self.proactive_refresh_fraction = float(proactive_refresh_fraction)
        if not 0.0 < self.proactive_refresh_fraction < 0.5:
            raise ValueError("proactive_refresh_fraction must be in (0, 0.5)")

        self.device = self.builders[0].device
        self.n_real = self.builders[0].n_real
        if any(builder.device != self.device for builder in self.builders):
            raise ValueError("all builders must use the same device")
        if any(builder.n_real != self.n_real for builder in self.builders):
            raise ValueError("batched runner currently requires equal atom counts")
        expected_shape = (self.batch_size, self.n_real, 3)
        if tuple(positions.shape) != expected_shape:
            raise ValueError(
                f"positions must have shape {expected_shape}, got {tuple(positions.shape)}"
            )
        if tuple(momenta.shape) != expected_shape:
            raise ValueError(
                f"momenta must have shape {expected_shape}, got {tuple(momenta.shape)}"
            )

        self.real_counts = torch.full(
            (self.batch_size,),
            self.n_real,
            dtype=torch.int32,
            device=self.device,
        )
        self.positions = positions.to(self.device, torch.float32).clone()
        self.momenta = momenta.to(self.device, torch.float32).clone()
        self.forces = torch.empty_like(self.positions)
        self.potential_energy = torch.zeros(
            self.batch_size, dtype=torch.float32, device=self.device
        )
        self.backup_positions = torch.empty_like(self.positions)
        self.backup_momenta = torch.empty_like(self.momenta)
        self.backup_forces = torch.empty_like(self.forces)
        self.backup_energy = torch.empty_like(self.potential_energy)
        self.window_overflow = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        self.window_capacity_overflow = torch.zeros_like(self.window_overflow)
        self.window_max_edges = torch.zeros(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        self.window_max_triplets = torch.zeros_like(self.window_max_edges)
        self.window_max_isolated = torch.zeros(
            self.batch_size, dtype=torch.int32, device=self.device
        )
        self.fallback_windows = 0
        self.candidate_refreshes = 1
        self.recovery_replays = 0
        self.launches = 0
        self.max_edge_counts = [0] * self.batch_size
        self.max_triplet_counts = [0] * self.batch_size
        self.max_displacements = [0.0] * self.batch_size
        self.last_edge_counts = [0] * self.batch_size
        self.last_triplet_counts = [0] * self.batch_size
        self.candidate_refresher = BatchedCandidateRefresher(self.builders)
        self.build_streams = (
            [torch.cuda.Stream(device=self.device) for _ in self.builders]
            if self.parallel_graph_build and self.batch_size > 1
            else []
        )

        self.candidate_refresher.refresh(self.positions)
        for builder in self.builders:
            builder.assert_candidate_capacity()
        with fusion_scope(self.enable_model_fusions):
            initial_forces, initial_energy, initial_statuses = self._evaluate(
                self.positions,
                rebuild_candidates=False,
            )
            if any(
                bool(fixed_capacity_exceeded(status, builder).item())
                for status, builder in zip(initial_statuses, self.builders)
            ):
                raise FixedCapacityOverflow(
                    max(int(status.edge_count.item()) for status in initial_statuses),
                    max(int(status.triplet_count.item()) for status in initial_statuses),
                )
            if self.validate_isolated_atoms:
                raise_if_isolated_atoms(
                    torch.cat(
                        [status.isolated_atom_count for status in initial_statuses]
                    )
                )
            if forces is None:
                self.forces.copy_(initial_forces)
            else:
                self.forces.copy_(forces.to(self.device, torch.float32))
            self.potential_energy.copy_(initial_energy)
        self._capture()

    @scoped_call
    def _evaluate(
        self,
        positions: Tensor,
        *,
        rebuild_candidates: bool,
    ) -> tuple[Tensor, Tensor, list[object]]:
        graphs = []
        statuses = []
        use_parallel_build = self.build_streams and not rebuild_candidates
        if use_parallel_build:
            origin_stream = torch.cuda.current_stream(self.device)
            for replica, (builder, stream) in enumerate(
                zip(self.builders, self.build_streams)
            ):
                stream.wait_stream(origin_stream)
                with torch.cuda.stream(stream):
                    graph, status = builder.build(
                        positions[replica],
                        rebuild_candidates=False,
                    )
                graphs.append(graph)
                statuses.append(status)
            for stream in self.build_streams:
                origin_stream.wait_stream(stream)
        else:
            for replica, builder in enumerate(self.builders):
                graph, status = builder.build(
                    positions[replica],
                    rebuild_candidates=rebuild_candidates,
                )
                graphs.append(graph)
                statuses.append(status)
        output = self.adapter.evaluate(
            graphs,
            task=self.task,
            is_training=False,
            n_real=self.real_counts,
        )
        force_parts = output["f"]
        if not isinstance(force_parts, (list, tuple)):
            raise RuntimeError("batched model did not return per-graph forces")
        forces = torch.stack(
            [force[: self.n_real] for force in force_parts],
            dim=0,
        )
        energy = self.adapter.extensive_energy(output["e"].reshape(self.batch_size), self.real_counts)
        return forces, energy, statuses

    def _step_body(self, *, rebuild_candidates: bool) -> list[object]:
        positions_new, momenta = self.integrator.step_pre_force(
            self.positions, self.momenta, self.forces
        )
        forces, energy, statuses = self._evaluate(
            positions_new,
            rebuild_candidates=rebuild_candidates,
        )
        momenta = self.integrator.step_post_force(momenta, forces)
        self.positions.copy_(positions_new)
        self.momenta.copy_(momenta)
        self.forces.copy_(forces.detach())
        self.potential_energy.copy_(energy.detach())
        return statuses

    def _window_body(self) -> None:
        self.backup_positions.copy_(self.positions)
        self.backup_momenta.copy_(self.momenta)
        self.backup_forces.copy_(self.forces)
        self.backup_energy.copy_(self.potential_energy)
        self.window_overflow.zero_()
        self.window_capacity_overflow.zero_()
        self.window_max_edges.zero_()
        self.window_max_triplets.zero_()
        self.window_max_isolated.zero_()
        for _ in range(self.steps_per_launch):
            statuses = self._step_body(rebuild_candidates=False)
            for replica, (status, builder) in enumerate(
                zip(statuses, self.builders)
            ):
                capacity_overflow = fixed_capacity_exceeded(status, builder)
                overflow = (
                    self.window_overflow[replica : replica + 1]
                    | status.overflow
                )
                if self.validate_isolated_atoms:
                    overflow = overflow | (status.isolated_atom_count > 0)
                self.window_overflow[replica : replica + 1].copy_(overflow)
                self.window_capacity_overflow[replica : replica + 1].copy_(
                    self.window_capacity_overflow[replica : replica + 1]
                    | capacity_overflow.reshape(1)
                )
                self.window_max_edges[replica : replica + 1].copy_(
                    torch.maximum(
                        self.window_max_edges[replica : replica + 1],
                        status.edge_count,
                    )
                )
                self.window_max_triplets[replica : replica + 1].copy_(
                    torch.maximum(
                        self.window_max_triplets[replica : replica + 1],
                        status.triplet_count,
                    )
                )
                self.window_max_isolated[replica : replica + 1].copy_(
                    torch.maximum(
                        self.window_max_isolated[replica : replica + 1],
                        status.isolated_atom_count,
                    )
                )

    def _restore_backup(self) -> None:
        self.positions.copy_(self.backup_positions)
        self.momenta.copy_(self.backup_momenta)
        self.forces.copy_(self.backup_forces)
        self.potential_energy.copy_(self.backup_energy)

    @scoped_call
    def _capture(self) -> None:
        saved_positions = self.positions.clone()
        saved_momenta = self.momenta.clone()
        saved_forces = self.forces.clone()
        saved_energy = self.potential_energy.clone()
        with fusion_scope(self.enable_model_fusions):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self.warmup):
                    self._window_body()
                    self.positions.copy_(saved_positions)
                    self.momenta.copy_(saved_momenta)
                    self.forces.copy_(saved_forces)
                    self.potential_energy.copy_(saved_energy)
            torch.cuda.current_stream().wait_stream(side)
            self.positions.copy_(saved_positions)
            self.momenta.copy_(saved_momenta)
            self.forces.copy_(saved_forces)
            self.potential_energy.copy_(saved_energy)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with capture_graph(self.graph):
                self._window_body()
        self.positions.copy_(saved_positions)
        self.momenta.copy_(saved_momenta)
        self.forces.copy_(saved_forces)
        self.potential_energy.copy_(saved_energy)
        torch.cuda.synchronize()

    @scoped_call
    def _fallback(self) -> None:
        self._restore_backup()
        if bool(self.window_capacity_overflow.any().item()):
            raise FixedCapacityOverflow(
                int(self.window_max_edges.max().item()),
                int(self.window_max_triplets.max().item()),
            )
        self.candidate_refresher.refresh(self.positions)
        for builder in self.builders:
            builder.assert_candidate_capacity()
        self.candidate_refreshes += 1
        self.graph.replay()
        self.recovery_replays += 1
        if bool(self.window_capacity_overflow.any().item()):
            self._restore_backup()
            raise FixedCapacityOverflow(
                int(self.window_max_edges.max().item()),
                int(self.window_max_triplets.max().item()),
            )
        if self.validate_isolated_atoms:
            isolated_count = int(self.window_max_isolated.sum().item())
            if isolated_count:
                self._restore_backup()
                raise_if_isolated_atoms(isolated_count)
        if bool(self.window_overflow.any().item()):
            self._restore_backup()
            with fusion_scope(self.enable_model_fusions):
                for _ in range(self.steps_per_launch):
                    statuses = self._step_body(rebuild_candidates=True)
                    capacity_overflow = any(
                        bool(fixed_capacity_exceeded(status, builder).item())
                        for status, builder in zip(statuses, self.builders)
                    )
                    if capacity_overflow:
                        self._restore_backup()
                        raise FixedCapacityOverflow(
                            max(int(status.edge_count.item()) for status in statuses),
                            max(
                                int(status.triplet_count.item())
                                for status in statuses
                            ),
                        )
                    if self.validate_isolated_atoms:
                        isolated_count = sum(
                            int(status.isolated_atom_count.item())
                            for status in statuses
                        )
                        if isolated_count:
                            self._restore_backup()
                            raise_if_isolated_atoms(isolated_count)
        self.fallback_windows += 1

    @scoped_call
    def replay(self, *, strict: bool = True) -> WholeStepState:
        self.graph.replay()
        self.launches += 1
        if strict and bool(self.window_overflow.any().item()):
            self._fallback()
        if strict:
            refresh_thresholds = [
                builder.skin * self.proactive_refresh_fraction
                for builder in self.builders
            ]
            refresh_needed = False
            for replica, builder in enumerate(self.builders):
                edge_count = int(builder.edge_offsets[-1].item())
                triplet_count = int(builder.pair_offsets[-1].item())
                displacement = float(builder._max_displacement.item())
                _require_finite_displacement(displacement, self.launches, replica)
                self.last_edge_counts[replica] = edge_count
                self.last_triplet_counts[replica] = triplet_count
                self.max_edge_counts[replica] = max(
                    self.max_edge_counts[replica], edge_count
                )
                self.max_triplet_counts[replica] = max(
                    self.max_triplet_counts[replica], triplet_count
                )
                self.max_displacements[replica] = max(
                    self.max_displacements[replica], displacement
                )
                if displacement >= refresh_thresholds[replica]:
                    refresh_needed = True
            if refresh_needed:
                self.candidate_refresher.refresh(self.positions)
                self.candidate_refreshes += 1
        return self.state

    @property
    def state(self) -> WholeStepState:
        return WholeStepState(
            positions=self.positions,
            momenta=self.momenta,
            forces=self.forces,
            potential_energy=self.potential_energy,
        )

    def stats(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "steps_per_launch": self.steps_per_launch,
            "parallel_graph_build": self.parallel_graph_build,
            "launches": self.launches,
            "candidate_refreshes": self.candidate_refreshes,
            "fallback_windows": self.fallback_windows,
            "recovery_replays": self.recovery_replays,
            "max_edge_counts": list(self.max_edge_counts),
            "max_triplet_counts": list(self.max_triplet_counts),
            "max_displacements": list(self.max_displacements),
            "u_capacities": [builder.u_capacity for builder in self.builders],
            "t_capacities": [builder.t_capacity for builder in self.builders],
        }
