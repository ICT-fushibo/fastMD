from __future__ import annotations

from collections.abc import Callable

import torch
from ase import units
from torch import Tensor

from fastmd._nvtx import nvtx_range


def kinetic_energy_fp64(momenta: Tensor, masses: Tensor) -> Tensor:
    """``Atoms.get_kinetic_energy`` (0.5 * vdot(p, p/m)) reduced in float64.

    ``momenta`` [N, 3] (0-dim result) or [B, N, 3] ([B]); ``masses`` [N] or
    [N, 1], any float dtype. For reporting only (the MD drivers' Ekin and T at
    every P); it never advances a state.
    """
    p = momenta.to(torch.float64)
    x = p * (p / masses.to(torch.float64).reshape(-1, 1))
    return 0.5 * (x.sum() if p.ndim == 2 else x.sum(dim=(-2, -1)))


def default_fix_com(ensemble: str) -> bool:
    """The MD drivers' ``fix_com=None``: remove the centre-of-mass momentum
    every step for nve/nvt, none for nvt_nhc (as ASE ``NoseHooverChainNVT``).

    GPUResidentMolecularDynamics (native and from_evaluator); :class:`GPUIntegrator`'s
    own ``None`` rule is unchanged (its callers pass the value).
    """
    return ensemble.lower() != "nvt_nhc"


class _PinnedUpload:
    """In-place host -> device rewrite of a small tensor without a host sync.

    A ring of pinned staging buffers, each guarded by the event of its last
    copy: a slot is refilled only after that copy finished (in steady state
    long ago, so the wait is free; the host may run SLOTS updates ahead of
    the device). The copy is ordered on the current stream. CPU tensors:
    plain copy.
    """

    SLOTS = 16

    def __init__(self, dst: Tensor) -> None:
        self.dst = dst
        self._slots = None
        self._next = 0

    def __call__(self, values) -> None:
        host = torch.tensor(values, dtype=self.dst.dtype).reshape(self.dst.shape)
        if self.dst.device.type != "cuda":
            self.dst.copy_(host)
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("thermostat constants are updated between replays, not in a capture")
        if self._slots is None:
            self._slots = [(host.pin_memory(), torch.cuda.Event()) for _ in range(self.SLOTS)]
        buffer, event = self._slots[self._next]
        self._next = (self._next + 1) % self.SLOTS
        event.synchronize()  # the slot's previous copy (no-op if never recorded)
        buffer.copy_(host)
        with torch.cuda.device(self.dst.device):
            self.dst.copy_(buffer, non_blocking=True)
            event.record()


class StatefulIntegrator:
    """Integrator that owns the authoritative MD state in its own buffers.

    :class:`GPUIntegrator` is functional: the MD driver keeps the float32
    state and passes it in. A stateful integrator (e.g. the float64
    :class:`~fastmd.execution.gpu_nhc.GPUNoseHooverChainNVT`) keeps
    positions, momenta, the cached forces F(positions), the potential energy
    and any thermostat variables itself; the drivers only keep float32
    mirrors of positions/momenta/forces/energy for graph building, logging
    and statistics. One MD step is split around the single force evaluation
    and must be capture-safe (no host syncs or allocations of state)::

        q = integrator.step_pre_force()                 # evaluation positions
        forces, energy = model(q)
        integrator.step_post_force(forces, energy=energy)

    Transactions: :meth:`backup` / :meth:`restore` snapshot the full state
    with device copies (recorded inside whole-step CUDA graphs);
    :meth:`snapshot` / :meth:`load_snapshot` return/restore host-held clones
    of state and backup (capture warmup, admission trials, precapture).

    Slot API (unified step 3.5, the evaluator-driven MD driver at any P):
    the state rows ``[0, n_owned)`` are this rank's atoms in the evaluator's
    row order, inside per-atom buffers of ``capacity`` rows. The driver
    hands the evaluator :meth:`positions32` and :meth:`migrating_fields`
    (the per-atom payload that migrates with the atoms); after an epoch
    refresh :meth:`reorder_` loads the returned rows in place and increments
    ``generation`` (snapshots of an older generation are refused). Native
    P=1 integrators keep ``capacity == n_owned`` (all N atoms) and
    ``generation == 0``.

    In-graph rows (unified step 3.5 PR C, slot integrators only): the P>1
    epoch whole-step graph runs the step halves on a FIXED row count, the
    topology's owned capacity, so one capture serves every epoch of its
    layout whatever the owned count. :meth:`reserve` makes the buffers hold
    that many rows; :meth:`step_pre_force_rows` / :meth:`step_post_force_rows`
    integrate rows ``[0, rows)`` with the device mask of the owned rows
    (padding rows stay zero, masses 1.0); :meth:`nonfinite_rows` is the
    step's non-finite flag. ``allocations`` counts buffer reallocations (a
    graph that reads the old buffers must be captured again).
    """

    positions: Tensor
    momenta: Tensor
    forces: Tensor
    potential_energy: Tensor
    generation: int = 0
    allocations: int = 0
    _slot_rows: int | None = None
    _slot_capacity: int | None = None

    @property
    def n_owned(self) -> int:
        """Rows ``[0, n_owned)`` of the state are this rank's atoms (native: all N)."""
        return int(self.positions.shape[0]) if self._slot_rows is None else self._slot_rows

    @n_owned.setter
    def n_owned(self, value: int) -> None:
        self._slot_rows = int(value)

    @property
    def capacity(self) -> int:
        """Rows allocated per per-atom field (native: ``n_owned``)."""
        return self.n_owned if self._slot_capacity is None else self._slot_capacity

    @capacity.setter
    def capacity(self, value: int) -> None:
        self._slot_capacity = int(value)

    def migrating_fields(self) -> dict[str, Tensor]:
        """Views ``[:n_owned]`` of the per-atom payload (``AtomState.payload`` names, shapes, dtypes)."""
        raise NotImplementedError

    def positions32(self) -> Tensor:
        """The evaluator's input positions, (n_owned, 3) in the evaluator dtype."""
        raise NotImplementedError

    def reorder_(self, state) -> bool:
        """Load the rows of ``state`` (an ``AtomState`` after an epoch refresh) in place; True if reallocated."""
        raise NotImplementedError

    def reserve(self, rows: int) -> bool:
        """At least ``rows`` rows per per-atom buffer (the owned rows are kept); True if reallocated."""
        raise NotImplementedError

    def step_pre_force_rows(self, rows: int) -> Tensor:
        """:meth:`step_pre_force` on rows ``[0, rows)`` with the owned-row mask; the float32 positions view."""
        raise NotImplementedError

    def step_post_force_rows(self, forces: Tensor, energy: Tensor | None, rows: int) -> None:
        """:meth:`step_post_force` on rows ``[0, rows)``; ``forces`` (rows, 3), padding rows ignored."""
        raise NotImplementedError

    def nonfinite_rows(self, rows: int) -> Tensor:
        """int64 (1,): 1 if a position or momentum of rows ``[0, rows)`` is not finite (device, no sync)."""
        raise NotImplementedError

    def step_pre_force(self) -> Tensor:
        raise NotImplementedError

    def step_post_force(self, forces: Tensor, energy: Tensor | None = None) -> None:
        raise NotImplementedError

    def set_forces(self, forces: Tensor, energy: Tensor | None = None) -> None:
        raise NotImplementedError

    def backup(self) -> None:
        raise NotImplementedError

    def restore(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError

    def load_snapshot(self, snapshot: object) -> None:
        raise NotImplementedError

    def kinetic_energy(self) -> Tensor:
        raise NotImplementedError

    def temperature(self) -> Tensor:
        raise NotImplementedError


class GPUIntegrator:
    """Small GPU-resident Velocity-Verlet / Berendsen NVT integrator.

    The equations intentionally mirror ASE's implementations for unconstrained
    systems.  Positions, momenta, forces, and masses stay as tensors, so an MD
    loop can consume model forces without copying them back to NumPy each step.

    A step is :meth:`step_pre_force` (Berendsen scaling, half kick with
    F(q_n), ``fix_com``, drift) -> forces F(q_{n+1}) -> :meth:`step_post_force`
    (half kick); :meth:`step` and the whole-step CUDA graphs
    (``WholeStepGraphRunner``, ``BatchedWholeStepGraphRunner``) run exactly
    that op sequence: the one Velocity-Verlet / Berendsen implementation.
    The Berendsen target is a device value (:meth:`set_temperature`, sync-free,
    between steps or graph replays), so a heating ramp needs no recapture.

    Domain decomposition (unified step 3.3): with a ``comm`` (the
    ``fastmd.engine.comm.Communicator`` protocol) the tensors are this rank's
    rows (a P>1 driver passes its current per-rank ``masses`` per call, as they
    migrate with the atoms), DOF come from ``n_global`` and the global
    quantities come from ONE ``comm.allreduce_sum_`` per step of the fused
    float64 vector ``[KE(p_n), sum p_n (3), sum F(q_n) (3)]``: the Berendsen
    scale uses KE, and ``fix_com`` subtracts the post-kick total momentum
    ``scale * sum p_n + dt/2 * sum F`` (linear in the reduced sums, so no
    second reduction) divided by ``n_global``. No reduction when neither is
    enabled; no host syncs. ``comm=None`` is the unchanged single-GPU path.
    """

    def __init__(
        self,
        masses: Tensor,
        dt_fs: float,
        ensemble: str = "nvt",
        temperature: float = 300.0,
        taut_fs: float = 100.0,
        fix_com: bool | None = None,
        degrees_of_freedom: int | None = None,
        *,
        comm=None,
        n_global: int | None = None,
    ) -> None:
        ensemble = ensemble.lower()
        if ensemble not in {"nve", "nvt"}:
            raise ValueError(
                f"unsupported GPUIntegrator ensemble: {ensemble} "
                "(Nose-Hoover chain: GPUNoseHooverChainNVT / ensemble='nvt_nhc')"
            )
        if comm is not None and (n_global is None or int(n_global) < 1):
            raise ValueError("a decomposed GPUIntegrator (comm) needs the global atom count n_global")
        self.masses = masses.reshape(-1, 1)
        self.dt = float(dt_fs) * units.fs
        self.ensemble = ensemble
        # The Berendsen target lives on the device (a CUDA graph input, so a
        # heating ramp needs no recapture), per momenta dtype; the float32
        # copy is the round-to-nearest value a Python float gets as a float32
        # multiply operand, so the scale is bitwise the scalar form's.
        target = float(temperature)
        self._target64 = torch.full((), target, dtype=torch.float64, device=self.masses.device)
        self._target32 = torch.full((), target, dtype=torch.float32, device=self.masses.device)
        self._upload_target = _PinnedUpload(self._target64)
        self._temperature_target = target
        self.taut = float(taut_fs) * units.fs
        self.fix_com = (ensemble == "nvt") if fix_com is None else bool(fix_com)
        self.comm = comm
        self.n_global = None if n_global is None else int(n_global)
        self.degrees_of_freedom = (
            int(degrees_of_freedom)
            if degrees_of_freedom is not None
            else int(self.masses.numel() * 3) if self.n_global is None
            else 3 * self.n_global
        )

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        """In the momenta's dtype (the Berendsen scale); the MD drivers report
        :func:`kinetic_energy_fp64`."""
        kinetic = 0.5 * momenta * momenta / self.masses
        if momenta.ndim == 2:
            return kinetic.sum()
        if momenta.ndim == 3:
            return kinetic.sum(dim=(-2, -1))
        raise ValueError(
            f"momenta must have shape [N, 3] or [B, N, 3], got {momenta.shape}"
        )

    def temperature(self, momenta: Tensor) -> Tensor:
        return 2.0 * self.kinetic_energy(momenta) / (
            self.degrees_of_freedom * units.kB
        )

    @property
    def temperature_target(self) -> float:
        """The Berendsen target (K), host mirror of the device value."""
        return self._temperature_target

    @temperature_target.setter
    def temperature_target(self, temperature: float) -> None:
        self.set_temperature(temperature)

    def set_temperature(self, temperature: float) -> None:
        """New Berendsen target (heating ramp) between steps or graph replays.

        Sync-free and stream ordered; raises inside a capture before anything
        changes (CPU: plain copy). ``set_temperature(T)`` equals a fresh
        integrator at ``T`` bitwise, and the current target changes no bit.
        """
        target = float(temperature)
        self._upload_target([target])
        self._target32.copy_(self._target64)
        self._temperature_target = target

    def target_row(self, temperature: float) -> list[float]:
        """The device target of ``temperature`` as one row of a multi-step window's per-step targets."""
        return [float(temperature)]

    def load_target_row(self, row: Tensor) -> None:
        """Load a :meth:`target_row` from the device (inside a captured multi-step window, before each
        step): the float64 value and its float32 rounding, as :meth:`set_temperature` (bitwise). The
        host mirror is the caller's (``set_temperature`` of the window's last target)."""
        self._target64.copy_(row[0])
        self._target32.copy_(row[0])

    def _berendsen_scale(self, old_temperature: Tensor) -> Tensor:
        """sqrt(1 + (T0 / T - 1) * dt / taut) clamped to [0.9, 1.1].

        ``T0 / T`` is ``T.reciprocal() * T0``: what ``float / Tensor``
        evaluates to, with the device target of ``T``'s dtype.
        """
        target = {torch.float32: self._target32, torch.float64: self._target64}.get(old_temperature.dtype)
        if target is None:
            raise TypeError(f"Berendsen needs float32/float64 momenta, got {old_temperature.dtype}")
        return torch.clamp(torch.sqrt(
            1.0 + (old_temperature.reciprocal() * target - 1.0) * (self.dt / self.taut)), 0.9, 1.1)

    def scale_velocities(self, momenta: Tensor) -> Tensor:
        if self.ensemble != "nvt":
            return momenta
        scale = self._berendsen_scale(self.temperature(momenta).clamp_min(1e-12))
        if momenta.ndim == 3:
            scale = scale[:, None, None]
        return momenta * scale

    def step_pre_force(
        self,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor,
        masses: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Berendsen scaling, p += dt/2 * F(q_n), ``fix_com``, drift: ``(q_{n+1}, p_{n+1/2})``.

        ``masses=None`` uses the constructor's masses.
        """
        masses = self.masses if masses is None else masses.reshape(-1, 1)
        if self.comm is not None:
            return self._decomposed_pre_force(positions, momenta, forces, masses)
        with nvtx_range("gpu_integrator"):
            momenta = self.scale_velocities(momenta)
        return self._kick_drift(positions, momenta, forces, masses)

    def step_post_force(self, momenta: Tensor, forces: Tensor) -> Tensor:
        """p += dt/2 * F(q_{n+1})."""
        with nvtx_range("gpu_integrator"):
            return momenta + 0.5 * self.dt * forces

    def _kick_drift(
        self, positions: Tensor, momenta: Tensor, forces: Tensor, masses: Tensor
    ) -> tuple[Tensor, Tensor]:
        with nvtx_range("gpu_integrator"):
            momenta = momenta + 0.5 * self.dt * forces
            if self.fix_com:
                atom_dim = -2
                momenta = momenta - momenta.sum(
                    dim=atom_dim, keepdim=True
                ) / float(momenta.shape[atom_dim])

            positions = positions + self.dt * momenta / masses
        return positions, momenta

    def _decomposed_pre_force(
        self, positions: Tensor, momenta: Tensor, forces: Tensor, masses: Tensor
    ) -> tuple[Tensor, Tensor]:
        """:meth:`step_pre_force` on this rank's rows; see the class docstring."""
        berendsen = self.ensemble == "nvt"
        with nvtx_range("gpu_integrator"):
            if berendsen or self.fix_com:
                p64 = momenta.to(torch.float64)
                totals = torch.cat((
                    (0.5 * p64 * p64 / masses.to(torch.float64)).sum().reshape(1),
                    p64.sum(dim=0), forces.to(torch.float64).sum(dim=0)))
                self.comm.allreduce_sum_(totals)
                total_momentum = totals[1:4]
            if berendsen:
                old_temperature = (2.0 * totals[0] / (
                    self.degrees_of_freedom * units.kB)).clamp_min(1e-12)
                scale = self._berendsen_scale(old_temperature)
                momenta = momenta * scale.to(momenta.dtype)
                total_momentum = total_momentum * scale
            momenta = momenta + 0.5 * self.dt * forces
            if self.fix_com:
                total_momentum = total_momentum + 0.5 * self.dt * totals[4:7]
                momenta = momenta - (total_momentum / self.n_global).to(momenta.dtype)
            positions = positions + self.dt * momenta / masses
        return positions, momenta

    def step(
        self,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor | None,
        force_fn: Callable[[Tensor], Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if forces is None:
            forces = force_fn(positions)
        positions, momenta = self.step_pre_force(positions, momenta, forces)
        forces = force_fn(positions)
        return positions, self.step_post_force(momenta, forces), forces


def slot_capacity(n: int) -> int:
    """Slot rows for ``n`` owned atoms: ``n + max(64, ceil(n / 8))`` rounded up to 64."""
    n = int(n)
    return -(-(n + max(64, -(-n // 8))) // 64) * 64


class SlotVelocityVerlet(StatefulIntegrator):
    """nve / Berendsen nvt of the evaluator-driven MD driver (any P): :class:`GPUIntegrator` on slot buffers.

    The state is ONE flat buffer ``[q 3S | p 3S | F 3S | m S | E 1]`` in the
    dtype of ``positions`` (the evaluator's: float32 for MatRIS) plus a
    backup of the same layout; S = :attr:`capacity` (default
    :func:`slot_capacity`). Rows ``[0, n_owned)`` are this rank's atoms in
    the evaluator's row order; padding rows are zero (masses 1.0) and never
    read. The step halves are ``self.core``'s (``GPUIntegrator(comm=comm,
    n_global=...)``, the masses passed per call) on the row views, copied
    back in place: bitwise the per-call path of the step 3.3 driver (one fused
    ``comm.allreduce_sum_`` per step for nvt or ``fix_com``, none for plain
    nve). ``comm`` is required (``LocalCommunicator()`` at P=1);
    ``n_global`` defaults to the owned count at one rank. ``fix_com=None``
    is the MD drivers' rule (:func:`default_fix_com`).

    :meth:`reorder_` loads a refreshed row order into the same buffers
    (growing them only when the owned count exceeds the capacity); the
    forces are not carried (F(q_n) was consumed by the pre-force half, the
    post-force half writes F(q_{n+1}) on the new rows).
    """

    def __init__(self, masses: Tensor, positions: Tensor, momenta: Tensor, forces: Tensor | None = None,
                 energy: Tensor | None = None, *, dt_fs: float, ensemble: str = "nvt",
                 temperature: float = 300.0, taut_fs: float = 100.0, fix_com: bool | None = None,
                 comm, n_global: int | None = None, capacity: int | None = None) -> None:
        if comm is None:
            raise ValueError("SlotVelocityVerlet needs a comm (LocalCommunicator() at one rank)")
        n = int(positions.shape[0])
        if n_global is None:
            if int(getattr(comm, "world_size", 1)) != 1:
                raise ValueError("SlotVelocityVerlet at P>1 needs the global atom count n_global")
            n_global = n
        self.dtype, self.device = positions.dtype, positions.device
        self._allocate(slot_capacity(n) if capacity is None else int(capacity), n)
        self._load_rows(positions, momenta, masses, n)
        if forces is not None:
            self.set_forces(forces, energy)
        elif energy is not None:
            self._E.copy_(energy.reshape(1))
        self.comm, self.n_global = comm, int(n_global)
        self.core = GPUIntegrator(
            self._m[:n], dt_fs, ensemble=ensemble, temperature=temperature, taut_fs=taut_fs,
            fix_com=default_fix_com(ensemble) if fix_com is None else bool(fix_com),
            comm=comm, n_global=self.n_global)
        self.ensemble, self.fix_com, self.dt = self.core.ensemble, self.core.fix_com, self.core.dt
        self.degrees_of_freedom = self.core.degrees_of_freedom
        self.backup()

    # ------------------------------------------------------------------ slots
    def _allocate(self, capacity: int, n: int) -> None:
        if capacity < n:
            raise ValueError(f"slot capacity {capacity} < {n} owned atoms")
        s = -(-int(capacity) // 4) * 4  # every field's rows start 4-element (vector) aligned
        self._flat = torch.zeros(10 * s + 1, dtype=self.dtype, device=self.device)
        self._backup = torch.zeros_like(self._flat)
        self._q = self._flat[:3 * s].view(s, 3)
        self._p = self._flat[3 * s:6 * s].view(s, 3)
        self._F = self._flat[6 * s:9 * s].view(s, 3)
        self._m = self._flat[9 * s:10 * s]
        self._E = self._flat[10 * s:]
        self._m.fill_(1.0)
        self._real = torch.zeros(s, dtype=torch.bool, device=self.device)  # rows [0, n_owned)
        self.capacity = s
        self.allocations += 1

    @torch.no_grad()
    def _load_rows(self, positions: Tensor, momenta: Tensor, masses: Tensor, n: int) -> None:
        if (positions.shape != (n, 3) or momenta.shape != (n, 3) or masses.numel() != n
                or masses.shape not in ((n,), (n, 1))):
            raise ValueError(f"slot rows need positions/momenta (n, 3) and masses (n,), got "
                             f"{tuple(positions.shape)}, {tuple(momenta.shape)}, {tuple(masses.shape)}")
        self._q[:n].copy_(positions)
        self._p[:n].copy_(momenta)
        self._m[:n].copy_(masses.reshape(n))
        self._q[n:].zero_()
        self._p[n:].zero_()
        self._F.zero_()
        self._m[n:].fill_(1.0)
        self._real.zero_()
        self._real[:n] = True
        self.n_owned = n
        if hasattr(self, "core"):  # unused by the decomposed core (masses per call); kept current
            self.core.masses = self._m[:n].reshape(-1, 1)

    @torch.no_grad()
    def reorder_(self, state) -> bool:
        """The owned rows after an epoch refresh: ``state.positions`` and the payload "momenta", "masses".

        In place into rows ``[0, n')`` (zero padding, masses 1.0, forces
        cleared, or loaded from a payload "forces": the refresh of the P>1
        in-graph step runs before the pre-force half, which reads them);
        reallocates (capacity :func:`slot_capacity` of n') only when n'
        exceeds the capacity. The backup follows; ``generation`` increments.
        Returns True if the buffers were reallocated.
        """
        sources = [state.positions, state.payload["momenta"], state.payload["masses"]]
        if "forces" in state.payload:
            sources.append(state.payload["forces"])
        storage = self._flat.untyped_storage().data_ptr()
        sources = [t.clone() if t.untyped_storage().data_ptr() == storage else t for t in sources]
        n = int(sources[0].shape[0])
        grown = n > self.capacity
        if grown:
            energy = self._E.clone()
            self._allocate(slot_capacity(n), n)
            self._E.copy_(energy)
        self._load_rows(*sources[:3], n)
        if len(sources) > 3:
            self._F[:n].copy_(sources[3].reshape(n, 3))
        self.backup()
        self.generation += 1
        return grown

    @torch.no_grad()
    def reserve(self, rows: int) -> bool:
        """At least ``rows`` slot rows (rounded up to 4); the owned rows, forces and energy are kept."""
        if int(rows) <= self.capacity:
            return False
        n = self.n_owned
        rows_n = [x[:n].clone() for x in (self._q, self._p, self._m, self._F)]
        energy = self._E.clone()
        self._allocate(int(rows), n)
        self._load_rows(*rows_n[:3], n)
        self._F[:n].copy_(rows_n[3])
        self._E.copy_(energy)
        self.backup()
        self.generation += 1
        return True

    def migrating_fields(self) -> dict[str, Tensor]:
        """``{"masses": (n,), "momenta": (n, 3)}`` views (the evaluator payload)."""
        n = self.n_owned
        return {"masses": self._m[:n], "momenta": self._p[:n]}

    def positions32(self) -> Tensor:
        return self._q[:self.n_owned]

    @property
    def positions(self) -> Tensor:
        return self._q[:self.n_owned]

    @property
    def momenta(self) -> Tensor:
        return self._p[:self.n_owned]

    @property
    def forces(self) -> Tensor:
        return self._F[:self.n_owned]

    @property
    def masses(self) -> Tensor:
        return self._m[:self.n_owned]

    @property
    def potential_energy(self) -> Tensor:
        return self._E

    # ------------------------------------------------------------ step halves
    @torch.no_grad()
    def step_pre_force(self) -> Tensor:
        """Berendsen scaling, p += dt/2 * F(q_n), ``fix_com``, drift on the owned rows; returns q_{n+1}."""
        n = self.n_owned
        q, p = self._q[:n], self._p[:n]
        positions, momenta = self.core.step_pre_force(q, p, self._F[:n], masses=self._m[:n])
        q.copy_(positions)
        p.copy_(momenta)
        return q

    @torch.no_grad()
    def step_post_force(self, forces: Tensor, energy: Tensor | None = None) -> None:
        """p += dt/2 * F(q_{n+1}); caches F(q_{n+1}) (and E)."""
        p = self._p[:self.n_owned]
        p.copy_(self.core.step_post_force(p, forces))
        self.set_forces(forces, energy)

    @torch.no_grad()
    def set_forces(self, forces: Tensor, energy: Tensor | None = None) -> None:
        self._F[:self.n_owned].copy_(forces.reshape(self.n_owned, 3))
        if energy is not None:
            self._E.copy_(energy.reshape(1))

    @torch.no_grad()
    def step_pre_force_rows(self, rows: int) -> Tensor:
        """:meth:`step_pre_force` on rows ``[0, rows)`` (capture-safe): padding rows carry zero momenta,
        forces and sums, and keep their zero positions and momenta (masked)."""
        q, p = self._q[:rows], self._p[:rows]
        positions, momenta = self.core.step_pre_force(q, p, self._F[:rows], masses=self._m[:rows])
        real = self._real[:rows, None]
        q.copy_(torch.where(real, positions, q))
        p.copy_(torch.where(real, momenta, p))
        return q

    @torch.no_grad()
    def step_post_force_rows(self, forces: Tensor, energy: Tensor | None, rows: int) -> None:
        """:meth:`step_post_force` on rows ``[0, rows)`` (capture-safe); padding rows of ``forces`` are ignored."""
        forces = torch.where(self._real[:rows, None], forces.reshape(rows, 3), 0.0)
        p = self._p[:rows]
        p.copy_(self.core.step_post_force(p, forces))
        self._F[:rows].copy_(forces)
        if energy is not None:
            self._E.copy_(energy.reshape(1))

    def nonfinite_rows(self, rows: int) -> Tensor:
        finite = torch.isfinite(self._q[:rows]).all() & torch.isfinite(self._p[:rows]).all()
        return (~finite).reshape(1).long()

    # ------------------------------------------------------------- thermostat
    @property
    def temperature_target(self) -> float:
        return self.core.temperature_target

    @temperature_target.setter
    def temperature_target(self, temperature: float) -> None:
        self.core.set_temperature(temperature)

    def set_temperature(self, temperature: float) -> None:
        """New Berendsen target (sync-free; see :meth:`GPUIntegrator.set_temperature`)."""
        self.core.set_temperature(temperature)

    # ------------------------------------------------------------ observables
    def kinetic_energy(self) -> Tensor:
        """Global ``kinetic_energy_fp64`` (0-dim float64). Collective: every rank calls it."""
        n = self.n_owned
        total = kinetic_energy_fp64(self._p[:n], self._m[:n]).reshape(1)
        self.comm.allreduce_sum_(total)
        return total.reshape(())

    def temperature(self) -> Tensor:
        """2 KE / (3 n_global kB). Collective."""
        return 2.0 * self.kinetic_energy() / (self.degrees_of_freedom * units.kB)

    # ------------------------------------------------------------ transactions
    def backup(self) -> None:
        self._backup.copy_(self._flat)

    def restore(self) -> None:
        self._flat.copy_(self._backup)

    def snapshot(self) -> tuple[Tensor, Tensor, int]:
        """Clones of state and backup, and the generation they belong to."""
        return self._flat.clone(), self._backup.clone(), self.generation

    def load_snapshot(self, snapshot: tuple[Tensor, Tensor, int]) -> None:
        state, backup, generation = snapshot
        if generation != self.generation:
            raise ValueError("snapshot predates an epoch refresh")
        self._flat.copy_(state)
        self._backup.copy_(backup)
