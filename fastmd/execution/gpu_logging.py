from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory
from ase.stress import full_3x3_to_voigt_6_stress
from torch import Tensor

from fastmd._nvtx import nvtx_range


@dataclass
class _FrameSlot:
    positions: Tensor
    momenta: Tensor
    forces: Tensor
    cell: Tensor
    scalars: Tensor
    extras: dict[str, Tensor]
    stress: Tensor | None = None
    event: torch.cuda.Event | None = None


@dataclass
class _PendingFrame:
    slot: _FrameSlot
    step: int
    time_ps: float


class AsyncGPUMDLogger:
    """Asynchronous trajectory/log writer for GPU-resident MD.

    The hot loop enqueues non-blocking CUDA -> pinned-CPU copies.  A background
    writer waits on the CUDA event and performs ASE trajectory/text I/O.
    If the writer fails, it records the error and keeps releasing ring slots
    (dropping later frames), and the next ``enqueue``/``flush``/``close`` in
    the calling thread raises it, so the caller never waits on a dead writer.

    ``state_dtype`` is the dtype of the logged positions/momenta (float64 for
    integrators with a float64 state). ``extras`` maps names to lengths of
    additional float64 per-frame vectors (e.g. thermostat variables); they
    are written to the trajectory frames' ``atoms.info``. With ``stress``
    every frame carries a 3x3 stress tensor in eV/A^3 (``enqueue(stress=...)``),
    stored in the trajectory frame's calculator in Voigt order, so that
    ``atoms.get_stress()`` returns what it returns for a calculator's 3x3
    result; the text log is unchanged. ``value_dtype`` is the dtype of the
    logged forces, cell and stress (float64 for a float64 model).
    """

    def __init__(
        self,
        atoms_template: Atoms,
        device: torch.device,
        trajectory=None,
        logfile: str | Path | IO[str] | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        ring: int = 8,
        state_dtype: torch.dtype = torch.float32,
        extras: dict[str, int] | None = None,
        stress: bool = False,
        value_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.enabled = trajectory is not None or logfile is not None
        self.state_dtype = state_dtype
        self.extras = dict(extras or {})
        self.stress = bool(stress)
        self.loginterval = max(int(loginterval), 1)
        self.device = device
        self.atoms_template = atoms_template.copy()
        self._stop = object()
        self._available: queue.Queue[_FrameSlot] = queue.Queue(maxsize=ring)
        self._pending: queue.Queue[_PendingFrame | object] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._traj = None
        self._log = None
        self._close_log = False
        # (step, exception) of the writer's first failure
        self._failure: tuple[int, BaseException] | None = None
        self._failure_reported = False

        if not self.enabled:
            return

        if trajectory is not None:
            if isinstance(trajectory, (str, Path)):
                mode = "a" if append_trajectory else "w"
                self._traj = Trajectory(
                    str(trajectory), mode, self.atoms_template
                )
            else:
                self._traj = trajectory

        if logfile is not None:
            if logfile == "-":
                self._log = sys.stdout
            elif isinstance(logfile, (str, Path)):
                self._log = open(logfile, "a", encoding="utf-8")
                self._close_log = True
            else:
                self._log = logfile
            self._write_log_header()

        pin = device.type == "cuda"
        n_atoms = len(self.atoms_template)
        for _ in range(max(int(ring), 1)):
            self._available.put(
                _FrameSlot(
                    positions=torch.empty((n_atoms, 3), dtype=state_dtype, pin_memory=pin),
                    momenta=torch.empty((n_atoms, 3), dtype=state_dtype, pin_memory=pin),
                    forces=torch.empty((n_atoms, 3), dtype=value_dtype, pin_memory=pin),
                    cell=torch.empty((3, 3), dtype=value_dtype, pin_memory=pin),
                    scalars=torch.empty((4,), dtype=torch.float64, pin_memory=pin),
                    extras={
                        name: torch.empty((int(size),), dtype=torch.float64, pin_memory=pin)
                        for name, size in self.extras.items()
                    },
                    stress=(
                        torch.empty((3, 3), dtype=value_dtype, pin_memory=pin)
                        if self.stress
                        else None
                    ),
                )
            )
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _write_log_header(self) -> None:
        if self._log is None:
            return
        natoms = len(self.atoms_template)
        if natoms <= 100:
            digits = 4
        elif natoms <= 1000:
            digits = 3
        elif natoms <= 10000:
            digits = 2
        else:
            digits = 1
        self._log_digits = digits
        self._log.write(
            "%-9s %12s %12s %12s  %6s\n"
            % ("Time[ps]", "Etot[eV]", "Epot[eV]", "Ekin[eV]", "T[K]")
        )
        self._log.flush()

    def should_log(self, step: int) -> bool:
        return self.enabled and step % self.loginterval == 0

    def enqueue(
        self,
        *,
        step: int,
        time_ps: float,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor,
        cell: Tensor,
        potential_energy: Tensor,
        kinetic_energy: Tensor,
        temperature: Tensor,
        extras: dict[str, Tensor] | None = None,
        stress: Tensor | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._raise_writer_failure()
        extras = extras or {}
        if set(extras) != set(self.extras):
            raise ValueError(
                f"logger extras {sorted(self.extras)} != enqueued {sorted(extras)}"
            )
        if (stress is not None) != self.stress:
            raise ValueError(
                f"logger stress={self.stress}, but the enqueued frame "
                f"{'has' if stress is not None else 'has no'} stress"
            )
        if stress is not None and stress.numel() != 9:
            raise ValueError(f"the stress must be a 3x3 tensor, got shape {tuple(stress.shape)}")
        with nvtx_range("logger enqueue"):
            slot = self._available.get()
            scalars = torch.stack(
                (
                    potential_energy.reshape(()).to(torch.float64),
                    kinetic_energy.reshape(()).to(torch.float64),
                    temperature.reshape(()).to(torch.float64),
                    torch.zeros((), dtype=torch.float64, device=potential_energy.device),
                )
            )
            if self._stream is None:
                slot.positions.copy_(positions.detach().cpu())
                slot.momenta.copy_(momenta.detach().cpu())
                slot.forces.copy_(forces.detach().cpu())
                slot.cell.copy_(cell.detach().cpu())
                slot.scalars.copy_(scalars.detach().cpu())
                for name, value in extras.items():
                    slot.extras[name].copy_(value.detach().reshape(-1).cpu())
                if stress is not None:
                    slot.stress.copy_(stress.detach().reshape(3, 3).cpu())
                slot.event = None
            else:
                producer = torch.cuda.current_stream(self.device)
                self._stream.wait_stream(producer)
                with torch.cuda.stream(self._stream):
                    slot.positions.copy_(positions.detach(), non_blocking=True)
                    slot.momenta.copy_(momenta.detach(), non_blocking=True)
                    slot.forces.copy_(forces.detach(), non_blocking=True)
                    slot.cell.copy_(cell.detach(), non_blocking=True)
                    slot.scalars.copy_(scalars.detach(), non_blocking=True)
                    for name, value in extras.items():
                        slot.extras[name].copy_(
                            value.detach().reshape(-1), non_blocking=True
                        )
                    if stress is not None:
                        slot.stress.copy_(stress.detach().reshape(3, 3), non_blocking=True)
                    slot.event = torch.cuda.Event()
                    slot.event.record(self._stream)
                # Write-after-read guard: the sources are the MD state buffers
                # (overwritten in place by the next whole-step replay) and
                # temporaries whose blocks the caching allocator hands out
                # again on the producer stream. Later producer work must not
                # start before the copy has read them. The host never blocks.
                # Trade-off: on logged steps the next step's kernels wait for
                # this D2H copy, so a copy engine saturated by other traffic
                # delays the MD stream (copying the sources to a device
                # staging buffer first would avoid it, at one extra D2D copy
                # and buffer per logged frame).
                producer.wait_event(slot.event)
            self._pending.put(_PendingFrame(slot=slot, step=int(step), time_ps=float(time_ps)))

    def _writer_loop(self) -> None:
        while True:
            item = self._pending.get()
            if item is self._stop:
                self._pending.task_done()
                return
            # Whatever happens, release the slot and mark the item done:
            # enqueue() waits for free slots and flush()/close() join the
            # queue. After the first failure later frames are dropped (not
            # written behind a gap); the caller thread re-raises the failure.
            try:
                if self._failure is None:
                    if item.slot.event is not None:
                        item.slot.event.synchronize()
                    self._write_frame(item)
            except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
                self._failure = (item.step, exc)
            finally:
                self._available.put(item.slot)
                self._pending.task_done()

    def _raise_writer_failure(self) -> None:
        failure = self._failure
        if failure is None:
            return
        step, exc = failure
        self._failure_reported = True
        detail = str(exc).strip().splitlines()  # full error: __cause__
        raise RuntimeError(
            "AsyncGPUMDLogger writer thread failed on the frame of step "
            f"{step} ({type(exc).__name__}: {detail[0] if detail else ''}); "
            "the trajectory/log is incomplete from that frame on"
        ) from exc

    def _write_frame(self, frame: _PendingFrame) -> None:
        scalars = frame.slot.scalars.numpy()
        epot = float(scalars[0])
        ekin = float(scalars[1])
        temp = float(scalars[2])

        if self._traj is not None:
            atoms = self.atoms_template.copy()
            atoms.set_cell(np.array(frame.slot.cell.numpy(), copy=True), scale_atoms=False)
            atoms.set_positions(np.array(frame.slot.positions.numpy(), copy=True))
            atoms.set_momenta(np.array(frame.slot.momenta.numpy(), copy=True))
            # Voigt (xx, yy, zz, yz, xz, xy) of the 3x3, as Atoms.get_stress does for a calculator's 3x3
            stress = {} if frame.slot.stress is None else {
                "stress": full_3x3_to_voigt_6_stress(frame.slot.stress.numpy())
            }
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=epot,
                forces=np.array(frame.slot.forces.numpy(), copy=True),
                **stress,
            )
            for name, value in frame.slot.extras.items():
                atoms.info[name] = np.array(value.numpy(), copy=True)
            self._traj.write(atoms)

        if self._log is not None:
            digits = getattr(self, "_log_digits", 3)
            fmt = "%-10.4f " + 3 * (f"%12.{digits}f ") + " %6.1f\n"
            self._log.write(fmt % (frame.time_ps, epot + ekin, epot, ekin, temp))
            self._log.flush()

        # Non-finite state check on values already on the host (no extra sync).
        # NaN positions make every atom isolated: the reference energy stays
        # finite and forces are zero, so the energy alone would not show it.
        bad = int((~np.isfinite(frame.slot.positions.numpy()).all(axis=1)).sum())
        stress_finite = frame.slot.stress is None or bool(np.isfinite(frame.slot.stress.numpy()).all())
        if bad or not np.isfinite(scalars[:3]).all() or not stress_finite:
            raise ValueError(
                f"non-finite MD state (frame written): Epot={epot}, Ekin={ekin}, T={temp}, "
                f"{bad} atoms with non-finite positions" + ("" if stress_finite else ", non-finite stress")
            )

    @property
    def copy_stream(self):
        """The CUDA stream of the device-to-host copies (None on a CPU device or without outputs). A frame
        enqueued while it is the current stream is copied without making any other stream wait: the
        caller keeps the frame's sources unchanged until that stream has read them."""
        return getattr(self, "_stream", None)

    def flush(self) -> None:
        if not self.enabled:
            return
        self._pending.join()
        self._raise_writer_failure()

    def close(self) -> None:
        if not self.enabled:
            return
        # Drain and stop the writer before raising its failure, so a failed
        # logger still releases its thread and files.
        self._pending.join()
        self._pending.put(self._stop)
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._traj is not None and hasattr(self._traj, "close"):
            self._traj.close()
        if self._log is not None and self._close_log:
            self._log.close()
        self.enabled = False
        # A failure already raised by enqueue()/flush() is not raised again,
        # e.g. from a context manager's __exit__ after the failed run().
        if not self._failure_reported:
            self._raise_writer_failure()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
