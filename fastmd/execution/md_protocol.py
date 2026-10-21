"""MD protocol pieces shared by the MD drivers: the heating ramp, the outputs and the restart file.

:class:`HeatingRamp` sets the thermostat target of
:class:`~fastmd.interfaces.gpu_md.GPUResidentMolecularDynamics` (the native
P=1 engine and the ``from_evaluator`` driver at any P); the target is a device
value of the integrators, so a ramp needs no CUDA graph recapture.

:class:`MDOutputs` writes a driver's output files at any P: rank 0 owns the
files and every rank joins the gathers (``comm.gather_by_id``, the
Communicator protocol; nothing here imports ``fastmd.distributed``). The files
follow md_pyrolysis, the external single-GPU heating MD script whose option
names the CLI keeps: thermo text, EXTXYZ frames with velocities, and an
ID-ordered, P-independent restart ``.npz`` (:func:`write_restart`,
:func:`read_restart`) that records the other files' offsets for truncation on
resume.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import weakref

from ase import units
import numpy as np
import torch

ENSEMBLES = ("nve", "nvt", "nvt_nhc")
THERMO_FILE, TRAJECTORY_FILE, RESTART_FILE = "md_thermo.txt", "md_trajectory.extxyz", "md_restart.npz"
THERMO_HEADER = ("# Step\tTime(fs)\tTargetTemperature(K)\tTemperature(K)\t"
                 "PotentialEnergy(eV)\tKineticEnergy(eV)\tTotalEnergy(eV)\n")
# The format name predates this module (fastmd.distributed.md); unchanged so older files load.
RESTART_FORMAT = "fastmd.distributed.md/1"
_ERRORS = (TypeError, ValueError, RuntimeError, AttributeError, IndexError, KeyError)


@dataclass(frozen=True)
class HeatingRamp:
    """md_pyrolysis's linear ramp (``--update-interval U``).

    Before the chunk starting at step s (s % U == 0; length L = min(U, total - s))
    the thermostat target becomes ``start + (end - start) * ((s + 0.5 * L) / total)``;
    from ``total_steps`` on it stays at ``end_K``. The thermo column is the ramp
    value at the step, ``start + (end - start) * (step / total)``. Host numbers
    only, so every rank computes the same bits.
    """

    start_K: float
    end_K: float
    total_steps: int
    update_interval: int = 1

    def __post_init__(self):
        if not all(math.isfinite(value) and value > 0 for value in (self.start_K, self.end_K)):
            raise ValueError("ramp temperatures must be finite and positive")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (self.total_steps, self.update_interval)):
            raise ValueError("ramp total_steps and update_interval must be positive integers")

    def setpoint(self, step: int) -> float | None:
        """The thermostat target to set before ``step``, or None inside a chunk."""
        if step % self.update_interval:
            return None
        if step >= self.total_steps:
            return float(self.end_K)
        length = min(self.update_interval, self.total_steps - step)
        return self.start_K + (self.end_K - self.start_K) * ((step + 0.5 * length) / self.total_steps)

    def target(self, step: int) -> float:
        return self.start_K + (self.end_K - self.start_K) * (min(step, self.total_steps) / self.total_steps)


def initialize_momenta(atoms, temperature_K: float, seed: int) -> None:
    """md_pyrolysis's initial momenta on the root's ASE Atoms (in place).

    ``np.random.seed(seed)``, ``MaxwellBoltzmannDistribution``, ``Stationary``,
    then an exact rescale to ``temperature_K`` with ``max(1, 3N - 3)`` DOF.
    """
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    np.random.seed(int(seed))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)
    ndof = max(1, 3 * len(atoms) - 3)
    current = 2.0 * atoms.get_kinetic_energy() / (ndof * units.kB)
    if current <= 0:
        raise RuntimeError("Initial kinetic temperature is non-positive.")
    atoms.set_momenta(atoms.get_momenta() * np.sqrt(temperature_K / current))


class ExtxyzTrajectory:
    """Truncatable EXTXYZ ``trajectory`` for AsyncGPUMDLogger (md_pyrolysis frames).

    Frames carry velocities instead of momenta, energy and forces, and the
    logger's per-frame extras as scalar ``info`` entries (``step`` an integer).
    Every frame is flushed, so after ``logger.flush()`` the file size is a frame
    boundary: the offset a restart records.
    """

    def __init__(self, path, *, append: bool):
        self._stream = open(path, "a" if append else "w", encoding="utf-8")

    def write(self, atoms) -> None:
        from ase.io import write

        velocities = atoms.get_velocities()
        atoms.arrays.pop("momenta", None)
        atoms.arrays["velocities"] = velocities
        for key, value in list(atoms.info.items()):
            if isinstance(value, np.ndarray) and value.size == 1:
                atoms.info[key] = int(value.item()) if key == "step" else float(value.item())
        write(self._stream, atoms, format="extxyz")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


@dataclass
class RestartData:
    """One restart file as host (numpy) arrays, rows in ID order.

    ``ids``/``numbers`` int64; ``masses``, ``positions`` (float64 for nvt_nhc,
    else the float32 values) and ``momenta`` float64; ``forces``, ``energy``
    and ``reference_energy`` the cached evaluation of that state (forces
    float32; a continuation does not re-evaluate); ``step`` the step count;
    ``parameters`` the driver's (see :func:`resume_mismatch`);
    ``temperature_K`` the thermostat target (None for nve); ``nhc_state`` the
    NHC ``state_dict`` (nvt_nhc, else None); ``offsets`` the output files'
    sizes; ``engine`` the writer, "domain" (the evaluator-driven driver) or
    "native" (the single-GPU engine). It is not part of ``parameters``.
    """

    ids: np.ndarray
    numbers: np.ndarray
    masses: np.ndarray
    positions: np.ndarray
    momenta: np.ndarray
    forces: np.ndarray
    energy: np.ndarray
    reference_energy: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    step: int
    parameters: dict
    temperature_K: float | None
    nhc_state: dict | None
    offsets: dict
    engine: str


def read_restart(path) -> RestartData:
    """Parse a :func:`write_restart` file (format check only; the caller checks cell and atoms).

    A file without the "engine" field (written before it existed) reads as "domain".
    """
    with np.load(path) as data:
        if str(data["format"]) != RESTART_FORMAT:
            raise ValueError(f"not a {RESTART_FORMAT} restart file")
        parameters = json.loads(str(data["parameters"]))
        temperature = float(data["temperature_K"])
        nhc = parameters["ensemble"] == "nvt_nhc"
        return RestartData(
            ids=data["ids"], numbers=data["numbers"], masses=data["masses"], positions=data["positions"],
            momenta=data["momenta"], forces=data["forces"], energy=data["potential_energy"],
            reference_energy=data["reference_energy"], cell=data["cell"], pbc=data["pbc"],
            step=int(data["step"]), parameters=parameters,
            temperature_K=None if math.isnan(temperature) else temperature,
            nhc_state={"temperature_K": temperature, "eta": np.array(data["nhc_eta"]),
                       "p_eta": np.array(data["nhc_p_eta"])} if nhc else None,
            offsets=json.loads(str(data["offsets"])),
            engine=str(data["engine"]) if "engine" in data.files else "domain")


def write_restart(path, data: RestartData) -> None:
    """Write ``data`` atomically (temporary file + os.replace) as an ``.npz``."""
    path = Path(path)
    nhc = {} if data.nhc_state is None else data.nhc_state
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, format=np.array(RESTART_FORMAT), ids=data.ids, numbers=data.numbers,
             masses=data.masses, positions=data.positions, momenta=data.momenta, forces=data.forces,
             potential_energy=data.energy, reference_energy=data.reference_energy, cell=data.cell,
             pbc=data.pbc, step=np.array(data.step, dtype=np.int64),
             parameters=np.array(json.dumps(data.parameters)),
             temperature_K=np.array(math.nan if data.temperature_K is None else data.temperature_K),
             nhc_eta=nhc.get("eta", np.zeros(0)), nhc_p_eta=nhc.get("p_eta", np.zeros(0)),
             offsets=np.array(json.dumps(data.offsets)), engine=np.array(data.engine))
    os.replace(temporary, path)


def resume_mismatch(saved: dict, requested: dict, *, ignore=()) -> dict:
    """``{key: (saved, requested)}`` for every parameter a continuation changes (sorted keys).

    ``total_steps`` is always ignored (a run without a ramp may be extended; a
    ramp's total is part of "ramp"), and so are the keys in ``ignore``.
    """
    skip = {"total_steps", *ignore}
    return {key: (saved.get(key), requested.get(key)) for key in sorted(saved.keys() | requested.keys())
            if key not in skip and saved.get(key) != requested.get(key)}


class MDOutputs:
    """A driver's output files: rank 0 owns them, every rank joins the gathers (all ranks call every method).

    ``source`` is the driver. It provides ``comm`` (``rank``,
    ``gather_by_id``), ``nsteps``, ``parameters`` (stored in the restart file;
    "ensemble" and "tchain" select the frame extras), ``cell`` (the restart
    file's and frames' cell), ``device`` (the logger's), ``md_fields()``,
    ``thermodynamics()`` (collective), ``thermostat_state()`` and
    ``collective_check(error)`` (every rank raises any rank's error).
    ``md_fields()`` returns views of the owned rows: "ids", "numbers",
    "positions" (the integrator's, float64 for nvt_nhc), "momenta", "masses"
    and the cached "forces", plus the cached "energy" and "reference_energy".

    Thermo line every ``thermo_interval`` steps, EXTXYZ frame every
    ``xyz_interval``, restart file every ``restart_interval`` (0 disables each;
    ``output_dir`` None disables all, but :meth:`save_restart` to a given path
    still works). ``total_steps`` marks the last step, which also gets the
    (enabled) thermo line and restart file. A fresh start (``offsets`` None)
    removes earlier outputs (thermo, frames and restart file) and creates the
    files (frame and restart at step 0 when enabled, as md_pyrolysis); a
    resume passes the restart's ``offsets`` and the files are truncated to
    them (a file shorter than its offset is rejected). ``engine`` is the
    restart files' "engine" field. The restart's ``pbc`` is ``source.pbc``
    when the driver has one (the native engine), else all periodic.

    Rank 0's file errors (the files' preparation and frame logger at
    construction, thermo append, the frame logger's writer failure, the
    restart write) are agreed: the constructor, :meth:`write`,
    :meth:`log_frame` and :meth:`save_restart` end with ONE
    ``source.collective_check`` on the steps that write anything, so every
    rank raises rank 0's error and no rank is left waiting in the next
    collective. :meth:`close` is rank-local (the driver agrees it).
    """

    def __init__(self, source, output_dir, thermo_interval: int, xyz_interval: int, restart_interval: int,
                 total_steps: int | None, offsets: dict | None, *, engine: str = "domain"):
        # A proxy: the driver owns its outputs, and a reference cycle would keep a dropped
        # driver's evaluator (and its CUDA graphs) alive until a GC pass.
        self.source, self.comm, self.engine = weakref.proxy(source), source.comm, engine
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.thermo_interval, self.xyz_interval = thermo_interval, xyz_interval
        self.restart_interval, self.total_steps = restart_interval, total_steps
        self.logger, self._tracked = None, {}
        if self.output_dir is None:
            return
        root, error = self.comm.rank == 0, None
        # The frame template's numbers and masses (collective), before root's file checks:
        # root builds its logger inside the agreed try below.
        fields = source.md_fields() if xyz_interval else None
        frame = (self.comm.gather_by_id(0, fields["ids"], numbers=fields["numbers"], masses=fields["masses"])
                 if xyz_interval else None)
        if root:
            # Both files have offsets even when this run writes no frames: a
            # later segment that does must not lose the earlier ones.
            self._tracked = {name: self.output_dir / name for name in (THERMO_FILE, TRAJECTORY_FILE)}
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                if offsets is None:
                    # No earlier run's restart may later resume into these files.
                    for path in (*self._tracked.values(), self.output_dir / RESTART_FILE):
                        path.unlink(missing_ok=True)
                    with open(self._tracked[THERMO_FILE], "w") as stream:
                        stream.write(THERMO_HEADER)
                else:
                    # Drop anything written after the checkpoint (md_pyrolysis). Files
                    # without an offset are left alone; a shorter one is another run's.
                    sizes = {name: int(offsets[name]) for name, path in self._tracked.items()
                             if name in offsets and path.exists()}
                    short = [name for name, size in sizes.items() if self._tracked[name].stat().st_size < size]
                    if short:
                        raise ValueError(f"cannot resume: {short} shorter than the restart file's offsets")
                    for name, size in sizes.items():
                        with open(self._tracked[name], "r+b") as stream:
                            stream.truncate(size)
                if xyz_interval:
                    self.logger = self._frame_logger(frame, fields["positions"].dtype, offsets is not None)
            except (OSError, *_ERRORS) as exc:
                error = str(exc)
        source.collective_check(error)
        if offsets is None and (xyz_interval or restart_interval):
            # md_pyrolysis writes the initial frame and restart file.
            error = self._log_frame(None) if xyz_interval else None
            if restart_interval:
                error = self._save_restart(None) or error
            try:
                source.collective_check(error)
            except BaseException:
                self.close_quietly()  # the constructor raises: nobody else closes the writer
                raise

    def _frame_logger(self, frame: dict, state_dtype, append: bool):
        """Root's EXTXYZ frame logger (the template from the gathered numbers and masses)."""
        from ase import Atoms
        from fastmd.execution.gpu_logging import AsyncGPUMDLogger
        source = self.source
        template = Atoms(numbers=frame["numbers"].cpu().numpy(),
                         cell=source.cell.detach().cpu().double().numpy(), pbc=self._pbc())
        template.set_masses(frame["masses"].double().cpu().numpy())
        extras = {"step": 1, "time_fs": 1}
        ensemble = source.parameters["ensemble"]
        if ensemble != "nve":
            extras["target_temp_K"] = 1
        if ensemble == "nvt_nhc":
            tchain = source.parameters["tchain"]
            extras.update(nhc_eta=tchain, nhc_p_eta=tchain, nhc_conserved_energy=1)
        trajectory = ExtxyzTrajectory(self._tracked[TRAJECTORY_FILE], append=append)
        try:
            return AsyncGPUMDLogger(template, source.device, trajectory=trajectory, state_dtype=state_dtype,
                                    extras=extras)
        except BaseException:
            trajectory.close()
            raise

    def write(self) -> dict | None:
        """Thermo line, frame and restart file due at the current step (call after every step).

        Returns the thermodynamics when any output was due, else None. The
        last step (``total_steps``) always gets a thermo line and a restart.
        """
        if self.output_dir is None:
            return None
        step = self.source.nsteps
        last = step == self.total_steps
        thermo_due = bool(self.thermo_interval) and (step % self.thermo_interval == 0 or last)
        xyz_due = bool(self.xyz_interval) and step % self.xyz_interval == 0
        restart_due = bool(self.restart_interval) and (step % self.restart_interval == 0 or last)
        if not (thermo_due or xyz_due or restart_due):
            return None
        thermo = self.source.thermodynamics()
        error = None
        if thermo_due and self.comm.rank == 0:
            target = thermo["target_temperature"]
            try:
                with open(self._tracked[THERMO_FILE], "a") as stream:
                    stream.write(f"{step:d}\t{thermo['time_fs']:.4f}\t"
                                 f"{math.nan if target is None else target:.4f}\t{thermo['temperature']:.4f}\t"
                                 f"{thermo['potential_energy']:.6f}\t{thermo['kinetic_energy']:.6f}\t"
                                 f"{thermo['total_energy']:.6f}\n")
            except (OSError, *_ERRORS) as exc:
                error = f"thermo file: {exc}"
        if xyz_due:
            error = self._log_frame(thermo) or error
        if restart_due:
            error = self._save_restart(None) or error
        self.source.collective_check(error)
        return thermo

    def log_frame(self, thermo: dict | None = None) -> None:
        """Gather the current frame by ID into rank 0's logger (asynchronous file write)."""
        if self.output_dir is None or not self.xyz_interval:
            raise RuntimeError("log_frame needs output_dir and xyz_interval")
        self.source.collective_check(self._log_frame(thermo))

    def _log_frame(self, thermo: dict | None) -> str | None:
        """The collective part of :meth:`log_frame`; rank 0's error (None elsewhere), not agreed."""
        thermo = self.source.thermodynamics() if thermo is None else thermo
        fields = self.source.md_fields()
        frame = self.comm.gather_by_id(0, fields["ids"], positions=fields["positions"],
                                       momenta=fields["momenta"], forces=fields["forces"])
        if frame is None:
            return None
        scalar = lambda value: torch.tensor([value], dtype=torch.float64)
        extras = {"step": scalar(thermo["step"]), "time_fs": scalar(thermo["time_fs"])}
        ensemble = self.source.parameters["ensemble"]
        if ensemble != "nve":
            extras["target_temp_K"] = scalar(thermo["target_temperature"])
        if ensemble == "nvt_nhc":
            extras.update(nhc_eta=torch.tensor(thermo["eta"], dtype=torch.float64),
                          nhc_p_eta=torch.tensor(thermo["p_eta"], dtype=torch.float64),
                          nhc_conserved_energy=scalar(thermo["conserved_energy"]))
        try:
            self.logger.enqueue(
                step=thermo["step"], time_ps=thermo["time_fs"] / 1000.0, positions=frame["positions"],
                momenta=frame["momenta"], forces=frame["forces"], cell=self.source.cell,
                potential_energy=scalar(thermo["potential_energy"]),
                kinetic_energy=scalar(thermo["kinetic_energy"]), temperature=scalar(thermo["temperature"]),
                extras=extras)
        except (OSError, *_ERRORS) as exc:  # e.g. the writer thread's failure
            return f"trajectory frame: {exc}"
        return None

    def save_restart(self, path=None) -> None:
        """Write the current state to rank 0's P-independent restart file (default ``output_dir``).

        ONE gather_by_id; rank 0 flushes the outputs and writes
        (:func:`write_restart`) the ID-ordered rows, the cached evaluation, the
        cell, the step, ``parameters``, the thermostat state and the output
        files' offsets; then one ``collective_check`` (rank 0's error on every rank).
        ``path`` is rank 0's (other ranks' are ignored), as the trajectory's.
        """
        self.source.collective_check(self._save_restart(path))

    def _save_restart(self, path) -> str | None:
        """The collective part of :meth:`save_restart`; rank 0's error (None elsewhere), not agreed."""
        fields = self.source.md_fields()
        frame = self.comm.gather_by_id(0, fields["ids"], numbers=fields["numbers"], masses=fields["masses"],
                                       positions=fields["positions"], momenta=fields["momenta"],
                                       forces=fields["forces"])
        if frame is None:
            return None
        try:
            if path is None and self.output_dir is None:  # root's path: agreed by the caller
                raise ValueError("save_restart needs a path (no output_dir)")
            thermostat = self.source.thermostat_state()
            path = self.output_dir / RESTART_FILE if path is None else Path(path)
            if self.logger is not None:
                self.logger.flush()
            offsets = {name: file.stat().st_size if file.exists() else 0 for name, file in self._tracked.items()}
            host = lambda value, dtype=None: (value if dtype is None else value.to(dtype)).detach().cpu().numpy()
            write_restart(path, RestartData(
                ids=host(frame["ids"], torch.int64), numbers=host(frame["numbers"], torch.int64),
                masses=host(frame["masses"], torch.float64), positions=host(frame["positions"], torch.float64),
                momenta=host(frame["momenta"], torch.float64), forces=host(frame["forces"]),
                energy=host(fields["energy"]), reference_energy=host(fields["reference_energy"]),
                cell=host(self.source.cell), pbc=self._pbc(), step=self.source.nsteps,
                parameters=self.source.parameters, temperature_K=thermostat.get("temperature_K"),
                nhc_state=thermostat if self.source.parameters["ensemble"] == "nvt_nhc" else None,
                offsets=offsets, engine=self.engine))
        except (OSError, *_ERRORS) as exc:
            return f"restart file: {exc}"
        return None

    def _pbc(self) -> np.ndarray:
        pbc = getattr(self.source, "pbc", None)
        if pbc is None:
            return np.ones(3, dtype=bool)
        return np.asarray(pbc.detach().cpu().numpy() if torch.is_tensor(pbc) else pbc, dtype=bool).reshape(3)

    def close(self) -> None:
        """Flush and close rank 0's trajectory logger (released even when the last write fails).

        Rank-local: rank 0 raises its writer's error; the driver's ``close``
        agrees it across ranks.
        """
        logger, self.logger = self.logger, None
        if logger is not None:
            logger.close()

    def close_quietly(self) -> None:
        """:meth:`close` without raising (a failed constructor or an exception already propagating)."""
        try:
            self.close()
        except Exception:  # noqa: BLE001 - the caller is already raising
            pass
