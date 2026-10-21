from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import copy
import gc
import itertools
import math
import time
import warnings
import weakref

import numpy as np
import torch
from ase import Atoms, units
from ase.io import Trajectory
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch import Tensor

from fastmd.adapters import ModelAdapter, load_adapter
from fastmd.adapters.negotiate import negotiated
from fastmd.graph import FixedCapacityGraphBuilder
from fastmd.graph import TensorGraphBuilder
from fastmd.graph.fixed_capacity_builder import line_graph_batch
from fastmd.config import ConfigurationWarning, InferenceConfig, env_value
from fastmd.engine.entry_config import (
    apply_checkpoint, configure_entrypoint, config_summary, current_adapter, release_adapter, scoped_call,
    use_adapter,
)
from fastmd.engine import saved_recompute
from fastmd.interfaces.calculator import require_finite_geometry
from fastmd.engine.capacity_policy import CapacityKey, JointCapacityPolicy, declared_capacity, held_demand, thermal_factors
from fastmd.execution.cuda_graph import BucketedGraphRunner
from fastmd.engine.energy_derivative import edges_only_builder_options
from fastmd.engine.energy_derivative import engine_derivative
from fastmd.engine.energy_derivative import stress_unit_scale
from fastmd.execution.gpu_integrator import GPUIntegrator
from fastmd.execution.gpu_integrator import SlotVelocityVerlet
from fastmd.execution.gpu_integrator import StatefulIntegrator
from fastmd.execution.gpu_integrator import default_fix_com
from fastmd.execution.gpu_integrator import kinetic_energy_fp64
from fastmd.execution.gpu_logging import AsyncGPUMDLogger
from fastmd.execution.gpu_nhc import GPUNoseHooverChainNVT
from fastmd.execution.gpu_nhc import NHCState
from fastmd.execution.md_protocol import ENSEMBLES
from fastmd.execution.md_protocol import _ERRORS
from fastmd.execution.md_protocol import HeatingRamp
from fastmd.execution.md_protocol import MDOutputs
from fastmd.execution.md_protocol import RestartData
from fastmd.execution.md_protocol import read_restart
from fastmd.execution.md_protocol import resume_mismatch
from fastmd.execution.whole_step_cuda_graph import FixedCapacityOverflow
from fastmd.execution.whole_step_cuda_graph import WholeStepGraphRunner


def _env_flag(name: str, default: str = "0") -> bool:
    return env_value(name, default) == "1"


# S8: whole-step tier cache — keep replaced-capacity graphs keyed by
# (u_capacity, t_capacity) with a shared graph pool, switch between them at
# transaction boundaries, and shrink behind hysteresis instead of paying a
# permanent padding tax after transient densification spikes.
_WHOLE_STEP_TIER_CACHE = _env_flag("MATRIS_WHOLE_STEP_TIER_CACHE")
_WHOLE_STEP_MAX_TIERS = int(env_value("MATRIS_WHOLE_STEP_MAX_TIERS", "3"))
_TIER_HYSTERESIS_WINDOW = int(env_value("MATRIS_TIER_HYSTERESIS_WINDOW", "2000"))
_TIER_HYSTERESIS_SIGMA = float(env_value("MATRIS_TIER_HYSTERESIS_SIGMA", "3.0"))
_TIER_CHECK_INTERVAL = int(env_value("MATRIS_TIER_CHECK_INTERVAL", "256"))
# After an overflow-driven up-switch, block shrinks for this many replays so
# a demand spike recurring on a period longer than the window cannot drive
# unbounded down/up flip-flop (default: two hysteresis windows).
_TIER_DOWN_COOLDOWN = int(
    env_value("MATRIS_TIER_DOWN_COOLDOWN", str(2 * _TIER_HYSTERESIS_WINDOW))
)
# Device-memory guard for tier captures. Per-tier byte attribution on a
# shared pool is path-dependent and ill-defined, so the budget asks the
# DEVICE instead: before capturing a tier, require mem_get_info free bytes
# to cover a conservative standalone estimate plus this floor. If not, the
# pool-reset valve runs first; if still short, the capture proceeds and an
# OOM unwinds through the exception-safe switch path.
_TIER_MIN_FREE_GIB = float(env_value("MATRIS_TIER_MIN_FREE_GIB", "6.0"))
_TIER_COST_MARGIN = 1.25
# S9: lattice cold starts under-estimate thermal triplet demand (measured
# +20% on Si-1000 within 0.3 ps of thermalization); an inflation factor at
# calibration time avoids the early grow-recapture chain.
_CAPACITY_THERMAL_FACTOR = float(env_value("MATRIS_CAPACITY_THERMAL_FACTOR", "1.0"))


@dataclass
class GPUMDState:
    positions: Tensor
    momenta: Tensor
    forces: Tensor | None
    potential_energy: Tensor | None


class GPUResidentMolecularDynamics:
    """Molecular dynamics with simulation state resident on one CUDA device.

    The adapter supplies topology and energy/derivative capabilities. The
    runtime owns graph construction, topology reuse, integration and logging.
    Supported ensembles are nve, Berendsen nvt and Nose-Hoover-chain nvt_nhc.
    InferenceConfig chooses eager, model_graph, whole_step or automatic
    execution; whole-step capture is used only where the adapter permits it.

    timestep, taut and tdamp are in fs; temperature is in K. Forces are
    conservative where the adapter's energy interface declares that property.
    Trajectories and thermodynamic/restart outputs are flushed by close();
    use this object as a context manager to guarantee writer cleanup.

    log_stress requires a trajectory and a fully periodic cell. Stress is
    computed only for logged frames and stored in ASE units (eV/Angstrom^3).
    steps_per_launch can integrate multiple steps in a whole-step launch;
    run(steps) requires a whole number of these windows.
    """

    # A whole-step frame staged for the logger (_stage_frame): (state copy, stress copy, event, thermostat-mass
    # copy) buffers, the step of the staged frame (None: none) and its thermostat constants. Class defaults:
    # These buffers are populated lazily by native capture/logging paths.
    _staged = None
    _staged_step = None
    _staged_constants = None
    # A model with a neighbour cap (TopologySpec.max_neighbors): the largest per-atom edge count of the builds, a
    # [1] int64 device maximum (None without a cap), and whether its excess was warned about.
    _neighbor_count_max = None
    _neighbor_cap_warned = False
    # Releases this driver's use of an adapter an entry point loaded by name (_config.use_adapter); None: none
    # (passed in by the caller, or from_evaluator's driver).
    _adapter_finalizer = None
    # run()'s whole-step windows launched behind an unsettled one, and those of them rolled back.
    _windows_ahead = 0
    _windows_rolled_back = 0

    @configure_entrypoint("gpu_md")
    def __init__(
        self,
        atoms: Atoms | Structure | Molecule,
        model_path: str | None = None,
        model: torch.nn.Module | ModelAdapter | str = "matris_10m_oam",
        task: str = "efsm",
        device: torch.device | str = "cuda",
        ensemble: str = "nvt",
        temperature: float = 300.0,
        timestep: float = 1.0,
        taut: float = 100.0,
        trajectory: str | Trajectory | None = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        logger_ring: int = 8,
        precapture_steps: int = 0,
        config: InferenceConfig | None = None,
        tdamp: float | None = None,
        tchain: int = 3,
        tloop: int = 1,
        fix_com: bool | None = None,
        ramp: HeatingRamp | None = None,
        output_dir=None,
        thermo_interval: int = 0,
        xyz_interval: int = 0,
        restart_interval: int = 0,
        total_steps: int | None = None,
        restart: RestartData | None = None,
        steps_per_launch: int = 1,
        log_stress: bool = False,
    ) -> None:
        if restart is not None and not isinstance(restart, RestartData):
            raise TypeError("GPUResidentMolecularDynamics: restart must be a RestartData (see from_restart)")
        if not isinstance(log_stress, bool):
            raise TypeError("GPUResidentMolecularDynamics: log_stress must be a bool")
        if log_stress and trajectory is None:
            raise ValueError("GPUResidentMolecularDynamics: log_stress=True writes the stress into the "
                             "trajectory frames; pass a trajectory")
        intervals = (thermo_interval, xyz_interval, restart_interval, 0 if total_steps is None else total_steps)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in intervals):
            raise ValueError("GPUResidentMolecularDynamics: output intervals and total_steps must be "
                             "nonnegative integers")
        if ramp is not None and not isinstance(ramp, HeatingRamp):
            raise TypeError("GPUResidentMolecularDynamics: ramp must be a HeatingRamp")
        if isinstance(steps_per_launch, bool) or not isinstance(steps_per_launch, int) or steps_per_launch < 1:
            raise ValueError("GPUResidentMolecularDynamics: steps_per_launch must be a positive integer")
        if steps_per_launch > 1:
            # Every output and log step must end a window (a window's inner steps are never visible).
            logs = trajectory is not None or logfile is not None
            ramp_steps = ramp.total_steps if total_steps is None and ramp is not None else total_steps
            cadence = {"loginterval": max(int(loginterval), 1) if logs else 0, "thermo_interval": thermo_interval,
                       "xyz_interval": xyz_interval, "restart_interval": restart_interval,
                       "total_steps": ramp_steps or 0, "restart step": 0 if restart is None else int(restart.step)}
            off = {name: value for name, value in cadence.items() if value % steps_per_launch}
            if off:
                raise ValueError(f"GPUResidentMolecularDynamics: steps_per_launch={steps_per_launch} must divide "
                                 f"{off} (a multi-step window's inner steps are not visible)")
        if ramp is not None and ensemble.lower() == "nve":
            raise ValueError("GPUResidentMolecularDynamics: a heating ramp needs a thermostat (nvt or nvt_nhc)")
        if ensemble.lower() == "nvt_nhc" and fix_com:
            raise ValueError(
                "GPUResidentMolecularDynamics: ensemble='nvt_nhc' has no centre-of-mass "
                "correction (ASE NoseHooverChainNVT); fix_com must be None or False"
            )
        if restart is not None and restart.temperature_K is not None:
            # The saved target continues (with a ramp: its current setpoint); a constant one must match.
            if ramp is None and temperature != restart.temperature_K:
                raise ValueError(f"GPUResidentMolecularDynamics: temperature {temperature} K cannot resume a "
                                 f"restart file at {restart.temperature_K} K (without a ramp the target is "
                                 "the saved one)")
            temperature = restart.temperature_K
        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        if log_stress and not atoms.get_pbc().all():
            raise ValueError("GPUResidentMolecularDynamics: log_stress needs a cell periodic in all three "
                             "directions (the stress is per cell volume)")
        # Only the initial state is checked on the host. A trajectory that turns
        # non-finite later is caught, without extra syncs, by the whole-step
        # replay's displacement readback and by the logger's frame readback
        # (gpu_logging); eager/model-graph steps without a logger do not check.
        require_finite_geometry(atoms.positions, atoms.cell.array, "GPUResidentMolecularDynamics")
        if atoms.has("momenta") and not np.isfinite(atoms.get_momenta()).all():
            raise ValueError("GPUResidentMolecularDynamics: non-finite initial momenta")
        self.atoms_template = atoms.copy()
        self.ramp = ramp
        self.steps_per_launch = steps_per_launch
        self._window_targets = None  # per-step targets of a multi-step whole-step window (thermostats)
        self.task = task
        self.force_task = "ef"
        self.log_stress = log_stress
        self._stress_at = None  # (step, stress in GPa): the stress of a logged state
        # Every whole-step window ends on a logged step: its runner captures the record window only.
        self._stress_every_window = log_stress and max(int(loginterval), 1) == steps_per_launch
        self.timestep = float(timestep)
        self.fix_com = default_fix_com(ensemble) if fix_com is None else bool(fix_com)
        self.nsteps = 0
        self._logged_initial = False
        self.device = torch.device(device)
        if self.device.index is None and self.device.type == "cuda":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self._adapter = load_adapter(
            model,
            model_path=model_path,
            device=str(self.device),
            enable_checkpoint=self._inference_config.checkpoint_enabled,
            optimization_profile=self._inference_config.optimization_profile,
        )
        use_adapter(self, model)  # an adapter loaded by name is closed with its last user (also on a failure below)
        # The model's declared capacity step where the configuration keeps the default one, and the thermal
        # factors declared with it (_capture_factors).
        self._inference_config = declared_capacity(self._adapter, self._inference_config)
        self._thermal_factors = thermal_factors(self._adapter, self._inference_config)
        self.model = self._adapter.module
        if self.model is model or self._adapter is model:  # supplied, not loaded
            self.model = self.model.to(self.device)
            apply_checkpoint(self.model, self._inference_config.checkpoint_enabled)
        self.model.eval()
        saved_recompute.check_model(
            self.model, saved_recompute.explicit_mode(self._inference_config.saved_tensor_recompute))
        saved_recompute.note_entry_point(self.model)
        for param in self.model.parameters():
            param.requires_grad_(False)
        # None: the model differentiates itself (adapter.evaluate); else the engine-owned derivative.
        self._derivative = engine_derivative(self._adapter)
        # A float64 model that the engine differentiates runs on float64 positions, cell, forces and
        # energy (the Nose-Hoover chain's float64 state); MatRIS runs float32.
        self._float64 = self._derivative is not None and self._adapter.topology_spec().model_dtype == "float64"
        if self._float64 and ensemble.lower() != "nvt_nhc":
            raise ValueError("GPUResidentMolecularDynamics: a float64 model needs ensemble='nvt_nhc' (the "
                             "integrator with a float64 state); nve and nvt keep a float32 state")

        self.atomic_numbers = torch.as_tensor(
            atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device
        )
        self.cell = torch.as_tensor(
            atoms.get_cell().array, dtype=torch.float32, device=self.device
        )
        self.pbc = torch.as_tensor(atoms.get_pbc(), dtype=torch.bool, device=self.device)
        # The cell the model reads (the builders' batches, the logged frames): float64 for a float64 model.
        self._model_cell = (torch.as_tensor(atoms.get_cell().array, dtype=torch.float64, device=self.device)
                            if self._float64 else self.cell)
        self._positions_dtype = {"positions_dtype": torch.float64} if self._float64 else {}
        self.masses = torch.as_tensor(
            atoms.get_masses(), dtype=torch.float32, device=self.device
        )
        self.n_atoms = int(self.atomic_numbers.shape[0])
        self.composition = atoms.get_chemical_formula()

        positions = torch.as_tensor(
            atoms.get_positions(), dtype=torch.float32, device=self.device
        )
        if atoms.has("momenta"):
            momenta_np = atoms.get_momenta()
        else:
            momenta_np = np.zeros((self.n_atoms, 3), dtype=np.float32)
        momenta = torch.as_tensor(momenta_np, dtype=torch.float32, device=self.device)
        self.state = GPUMDState(
            positions=positions,
            momenta=momenta,
            forces=None,
            potential_energy=None,
        )

        if ensemble.lower() == "nvt_nhc":
            # float64 state from the float64 ASE arrays (ASE-exact dynamics)
            self.integrator = GPUNoseHooverChainNVT(
                atoms.get_masses(),
                atoms.get_positions(),
                atoms.get_momenta(),
                temperature_K=temperature,
                timestep=self.timestep * units.fs,
                tdamp=(100.0 * self.timestep if tdamp is None else float(tdamp))
                * units.fs,
                tchain=tchain,
                tloop=tloop,
                device=self.device,
                force_dtype=torch.float64 if self._float64 else torch.float32,
            )
            self.integrator.warmup_compile()
        else:
            self.integrator = GPUIntegrator(
                self.masses,
                dt_fs=timestep,
                ensemble=ensemble,
                temperature=temperature,
                taut_fs=taut,
                fix_com=self.fix_com,
                degrees_of_freedom=self.n_atoms * 3,
            )
        self.stateful_integrator = (
            self.integrator
            if isinstance(self.integrator, StatefulIntegrator)
            else None
        )
        # The restart file's "parameters" (the from_evaluator driver's keys; the
        # native engine does not honor its NHC options scalar_sync/ke2_update).
        self.parameters = {
            "ensemble": ensemble.lower(), "dt_fs": self.timestep,
            "tdamp_fs": float(100.0 * self.timestep if tdamp is None else tdamp), "taut_fs": float(taut),
            "tchain": tchain, "tloop": tloop, "fix_com": self.fix_com,
            "ramp": None if ramp is None else asdict(ramp), "scalar_sync": "broadcast",
            "ke2_update": "scaled",
            "total_steps": ramp.total_steps if total_steps is None and ramp is not None else total_steps,
        }
        if restart is not None:
            self._seed_restart(restart, atoms)
        self.whole_step_cuda_graph = self._inference_config.execution == "whole_step"
        self.enable_model_fusions = self._inference_config.optimization_profile != "generic"
        self.validate_isolated_atoms = (
            not self._inference_config.handle_isolated_atoms
            or self.adapter.isolated_atom_policy() == "error"
        )
        self.candidate_skin = self._inference_config.capacity.candidate_skin
        self.runner = None
        self._stress_runner = None  # model_graph with log_stress: the logged steps' runner (task "efs")
        self.whole_step_runner = None
        self.whole_step_recaptures = 0
        self._whole_step_accumulated = {
            "launches": 0,
            "candidate_refreshes": 0,
            "fallback_windows": 0,
            "recovery_replays": 0,
            "max_edge_count": 0,
            "max_triplet_count": 0,
            "max_displacement": 0.0,
        }
        self.whole_step_capacity_history: list[tuple[int, int]] = []
        self.capacity_policy: JointCapacityPolicy | None = None
        self._tier_runners: dict[CapacityKey, WholeStepGraphRunner] = {}
        self._tier_lru: list[CapacityKey] = []
        self._tier_pool = None
        self._tier_demand_u: deque[int] = deque(maxlen=_TIER_HYSTERESIS_WINDOW)
        self._tier_demand_t: deque[int] = deque(maxlen=_TIER_HYSTERESIS_WINDOW)
        self._tier_switch_up = 0
        self._tier_switch_down = 0
        self._tier_captures = 0
        self._tier_evictions = 0
        self._tier_check_counter = 0
        self._tier_replay_counter = 0
        self._tier_last_up_at = -(10**18)
        self._tier_cost_per_triplet = 0.0
        self._tier_pool_resets = 0
        topology = self.adapter.topology_spec()
        self.atom_graph_cutoff = float(topology.cutoff)
        # A model without triplets reads the edges only (graph batches of the builders' edges-only mode).
        self._edges_only = topology.triplet_cutoff is None
        self.line_graph_cutoff = None if self._edges_only else float(topology.triplet_cutoff)
        # The whole-step builder's options for this model (none for MatRIS): float64 positions, and no
        # per-build sort of the edge rows for a model that only needs them grouped by centre.
        self._fixed_builder_options = edges_only_builder_options(self.adapter) if self._edges_only else {}
        if self._edges_only and topology.max_neighbors is not None:
            # The builds list every pair within the cutoff and fold their largest per-atom edge count into
            # this device maximum (no host read); neighbor_cap_summary() reports it.
            self._neighbor_count_max = torch.zeros(1, dtype=torch.int64, device=self.device)
            self._fixed_builder_options["max_neighbor_count"] = self._neighbor_count_max
        # A model with a line graph that the engine differentiates reads graph batches with positions
        # (its leaf): the whole-step builder publishes them, and the eager path views MatRIS's graph.
        self._line_graph_batches = self._derivative is not None and not self._edges_only
        if self._line_graph_batches:
            self._fixed_builder_options["publish_positions"] = True
        self.graph_builder = TensorGraphBuilder(
            cell=self._model_cell,
            atomic_numbers=self.atomic_numbers,
            pbc=self.pbc,
            atom_graph_cutoff=self.atom_graph_cutoff,
            line_graph_cutoff=self.line_graph_cutoff,
            device=self.device,
            composition=self.composition,
            check_isolated_atoms=self.validate_isolated_atoms,
            **self._positions_dtype,
        )
        self._auto_reason = None  # why execution="auto" chose its mode, or left whole_step (a CUDA OOM)
        self._auto_choice = None  # the mode execution="auto" chose here
        self._engine_plan = None  # another model's negotiated plan: (adapter, EnginePlanRecord)
        if negotiated(self.adapter):
            # Another model: its negotiated plan decides the execution and the forwards' precision.
            plan = self._negotiate_plan()
            execution = plan.decision("execution")
            if self._execution_auto:
                self._auto_reason, self._auto_choice = execution.reason, execution.chosen
            self._inference_config = replace(self._inference_config, execution=execution.chosen,
                                             matmul_precision=plan.matmul_precision)
            self.whole_step_cuda_graph = execution.chosen == "whole_step"
        elif self._execution_auto:
            mode, self._auto_reason = self._auto_mode(ensemble)
            self._auto_choice = mode
            self._inference_config = replace(self._inference_config, execution=mode)
            self.whole_step_cuda_graph = mode == "whole_step"
        if self.whole_step_cuda_graph:
            try:
                self.enable_whole_step_cuda_graph(precapture_steps=precapture_steps)
            except torch.OutOfMemoryError as error:
                if not self._execution_auto:
                    raise
                self._whole_step_to_eager(error)
        elif self._inference_config.execution == "model_graph":
            failed = None
            try:
                self.enable_model_cuda_graph(precapture_steps=precapture_steps)
            except torch.OutOfMemoryError as error:
                # Another model's execution auto (a measured bound, not K6's) falls back to eager.
                if not (self._execution_auto and self._engine_plan is not None):
                    raise
                failed = str(error).splitlines()[0][:160]
            if failed is not None:  # outside the except block: its traceback would pin the graph pool
                self._drop_model_graph()
                self._auto_reason = f"model_graph ran out of CUDA memory ({failed}); eager"
                warnings.warn(self._auto_reason, RuntimeWarning, stacklevel=2)
        from fastmd.engine.comm import LocalCommunicator

        self.comm = LocalCommunicator()  # the outputs gather "by ID" (ids = arange(N))
        # The outputs first: a rejected resume must not open (truncate) the trajectory.
        self.logger = None
        self.outputs = MDOutputs(self, output_dir, thermo_interval, xyz_interval, restart_interval,
                                 self.parameters["total_steps"], None if restart is None else restart.offsets,
                                 engine="native")
        try:
            self.logger = self._native_logger(trajectory, logfile, loginterval, append_trajectory, logger_ring)
        except BaseException:
            self.outputs.close_quietly()
            raise
        # A dropped driver (no close()) still stops its loggers' writer threads.
        self._finalizer = weakref.finalize(self, _close_quietly, self.logger, self.outputs)

    def _native_logger(self, trajectory, logfile, loginterval, append_trajectory, logger_ring):
        return AsyncGPUMDLogger(
            atoms_template=self.atoms_template,
            device=self.device,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
            ring=logger_ring,
            **({"stress": True} if self.log_stress else {}),
            # a float64 model: float64 forces, cell and stress in the frames
            **({"value_dtype": torch.float64} if self._float64 else {}),
            # nvt_nhc: log the float64 state and the thermostat variables
            **(
                {
                    "state_dtype": torch.float64,
                    "extras": {
                        "nhc_eta": self.integrator.tchain,
                        "nhc_p_eta": self.integrator.tchain,
                        "nhc_conserved_energy": 1,
                    },
                }
                if isinstance(self.integrator, GPUNoseHooverChainNVT)
                else {}
            ),
        )

    def _seed_restart(self, data: RestartData, atoms: Atoms) -> None:
        """Continue a restart file: its step, cached forces/energy and thermostat state (before any capture)."""
        if len(data.ids) != self.n_atoms or not np.array_equal(
                np.asarray(data.numbers), atoms.get_atomic_numbers()):
            raise ValueError("GPUResidentMolecularDynamics: the restart file's atoms differ from the structure")
        # Either side native: the decomposed chain's options are not compared.
        changed = resume_mismatch(data.parameters, self.parameters, ignore=("scalar_sync", "ke2_update"))
        if changed:
            raise ValueError("GPUResidentMolecularDynamics: MD parameters differ from the restart file's "
                             f"(saved, requested): {changed}")
        forces = torch.as_tensor(np.asarray(data.forces, dtype=np.float32), device=self.device)
        energy = torch.as_tensor(np.asarray(data.energy, dtype=np.float32), device=self.device).reshape(())
        if self.stateful_integrator is not None:
            self.stateful_integrator.set_forces(forces, energy=energy.reshape(1))
            if data.nhc_state is not None:
                self.integrator.load_state_dict(data.nhc_state)
        self.state.forces, self.state.potential_energy = forces.reshape(self.n_atoms, 3), energy
        self.nsteps = int(data.step)

    @classmethod
    def from_restart(cls, path, *, model: torch.nn.Module | ModelAdapter | str = "matris_10m_oam",
                     model_path: str | None = None, **kwargs) -> "GPUResidentMolecularDynamics":
        """Resume a restart file (a compatible single-device simulation) on one GPU.

        The structure comes from the file (numbers, cell, pbc, masses,
        float64 positions and momenta); the MD options default to the file's
        ``parameters``, and an option given here must equal the file's except
        ``total_steps`` and the decomposed chain's ``scalar_sync``/``ke2_update``.
        The thermostat target is the saved one: with a ramp the ramp continues
        from it (a ``temperature`` given here is not used), without one a
        ``temperature`` that differs from it is rejected. Other keywords are
        the constructor's (``config``, ``trajectory``, ``output_dir``, ...);
        ``output_dir`` resumes its md_pyrolysis files (truncated to the
        file's offsets).
        """
        data = read_restart(path)
        atoms = Atoms(numbers=np.asarray(data.numbers), positions=np.asarray(data.positions, dtype=np.float64),
                      cell=np.asarray(data.cell, dtype=np.float64), pbc=np.asarray(data.pbc, dtype=bool))
        atoms.set_masses(np.asarray(data.masses, dtype=np.float64))
        atoms.set_momenta(np.asarray(data.momenta, dtype=np.float64))
        saved = data.parameters
        defaults = {"ensemble": saved["ensemble"], "timestep": saved["dt_fs"], "taut": saved["taut_fs"],
                    "tdamp": saved["tdamp_fs"], "tchain": saved["tchain"], "tloop": saved["tloop"],
                    "fix_com": saved["fix_com"], "total_steps": saved.get("total_steps"),
                    "ramp": None if saved.get("ramp") is None else HeatingRamp(**saved["ramp"])}
        if data.temperature_K is not None:
            defaults["temperature"] = data.temperature_K
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        return cls(atoms, model_path=model_path, model=model, restart=data, **kwargs)

    adapter = property(current_adapter)

    def config_summary(self) -> dict:
        """Requested/resolved configuration and observed execution state."""
        return config_summary(self)

    def plan_record(self, op_plan=None):
        """The engine plan record (``fastmd.adapters.EnginePlanRecord``; ``as_dict()``/``to_json()``):
        every runtime decision (neighbor list, topology reuse, padding, kernels, execution, precision,
        recompute, multi-GPU) with its reason and evidence, the model's fingerprint and the accuracy
        class. For MatRIS it records the decisions made here; it changes none. ``op_plan``: an
        ``OpPlanRecorder`` (or its snapshot) of runs to link."""
        from fastmd.adapters import entry_plan_record
        return entry_plan_record(self, op_plan=op_plan)

    @scoped_call
    def enable_model_cuda_graph(self, precapture_steps: int = 0) -> None:
        """Enable model-only replay from the current GPU-resident MD state."""
        if self.device.type != "cuda":
            raise ValueError("model CUDA graph capture requires a CUDA device")
        if self.whole_step_runner is not None:
            raise ValueError("cannot enable model replay with a whole-step runner")
        if self.runner is not None:
            return
        capacity = self._inference_config.capacity
        self.runner = BucketedGraphRunner(
            self.adapter,
            task=self.force_task,
            enable_model_fusions=self.enable_model_fusions,
            u_step=capacity.u_step, t_step=capacity.t_step,
            n_dummy=capacity.n_dummy, min_pad_u=capacity.min_pad_u,
            warmup=capacity.warmup,
            config=self._inference_config,
        )
        self._inference_config = replace(self._inference_config, execution="model_graph")
        self.runner._inference_config = self._inference_config
        if self.log_stress:
            # The logged steps' runner: its own graphs and pool. Both capture streams are created here, back to
            # back: consecutive streams of torch's pool never alias, so one runner's pool release cannot free a
            # cuBLAS workspace that the other runner's graphs use.
            self._stress_runner = BucketedGraphRunner(
                self.adapter, task=self.force_task + "s", enable_model_fusions=self.enable_model_fusions,
                u_step=capacity.u_step, t_step=capacity.t_step, n_dummy=capacity.n_dummy,
                min_pad_u=capacity.min_pad_u, warmup=capacity.warmup, config=self._inference_config)
            self._stress_runner._inference_config = self._inference_config
            for runner in (self.runner, self._stress_runner):
                runner._capture_stream = torch.cuda.Stream()
        self.precapture(steps=precapture_steps, proactive=True)

    @scoped_call
    def enable_whole_step_cuda_graph(self, precapture_steps: int = 0) -> None:
        """Capture the whole-step path from the current GPU-resident MD state."""
        if self.device.type != "cuda":
            raise ValueError("whole-step CUDA graph capture requires a CUDA device")
        if self.whole_step_runner is not None:
            return
        if self.runner is not None:
            raise ValueError("cannot replace an active bucketed CUDA graph runner")

        self.whole_step_cuda_graph = True
        sample_graph = self.graph_builder.build(self.state.positions)
        capacity = self._inference_config.capacity
        self.capacity_policy = JointCapacityPolicy(
            u_step=capacity.u_step, t_step=capacity.t_step,
            min_pad_u=capacity.min_pad_u, min_pad_t=capacity.min_pad_t,
            triplets=not self._edges_only,
        )
        if self._edges_only:
            u_demand, t_demand = sample_graph.edge_index.shape[0] // 2, 0
        else:
            u_demand = sample_graph.undirected2directed.shape[0]
            t_demand = sample_graph.line_graph.shape[0]
        if _CAPACITY_THERMAL_FACTOR != 1.0:
            t_demand = int(round(t_demand * _CAPACITY_THERMAL_FACTOR))
        # A large system's first capture already has the whole-step headroom
        # (_whole_step_headroom): thermal fluctuations of the starting structure
        # otherwise grow (recapture) it within the first steps (10,752 RDX atoms
        # at 300 K: twice in 23 steps). A declared thermal factor holds more where larger.
        u_factor, t_factor = self._capture_factors(first=True)
        initial_capacity = self.capacity_policy.start(
            held_demand(u_demand, self._whole_step_headroom(u_demand, capacity.u_step), u_factor),
            held_demand(t_demand, self._whole_step_headroom(t_demand, capacity.t_step), t_factor))
        self._whole_u_capacity = initial_capacity.u_capacity
        self._whole_t_capacity = initial_capacity.t_capacity
        self.whole_step_capacity_history.append(
            (self._whole_u_capacity, self._whole_t_capacity)
        )
        if _WHOLE_STEP_TIER_CACHE and self._tier_pool is None:
            self._tier_pool = torch.cuda.graph_pool_handle()
        while True:
            try:
                self.whole_step_runner = self._make_whole_step_runner(
                    positions=self.state.positions,
                    momenta=self.state.momenta,
                    forces=self.state.forces,
                )
                break
            except FixedCapacityOverflow as overflow:
                # The sizing graph comes from the dynamic builder; if the
                # fixed-capacity build of the same state does not fit (e.g. a
                # pair exactly at the cutoff), grow before the first capture
                # instead of running on the substitute topology.
                current = CapacityKey(self._whole_u_capacity, self._whole_t_capacity)
                grown = self.capacity_policy.grow(
                    current, (overflow.edge_count + 1) // 2, overflow.triplet_count
                )
                if grown == current:
                    raise
                self._whole_u_capacity = grown.u_capacity
                self._whole_t_capacity = grown.t_capacity
                self.whole_step_capacity_history.append(
                    (self._whole_u_capacity, self._whole_t_capacity)
                )
        initial_capacity = CapacityKey(self._whole_u_capacity, self._whole_t_capacity)
        self._inference_config = replace(self._inference_config, execution="whole_step")
        self.whole_step_runner._inference_config = self._inference_config
        if _WHOLE_STEP_TIER_CACHE:
            self._tier_runners[initial_capacity] = self.whole_step_runner
            if initial_capacity in self._tier_lru:
                self._tier_lru.remove(initial_capacity)
            self._tier_lru.append(initial_capacity)
        self._sync_whole_state()
        if self.log_stress:
            # The stress of the current state (the runner's initial evaluation), copied: precapture replays
            # of a record-only runner overwrite the runner's buffer before the first frame.
            self._stress_at = (self.nsteps, self.whole_step_runner.stress.clone())
        if precapture_steps <= 0:
            return

        initial = (
            self.state.positions.clone(),
            self.state.momenta.clone(),
            self.state.forces.clone(),
            self.state.potential_energy.clone(),
        )
        integrator_initial = (
            self.stateful_integrator.snapshot()
            if self.stateful_integrator is not None
            else None
        )
        try:
            for _ in range(int(precapture_steps)):
                self._replay_whole_step()
        except BaseException:
            # A failure (e.g. a CUDA OOM that execution auto continues from
            # eagerly) leaves the input state, not the probe's.
            if integrator_initial is not None:
                self.stateful_integrator.load_snapshot(integrator_initial)
            if self.whole_step_runner is not None:
                self.whole_step_runner.reset(positions=initial[0], momenta=initial[1], forces=initial[2],
                                             potential_energy=initial[3])
            else:
                self.state = GPUMDState(positions=initial[0], momenta=initial[1], forces=initial[2],
                                        potential_energy=initial[3])
            raise
        if integrator_initial is not None:
            self.stateful_integrator.load_snapshot(integrator_initial)
        self.whole_step_runner.reset(
            positions=initial[0],
            momenta=initial[1],
            forces=initial[2],
            potential_energy=initial[3],
        )
        self._sync_whole_state()
        self.capacity_policy.reset_observations()
        # The precapture trajectory was rolled back; its demand samples must
        # not seed the production hysteresis window either.
        self._tier_demand_u.clear()
        self._tier_demand_t.clear()
        self._tier_check_counter = 0

    @scoped_call
    def _make_whole_step_runner(
        self,
        *,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor | None = None,
        recapture: bool = False,
    ) -> WholeStepGraphRunner:
        fixed_builder = FixedCapacityGraphBuilder(
            cell=self._model_cell,
            atomic_numbers=self.atomic_numbers,
            pbc=self.pbc,
            u_capacity=self._whole_u_capacity,
            t_capacity=self._whole_t_capacity,
            atom_graph_cutoff=self.atom_graph_cutoff,
            line_graph_cutoff=self.line_graph_cutoff,
            candidate_cutoff=self.atom_graph_cutoff + self.candidate_skin,
            device=self.device,
            composition=self.composition,
            n_dummy=self._inference_config.capacity.n_dummy,
            padding=self.adapter.padding_rule(),
            prepare=self.adapter.prepare,
            **self._fixed_builder_options,
        )
        return WholeStepGraphRunner(
            model=self.adapter,
            builder=fixed_builder,
            integrator=self.integrator,
            positions=positions,
            momenta=momenta,
            forces=forces,
            task=self.force_task,
            steps_per_launch=self.steps_per_launch,
            step_targets=self._step_targets(),
            enable_model_fusions=self.enable_model_fusions,
            graph_pool=self._tier_pool,
            validate_isolated_atoms=self.validate_isolated_atoms,
            warmup=self._inference_config.capacity.warmup,
            config=self._inference_config,
            recapture=recapture,
            **({"record_stress": True, "record_every_window": self._stress_every_window} if self.log_stress else {}),
        )

    def _capture_factors(self, first: bool = False) -> tuple[float, float]:
        """The model's declared (pair, triplet) thermal factors (capacity_policy.thermal_factors, held_demand);
        at a first capture MATRIS_CAPACITY_THERMAL_FACTOR, where set to another value than 1, scales the triplet
        demand instead."""
        u_factor, t_factor = self._thermal_factors
        return u_factor, 1.0 if first and _CAPACITY_THERMAL_FACTOR != 1.0 else t_factor

    def _whole_step_headroom(self, capacity: int, step: int) -> int:
        """Headroom of a whole-step capture above its demand: CapturePolicy's growth fraction in whole
        steps, none below 128 steps. Unlike a model-graph bucket's there is no one-step minimum: a
        small system's padding step costs more per MD step than its rare recaptures (GB200, ABBA:
        RDX 1,344 atoms +1%, Si 1,000 +2% per step), while a large system's fluctuations otherwise
        recapture it every few steps."""
        from fastmd.engine.capacity_policy import CapturePolicy
        fraction = CapturePolicy.from_config(self._inference_config.capacity).growth_headroom
        return int(capacity * fraction) // step * step

    def _step_targets(self) -> "_WindowTargets | None":
        """The per-step target table of a multi-step window (nvt, nvt_nhc), shared by every runner of the MD.

        Created with the current target; a recapture (growth inside a window) keeps the window's rows."""
        if self.steps_per_launch == 1 or self.temperature_target is None:
            return None
        if self._window_targets is None:
            self._window_targets = _WindowTargets(self.integrator, self.steps_per_launch, self.device)
            self._window_targets.fill([self.temperature_target] * self.steps_per_launch)
        return self._window_targets

    def _set_window_targets(self, start: int) -> None:
        """Before the whole-step window from step ``start``: its thermostat targets (the multi-step table, else
        the ramp's setpoint)."""
        if self._window_targets is not None:
            self._load_window_targets(start)
        elif self.ramp is not None:
            setpoint = self.ramp.setpoint(start)
            if setpoint is not None:
                self.set_temperature(setpoint)

    def _load_window_targets(self, start: int) -> None:
        """Before a multi-step window: each step's target (the ramp's setpoints, else the current one) into
        the table; the integrator's host target becomes the window's last (the graph ends with it loaded)."""
        targets, target = [], self.temperature_target
        for k in range(self.steps_per_launch):
            setpoint = None if self.ramp is None else self.ramp.setpoint(start + k)
            target = target if setpoint is None else setpoint
            targets.append(target)
        self._window_targets.fill(targets)
        if targets[-1] != self.temperature_target:
            self.set_temperature(targets[-1])

    def _sync_whole_state(self) -> None:
        assert self.whole_step_runner is not None
        whole_state = self.whole_step_runner.state
        self.state = GPUMDState(
            positions=whole_state.positions,
            momenta=whole_state.momenta,
            forces=whole_state.forces,
            potential_energy=whole_state.potential_energy[0],
        )

    def set_activation_checkpoint(self, enabled: bool | None) -> None:
        """Update inference checkpointing before capture without reloading weights."""
        if enabled is not None and not isinstance(enabled, bool):
            raise TypeError("enabled must be bool or None")
        captured = self.whole_step_runner is not None or any(
            runner is not None and runner.captures > 0 for runner in (self.runner, self._stress_runner)
        )
        if captured:
            raise ValueError("Changing checkpoint policy does not change already captured CUDA graphs; create a new instance with the desired config")
        if enabled is not False and self._inference_config.saved_tensor_recompute in ("on", "deep"):
            raise ValueError(f"saved_tensor_recompute={self._inference_config.saved_tensor_recompute!r} "
                             "excludes checkpoint: use checkpoint off")
        apply_checkpoint(self.model, enabled)
        self._inference_config = replace(self._inference_config, checkpoint_enabled=enabled)
        for runner in (self.runner, self._stress_runner):
            if runner is not None:
                runner._inference_config = self._inference_config

    def _clone_state(self) -> tuple[GPUMDState, int, object, float | None]:
        return (
            GPUMDState(
                positions=self.state.positions.clone(),
                momenta=self.state.momenta.clone(),
                forces=None if self.state.forces is None else self.state.forces.clone(),
                potential_energy=(
                    None
                    if self.state.potential_energy is None
                    else self.state.potential_energy.clone()
                ),
            ),
            self.nsteps,
            (
                self.stateful_integrator.snapshot()
                if self.stateful_integrator is not None
                else None
            ),
            self.temperature_target,
        )

    def _restore_dynamic_state(self, checkpoint: tuple[GPUMDState, int, object, float | None]) -> None:
        state, nsteps, integrator_state, target = checkpoint
        if integrator_state is not None:
            self.stateful_integrator.load_snapshot(integrator_state)
        if target is not None and target != self.temperature_target:
            self.set_temperature(target)
        self.state = GPUMDState(
            positions=state.positions.clone(),
            momenta=state.momenta.clone(),
            forces=None if state.forces is None else state.forces.clone(),
            potential_energy=(
                None
                if state.potential_energy is None
                else state.potential_energy.clone()
            ),
        )
        self.nsteps = int(nsteps)

    def _timed_trial(self, steps: int) -> list[float]:
        samples_ms = []
        torch.cuda.synchronize(self.device)
        for _ in range(steps):
            started = time.perf_counter()
            self.step()
            torch.cuda.synchronize(self.device)
            samples_ms.append((time.perf_counter() - started) * 1.0e3)
        return samples_ms

    @scoped_call
    def select_whole_step_cuda_graph(
        self,
        *,
        warmup_steps: int = 5,
        min_speedup: float = 1.01,
        baseline: str = "eager",
    ) -> dict[str, object]:
        """Select dynamic execution or whole-step replay from matched states. ``baseline``
        "model_graph" times model-graph replay instead of eager and keeps it (recaptured) where
        whole_step is not ``min_speedup`` faster (the "dynamic" fields then hold its samples)."""
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if min_speedup < 1.0:
            raise ValueError("min_speedup must be at least 1.0")
        if baseline not in ("eager", "model_graph"):
            raise ValueError("baseline must be 'eager' or 'model_graph'")
        if self.whole_step_runner is not None or self.runner is not None:
            raise ValueError("execution-path selection requires uncaptured GPU MD")
        if self.logger.enabled:
            raise ValueError("execution-path selection requires logging to be disabled")

        self._ensure_forces()
        initial = self._clone_state()
        if baseline == "model_graph":
            self.enable_model_cuda_graph()
        dynamic_samples = self._timed_trial(warmup_steps)
        dynamic_final = self._clone_state()
        if baseline == "model_graph":
            self._drop_model_graph()

        self._restore_dynamic_state(initial)
        self.enable_whole_step_cuda_graph()
        graph_samples = self._timed_trial(warmup_steps)
        graph_trial_stats = self.whole_step_stats()
        dynamic_median = float(np.median(dynamic_samples))
        graph_median = float(np.median(graph_samples))
        graph_speedup = dynamic_median / graph_median
        selected_path = "whole_step_cuda_graph"
        if graph_speedup < min_speedup:
            self._drop_whole_step()
            self._restore_dynamic_state(dynamic_final)
            selected_path = "dynamic_gpu"
            if baseline == "model_graph":
                self.enable_model_cuda_graph()
                selected_path = "model_graph"

        return {
            "baseline": baseline,
            "warmup_steps": warmup_steps,
            "min_speedup": min_speedup,
            "dynamic_samples_ms": dynamic_samples,
            "graph_samples_ms": graph_samples,
            "dynamic_median_ms": dynamic_median,
            "graph_median_ms": graph_median,
            "graph_speedup": graph_speedup,
            "selected_path": selected_path,
            "graph_trial_stats": graph_trial_stats,
        }

    def _drop_whole_step(self) -> None:
        """Release the whole-step runner (graphs and pool): back to eager. The tier bookkeeping goes too:
        the runner is registered in the tier cache, which would otherwise pin its graph and workspaces
        and leave duplicate LRU entries for a later re-enable."""
        runner = self.whole_step_runner
        self.whole_step_runner = None
        self.whole_step_cuda_graph = False
        self._inference_config = replace(self._inference_config, execution="eager")
        self.capacity_policy = None
        self._tier_runners.clear()
        self._tier_lru.clear()
        self._tier_pool = None
        self._tier_demand_u.clear()
        self._tier_demand_t.clear()
        self._tier_check_counter = 0
        del runner
        gc.collect()
        torch.cuda.empty_cache()

    def _negotiate_plan(self):
        """Another model's plan (``fastmd.adapters.negotiate``), made at construction and kept for
        ``plan_record()``. Its precision applies from here on (IEEE FP32 unless TF32 is asked for). Under
        execution auto a model the runtime may capture (``can_capture``) that reads edges only, or a line
        graph the engine differentiates, gets, at that precision, a memory model measured on its own
        evaluation (K6 has no calibration for it) at the largest capacity either graph path would capture
        first (with triplets: the largest edge and triplet capacities), and the whole-step trial against
        model_graph (``_whole_step_trial``, which leaves the initial state and no graph); with ``log_stress``
        model_graph's bound covers its two runners' pools (the trial runs only where they fit). An explicit
        execution is kept where the model allows it."""
        from fastmd.adapters.memory import MeasuredMemoryModel
        from fastmd.adapters.negotiate import can_capture
        from fastmd.adapters.negotiate import negotiate
        from fastmd.adapters.negotiate import renegotiate_execution
        from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
        from fastmd.execution._memory_probe import edges_only_memory_runs
        from fastmd.execution._memory_probe import measured_memory
        config = self._inference_config
        # What was requested, not the mode configure_entrypoint resolved auto to.
        requested = replace(config, execution="auto") if self._execution_auto else config
        pbc = tuple(bool(axis) for axis in self.pbc.tolist())
        # model_graph with log_stress holds two runners' graph pools (the logged steps' next to the plain one).
        pools = 2 if self.log_stress else 1
        plan = negotiate(self.adapter, requested, entry="gpu_md", device=self.device, pbc=pbc,
                         model_graph_pools=pools)
        self._inference_config = replace(config, matmul_precision=plan.matmul_precision)
        # (configure_entrypoint resolved auto to eager without CUDA or under checkpointing.)
        if (self._execution_auto and config.execution != "eager" and self._edges_only and _fixed_builder_available
                and can_capture(self.adapter)):
            capacity = config.capacity
            with self._inference_config.scope():
                u_demand = int(self.graph_builder.build(self.state.positions).edge_index.shape[0]) // 2
                u_factor, _ = self._capture_factors(first=True)
                policy = JointCapacityPolicy(u_step=capacity.u_step, t_step=capacity.t_step,
                                             min_pad_u=capacity.min_pad_u, triplets=False)
                # whole_step's first capture, or model_graph's precapture envelope (one step above its bucket).
                target = 2 * max(policy.key_for(held_demand(u_demand, self._whole_step_headroom(
                                                    u_demand, capacity.u_step), u_factor), 0).u_capacity,
                                 policy.key_for(held_demand(u_demand, 0, u_factor), 0).u_capacity + capacity.u_step)
                runs = edges_only_memory_runs(
                    self.adapter, cell=self.cell, atomic_numbers=self.atomic_numbers, pbc=self.pbc,
                    positions=self.state.positions, n_real=self.n_atoms, device=self.device,
                    task=self.force_task + ("s" if self.log_stress else ""), candidate_skin=self.candidate_skin,
                    n_dummy=capacity.n_dummy)
                memory = measured_memory(MeasuredMemoryModel, runs, target, self.device)
                plan = renegotiate_execution(plan, self.adapter, requested, entry="gpu_md", device=self.device,
                                             pbc=pbc, capacity={"edges": target}, memory=memory,
                                             whole_step_trial=self._whole_step_trial, model_graph_pools=pools)
        elif (self._execution_auto and config.execution != "eager" and self._line_graph_batches
                and _fixed_builder_available and can_capture(self.adapter)):
            with self._inference_config.scope():
                target, runs = self._line_graph_memory_runs()
                memory = measured_memory(MeasuredMemoryModel, runs, target["edges"], self.device, target["triplets"])
                plan = renegotiate_execution(plan, self.adapter, requested, entry="gpu_md", device=self.device,
                                             pbc=pbc, capacity=target, memory=memory,
                                             whole_step_trial=self._whole_step_trial, model_graph_pools=pools)
        execution = plan.decision("execution")
        if not self._execution_auto and execution.chosen != config.execution:
            warnings.warn(f"GPUResidentMolecularDynamics: execution={config.execution!r} for model "
                          f"{self.adapter.name!r}: {execution.reason}", ConfigurationWarning, stacklevel=4)
        self._engine_plan = (self.adapter, plan)
        return plan

    def _line_graph_memory_runs(self):
        """The capacities execution auto measures a line-graph model the engine differentiates for (the
        larger edge and triplet capacities of whole_step's first capture and of model_graph's precapture
        envelope, one step above its bucket in each) and the probe's runs."""
        from fastmd.engine.capacity_policy import tier
        from fastmd.execution._memory_probe import line_graph_memory_runs
        capacity = self._inference_config.capacity
        graph = self.graph_builder.build(self.state.positions)
        u_demand, t_demand = int(graph.undirected2directed.shape[0]), int(graph.line_graph.shape[0])
        if _CAPACITY_THERMAL_FACTOR != 1.0:
            t_demand = int(round(t_demand * _CAPACITY_THERMAL_FACTOR))
        u_factor, t_factor = self._capture_factors(first=True)
        policy = JointCapacityPolicy(u_step=capacity.u_step, t_step=capacity.t_step, min_pad_u=capacity.min_pad_u,
                                     min_pad_t=capacity.min_pad_t, triplets=True)
        first = policy.start(held_demand(u_demand, self._whole_step_headroom(u_demand, capacity.u_step), u_factor),
                             held_demand(t_demand, self._whole_step_headroom(t_demand, capacity.t_step), t_factor))
        target = {"edges": 2 * max(first.u_capacity, tier(held_demand(u_demand, 0, u_factor), capacity.u_step,
                                                          capacity.min_pad_u) + capacity.u_step),
                  "triplets": max(first.t_capacity, tier(held_demand(t_demand, 0, t_factor), capacity.t_step, 0)
                                  + capacity.t_step)}
        runs = line_graph_memory_runs(
            self.adapter, cell=self._model_cell, atomic_numbers=self.atomic_numbers, pbc=self.pbc,
            positions=self.state.positions, n_real=self.n_atoms, device=self.device,
            task=self.force_task + ("s" if self.log_stress else ""), candidate_skin=self.candidate_skin,
            n_dummy=capacity.n_dummy)
        return target, runs

    # What a whole-step trial changes besides the state (restored after it).
    _TRIAL_KEPT = ("whole_step_capacity_history", "whole_step_recaptures", "_whole_step_accumulated", "_tier_switch_up",
                   "_tier_switch_down", "_tier_captures", "_tier_evictions", "_tier_replay_counter", "_tier_last_up_at",
                   "_tier_cost_per_triplet", "_tier_pool_resets", "_stress_at", "_evaluated_stress", "_window_targets")

    def _whole_step_trial(self, windows: int = 5) -> float:
        """execution auto's whole-step trial for a model other than MatRIS: model_graph's median time of one
        ``steps_per_launch`` window over whole_step's, ``windows`` windows each from the current state, each
        computing the stress where :meth:`step` would on every window (``log_stress`` logging every window).
        Neither runner stays; the state (its forces: this eager evaluation's) and the whole-step bookkeeping
        are restored, and no step is logged or counted. A CUDA OOM fails the trial (NaN: model_graph, whose
        own capture then falls back to eager)."""
        window, stress = self.steps_per_launch, self._stress_every_window

        def model_graph_window() -> None:
            for k in range(window):
                self._dynamic_step(stress=stress and k == window - 1)

        missing = object()
        self._ensure_forces()
        initial = self._clone_state()
        kept = {name: getattr(self, name, missing) for name in self._TRIAL_KEPT}
        kept = {name: copy.copy(value) if isinstance(value, (list, dict, deque)) else value
                for name, value in kept.items()}
        try:
            self.enable_model_cuda_graph()
            baseline = self._trial_times(windows, model_graph_window)
            self._drop_model_graph()
            self._restore_dynamic_state(initial)
            self.enable_whole_step_cuda_graph()
            speedup = float(np.median(baseline) / np.median(
                self._trial_times(windows, lambda: self._replay_whole_step(record=stress))))
        except torch.OutOfMemoryError:
            speedup = float("nan")
        self._drop_model_graph()
        self._drop_whole_step()
        for name, value in kept.items():
            if value is missing:
                self.__dict__.pop(name, None)
            else:
                setattr(self, name, value)
        self._restore_dynamic_state(initial)
        return speedup

    def _trial_times(self, steps: int, step) -> list[float]:
        """Wall times (ms) of ``steps`` calls of ``step``, each synchronized."""
        times = []
        torch.cuda.synchronize(self.device)
        for _ in range(steps):
            started = time.perf_counter()
            step()
            torch.cuda.synchronize(self.device)
            times.append((time.perf_counter() - started) * 1e3)
        return times

    def _drop_model_graph(self) -> None:
        """Release the model-graph runners (graphs and pools): back to eager."""
        for runner in (self.runner, self._stress_runner):
            if runner is not None and runner.cache:
                runner._release_pool()
        self.runner = self._stress_runner = None
        self._inference_config = replace(self._inference_config, execution="eager")
        gc.collect()
        torch.cuda.empty_cache()

    def _accumulate_whole_step_stats(self, runner: WholeStepGraphRunner) -> None:
        old_stats = runner.stats()
        for name in (
            "launches",
            "candidate_refreshes",
            "fallback_windows",
            "recovery_replays",
        ):
            self._whole_step_accumulated[name] += old_stats[name]
        for name in ("max_edge_count", "max_triplet_count", "max_displacement"):
            self._whole_step_accumulated[name] = max(
                self._whole_step_accumulated[name], old_stats[name]
            )

    def _tier_capture_estimate_bytes(self, key: CapacityKey) -> int:
        """Conservative standalone-size estimate for capturing a tier."""
        if self._tier_cost_per_triplet <= 0.0:
            return 0
        return int(
            self._tier_cost_per_triplet * key.t_capacity * _TIER_COST_MARGIN
        )

    @scoped_call
    def reset_whole_step_graph_pool(self) -> None:
        """Whole-pool reset: the only true memory-release valve.

        Shared-pool eviction is reuse-only (deleting one graph returns its
        blocks to the pool, not to the device), so genuinely reclaiming
        graph memory requires dropping EVERY runner sharing the pool,
        emptying the cache, and recapturing the active tier into a fresh
        pool. Intended triggers: budget pressure before a tier capture,
        protocol phase boundaries (e.g. between annealing segments), and
        before switching to checkpoint-off replay. If the recapture fails,
        gpu_md.state is restored from a snapshot and the exception
        propagates with no whole-step runner active; callers may fall back
        to the dynamic path.
        """
        if self.whole_step_runner is None:
            return
        state = self.whole_step_runner.state
        snapshot = (
            state.positions.clone(),
            state.momenta.clone(),
            state.forces.clone(),
            state.potential_energy.clone(),
        )
        self._accumulate_whole_step_stats(self.whole_step_runner)
        self.whole_step_runner = None
        self._tier_runners.clear()
        self._tier_lru.clear()
        gc.collect()
        torch.cuda.empty_cache()
        self._tier_pool = (
            torch.cuda.graph_pool_handle() if _WHOLE_STEP_TIER_CACHE else None
        )
        self._tier_pool_resets += 1
        key = CapacityKey(self._whole_u_capacity, self._whole_t_capacity)
        try:
            runner = self._make_whole_step_runner(
                positions=snapshot[0],
                momenta=snapshot[1],
                forces=snapshot[2],
                recapture=True,
            )
        except BaseException:
            self.state = GPUMDState(
                positions=snapshot[0],
                momenta=snapshot[1],
                forces=snapshot[2],
                potential_energy=snapshot[3][0],
            )
            raise
        self.whole_step_runner = runner
        if _WHOLE_STEP_TIER_CACHE:
            self._tier_runners[key] = runner
            self._tier_lru.append(key)
        self._sync_whole_state()

    def _switch_whole_step_tier(self, key: CapacityKey) -> None:
        """Switch the active whole-step tier, capturing it on first visit.

        Tier switches happen only between transactions: the caller holds a
        committed (or restored) state, so moving it into another tier's
        persistent buffers is the same reset pattern the post-precapture
        path already relies on. Cached tiers replay mutually exclusively
        and share one graph memory pool.
        """
        assert self.whole_step_runner is not None
        current = CapacityKey(self._whole_u_capacity, self._whole_t_capacity)
        if key == current:
            return
        old_runner = self.whole_step_runner
        old_state = old_runner.state
        # Run every fallible operation (capture can OOM, the candidate
        # assert can raise) BEFORE committing any bookkeeping, so an
        # exception leaves the old tier fully active and consistent.
        cached = self._tier_runners.get(key)
        created = cached is None
        if created:
            # Evict BEFORE capturing: pool eviction is reuse-only (deleting a
            # graph returns its blocks to the shared pool, not to the device),
            # so the new capture can only consume evicted blocks if the
            # eviction happens first — the reverse order grows the pool by a
            # full tier and strands the evicted blocks. Losing a cached tier
            # if the capture below then fails is acceptable: the active
            # runner and all bookkeeping remain consistent.
            while len(self._tier_runners) >= _WHOLE_STEP_MAX_TIERS:
                evict_key = next(
                    (k for k in self._tier_lru if k != current), None
                )
                if evict_key is None:
                    break
                self._tier_lru.remove(evict_key)
                evicted = self._tier_runners.pop(evict_key)
                del evicted
                self._tier_evictions += 1
                gc.collect()
                torch.cuda.empty_cache()
            # Budget guard: ask the device, not a ledger. If free memory
            # cannot cover a conservative estimate of the new tier, fire the
            # pool-reset valve (which also re-registers the ACTIVE tier and
            # refreshes old_state's backing runner) before capturing.
            estimate = self._tier_capture_estimate_bytes(key)
            floor_bytes = int(_TIER_MIN_FREE_GIB * (1 << 30))
            if estimate:
                free, _ = torch.cuda.mem_get_info()
                if free < estimate + floor_bytes:
                    self.reset_whole_step_graph_pool()
                    old_runner = self.whole_step_runner
                    old_state = old_runner.state
            self._whole_u_capacity = key.u_capacity
            self._whole_t_capacity = key.t_capacity
            free_before, _ = torch.cuda.mem_get_info()
            try:
                cached = self._make_whole_step_runner(
                    positions=old_state.positions,
                    momenta=old_state.momenta,
                    forces=old_state.forces,
                    recapture=True,
                )
            except BaseException:
                self._whole_u_capacity = current.u_capacity
                self._whole_t_capacity = current.t_capacity
                raise
            free_after, _ = torch.cuda.mem_get_info()
            consumed = max(0, free_before - free_after)
            if key.t_capacity > 0 and consumed > 0:
                self._tier_cost_per_triplet = max(
                    self._tier_cost_per_triplet,
                    consumed / key.t_capacity,
                )
            self._tier_runners[key] = cached
        else:
            cached.reset(
                positions=old_state.positions,
                momenta=old_state.momenta,
                forces=old_state.forces,
                potential_energy=old_state.potential_energy,
                reset_counters=True,
            )
            cached.builder.assert_candidate_capacity()
            self._whole_u_capacity = key.u_capacity
            self._whole_t_capacity = key.t_capacity
        self._accumulate_whole_step_stats(old_runner)
        self.whole_step_capacity_history.append(
            (key.u_capacity, key.t_capacity)
        )
        if created:
            self.whole_step_recaptures += 1
            self._tier_captures += 1
        self.whole_step_runner = cached
        if key in self._tier_lru:
            self._tier_lru.remove(key)
        self._tier_lru.append(key)
        while len(self._tier_runners) > _WHOLE_STEP_MAX_TIERS:
            evict_key = next((k for k in self._tier_lru if k != key), None)
            if evict_key is None:
                break
            self._tier_lru.remove(evict_key)
            evicted = self._tier_runners.pop(evict_key)
            del evicted
            self._tier_evictions += 1
            gc.collect()
        self._tier_demand_u.clear()
        self._tier_demand_t.clear()
        self._sync_whole_state()

    def _maybe_switch_down_tier(self) -> None:
        """Shrink to a smaller tier once demand has stably receded.

        Hysteresis: the window must be full, the bucket-rounded p95 plus a
        sigma-scaled margin must select a strictly smaller tier, and the
        window maximum must still fit that tier with minimum padding. The
        window is cleared on every switch, so consecutive switches are at
        least one full window apart.
        """
        maxlen = self._tier_demand_t.maxlen
        if not maxlen or len(self._tier_demand_t) < maxlen:
            return
        if self._tier_replay_counter - self._tier_last_up_at < _TIER_DOWN_COOLDOWN:
            return
        assert self.capacity_policy is not None
        u_values = sorted(self._tier_demand_u)
        t_values = sorted(self._tier_demand_t)
        index = int(0.95 * (len(u_values) - 1))
        mean_u = sum(u_values) / len(u_values)
        mean_t = sum(t_values) / len(t_values)
        std_u = (sum((v - mean_u) ** 2 for v in u_values) / len(u_values)) ** 0.5
        std_t = (sum((v - mean_t) ** 2 for v in t_values) / len(t_values)) ** 0.5
        # Sigma margins only; key_for already enforces min_pad headroom on
        # top, so folding min_pad in here would double-count padding and can
        # round the candidate ABOVE the current tier, blocking every shrink.
        margin_u = int(_TIER_HYSTERESIS_SIGMA * std_u)
        margin_t = int(_TIER_HYSTERESIS_SIGMA * std_t)
        candidate = self.capacity_policy.key_for(
            u_values[index] + margin_u,
            t_values[index] + margin_t,
        )
        current = CapacityKey(self._whole_u_capacity, self._whole_t_capacity)
        if candidate == current:
            return
        if (
            candidate.u_capacity > current.u_capacity
            or candidate.t_capacity > current.t_capacity
        ):
            return
        # The fixed-capacity builder needs at least one padding undirected
        # pair (u == u_capacity is an overflow), even with min_pad_u == 0.
        if (
            candidate.u_capacity
            < u_values[-1] + max(self.capacity_policy.min_pad_u, 1)
            or candidate.t_capacity < t_values[-1] + self.capacity_policy.min_pad_t
        ):
            return
        self._tier_switch_down += 1
        self._switch_whole_step_tier(candidate)

    def _grow_whole_step_runner(self, overflow: FixedCapacityOverflow) -> None:
        assert self.whole_step_runner is not None
        assert self.capacity_policy is not None
        old_runner = self.whole_step_runner
        old_state = old_runner.state
        if not _WHOLE_STEP_TIER_CACHE:
            # Pre-tier-cache ordering: stats accumulate before the
            # un-growable re-raise below (behavior-identical default path).
            self._accumulate_whole_step_stats(old_runner)
        required_u = (overflow.edge_count + 1) // 2
        self.capacity_policy.record_overflow(required_u, overflow.triplet_count)
        # A dimension that grows gets the whole-step headroom: one step per
        # growth recaptured every few steps while a large system warms up
        # (10,752 RDX atoms: 6 times in 300 steps), and the declared thermal factor.
        needed = self.capacity_policy.key_for(required_u, overflow.triplet_count)
        u_step, t_step = self.capacity_policy.u_step, self.capacity_policy.t_step
        u_factor, t_factor = self._capture_factors()
        grown = self.capacity_policy.grow(
            CapacityKey(self._whole_u_capacity, self._whole_t_capacity),
            held_demand(required_u, self._whole_step_headroom(needed.u_capacity, u_step), u_factor)
            if needed.u_capacity > self._whole_u_capacity else required_u,
            held_demand(overflow.triplet_count, self._whole_step_headroom(needed.t_capacity, t_step), t_factor)
            if needed.t_capacity > self._whole_t_capacity else overflow.triplet_count,
        )
        if (
            grown.u_capacity <= self._whole_u_capacity
            and grown.t_capacity <= self._whole_t_capacity
        ):
            raise overflow
        if _WHOLE_STEP_TIER_CACHE:
            self._tier_switch_up += 1
            self._tier_last_up_at = self._tier_replay_counter
            self._switch_whole_step_tier(grown)
            return
        self._whole_u_capacity = grown.u_capacity
        self._whole_t_capacity = grown.t_capacity
        self.whole_step_recaptures += 1
        self.whole_step_capacity_history.append(
            (self._whole_u_capacity, self._whole_t_capacity)
        )
        # Release the old runner (graph and pool) before capturing the grown
        # one: both at once would need two working sets. The state is kept.
        self.state = GPUMDState(
            positions=old_state.positions.clone(), momenta=old_state.momenta.clone(),
            forces=old_state.forces.clone(), potential_energy=old_state.potential_energy.clone())
        self.whole_step_runner = None
        del old_runner, old_state
        gc.collect()
        torch.cuda.empty_cache()
        self.whole_step_runner = self._make_whole_step_runner(
            positions=self.state.positions,
            momenta=self.state.momenta,
            forces=self.state.forces,
            recapture=True,
        )
        self._sync_whole_state()

    def _replay_whole_step(self, record: bool = False, settle: tuple | None = None) -> None:
        """One whole-step window (``record``: the record window), growing the capacity until it fits.
        ``settle``: the window was launched already (:meth:`run`), with nothing launched behind it: settle
        it, with ``settle[0]`` its flags on the host (None: the device's)."""
        assert self.whole_step_runner is not None
        while True:
            try:
                if settle is not None:
                    flags, settle = settle[0], None
                    self.whole_step_runner.settle(record=record or self.whole_step_runner.record_every_window,
                                                  flags=flags)
                else:
                    self.whole_step_runner.replay(strict=True, **({"record": True} if record else {}),
                                                  **({"launched": self._log_staged}
                                                     if self._staged_step is not None else {}))
                break
            except FixedCapacityOverflow as error:
                overflow = error
            # Grow outside the except block, without the traceback: its frames
            # would keep the old runner (graph and pool) alive during the new
            # capture (two working sets reserved at once).
            overflow.__traceback__ = None
            self._grow_whole_step_runner(overflow)
            del overflow
        if record:  # this runner's buffer (a tier switch below may replace the runner)
            self._stress_at = (self.nsteps + self.steps_per_launch, self.whole_step_runner.stress)
        self._after_window()

    def _after_window(self) -> None:
        """After a settled whole-step window: the capacity policy's observation, the tier cache's demand
        (and its shrink check), the state's views."""
        assert self.capacity_policy is not None
        self.capacity_policy.observe(
            self.whole_step_runner.last_edge_count // 2,
            self.whole_step_runner.last_triplet_count,
            CapacityKey(self._whole_u_capacity, self._whole_t_capacity),
        )
        if _WHOLE_STEP_TIER_CACHE:
            self._tier_replay_counter += 1
            self._tier_demand_u.append(self.whole_step_runner.last_edge_count // 2)
            self._tier_demand_t.append(self.whole_step_runner.last_triplet_count)
            self._tier_check_counter += 1
            if self._tier_check_counter >= _TIER_CHECK_INTERVAL:
                self._tier_check_counter = 0
                try:
                    self._maybe_switch_down_tier()
                except torch.OutOfMemoryError as error:
                    if not self._execution_auto:
                        raise
                    # The step is committed; only a smaller tier's capture
                    # failed. The active tier stays (no new shrink attempt
                    # before the cooldown), unless the pool-reset valve
                    # dropped every graph (its snapshot is this step).
                    self._tier_last_up_at = self._tier_replay_counter
                    self._tier_demand_u.clear()
                    self._tier_demand_t.clear()
                    if self.whole_step_runner is None:
                        self._whole_step_to_eager(error, rollback=False)
                        return
        self._sync_whole_state()

    @scoped_call
    def build_graph(self, positions: Tensor):
        graph = self.graph_builder.build(positions)
        if self._neighbor_count_max is not None:
            centres = graph.edge_index[:, 0].long()
            counts = torch.zeros(self.n_atoms, dtype=torch.int64, device=centres.device)
            counts.index_add_(0, centres, torch.ones_like(centres))
            torch.maximum(self._neighbor_count_max, counts.max().reshape(1), out=self._neighbor_count_max)
        return graph

    def neighbor_cap_summary(self) -> dict | None:
        """For a model that keeps at most ``max_neighbors`` neighbours per atom (its own graph drops the
        farthest ones): the cap, the largest number of neighbours within the cutoff one atom had in the
        builds so far (read from the device) and whether it exceeded the cap, in which case the engine
        evaluated every pair (the model's intent, not its own truncated graph) and this warns once.
        None without a cap."""
        if self._neighbor_count_max is None:
            return None
        cap = self.adapter.topology_spec().max_neighbors
        largest = int(self._neighbor_count_max.item())
        if largest > cap and not self._neighbor_cap_warned:
            self._neighbor_cap_warned = True
            warnings.warn(f"model {self.adapter.name!r} keeps at most {cap} neighbours per atom in its own graph; "
                          f"an atom had {largest} within the {self.atom_graph_cutoff:g} A cutoff, and the engine "
                          "evaluated every pair", RuntimeWarning, stacklevel=2)
        return {"max_neighbors": cap, "largest": largest, "exceeded": largest > cap}

    @scoped_call
    def evaluate(self, positions: Tensor, stress: bool = False) -> tuple[Tensor, Tensor]:
        """Forces and energy at ``positions``; ``stress``: task "efs", its stress (the model's GPa, or
        eV/A^3 from the engine-owned derivative) kept for the frame."""
        graph = self.build_graph(positions)
        runner = self._stress_runner if stress else self.runner
        if runner is not None:
            out, n_real = runner.run(graph)
            forces = out["f"][0][:n_real]
        else:
            batch = graph
            if self._edges_only:
                graph = self.adapter.prepare(graph)  # the model's input over this build's batch
            elif self._line_graph_batches:
                batch = line_graph_batch(graph)  # the graph batch over this build's graph, with positions
                graph = self.adapter.prepare(batch)
            if self._derivative is None:
                out = self.adapter.evaluate(
                    [graph], task=self.force_task + "s" if stress else self.force_task, is_training=False,
                    handle_isolated_atoms=not self.validate_isolated_atoms,
                )
            else:
                out = self._derivative(batch, graph, task=self.force_task + "s" if stress else self.force_task)
            forces = out["f"][0] if isinstance(out["f"], (list, tuple)) else out["f"]
        if stress:
            self._evaluated_stress = out["s"][0].detach()

        energy = (self.adapter if self._derivative is None else self._derivative).extensive_energy(
            out["e"].reshape(-1)[0], self.n_atoms)
        return forces.detach(), energy.detach()

    def force_fn(self, positions: Tensor, stress: bool = False) -> Tensor:
        forces, energy = self.evaluate(positions, stress=True) if stress else self.evaluate(positions)
        self.state.potential_energy = energy
        return forces

    def _ensure_forces(self, stress: bool = False) -> None:
        if self.state.forces is None:
            force_fn = (lambda positions: self.force_fn(positions, stress=True)) if stress else self.force_fn
            if self.stateful_integrator is not None:
                # F(q_0) at the float64 positions (the builder casts to fp32)
                forces = force_fn(self.stateful_integrator.positions)
                self.stateful_integrator.set_forces(
                    forces, energy=self.state.potential_energy.reshape(1)
                )
                self.state.forces = forces
            else:
                self.state.forces = force_fn(self.state.positions)
            if stress:
                self._stress_at = (self.nsteps, self._evaluated_stress)

    def _stress_due(self, step: int) -> bool:
        """Whether the force evaluation of the state at ``step`` computes its stress (a logged frame)."""
        return self.log_stress and self.logger.should_log(step)

    def _logged_stress(self, step: int | None = None) -> Tensor:
        """The current state's stress for its frame, eV/A^3 (MLIPCalculator's float32 GPa conversion):
        the logged step's evaluation; one evaluation now only for a first frame whose forces exist already
        (a resumed run, or forces evaluated before the first step). ``step``: the state's step (default
        ``nsteps``; a window that is not settled yet ends on a later one)."""
        step = self.nsteps if step is None else step
        if self._stress_at is None:
            source = self.stateful_integrator if self.stateful_integrator is not None else self.state
            self.evaluate(source.positions, stress=True)
            self._stress_at = (step, self._evaluated_stress)
        if self._stress_at[0] != step:
            raise RuntimeError(f"GPUResidentMolecularDynamics: no stress was computed for the logged step "
                               f"{step} (last: step {self._stress_at[0]})")
        scale = stress_unit_scale(self.adapter)
        return self._stress_at[1] if scale == 1.0 else self._stress_at[1] * scale

    def _kinetic_energy(self) -> Tensor:
        if self.stateful_integrator is not None:
            return self.stateful_integrator.kinetic_energy()
        return kinetic_energy_fp64(self.state.momenta, self.masses)

    def _log_current(self) -> None:
        if not self.logger.enabled:
            return
        self._log_staged()  # frames stay in step order
        self._ensure_forces(**({"stress": True} if self.log_stress else {}))
        assert self.state.forces is not None
        assert self.state.potential_energy is not None
        # a stateful integrator's float64 state, else the float32 state
        source = (
            self.stateful_integrator
            if self.stateful_integrator is not None
            else self.state
        )
        extras = None
        if isinstance(self.integrator, GPUNoseHooverChainNVT):
            extras = {
                "nhc_eta": self.integrator.eta,
                "nhc_p_eta": self.integrator.p_eta,
                "nhc_conserved_energy": self.integrator.conserved_energy(),
            }
        self.logger.enqueue(
            step=self.nsteps,
            time_ps=self.nsteps * self.timestep / 1000.0,
            positions=source.positions,
            momenta=source.momenta,
            forces=self.state.forces,
            cell=self._model_cell,
            potential_energy=self.state.potential_energy,
            kinetic_energy=self._kinetic_energy(),
            temperature=self.temperature(),
            extras=extras,
            **({"stress": self._logged_stress()} if self.log_stress else {}),
        )

    def _stages_frames(self) -> bool:
        """Whole-step MD on the Nose-Hoover chain: a logged window's frame is staged on the device and
        enqueued once the next window is launched; its thermostat sums and device-to-host copies run on
        the logger's copy stream, so that they overlap that window instead of idling the GPU between
        windows. (A float32 model's float32 forces and energy stay exact in the float64 staged state.)"""
        return (self.whole_step_runner is not None
                and isinstance(self.integrator, GPUNoseHooverChainNVT) and self.logger.enabled)

    def _stage_frame(self, step: int | None = None) -> None:
        """Copy the current (logged) state once the previous frame's reads of the copies on the logger's
        copy stream are done: the integrator's flat state, its thermostat masses and the stress (copies
        on the MD stream), and the target's kT and 3N kT (host values); an event marks the copies for the
        copy stream. The frame's thermostat sums read this snapshot, not a later target's constants.
        ``step``: the state's step (default ``nsteps``)."""
        step = self.nsteps if step is None else step
        integrator = self.stateful_integrator
        if self._staged is None:
            self._staged = (NHCState.allocate(self.n_atoms, integrator.tchain, self.device),
                            torch.zeros((3, 3), dtype=torch.float64, device=self.device) if self.log_stress else None,
                            torch.cuda.Event(), torch.empty_like(integrator.Q))
        state, stress, staged, masses = self._staged
        md_stream = torch.cuda.current_stream(self.device)
        if self.logger.copy_stream is not None:
            md_stream.wait_stream(self.logger.copy_stream)
        state.flat.copy_(integrator.state.flat)
        masses.copy_(integrator.Q)
        if stress is not None:
            stress.copy_(self._logged_stress(step))
        staged.record(md_stream)
        self._staged_step = step
        self._staged_constants = (integrator.kT, integrator.ndof_kT, masses)

    def _log_staged(self) -> None:
        """Enqueue the staged frame, if any (what :meth:`_log_current` logs, from the copies): on the
        logger's copy stream after the staging copies, so that no MD-stream work waits for it."""
        if self._staged_step is None:
            return
        step, self._staged_step = self._staged_step, None
        state, stress, staged, _ = self._staged
        constants = self._staged_constants
        stream = self.logger.copy_stream
        if stream is None:
            self._enqueue_staged(step, state, stress, constants)
            return
        stream.wait_event(staged)
        with torch.cuda.stream(stream):
            self._enqueue_staged(step, state, stress, constants)

    def _enqueue_staged(self, step: int, state: NHCState, stress: Tensor | None, constants: tuple) -> None:
        integrator = self.stateful_integrator
        self.logger.enqueue(
            step=step,
            time_ps=step * self.timestep / 1000.0,
            positions=state.positions,
            momenta=state.momenta,
            forces=state.forces,
            cell=self._model_cell,
            potential_energy=state.potential_energy[0],
            kinetic_energy=integrator.kinetic_energy(state),
            temperature=integrator.temperature(state),
            extras={
                "nhc_eta": state.eta,
                "nhc_p_eta": state.p_eta,
                "nhc_conserved_energy": integrator.conserved_energy(state=state, constants=constants),
            },
            **({"stress": stress} if stress is not None else {}),
        )

    def _log_initial_if_needed(self) -> None:
        if self.logger.enabled and not self._logged_initial:
            self._log_current()
            self._logged_initial = True

    @scoped_call
    def precapture(self, steps: int = 0, proactive: bool = True) -> None:
        if self.whole_step_runner is not None:
            return
        if self.runner is None:
            return
        if steps > 0 and self.stateful_integrator is not None:
            integrator = self.stateful_integrator
            saved = integrator.snapshot()
            try:
                with self.runner.exact_capacity_capture():
                    forces, energy = self.evaluate(integrator.positions)
                    integrator.set_forces(forces, energy=energy.reshape(1))
                    for _ in range(steps):
                        forces, energy = self.evaluate(integrator.step_pre_force())
                        integrator.step_post_force(forces, energy=energy.reshape(1))
                sample_positions = integrator.positions.to(torch.float32)
            finally:
                integrator.load_snapshot(saved)
        elif steps > 0:
            positions = self.state.positions.clone()
            momenta = self.state.momenta.clone()
            with self.runner.exact_capacity_capture():
                forces, _ = self.evaluate(positions)
                force_fn = lambda pos: self.evaluate(pos)[0]
                for _ in range(steps):
                    positions, momenta, forces = self.integrator.step(
                        positions, momenta, forces, force_fn
                    )
            sample_positions = positions
        else:
            sample_positions = self.state.positions
        sample_graph = self.build_graph(sample_positions)
        self.runner.precapture(sample_graph, proactive=proactive)
        if self._stress_runner is not None:  # the buckets of the same sample (not the probe's exact ones)
            self._stress_runner.precapture(sample_graph, proactive=proactive)

    def _auto_mode(self, ensemble: str) -> tuple[str, str]:
        """execution="auto" (the default): (mode, reason). whole_step where it
        applies, else model_graph, else eager; a capture only where the
        calculator's rule allows one (capacity_policy.graph_execution: a measured
        K6 calibration bounds the model and its model_graph prediction at the
        recompute level eager takes fits the device)."""
        from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
        from fastmd.graph.fixed_capacity_builder import _fixed_builder_unavailable
        from fastmd.engine.capacity_policy import graph_execution
        config = self._inference_config
        if config.execution == "eager":
            return "eager", "CPU or checkpoint (no CUDA graph)"
        if self.adapter.compiled:
            return "eager", "compiled model"
        if config.capacity.warmup < 1:
            return "eager", "capacity.warmup=0 (a capture needs a warmup)"
        rows = 0 if self._edges_only else int(self.graph_builder.build(self.state.positions).line_graph.shape[0])
        graph, reason = graph_execution(self.model, config, rows, self.device)
        if not graph:
            return "eager", reason
        if not bool(self.pbc.all()) or self.candidate_skin <= 0 or not _fixed_builder_available:
            return "model_graph", (f"{reason}; whole_step needs full PBC, a skin and the fixed-capacity builder"
                                   + ("" if _fixed_builder_available else f" ({_fixed_builder_unavailable})"))
        return "whole_step", reason

    def _whole_step_to_eager(self, error, rollback: bool = True) -> None:
        """execution="auto": a CUDA OOM in a whole-step capture (construction,
        capacity growth) continues on the eager path. The failed window was
        rolled back to its start (the runner's backup, and the stateful
        integrator's), which becomes the eager state; graphs and pool go."""
        runner = self.whole_step_runner
        if runner is not None:
            if rollback and runner.launches:
                runner._restore_backup()
            self._sync_whole_state()
            state = self.state
            self.state = GPUMDState(
                positions=state.positions.clone(), momenta=state.momenta.clone(),
                forces=None if state.forces is None else state.forces.clone(),
                potential_energy=None if state.potential_energy is None else state.potential_energy.clone())
        self.whole_step_runner = None
        self.whole_step_cuda_graph = False
        self._inference_config = replace(self._inference_config, execution="eager")
        self.capacity_policy = None
        self._tier_runners.clear()
        self._tier_lru.clear()
        self._tier_pool = None
        self._tier_demand_u.clear()
        self._tier_demand_t.clear()
        del runner
        gc.collect()
        torch.cuda.empty_cache()
        self._auto_reason = ("whole_step ran out of CUDA memory "
                               f"({str(error).splitlines()[0][:160]}); eager")
        warnings.warn(self._auto_reason, RuntimeWarning, stacklevel=3)

    @property
    def temperature_target(self) -> float | None:
        """The thermostat target (K); None for nve."""
        if isinstance(self.integrator, GPUNoseHooverChainNVT):
            return float(self.integrator.temperature_K)
        return self.integrator.temperature_target if self.integrator.ensemble == "nvt" else None

    def set_temperature(self, temperature: float) -> None:
        """New thermostat target before the next step (sync-free; graphs read it at replay)."""
        if self.temperature_target is None:
            raise ValueError("GPUResidentMolecularDynamics: nve has no thermostat temperature")
        self.integrator.set_temperature(float(temperature))

    @scoped_call
    def step(self) -> GPUMDState:
        if self.whole_step_runner is None:
            self._log_staged()
        self._log_initial_if_needed()
        window = self.steps_per_launch
        if self.whole_step_runner is not None:
            start_target = self.temperature_target
            self._set_window_targets(self.nsteps)
            try:
                if self._stress_due(self.nsteps + window):
                    self._replay_whole_step(record=True)
                else:
                    self._replay_whole_step()
            except torch.OutOfMemoryError as error:
                if not self._execution_auto:
                    raise
                self._whole_step_to_eager(error)
                if self._window_targets is not None:
                    self.set_temperature(start_target)
            else:
                self.nsteps += window
                if self.logger.should_log(self.nsteps):
                    if self._stages_frames():
                        self._stage_frame()
                    else:
                        self._log_current()
                return self.state
        for k in range(window):
            if self.ramp is not None:
                setpoint = self.ramp.setpoint(self.nsteps)
                if setpoint is not None:
                    self.set_temperature(setpoint)
            if k == window - 1 and self._stress_due(self.nsteps + 1):
                self._dynamic_step(stress=True)
                self._stress_at = (self.nsteps + 1, self._evaluated_stress)
            else:
                self._dynamic_step()
            self.nsteps += 1
        if self.logger.should_log(self.nsteps):
            self._log_current()
        return self.state

    def _dynamic_step(self, stress: bool = False) -> None:
        """One eager or model-graph step; ``stress``: its force evaluation also computes the stress."""
        self._ensure_forces()
        if self.stateful_integrator is not None:
            self._stateful_step(**({"stress": True} if stress else {}))
        else:
            positions, momenta, forces = self.integrator.step(
                self.state.positions,
                self.state.momenta,
                self.state.forces,
                (lambda positions: self.force_fn(positions, stress=True)) if stress else self.force_fn,
            )
            self.state = GPUMDState(
                positions=positions,
                momenta=momenta,
                forces=forces,
                potential_energy=self.state.potential_energy,
            )

    def _stateful_step(self, stress: bool = False) -> None:
        """Dynamic (eager or model-graph) step of a stateful integrator:
        pre-force half -> one force evaluation -> post-force half."""
        integrator = self.stateful_integrator
        positions = integrator.step_pre_force()
        forces = self.force_fn(positions, stress=True) if stress else self.force_fn(positions)
        energy = self.state.potential_energy
        integrator.step_post_force(forces, energy=energy.reshape(1))
        self.state = GPUMDState(
            positions=positions.to(torch.float32),
            momenta=integrator.momenta.to(torch.float32),
            forces=forces,
            potential_energy=energy,
        )

    @scoped_call
    def run(self, steps: int) -> GPUMDState:
        """``steps`` MD steps (a multiple of ``steps_per_launch``: each :meth:`step` advances one window)."""
        if int(steps) % self.steps_per_launch:
            raise ValueError(f"GPUResidentMolecularDynamics.run: {steps} steps are not whole windows of "
                             f"steps_per_launch={self.steps_per_launch}")
        windows = int(steps) // self.steps_per_launch
        while windows:
            if self.whole_step_runner is not None and windows > 1:
                windows -= self._run_windows(windows)
            else:
                self.step()
                windows -= 1
        self._log_staged()
        return self.state

    def _run_windows(self, count: int) -> int:
        """Up to ``count`` whole-step windows of :meth:`run`, each next one launched before the host reads
        this one's flags, so that the MD stream does not wait for the host between windows. Every window is
        settled as :meth:`step` settles it, from a host copy of its flags. When a window with one launched
        behind it needs more than host bookkeeping (a candidate refresh, an overflow, a non-finite state),
        both are rolled back to its start and :meth:`step` runs it as if neither had been launched (none is
        launched behind a likely refresh). None is launched behind a logged window whose frame is not staged
        on the device, nor where the tier cache checks its demand. Returns the windows run (fewer once a CUDA
        OOM left the whole-step path)."""
        window = self.steps_per_launch
        self._log_initial_if_needed()
        # ahead: the flags slot of the window launched behind this one; staged: this window's frame, staged
        # before that launch; target, target_ahead: the thermostat target before this window's targets and
        # before the next one's were set (a window's targets build on it), restored with a rollback
        done, ahead, runner, staged, target, target_ahead = 0, None, None, False, None, None
        try:
            while done < count and self.whole_step_runner is not None:
                runner, start = self.whole_step_runner, self.nsteps
                record = self._stress_due(start + window)
                if ahead is None:
                    target = self.temperature_target
                    self._set_window_targets(start)
                    slot = runner.launch(record=record)
                    if record:
                        self._stress_at = (start + window, runner.stress)
                    self._log_staged()  # a frame staged before this window, enqueued while it runs
                else:
                    slot, ahead, target = ahead, None, target_ahead
                logged = self.logger.should_log(start + window)
                staged = False
                if done + 1 < count and self._launches_behind(runner, logged):
                    if logged:  # before the next window overwrites the state
                        self._stage_frame(start + window)
                        staged = True
                    target_ahead = self.temperature_target
                    self._set_window_targets(start + window)
                    record_next = self._stress_due(start + 2 * window)
                    ahead = runner.launch(record=record_next, behind=True)
                    if record_next:
                        self._stress_at = (start + 2 * window, runner.stress)
                    self._windows_ahead += 1
                flags = runner.launched_flags(slot)
                work = runner.settling_work(flags)
                if ahead is not None and work is not None:
                    # As if neither window had been launched: step() runs this one (its refresh, fallback,
                    # growth or eager steps). No local reference may keep a runner that a growth replaces.
                    runner.rewind()
                    ahead = runner = None
                    self._windows_rolled_back += 1
                    if staged:  # a frame of the window rolled back
                        self._staged_step, staged = None, False
                    if target != self.temperature_target:
                        self.set_temperature(target)
                    self.step()
                    done += 1
                    continue
                if ahead is not None:  # host bookkeeping only; the window behind stands
                    runner.settle(record=record, flags=flags)
                    self._after_window()
                else:  # as step() settles it (the device's flags after an overflow); a CUDA OOM goes eager
                    runner = None
                    if not self._settle_window(record, None if work == "overflow" else flags, target):
                        self.step()  # this window's eager steps
                        done += 1
                        continue
                self.nsteps += window
                if logged:
                    if staged:
                        staged = False
                        self._log_staged()
                    elif self._stages_frames():
                        self._stage_frame()
                    else:
                        self._log_current()
                done += 1
        except BaseException:
            if ahead is not None:  # back to the state of the window settled (or being settled)
                runner.drop()
                if target_ahead != self.temperature_target:
                    self.set_temperature(target_ahead)
            if staged:  # the frame of a window that was not committed
                self._staged_step = None
            raise
        return done

    def _settle_window(self, record: bool, flags: tuple | None, target) -> bool:
        """:meth:`step`'s checks of the window :meth:`run` launched last, with nothing launched behind it
        (``flags``: its host copy; None: the device's, after an overflow), and its capacity growth. False: a
        CUDA OOM under execution auto left the whole-step path with the window rolled back to its start, and
        ``target`` (the thermostat target before its targets were set) restored for its eager steps."""
        try:
            self._replay_whole_step(record, settle=(flags,))
        except torch.OutOfMemoryError as error:
            if not self._execution_auto:
                raise
            self._whole_step_to_eager(error)
            if self._window_targets is not None:
                self.set_temperature(target)
            return False
        return True

    def _launches_behind(self, runner: WholeStepGraphRunner, logged: bool) -> bool:
        """Whether :meth:`run` launches the next window behind this unsettled one: not when this one's frame
        is not staged on the device (``_log_current`` reads the live state), not when the tier cache checks
        its demand after it (a switch moves the state), not when it will likely refresh the candidates."""
        if logged and not self._stages_frames():
            return False
        if _WHOLE_STEP_TIER_CACHE and self._tier_check_counter + 1 >= _TIER_CHECK_INTERVAL:
            return False
        return runner.refresh_unlikely()

    def temperature(self) -> Tensor:
        """2 Ekin / (3N kB), a float64 reduction in every ensemble."""
        if self.stateful_integrator is not None:
            return self.stateful_integrator.temperature()
        return 2.0 * self._kinetic_energy() / (self.integrator.degrees_of_freedom * units.kB)

    def conserved_energy(self) -> Tensor:
        """NHC conserved quantity E_pot + E_kin + E_thermostat (``nvt_nhc``)."""
        if not isinstance(self.integrator, GPUNoseHooverChainNVT):
            raise ValueError("conserved_energy is defined for ensemble='nvt_nhc'")
        self._ensure_forces()
        return self.integrator.conserved_energy()

    @scoped_call
    def whole_step_stats(self, *, capacity_plan: bool = True) -> dict[str, object] | None:
        """Whole-step runner statistics; ``capacity_plan=False`` leaves out the capacity policy's
        observation summary and recommended keys (a Python search over every replay's demand)."""
        if self.whole_step_runner is None:
            return None
        stats: dict[str, object] = self.whole_step_runner.stats()
        for name in (
            "launches",
            "candidate_refreshes",
            "fallback_windows",
            "recovery_replays",
        ):
            stats[name] += self._whole_step_accumulated[name]
        for name in ("max_edge_count", "max_triplet_count", "max_displacement"):
            stats[name] = max(stats[name], self._whole_step_accumulated[name])
        stats["capacity_recaptures"] = self.whole_step_recaptures
        stats["pool_resets"] = self._tier_pool_resets
        stats["windows_launched_ahead"] = self._windows_ahead
        stats["windows_rolled_back"] = self._windows_rolled_back
        stats["capacity_history"] = list(self.whole_step_capacity_history)
        if self.capacity_policy is not None and capacity_plan:
            stats["capacity_policy"] = self.capacity_policy.stats()
        if _WHOLE_STEP_TIER_CACHE:
            stats["tier_cache"] = {
                "tiers_cached": len(self._tier_runners),
                "tier_keys": [
                    (key.u_capacity, key.t_capacity) for key in self._tier_lru
                ],
                "switch_up": self._tier_switch_up,
                "switch_down": self._tier_switch_down,
                "captures": self._tier_captures,
                "evictions": self._tier_evictions,
            }
        return stats

    def model_graph_stats(self) -> dict[str, object] | None:
        """The bucketed runner's statistics; with log_stress also its logged steps' runner ("stress_runner")."""
        if self.runner is None:
            return None
        stats = self.runner.stats()
        if self._stress_runner is not None:
            stats["stress_runner"] = self._stress_runner.stats()
        return stats

    def snapshot_atoms(self) -> Atoms:
        atoms = self.atoms_template.copy()
        source = self.stateful_integrator if self.stateful_integrator is not None else self.state
        atoms.set_positions(source.positions.detach().cpu().numpy())
        atoms.set_momenta(source.momenta.detach().cpu().numpy())
        return atoms

    def flush_log(self) -> None:
        """Wait for queued trajectory and log frames to reach their files."""
        self._log_staged()
        for logger in (self.outputs.logger, self.logger):
            if logger is not None:
                logger.flush()

    def close(self) -> None:
        """Flush outputs and release the loaded adapter after its last user closes."""
        try:
            self._close()
        finally:
            release_adapter(self)


    def _close(self) -> None:
        self._finalizer.detach()
        try:
            if self.logger is not None:
                self._log_staged()
                self.logger.close()
        finally:
            self.outputs.close()
        if self._neighbor_count_max is not None:
            self.neighbor_cap_summary()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._close()
        finally:
            release_adapter(self)
        return False

    # --- outputs (MDOutputs: rank 0 owns the files; every rank joins the gathers) ---
    def write_outputs(self) -> dict | None:
        """Thermo line, frame and restart file due at the current step (call after every step).

        Returns the thermodynamics when any output was due, else None. The
        last step (``total_steps``) always gets a thermo line and a restart.
        """
        return self.outputs.write()

    def log_frame(self, thermo: dict | None = None) -> None:
        """Gather the current frame by ID into rank 0's EXTXYZ logger (asynchronous file write)."""
        self.outputs.log_frame(thermo)

    def save_restart(self, path=None) -> None:
        """Write the current state to rank 0's P-independent restart file (default ``output_dir``).

        ONE gather_by_id; ID-ordered rows, the cached forces and energies (a
        continuation does not re-evaluate), cell, step, ``parameters``, the
        thermostat state, the output files' offsets and ``engine`` ("domain"
        for historical files, "native" for FastMD with ``ids = arange(N)``).
        """
        self.outputs.save_restart(path)


    def md_fields(self) -> dict:
        """Views of the current single-device state for output and restart."""
        return self._native_fields()

    def _native_fields(self) -> dict:
        """md_fields of the native engine: all atoms, ``ids = arange(N)`` (forces evaluated if missing)."""
        self._ensure_forces()
        integrator = self.stateful_integrator  # the float64 state of nvt_nhc
        source = self.state if integrator is None else integrator
        if integrator is None:
            masses = self.masses
        else:
            if getattr(self, "_masses64", None) is None:
                self._masses64 = torch.as_tensor(self.atoms_template.get_masses(), dtype=torch.float64,
                                                 device=self.device)
            masses = self._masses64
        table = self.adapter.reference_energies()
        reference = (self.state.potential_energy.new_zeros(()) if table is None else
                     table[0, self.atomic_numbers.long() - 1].sum().detach().to(torch.float32))
        return {"ids": torch.arange(self.n_atoms, device=self.device), "numbers": self.atomic_numbers,
                "positions": source.positions, "momenta": source.momenta, "masses": masses,
                "forces": self.state.forces, "energy": self.state.potential_energy.reshape(()),
                "reference_energy": reference}

    def thermostat_state(self) -> dict:
        """Replicated thermostat state (host values; downloads eta/p_eta for nvt_nhc)."""
        if self.thermostat is not None:
            return self.thermostat.state_dict()
        if self.integrator.ensemble == 'nvt':
            return {'temperature_K': float(self.integrator.temperature_target)}
        return {}

    def collective_check(self, error: str | None) -> None:
        """Local implementation of the output protocol's error check."""
        if error:
            raise ValueError(f"GPUResidentMolecularDynamics outputs: {error}")

    @property
    def thermostat(self) -> GPUNoseHooverChainNVT | None:
        """The Nose-Hoover chain integrator (nvt_nhc), else None."""
        return self.integrator if isinstance(self.integrator, GPUNoseHooverChainNVT) else None

    @property
    def temperature_K(self) -> float | None:
        """The thermostat's current target (None for nve)."""
        return self.temperature_target

    def target_temperature(self) -> float | None:
        """The thermo column: the ramp at the current step, else the thermostat target."""
        return self.ramp.target(self.nsteps) if self.ramp is not None else self.temperature_K

    def thermodynamics(self) -> dict:
        """Global observables of the current state: ONE fused allreduce and ONE download (collective).

        Identical on every rank, as Python numbers: step, time_fs,
        target_temperature, potential_energy, kinetic_energy, total_energy,
        temperature, momentum (the total, 3 floats), dof (3N; the
        temperature's) and, for nvt_nhc, eta, p_eta, thermostat_energy and
        conserved_energy (ASE ``get_conserved_energy``). The kinetic energy is
        the float64 ``kinetic_energy_fp64``.
        """
        fields = self.md_fields()  # the domain's: the prediction's state and energy
        momenta, masses = fields["momenta"], fields["masses"]
        totals = torch.cat((kinetic_energy_fp64(momenta, masses).reshape(1),
                            momenta.to(torch.float64).sum(dim=0)))
        self.comm.allreduce_sum_(totals)
        values = [fields["energy"].to(torch.float64).reshape(1), totals]
        thermostat = self.thermostat
        if thermostat is not None:
            values += [thermostat.eta.reshape(-1), thermostat.p_eta.reshape(-1),
                       thermostat.thermostat_energy().to(torch.float64).reshape(1)]
        host = torch.cat(values).cpu().tolist()
        potential, kinetic = host[0], host[1]
        dof = 3 * self.n_atoms
        result = {"step": self.nsteps, "time_fs": self.nsteps * self.timestep,
                  "target_temperature": self.target_temperature(), "potential_energy": potential,
                  "kinetic_energy": kinetic, "total_energy": potential + kinetic,
                  "temperature": 2 * kinetic / (dof * units.kB), "momentum": host[2:5], "dof": dof}
        if thermostat is not None:
            tchain = self.parameters["tchain"]
            energy = host[5 + 2 * tchain]
            result.update(eta=host[5:5 + tchain], p_eta=host[5 + tchain:5 + 2 * tchain],
                          thermostat_energy=energy, conserved_energy=potential + kinetic + energy)
        return result





class _WindowTargets:
    """Per-step thermostat targets of a multi-step whole-step window (``steps_per_launch > 1``).

    Row k of a device table holds the integrator's ``target_row`` of step k's target; the captured window
    loads it before step k (``load_target_row``), so a ramp inside a window follows the same targets as one
    step per launch, bitwise. The host refills the table between replays (pinned, sync-free) only when the
    window's targets change.
    """

    def __init__(self, integrator, steps: int, device) -> None:
        from fastmd.execution.gpu_integrator import _PinnedUpload
        self.integrator = integrator
        width = len(integrator.target_row(1.0))
        self.rows = torch.empty((steps, width), dtype=torch.float64, device=device)
        self._upload = _PinnedUpload(self.rows)
        self._loaded = None

    def fill(self, temperatures) -> None:
        temperatures = tuple(float(value) for value in temperatures)
        if temperatures != self._loaded:
            self._upload([self.integrator.target_row(value) for value in temperatures])
            self._loaded = temperatures

    def apply(self, k: int) -> None:
        self.integrator.load_target_row(self.rows[k])


def _close_quietly(*closables) -> None:
    """weakref.finalize callback of a dropped driver: stop its writer threads (errors are not raised)."""
    for closable in closables:
        try:
            if closable is not None:
                closable.close()
        except Exception:  # noqa: BLE001 - nobody is left to raise to
            pass


