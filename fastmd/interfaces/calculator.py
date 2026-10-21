from __future__ import annotations
import pickle

from dataclasses import replace
import warnings

from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes, all_properties
import numpy as np
import torch

from fastmd.adapters import ModelAdapter, load_adapter
from fastmd.adapters.negotiate import can_capture, negotiated
from fastmd.config import ConfigurationWarning, InferenceConfig
from fastmd.engine.entry_config import (apply_checkpoint, configure_entrypoint, config_summary, current_adapter,
                                   release_adapter, scoped_call, use_adapter)
from fastmd.engine.energy_derivative import engine_derivative
from fastmd.engine import saved_recompute

from pymatgen.io.ase import AseAtomsAdaptor
from fastmd.graph import RadiusGraph
from fastmd.graph.fixed_capacity_builder import line_graph_batch

from ase.optimize import (
    BFGS, BFGSLineSearch, 
    FIRE, LBFGS, 
    LBFGSLineSearch, MDMin
)

names = [
    "BFGS", "BFGSLineSearch", 
    "FIRE", "LBFGS", 
    "LBFGSLineSearch", "MDMin"
]

OPTIMIZERS = {name: globals()[name] for name in names}


def require_finite_geometry(positions, cell, where: str) -> None:
    """Reject non-finite host coordinates before a graph is built.

    A NaN distance fails every cutoff test, so NaN atoms would silently count
    as isolated: under isolated_atoms='reference' the model then returns the
    finite reference energy and zero forces. Host-side, ~10 us per 10k atoms.
    """
    finite = np.isfinite(positions)
    if not finite.all():
        bad = np.flatnonzero(~finite.all(axis=1))
        raise ValueError(
            f"{where}: {bad.size} of {len(finite)} atoms have non-finite positions "
            f"(first index {bad[0]})"
        )
    if not np.isfinite(cell).all():
        raise ValueError(f"{where}: the cell is not finite: {np.asarray(cell).tolist()}")


class MLIPCalculator(Calculator):
    """MatRIS Calculator for ASE applications."""
    
    implemented_properties = ("energy", "forces", "stress", "magmoms")  # type: ignore
    
    @configure_entrypoint("calculator")
    def __init__(
        self,
        model_path: str = None,
        model: torch.nn.Module | ModelAdapter | str = "matris_10m_oam",
        task: str = "efs",
        device: str = "cpu",
        enable_compile: bool = False,
        config: InferenceConfig | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            model (MatRIS): Instance of a MatRIS model. If set to None, the default MatRIS is loaded.
                A model adapter (``fastmd.adapters``) is used as it is; one whose model reads
                edges only (no triplets) needs a CUDA device and cells periodic in all three
                directions.
            task (str): The prediction task. Can be 'e', 'em', 'ef', 'efs', 'efsm'.
            device (str): The device to be used for predictions,
            enable_compile (bool): Whether to compile interaction blocks.
            config: Unified execution, optimization, checkpoint, isolated-atom,
                and capacity settings. Defaults to eager execution, reference
                isolated-atom handling, and checkpointing off. These settings
                also apply when passing an already constructed MatRIS model.
                The model (loaded or supplied) is put in eval mode and its
                parameters are frozen; a supplied instance is modified in place,
                as by GPUResidentMolecularDynamics and the distributed evaluator.
            stress_unit (float): the conversion factor to convert GPa(MatRIS default) to eV/A^3.
            **kwargs: Passed to the Calculator parent class.
        """
        super().__init__(**kwargs)
        self.task=task
        self.device = device
        self.handle_isolated_atoms = self._inference_config.handle_isolated_atoms
        enable_checkpoint = self._inference_config.checkpoint_enabled
        self._adapter = load_adapter(
            model,
            model_path=model_path,
            device=self.device,
            enable_compile=enable_compile,
            enable_checkpoint=enable_checkpoint,
            optimization_profile=self._inference_config.optimization_profile,
        )
        use_adapter(self, model)  # an adapter loaded by name is closed with its last user (also on a failure below)
        # The model's declared capacity step where the configuration keeps the default one.
        from fastmd.engine.capacity_policy import declared_capacity
        self._inference_config = declared_capacity(self._adapter, self._inference_config)
        self.model = self._adapter.module
        if self.model is model or self._adapter is model:  # supplied, not loaded
            if model_path is not None:
                raise ValueError("Pass either model=MatRIS(...) or model_path, not both")
            self.model = self.model.to(self.device)
            apply_checkpoint(self.model, enable_checkpoint)
        # Fails here, not at the first call, for a compiled/checkpointed model.
        saved_recompute.check_model(
            self.model, saved_recompute.explicit_mode(self._inference_config.saved_tensor_recompute))
        saved_recompute.note_entry_point(self.model)
        # Inference: eval mode (Dropout is the identity, BatchNorm uses its running
        # statistics, as in every other entry point) and frozen params: forces/stress
        # are autograd w.r.t. coords/strains, never params. Freezing params avoids
        # computing unused parameter grads in the backward (cuts backward work and
        # enables the dx-only fast path of fused LN+act).
        self.model.eval()
        for _p in self.model.parameters():
            _p.requires_grad_(False)
        # A model without triplets reads edges only: its graphs are the GPU
        # graph builder's graph batches (edges-only mode), not MatRIS's converter's.
        self._edges_only = self.adapter.topology_spec().triplet_cutoff is None
        # None: the model differentiates itself (adapter.evaluate); else the engine-owned derivative
        # (stress in eV/A^3; a float64 model reads float64 graph batches).
        self._derivative = engine_derivative(self._adapter)
        self._float64 = self._derivative is not None and self.adapter.topology_spec().model_dtype == "float64"
        # A model with a line graph that the engine differentiates reads the GPU graph builder's graphs
        # as graph batches with positions (line_graph_batch), not MatRIS's converter's.
        self._line_graph_batches = self._derivative is not None and not self._edges_only
        if self._edges_only or self._line_graph_batches:
            from fastmd.graph.gpu_graph_builder import op_available
            if not (op_available and torch.cuda.is_available() and torch.device(self.device).type == "cuda"):
                from fastmd.graph.neighbor_list_backend import UNAVAILABLE_REASON
                kind = "reads edges only" if self._edges_only else "has a line graph the engine differentiates"
                raise ValueError(
                    f"MLIPCalculator: a model that {kind} needs a CUDA device and the GPU "
                    f"graph builder (nvalchemiops); got device {self.device!r}"
                    + ("" if op_available else f" ({UNAVAILABLE_REASON})")
                )
            self._gpu_graph = True
            if self._edges_only and self.adapter.topology_spec().max_neighbors is not None:
                # A model built for at most max_neighbors neighbours per atom: every build (rebuilt or the skin
                # path's) folds its largest per-atom edge count into this device maximum, which each call
                # checks (neighbor_cap_summary).
                self._neighbor_count_max = torch.zeros(1, dtype=torch.int64, device=self.device)
        else:
            # GPU graphs retain isolated nodes and mask their learned readouts on
            # device, avoiding the legacy per-step pymatgen neighbor-list scan.
            self._gpu_graph = (
                getattr(self.model.graph_converter, "algorithm", "legacy") == "gpu"
                and torch.cuda.is_available()
            )

        # execution="auto" (the default): model_graph where the replay applies
        # (GPU graph converter, not compiled; CUDA and no checkpoint were
        # checked by configure_entrypoint). The first call of a system (atom
        # count) runs eager and measures it; a repeat call switches to
        # model_graph if that fits the device (_auto_decide), so a single-point
        # loop over many structures never captures.
        self._auto = None
        # Why execution auto keeps every call eager (None: it decides per system, or no auto).
        self._auto_static = "CPU or checkpoint (no CUDA graph)" if self._execution_auto else None
        if self._execution_auto and self._inference_config.execution == "model_graph":
            # (Another model's compiled modules count unless its contract declares them capture-safe.)
            compiled = enable_compile or (self.adapter.compiled and not (
                negotiated(self.adapter) and can_capture(self.adapter)))
            if self._gpu_graph and not compiled and self._inference_config.capacity.warmup >= 1:
                from collections import deque
                self._auto = {"n": None, "rows": None, "eager_peak": None, "mode": "eager", "reason": None,
                              "decisions": deque(maxlen=64), "decision_count": 0}
            self._auto_static = (None if self._auto is not None else "no GPU graph converter" if not self._gpu_graph
                                 else "compiled model" if compiled else "capacity.warmup=0 (a capture needs a warmup)")
            self._inference_config = replace(self._inference_config, execution="eager")
        self._engine_plan = None  # another model's negotiated plan: (adapter, EnginePlanRecord)
        self._initial_plan = None  # that plan at construction (a new system's first call)
        if negotiated(self.adapter):
            # Another model: its negotiated plan decides an explicit execution where the model allows it
            # (auto: at a system's first repeat call, _auto_decide) and the forwards' precision.
            self._negotiate_plan()

        # Optional bucketed CUDA-graph replay of the (sync-free) forward+backward.
        self._graph_runner = None
        if (
            self._inference_config.execution == "model_graph"
            and self._gpu_graph
            and torch.cuda.is_available()
        ):
            from fastmd.execution.cuda_graph import BucketedGraphRunner
            capacity = self._inference_config.capacity
            self._graph_runner = BucketedGraphRunner(
                self.adapter, task=self.task,
                u_step=capacity.u_step, t_step=capacity.t_step,
                n_dummy=capacity.n_dummy, min_pad_u=capacity.min_pad_u,
                warmup=capacity.warmup,
                enable_model_fusions=self._inference_config.optimization_profile != "generic",
                config=self._inference_config,
            )

        # Step C7: skin-epoch topology reuse (InferenceConfig.topology_reuse).
        # The runner of the current system is created at its first eligible
        # call; ``_skin_off_reason`` records why a call used the per-call
        # rebuild instead (None when it did not).
        self._skin = None
        self._skin_cell = None      # cell of the last periodic call
        self._skin_n = None         # its atom count
        self._skin_hold = 0         # rebuild calls left after a cell change
        self._skin_episodes = 0     # cell-change episodes so far
        self._skin_unavailable = None  # (atom count, cell) the skin path could not evaluate
        self._skin_static = self._skin_static_reason(enable_compile)
        self._skin_off_reason = self._skin_static
        self._skin_last = "off"
        self._skin_fallbacks = 0

        from fastmd.engine.energy_derivative import stress_unit_scale

        self.stress_unit = stress_unit_scale(self.adapter)
        key = ["atoms_per_graph", "ref_energy"]
        for t in task:
            key.append(t)
        self.key = set(key)

    adapter = property(current_adapter)
    _closed = False
    # A model with a neighbour cap (TopologySpec.max_neighbors): the largest per-atom edge count of the builds, a
    # [1] int64 device maximum (None without a cap), and whether its excess was warned about.
    _neighbor_count_max = None
    _neighbor_cap_warned = False
    _checked_numbers = None  # the atomic numbers the model's input check last passed (_check_new_numbers)

    def close(self) -> None:
        """Release this calculator's captured graphs and its model adapter: one loaded by name (here, or by
        another entry point) is closed when its last user is (its upstream patches are reverted); an adapter
        or a model the caller passed in stays open. Idempotent; a later calculation raises. Context exit
        closes too, and a dropped calculator releases its adapter when it is finalised."""
        self._closed = True
        try:
            self._drop_skin()
            runner, self._graph_runner = self._graph_runner, None
            if runner is not None and runner.cache:
                runner._release_pool()
        finally:
            release_adapter(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

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

    def _skin_static_reason(self, enable_compile) -> str | None:
        """Why topology reuse never applies to this calculator (None: it may)."""
        mode = self._inference_config.topology_reuse
        if mode == "off":
            return "topology_reuse='off'"
        contract = self._engine_plan[1].contract if self._engine_plan is not None else None
        if contract is not None and not (contract.exact_cutoff or contract.exact_at_cutoff):
            return "no exact cutoff declared (a stale or kNN list is never reused)"
        if not self._gpu_graph:
            return "no GPU graph converter"
        if self._float64:
            return "a float64 model (the skin runner reads float32 coordinates)"
        parameter = next(iter(self.model.parameters()), None)
        if not isinstance(parameter, torch.Tensor) or parameter.device.type != "cuda":
            return "the model is not on a CUDA device"
        from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
        from fastmd.graph.fixed_capacity_builder import _fixed_builder_unavailable
        if not _fixed_builder_available:
            return f"fixed-capacity builder unavailable: {_fixed_builder_unavailable}"
        if mode == "auto" and enable_compile:
            return "auto: enable_compile"
        if self._inference_config.capacity.candidate_skin <= 0:
            return "capacity.candidate_skin is 0"
        return None

    def neighbor_cap_summary(self) -> dict | None:
        """For a model that keeps at most ``max_neighbors`` neighbours per atom (its own graph drops the
        farthest ones): the cap, the largest number of neighbours within the cutoff one atom had in the
        calls so far (read from the device) and whether it exceeded the cap, in which case the calculator
        evaluated every pair (the model's intent, not its own truncated graph) and this warns once (each
        call checks). None without a cap."""
        if self._neighbor_count_max is None:
            return None
        topology = self.adapter.topology_spec()
        cap, largest = topology.max_neighbors, int(self._neighbor_count_max.item())
        if largest > cap and not self._neighbor_cap_warned:
            self._neighbor_cap_warned = True
            warnings.warn(f"model {self.adapter.name!r} keeps at most {cap} neighbours per atom in its own graph; "
                          f"an atom had {largest} within the {topology.cutoff:g} A cutoff, and the calculator "
                          "evaluated every pair", RuntimeWarning, stacklevel=2)
        return {"max_neighbors": cap, "largest": largest, "exceeded": largest > cap}

    def topology_reuse_summary(self) -> dict:
        """Topology reuse: requested mode, what the last call used and why, runner counters."""
        return {
            "requested": self._inference_config.topology_reuse,
            "last_call": self._skin_last,
            "reason": self._skin_off_reason,
            "fallbacks": self._skin_fallbacks,
            "runner": None if self._skin is None else self._skin.stats(),
        }

    def _drop_skin(self) -> None:
        if self._skin is not None:
            self._skin.close()
            self._skin = None

    def _skin_runner(self, atoms):
        """The skin runner for this call's system, or None (per-call rebuild).

        The runner is kept per (atom count, cell); new atomic numbers of the
        same atoms are set in place. A changed cell of the same atom count
        (cell relaxation, strain scans) drops the runner and rebuilds per call
        until the cell has been unchanged for STABLE_CELL_CALLS calls
        (model_graph: 4x per cell-change episode up to STABLE_CELL_CALLS_MAX,
        so a cell relaxation or a loop over cells of one atom count stays on
        the rebuild path instead of recapturing every few calls).
        """
        from fastmd.execution.skin_topology import (SkinTopologyRunner, STABLE_CELL_CALLS, STABLE_CELL_CALLS_EAGER,
                                               STABLE_CELL_CALLS_MAX)
        if self._skin_static is not None:
            self._skin_off_reason = self._skin_static
            return None
        if self._inference_config.topology_reuse == "auto" and self._graph_runner is None:
            # Eager gains 1-2% at a fixed demand and re-keys its padded bucket
            # as the demand drifts (heating); "skin" selects it explicitly.
            self._skin_off_reason = "auto: eager execution rebuilds per call"
            return None
        pbc = atoms.get_pbc()
        if not pbc.all():
            self._skin_off_reason = "not periodic in all directions"
            return None
        cell = np.array(atoms.cell.array, dtype=np.float64)
        n = len(atoms)
        if n != self._skin_n:
            self._drop_skin()
            # Eager: a new system's first call rebuilds (single-point loops
            # never set a runner up); model_graph captures on it either way.
            self._skin_hold = 0 if self._graph_runner is not None else 1
        elif not np.array_equal(cell, self._skin_cell):
            self._drop_skin()
            self._skin_episodes += 1
            if self._graph_runner is None:
                self._skin_hold = STABLE_CELL_CALLS_EAGER
            else:
                self._skin_hold = min(STABLE_CELL_CALLS << min(2 * (self._skin_episodes - 1), 8),
                                      STABLE_CELL_CALLS_MAX)
        self._skin_n, self._skin_cell = n, cell
        if self._skin_hold > 0:
            self._skin_hold -= 1
            self._skin_off_reason = "a new system or cell within the last calls"
            return None
        if self._skin_unavailable is not None and self._skin_unavailable[0] == n and np.array_equal(
                self._skin_unavailable[1], cell):
            self._skin_off_reason = "candidate list overflow for this system"
            return None
        numbers = atoms.get_atomic_numbers()
        if self._skin is None:
            if self._graph_runner is not None and self._graph_runner.cache:
                # One pool at a time: the rebuild path's buckets go.
                self._graph_runner._release_pool()
                self._graph_runner._tight.clear()
                self._graph_runner._envelope = None
            self._skin = SkinTopologyRunner(
                self.adapter, task=self.task, atomic_numbers=numbers, cell=cell, pbc=pbc,
                skin=self._inference_config.capacity.candidate_skin,
                capture=self._graph_runner is not None,
                check_isolated_atoms=not self.handle_isolated_atoms or self.adapter.isolated_atom_policy() == "error",
                handle_isolated_atoms=self.handle_isolated_atoms,
                device=self._model_device(),
                enable_model_fusions=self._inference_config.optimization_profile != "generic",
                composition=atoms.get_chemical_formula(),
                config=self._inference_config,
                **({"max_neighbor_count": self._neighbor_count_max} if self._neighbor_count_max is not None else {}),
            )
        else:
            self._skin.set_atomic_numbers(numbers)
        self._skin_off_reason = None
        return self._skin

    def _skin_predict(self, skin, atoms):
        """The skin path's prediction, or None after a fallback to the rebuild."""
        from fastmd.execution.skin_topology import SkinTopologyUnavailable
        try:
            out = skin.evaluate(atoms.get_positions(), atoms=atoms)
        except SkinTopologyUnavailable as error:
            self._skin_unavailable = (self._skin_n, self._skin_cell)
            self._skin_off_reason = f"skin topology unavailable: {error}"
            self._skin_fallbacks += 1
            self._drop_skin()
            return None
        return {key: out[key] for key in self.key if key in out}

    def _get_isolated_atom_indices(self, structure) -> np.ndarray:
        center_index, _, _, _ = structure.get_neighbor_list(
            r=self.model.graph_converter.atom_graph_cutoff,
            sites=structure.sites,
            numerical_tol=1e-8,
        )
        return np.setdiff1d(
            np.arange(len(structure), dtype=int),
            np.unique(center_index),
            assume_unique=True,
        )

    def _get_atomic_reference_energy(self, atomic_numbers: np.ndarray) -> float:
        if len(atomic_numbers) == 0:
            return 0.0
        if self.adapter.isolated_atom_policy() == "error":
            raise ValueError(
                "Cannot add isolated atomic energy because this model has no "
                "reference_energy table."
            )

        reference_energies = self.adapter.reference_energies()
        atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        max_num_elements = reference_energies.shape[1]
        if np.any(atomic_numbers < 1) or np.any(atomic_numbers > max_num_elements):
            raise ValueError(
                f"Atomic numbers must be in [1, {max_num_elements}] to use "
                "reference atomic energies."
            )

        ref_weights = reference_energies.detach().cpu().numpy()[0]
        return float(ref_weights[atomic_numbers - 1].sum())

    def _as_numpy_prediction(self, value):
        if isinstance(value, (list, tuple)):
            value = value[0]
        elif hasattr(value, "ndim") and value.ndim > 0:
            value = value[0]

        if hasattr(value, "cpu"):
            return value.cpu().detach().numpy()
        return np.array(value)

    def _model_device(self):
        """The device of the model's parameters (the calculator's device for a module without any)."""
        parameter = next(iter(self.model.parameters()), None)
        return torch.device(self.device) if parameter is None else parameter.device

    def _check_new_numbers(self, graph, atoms) -> None:
        """The model's own input check (its element range, in ``adapter.prepare``) of atomic numbers that
        a model-graph bucket takes in place (its input was prepared once, at its capture): once per new set
        of numbers, on this call's graph (the eager path prepares every call's graph)."""
        numbers = atoms.get_atomic_numbers()
        if self._checked_numbers is not None and np.array_equal(numbers, self._checked_numbers):
            return
        self.adapter.prepare(graph if self._edges_only else line_graph_batch(graph))
        self._checked_numbers = numbers.copy()

    def _per_call_graph(self, structure, atoms):
        """This call's ragged graph (the rebuild path): MatRIS's graph converter's, or for a model
        that reads edges only the GPU graph builder's ``GraphBatch`` of ``atoms`` (such a model
        always takes the GPU path, which passes ``atoms``)."""
        check_isolated_atoms = not self.handle_isolated_atoms or self.adapter.isolated_atom_policy() == "error"
        if self._edges_only:
            from fastmd.graph.gpu_graph_builder import atoms_to_edges_gpu
            batch = atoms_to_edges_gpu(atoms, self.adapter.topology_spec().cutoff, self.device,
                                       check_isolated_atoms=check_isolated_atoms,
                                       **({"positions_dtype": torch.float64} if self._float64 else {}))
            if self._neighbor_count_max is not None:
                centres = batch.edge_index[:, 0].long()
                counts = torch.zeros(batch.n_real, dtype=torch.int64, device=centres.device)
                counts.index_add_(0, centres, torch.ones_like(centres))
                torch.maximum(self._neighbor_count_max, counts.max().reshape(1), out=self._neighbor_count_max)
            return batch
        if self._line_graph_batches:
            from fastmd.graph.gpu_graph_builder import atoms_to_graph_gpu
            topology = self.adapter.topology_spec()
            return atoms_to_graph_gpu(atoms, topology.cutoff, topology.triplet_cutoff, self.device,
                                      check_isolated_atoms=check_isolated_atoms)
        return self.model.graph_converter(
            structure, atoms=atoms, check_isolated_atoms=check_isolated_atoms,
        ).to(self.device)

    def _new_graph_runner(self):
        from fastmd.execution.cuda_graph import BucketedGraphRunner
        capacity = self._inference_config.capacity
        return BucketedGraphRunner(
            self.adapter, task=self.task,
            u_step=capacity.u_step, t_step=capacity.t_step,
            n_dummy=capacity.n_dummy, min_pad_u=capacity.min_pad_u,
            warmup=capacity.warmup,
            enable_model_fusions=self._inference_config.optimization_profile != "generic",
            config=self._inference_config,
        )

    def _set_execution(self, mode: str, reason: str) -> None:
        """execution="auto": record a decision and switch between eager and
        model_graph if it changes the mode (the old mode's graphs and pools
        released first)."""
        import gc
        auto = self._auto
        auto["mode"], auto["reason"] = mode, reason
        auto["decisions"].append({"n": auto["n"], "mode": mode, "reason": reason})
        auto["decision_count"] += 1
        if (mode == "model_graph") == (self._graph_runner is not None):
            return
        if mode == "eager":
            self._drop_skin()
            runner, self._graph_runner = self._graph_runner, None
            if runner is not None and runner.cache:
                runner._release_pool()
            del runner
        gc.collect()
        # Either way the old mode's cached blocks go back to the device (the
        # eager call's would otherwise stay reserved next to the graph pool).
        torch.cuda.empty_cache()
        # Before the runner: its captures decide recompute with this config.
        self._inference_config = replace(self._inference_config, execution=mode)
        if mode == "model_graph":
            self._graph_runner = self._new_graph_runner()

    def _auto_decide(self, atoms) -> tuple[str, str]:
        """model_graph or eager for the measured system (its eager call's rows):
        capacity_policy.graph_execution, the rule GPU-resident MD shares (K6's
        model_graph prediction for this model, at the recompute level eager takes,
        must fit AUTO_THRESHOLD of the budget; a model outside every measured
        envelope, or a configuration without a prediction, runs eager). Admission off:
        model_graph (the OOM safety net remains). Another model: its negotiated plan
        (``_negotiated_decide``, for ``atoms``).
        """
        if negotiated(self.adapter):
            return self._negotiated_decide(atoms)
        from fastmd.engine.capacity_policy import graph_execution
        graph, reason = graph_execution(self.model, self._inference_config, self._auto["rows"] or 0,
                                        self._model_device(), admission_off_trusts_any_model=True)
        return ("model_graph" if graph else "eager"), reason

    def _negotiate_plan(self) -> None:
        """Another model's plan (``fastmd.adapters.negotiate``) at construction, kept for
        ``plan_record()``: an explicit execution as requested where the model allows it, the forwards'
        precision (IEEE FP32 unless TF32 is asked for); under execution auto eager until a system's
        first repeat call decides (``_negotiated_decide``)."""
        from fastmd.adapters.negotiate import can_capture
        from fastmd.adapters.negotiate import negotiate
        from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
        config = self._inference_config
        requested = replace(config, execution="auto") if self._execution_auto else config
        plan = negotiate(self.adapter, requested, entry="calculator", device=self._model_device())
        execution = plan.decision("execution")
        reason = None
        if (self._auto is not None and (self._edges_only or self._line_graph_batches) and _fixed_builder_available
                and can_capture(self.adapter)):
            reason = ("auto: a system's first call runs eager; its first repeat call decides, with a memory model "
                      "measured on the system")
        elif self._execution_auto and self._auto is None:
            reason = f"auto: {self._auto_static}: eager"
        elif not self._execution_auto and execution.chosen != config.execution:
            warnings.warn(f"MLIPCalculator: execution={config.execution!r} for model {self.adapter.name!r}: "
                          f"{execution.reason}", ConfigurationWarning, stacklevel=4)
        if reason is not None:
            plan.decisions = [replace(decision, reason=reason) if decision.id == "execution" else decision
                              for decision in plan.decisions]
        self._engine_plan = self._initial_plan = (self.adapter, plan)
        self._inference_config = replace(config, matmul_precision=plan.matmul_precision,
                                         **({} if self._execution_auto else {"execution": execution.chosen}))

    def _negotiated_decide(self, atoms) -> tuple[str, str]:
        """execution auto for another model at a system's first repeat call: the plan's execution decided
        again (``renegotiate_execution``) at the capacity its model graph would capture (the skin path's
        bucket, which adds the minimum padding), with a memory model measured (at the plan's precision) on
        the model's own evaluation of this system (edges only, or with the line graph of a model the engine
        differentiates: also at the bucket's triplet capacity; K6 has no calibration for it); the plan is
        kept for ``plan_record()``."""
        from fastmd.adapters.memory import MeasuredMemoryModel
        from fastmd.adapters.negotiate import can_capture
        from fastmd.adapters.negotiate import negotiate
        from fastmd.adapters.negotiate import renegotiate_execution
        from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
        from fastmd.execution._memory_probe import edges_only_memory_runs
        from fastmd.execution._memory_probe import line_graph_memory_runs
        from fastmd.execution._memory_probe import measured_memory
        from fastmd.engine.capacity_policy import CapturePolicy
        config = replace(self._inference_config, execution="auto")
        device = self._model_device()
        plan = (self._engine_plan[1] if self._engine_plan is not None and self._engine_plan[0] is self.adapter
                else negotiate(self.adapter, config, entry="calculator", device=device))
        capacity = memory = None
        pbc = tuple(bool(axis) for axis in atoms.get_pbc())
        if self._edges_only and _fixed_builder_available and can_capture(self.adapter):
            with self._inference_config.scope():
                batch = self._per_call_graph(None, atoms)
                u = int(batch.edge_index.shape[0]) // 2
                capacity = {"edges": 2 * CapturePolicy.from_config(config.capacity).bucket_key(
                    u + config.capacity.min_pad_u, 0)[0]}
                n = batch.n_real
                runs = edges_only_memory_runs(
                    self.adapter, cell=batch.cell, atomic_numbers=batch.species[:n],
                    pbc=torch.tensor(pbc, device=device), positions=batch.positions[:n], n_real=n,
                    device=device, task=self.task, n_dummy=config.capacity.n_dummy,
                    candidate_skin=config.capacity.candidate_skin if self._skin_static is None else 0.0)
                del batch
                memory = measured_memory(MeasuredMemoryModel, runs, capacity["edges"], device)
        elif self._line_graph_batches and _fixed_builder_available and can_capture(self.adapter):
            with self._inference_config.scope():
                batch = line_graph_batch(self._per_call_graph(None, atoms))
                u, t = int(batch.pair_u2d.shape[0]), int(batch.triplets.shape[0])
                key = CapturePolicy.from_config(config.capacity).bucket_key(u + config.capacity.min_pad_u,
                                                                             t + config.capacity.min_pad_t)
                capacity = {"edges": 2 * key[0], "triplets": key[1]}
                runs = line_graph_memory_runs(
                    self.adapter, cell=batch.cell, atomic_numbers=batch.species, pbc=torch.tensor(pbc, device=device),
                    positions=batch.positions, n_real=batch.n_real, device=device, task=self.task,
                    n_dummy=config.capacity.n_dummy,
                    candidate_skin=config.capacity.candidate_skin if self._skin_static is None else 0.0)
                del batch
                memory = measured_memory(MeasuredMemoryModel, runs, capacity["edges"], device, capacity["triplets"])
        plan = renegotiate_execution(plan, self.adapter, config, entry="calculator", device=device, pbc=pbc,
                                     capacity=capacity, memory=memory)
        self._engine_plan = (self.adapter, plan)
        execution = plan.decision("execution")
        return execution.chosen, execution.reason

    def _auto_before_call(self, atoms) -> None:
        """execution="auto": a new system's call runs eager (and is measured:
        it resets torch's CUDA peak-memory statistics); its first repeat after
        a successful call decides (_auto_decide)."""
        auto = self._auto
        n = len(atoms)
        if n != auto["n"]:
            auto.update(n=n, rows=None, eager_peak=None)
            if self._initial_plan is not None:
                self._engine_plan = self._initial_plan  # this system's first call: eager until it repeats
            if self._graph_runner is not None or auto["mode"] != "eager":
                self._set_execution("eager", f"new system ({n} atoms): eager first call")
            auto["mode"], auto["measure"] = "eager", True
            auto.pop("decided", None)
            device = self._model_device()
            auto["baseline"] = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        elif not auto.get("decided") and auto["rows"] is not None and auto["eager_peak"] is not None:
            # (Decided once the decision returns: a call it rejects, e.g. an element outside the model, leaves
            # the next repeat call to decide.)
            mode, reason = self._auto_decide(atoms)
            auto["decided"] = True
            self._set_execution(mode, reason)

    def _auto_after_call(self) -> None:
        auto = self._auto
        if auto.pop("measure", False):
            device = self._model_device()
            auto["eager_peak"] = torch.cuda.max_memory_allocated(device) - auto["baseline"]

    def _auto_capture(self, run):
        """``run()`` on the graph path; under execution="auto" a CUDA OOM or a
        failed capture switches this system to eager (a RuntimeWarning).
        Returns (result, whether it ran)."""
        if self._auto is None:
            return run(), True
        message = None
        try:
            return run(), True
        except torch.OutOfMemoryError as error:
            message = str(error)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() and "capture" not in str(error).lower():
                raise
            message = str(error)
        # Outside the except block: its traceback would pin the graph pool.
        reason = f"model_graph failed ({message.splitlines()[0][:160] if message else ''}); eager"
        self._set_execution("eager", reason)
        self._auto["decided"] = True
        warnings.warn(f"execution auto: {reason}", RuntimeWarning, stacklevel=4)
        return None, False

    def _predict_structure(self, structure, atoms=None):
        if self._auto is None or atoms is None:
            return self._predict_structure_scoped(structure, atoms=atoms)
        self._auto_before_call(atoms)
        if self._graph_runner is None:
            result = self._predict_structure_scoped(structure, atoms=atoms)
            self._auto_after_call()
            return result
        result, ran = self._auto_capture(lambda: self._predict_structure_scoped(structure, atoms=atoms))
        return result if ran else self._predict_structure_scoped(structure, atoms=atoms)

    @scoped_call
    def _predict_structure_scoped(self, structure, atoms=None):
        # atoms!=None takes the GPU fast path (build graph straight from ASE atoms,
        # skipping the ASE<->pymatgen round trip).
        skin = self._skin_runner(atoms) if atoms is not None else None
        result = self._skin_predict(skin, atoms) if skin is not None else None
        self._skin_last = "skin" if result is not None else "off"
        if result is not None:
            return result
        if self._skin is not None:
            # This call rebuilds: one pool at a time.
            self._drop_skin()
        if self._graph_runner is not None and atoms is not None:
            # Bucketed CUDA-graph replay of forward+backward (padded, dummy-sink
            # masked -> exact on the n_real real atoms; edges only: padded by the
            # adapter's padding rule).
            g = self._per_call_graph(None, atoms)
            if self._edges_only or self._line_graph_batches:
                self._check_new_numbers(g, atoms)
            out, n_real = self._graph_runner.run(g)
            result = {}
            if "e" in self.key:
                result["e"] = float(out["e"][0])
            if "ref_energy" in self.key and "ref_energy" in out:
                re = out["ref_energy"]
                result["ref_energy"] = float(re[0]) if torch.is_tensor(re) else float(re)
            if "f" in self.key and "f" in out:
                result["f"] = out["f"][0][:n_real].detach().cpu().numpy()
            if "s" in self.key and "s" in out:
                result["s"] = out["s"][0].detach().cpu().numpy()
            if "m" in self.key and "m" in out:
                result["m"] = out["m"][0][:n_real].detach().cpu().numpy()
            return result

        graph = self._per_call_graph(structure, atoms)
        if self._edges_only:
            graphs = [self.adapter.prepare(graph)]  # the model's own input over this call's graph batch
        elif self._line_graph_batches:
            graph = line_graph_batch(graph)  # the graph batch over this call's graph, with positions
            graphs = [self.adapter.prepare(graph)]
        else:
            graphs = [graph] if isinstance(graph, RadiusGraph) else graph
        if self._auto is not None:
            # Host-known shapes (a model that reads edges only has no triplet rows).
            self._auto["rows"] = (0 if self._edges_only else int(graph.triplets.shape[0]) if self._line_graph_batches
                                  else int(sum(g.line_graph.shape[0] for g in graphs)))

        if self._derivative is None:
            model_prediction = self.adapter.evaluate(
                graphs,
                task = self.task,
                is_training = False,
                handle_isolated_atoms=self.handle_isolated_atoms,
            )
        else:
            model_prediction = self._derivative(graph, graphs[0], task=self.task)

        return {
            key: self._as_numpy_prediction(model_prediction[key])
            for key in self.key & set(model_prediction.keys())
        }
     
    @scoped_call
    def precapture(self, atoms, n_steps: int = 120, temperature: float = 300.0,
                   timestep: float = 1.0) -> None:
        """Warm the CUDA-graph bucket cache up front by driving a throwaway NVT
        trajectory on a COPY of `atoms`. This covers the one-time thermalization
        drift plus the equilibrium fluctuation of the edge/triplet counts, so the
        production run replays with no mid-run capture spikes. No-op unless
        execution='model_graph' is active, or execution auto chose it for these atoms."""
        if self._graph_runner is None and self._auto is None:
            return
        if (self._auto is not None and self._auto["n"] == len(atoms) and self._auto.get("decided")
                and self._graph_runner is None):
            return  # execution auto already keeps these atoms eager
        from ase.md.nvtberendsen import NVTBerendsen
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        probe = atoms.copy()
        MaxwellBoltzmannDistribution(probe, temperature_K=temperature, force_temp=True)
        probe.calc = self
        dyn = NVTBerendsen(probe, timestep=timestep * units.fs,
                           temperature_K=temperature, taut=100 * units.fs)
        dyn.run(n_steps)
        if self._skin is not None and self._skin_last == "skin":
            # The probe ran on the skin path: its buckets around the final demand.
            skin = self._skin
            if self._auto_capture(lambda: skin.precapture(proactive=True))[1]:
                print(f"CUDA-graph precapture: {skin.captures} graphs / "
                      f"{len(skin.cache)} buckets (skin topology)")
            return
        if self._graph_runner is None:
            return  # execution auto kept eager for these atoms
        # proactive ±1 neighbor buckets around the final state
        g = self._per_call_graph(None, probe)
        runner = self._graph_runner
        if self._auto_capture(lambda: runner.precapture(g, proactive=True))[1]:
            print(f"CUDA-graph precapture: {runner.captures} graphs / {len(runner.cache)} buckets")

    @scoped_call
    def calculate(
        self,
        atoms: Atoms,
        properties: list,
        system_changes: list,
    ) -> None:
        
        if self._closed:
            raise RuntimeError("MLIPCalculator: the calculator is closed")
        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )
        require_finite_geometry(atoms.positions, atoms.cell.array, "MLIPCalculator")

        pbc = atoms.get_pbc()
        if (not pbc[0]) or (not pbc[1]) or (not pbc[2]):
            if self._edges_only or self._line_graph_batches:
                # The extended cell below wraps the coordinates of a non-periodic
                # direction, whose edges have no image across the cell face.
                kind = "reads edges only" if self._edges_only else "has a line graph the engine differentiates"
                raise ValueError(f"MLIPCalculator: a model that {kind} needs a cell periodic in "
                                 f"all three directions; got pbc={pbc.tolist()}")
            pos = atoms.get_positions()
            cell = np.array(atoms.get_cell())
            
            pbc_x = pbc[0]
            pbc_y = pbc[1]
            pbc_z = pbc[2]
            identity = np.identity(3, dtype=float)
            max_positions = np.max(np.absolute(pos)) + 1
            
            topology = self.adapter.topology_spec()
            cutoff = topology.cutoff
            expand = max(5, topology.num_interactions)

            # Extend cell in non-periodic directions
            if not pbc_x:
                cell[0, :] = max_positions * expand * cutoff * identity[0, :]
            if not pbc_y:
                cell[1, :] = max_positions * expand * cutoff * identity[1, :]
            if not pbc_z:
                cell[2, :] = max_positions * expand * cutoff * identity[2, :]
            
            # update
            atoms.set_cell(cell, scale_atoms=False)
        
        total_atoms = len(atoms)
        active_indices = np.arange(total_atoms, dtype=int)
        isolated_indices = np.array([], dtype=int)
        isolated_ref_energy = 0.0

        use_gpu_fast_path = self._gpu_graph
        if use_gpu_fast_path:
            # Fast path: build the graph straight from ASE atoms (no ASE->pymatgen
            # ->ASE round trip). Isolated nodes stay in place; the model masks
            # learned readouts on device and retains their reference energies.
            # Without a reference table the converter rejects isolated atoms
            # strictly; connected systems still use the requested CUDA replay.
            structure = None
            pred_structure = None
            pred_atoms = atoms
        else:
            structure = AseAtomsAdaptor.get_structure(atoms)
            pred_structure = structure
            pred_atoms = None
            # "model": the model evaluates isolated atoms itself (no reference-energy substitution).
            if self.handle_isolated_atoms and self.adapter.isolated_atom_policy() != "model":
                isolated_indices = self._get_isolated_atom_indices(structure)
                if len(isolated_indices) != 0:
                    active_indices = np.setdiff1d(
                        active_indices, isolated_indices, assume_unique=True
                    )
                    isolated_ref_energy = self._get_atomic_reference_energy(
                        atoms.get_atomic_numbers()[isolated_indices]
                    )
                    pred_structure = (
                        AseAtomsAdaptor.get_structure(atoms[active_indices])
                        if len(active_indices) != 0
                        else None
                    )

        model_predictions = {}
        forces = None
        stress = None
        magmoms = None
        ref_energy = isolated_ref_energy
        energy = isolated_ref_energy

        do_predict = (pred_structure is not None) or (pred_atoms is not None)
        if do_predict:
            model_predictions = self._predict_structure(pred_structure, atoms=pred_atoms)

            n_atoms = len(active_indices)
            energy_of = self.adapter if self._derivative is None else self._derivative
            if "ref_energy" in model_predictions:  # a model without reference energies may report none
                ref_energy += float(energy_of.extensive_energy(model_predictions["ref_energy"], n_atoms))
            energy += float(energy_of.extensive_energy(model_predictions["e"], n_atoms))
            forces = model_predictions.get("f", None)
            stress = model_predictions.get("s", None)
            magmoms = model_predictions.get("m", None)

            if len(isolated_indices) != 0:
                if forces is not None:
                    full_forces = np.zeros((total_atoms, 3), dtype=forces.dtype)
                    full_forces[active_indices] = forces
                    forces = full_forces
                if magmoms is not None:
                    full_magmoms = np.zeros(total_atoms, dtype=magmoms.dtype)
                    full_magmoms[active_indices] = magmoms
                    magmoms = full_magmoms
        else:
            if "f" in self.task:
                forces = np.zeros((total_atoms, 3), dtype=float)
            if "s" in self.task:
                stress = np.zeros((3, 3), dtype=float)
            if "m" in self.task:
                magmoms = np.zeros(total_atoms, dtype=float)

        if stress is not None:
            stress = stress * self.stress_unit
        if self._neighbor_count_max is not None:
            self.neighbor_cap_summary()  # warns once if an atom exceeded the model's neighbour cap

        self.results.update(
            ref_energy=ref_energy,
            energy=energy,  # Total Energy
            forces=forces,
            # Stress: GPa -> eV/A^3
            stress=stress,
            magmoms=magmoms,
        )


class TrajectoryObserver:
    # ref: https://github.com/CederGroupHub/chgnet

    def __init__(self, atoms: Atoms) -> None:
        
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.magmoms: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class CrystalFeasObserver:
    # ref: https://github.com/CederGroupHub/chgnet

    def __init__(self, atoms: Atoms) -> None:
        """Create a CrystalFeasObserver from an Atoms object."""
        self.atoms = atoms
        self.crystal_feature_vectors: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record Atoms crystal feature vectors after an MD/relaxation step."""
        self.crystal_feature_vectors.append(self.atoms._calc.results["crystal_fea"])

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.crystal_feature_vectors)

    def save(self, filename: str) -> None:
        """Save the crystal feature vectors to filename in pickle format."""
        out_pkl = {"crystal_feas": self.crystal_feature_vectors}
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)
                      
