"""Capability negotiation: the engine plan of a model adapter, with a reason for every decision.

For one adapter under one resolved :class:`~fastmd.config.InferenceConfig` (the one knob: its
defaults are the fastest options; nothing new to set) the negotiation resolves the runtime
decisions, each from the fastest option down a fallback chain:

==================  ================================================  =========================================
decision            chain (fastest first)                             the fastest option needs
==================  ================================================  =========================================
``neighbor_list``   fixed_capacity_builder -> the rebuilding builder  topology reuse (its candidates); the
                    -> host                                           rebuilding builder: tensor_graph_builder
                                                                      (GPU MD), gpu_graph_builder (calculator)
``topology_reuse``  exact_skin_refilter -> rebuild                    a path that reuses (whole_step; the
                                                                      calculator's model_graph or "skin"), an
                                                                      exact cutoff, full PBC, a skin
``padding``         the adapter's rule -> none                        a padded path (a CUDA graph, or the
                                                                      calculator's skin path; eager: exact
                                                                      shapes)
``kernels.<site>``  the site's options in order -> its reference      install, import and JIT warm-up succeed
``derivative``      recorded: the contract's force mode, stress unit
``execution``       whole_step -> model_graph -> eager                sync-free, CUDA, a memory bound that fits
``precision``       ieee (fp64 for an FP64 model); tf32 only asked    (measured, not assumed)
``recompute``       the requested level -> off                        the adapter implements the level
``multi_gpu``       single_gpu (P=1); other sizes rejected            -
==================  ================================================  =========================================

FastMD rejects distributed entry points and any world_size other than 1.

A capability the adapter lacks resolves to the conservative option with a reason, never to an error.
The errors (:class:`NegotiationError`): world_size != 1,
``optimization_profile="legacy"`` for a model other than MatRIS, a kernel site whose reference
fails to install, runtime rules the runtime does not have, declarations that contradict each other
(a contract exact at the cutoff only with ``far_self_loops`` padding), and replicas of a model that does not
declare, or fails the check of, its systems' union (:func:`batch_systems`); a request the entry point itself
rejects (an execution mode it does not support, a CUDA graph without CUDA) raises its ``ValueError``.
"reduced" (TF32) is never a default: a model other than MatRIS runs IEEE FP32 unless the configuration
asks for TF32 (``matmul_precision``), and a kernel option of class "reduced" is never installed by default.
A timing trial chooses among options of one accuracy class only.

MatRIS is not negotiated: the runtime already decides for it (execution auto in GPU-resident MD and in
the calculator's capacity policy, K6 admission, saved-tensor recompute, topology reuse, precision); its
adapter declares that capability (:class:`~fastmd.adapters.protocol.RuntimePlanned`, the rules "matris").
:func:`entry_plan_record` reads those decisions from a constructed entry point, and :func:`dry_run`
(``python -m fastmd.explain``) mirrors the same rules without one; both only record. The whole-step
timing trial is not run for MatRIS (its auto rule stays as it is). MatRIS's record is made in its own
package (``fastmd.models.matris.plan_record``), which the model registry names for its rules
(``fastmd.models.plan_recorder``): the negotiation imports no model.

The :class:`EnginePlanRecord` is JSON: the model, its upstream package and fingerprint (weights,
static system inputs, kernel choices), the profile, every decision with its reason and evidence, the
patches, the accuracy class and the measured precision; it can link an ``OpPlanRecorder`` snapshot
(``fastmd.engine.recorder``: what executed, where the record says what was chosen).
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Callable, Iterator, Mapping, Sequence

from fastmd.adapters.patches import applied_pins, patch_report
from fastmd.adapters.protocol import (
    ACCURACY_CLASSES,
    CaptureContract,
    CaptureDeclaration,
    EdgeShiftInput,
    EnergyTap,
    KernelBackends,
    KernelChoice,
    ModelOwnedDerivative,
    NegotiationError,
    RecomputeParticipant,
    SYSTEM_BATCHING_KINDS,
    SystemBatching,
    SystemEnergyTap,
    SystemInputs,
    UpstreamPatches,
    check_profile,
    runtime_planned,
)

SCHEMA = "fastmd.engine_plan/1"
EXECUTION_CHAIN = ("whole_step", "model_graph", "eager")
# The entry point of a model with a declared shard on several GPUs (fastmd.distributed.ShardedEvaluator).
DISTRIBUTED = "distributed"
# The whole-step trial keeps whole_step at this speedup over model_graph (select_whole_step_cuda_graph's default).
WHOLE_STEP_MIN_SPEEDUP = 1.01
# An adapter that declares no contract: not capturable, no exact cutoff (rebuild every step).
UNDECLARED_CONTRACT = CaptureContract(sync_free=False, exact_cutoff=False)


@dataclass(frozen=True)
class Decision:
    """One runtime decision: the chosen option of its chain, why, and what was measured."""

    id: str
    chosen: str
    reason: str
    chain: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    # The accuracy class this decision gives the arithmetic, where it sets one (precision, kernels).
    accuracy_class: str | None = None

    def as_dict(self) -> dict:
        out = {"id": self.id, "chosen": self.chosen, "reason": self.reason}
        if self.chain:
            out["chain"] = list(self.chain)
        if self.evidence:
            out["evidence"] = _jsonable(self.evidence)
        if self.accuracy_class is not None:
            out["accuracy_class"] = self.accuracy_class
        return out


@dataclass
class EnginePlanRecord:
    """The negotiated (or, for MatRIS, recorded) engine plan of one model in one entry point."""

    model: str
    upstream: str | None
    fingerprint: str
    profile: str
    entry: str
    world_size: int
    decisions: list[Decision]
    contract: CaptureContract
    accuracy_class: str
    isolated_atoms: Mapping[str, Any]
    memory: Mapping[str, Any]
    precision: dict
    patches: dict = field(default_factory=lambda: {"applied": [], "declined": {}})
    # The FP32 GEMM precision the adapter's forwards run with (forward_scope); None: the process's.
    matmul_precision: str | None = None
    op_plan: dict | None = None
    # Distinct global precision changes made inside forward_scope (by upstream code) and undone there.
    precision_restores: list = field(default_factory=list)
    # False: the adapter declares no contract and ``contract`` holds the conservative defaults.
    contract_declared: bool = True
    # P>1 of the distributed entry point only: the decomposition, the exchange schedule, the per-rank sizes and
    # memory, the rank contract's hash, the stress method, the source pins (None: absent from the record)
    distributed: dict | None = None

    def decision(self, decision_id: str) -> Decision:
        for decision in self.decisions:
            if decision.id == decision_id:
                return decision
        raise KeyError(decision_id)

    def as_dict(self) -> dict:
        return _jsonable({
            "schema": SCHEMA, "model": self.model, "upstream": self.upstream, "fingerprint": self.fingerprint,
            "profile": self.profile, "entry": self.entry, "world_size": self.world_size,
            "decisions": [decision.as_dict() for decision in self.decisions], "patches": self.patches,
            "accuracy_class": self.accuracy_class, "contract": asdict(self.contract),
            "contract_declared": self.contract_declared, "isolated_atoms": self.isolated_atoms,
            "memory": self.memory, "precision": self.precision, "matmul_precision": self.matmul_precision,
            "precision_restores": self.precision_restores, "op_plan": self.op_plan,
            **({} if self.distributed is None else {"distributed": self.distributed}),
        })

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def __str__(self) -> str:
        lines = [f"engine plan: {self.model} ({self.upstream}), {self.entry}, profile {self.profile}, "
                 f"P={self.world_size}, accuracy class {self.accuracy_class}", f"  fingerprint {self.fingerprint}"]
        for decision in self.decisions:
            lines.append(f"  {decision.id:<16} {decision.chosen:<22} {decision.reason}")
        lines.append(f"  {'isolated_atoms':<16} {self.isolated_atoms.get('policy')!s:<22} "
                     f"{self.isolated_atoms.get('effect')}")
        lines.append(f"  {'memory':<16} {self.memory.get('calibration')!s:<22} {self.memory.get('note', '')}")
        if self.distributed is not None:
            block = self.distributed
            lines.append(f"  {'distributed':<16} {block.get('strategy')!s:<22} grid {block.get('grid')}, halo "
                         f"{block.get('halo_A')} Å" + (" (estimates)" if block.get("estimate") else ""))
        return "\n".join(lines)

    def link_op_plan(self, snapshot) -> "EnginePlanRecord":
        """A deep copy of this record linking an ``OpPlanRecorder`` snapshot (or the recorder) of this
        plan's runs: its schema, sha256 over the canonical JSON, kernel-policy fingerprints and entry
        counts per section."""
        from fastmd.engine.recorder import canonical_json
        if hasattr(snapshot, "snapshot"):
            snapshot = snapshot.snapshot()
        return replace(copy.deepcopy(self), op_plan={
            "schema": snapshot.get("schema"),
            "sha256": hashlib.sha256(canonical_json(snapshot).encode()).hexdigest(),
            "policy_fingerprints": list(snapshot.get("policy_fingerprints", ())),
            "sections": [{"label": section.get("label"), "entry_counts": section.get("entry_counts", {}),
                          "fallback_counts": section.get("fallback_counts", {})}
                         for section in snapshot.get("sections", ())],
        })

    @contextmanager
    def forward_scope(self) -> Iterator[None]:
        """Run an adapter forward under the plan's precision (``matmul_precision``), and undo any
        global FP32 GEMM precision change made inside (upstream code calling
        ``torch.set_float32_matmul_precision`` or setting ``allow_tf32``), recording each distinct one."""
        from fastmd.precision import matmul_precision_scope
        from fastmd.precision import restore_fp32_state
        with matmul_precision_scope(self.matmul_precision), restore_fp32_state(self.precision_restores):
            yield

    def check_precision(self, stage: str) -> dict:
        """Measure the cuBLAS FP32 GEMM math the adapter's forwards run with now (``stage``: e.g.
        "before_capture"), record it, and say whether it is still the negotiated one."""
        from fastmd.precision import fp32_gemm_mode
        from fastmd.precision import matmul_precision_scope
        with matmul_precision_scope(self.matmul_precision):
            mode = fp32_gemm_mode()
        expected = self.precision.get("cublas")
        # FP64 GEMMs, and a plan on the CPU, do not depend on the cuBLAS FP32 flags.
        unaffected = (expected is None or self.precision.get("chosen") == "fp64"
                      or self.precision.get("device") == "cpu")
        self.precision[stage] = {"cublas": mode, "as_negotiated": unaffected or mode == expected}
        return self.precision[stage]


# --- shared helpers ----------------------------------------------------------------------------------

def _finite(value):
    """``value`` with non-finite floats as None and sequences as lists (JSON without NaN/Infinity)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_finite(item) for item in value]
    return value


def _jsonable(value):
    return json.loads(json.dumps(_finite(value), default=str, sort_keys=True, allow_nan=False))


def loosest_class(*classes: str | None) -> str:
    """The least strict of accuracy classes (None ignored); "bitwise" without any."""
    present = [value for value in classes if value is not None]
    return max(present, key=ACCURACY_CLASSES.index) if present else ACCURACY_CLASSES[0]


def trial_candidates(options: Sequence[tuple[str, str]]) -> list[str]:
    """The options a timing trial may choose among (``options``: (name, class) pairs, the default
    first): those of the default's accuracy class, so a trial never changes the class of the
    arithmetic (and never turns a default into "reduced")."""
    if not options:
        return []
    first = options[0][1]
    return [name for name, cls in options if cls == first]


def contract_of(adapter) -> tuple[CaptureContract, bool]:
    """The adapter's capture contract and whether it declared one."""
    if isinstance(adapter, CaptureDeclaration):
        return adapter.capture_contract(), True
    return UNDECLARED_CONTRACT, False


def negotiated(adapter) -> bool:
    """Whether the entry points negotiate ``adapter``'s plan: every model but MatRIS (whose decisions
    the runtime makes and :func:`entry_plan_record` records: ``protocol.runtime_planned``)."""
    return not runtime_planned(adapter)


def measured_by_entry_points(adapter) -> bool:
    """Whether the entry points measure ``adapter``'s memory for execution auto on its own evaluation (where
    it may be captured): a model that reads edges only, or one with a line graph whose energy the engine
    differentiates (an energy tap: the probe builds the line graph with the positions its derivative needs)."""
    if adapter.topology_spec().triplet_cutoff is None:
        return True
    return not isinstance(adapter, ModelOwnedDerivative) and isinstance(adapter, (EnergyTap, SystemEnergyTap))


def can_capture(adapter) -> bool:
    """Whether a CUDA graph may capture the adapter's evaluation: a declared contract that is sync-free,
    without dynamic torch.compile (``compiled`` counts as dynamic unless the contract says capture-safe)."""
    contract, declared = contract_of(adapter)
    return (declared and contract.sync_free and contract.compiled != "dynamic"
            and (not adapter.compiled or contract.compiled == "capture_safe"))


def multi_gpu_decision(adapter, world_size: int, shard=None, skin: float | None = None) -> Decision:
    """FastMD supports one device per simulation."""
    if world_size != 1:
        raise NegotiationError("FastMD supports one device per simulation")
    return Decision("multi_gpu", "single_gpu", "world_size=1", evidence={"world_size": world_size})




def plan_fingerprint(adapter, *, profile: str, kernels: Mapping[str, str], model_dtype: str,
                     shard: Mapping[str, Any] | None = None) -> str:
    """sha256 over the model's weights (``state_dict``: names, dtypes, shapes, bytes), its name and
    upstream package, its static system inputs (tensors by their bytes), the profile, the kernel
    choices and the applied upstream patch records (names and source hashes); at P>1 also ``shard``
    (the declared shard's digest and its source pins: never through ``upstream_patches``, which P=1
    reads). Copies every weight to the host: a report, never inside a capture."""
    import torch
    digest = hashlib.sha256()
    system = dict(adapter.system_inputs()) if isinstance(adapter, SystemInputs) else {}
    header = {"model": adapter.name, "upstream": getattr(adapter, "upstream", None), "profile": profile,
              "kernels": dict(kernels), "model_dtype": model_dtype, "system_inputs": sorted(system)}
    pins = applied_pins(adapter.upstream_patches()) if isinstance(adapter, UpstreamPatches) else []
    if pins:  # (a model without applied records keeps its fingerprint)
        header["patches"] = pins
    if shard is not None:  # P>1 only
        header["shard"] = dict(shard)
    digest.update(json.dumps(header, sort_keys=True, default=str).encode())

    def update(name, value):
        import numpy as np
        if isinstance(value, (torch.Tensor, np.ndarray)):
            value = value.detach().reshape(-1).contiguous().cpu() if isinstance(value, torch.Tensor) else value
            digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
            data = value.view(torch.uint8).numpy() if isinstance(value, torch.Tensor) else np.ascontiguousarray(value)
            digest.update(data.tobytes())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                update(f"{name}.{key}", value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(f"{name}|{type(value).__name__}|{len(value)}".encode())
            for index, item in enumerate(value):
                update(f"{name}[{index}]", item)
        elif value is None or isinstance(value, (bool, int, float, str)):
            digest.update(f"{name}|{type(value).__name__}|{value!r}".encode())  # repr keeps nan, inf apart
        else:
            raise TypeError(f"{name}: a static system input must be a tensor, an array, a number, a string, "
                            f"None or a list/mapping of them, not {type(value).__name__}")
    for name in sorted(system):
        update(f"system_inputs.{name}", system[name])
    state = adapter.module.state_dict() if hasattr(adapter.module, "state_dict") else {}
    for name, tensor in state.items():
        if isinstance(tensor, torch.Tensor):
            update(name, tensor)
    return "sha256:" + digest.hexdigest()


def isolated_atoms_field(adapter, handle_isolated_atoms: bool) -> dict:
    policy = adapter.isolated_atom_policy()
    if not handle_isolated_atoms or policy == "error":
        effect = "isolated atoms are rejected" + (
            " (the model has no reference energies)" if handle_isolated_atoms else " (isolated_atoms='error')")
    elif policy == "model":
        effect = "admitted; the model evaluates them itself (no reference-energy substitution)"
    else:
        effect = "admitted; an isolated atom keeps its element's reference energy"
    return {"policy": policy, "requested": "reference" if handle_isolated_atoms else "error", "effect": effect}


def derivative_decision(contract: CaptureContract, declared: bool = True) -> Decision:
    """Who differentiates the energy (the contract's force mode) and the unit of the stress it returns
    (the runtime reports eV/Å³: a stress in GPa is multiplied by ``ase.units.GPa``)."""
    reason = {"model_owned": "the model's own call returns forces and stress",
              "engine_autograd": "one engine-owned backward pass of the model's energy (forces and stress in one "
                                 "VJP)",
              "direct_head": "the model's force head (no backward pass)"}[contract.force_mode]
    reason = f"{reason}; stress in {contract.stress_unit}"
    return Decision("derivative", contract.force_mode, reason if declared else f"not declared, the default: {reason}",
                    chain=("model_owned", "engine_autograd", "direct_head"),
                    evidence={"stress_unit": contract.stress_unit})


def precision_decision(state: Mapping[str, Any], requested: str | None, model_dtype: str, *,
                       default_ieee: bool, cuda: bool = True) -> tuple[Decision, str | None]:
    """The precision decision from a measured state (``fastmd.precision.precision_state``) and the
    configuration's ``matmul_precision``; returns it and the matmul precision the forwards must run
    with (``default_ieee``: a model other than MatRIS, IEEE FP32 unless TF32 was asked for)."""
    environment = {name: value for name, value in (state.get("environment") or {}).items() if value is not None}
    evidence = {"measured": {key: state.get(key) for key in ("fp32_gemm", "float32_matmul_precision")},
                "requested_matmul_precision": requested, "model_dtype": model_dtype, "environment": environment}
    if model_dtype == "float64":
        return Decision("precision", "fp64", "FP64 model: FP64 GEMMs (3xTF32 and TF32 do not apply); "
                        "whole_step stays available", chain=("fp64", "fp32"), evidence=evidence), requested
    if not cuda:
        return Decision("precision", "ieee", "CPU: FP32 GEMMs (the cuBLAS TF32 state does not apply)",
                        chain=("ieee", "tf32"), evidence=evidence), requested
    mode = state.get("fp32_gemm")
    # The Triton FP32 dot sites are MatRIS operator sites: another model runs none of them.
    triton = {} if default_ieee else {site: value for site, value in (state.get("triton_dot") or {}).items()
                                      if site != "requested"}
    tf32_dots = sorted(site for site, value in triton.items() if value == "tf32")
    if triton:
        evidence["triton_dot"] = dict(state.get("triton_dot") or {})
    asked_tf32 = requested in ("high", "medium")
    if default_ieee and mode == "tf32" and not asked_tf32:
        # Never TF32 by default: the forwards run under "highest" (IEEE FP32).
        return Decision("precision", "ieee", "IEEE FP32 by default for this model (the process default is "
                        "TF32" + (f": {environment}" if environment else "") + "; forwards run at 'highest')",
                        chain=("ieee", "tf32"), evidence=evidence), "highest"
    if mode == "tf32" or tf32_dots:
        where = "cuBLAS runs" if mode == "tf32" else "Triton dots " + ", ".join(tf32_dots) + " run"
        why = ("requested (matmul_precision=%r)" % requested if asked_tf32
               else "not requested: the process default" + (f" ({environment})" if environment else ""))
        return Decision("precision", "tf32", f"{where} TF32, {why}", chain=("ieee", "tf32"),
                        evidence=evidence, accuracy_class="reduced"), requested
    dots = sorted(site for site, value in triton.items() if value == "tf32x3")
    reason = "measured: cuBLAS FP32 GEMMs run IEEE FP32" + (f"; 3xTF32 Triton dots at {', '.join(dots)}"
                                                            if dots else "")
    if default_ieee and requested is None:
        # Pinned, so a later process-wide TF32 switch does not reach the forwards.
        return Decision("precision", "ieee", reason + "; forwards run at 'highest'", chain=("ieee", "tf32"),
                        evidence=evidence), "highest"
    return Decision("precision", "ieee", reason, chain=("ieee", "tf32"), evidence=evidence), requested


def measured_precision(matmul_precision: str | None, triton_dot_precision: str | None = None) -> dict:
    """``precision_state()`` under a configuration's precision scopes (what its calls run with)."""
    from fastmd.precision import matmul_precision_scope
    from fastmd.precision import precision_state
    from fastmd.precision import triton_dot_precision_scope
    with matmul_precision_scope(matmul_precision), triton_dot_precision_scope(triton_dot_precision):
        return precision_state()


def _precision_field(precision: Decision, state: Mapping[str, Any], model_dtype: str, cuda: bool,
                     cublas: str | None = None) -> dict:
    """The record's precision: the decision, the cuBLAS FP32 GEMM math the forwards run with, the device
    and the measured state."""
    return {"chosen": precision.chosen, "cublas": cublas or state.get("fp32_gemm"),
            "device": "cuda" if cuda else "cpu", "model_dtype": model_dtype, "state": state}


# --- models other than MatRIS: negotiated ------------------------------------------------------------

def negotiate(adapter, config, *, entry: str = "gpu_md", device=None, pbc=(True, True, True),
              world_size: int = 1, capacity: Mapping[str, int] | None = None, memory=None,
              budget: int | None = None, whole_step_trial: Callable[[], float] | None = None,
              model_graph_pools: int = 1, shard=None, skin: float | None = None,
              agreed_kernels: Mapping[str, str] | None = None) -> EnginePlanRecord:
    """The engine plan of a model other than MatRIS (MatRIS: :func:`entry_plan_record`, :func:`dry_run`).

    ``config``: a ``ResolvedConfig`` (or an ``InferenceConfig``, resolved with ``entry``'s defaults).
    ``capacity``: the edge (and triplet) capacity the graph paths would hold, for the memory bound;
    ``memory``: the adapter's :class:`~fastmd.adapters.memory.MeasuredMemoryModel` (K6 has no
    calibration for it), ``budget``: the device bytes a capture may use above what is allocated now
    (default: free + reserved - allocated - the admission margin); ``whole_step_trial``: returns
    whole_step's speedup over model_graph (GPU-resident MD's trial under execution auto); without one
    the execution decision says that the trial is pending. ``model_graph_pools``: the captures (graph
    pools) model_graph holds at once (GPU-resident MD with ``log_stress``: 2, its logged steps' runner
    next to its plain one; whole_step's windows share one pool): its memory bound is that many times the
    measured capture's. Kernel options are installed here (``select_kernel``), outside every graph,
    before the precision is measured; the contract is read again after every kernel installation. Topology
    reuse and padding follow the execution decision (the paths that implement them).

    FastMD supports a single device; distributed-only inputs are rejected.
    """
    import torch
    from fastmd.config import InferenceConfig, ResolvedConfig, resolve_config
    if runtime_planned(adapter):
        raise TypeError("MatRIS is not negotiated: use entry_plan_record(entry point) or dry_run()")
    if config is None or isinstance(config, InferenceConfig):
        config = resolve_config(config, default_execution="auto" if entry in ("gpu_md", "calculator", DISTRIBUTED)
                                else "eager", default_isolated_atoms="error" if entry == "torchsim" else "reference")
    if not isinstance(config, ResolvedConfig):
        raise TypeError("config must be an InferenceConfig, a ResolvedConfig or None")
    check_profile(adapter, config.optimization_profile)
    if entry == DISTRIBUTED or shard is not None or skin is not None or agreed_kernels is not None:
        raise NegotiationError("FastMD supports single-device entry points only")
    multi_gpu = multi_gpu_decision(adapter, world_size, shard=shard if entry == DISTRIBUTED else None, skin=skin)
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda = device.type == "cuda" and torch.cuda.is_available()
    contract, declared = contract_of(adapter)
    _check_padding(adapter, contract)  # before any kernel is installed
    topology = adapter.topology_spec()
    full_pbc = all(bool(axis) for axis in pbc)
    restores: list = []

    from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
    builder = cuda and _fixed_builder_available

    # Kernels are installed for the requested execution, under auto for the first mode of the entry
    # point's chain (what auto aims for); a site that ranks its options per mode follows the decision below.
    target = config.execution if config.execution not in ("auto", None) else _entry_chain(entry)[0]
    choices = tuple(adapter.kernel_backends()) if isinstance(adapter, KernelBackends) else ()
    kernels, kernel_decisions = {}, []
    for choice in choices:
        decision = _kernel_decision(adapter, choice, config.optimization_profile, contract, restores, target)
        kernels[choice.site] = decision.chosen
        kernel_decisions.append(decision)
    if not kernel_decisions:
        kernel_decisions.append(Decision("kernels", "upstream", "the adapter declares no kernel choices: the "
                                         "upstream module as it is", chain=("upstream",)))
    # The contract of the installed kernels (a compiled kernel changes what the model declares about capture).
    contract, declared = contract_of(adapter)

    # Precision, measured after the kernels were installed (check_precision measures it again on request).
    model_dtype = topology.model_dtype
    state = measured_precision(config.matmul_precision)
    precision, matmul_precision = precision_decision(state, config.matmul_precision, model_dtype, default_ieee=True,
                                                     cuda=cuda)
    cublas = measured_precision(matmul_precision)["fp32_gemm"] if cuda else None
    execution, memory_field = _foreign_execution(
        adapter, config, entry, cuda, contract, full_pbc, builder, capacity, memory, budget, whole_step_trial, device,
        model_graph_pools)
    # Under auto, where only a memory model is missing, the entry points measure one (measured_by_entry_points)
    # and decide again (renegotiate_execution): the kernels stay those of auto's aim until then.
    pending = (execution.evidence.get("memory_model") == "not measured" and builder
               and measured_by_entry_points(adapter))
    if execution.chosen != target and not pending:
        compiled = adapter.compiled
        kernel_decisions = _execution_kernels(adapter, config, contract, execution.chosen, kernel_decisions,
                                              restores)
        kernels = {d.id.removeprefix("kernels."): d.chosen for d in kernel_decisions if d.id.startswith("kernels.")}
        contract, declared, execution = _kernels_contract(
            adapter, config, entry, cuda, contract, compiled, full_pbc, builder, capacity, memory, budget, device,
            execution, model_graph_pools)
    neighbor_list, topology_reuse, padding = _graph_decisions(
        adapter, config, entry, cuda, builder, contract, topology, pbc, execution.chosen)

    _check_padding(adapter, contract)
    recompute = _foreign_recompute(adapter, config.saved_tensor_recompute)
    classes = [contract.accuracy_class, precision.accuracy_class] + [d.accuracy_class for d in kernel_decisions]
    patches = adapter.upstream_patches() if isinstance(adapter, UpstreamPatches) else ()
    return EnginePlanRecord(
        model=adapter.name, upstream=getattr(adapter, "upstream", None),
        fingerprint=plan_fingerprint(adapter, profile=config.optimization_profile, kernels=kernels,
                                     model_dtype=model_dtype),
        profile=config.optimization_profile, entry=entry, world_size=int(world_size),
        decisions=[neighbor_list, topology_reuse, padding, *kernel_decisions, derivative_decision(contract, declared),
                   execution, precision, recompute, multi_gpu],
        contract=contract, accuracy_class=loosest_class(*classes),
        isolated_atoms=isolated_atoms_field(adapter, config.handle_isolated_atoms), memory=memory_field,
        precision=_precision_field(precision, state, model_dtype, cuda, cublas),
        matmul_precision=matmul_precision, precision_restores=restores, contract_declared=declared,
        patches=patch_report(patches))


def _graph_decisions(adapter, config, entry, cuda, builder, contract, topology, pbc, mode) -> tuple[Decision, ...]:
    """The neighbor list, topology reuse and padding of a model other than MatRIS, which follow its
    execution decision ``mode`` (the paths that implement them): GPU-resident MD rebuilds its graph every
    step (``TensorGraphBuilder``) except under whole_step, whose fixed-capacity builder reuses candidates;
    the calculator rebuilds per call (``atoms_to_edges_gpu``) or reuses candidates on its skin path. A
    padded path (a CUDA graph, or the skin path's fixed capacities) pads with the adapter's rule."""
    full_pbc = all(bool(axis) for axis in pbc)
    skin = config.capacity.candidate_skin
    if entry == "gpu_md":
        reuse_off = [why for ok, why in ((mode == "whole_step", f"{mode}: the graph is rebuilt every step"),)
                     if not ok]
    else:
        reuse_off = [why for ok, why in (
            (entry == "calculator", f"{entry}: the graph is rebuilt every call"),
            (config.topology_reuse != "off", "topology_reuse='off'"),
            (config.topology_reuse == "skin" or mode == "model_graph", f"auto: {mode} execution rebuilds per call"),
            (contract.exact_cutoff or contract.exact_at_cutoff,
             "no exact cutoff declared (a stale or kNN list is never reused)"),
            (full_pbc, "not periodic in all directions"), (skin > 0, "capacity.candidate_skin is 0"),
            (builder, "no engine builder")) if not ok][:1]
    topology_reuse = Decision(
        "topology_reuse", "rebuild" if reuse_off else "exact_skin_refilter",
        reuse_off[0] if reuse_off else
        f"exact cutoff: candidates within cutoff + {skin} Å, re-filtered to r < {topology.cutoff} Å every evaluation",
        chain=("exact_skin_refilter", "rebuild"), evidence={"candidate_skin": skin, "cutoff": topology.cutoff})
    reuse = topology_reuse.chosen == "exact_skin_refilter"
    rebuilt = "tensor_graph_builder" if entry == "gpu_md" else "gpu_graph_builder"
    graph = ("edges only" if topology.triplet_cutoff is None
             else f"edges and triplets (triplet cutoff {topology.triplet_cutoff} Å)")
    neighbor_list = Decision(
        "neighbor_list", ("fixed_capacity_builder" if reuse else rebuilt) if cuda else "host",
        ("no CUDA device" if not cuda else
         f"{graph}: candidates within cutoff + skin on the device, re-filtered exactly every evaluation" if reuse
         else f"{graph}: the device neighbor list, rebuilt every evaluation ({topology_reuse.reason})"),
        chain=("fixed_capacity_builder", rebuilt, "host"),
        evidence={"pbc": [bool(axis) for axis in pbc], "triplets": topology.triplet_cutoff is not None})
    rule = getattr(adapter, "padding_rule", None)
    padding = _padding_decision(mode if mode != "eager" or not reuse else "skin", config,
                                rule() if callable(rule) else rule)
    return neighbor_list, topology_reuse, padding


def renegotiate_execution(record: EnginePlanRecord, adapter, config, *, entry: str, device=None,
                          pbc=(True, True, True), capacity: Mapping[str, int] | None = None, memory=None,
                          budget: int | None = None,
                          whole_step_trial: Callable[[], float] | None = None,
                          model_graph_pools: int = 1) -> EnginePlanRecord:
    """A copy of ``record`` (``adapter``'s negotiated plan) whose execution decision, and the neighbor
    list, topology reuse and padding that follow it, are decided again for a system: its ``capacity``,
    a ``memory`` model measured on it, an optional ``whole_step_trial`` and ``model_graph_pools`` (see
    :func:`negotiate`). A kernel site that orders its options per execution mode follows the new decision
    (the fingerprint, upstream, contract, derivative and accuracy class with it); the patch report (and the
    fingerprint) follow the adapter's applied records now; the precision, recompute and multi-GPU decisions
    stay."""
    import torch
    from fastmd.config import ResolvedConfig
    if not isinstance(config, ResolvedConfig):
        raise TypeError("config must be a ResolvedConfig")
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda = device.type == "cuda" and torch.cuda.is_available()
    contract, _ = contract_of(adapter)
    topology = adapter.topology_spec()
    from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
    builder = cuda and _fixed_builder_available
    full_pbc = all(bool(axis) for axis in pbc)
    execution, memory_field = _foreign_execution(
        adapter, config, entry, cuda, contract, full_pbc, builder, capacity, memory, budget, whole_step_trial, device,
        model_graph_pools)
    plan = copy.deepcopy(record)
    kernel_decisions = [decision for decision in plan.decisions if decision.id.startswith("kernels.")]
    compiled = adapter.compiled
    decided = _execution_kernels(adapter, config, contract, execution.chosen, kernel_decisions, plan.precision_restores)
    contract, declared, execution = _kernels_contract(
        adapter, config, entry, cuda, contract, compiled, full_pbc, builder, capacity, memory, budget, device,
        execution, model_graph_pools)
    _check_padding(adapter, contract)
    replaced = dict(zip(("neighbor_list", "topology_reuse", "padding"), _graph_decisions(
        adapter, config, entry, cuda, builder, contract, topology, pbc, execution.chosen)), execution=execution,
        derivative=derivative_decision(contract, declared))
    for decision in decided:
        replaced[decision.id] = decision
    plan.decisions = [replaced.get(decision.id, decision) for decision in plan.decisions]
    plan.memory = memory_field
    plan.contract, plan.contract_declared = contract, declared
    patches = patch_report(adapter.upstream_patches()) if isinstance(adapter, UpstreamPatches) else plan.patches
    # Other kernels, or other applied patch records (e.g. the adapter closed since): a new plan identity.
    if [d.chosen for d in decided] != [d.chosen for d in kernel_decisions] or patches != plan.patches:
        kernels = {d.id.removeprefix("kernels."): d.chosen for d in decided}
        plan.upstream, plan.patches = getattr(adapter, "upstream", None), patches
        plan.fingerprint = plan_fingerprint(adapter, profile=config.optimization_profile, kernels=kernels,
                                            model_dtype=topology.model_dtype)
    plan.accuracy_class = loosest_class(contract.accuracy_class, plan.decision("precision").accuracy_class,
                                        *(d.accuracy_class for d in decided))
    return plan


def _check_padding(adapter, contract: CaptureContract) -> None:
    """A model whose value and gradient are exactly 0 at its cutoff but not beyond (``exact_at_cutoff``
    without ``exact_cutoff``) must pad at the cutoff (``exact_rc_loops``): a far self-loop lies beyond the
    cutoff, where such a model's padded rows would add energy."""
    rule = getattr(adapter, "padding_rule", None)
    kind = getattr(rule() if callable(rule) else rule, "kind", None)
    if contract.exact_at_cutoff and not contract.exact_cutoff and kind == "far_self_loops":
        raise NegotiationError(f"model {adapter.name!r} declares a cutoff exact at the cutoff only (exact_at_cutoff), "
                               "so its padded rows must sit at the cutoff (PaddingRule('exact_rc_loops')), not beyond "
                               "it (far_self_loops)")


def _kernels_contract(adapter, config, entry, cuda, contract, compiled: bool, full_pbc, builder, capacity, memory,
                      budget, device, execution: Decision,
                      model_graph_pools: int) -> tuple[CaptureContract, bool, Decision]:
    """The contract of the kernels just installed for ``execution`` (an installed kernel may change what the
    model declares about capture, e.g. a compiled backbone, or its ``compiled`` flag, ``compiled`` before),
    and the execution decision: kept where that contract still allows it, else decided again with it (without
    a timing trial), which can only fall back along the chain; the kernels stay those installed for
    ``execution`` (the fallback modes run any kernel). A contract that no longer allows the decided mode never
    leaves it decided."""
    installed, declared = contract_of(adapter)
    fields = ("sync_free", "compiled", "exact_cutoff", "exact_at_cutoff")
    if adapter.compiled == compiled and all(getattr(installed, name) == getattr(contract, name) for name in fields):
        return installed, declared, execution
    decided, _ = _foreign_execution(adapter, config, entry, cuda, installed, full_pbc, builder, capacity, memory,
                                    budget, None, device, model_graph_pools)
    order = tuple(reversed(EXECUTION_CHAIN))  # slowest first
    if order.index(decided.chosen) >= order.index(execution.chosen):
        return installed, declared, execution
    return installed, declared, replace(decided, reason=f"{decided.reason} (the contract of the kernels installed "
                                       f"for {execution.chosen})")


def _entry_chain(entry: str) -> tuple[str, ...]:
    """The execution modes an entry point offers, fastest first."""
    allowed = {"gpu_md": EXECUTION_CHAIN, "calculator": ("model_graph", "eager"),
               DISTRIBUTED: ("model_graph", "eager")}.get(entry, ("eager",))
    return tuple(mode for mode in EXECUTION_CHAIN if mode in allowed)


def _execution_kernels(adapter, config, contract: CaptureContract, mode: str, decisions: Sequence[Decision],
                       restores: list) -> list[Decision]:
    """``decisions`` (the kernel decisions) with each site that orders its options per execution mode
    (``KernelChoice.by_execution``) decided again for ``mode``: its first option there that installs
    (a no-op where it is installed already); "generic" keeps the reference."""
    if not isinstance(adapter, KernelBackends) or config.optimization_profile == "generic":
        return list(decisions)
    choices = {f"kernels.{choice.site}": choice for choice in adapter.kernel_backends() if choice.by_execution}
    return [_kernel_decision(adapter, choices[decision.id], config.optimization_profile, contract, restores, mode)
            if decision.id in choices else decision for decision in decisions]


def _kernel_decision(adapter, choice: KernelChoice, profile: str, contract: CaptureContract,
                     restores: list, execution: str | None = None, agreed: str | None = None) -> Decision:
    """The first option of the site that installs and warms up, else its reference; "generic" takes
    the reference; options of class "reduced" are skipped for a model that is not itself "reduced"
    (reduced accuracy is never a default). The options rank in the site's order under ``execution``
    (``KernelChoice.order``); ``agreed`` (the distributed ranks' agreement): the chain starts there."""
    from fastmd.precision import restore_fp32_state
    site = f"kernels.{choice.site}"
    cls = lambda option: choice.accuracy.get(option, contract.accuracy_class)
    ranked = choice.order(execution)
    per_mode = execution in choice.by_execution
    if profile == "generic":
        order, why = [choice.reference], "optimization_profile='generic': the reference"
    else:
        looser = lambda option: cls(option) == "reduced" and contract.accuracy_class != "reduced"
        order = [option for option in ranked if not looser(option)]  # a reference is never declared reduced
        order += [choice.reference] if choice.reference not in order else []
        why = None
    if agreed is not None and agreed in order:
        order = order[order.index(agreed):]
    failures = {}
    skipped = [option for option in ranked if option not in order] if profile != "generic" else []
    for option in order:
        try:
            with restore_fp32_state(restores):
                adapter.select_kernel(choice.site, option)
        except Exception as error:  # noqa: BLE001 - an unavailable option falls back, with its reason
            failures[option] = f"{type(error).__name__}: {str(error).splitlines()[0][:160] if str(error) else ''}"
            if option == choice.reference:
                raise NegotiationError(f"{site}: the reference {option!r} failed to install ({failures[option]})")
            continue
        reason = why or ("fastest option" if option == ranked[0] else
                         f"fell back past {', '.join(failures) or 'nothing'}")
        if agreed is not None:
            reason += f" (agreed over the ranks: the first option every rank installs, from {agreed!r})"
        if per_mode and why is None:
            reason += f" under {execution} execution (the options rank differently there)"
        if skipped:
            reason += f" ({', '.join(skipped)} skipped: reduced accuracy is never a default)"
        evidence = {"failed": failures} if failures else {}
        if choice.by_execution and why is None:
            evidence["execution"] = execution
        return Decision(site, option, reason, chain=ranked, evidence=evidence,
                        accuracy_class=cls(option) if cls(option) != contract.accuracy_class else None)
    raise NegotiationError(f"{site}: no option installed")  # unreachable: the reference raises above


def _foreign_execution(adapter, config, entry, cuda, contract, full_pbc, builder, capacity, memory, budget,
                       trial, device, model_graph_pools: int = 1) -> tuple[Decision, dict]:
    """whole_step -> model_graph -> eager for a model other than MatRIS; K6 has no calibration for it,
    so a CUDA graph needs a measured memory bound (a valid MeasuredMemoryModel fit): one capture for
    whole_step (its windows share one pool), ``model_graph_pools`` captures for model_graph (where it holds
    more than one pool, the trial against it runs only where they fit)."""
    import torch
    from fastmd.engine.entry_config import check_execution
    requested = config.execution
    chain = _entry_chain(entry)
    memory_field = {"calibration": "uncalibrated", "note": "K6 has no calibration for this model"}
    if memory is not None:
        memory_field = {**memory.summary(), "note": "measured memory model (K6 uncalibrated)"}
    # ``compiled`` (the ModelAdapter flag) is dynamic compilation unless the contract says capture-safe.
    dynamic = contract.compiled == "dynamic" or (adapter.compiled and contract.compiled != "capture_safe")
    unsafe = [why for ok, why in ((contract.sync_free, "not declared sync-free"),
                                  (not dynamic, "dynamic torch.compile (traced at run time)")) if not ok]
    # whole_step reuses candidates within a skin, re-filtered exactly: it needs an exact cutoff.
    no_whole_step = [why for ok, why in ((full_pbc, "full PBC"), (builder, "the fixed-capacity builder"),
                                         (config.capacity.candidate_skin > 0, "a skin"),
                                         (contract.exact_cutoff or contract.exact_at_cutoff, "an exact cutoff"))
                     if not ok]
    if requested not in ("auto", None):
        check_execution(entry, requested, device if cuda else torch.device("cpu"))  # the entry point's rejection
        evidence = {"requested": requested}
        if requested != "eager" and unsafe:
            return Decision("execution", "eager", f"requested {requested}, but {'; '.join(unsafe)}: eager",
                            chain=chain, evidence=evidence), memory_field
        if requested == "whole_step" and no_whole_step:
            return Decision("execution", "model_graph", f"requested whole_step, which needs "
                            f"{', '.join(no_whole_step)}: model_graph", chain=chain, evidence=evidence), memory_field
        return Decision("execution", requested, "explicit request", chain=chain, evidence=evidence), memory_field
    blockers = [why for ok, why in (
        (cuda, "no CUDA device"), (config.checkpoint_enabled is False, "checkpointing (no CUDA graph)"),
        (config.capacity.warmup >= 1, "capacity.warmup=0 (a capture needs a warmup)")) if not ok] + unsafe
    evidence: dict = {"requested": "auto"}
    model_graph_fits, model_graph_why = True, None
    if not blockers and len(chain) > 1:
        fits, why, numbers = _memory_bound(memory, capacity, budget, device)
        evidence["memory"] = numbers
        if not fits:
            blockers.append(why)
            if memory is None:
                evidence["memory_model"] = "not measured"  # an entry point may measure one and decide again
        else:
            evidence["memory_reason"] = why
            if model_graph_pools > 1:
                model_graph_fits, model_graph_why, numbers = _memory_bound(memory, capacity, budget, device,
                                                                           pools=model_graph_pools)
                evidence["memory"]["model_graph"] = numbers
    if blockers or len(chain) == 1:
        return Decision("execution", "eager", "; ".join(blockers) or f"{entry}: eager only", chain=chain,
                        evidence=evidence), memory_field
    graph_only = "whole_step" not in chain or no_whole_step
    if not model_graph_fits and graph_only:
        return Decision("execution", "eager", f"model_graph holds {model_graph_pools} graph pools: {model_graph_why}"
                        + (f"; whole_step needs {', '.join(no_whole_step)}" if no_whole_step else ""), chain=chain,
                        evidence=evidence), memory_field
    if "whole_step" not in chain:
        return Decision("execution", "model_graph", evidence["memory_reason"], chain=chain,
                        evidence=evidence), memory_field
    if no_whole_step:
        return Decision("execution", "model_graph", f"{evidence['memory_reason']}; whole_step needs "
                        f"{', '.join(no_whole_step)}", chain=chain, evidence=evidence), memory_field
    if not model_graph_fits:
        return Decision("execution", "whole_step", f"{evidence['memory_reason']}; model_graph holds "
                        f"{model_graph_pools} graph pools ({model_graph_why}): whole_step without the trial "
                        "against it", chain=chain, evidence=evidence), memory_field
    if trial is None:
        return Decision("execution", "whole_step", f"{evidence['memory_reason']}; the whole-step trial against "
                        f"model_graph (kept at >= {WHOLE_STEP_MIN_SPEEDUP}x) is pending", chain=chain,
                        evidence=evidence), memory_field
    # Both replay the same model arithmetic: one accuracy class, so the trial may choose either.
    candidates = trial_candidates([("whole_step", contract.accuracy_class), ("model_graph", contract.accuracy_class)])
    speedup = float(trial())
    evidence["whole_step_speedup_over_model_graph"] = speedup
    if not math.isfinite(speedup) or speedup <= 0:
        return Decision("execution", "model_graph", f"{evidence['memory_reason']}; the whole-step trial failed "
                        f"(speedup {speedup})", chain=chain, evidence=evidence), memory_field
    chosen = "whole_step" if speedup >= WHOLE_STEP_MIN_SPEEDUP and "whole_step" in candidates else "model_graph"
    return Decision("execution", chosen, f"{evidence['memory_reason']}; trial: whole_step {speedup:.3f}x of "
                    f"model_graph ({'>=' if chosen == 'whole_step' else '<'} {WHOLE_STEP_MIN_SPEEDUP})",
                    chain=chain, evidence=evidence), memory_field


def _memory_bound(memory, capacity, budget, device, pools: int = 1) -> tuple[bool, str, dict]:
    """Whether the measured bound of ``pools`` captures (one graph pool each) fits the auto share of the
    budget."""
    from fastmd.engine.saved_recompute import AUTO_THRESHOLD
    if memory is None:
        return False, "no memory bound: K6 uncalibrated and no measured memory model", {}
    if not memory.valid:
        return False, f"measured memory fit invalid ({memory.reason}): no memory bound", {}
    if capacity is None:
        return False, "no capacity to predict the capture at", {}
    edges, triplets = int(capacity.get("edges", 0)), int(capacity.get("triplets", 0))
    bound = memory.bound("capture", edges, triplets)
    if bound is None:
        return False, f"no measured prediction at {edges} edges, {triplets} triplets (outside the measured range)", {}
    bound *= pools
    if budget is None:
        budget = capture_budget(device)
    fits = bound <= AUTO_THRESHOLD * budget
    numbers = {"capture_bound_gib": round(bound / 2**30, 3), "budget_gib": round(budget / 2**30, 3),
               "threshold": AUTO_THRESHOLD, "edges": edges, "triplets": triplets}
    what = "capture bound" if pools == 1 else f"bound of {pools} captures"
    if pools > 1:
        numbers["pools"] = pools
    return fits, (f"measured {what} {bound / 2**30:.2f} GiB {'<=' if fits else '>'} "
                  f"{AUTO_THRESHOLD:.0%} of {budget / 2**30:.1f} GiB"), numbers


def capture_budget(device) -> int:
    """The device bytes a capture may use above what is allocated now (the measured bound is a peak
    above the allocation): free + reserved - allocated, less the admission margin."""
    import torch
    from fastmd.engine import memory_admission
    free, total = torch.cuda.mem_get_info(device)
    return (free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
            - memory_admission.margin_bytes(total))


def _foreign_recompute(adapter, requested: str) -> Decision:
    levels = tuple(getattr(adapter, "recompute_levels", ())) if isinstance(adapter, RecomputeParticipant) else ()
    chain = levels or ("off",)
    if requested == "off":
        return Decision("recompute", "off", "requested off", chain=chain)
    if requested in levels:
        return Decision("recompute", requested, "explicit request", chain=chain)
    if not levels:
        return Decision("recompute", "off", f"saved_tensor_recompute={requested!r}: the model implements no "
                        "saved-tensor recompute", chain=chain)
    if requested == "auto":
        return Decision("recompute", "off", "auto: no calibrated recompute rule for this model (K6 uncalibrated)",
                        chain=chain)
    return Decision("recompute", "off", f"saved_tensor_recompute={requested!r}: the model implements "
                    f"{', '.join(levels)}", chain=chain)


def check_system_batching(adapter) -> str:
    """The system batching ``adapter`` declares for its loaded module
    (:class:`~fastmd.adapters.protocol.SystemBatching`), if the runtime may evaluate several of its systems as
    one disjoint union: "union" with a per-atom energy tap, or "segments" with an energy tap of either kind
    (the model keeps the systems apart itself); the graph padded with self-loops (the union builder's: a line
    graph, whose triplets the union builder lists per system as well, with far self-loops). Raises
    :class:`NegotiationError` otherwise. Host-only (no model call)."""
    name = getattr(adapter, "name", type(adapter).__name__)
    if runtime_planned(adapter) or isinstance(adapter, ModelOwnedDerivative):
        raise NegotiationError(f"model {name!r} differentiates its own energy: replicas of it run in one whole-step "
                               "graph through fastmd.execution.whole_step_cuda_graph.BatchedWholeStepGraphRunner, not "
                               "as a union")
    if not isinstance(adapter, (EnergyTap, SystemEnergyTap)):
        raise NegotiationError(f"model {name!r} has no energy tap: a union of systems is differentiated by the "
                               "engine")
    declared = adapter.system_batching() if isinstance(adapter, SystemBatching) else None
    if declared is None:
        raise NegotiationError(f"model {name!r} does not declare how its systems may share one graph (SystemBatching "
                               "'union' or 'segments' for the loaded module): its systems cannot be evaluated as one "
                               "union; run them one by one (GPUResidentMolecularDynamics)")
    if declared not in SYSTEM_BATCHING_KINDS:
        raise NegotiationError(f"model {name!r} declares system batching {declared!r}; the runtime batches "
                               f"{' and '.join(map(repr, SYSTEM_BATCHING_KINDS))} only")
    if declared == "union" and not isinstance(adapter, EnergyTap):
        raise NegotiationError(f"model {name!r} declares 'union' with a whole-system energy tap, which gives one energy "
                               "for the whole union: a model that pools over its system declares 'segments' and "
                               "returns one energy per system")
    padding = adapter.padding_rule().kind
    if padding == "sink_atoms":
        raise NegotiationError(f"model {name!r} pads with sink atoms: a union pads with self-loops (sink rows would "
                               "sit after the last system's rows)")
    triplet_cutoff = adapter.topology_spec().triplet_cutoff
    if triplet_cutoff is not None and padding != "far_self_loops":
        raise NegotiationError(f"model {name!r} has a line graph (triplets within {triplet_cutoff} Å) padded with "
                               f"{padding}: a union's line graph pads with far self-loops (padded triplets on padded "
                               "pairs)")
    return declared


def check_system_cells(adapter) -> str:
    """:func:`check_system_batching` for different systems in one union, each with its own atoms and cell
    (``SystemSegments.cells``, ``fastmd.interfaces.multi_system_md``): the model must also take the engine's
    per-edge shifts (:class:`~fastmd.adapters.protocol.EdgeShiftInput`) and declare "union" (a model that pools
    per system reads the batch's segments, which a slot's vacant rows would join), without a line graph (the
    union builder's line graph has one cell). Raises :class:`NegotiationError` otherwise. Host-only."""
    name = getattr(adapter, "name", type(adapter).__name__)
    declared = check_system_batching(adapter)
    if not (isinstance(adapter, EdgeShiftInput) and adapter.edge_shift_input()):
        raise NegotiationError(f"model {name!r} computes its edge vectors from one cell per graph batch (it does not "
                               "take the engine's per-edge shifts, EdgeShiftInput): its replicas share a union, "
                               "systems with cells of their own do not")
    if declared != "union":
        raise NegotiationError(f"model {name!r} declares {declared!r}: systems with cells of their own run in slots "
                               "for a model that declares 'union' (a model that pools per system would pool a slot's "
                               "vacant rows too)")
    if adapter.topology_spec().triplet_cutoff is not None:
        raise NegotiationError(f"model {name!r} has a line graph: systems with cells of their own run edges only")
    return declared


def batch_systems(record: EnginePlanRecord, adapter, *, systems: int,
                  self_check: Callable[[], Mapping[str, Any]] | None = None,
                  own_cells: bool = False) -> EnginePlanRecord:
    """A copy of ``record`` (``adapter``'s negotiated plan) with the decision to evaluate ``systems`` systems
    as one disjoint union (its declaration, "union" or "segments"), for the replica entry point
    (``fastmd.interfaces.replica_md``) and, with ``own_cells`` (systems with atoms and cells of their own,
    :func:`check_system_cells`), the multi-system one (``fastmd.interfaces.multi_system_md``).

    The model must pass :func:`check_system_batching` (with ``own_cells``, :func:`check_system_cells`);
    ``self_check`` (the entry point's evaluation of two differently rattled copies of the system, or of two of
    its systems, as a union and alone) must then pass,
    and its numbers are the decision's evidence. Anything else raises :class:`NegotiationError`: the runtime
    never batches a model silently, and never falls back to running the systems one by one."""
    name = getattr(adapter, "name", type(adapter).__name__)
    declared = check_system_cells(adapter) if own_cells else check_system_batching(adapter)
    evidence = {"systems": int(systems)}
    reason = f"declared {declared!r} for the loaded module; {int(systems)} systems in one graph"
    if self_check is not None:
        check = dict(self_check())
        evidence["self_check"] = check
        if not check.get("passed"):
            if check.get("failed") == "cell":
                raise NegotiationError(
                    f"model {name!r} takes the engine's per-edge shifts (EdgeShiftInput), but its energies and forces "
                    f"change with its cell argument (by {check.get('cell_argument_force_difference')} eV/Å and "
                    f"{check.get('cell_argument_energy_difference_per_atom')} eV/atom, beyond the tolerances "
                    f"{check.get('tolerance')} eV/Å and {check.get('energy_tolerance_per_atom')} eV/atom): it reads "
                    "the batch's cell, which holds one system's")
            if check.get("failed") == "energies":
                found = (f"in energy by {check.get('max_energy_difference_per_atom')} eV/atom, beyond the tolerance "
                         f"{check.get('energy_tolerance_per_atom')} eV/atom")
            else:
                found = (f"in force by {check.get('max_force_difference')} eV/Å, beyond the tolerance "
                         f"{check.get('tolerance')} eV/Å")
            evaluated = check.get("evaluated", "two copies of the system")
            alone = "the same systems" if evaluated == "two of its systems" else "the same copies"
            raise NegotiationError(
                f"model {name!r} declares {declared!r}, but {evaluated} evaluated as one union differ "
                f"from {alone} alone {found} (the noise floor of both paths): its atoms see the other system")
        reason += (f"; self-check: union vs alone max |dF| {check['max_force_difference']:.3g} eV/Å <= "
                   f"{check['tolerance']:.3g} eV/Å")
    plan = copy.deepcopy(record)
    plan.decisions = [decision for decision in plan.decisions if decision.id != "system_batching"] + [
        Decision("system_batching", declared, reason, chain=(declared,), evidence=evidence)]
    return plan


def size_unions(record: EnginePlanRecord, *, requested: int, memory, union_edges: Callable[[int], int], device,
                replicas: int | None = None, persistent_per_replica: int = 0, budget: int | None = None,
                union_triplets: Callable[[int], int] | None = None,
                unit: str = "replicas") -> tuple[int, EnginePlanRecord]:
    """The replicas per union that the memory admits, and a copy of ``record`` with that decision
    ("union_size"): ``requested``, unless the unions would not fit the auto share of ``budget`` (default: the
    device's capture budget), when the largest of ``requested`` halved as often as needed that does. Several
    unions share one graph pool: the need of unions of ``k`` replicas is the bound of one union's capture
    (``memory``: a :class:`~fastmd.adapters.memory.MeasuredMemoryModel` of unions of the system, fitted over
    their edge capacities; it holds that union's buffers) at the union's edge capacity ``union_edges(k)`` (and
    triplet capacity ``union_triplets(k)`` for a model with a line graph, whose fit has a per-triplet cost), plus
    the buffers the other ``replicas - k`` replicas keep outside the pool (``persistent_per_replica`` bytes
    each); a union beyond the measured range is halved like one that does not fit. ``memory`` None: not
    measured (a union too small), kept as requested, as without a valid capture fit (a failed measurement).
    Unions that do not fit at one replica raise :class:`NegotiationError`. ``unit`` names what a union holds
    (the multi-system entry point sizes its unions in "rows": the slots' atom ceilings)."""
    from fastmd.engine.saved_recompute import AUTO_THRESHOLD
    requested, replicas = int(requested), int(requested if replicas is None else replicas)
    triplets = (lambda size: 0) if union_triplets is None else union_triplets
    evidence: dict = {"requested": requested, "replicas": replicas, "union_edges": int(union_edges(requested)),
                      "persistent_bytes_per_replica": int(persistent_per_replica)}
    if union_triplets is not None:
        evidence["union_triplets"] = int(triplets(requested))
    chosen, reason = requested, f"{requested} {unit} per union"
    if memory is None:
        reason += ": too small to measure (not bounded)"
    else:
        budget = capture_budget(device) if budget is None else int(budget)
        limit = AUTO_THRESHOLD * budget

        def need(size: int) -> int | None:
            bound = memory.bound("capture", int(union_edges(size)), int(triplets(size)))
            return None if bound is None else bound + (replicas - size) * int(persistent_per_replica)

        evidence.update(memory=memory.summary(), budget_gib=round(budget / 2**30, 3), threshold=AUTO_THRESHOLD)
        first = need(requested)
        evidence["need_gib"] = None if first is None else round(first / 2**30, 3)
        capture = memory.fits.get("capture")
        if first is None and not (capture is not None and capture.valid):
            reason += f"; memory not bounded ({memory.reason})"
        elif first is not None and first <= limit:
            reason += (f": measured need {first / 2**30:.2f} GiB <= {AUTO_THRESHOLD:.0%} of the budget "
                       f"{budget / 2**30:.1f} GiB")
        else:
            while chosen > 1 and (need(chosen) is None or need(chosen) > limit):
                chosen = max(1, chosen // 2)
            fitted = need(chosen)
            if fitted is None or fitted > limit:
                smallest = "one replica" if unit == "replicas" else f"{chosen} {unit}"
                raise NegotiationError(f"{replicas} {unit} do not fit: unions of {smallest} need "
                                       f"{(fitted or 0) / 2**30:.2f} GiB > {AUTO_THRESHOLD:.0%} of the budget "
                                       f"{budget / 2**30:.1f} GiB")
            evidence.update(fitted_need_gib=round(fitted / 2**30, 3), fitted_union_edges=int(union_edges(chosen)))
            if union_triplets is not None:
                evidence["fitted_union_triplets"] = int(triplets(chosen))
            needed = ("are beyond the measured range" if first is None
                      else f"need {first / 2**30:.2f} GiB > {AUTO_THRESHOLD:.0%} of the budget {budget / 2**30:.1f} GiB")
            reason = f"{requested} {unit} per union {needed}: {chosen} per union ({fitted / 2**30:.2f} GiB)"
    plan = copy.deepcopy(record)
    plan.decisions = [decision for decision in plan.decisions if decision.id != "union_size"] + [
        Decision("union_size", str(chosen), reason, evidence=evidence)]
    return chosen, plan


# --- the plan record of an entry point (MatRIS's: recorded, not decided) ------------------------------

def entry_plan_record(instance, *, op_plan=None) -> EnginePlanRecord:
    """The plan record of a constructed entry point (``GPUResidentMolecularDynamics``,
    ``MLIPCalculator``): for MatRIS the decisions the runtime made, read from the instance; for
    another model the plan its entry point negotiated (``_engine_plan``: (adapter, record)), else
    one negotiated now and kept, so kernels are installed once. Where the entry point runs otherwise
    since (``_observed``) the record says so. The distributed entry point of a declared shard
    (``fastmd.distributed.ShardedEvaluator``) keeps its negotiated plan and reports it as it runs
    (its ``running_plan()``). ``op_plan`` is linked on a copy."""
    adapter = instance.adapter
    if callable(getattr(type(instance), "running_plan", None)):
        record = instance.running_plan()
    elif runtime_planned(adapter):
        record = _plan_recorder(adapter).entry_record(instance)
    else:
        plan = getattr(instance, "_engine_plan", None)
        if plan is None or plan[0] is not adapter:
            config = instance._inference_config
            if getattr(instance, "_execution_auto", False):
                config = replace(config, execution="auto")  # what was requested, not the mode resolved since
            plan = instance._engine_plan = (adapter, negotiate(adapter, config, entry=instance._config_kind,
                                                               device=_entry_device(instance)))
        record = _observed(instance, plan[1])
    return record if op_plan is None else record.link_op_plan(op_plan)


def _plan_recorder(adapter):
    """The module that records the plan of ``adapter``, which the runtime plans by its own rules
    (``runtime_planned``): the model registry names one per rule set (MatRIS's: ``fastmd.models.matris.plan_record``,
    with ``entry_record`` and ``dry_run_record``)."""
    import inspect
    from fastmd.models import plan_recorder
    return plan_recorder(inspect.getattr_static(adapter.__class__, "runtime_rules"))


# MatRIS's public rule functions, moved to its plan record, stay readable here (module attribute or from-import)
# until the old import paths are removed.
_MOVED_RULES = frozenset({"matris_execution_rule", "matris_recompute_decision"})


def __getattr__(name):
    if name in _MOVED_RULES:
        from fastmd.models import plan_recorder
        return getattr(plan_recorder("matris"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _observed(instance, record: EnginePlanRecord) -> EnginePlanRecord:
    """``record`` (a negotiated plan), or a copy saying where the entry point runs otherwise now: its
    execution (a fallback after a CUDA OOM, an explicit enable_* call), the calculator's last call's
    topology reuse (a system's first call and the calls after a cell change rebuild), and the neighbor
    list and padding of those paths."""
    from fastmd.engine.entry_config import config_summary
    effective = config_summary(instance)["effective"]
    execution = effective["execution"]
    observed = {"execution": (execution, effective.get("auto_execution", {}).get("reason"))}
    if instance._config_kind == "gpu_md":
        reuse, why = execution == "whole_step", None
    elif instance._config_has_evaluated:
        topology = instance.topology_reuse_summary()
        reuse, why = topology["last_call"] == "skin", topology["reason"]
    else:
        reuse, why = record.decision("topology_reuse").chosen == "exact_skin_refilter", None
    observed["topology_reuse"] = ("exact_skin_refilter" if reuse else "rebuild", why)
    rebuilt = "tensor_graph_builder" if instance._config_kind == "gpu_md" else "gpu_graph_builder"
    if record.decision("neighbor_list").chosen != "host":
        observed["neighbor_list"] = ("fixed_capacity_builder" if reuse else rebuilt, None)
    observed["padding"] = (record.decision("padding").chain[0] if reuse or execution != "eager" else "none", None)
    changed = {key: value for key, value in observed.items() if record.decision(key).chosen != value[0]}
    if not changed:
        return record
    record = copy.deepcopy(record)
    record.decisions = [replace(decision, chosen=changed[decision.id][0], reason=(
        f"{decision.reason} (negotiated {decision.chosen}; {changed[decision.id][0]} now"
        + (f": {changed[decision.id][1]})" if changed[decision.id][1] else ")")))
        if decision.id in changed else decision for decision in record.decisions]
    return record


def _entry_device(instance):
    """The device of an entry point's model (its first parameter), else the entry point's own."""
    import torch
    parameter = next(iter(instance.model.parameters()), None) if hasattr(instance.model, "parameters") else None
    if isinstance(parameter, torch.Tensor):
        return parameter.device
    device = getattr(instance, "device", None)
    return torch.device(device) if device is not None else torch.device("cpu")


def _padding_decision(execution: str, resolved, rule) -> Decision:
    """Eager evaluates exact shapes; a graph path (or "skin", the calculator's eager skin path) pads to
    its capacities with the adapter's padding rule (``ModelAdapter.padding_rule()``; None: sink atoms,
    MatRIS's, masked out of the energy by n_real)."""
    kind = getattr(rule, "kind", None) or "sink_atoms"
    chain = (kind, "none")
    if execution == "eager":
        return Decision("padding", "none", "eager: exact shapes", chain=chain)
    if kind == "sink_atoms":
        reason = (f"capacity-padded graphs: padded rows point at capacity.n_dummy={resolved.capacity.n_dummy} "
                  "sink atoms (masked by n_real)")
    elif kind == "far_self_loops":
        reason = ("the adapter's rule: self-loops round robin over the real atoms, with an integer image at least "
                  f"max({rule.min_length} Å, twice the cutoff) long")
    else:
        vector = "(cutoff, 0, 0)" if rule.pad_vector is None else str(tuple(rule.pad_vector))
        reason = f"the adapter's rule: image-0 self-loops round robin over the real atoms, edge vector {vector}"
    return Decision("padding", kind, reason, chain=chain)


def dry_run(adapter, config, *, entry: str = "gpu_md", device=None, atoms, world_size: int = 1) -> EnginePlanRecord:
    """The plan an entry point of ``entry`` ("gpu_md" or "calculator") would record for ``atoms``,
    without constructing it or running MD. Any other model: :func:`negotiate` without a measurement, so
    under execution auto eager with the reason, where the entry point measures a capturable model
    (:func:`measured_by_entry_points`) and may capture it (GPU-resident MD at construction, the calculator at
    a system's first repeat call). For MatRIS a mirror
    of the runtime's rules (``fastmd.models.matris.plan_record.dry_run_record``); its budget is the device's
    before the entry point allocates its state, so near the auto threshold the entry point may decide
    otherwise. Under auto the saved-tensor recompute level is previewed with the rule of the entry point's
    first evaluation. MatRIS at P>1 runs the distributed evaluator, whose plan this dry run does not cover
    (an error)."""
    import torch
    from fastmd.config import InferenceConfig
    from fastmd.config import resolve_config
    if entry not in ("gpu_md", "calculator"):
        raise ValueError("entry must be 'gpu_md' or 'calculator'")
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    pbc = tuple(bool(axis) for axis in atoms.get_pbc())
    resolved = resolve_config(config, default_execution="auto") if config is None or isinstance(
        config, InferenceConfig) else config
    if not runtime_planned(adapter):
        from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
        plan = negotiate(adapter, resolved, entry=entry, device=device, pbc=pbc, world_size=world_size)
        execution = plan.decision("execution")
        # Where the entry point measures: a model it measures with the fixed-capacity builder (the calculator:
        # not compiled at all, as its execution auto requires).
        if (resolved.execution == "auto" and execution.chosen == "eager" and "no memory bound" in execution.reason
                and measured_by_entry_points(adapter) and _fixed_builder_available
                and (entry == "gpu_md" or not adapter.compiled)):
            plan.decisions = [replace(decision, reason=decision.reason + " (this dry run measures nothing; the "
                                      "entry point measures the model on the system and may capture it)")
                              if decision.id == "execution" else decision for decision in plan.decisions]
        return plan
    return _plan_recorder(adapter).dry_run_record(adapter, resolved, entry=entry, device=device, atoms=atoms,
                                                  pbc=pbc, world_size=world_size)
