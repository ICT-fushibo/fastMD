"""MatRIS's engine plan record: the decisions the runtime makes for MatRIS by its own rules (``RuntimePlanned``,
the rule set "matris": execution auto in GPU-resident MD and in the calculator's capacity policy, K6 memory
admission, saved-tensor recompute, topology reuse, precision, the operator dispatch's kernel policy), recorded in the
negotiation's :class:`~fastmd.adapters.negotiate.EnginePlanRecord`, not decided anew: read from a constructed entry
point (:func:`entry_record`), or without one by a mirror of the same rules (:func:`dry_run_record`). The negotiation
(:func:`~fastmd.adapters.negotiate.entry_plan_record`, :func:`~fastmd.adapters.negotiate.dry_run`) reaches this
module through the model registry (``fastmd.models.plan_recorder``)."""
from __future__ import annotations

from typing import Any, Mapping

# The negotiation's record format and the record-building helpers it shares with the other models' plans.
from fastmd.adapters.negotiate import (
    EXECUTION_CHAIN,
    Decision,
    EnginePlanRecord,
    _entry_device,
    _padding_decision,
    _precision_field,
    derivative_decision,
    isolated_atoms_field,
    loosest_class,
    measured_precision,
    multi_gpu_decision,
    plan_fingerprint,
    precision_decision,
)


def entry_record(instance) -> EnginePlanRecord:
    """The plan record of a constructed MatRIS entry point: the decisions its runtime made, read from it."""
    from fastmd.engine.entry_config import config_summary
    adapter, resolved, kind = instance.adapter, instance._inference_config, instance._config_kind
    summary = config_summary(instance)
    effective = summary["effective"]
    execution = effective["execution"]
    if kind == "gpu_md":
        neighbor_list, reuse = _gpu_md_topology(execution, resolved)
    else:
        neighbor_list = _converter_decision(instance._gpu_graph)
        topology = instance.topology_reuse_summary()
        skin = topology["last_call"] == "skin"
        reason = topology["reason"] or ("skin epoch: candidates re-filtered exactly every call" if skin
                                        else "no call yet")
        reuse = Decision("topology_reuse", "exact_skin_refilter" if skin else "rebuild", reason,
                         chain=("exact_skin_refilter", "rebuild"),
                         evidence={key: topology[key] for key in ("requested", "last_call", "fallbacks")})
    execution_decision = _matris_execution_record(instance, kind, summary)
    state = measured_precision(resolved.matmul_precision, resolved.triton_dot_precision)
    model_dtype = adapter.topology_spec().model_dtype
    cuda = _entry_device(instance).type == "cuda"
    precision, _ = precision_decision(state, resolved.matmul_precision, model_dtype, default_ieee=False, cuda=cuda)
    decisions = [neighbor_list, reuse, _padding_decision(execution, resolved, None),
                 _matris_kernels(resolved, summary["kernel_policy"]), derivative_decision(adapter.capture_contract()),
                 execution_decision, precision, matris_recompute_decision(instance.model, resolved),
                 multi_gpu_decision(adapter, 1)]
    return _matris_record(adapter, resolved, kind, decisions, precision, state, model_dtype, cuda)


def _gpu_md_topology(execution: str, resolved) -> tuple[Decision, Decision]:
    """GPU-resident MD's graph source and topology reuse, both set by its execution mode."""
    whole_step = execution == "whole_step"
    neighbor_list = Decision(
        "neighbor_list", "fixed_capacity_builder" if whole_step else "tensor_graph_builder",
        "whole_step: candidates within cutoff + skin on the device, re-filtered exactly every step" if whole_step
        else f"{execution}: device neighbor list (nvalchemiops) every step",
        chain=("fixed_capacity_builder", "tensor_graph_builder"))
    reuse = Decision("topology_reuse", "exact_skin_refilter" if whole_step else "rebuild",
                     f"whole_step: candidate_skin {resolved.capacity.candidate_skin} Å" if whole_step
                     else f"{execution}: the graph is rebuilt every step", chain=("exact_skin_refilter", "rebuild"))
    return neighbor_list, reuse


def _converter_decision(gpu_graph: bool) -> Decision:
    return Decision("neighbor_list", "gpu_graph_converter" if gpu_graph else "host_graph_converter",
                    "the model's graph converter on the device" if gpu_graph else
                    "the model's graph converter on the host (legacy algorithm or no CUDA)",
                    chain=("gpu_graph_converter", "host_graph_converter"))


def _matris_record(adapter, resolved, kind, decisions, precision, state, model_dtype, cuda) -> EnginePlanRecord:
    from fastmd.engine import memory_admission
    memory = memory_admission.memory_model(adapter.module)
    contract = adapter.capture_contract()
    return EnginePlanRecord(
        model=adapter.name, upstream=adapter.upstream,
        fingerprint=plan_fingerprint(adapter, profile=resolved.optimization_profile, kernels={},
                                     model_dtype=model_dtype),
        profile=resolved.optimization_profile, entry=kind, world_size=1, decisions=decisions, contract=contract,
        accuracy_class=loosest_class(contract.accuracy_class, precision.accuracy_class),
        isolated_atoms=isolated_atoms_field(adapter, resolved.handle_isolated_atoms),
        memory={**memory.summary(), "calibrated": memory.calibrated, "strict": memory.strict,
                "note": "K6 (memory_admission)"},
        precision=_precision_field(precision, state, model_dtype, cuda),
        matmul_precision=resolved.matmul_precision)


def _matris_kernels(resolved, kernel_policy: Mapping[str, Any]) -> Decision:
    profile = resolved.optimization_profile
    reason = {"topology": "the fused kernels of the MatRIS operator dispatch",
              "generic": "plain torch reference operators",
              "legacy": "registry defaults plus MATRIS_* environment options"}[profile]
    return Decision("kernels", profile, f"optimization_profile={profile!r}: {reason}",
                    chain=("topology", "generic", "legacy"), evidence={"kernel_policy": dict(kernel_policy)})


def _matris_execution_record(instance, kind, summary) -> Decision:
    requested = summary["requested"].get("execution")
    effective = summary["effective"]
    chosen = effective["execution"]
    chain = EXECUTION_CHAIN if kind == "gpu_md" else ("model_graph", "eager")
    evidence = {"requested": requested}
    runner = getattr(instance, "runner" if kind == "gpu_md" else "_graph_runner", None)
    note = getattr(runner, "precapture_note", None)
    if note:
        evidence["model_graph_neighbor_buckets"] = note  # the capacity policy's precapture decision
    if not getattr(instance, "_execution_auto", False):
        reason = "explicit request"
        if chosen != requested:
            reason += f" of {requested}; {chosen} now (CUDA replay requirements, or changed after construction)"
        return Decision("execution", chosen, reason, chain=chain, evidence=evidence)
    evidence["auto"] = effective.get("auto_execution")
    if kind == "gpu_md":
        reason, choice = instance._auto_reason or "auto", getattr(instance, "_auto_choice", None)
        if choice is not None and choice != chosen:
            reason = f"{reason} (auto chose {choice} at construction; {chosen} now)"
        return Decision("execution", chosen, reason, chain=chain, evidence=evidence)
    auto = getattr(instance, "_auto", None)
    if auto is None:
        return Decision("execution", chosen, f"auto: {getattr(instance, '_auto_static', None) or 'eager'}",
                        chain=chain, evidence=evidence)
    reason = auto["reason"] or ("auto: a new system's first call runs eager and is measured; its first repeat "
                                "decides (K6 model_graph prediction)")
    return Decision("execution", auto["mode"], reason, chain=chain, evidence=evidence)


def matris_recompute_decision(model, resolved) -> Decision:
    """The saved-tensor recompute level a MatRIS entry point's forwards use now (``current_mode``),
    with the auto state's decision records."""
    from fastmd.engine import saved_recompute
    requested = resolved.saved_tensor_recompute
    chosen = {"off": "off", "narrow": "on", "deep": "deep"}[saved_recompute.current_mode(model, resolved)]
    state = saved_recompute.auto_record(model, resolved)
    evidence = {"requested": requested, **({"auto": state} if state is not None else {})}
    if requested != "auto":
        reason = "explicit request" if requested != "off" else "off (requested or resolved)"
    elif state is None:
        reason = "auto: decided at the first evaluation (none yet)"
    else:
        last = state["decisions"][-1] if state["decisions"] else {}
        reason = f"auto: {last.get('event', 'decided')}" + (
            f", {last['rows']} triplet rows, budget {last.get('budget_gib')} GiB" if "rows" in last else "")
    return Decision("recompute", chosen, reason, chain=("off", "on", "deep"), evidence=evidence)


def matris_execution_rule(adapter, resolved, kind: str, device, rows: int | None, *, full_pbc: bool,
                          builder: bool) -> Decision:
    """The execution decision a MatRIS entry point of ``kind`` makes for a system of ``rows`` triplet
    rows. A mirror of the entry points' rules (``configure_entrypoint``, then GPU-resident MD's
    ``_auto_mode``, or the calculator's construction and first-repeat rule; the calculator decides at a
    system's first repeat call, from its eager call's rows); tests compare it with the entry points.
    A request the entry point rejects raises its error."""
    import torch
    from fastmd.engine.entry_config import _auto_execution, check_execution
    from fastmd.engine.capacity_policy import graph_execution
    from fastmd.graph.fixed_capacity_builder import _fixed_builder_unavailable
    chain = EXECUTION_CHAIN if kind == "gpu_md" else ("model_graph", "eager")
    device = torch.device(device)
    if resolved.execution != "auto":
        check_execution(kind, resolved.execution, device)  # the entry point's own rejection
        return Decision("execution", resolved.execution, "explicit request", chain=chain,
                        evidence={"requested": resolved.execution})
    evidence = {"requested": "auto"}
    mode = _auto_execution(kind, device, resolved)
    if kind == "gpu_md":
        if mode == "eager":
            return Decision("execution", "eager", "CPU or checkpoint (no CUDA graph)", chain=chain, evidence=evidence)
        if adapter.compiled:
            return Decision("execution", "eager", "compiled model", chain=chain, evidence=evidence)
        if resolved.capacity.warmup < 1:
            return Decision("execution", "eager", "capacity.warmup=0 (a capture needs a warmup)", chain=chain,
                            evidence=evidence)
        graph, reason = graph_execution(adapter.module, resolved, int(rows or 0), device)
        evidence["rows"] = rows
        if not graph:
            return Decision("execution", "eager", reason, chain=chain, evidence=evidence)
        if not full_pbc or resolved.capacity.candidate_skin <= 0 or not builder:
            return Decision("execution", "model_graph",
                            f"{reason}; whole_step needs full PBC, a skin and the fixed-capacity builder"
                            + ("" if builder else f" ({_fixed_builder_unavailable})"),
                            chain=chain, evidence=evidence)
        return Decision("execution", "whole_step", reason, chain=chain, evidence=evidence)
    if mode == "eager":
        return Decision("execution", "eager", "auto: CPU or checkpoint (no CUDA graph)", chain=chain,
                        evidence=evidence)
    if not _gpu_converter(adapter):
        return Decision("execution", "eager", "auto: no GPU graph converter", chain=chain, evidence=evidence)
    if adapter.compiled:
        return Decision("execution", "eager", "auto: compiled model", chain=chain, evidence=evidence)
    if resolved.capacity.warmup < 1:
        return Decision("execution", "eager", "auto: capacity.warmup=0 (a capture needs a warmup)", chain=chain,
                        evidence=evidence)
    graph, reason = graph_execution(adapter.module, resolved, int(rows or 0), device,
                                    admission_off_trusts_any_model=True)
    evidence["rows"] = rows
    return Decision("execution", "model_graph" if graph else "eager",
                    f"from a system's first repeat call (its first call runs eager): {reason}", chain=chain,
                    evidence=evidence)


def _gpu_converter(adapter) -> bool:
    import torch
    return getattr(adapter.module.graph_converter, "algorithm", "legacy") == "gpu" and torch.cuda.is_available()


def dry_run_record(adapter, resolved, *, entry: str, device, atoms, pbc, world_size: int = 1) -> EnginePlanRecord:
    """The plan a MatRIS entry point of ``entry`` would record for ``atoms`` (``resolved``: its configuration,
    ``device`` and ``pbc`` as :func:`~fastmd.adapters.negotiate.dry_run` resolved them): a mirror of the runtime's
    rules (:func:`matris_execution_rule`, :func:`_matris_recompute_preview`)."""
    import torch
    if world_size != 1:
        raise ValueError("FastMD supports one device per simulation")
    from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
    # Only a CUDA device reaches a rule that reads the rows (auto runs eager on a CPU).
    cuda = device.type == "cuda" and torch.cuda.is_available()
    rows = _triplet_rows(adapter, atoms, device, entry) if cuda and (
        entry == "gpu_md" or _gpu_converter(adapter)) else None
    full_pbc = all(pbc)
    execution = matris_execution_rule(adapter, resolved, entry, device, rows, full_pbc=full_pbc,
                                      builder=_fixed_builder_available)
    chosen = execution.chosen
    if entry == "gpu_md":
        neighbor_list, reuse = _gpu_md_topology(chosen, resolved)
    else:
        neighbor_list = _converter_decision(_gpu_converter(adapter))
        reuse = _calculator_reuse_preview(adapter, resolved, chosen, device, full_pbc, _fixed_builder_available)
    from fastmd.engine.policy import policy_summary
    with resolved.scope():
        kernels = _matris_kernels(resolved, policy_summary())
    state = measured_precision(resolved.matmul_precision, resolved.triton_dot_precision)
    model_dtype = adapter.topology_spec().model_dtype
    precision, _ = precision_decision(state, resolved.matmul_precision, model_dtype, default_ieee=False,
                                      cuda=device.type == "cuda")
    recompute = _matris_recompute_preview(adapter, resolved, chosen, rows, device)
    decisions = [neighbor_list, reuse, _padding_decision(chosen, resolved, None), kernels,
                 derivative_decision(adapter.capture_contract()), execution, precision, recompute,
                 multi_gpu_decision(adapter, 1)]
    return _matris_record(adapter, resolved, entry, decisions, precision, state, model_dtype, device.type == "cuda")


def _calculator_reuse_preview(adapter, resolved, execution, device, full_pbc, builder) -> Decision:
    """The calculator's topology reuse for a system, in the order of its rules (``_skin_static_reason``,
    then ``_skin_runner``): the first condition that fails, else skin epochs and the call they start at."""
    from fastmd.graph.fixed_capacity_builder import _fixed_builder_unavailable
    mode, skin = resolved.topology_reuse, resolved.capacity.candidate_skin
    off = next((why for ok, why in (
        (mode != "off", "topology_reuse='off'"), (_gpu_converter(adapter), "no GPU graph converter"),
        (device.type == "cuda", "the model is not on a CUDA device"),
        (builder, f"fixed-capacity builder unavailable: {_fixed_builder_unavailable}"),
        (not (mode == "auto" and adapter.compiled), "auto: enable_compile"),
        (skin > 0, "capacity.candidate_skin is 0"),
        (not (mode == "auto" and execution != "model_graph"), "auto: eager execution rebuilds per call"),
        (full_pbc, "not periodic in all directions")) if not ok), None)
    if off is not None:
        return Decision("topology_reuse", "rebuild", off, chain=("exact_skin_refilter", "rebuild"))
    explicit_graph = resolved.execution == "model_graph"
    start = ("from the first call" if explicit_graph else "from a system's first repeat call (its first call runs "
             "eager)" if execution == "model_graph" else "from the second call (eager: the first call rebuilds)")
    return Decision("topology_reuse", "exact_skin_refilter", f"skin epochs {start}, candidates within cutoff + "
                    f"{skin} Å re-filtered exactly every call", chain=("exact_skin_refilter", "rebuild"))


def _triplet_rows(adapter, atoms, device, entry: str) -> int:
    """Triplet rows of ``atoms`` as the entry point counts them: GPU-resident MD's graph builder on
    the FP32 positions, the calculator's GPU graph converter."""
    import torch
    if entry == "calculator":
        graph = adapter.module.graph_converter(None, atoms=atoms, check_isolated_atoms=False)
        return int(sum(g.line_graph.shape[0] for g in (graph if isinstance(graph, list) else [graph])))
    from fastmd.graph import TensorGraphBuilder
    topology = adapter.topology_spec()
    builder = TensorGraphBuilder(
        cell=torch.as_tensor(atoms.get_cell().array, dtype=torch.float32, device=device),
        atomic_numbers=torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=device),
        pbc=torch.as_tensor(atoms.get_pbc(), dtype=torch.bool, device=device),
        atom_graph_cutoff=float(topology.cutoff), line_graph_cutoff=float(topology.triplet_cutoff),
        device=device, composition=atoms.get_chemical_formula(), check_isolated_atoms=False)
    positions = torch.as_tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    return int(builder.build(positions).line_graph.shape[0])


def _matris_recompute_preview(adapter, resolved, execution, rows, device) -> Decision:
    """The recompute decision a MatRIS entry point makes at its first evaluation, previewed with the
    rule of ``saved_recompute.AutoRecompute.decide_local``: explicit levels as given; auto on a CPU, with
    memory admission off or with compiled or checkpointed blocks "off" (no admission model); else the
    first level of the ladder whose K6 prediction fits the share of the current budget (the state's
    share: ``recompute_policy`` by execution, "safe" for model_graph with ``capacity.warmup < 2``, and
    for eager "safe" where "fast" would need expandable segments the allocator cannot switch to). The
    rows are those of the first evaluation: the system's for eager, its padded capacity for model_graph
    (its bucket) and whole_step (its start capacity). The entry point decides after allocating its
    state, and later captures at larger capacities (neighbor buckets, growth) can climb further."""
    chain = ("off", "on", "deep")
    requested = resolved.saved_tensor_recompute
    if requested != "auto":
        return Decision("recompute", requested, "explicit request" if requested != "off" else
                        "off (requested or resolved)", chain=chain, evidence={"requested": requested})
    if device.type != "cuda":
        return Decision("recompute", "off", "auto on a CPU: no admission model", chain=chain,
                        evidence={"requested": requested})
    from fastmd.engine.capacity_policy import CapturePolicy, graph_budget, tier
    from fastmd.config import env_value
    from fastmd.engine import memory_admission
    from fastmd.engine.saved_recompute import AUTO_THRESHOLD, LADDER, _PUBLIC, auto_level, policy_threshold
    if not memory_admission.enabled() or any(
            getattr(block, "enable_compile", False) or getattr(block, "enable_checkpoint", False) is not False
            for block in getattr(adapter.module, "interaction_block", ())):
        return Decision("recompute", "off", "auto: no admission model (memory admission off, or compiled or "
                        "checkpointed blocks)", chain=chain, evidence={"requested": requested})
    capacity = resolved.capacity
    decided = int(rows or 0)
    if execution == "model_graph":
        decided = tier(decided, capacity.t_step, 0)
    elif execution == "whole_step":
        demand = int(round(decided * float(env_value("MATRIS_CAPACITY_THERMAL_FACTOR", "1.0"))))
        headroom = int(demand * CapturePolicy.from_config(capacity).growth_headroom) // capacity.t_step * capacity.t_step
        decided = tier(demand + headroom, capacity.t_step, capacity.min_pad_t)
    memory = memory_admission.memory_model(adapter.module)
    options = resolved.kernel_options
    mode = "eager" if execution == "eager" else "model_graph"
    peaks = {level: [memory_admission.predict(
        "single", resolved.optimization_profile, mode, decided, recompute=level,
        mixed_width=memory_admission.mixed_width_active(options),
        single_bwd=memory_admission.single_bwd_mode(options) == "on", memory=memory)] for level in LADDER}
    calibrated = memory.calibrated and all(memory_admission.entry_calibrated(
        memory, "single", resolved.optimization_profile, mode, level) for level in LADDER)
    kind = "safe" if execution == "model_graph" and capacity.warmup < 2 else execution
    threshold = policy_threshold(resolved.recompute_policy, kind) if calibrated else AUTO_THRESHOLD
    if execution == "eager" and threshold > AUTO_THRESHOLD and not _expandable_segments_possible():
        threshold = AUTO_THRESHOLD
    budget = graph_budget(adapter.module, device)
    level = _PUBLIC[auto_level(peaks, [budget], memory.strict, threshold)]
    gib = {_PUBLIC[name]: None if values[0] is None else round(values[0] / 2**30, 2) for name, values in peaks.items()}
    return Decision("recompute", level, f"auto (previewed): the first K6 {memory.calibration} level within "
                    f"{threshold:.0%} of the current budget at the first evaluation's {decided} triplet rows",
                    chain=chain, evidence={"requested": requested, "rows": rows, "decided_rows": decided,
                                           "predicted_gib": gib, "budget_gib": round(budget / 2**30, 2),
                                           "threshold": threshold})


def _expandable_segments_possible() -> bool:
    """Whether the caching allocator uses, or could switch to, expandable segments (the condition of
    recompute_policy "fast" at P=1 eager), read without switching it."""
    import os
    import torch
    getter = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    conf = getter() if getter is not None else ",".join(
        os.environ.get(name, "") for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"))
    conf = conf.replace(" ", "").lower()
    if conf.strip(","):
        return "expandable_segments:true" in conf
    return not ((torch.distributed.is_available() and torch.distributed.is_initialized()
                 and torch.distributed.get_world_size() > 1)
                or torch.cuda.memory.get_allocator_backend() != "native")
