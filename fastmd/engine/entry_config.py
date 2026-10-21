"""Scoped application configuration, validation, and runtime reporting."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
from functools import wraps
import inspect
import threading
import warnings
import weakref

from fastmd.adapters import as_adapter
from fastmd.config import ConfigurationWarning
from fastmd.config import current_config
from fastmd.config import resolve_config


REMOVED_INFERENCE_ARGUMENTS = frozenset({
    "handle_isolated_atoms", "validate_isolated_atoms", "enable_checkpoint",
    "use_cuda_graph", "whole_step_cuda_graph", "candidate_skin",
    "enable_model_fusions", "u_step", "t_step", "n_dummy", "min_pad_u",
    "min_pad_t", "warmup",
})


def reject_removed_arguments(kwargs):
    """Do not let ASE's permissive **kwargs silently accept removed flags."""
    removed = REMOVED_INFERENCE_ARGUMENTS.intersection(kwargs)
    if removed:
        raise TypeError(
            "Removed inference arguments: " + ", ".join(sorted(removed))
            + ". Use config=InferenceConfig(...) instead."
        )


def scoped_init(init):
    """Runner constructor scope; ``config`` is an already resolved config."""
    signature = inspect.signature(init)
    @wraps(init)
    def wrapped(self, *args, **kwargs):
        self._inference_config = kwargs.get("config") or current_config()
        resolved = self._inference_config
        if resolved is not None:
            bound = signature.bind(self, *args, **kwargs)
            bound.arguments["config"] = resolved
            for name in ("u_step", "t_step", "n_dummy", "min_pad_u", "warmup"):
                if name not in signature.parameters:
                    continue
                value = getattr(resolved.capacity, name)
                if name in bound.arguments and bound.arguments[name] != value:
                    raise ValueError(f"Runner argument {name} conflicts with config.capacity.{name}")
                bound.arguments[name] = value
            args, kwargs = bound.args[1:], bound.kwargs
        with resolved.scope() if resolved is not None else nullcontext():
            return init(self, *args, **kwargs)
    return wrapped


def scoped_call(method):
    """Run an application method under its instance's kernel configuration."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        resolved = getattr(self, "_inference_config", None)
        with resolved.scope() if resolved is not None else nullcontext():
            result = method(self, *args, **kwargs)
        if method.__name__ in {"_predict_structure", "_predict_structure_scoped", "evaluate", "forward"}:
            self._config_has_evaluated = True
        return result
    return wrapped


def configure_entrypoint(kind):
    """Resolve the single public configuration before loading or inference."""
    def decorate(init):
        signature = inspect.signature(init)

        @wraps(init)
        def wrapped(self, *args, **kwargs):
            import torch
            reject_removed_arguments(kwargs)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            config = bound.arguments.get("config")
            device_arg = bound.arguments.get("device")
            device = (
                torch.device(device_arg) if device_arg is not None
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            default_execution = "eager" if kind == "torchsim" else "auto"
            default_isolated = "error" if kind == "torchsim" else "reference"
            resolved = resolve_config(
                config, default_execution=default_execution,
                default_isolated_atoms=default_isolated,
            )
            # execution="auto": the fastest mode that applies here (the entry
            # point falls back to eager after a CUDA OOM in a graph capture).
            self._execution_auto = resolved.execution == "auto"
            if self._execution_auto:
                resolved = replace(resolved, execution=_auto_execution(kind, device, resolved))
            check_execution(kind, resolved.execution, device)
            if resolved.execution != "eager" and resolved.checkpoint_enabled is not False:
                warnings.warn(
                    "Activation checkpointing inside CUDA Graph capture is not validated. "
                    "Use checkpoint='off' for the validated graph path; 'auto' is a row-count "
                    "threshold, not a capture-safety policy. The requested setting is unchanged.",
                    ConfigurationWarning, stacklevel=2,
                )
            self._inference_config = resolved
            self._config_has_evaluated = False
            self._config_kind = kind
            self._requested_inference_config = (
                {
                    "execution": config.execution,
                    "optimization_profile": config.optimization_profile,
                    "checkpoint": config.checkpoint,
                    "isolated_atoms": config.isolated_atoms,
                    "capacity": asdict(config.capacity),
                    "expert_overrides": dict(config.expert_overrides),
                    "matmul_precision": config.matmul_precision,
                    "topology_reuse": config.topology_reuse,
                    "triton_dot_precision": config.triton_dot_precision,
                } if config is not None else {
                    "source": "entrypoint defaults",
                    "execution": default_execution,
                    "isolated_atoms": default_isolated,
                    "checkpoint": "off",
                }
            )
            try:
                with resolved.scope():
                    init(*bound.args, **bound.kwargs)
                if device.type == "cuda":
                    from fastmd.precision import warn_implicit_tf32
                    # After init, at the instance's precision: another model's negotiated plan pins its
                    # forwards to IEEE FP32.
                    with self._inference_config.scope():
                        warn_implicit_tf32(self._inference_config.matmul_precision, f"MatRIS {kind}")
                if kind == "calculator" and self._inference_config.execution == "model_graph" and self._graph_runner is None:
                    raise ValueError("execution='model_graph' requires the GPU graph converter; the loaded model uses a CPU/legacy converter")
            except BaseException:
                release_adapter(self)  # a failed constructor gives its adapter back now, not at its collection
                raise
            if resolved.report:
                summary = config_summary(self)
                effective = summary["effective"]
                print(
                    f"MatRIS config: execution={effective['execution']}, "
                    f"profile={resolved.optimization_profile}, checkpoint={resolved.as_dict()['checkpoint']}, "
                    f"isolated_atoms={effective['isolated_atoms']}; "
                    f"merged projections={effective['merged_projections']['status']}"
                )
                for reason in effective["notes"]:
                    print(f"  {reason}")
        return wrapped
    return decorate

def check_execution(kind, execution, device):
    """Reject an execution mode the entry point ``kind`` does not support, or a CUDA graph without CUDA."""
    import torch
    allowed = {"eager", "model_graph", "whole_step"} if kind == "gpu_md" else {"eager", "model_graph"}
    if kind == "torchsim":
        allowed = {"eager"}
    if execution not in allowed:
        raise ValueError(f"{kind} does not support execution={execution!r}; supported: {sorted(allowed)}")
    if execution != "eager" and (device.type != "cuda" or not torch.cuda.is_available()):
        raise ValueError(f"execution={execution!r} requires an available CUDA device; got {device}")


def _auto_summary(instance):
    """execution="auto": the concrete mode, why, and the decisions so far."""
    state = getattr(instance, "_auto", None)
    if isinstance(state, dict):
        return {"mode": state["mode"], "reason": state["reason"], "decisions": list(state["decisions"]),
                "decision_count": state["decision_count"]}
    reason = getattr(instance, "_auto_reason", None)
    return {"mode": instance._inference_config.execution, "reason": reason or "resolved at construction"}


def _auto_execution(kind, device, resolved):
    """The concrete mode of ``execution="auto"`` for an entry point.

    CUDA graphs need a CUDA device and no checkpoint recomputation (not
    validated inside a capture); TorchSim's batched SimState path is eager
    only. Otherwise the fastest mode: whole_step for GPU MD, model_graph for
    the calculator (which also needs the GPU graph converter; it checks).
    """
    import torch
    if kind == "torchsim" or device.type != "cuda" or not torch.cuda.is_available():
        return "eager"
    if resolved.checkpoint_enabled is not False:
        return "eager"
    return "whole_step" if kind == "gpu_md" else "model_graph"


def current_adapter(instance):
    """An entry point's ``adapter``: the one it resolved, or a new one once ``model`` was rebound
    (its eager calls follow ``model``, as they always did)."""
    adapter = instance._adapter
    if adapter.module is not instance.model:
        adapter = instance._adapter = as_adapter(instance.model)
    return adapter


# Adapters an entry point loaded (by id): [adapter, the number of entry points using it]. Such an adapter is
# closed when the last entry point using it closes or is finalised, so one passed on (``entry.adapter`` given to
# another entry point) stays open while that one runs.
_LOADED_ADAPTERS: dict[int, list] = {}
# Guards the counts; its sections run no Python code (a finalizer releasing a use could run in one otherwise).
_LOADED_LOCK = threading.RLock()


def use_adapter(instance, model) -> None:
    """Register an entry point as a user of its adapter (``instance._adapter``) where an entry point loaded that
    adapter (this one, from a name or a checkpoint path, or another one whose ``adapter`` was passed in): the
    adapter is closed when its last such user closes, exits its context or is finalised (also after a failed
    constructor); an adapter's ``close`` reverts its upstream patches (reference-counted per record). An
    adapter or a model object the caller passed in that no entry point loaded (``model``) stays the caller's."""
    adapter = instance._adapter
    loaded = adapter is not model and adapter.module is not model and callable(getattr(adapter, "close", None))
    with _LOADED_LOCK:
        if loaded:
            _LOADED_ADAPTERS[id(adapter)] = [adapter, 0]
        entry = _LOADED_ADAPTERS.get(id(adapter))
        counted = entry is not None and entry[0] is adapter
        if counted:
            entry[1] += 1
    # (Outside the lock: setting an attribute may run Python code. The finalizer holds the adapter's id, never the
    # entry point.)
    instance._adapter_finalizer = weakref.finalize(instance, _release, id(adapter)) if counted else None


def release_adapter(instance) -> None:
    """The entry point stops using its adapter (once): the last user of an adapter an entry point loaded closes
    it; nothing for one the caller passed in."""
    finalizer = getattr(instance, "_adapter_finalizer", None)
    if finalizer is not None:
        finalizer()


def _release(key: int) -> None:
    with _LOADED_LOCK:
        entry = _LOADED_ADAPTERS.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1]:
            return
        del _LOADED_ADAPTERS[key]
    entry[0].close()


def apply_checkpoint(model, enabled):
    """Keep model metadata and every interaction block in agreement (MatRIS's activation
    checkpointing; a model without interaction blocks has none, and only "off" applies to it)."""
    if not hasattr(model, "interaction_block"):
        if enabled:
            raise ValueError("activation checkpointing applies to MatRIS models only")
        return
    model.enable_checkpoint = enabled
    model.config["enable_checkpoint"] = enabled
    for block in model.interaction_block:
        block.enable_checkpoint = enabled


def kernel_policy_summary(resolved, *, strict_errors=False):
    """The kernel policy an instance's scoped calls run with (fusion scope excluded)."""
    from fastmd.engine.policy import policy_summary
    with resolved.scope():
        return policy_summary(strict_errors=strict_errors)


def config_summary(instance):
    """Return requested/resolved settings and observed state, not promises."""
    resolved = instance._inference_config
    kind = instance._config_kind
    if kind == "gpu_md":
        execution = "whole_step" if instance.whole_step_runner is not None else "model_graph" if instance.runner is not None else "eager"
    elif kind == "calculator":
        execution = "model_graph" if instance._graph_runner is not None else "eager"
    else:
        execution = "eager"
    notes = []
    if getattr(instance, "_closed", False):
        notes.append("Closed: no further calculation.")
    elif execution != resolved.execution:
        notes.append(f"Requested {resolved.execution}; effective {execution} because CUDA replay requirements were not met.")
    isolated = "reference" if resolved.handle_isolated_atoms else "error"
    if resolved.handle_isolated_atoms and instance.adapter.isolated_atom_policy() == "model":
        isolated = "model"  # the model evaluates isolated atoms itself
    if instance.adapter.isolated_atom_policy() == "error":
        isolated = "error"
        if resolved.handle_isolated_atoms:
            notes.append("Model has no reference-energy table; isolated atoms are rejected.")
    try:
        device = next(instance.model.parameters()).device.type
    except (StopIteration, TypeError):
        device = str(getattr(instance, "device", "unknown")).split(":")[0]
    if device != "cuda":
        notes.append("CUDA-only kernel optimizations are not used on this device.")
    observed = bool(instance._config_has_evaluated)
    # Whole-step constructors already evaluate and capture before returning;
    # unlike the bucket runner, they do not expose a captures counter.
    observed = observed or getattr(instance, "whole_step_runner", None) is not None
    runners = (getattr(instance, "runner", None), getattr(instance, "_graph_runner", None), getattr(instance, "whole_step_runner", None))
    observed = observed or any(getattr(r, "captures", 0) > 0 for r in runners if r is not None)
    materialized = sum(
        getattr(module, "_merged_attn_proj_weight", None) is not None
        for module in instance.model.modules()
    ) if observed else None
    notes.append("Merged projection caches indicate materialization, not per-call use; CUDA device, frozen weights, dtype, feature widths, and row thresholds still gate kernel selection.")
    from fastmd.precision import matmul_precision_scope
    from fastmd.precision import precision_state
    from fastmd.precision import triton_dot_precision_scope
    with matmul_precision_scope(resolved.matmul_precision), triton_dot_precision_scope(resolved.triton_dot_precision):
        # What this instance's scoped calls run with, not the caller's state.
        matmul = precision_state()
    return {
        "requested": instance._requested_inference_config,
        "configuration": resolved.as_dict(),
        "fingerprint": resolved.fingerprint,
        "kernel_policy": kernel_policy_summary(resolved),
        "effective": {
            "execution": execution,
            "checkpoint": getattr(instance.model, "enable_checkpoint", None),
            "isolated_atoms": isolated,
            "matmul_precision": matmul,
            # Per Triton FP32 dot site: the input precision, or "cublas" (its cuBLAS path).
            "triton_dot_precision": matmul.pop("triton_dot"),
            "merged_projections": {
                "status": "observed" if observed else "pending first evaluation",
                "materialized_module_count": materialized,
            },
            "notes": notes,
            **({"topology_reuse": instance.topology_reuse_summary()} if kind == "calculator" else {}),
            **({"neighbor_cap": instance.neighbor_cap_summary()}
               if getattr(instance, "_neighbor_count_max", None) is not None else {}),
            **({"auto_execution": _auto_summary(instance)} if getattr(instance, "_execution_auto", False) else {}),
        },
    }
