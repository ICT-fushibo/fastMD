"""Reversible, instance-local precision patches for frozen FP32 linear layers.

Move a model to its final device, call ``eval().requires_grad_(False)``, then
install a plan BEFORE compiling or capturing it. Casts of the frozen weights
are cached at installation; input casts and their input-gradient backward stay
in the captured tensor program. The checkpoint tensors and state-dict keys do
not change. No autocast context or process precision flag is modified.

Only selected ``nn.Linear`` calls change. Their inputs, weights and bias are
rounded to BF16/FP16, ``F.linear`` runs in that dtype, and its output is cast
back to FP32. Autograd differentiates precisely this cast/linear/cast program;
this is not an IEEE derivative of a discontinuous floating-point function.
Accumulator precision is backend-dependent, not guaranteed by this wrapper.
Unselected geometry, nonlinearities, reductions and readouts retain their
original execution (and the caller must still set IEEE FP32 global math).

Lifecycle contract: do not train, mutate/move weights, enable other operator
rewrites, or change a plan while installed. Call ``validate()`` between trials
to detect changes to weights. Destroy every CUDA graph that references the
plan before ``restore()`` or context exit; old captured graphs cannot be
invalidated here. Compilation/capture is allowed AFTER installation and must
be redone for each plan. Do not share the patched instance across threads.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_OWNER = "_fastmd_layered_precision_owner"
_WEIGHT = "_fastmd_layered_precision_weight"
_BIAS = "_fastmd_layered_precision_bias"
_MISSING = object()
_DTYPES = {
    "fp32": torch.float32, "float32": torch.float32,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
}
_FAMILY_GROUPS = {
    "chgnet": ("atom_message", "bond_message", "angle_message", "protected"),
    "orb": ("edge_message", "node_message", "attention", "protected"),
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, str):
        result = _DTYPES.get(value.lower())
    else:
        result = value if value in _DTYPES.values() else None
    if result is None:
        raise ValueError(f"unsupported linear precision {value!r}; use fp32, bf16 or fp16")
    return result


def _linears(model: nn.Module) -> dict[str, nn.Linear]:
    if not isinstance(model, nn.Module):
        raise TypeError("expected the model nn.Module (adapter.module), not its adapter")
    # Include aliases: a shared layer must not be both protected and reduced.
    return {name: site for name, site in model.named_modules(remove_duplicate=False)
            if isinstance(site, nn.Linear)}


def discover_linear_groups(model: nn.Module, *, family: str) -> dict[str, list[str]]:
    """Classify CHGNet/ORB linear calls, protecting every unrecognized path.

    CHGNet's basis/embedding weights, site-wise head and final MLP are protected.
    ORB's encoder, decoder and all output heads are protected. Only standard
    message-layer MLPs and ORB attention projections are candidates. Counts
    describe modules/parameters, not FLOPs or measured execution coverage.
    """
    if family not in _FAMILY_GROUPS:
        raise ValueError(f"unknown model family {family!r}; expected chgnet or orb")
    groups = {name: [] for name in _FAMILY_GROUPS[family]}
    for name in _linears(model):
        group = "protected"
        parts = name.split(".")
        if family == "chgnet":
            group = {"atom_conv_layers": "atom_message", "bond_conv_layers": "bond_message",
                     "angle_layers": "angle_message"}.get(parts[0], "protected")
        elif len(parts) >= 4 and parts[:2] == ["model", "gnn_stacks"]:
            projection = parts[3]
            if projection == "_edge_mlp":
                group = "edge_message"
            elif projection == "_node_mlp":
                group = "node_message"
            elif projection in {"_receive_attn", "_send_attn"}:
                group = "attention"
        groups[group].append(name)
    return groups


def _explicit_groups(sites: Mapping[str, nn.Linear], groups: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for group, names in groups.items():
        if not isinstance(group, str) or not group:
            raise ValueError("group names must be nonempty strings")
        if isinstance(names, str):
            raise TypeError(f"group {group!r} needs a sequence of module paths, not one string")
        result[group] = list(names)
        for name in result[group]:
            if name not in sites:
                raise ValueError(f"group {group!r} names unknown linear module {name!r}")
            if name in assigned:
                raise ValueError(f"linear module {name!r} is assigned to more than one group")
            assigned.add(name)
    result.setdefault("protected", []).extend(name for name in sites if name not in assigned)
    return result


def _linear_forward(self: nn.Linear, input: Tensor) -> Tensor:
    # Static metadata checks are compile/capture friendly; no tensor readback.
    if input.dtype != torch.float32:
        raise TypeError("layered precision expects an FP32 input boundary")
    weight = self._fastmd_layered_precision_weight
    if input.device != weight.device:
        raise RuntimeError("model/input moved after precision installation; restore and reinstall")
    return F.linear(input.to(weight.dtype), weight,
                    self._fastmd_layered_precision_bias).to(input.dtype)


def _signature(parameter: Tensor | None) -> tuple | None:
    if parameter is None:
        return None
    return (id(parameter), parameter.data_ptr(), parameter._version,
            parameter.dtype, parameter.device, parameter.requires_grad)


@dataclass
class _Patch:
    module: nn.Linear
    names: list[str]
    precision: torch.dtype
    old_forward: Any
    weight_signature: tuple
    bias_signature: tuple | None
    forward: Any = None


class LayeredPrecisionInstallation:
    """An installed immutable plan. Use the factory or its context manager.

    ``restore`` is idempotent and only changes this instance's patched layers.
    It must run after all runners/captured graphs using this plan are destroyed.
    ``manifest`` remains available after restoration for experiment records.
    """

    def __init__(self, model: nn.Module, group_precisions: Mapping[str, str | torch.dtype], *,
                 family: str | None = None, groups: Mapping[str, Sequence[str]] | None = None) -> None:
        self.model = model
        self.family = family
        self._active = False
        self._patches: list[_Patch] = []
        sites = _linears(model)
        if groups is None:
            if family is None:
                raise ValueError("supply family='chgnet'/'orb' or explicit groups")
            groups = discover_linear_groups(model, family=family)
        self.groups = _explicit_groups(sites, groups)
        unknown = set(group_precisions) - set(self.groups)
        if unknown:
            raise ValueError(f"unknown precision groups: {sorted(unknown)}")
        self.precisions = {group: _dtype(group_precisions.get(group, "fp32")) for group in self.groups}
        if self.precisions["protected"] != torch.float32:
            raise ValueError("the protected group must stay fp32")
        if hasattr(model, _OWNER):
            raise RuntimeError("this model already has a layered precision installation")
        for name, module in model.named_modules():
            if module.training:
                raise ValueError(f"module {name!r} is training; call eval() before installation")
            if getattr(module, "_compiled_call_impl", None) is not None or hasattr(module, "_orig_mod"):
                raise ValueError("install layered precision before torch.compile; load ORB with compile=False")
            if hasattr(module, "_fastmd_projection_weights") or hasattr(module, "_fastmd_operator_backend"):
                raise ValueError("operator rewrites may bypass nn.Linear; use an unmodified model instance")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("layered precision is inference-only; freeze model parameters before installation")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters() if parameter.is_floating_point()):
            # ORB legitimately retains FP64 reference-energy buffers, but its
            # learned parameters and the selected linears must be FP32.
            raise ValueError("layered precision currently requires FP32 model parameters")

        assignments = {name: self.precisions[group] for group, names in self.groups.items() for name in names}
        aliases: dict[int, list[str]] = {}
        for name, module in sites.items():
            aliases.setdefault(id(module), []).append(name)
        self._sites = sites
        self._rows = []
        for names in aliases.values():
            module = sites[names[0]]
            precisions = {assignments[name] for name in names}
            if len(precisions) != 1:
                raise ValueError(f"shared linear aliases have conflicting precisions: {names}")
            precision = next(iter(precisions))
            self._rows.append({"paths": list(names), "groups": sorted({group for group, paths in self.groups.items()
                                                                      if any(name in paths for name in names)}),
                               "in_features": module.in_features, "out_features": module.out_features,
                               "parameter_count": module.weight.numel() + (0 if module.bias is None else module.bias.numel()),
                               "source_dtype": _dtype_name(module.weight.dtype),
                               "forward_compute_dtype": _dtype_name(precision),
                               "backward_input_gradient_compute_dtype": _dtype_name(precision),
                               "boundary_dtype": "float32", "patched": precision != torch.float32})
            if precision == torch.float32:
                continue
            if type(module) is not nn.Linear or getattr(module.forward, "__func__", None) is not nn.Linear.forward:
                raise ValueError(f"{names[0]!r} is not a standard nn.Linear.forward; refusing to replace custom arithmetic")
            if any(hasattr(module, key) for key in (_OWNER, _WEIGHT, _BIAS)):
                raise RuntimeError(f"linear {names[0]!r} already has precision patch attributes")
            self._patches.append(_Patch(module, names, precision, module.__dict__.get("forward", _MISSING),
                                        _signature(module.weight), _signature(module.bias)))

        # Allocate every cache before mutating any forward: preflight failure
        # cannot leave half a plan installed. Tensors are ordinary attributes,
        # not parameters/buffers, so state_dict is exactly the original one.
        caches = [(patch.module.weight.detach().to(patch.precision),
                   None if patch.module.bias is None else patch.module.bias.detach().to(patch.precision))
                  for patch in self._patches]
        installed: list[_Patch] = []
        try:
            setattr(model, _OWNER, self)
            self._active = True
            for patch, (weight, bias) in zip(self._patches, caches):
                installed.append(patch)
                setattr(patch.module, _OWNER, self)
                setattr(patch.module, _WEIGHT, weight)
                setattr(patch.module, _BIAS, bias)
                patch.forward = MethodType(_linear_forward, patch.module)
                patch.module.forward = patch.forward
        except BaseException:
            self._restore(installed)
            raise
        self._cache_bytes = sum(weight.numel() * weight.element_size()
                                + (0 if bias is None else bias.numel() * bias.element_size())
                                for weight, bias in caches)

    @property
    def active(self) -> bool:
        return self._active

    def validate(self) -> None:
        """Check frozen state outside compiled/captured work; never call per step."""
        if not self._active:
            raise RuntimeError("the precision plan has been restored")
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("the installed inference model was switched to training")
        for patch in self._patches:
            if patch.module.__dict__.get("forward") is not patch.forward:
                raise RuntimeError(f"forward changed after installation: {patch.names}")
            if (_signature(patch.module.weight) != patch.weight_signature
                    or _signature(patch.module.bias) != patch.bias_signature):
                raise RuntimeError(f"weights changed after installation: {patch.names}; restore and reinstall")

    def manifest(self) -> dict:
        """JSON-safe plan and structural coverage, without GPU synchronization."""
        rows = [{**row, "paths": list(row["paths"]), "groups": list(row["groups"])} for row in self._rows]
        total = len(rows)
        selected = sum(row["patched"] for row in rows)
        parameters = sum(row["parameter_count"] for row in rows)
        selected_parameters = sum(row["parameter_count"] for row in rows if row["patched"])
        return {"schema": "fastmd.layered_precision/1", "family": self.family,
                "active": self._active, "group_precisions": {group: _dtype_name(dtype) for group, dtype in self.precisions.items()},
                "groups": {group: list(names) for group, names in self.groups.items()}, "linear_sites": rows,
                "coverage": {"linear_modules": total, "patched_linear_modules": selected,
                             "linear_module_fraction": selected / total if total else 0.0,
                             "linear_parameters": parameters, "patched_linear_parameters": selected_parameters,
                             "linear_parameter_fraction": selected_parameters / parameters if parameters else 0.0,
                             "meaning": "structural counts; not FLOPs or measured call coverage"},
                "cached_weight_bytes": self._cache_bytes, "checkpoint_tensors_changed": False,
                "accumulator_precision": "backend-dependent; no global flags changed",
                "backward": "autograd of input cast, low-dtype linear, FP32 output cast; frozen weights"}

    def _restore(self, patches: Sequence[_Patch]) -> None:
        for patch in reversed(patches):
            if patch.old_forward is _MISSING:
                patch.module.__dict__.pop("forward", None)
            else:
                patch.module.forward = patch.old_forward
            for key in (_OWNER, _WEIGHT, _BIAS):
                patch.module.__dict__.pop(key, None)
        self.model.__dict__.pop(_OWNER, None)
        self._active = False

    def restore(self) -> None:
        """Restore original forwards; caller must first destroy captured runners."""
        if not self._active:
            return
        # Do not silently overwrite a later patch installed by another owner.
        for patch in self._patches:
            if patch.module.__dict__.get("forward") is not patch.forward:
                raise RuntimeError(f"another patch changed forward; restore it first: {patch.names}")
        self._restore(self._patches)

    def __enter__(self) -> "LayeredPrecisionInstallation":
        self.validate()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.restore()


def apply_layered_precision(model: nn.Module, group_precisions: Mapping[str, str | torch.dtype], *,
                            family: str | None = None,
                            groups: Mapping[str, Sequence[str]] | None = None) -> LayeredPrecisionInstallation:
    """Install a plan immediately, optionally using ``with`` for restoration.

    Omitted groups stay FP32; ``protected`` cannot be lowered. For a custom
    model, supply explicit group-to-module-path lists; omitted paths are
    protected. An all-FP32 plan changes no forwards and is a baseline identity.
    """
    return LayeredPrecisionInstallation(model, group_precisions, family=family, groups=groups)
