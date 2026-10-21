"""torch.compile-generated lowerings behind the topology-contract interface.

The lowering registry is kernel-source-agnostic: a lowering may be a
hand-written Triton kernel, a generic eager chain, or — here — a pair of
Inductor-generated kernels produced by ``torch.compile`` from the idiomatic
formula. Each compiled lowering follows the repository's standard custom
autograd Function idiom: an explicit forward formula and an explicit VJP
formula, each compiled as a forward-only artifact. This keeps AOTAutograd
out of the captured region and matches the forward/VJP contract every other
lowering obeys.

Capture discipline (same as the merged-weight caches): artifacts compile and
warm on the first eager call; inside CUDA-graph capture a cold artifact
returns None and the caller falls back, so an unwarmed compiled path can
never be baked into a graph implicitly. ``mode="max-autotune-no-cudagraphs"``
keeps Inductor's cudagraph trees out of the externally captured stream.
"""

from __future__ import annotations

from fastmd.config import env_value


import torch
from torch.autograd import Function

from fastmd.engine import recorder as _recorder


def _env_flag(name: str, default: bool) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "off", "no"}


COMPILED_LOWERINGS_ENABLED = _env_flag("MATRIS_COMPILED_LOWERINGS", False)
_COMPILE_MODE = env_value(
    "MATRIS_COMPILED_LOWERINGS_MODE", "max-autotune-no-cudagraphs"
)


def _compile(fn):
    # dynamic=True: one symbolic-shape artifact covers the ragged eager/setup
    # steps (T drifts per MD step) and the padded captured shape alike. The
    # historical dynamic-compile regression was whole-model recompilation and
    # graph breaks; a single fullgraph formula is Inductor's easy case.
    return torch.compile(fn, mode=_COMPILE_MODE, dynamic=True, fullgraph=True)


class _ArtifactPair:
    """Lazily compiled (forward, vjp) artifacts with capture-cold fallback."""

    def __init__(self, fwd_formula, vjp_formula) -> None:
        self._fwd_formula = fwd_formula
        self._vjp_formula = vjp_formula
        # A process may serve several scoped inference configurations. Never
        # reuse an artifact compiled with a different requested compile mode.
        self._compiled = {}

    def get(self):
        mode = _COMPILE_MODE
        if mode not in self._compiled:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return None
            self._compiled[mode] = (
                _compile(self._fwd_formula),
                _compile(self._vjp_formula),
            )
        return self._compiled[mode]


# ---------------------------------------------------------------- line envelope

def _envelope_fwd_formula(base, source_index, target_index):
    return torch.index_select(base, 0, source_index) * torch.index_select(
        base, 0, target_index
    )


def _envelope_vjp_formula(base, source_index, target_index, grad_output):
    grad_base = torch.zeros_like(base)
    grad_base.index_add_(
        0, source_index, torch.index_select(base, 0, target_index) * grad_output
    )
    grad_base.index_add_(
        0, target_index, torch.index_select(base, 0, source_index) * grad_output
    )
    return grad_base


_ENVELOPE_ARTIFACTS = _ArtifactPair(_envelope_fwd_formula, _envelope_vjp_formula)


class _CompiledLineEnvelope(Function):
    @staticmethod
    def forward(ctx, base, source_index, target_index):
        artifacts = _ENVELOPE_ARTIFACTS.get()
        assert artifacts is not None
        ctx.save_for_backward(base, source_index, target_index)
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("line_envelope", "compiled", {
                "rows": source_index.numel(), "width": base.shape[1], "mode": _COMPILE_MODE,
                "dtype": base.dtype, "index_dtype": [source_index.dtype, target_index.dtype],
            })
        return artifacts[0](base, source_index, target_index)

    @staticmethod
    def backward(ctx, grad_output):
        base, source_index, target_index = ctx.saved_tensors
        artifacts = _ENVELOPE_ARTIFACTS.get()
        assert artifacts is not None
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("line_envelope", "compiled", {
                "rows": source_index.numel(), "width": base.shape[1], "mode": _COMPILE_MODE,
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        grad_base = artifacts[1](
            base, source_index, target_index, grad_output.contiguous()
        )
        return grad_base, None, None


def compiled_line_envelope(base, source_index, target_index):
    """Inductor-generated line-envelope product, or None to fall back."""
    if not (
        COMPILED_LOWERINGS_ENABLED
        and base.is_cuda
        and base.dim() == 2
        and base.is_contiguous()
        and source_index.numel() == target_index.numel()
    ):
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.fallback("line_envelope", "compiled",
                         "tensor_guard" if COMPILED_LOWERINGS_ENABLED else "disabled")
        return None
    if _ENVELOPE_ARTIFACTS.get() is None:
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.fallback("line_envelope", "compiled", "cold_under_capture")
        return None
    return _CompiledLineEnvelope.apply(base, source_index, target_index)


# ---------------------------------------------------------------- residual add

def _residual_fwd_formula(delta, residual, weight):
    return delta + weight * residual


def _residual_vjp_formula(grad_output, weight):
    return grad_output * weight


_RESIDUAL_ARTIFACTS = _ArtifactPair(_residual_fwd_formula, _residual_vjp_formula)


class _CompiledResidualAdd(Function):
    @staticmethod
    def forward(ctx, delta, residual, weight):
        artifacts = _RESIDUAL_ARTIFACTS.get()
        assert artifacts is not None
        ctx.save_for_backward(weight)
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("residual_add", "compiled", {
                "rows": delta.shape[0], "width": delta.shape[1], "mode": _COMPILE_MODE,
                "dtype": delta.dtype,
            })
        return artifacts[0](delta, residual, weight)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        artifacts = _RESIDUAL_ARTIFACTS.get()
        assert artifacts is not None
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("residual_add", "compiled", {
                "rows": grad_output.shape[0], "width": grad_output.shape[1],
                "mode": _COMPILE_MODE,
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        # Same contract as the Triton lowering: no weight gradient
        # (inference-frozen weight, guarded by the caller).
        return grad_output, artifacts[1](grad_output, weight), None


def compiled_residual_add(delta, residual, weight):
    """Inductor-generated residual add, or None to fall back."""
    if not (
        COMPILED_LOWERINGS_ENABLED
        and delta.is_cuda
        and delta.dim() == 2
        and residual.shape == delta.shape
        and delta.shape[1] == weight.numel()
        and delta.dtype == residual.dtype == weight.dtype
        and not weight.requires_grad
    ):
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.fallback("residual_add", "compiled",
                         "tensor_guard" if COMPILED_LOWERINGS_ENABLED else "disabled")
        return None
    if _RESIDUAL_ARTIFACTS.get() is None:
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.fallback("residual_add", "compiled", "cold_under_capture")
        return None
    return _CompiledResidualAdd.apply(delta, residual, weight.reshape(1, -1))
