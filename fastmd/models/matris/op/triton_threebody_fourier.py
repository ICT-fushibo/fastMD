"""Fused Fourier three-body basis for line-graph angles.

Computes the current PyTorch sequence:

    edge_i = unit_edge_vectors[target_index]
    edge_j = unit_edge_vectors[source_index]
    angle = acos(sum(edge_i * edge_j) * (1 - 1e-6))
    basis = [1/sqrt(2), sin(angle * f), cos(angle * f)] / sqrt(pi)

The Fourier frequencies must be frozen; backward returns gradients only for
``unit_edge_vectors``. The row (angle) count is a runtime argument excluded from
Triton's value specialization, so new row counts reuse the compiled kernels.
Under ``private_sink_atomics`` (sink_atomics.py) the backward's atomics skip
zero addends (``SKIP_ZERO``); outside it the kernels are unchanged. In
deterministic mode (read in the forward) the backward runs in torch with
ordered sums (``ordered_index_add_``).
"""
from __future__ import annotations

from fastmd.config import env_value


import torch
import triton
import triton.language as tl
from torch.autograd import Function
from triton.language.extra import libdevice

from fastmd.engine import recorder as _recorder
from fastmd.precision import deterministic_mode
from fastmd.engine.kernels.row_count import static_under_compile
from fastmd.engine.kernels.sink_atomics import zero_skip_atomics
from fastmd.engine.kernels.triton_zero_skip_index import ordered_index_add_


def _env_int(name: str, default: int) -> int:
    value = env_value(name)
    if value is None:
        return default
    return int(value)


_BM = _env_int("MATRIS_THREEBODY_FOURIER_BM", 128)
_EPS_SCALE = 1.0 - 1.0e-6


@triton.jit(do_not_specialize=["N"])
def _threebody_fourier_fwd(
    unit_vec,
    target_index,
    source_index,
    freqs,
    out,
    N,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    mask = rows < N
    tgt = tl.load(target_index + rows, mask=mask, other=0).to(tl.int64)
    src = tl.load(source_index + rows, mask=mask, other=0).to(tl.int64)

    ti0 = tl.load(unit_vec + tgt * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    ti1 = tl.load(unit_vec + tgt * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    ti2 = tl.load(unit_vec + tgt * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    sj0 = tl.load(unit_vec + src * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    sj1 = tl.load(unit_vec + src * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    sj2 = tl.load(unit_vec + src * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    theta = (ti0 * sj0 + ti1 * sj1 + ti2 * sj2) * 0.999999
    angle = libdevice.acos(theta)
    inv_scale = 1.0 / SCALE
    base = out + rows * 7
    tl.store(base + 0, 0.7071067811865476 * inv_scale, mask=mask)

    f0 = tl.load(freqs + 0).to(tl.float32)
    f1 = tl.load(freqs + 1).to(tl.float32)
    f2 = tl.load(freqs + 2).to(tl.float32)
    a0 = angle * f0
    a1 = angle * f1
    a2 = angle * f2
    tl.store(base + 1, libdevice.sin(a0) * inv_scale, mask=mask)
    tl.store(base + 2, libdevice.sin(a1) * inv_scale, mask=mask)
    tl.store(base + 3, libdevice.sin(a2) * inv_scale, mask=mask)
    tl.store(base + 4, libdevice.cos(a0) * inv_scale, mask=mask)
    tl.store(base + 5, libdevice.cos(a1) * inv_scale, mask=mask)
    tl.store(base + 6, libdevice.cos(a2) * inv_scale, mask=mask)


@triton.jit
def _atomic_mask(mask, value, SKIP_ZERO: tl.constexpr):
    # With SKIP_ZERO (design D3) zero addends, e.g. of the padding triplets of a
    # private sink, are not added; NaN/Inf compare != 0, so the sums are unchanged.
    if SKIP_ZERO:
        mask = mask & (value != 0.0)
    return mask


@triton.jit(do_not_specialize=["N"])
def _threebody_fourier_bwd(
    gout,
    unit_vec,
    target_index,
    source_index,
    freqs,
    grad_unit,
    N,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    SKIP_ZERO: tl.constexpr = False,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    mask = rows < N
    tgt = tl.load(target_index + rows, mask=mask, other=0).to(tl.int64)
    src = tl.load(source_index + rows, mask=mask, other=0).to(tl.int64)

    ti0 = tl.load(unit_vec + tgt * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    ti1 = tl.load(unit_vec + tgt * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    ti2 = tl.load(unit_vec + tgt * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    sj0 = tl.load(unit_vec + src * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    sj1 = tl.load(unit_vec + src * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    sj2 = tl.load(unit_vec + src * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    theta = (ti0 * sj0 + ti1 * sj1 + ti2 * sj2) * 0.999999
    angle = libdevice.acos(theta)
    inv_scale = 1.0 / SCALE
    base = gout + rows * 7

    f0 = tl.load(freqs + 0).to(tl.float32)
    f1 = tl.load(freqs + 1).to(tl.float32)
    f2 = tl.load(freqs + 2).to(tl.float32)
    a0 = angle * f0
    a1 = angle * f1
    a2 = angle * f2
    gsin0 = tl.load(base + 1, mask=mask, other=0.0).to(tl.float32)
    gsin1 = tl.load(base + 2, mask=mask, other=0.0).to(tl.float32)
    gsin2 = tl.load(base + 3, mask=mask, other=0.0).to(tl.float32)
    gcos0 = tl.load(base + 4, mask=mask, other=0.0).to(tl.float32)
    gcos1 = tl.load(base + 5, mask=mask, other=0.0).to(tl.float32)
    gcos2 = tl.load(base + 6, mask=mask, other=0.0).to(tl.float32)

    dangle = (
        gsin0 * libdevice.cos(a0) * f0
        + gsin1 * libdevice.cos(a1) * f1
        + gsin2 * libdevice.cos(a2) * f2
        - gcos0 * libdevice.sin(a0) * f0
        - gcos1 * libdevice.sin(a1) * f1
        - gcos2 * libdevice.sin(a2) * f2
    ) * inv_scale
    denom = tl.sqrt(tl.maximum(1.0 - theta * theta, 1.0e-24))
    dtheta = -dangle / denom
    ddot = dtheta * 0.999999

    gt0 = ddot * sj0
    gt1 = ddot * sj1
    gt2 = ddot * sj2
    gs0 = ddot * ti0
    gs1 = ddot * ti1
    gs2 = ddot * ti2

    tl.atomic_add(grad_unit + tgt * 3 + 0, gt0, mask=_atomic_mask(mask, gt0, SKIP_ZERO), sem="relaxed")
    tl.atomic_add(grad_unit + tgt * 3 + 1, gt1, mask=_atomic_mask(mask, gt1, SKIP_ZERO), sem="relaxed")
    tl.atomic_add(grad_unit + tgt * 3 + 2, gt2, mask=_atomic_mask(mask, gt2, SKIP_ZERO), sem="relaxed")
    tl.atomic_add(grad_unit + src * 3 + 0, gs0, mask=_atomic_mask(mask, gs0, SKIP_ZERO), sem="relaxed")
    tl.atomic_add(grad_unit + src * 3 + 1, gs1, mask=_atomic_mask(mask, gs1, SKIP_ZERO), sem="relaxed")
    tl.atomic_add(grad_unit + src * 3 + 2, gs2, mask=_atomic_mask(mask, gs2, SKIP_ZERO), sem="relaxed")


def _ordered_backward(gout, unit_vec, target_index, source_index, freqs, scale, skip_zero):
    """Deterministic mode: the kernel's VJP in torch (same formulas, FP32) with
    the per-angle addends summed by ordered index_add_ instead of atomics."""
    edge_i = unit_vec.index_select(0, target_index)
    edge_j = unit_vec.index_select(0, source_index)
    theta = torch.sum(edge_i * edge_j, dim=1) * _EPS_SCALE
    phase = torch.outer(torch.acos(theta), freqs)
    dangle = torch.sum((gout[:, 1:4] * torch.cos(phase) - gout[:, 4:7] * torch.sin(phase)) * freqs, dim=1) / scale
    ddot = (-dangle / torch.sqrt(torch.clamp(1.0 - theta * theta, min=1.0e-24)) * _EPS_SCALE)[:, None]
    grad_unit = torch.zeros_like(unit_vec)
    ordered_index_add_(grad_unit, target_index, ddot * edge_j, skip_zero=skip_zero)
    ordered_index_add_(grad_unit, source_index, ddot * edge_i, skip_zero=skip_zero)
    return grad_unit


class _ThreebodyFourierBasis(Function):
    @staticmethod
    def forward(
        ctx,
        unit_vec: torch.Tensor,
        target_index: torch.Tensor,
        source_index: torch.Tensor,
        freqs: torch.Tensor,
        scale: float,
    ):
        if unit_vec.dim() != 2 or unit_vec.shape[1] != 3:
            raise ValueError(f"expected unit_vec [E,3], got {unit_vec.shape}")
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = [not tensor.is_contiguous() for tensor in
                      (unit_vec, target_index, source_index, freqs)]
        unit_vec = unit_vec.contiguous()
        target_index = target_index.contiguous()
        source_index = source_index.contiguous()
        freqs = freqs.contiguous()
        n = target_index.numel()
        out = torch.empty((n, 7), device=unit_vec.device, dtype=unit_vec.dtype)
        grid = (triton.cdiv(n, _BM),)
        _threebody_fourier_fwd[grid](
            unit_vec,
            target_index,
            source_index,
            freqs,
            out,
            N=static_under_compile(n),
            SCALE=float(scale),
            BM=_BM,
            num_warps=4,
        )
        ctx.save_for_backward(unit_vec, target_index, source_index, freqs)
        ctx.scale = float(scale)
        ctx.skip_zero = zero_skip_atomics()
        ctx.ordered = deterministic_mode()
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("threebody_fourier_basis", "triton", {
                "rows": n, "vectors": unit_vec.shape[0], "BM": _BM, "num_warps": 4,
                "SCALE": float(scale), "dtype": unit_vec.dtype,
                "index_dtype": [target_index.dtype, source_index.dtype], "contiguous_copy": copied,
            })
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        unit_vec, target_index, source_index, freqs = ctx.saved_tensors
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = not gout.is_contiguous()
        gout = gout.contiguous()
        n = target_index.numel()
        if ctx.ordered:
            grad_unit = _ordered_backward(gout, unit_vec, target_index, source_index, freqs, ctx.scale,
                                          ctx.skip_zero)
            if rec is not None:
                rec.record("threebody_fourier_basis", "ordered_index_add", {
                    "rows": n, "vectors": unit_vec.shape[0]}, phase="bwd", site=getattr(ctx, "_opsel_site", None))
            return grad_unit, None, None, None, None
        grad_unit = torch.zeros_like(unit_vec)
        grid = (triton.cdiv(n, _BM),)
        _threebody_fourier_bwd[grid](
            gout,
            unit_vec,
            target_index,
            source_index,
            freqs,
            grad_unit,
            N=static_under_compile(n),
            SCALE=ctx.scale,
            BM=_BM,
            SKIP_ZERO=ctx.skip_zero,
            num_warps=4,
        )
        if rec is not None:
            params = {
                "rows": n, "vectors": unit_vec.shape[0], "BM": _BM, "num_warps": 4,
                "SCALE": ctx.scale, "contiguous_copy": copied,
            }
            if ctx.skip_zero:  # new key only under the private sink: P=1 records as before
                params["SKIP_ZERO"] = True
            rec.record("threebody_fourier_basis", "triton", params,
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return grad_unit, None, None, None, None


def threebody_fourier_basis(
    unit_vec: torch.Tensor,
    target_index: torch.Tensor,
    source_index: torch.Tensor,
    freqs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if unit_vec.is_cuda and target_index.is_cuda and source_index.is_cuda and freqs.is_cuda:
        return _ThreebodyFourierBasis.apply(
            unit_vec, target_index, source_index, freqs, float(scale)
        )
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("threebody_fourier_basis", "torch", {
            "rows": target_index.numel(), "vectors": unit_vec.shape[0],
        })
    edge_i = torch.index_select(unit_vec, 0, target_index)
    edge_j = torch.index_select(unit_vec, 0, source_index)
    theta = torch.sum(edge_i * edge_j, dim=1) * _EPS_SCALE
    angle = torch.acos(theta)
    result = angle.new_empty((angle.shape[0], 7))
    result[:, 0] = 2**-0.5
    tmp = torch.outer(angle, freqs)
    result[:, 1:4] = torch.sin(tmp)
    result[:, 4:7] = torch.cos(tmp)
    return result / float(scale)


if __name__ == "__main__":
    # The kernel and the eager reference both compute in FP32, and
    # acos'(t) = -1/sqrt(1 - t^2) is ill-conditioned for near-(anti)parallel pairs:
    # at N=152446 random pairs reach 1 - |t| ~ 3e-5, where FP32 rounding of t alone
    # moves a gradient by ~1e-2. Both are therefore measured against FP64 (bwd
    # relative to the largest gradient): the kernel must match eager FP32 accuracy,
    # and FP32 tolerance on well-conditioned pairs (|t| < 0.99).
    torch.manual_seed(0)
    dev = "cuda"
    freqs = torch.tensor([0.833447, 1.5861181, 2.5042067], device=dev)
    scale = 3.141592653589793**0.5

    def reference(unit, target, source):
        edge_i = torch.index_select(unit, 0, target)
        edge_j = torch.index_select(unit, 0, source)
        theta = torch.sum(edge_i * edge_j, dim=1) * _EPS_SCALE
        angle = torch.acos(theta)
        ref = angle.new_empty((angle.shape[0], 7))
        ref[:, 0] = 2**-0.5
        tmp = torch.outer(angle, freqs.to(angle.dtype))
        ref[:, 1:4] = torch.sin(tmp)
        ref[:, 4:7] = torch.cos(tmp)
        return ref / scale

    def errors(unit, target, source, grad):
        results = {}
        for name, fn, dtype in (
            ("fp64", reference, torch.float64),
            ("eager", reference, torch.float32),
            ("triton", lambda u, t, s: threebody_fourier_basis(u, t, s, freqs, scale), torch.float32),
        ):
            u = unit.detach().to(dtype, copy=True).requires_grad_(True)
            out = fn(u, target, source)
            out.backward(grad.to(dtype))
            results[name] = (out.detach().double(), u.grad.double())
        out64, grad64 = results.pop("fp64")
        return {
            name: ((out - out64).abs().max().item(),
                   ((g - grad64).abs().max() / grad64.abs().max()).item())
            for name, (out, g) in results.items()
        }

    for n, e in [(1024, 256), (152446, 26174)]:
        unit = torch.nn.functional.normalize(torch.randn(e, 3, device=dev), dim=1)
        target = torch.randint(0, e, (n,), device=dev)
        source = torch.randint(0, e, (n,), device=dev)
        source = torch.where(source == target, (source + 1) % e, source)
        grad = torch.randn(n, 7, device=dev)
        err = errors(unit, target, source, grad)
        well = (unit[target] * unit[source]).sum(1).abs() < 0.99
        err_well = errors(unit, target[well], source[well], grad[well])
        print(
            f"N={n} E={e} vs FP64: "
            f"triton fwd={err['triton'][0]:.2e} bwd={err['triton'][1]:.2e}, "
            f"eager FP32 fwd={err['eager'][0]:.2e} bwd={err['eager'][1]:.2e}; "
            f"|t|<0.99 triton fwd={err_well['triton'][0]:.2e} bwd={err_well['triton'][1]:.2e}"
        )
        assert err["triton"][0] <= 2 * err["eager"][0] + 1e-6
        assert err["triton"][1] <= 4 * err["eager"][1] + 1e-5
        assert err_well["triton"][0] < 5e-6 and err_well["triton"][1] < 1e-5
