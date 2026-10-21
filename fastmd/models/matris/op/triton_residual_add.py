"""Fused residual add for frozen-parameter inference.

Computes ``delta + weight * residual`` for 2D feature tensors.  The wrapper in
``interaction_block`` only uses this when ``weight`` is frozen; backward returns
input gradients for ``delta`` and ``residual`` and no parameter gradient.

The row count is a runtime argument excluded from Triton's value
specialization, so new row counts reuse the compiled kernel. The forward reads
row-strided views (unit column stride, e.g. ``packed[:, :D]``) in place: row
strides are compile-time constants, so a contiguous input compiles exactly the
code it did before; only inputs with a non-unit column stride are copied.
"""
from __future__ import annotations

from fastmd.config import env_value


import torch
import triton
import triton.language as tl
from torch.autograd import Function

from fastmd.engine import recorder as _recorder
from fastmd.engine.kernels.row_count import static_under_compile
from fastmd.engine.kernels.row_tiles import row_tile


def _env_int(name: str, default: int | None) -> int | None:
    value = env_value(name)
    if value is None:
        return default
    return int(value)


# None: by width (row_tiles).
_BM = _env_int("MATRIS_FUSED_RESIDUAL_ADD_BM", None)
_BD = _env_int("MATRIS_FUSED_RESIDUAL_ADD_BD", 128)


@triton.jit(do_not_specialize=["N"])
def _residual_add_fwd(
    delta,
    residual,
    weight,
    out,
    N,
    D: tl.constexpr,
    STRIDE_DELTA: tl.constexpr,
    STRIDE_RESIDUAL: tl.constexpr,
    STRIDE_OUT: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    # Equal strides give one common offset, i.e. exactly the contiguous code.
    offs_delta = rows[:, None] * STRIDE_DELTA + cols[None, :]
    offs_residual = rows[:, None] * STRIDE_RESIDUAL + cols[None, :]
    offs_out = rows[:, None] * STRIDE_OUT + cols[None, :]
    d = tl.load(delta + offs_delta, mask=mask, other=0.0)
    r = tl.load(residual + offs_residual, mask=mask, other=0.0)
    w = tl.load(weight + cols, mask=cols < D, other=0.0)
    tl.store(out + offs_out, d + r * w[None, :], mask=mask)


@triton.jit(do_not_specialize=["N"])
def _residual_add_bwd(
    gout,
    weight,
    grad_residual,
    N,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    offs = rows[:, None] * D + cols[None, :]
    go = tl.load(gout + offs, mask=mask, other=0.0)
    w = tl.load(weight + cols, mask=cols < D, other=0.0)
    tl.store(grad_residual + offs, go * w[None, :], mask=mask)


class _ResidualAdd(Function):
    @staticmethod
    def forward(ctx, delta: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
        n, d = delta.shape
        stride_delta = stride_residual = stride_out = d
        copied = [False, False]
        if not (delta.is_contiguous() and residual.is_contiguous()):
            # Unit column stride: read the row-strided view in place (then
            # n > 1, since a single row with unit column stride is contiguous).
            # Any other layout is copied, as before.
            if not delta.is_contiguous():
                if delta.stride(1) == 1:
                    stride_delta = delta.stride(0)
                else:
                    delta, copied[0] = delta.contiguous(), True
            if not residual.is_contiguous():
                if residual.stride(1) == 1:
                    stride_residual = residual.stride(0)
                else:
                    residual, copied[1] = residual.contiguous(), True
        weight = weight.reshape(-1).contiguous()
        # A row-strided delta is not dense, so its empty_like is contiguous.
        out = torch.empty_like(delta)
        if stride_delta != d and n > 1:
            stride_out = out.stride(0)
        bd = triton.next_power_of_2(min(_BD, d))
        bm = row_tile("residual_add", bd, _BM)
        grid = (triton.cdiv(n, bm), triton.cdiv(d, bd))
        # Positional launch: keyword binding costs ~0.2 us per argument.
        _residual_add_fwd[grid](
            delta, residual, weight, out, static_under_compile(n),
            d, stride_delta, stride_residual, stride_out, bm, bd,  # constexprs
            num_warps=4,
        )
        ctx.save_for_backward(weight)
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
            params = {
                "rows": n, "width": d, "BM": bm, "BD": bd, "num_warps": 4,
                "dtype": delta.dtype, "contiguous_copy": copied,
            }
            if (stride_delta, stride_residual, stride_out) != (d, d, d):
                params["row_stride"] = [stride_delta, stride_residual, stride_out]
            rec.record("residual_add", "triton", params)
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        (weight,) = ctx.saved_tensors
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = not gout.is_contiguous()
        gout = gout.contiguous()
        n, d = gout.shape
        grad_residual = torch.empty_like(gout)
        bd = triton.next_power_of_2(min(_BD, d))
        bm = row_tile("residual_add", bd, _BM)
        grid = (triton.cdiv(n, bm), triton.cdiv(d, bd))
        _residual_add_bwd[grid](
            gout,
            weight,
            grad_residual,
            N=static_under_compile(n),
            D=d,
            BM=bm,
            BD=bd,
            num_warps=4,
        )
        if rec is not None:
            rec.record("residual_add", "triton", {
                "rows": n, "width": d, "BM": bm, "BD": bd, "num_warps": 4,
                "dtype": gout.dtype, "contiguous_copy": copied,
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return gout, grad_residual, None


def residual_add(
    delta: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return _ResidualAdd.apply(delta, residual, weight)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for n, d in [(1024, 128), (13824, 128), (155648, 128)]:
        delta = torch.randn(n, d, device=dev, requires_grad=True)
        residual = torch.randn(n, d, device=dev, requires_grad=True)
        weight = torch.randn(1, d, device=dev)
        d2 = delta.detach().clone().requires_grad_(True)
        r2 = residual.detach().clone().requires_grad_(True)
        ref = delta + weight * residual
        tri = residual_add(d2, r2, weight)
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        torch.cuda.synchronize()
        print(
            f"N={n}: fwd={(ref - tri).abs().max().item():.2e} "
            f"bwd={max((delta.grad - d2.grad).abs().max().item(), (residual.grad - r2.grad).abs().max().item()):.2e}"
        )
