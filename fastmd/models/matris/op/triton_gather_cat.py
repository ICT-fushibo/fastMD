"""Fused gather+cat helpers for graph-captured interaction blocks.

``out = cat([aligned, gathered[index_a], gathered[index_b]], 1)`` (cat3) and the
4-part variant. The aligned and gathered matrices may be row-strided views with
unit column stride (e.g. ``packed[:, D:2D]`` of ``[R, 3D]``): their row strides are
compile-time keys (fixed per model width and layout; equal to ``D`` for
contiguous callers, whose code generation is unchanged), so they are read in
place instead of being copied. The row count ``E`` is excluded from Triton's
value specialization (it is not read by the kernels), so eager steps with new
row counts reuse the compiled kernel. Row/column addresses are 64-bit.
Under ``private_sink_atomics`` (sink_atomics.py) the VJP's atomics skip rows
whose addends are all zero (``SKIP_ZERO``); outside it the kernels are unchanged.
In deterministic mode (read in the forward) the VJP's sums are ordered
(``ordered_index_add_``) instead of atomic.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function

from fastmd.engine import recorder as _recorder
from fastmd.precision import deterministic_mode
from fastmd.engine.kernels.sink_atomics import zero_skip_atomics
from fastmd.engine.kernels.triton_zero_skip_index import ordered_index_add_


@triton.jit(do_not_specialize=["E"])
def _aligned_gather_cat3_fwd(s0, s1, i1, i2, out, E, D: tl.constexpr, BLK: tl.constexpr,
                             S0_STRIDE: tl.constexpr, S1_STRIDE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLK)
    mask = c < D
    b = tl.load(i1 + row).to(tl.int64)
    d = tl.load(i2 + row).to(tl.int64)
    v0 = tl.load(s0 + row * S0_STRIDE + c, mask=mask, other=0.0)
    v1 = tl.load(s1 + b * S1_STRIDE + c, mask=mask, other=0.0)
    v2 = tl.load(s1 + d * S1_STRIDE + c, mask=mask, other=0.0)
    o = out + row * (3 * D)
    tl.store(o + c, v0, mask=mask)
    tl.store(o + D + c, v1, mask=mask)
    tl.store(o + 2 * D + c, v2, mask=mask)


@triton.jit
def _atomic_row(mask, value, SKIP_ZERO: tl.constexpr):
    # The atomic mask of one row's ``value``. With SKIP_ZERO (design D3) a row
    # whose addends are all zero, e.g. a padding row of a private sink, is not
    # added. The mask stays uniform along the row (vector atomics keep their
    # width); zeros inside a live row are still added, as without SKIP_ZERO,
    # and NaN/Inf compare != 0, so the sums are unchanged.
    if SKIP_ZERO:
        mask = mask & (tl.max((value != 0.0).to(tl.int32), axis=0) != 0)
    return mask


@triton.jit(do_not_specialize=["E"])
def _aligned_gather_cat3_bwd(s0_grad, s1_grad, i1, i2, gout, E, D: tl.constexpr, BLK: tl.constexpr,
                             SKIP_ZERO: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLK)
    mask = c < D
    b = tl.load(i1 + row).to(tl.int64)
    d = tl.load(i2 + row).to(tl.int64)
    go = gout + row * (3 * D)
    g0 = tl.load(go + c, mask=mask, other=0.0)
    g1 = tl.load(go + D + c, mask=mask, other=0.0)
    g2 = tl.load(go + 2 * D + c, mask=mask, other=0.0)
    tl.store(s0_grad + row * D + c, g0, mask=mask)  # empty: never masked by SKIP_ZERO
    tl.atomic_add(s1_grad + b * D + c, g1, mask=_atomic_row(mask, g1, SKIP_ZERO), sem="relaxed")
    tl.atomic_add(s1_grad + d * D + c, g2, mask=_atomic_row(mask, g2, SKIP_ZERO), sem="relaxed")


def _row_view(matrix):
    """``(matrix, row_stride)`` with unit column stride; copies only when needed.

    Row-strided views are read in place. The stride of a single-row matrix is
    never read, so it is normalized to the width (one compiled variant).
    """
    rows, dim = matrix.shape
    if matrix.stride(1) != 1 and dim > 1:
        matrix = matrix.contiguous()
    return matrix, (matrix.stride(0) if rows > 1 else dim)


def _record_row_strides(params, dim, strides):
    # New key only for non-default layouts: contiguous calls record as before.
    if any(stride != dim for stride in strides):
        params["row_stride"] = list(strides)
    return params


def _record_skip_zero(params, skip_zero):
    # New key only under the private sink: P=1 calls record as before.
    if skip_zero:
        params["SKIP_ZERO"] = True
    return params


def _ordered_vjp(ctx, op, grad_out, dim, sums):
    """Deterministic mode: the aligned block's copy and, per ``(gradient, index)``
    of ``sums`` (output blocks 1, 2, ...), its ordered index_add_."""
    grad_aligned = grad_out[:, :dim].contiguous()
    for block, (gradient, index) in enumerate(sums, 1):
        ordered_index_add_(gradient, index, grad_out[:, block * dim:(block + 1) * dim], skip_zero=ctx.skip_zero)
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record(op, "ordered_index_add", {"rows": grad_out.shape[0], "width": dim},
                   phase="bwd", site=getattr(ctx, "_opsel_site", None))
    return grad_aligned


class _AlignedGatherCat3(Function):
    @staticmethod
    def forward(ctx, aligned, gathered, index_a, index_b):
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = [not tensor.is_contiguous() for tensor in (index_a, index_b)]
        inputs = (aligned, gathered)
        aligned, aligned_stride = _row_view(aligned)
        gathered, gathered_stride = _row_view(gathered)
        index_a = index_a.contiguous()
        index_b = index_b.contiguous()
        rows, dim = aligned.shape
        block = triton.next_power_of_2(dim)
        out = torch.empty(rows, 3 * dim, device=aligned.device, dtype=aligned.dtype)
        _aligned_gather_cat3_fwd[(rows,)](
            aligned, gathered, index_a, index_b, out, rows, D=dim, BLK=block,
            S0_STRIDE=aligned_stride, S1_STRIDE=gathered_stride, num_warps=1
        )
        ctx.save_for_backward(index_a, index_b)
        ctx.shapes = (aligned.shape, gathered.shape)
        ctx.skip_zero = zero_skip_atomics()
        ctx.ordered = deterministic_mode()
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("gather_cat3", "triton", _record_row_strides({
                "rows": rows, "width": dim, "source_rows": gathered.shape[0], "BLK": block,
                "num_warps": 1, "dtype": aligned.dtype,
                "index_dtype": [index_a.dtype, index_b.dtype],
                "contiguous_copy": [view is not given for view, given in
                                    zip((aligned, gathered), inputs)] + copied,
            }, dim, (aligned_stride, gathered_stride)))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        index_a, index_b = ctx.saved_tensors
        aligned_shape, gathered_shape = ctx.shapes
        dim = aligned_shape[1]
        block = triton.next_power_of_2(dim)
        if ctx.ordered:
            grad_gathered = torch.zeros(gathered_shape, device=grad_out.device, dtype=grad_out.dtype)
            grad_aligned = _ordered_vjp(ctx, "gather_cat3", grad_out, dim,
                                        ((grad_gathered, index_a), (grad_gathered, index_b)))
            return grad_aligned, grad_gathered, None, None
        grad_aligned = torch.empty(aligned_shape, device=grad_out.device, dtype=grad_out.dtype)
        grad_gathered = torch.zeros(gathered_shape, device=grad_out.device, dtype=grad_out.dtype)
        _aligned_gather_cat3_bwd[(aligned_shape[0],)](
            grad_aligned,
            grad_gathered,
            index_a,
            index_b,
            grad_out.contiguous(),
            aligned_shape[0],
            D=dim,
            BLK=block,
            SKIP_ZERO=ctx.skip_zero,
            num_warps=1,
        )
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("gather_cat3", "triton", _record_skip_zero({
                "rows": aligned_shape[0], "width": dim, "source_rows": gathered_shape[0],
                "BLK": block, "num_warps": 1, "contiguous_copy": not grad_out.is_contiguous(),
            }, ctx.skip_zero), phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return grad_aligned, grad_gathered, None, None


def aligned_gather_cat3(
    aligned: torch.Tensor,
    gathered: torch.Tensor,
    index_a: torch.Tensor,
    index_b: torch.Tensor,
) -> torch.Tensor:
    if aligned.is_cuda and gathered.is_cuda:
        return _AlignedGatherCat3.apply(aligned, gathered, index_a, index_b)
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("gather_cat3", "torch", {"rows": aligned.shape[0], "width": aligned.shape[1]})
    return torch.cat(
        [
            aligned,
            torch.index_select(gathered, 0, index_a),
            torch.index_select(gathered, 0, index_b),
        ],
        dim=1,
    )


@triton.jit(do_not_specialize=["E"])
def _aligned_gather_cat4_fwd(
    s0,
    s1,
    i1,
    s2,
    i2,
    i3,
    out,
    E,
    D: tl.constexpr,
    BLK: tl.constexpr,
    S0_STRIDE: tl.constexpr,
    S1_STRIDE: tl.constexpr,
    S2_STRIDE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLK)
    mask = c < D
    a = tl.load(i1 + row).to(tl.int64)
    b = tl.load(i2 + row).to(tl.int64)
    d = tl.load(i3 + row).to(tl.int64)
    v0 = tl.load(s0 + row * S0_STRIDE + c, mask=mask, other=0.0)
    v1 = tl.load(s1 + a * S1_STRIDE + c, mask=mask, other=0.0)
    v2 = tl.load(s2 + b * S2_STRIDE + c, mask=mask, other=0.0)
    v3 = tl.load(s2 + d * S2_STRIDE + c, mask=mask, other=0.0)
    o = out + row * (4 * D)
    tl.store(o + c, v0, mask=mask)
    tl.store(o + D + c, v1, mask=mask)
    tl.store(o + 2 * D + c, v2, mask=mask)
    tl.store(o + 3 * D + c, v3, mask=mask)


@triton.jit(do_not_specialize=["E"])
def _aligned_gather_cat4_bwd(
    s0_grad,
    s1_grad,
    s2_grad,
    i1,
    i2,
    i3,
    gout,
    E,
    D: tl.constexpr,
    BLK: tl.constexpr,
    SKIP_ZERO: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLK)
    mask = c < D
    a = tl.load(i1 + row).to(tl.int64)
    b = tl.load(i2 + row).to(tl.int64)
    d = tl.load(i3 + row).to(tl.int64)
    go = gout + row * (4 * D)
    g0 = tl.load(go + c, mask=mask, other=0.0)
    g1 = tl.load(go + D + c, mask=mask, other=0.0)
    g2 = tl.load(go + 2 * D + c, mask=mask, other=0.0)
    g3 = tl.load(go + 3 * D + c, mask=mask, other=0.0)
    tl.store(s0_grad + row * D + c, g0, mask=mask)  # empty: never masked by SKIP_ZERO
    tl.atomic_add(s1_grad + a * D + c, g1, mask=_atomic_row(mask, g1, SKIP_ZERO), sem="relaxed")
    tl.atomic_add(s2_grad + b * D + c, g2, mask=_atomic_row(mask, g2, SKIP_ZERO), sem="relaxed")
    tl.atomic_add(s2_grad + d * D + c, g3, mask=_atomic_row(mask, g3, SKIP_ZERO), sem="relaxed")


class _AlignedGatherCat4(Function):
    @staticmethod
    def forward(ctx, aligned, gathered_a, index_a, gathered_b, index_b, index_c):
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = [not tensor.is_contiguous() for tensor in (index_a, index_b, index_c)]
        inputs = (aligned, gathered_a, gathered_b)
        aligned, aligned_stride = _row_view(aligned)
        gathered_a, gathered_a_stride = _row_view(gathered_a)
        gathered_b, gathered_b_stride = _row_view(gathered_b)
        index_a = index_a.contiguous()
        index_b = index_b.contiguous()
        index_c = index_c.contiguous()
        rows, dim = aligned.shape
        block = triton.next_power_of_2(dim)
        out = torch.empty(rows, 4 * dim, device=aligned.device, dtype=aligned.dtype)
        _aligned_gather_cat4_fwd[(rows,)](
            aligned,
            gathered_a,
            index_a,
            gathered_b,
            index_b,
            index_c,
            out,
            rows,
            D=dim,
            BLK=block,
            S0_STRIDE=aligned_stride,
            S1_STRIDE=gathered_a_stride,
            S2_STRIDE=gathered_b_stride,
            num_warps=1,
        )
        ctx.save_for_backward(index_a, index_b, index_c)
        ctx.shapes = (aligned.shape, gathered_a.shape, gathered_b.shape)
        ctx.skip_zero = zero_skip_atomics()
        ctx.ordered = deterministic_mode()
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("gather_cat4", "triton", _record_row_strides({
                "rows": rows, "width": dim,
                "source_rows": [gathered_a.shape[0], gathered_b.shape[0]], "BLK": block,
                "num_warps": 1, "dtype": aligned.dtype,
                "index_dtype": [index_a.dtype, index_b.dtype, index_c.dtype],
                "contiguous_copy": [view is not given for view, given in
                                    zip((aligned, gathered_a, gathered_b), inputs)] + copied,
            }, dim, (aligned_stride, gathered_a_stride, gathered_b_stride)))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        index_a, index_b, index_c = ctx.saved_tensors
        aligned_shape, gathered_a_shape, gathered_b_shape = ctx.shapes
        dim = aligned_shape[1]
        block = triton.next_power_of_2(dim)
        if ctx.ordered:
            grad_gathered_a = torch.zeros(gathered_a_shape, device=grad_out.device, dtype=grad_out.dtype)
            grad_gathered_b = torch.zeros(gathered_b_shape, device=grad_out.device, dtype=grad_out.dtype)
            grad_aligned = _ordered_vjp(ctx, "gather_cat4", grad_out, dim, (
                (grad_gathered_a, index_a), (grad_gathered_b, index_b), (grad_gathered_b, index_c)))
            return grad_aligned, grad_gathered_a, None, grad_gathered_b, None, None
        grad_aligned = torch.empty(aligned_shape, device=grad_out.device, dtype=grad_out.dtype)
        grad_gathered_a = torch.zeros(gathered_a_shape, device=grad_out.device, dtype=grad_out.dtype)
        grad_gathered_b = torch.zeros(gathered_b_shape, device=grad_out.device, dtype=grad_out.dtype)
        _aligned_gather_cat4_bwd[(aligned_shape[0],)](
            grad_aligned,
            grad_gathered_a,
            grad_gathered_b,
            index_a,
            index_b,
            index_c,
            grad_out.contiguous(),
            aligned_shape[0],
            D=dim,
            BLK=block,
            SKIP_ZERO=ctx.skip_zero,
            num_warps=1,
        )
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("gather_cat4", "triton", _record_skip_zero({
                "rows": aligned_shape[0], "width": dim,
                "source_rows": [gathered_a_shape[0], gathered_b_shape[0]], "BLK": block,
                "num_warps": 1, "contiguous_copy": not grad_out.is_contiguous(),
            }, ctx.skip_zero), phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return grad_aligned, grad_gathered_a, None, grad_gathered_b, None, None


def aligned_gather_cat4(
    aligned: torch.Tensor,
    gathered_a: torch.Tensor,
    index_a: torch.Tensor,
    gathered_b: torch.Tensor,
    index_b: torch.Tensor,
    index_c: torch.Tensor,
) -> torch.Tensor:
    if aligned.is_cuda and gathered_a.is_cuda and gathered_b.is_cuda:
        return _AlignedGatherCat4.apply(aligned, gathered_a, index_a, gathered_b, index_b, index_c)
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("gather_cat4", "torch", {"rows": aligned.shape[0], "width": aligned.shape[1]})
    return torch.cat(
        [
            aligned,
            torch.index_select(gathered_a, 0, index_a),
            torch.index_select(gathered_b, 0, index_b),
            torch.index_select(gathered_b, 0, index_c),
        ],
        dim=1,
    )
