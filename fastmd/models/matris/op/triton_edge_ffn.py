"""The line-graph refinement's edge FFN in one kernel per direction (C4, residual round 2).

``Refinement.refinement_edges`` computes ``W2 silu(W1 x)`` over the line-graph rows (``x`` the edge input,
D = 32, frozen weights without bias) before its residual add. Unfused that is two [rows, 32] GEMMs and a
SiLU, each reading and writing [rows, 32]; the forward saves ``h = W1 x`` for the SiLU backward, and the
backward runs two dX GEMMs and the SiLU backward.

Forward, per row tile: both dots and the SiLU in registers; only the output is written and only ``x`` is
saved. Backward, per row tile: ``h`` recomputed from ``x`` (one dot; cheaper than saving and re-reading
it) and ``dx = W1^T (silu'(h) * W2^T dy)`` in the same kernel. The residual add stays the dispatcher's
kernel: folding it in as well measured 0.4 ms faster at 10,752 atoms, but raised the reserved peak under
saved-tensor recompute and the layer checkpoint (up to +6.9 GiB at 21,504 atoms, the allocated peak
unchanged). For the same reason, under saved-tensor recompute the forward allocates, and frees unused, the
unfused layers' two intermediate blocks before its output (see ``_EdgeFFN.forward``). Dot precision: ``InferenceConfig.triton_dot_precision`` (by default 3xTF32 where cuBLAS runs IEEE
FP32, TF32 where it runs TF32). The rows are a runtime argument excluded from Triton's value specialization.
A backward that builds a graph (``create_graph``) takes the same formula in torch ops.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function

from fastmd.engine import recorder as _recorder
from fastmd.engine import saved_recompute as _saved_recompute

from fastmd.precision import triton_dot_input_precision

# (BM rows, num_warps), GB200, 7.6M rows, D = 32.
_FWD_TILE = (128, 8)
_BWD_TILE = (128, 8)


@triton.jit(do_not_specialize=["M"])
def _edge_ffn_fwd(x, w1, w2, y, M, SX, D: tl.constexpr, BM: tl.constexpr, PRECISION: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    live = rows < M
    xt = tl.load(x + rows[:, None] * SX + d[None, :], mask=live[:, None], other=0.0)
    # W [out, in]: the transposes (in, out) for x @ W^T.
    h = tl.dot(xt, tl.load(w1 + d[None, :] * D + d[:, None]), input_precision=PRECISION)
    a = h * tl.sigmoid(h)
    out = tl.dot(a, tl.load(w2 + d[None, :] * D + d[:, None]), input_precision=PRECISION)
    tl.store(y + rows[:, None] * D + d[None, :], out, mask=live[:, None])


@triton.jit(do_not_specialize=["M"])
def _edge_ffn_bwd(x, w1, w2, dy, dx, M, SX, SDY, D: tl.constexpr, BM: tl.constexpr, PRECISION: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    live = rows < M
    xt = tl.load(x + rows[:, None] * SX + d[None, :], mask=live[:, None], other=0.0)
    g = tl.load(dy + rows[:, None] * SDY + d[None, :], mask=live[:, None], other=0.0)
    h = tl.dot(xt, tl.load(w1 + d[None, :] * D + d[:, None]), input_precision=PRECISION)
    s = tl.sigmoid(h)
    da = tl.dot(g, tl.load(w2 + d[:, None] * D + d[None, :]), input_precision=PRECISION)
    dh = da * (s * (1.0 + h * (1.0 - s)))
    tl.store(dx + rows[:, None] * D + d[None, :],
             tl.dot(dh, tl.load(w1 + d[:, None] * D + d[None, :]), input_precision=PRECISION), mask=live[:, None])


def edge_ffn_supported(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> str | None:
    """None if the fused edge FFN runs on these tensors, else the reason it does not."""
    tensors = (x, w1, w2)
    if not all(t.is_cuda and t.device == x.device for t in tensors) or x.device.index != torch.cuda.current_device():
        return "device"
    if not all(t.dtype == torch.float32 for t in tensors) or torch.is_autocast_enabled("cuda"):
        return "dtype"
    if x.dim() != 2 or x.shape[1] != 32 or tuple(w1.shape) != (32, 32) or tuple(w2.shape) != (32, 32):
        return "unsupported_shape"
    if x.stride(1) != 1 or not (w1.is_contiguous() and w2.is_contiguous()):
        return "layout"
    if w1.requires_grad or w2.requires_grad:
        return "trainable_params"
    if triton_dot_input_precision("edge_ffn") is None:
        return "cublas_precision"
    return None


class _EdgeFFN(Function):
    @staticmethod
    def forward(ctx, x, w1, w2, prec):
        rows, dim = x.shape
        slots = _saved_recompute.active() is not None
        if slots:
            # Under saved-tensor recompute: the unfused layers' h and silu(h) blocks, allocated before the
            # output and freed right after (no data written), so the caching allocator keeps the unfused
            # path's block sequence. Without them a later layer's [rows, 64] projection output found no free
            # pair of [rows, 32] blocks and opened a new segment: up to +5.5 GiB reserved at 21,504 atoms
            # (allocated unchanged). Not otherwise: there they raised the layer-checkpoint peaks instead.
            h_slot = torch.empty((rows, dim), device=x.device, dtype=torch.float32)
            a_slot = torch.empty((rows, dim), device=x.device, dtype=torch.float32)
        y = torch.empty((rows, dim), device=x.device, dtype=torch.float32)
        if slots:
            del a_slot, h_slot
        bm, warps = _FWD_TILE
        if rows:
            _edge_ffn_fwd[(triton.cdiv(rows, bm),)](x, w1, w2, y, rows, x.stride(0), D=dim, BM=bm, PRECISION=prec,
                                                    num_warps=warps)
        ctx.save_for_backward(x, w1, w2)
        ctx.precision = prec
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("edge_ffn", "triton", {"rows": rows, "D": dim, "BM": bm, "num_warps": warps,
                                              "PRECISION": prec, "stride": x.stride(0)})
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w1, w2 = ctx.saved_tensors
        if torch.is_grad_enabled():
            # A graph-building backward (create_graph): the same formula, differentiable.
            h = torch.nn.functional.linear(x, w1)
            s = torch.sigmoid(h)
            return (grad_y @ w2) * (s * (1 + h * (1 - s))) @ w1, None, None, None
        rows, dim = x.shape
        if x.stride(1) != 1:  # a rebuilt saved input in another layout
            x = x.contiguous()
        g = grad_y if grad_y.stride(1) == 1 else grad_y.contiguous()
        dx = torch.empty((rows, dim), device=x.device, dtype=torch.float32)
        bm, warps = _BWD_TILE
        if rows:
            _edge_ffn_bwd[(triton.cdiv(rows, bm),)](x, w1, w2, g, dx, rows, x.stride(0), g.stride(0), D=dim, BM=bm,
                                                    PRECISION=ctx.precision, num_warps=warps)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("edge_ffn", "triton", {"rows": rows, "D": dim, "BM": bm, "num_warps": warps,
                                              "PRECISION": ctx.precision, "grad_stride": g.stride(0)},
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return dx, None, None, None


def fused_edge_ffn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, prec: str | None = None) -> torch.Tensor:
    """``W2 silu(W1 x)`` (W [out, in] contiguous); the caller checked ``edge_ffn_supported``. ``prec``: the
    dots' input precision, None: ``triton_dot_input_precision("edge_ffn")``."""
    prec = triton_dot_input_precision("edge_ffn") if prec is None else prec
    if prec is None:
        raise ValueError("the edge FFN site takes cuBLAS at this dot precision (triton_dot_input_precision)")
    return _EdgeFFN.apply(x, w1, w2, prec)
