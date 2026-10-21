"""Two-part segment attention: per-part (log-sum-exp, weighted mean) and the merge.

The A2 bond-partial exchange (``fastmd.models.matris.distributed.bond_partial``) splits the
segment softmax of a boundary bond between the two ranks that own its ends:
each end reduces its own triplet rows to a partial ``(lse, y)`` per channel
(``y`` the softmax-weighted mean over those rows, ``lse`` their log-sum-exp;
an empty part is ``(-inf, 0)``), the ends swap partials, and both evaluate
``merge_lse_partials`` on the same two operands in canonical order (the lower
rank's partial first). The merge is exact in real arithmetic: with
``s_k = exp(lse_k - m)`` for any shift ``m``, the softmax-weighted mean over
the union of both parts is ``y = (s_1 y_1 + s_2 y_2) / (s_1 + s_2)`` (the
implementation shifts by ``max(lse_1, lse_2)``; ``y = 0`` when both parts
are empty). Evaluated from the same operands in the same order, it is
bitwise equal on both ends.

Empty parts are guarded inside the formulas, never with a where() around a
0/0 (its VJP would still evaluate the NaN), and the stabilizing shifts are
stop-gradient: the merged value does not depend on them.

Layout of a merge operand: ``[rows, 2 * width * groups]``, one ``(lse, y)``
pair of ``width`` channels per group, e.g. ``(lse_t, y_t, lse_s, y_s)`` for
the target and source sides of the line attention; the result is
``[rows, width * groups]`` (``y_0, y_1, ...``). ``first_is_own`` lets a
caller pass ``(own, peer)`` and a per-row bool (own is the canonical first
operand) instead of forming the canonical pair with two ``where`` passes.

On CUDA, float32 operands take one fused Triton kernel per direction. The
forward evaluates the torch expression in the same order with the same
roundings (fp contraction off, libdevice ``exp``, IEEE division): it matched
the eager CUDA ``merge_lse_partials_torch`` bit for bit on GB300, and it
depends only on the canonical operands, not on which of them is own. The
backward is
``d/dy_k = w_k``, ``d/dlse_1 = -d/dlse_2 = w_1 w_2 (y_1 - y_2)`` with the
weights ``w_k = s_k / total`` (finite, 0 for an empty part); under
``create_graph`` it runs the same formula in torch ops, so it stays
differentiable.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import Function
import triton
import triton.language as tl
from triton.language.extra import libdevice

from fastmd.engine import recorder as _recorder

_MERGE_BLOCK = 1024
_MERGE_NUM_WARPS = 4


def segment_logsumexp(values: Tensor, segment: Tensor, count: int) -> Tensor:
    """Per-channel log-sum-exp of the rows of ``values`` per segment; -inf where empty.

    ``segment[k]`` (int64) is the segment, in ``[0, count)``, of row ``k``.
    """
    width = values.shape[1]
    with torch.no_grad():
        peak = values.new_full((count, width), float("-inf")).scatter_reduce(
            0, segment[:, None].expand(-1, width), values, reduce="amax", include_self=False)
        peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    total = values.new_zeros((count, width)).index_add(
        0, segment, torch.exp(values - peak.index_select(0, segment)))
    positive = total > 0
    return torch.where(positive, peak + torch.log(torch.where(positive, total, torch.ones_like(total))),
                       torch.full_like(total, float("-inf")))


def slot_logsumexp(logits: Tensor, rows: Tensor, segment: Tensor, count: int) -> Tensor:
    """Per-channel log-sum-exp of ``logits[rows]`` per segment; -inf where empty.

    ``segment[k]`` is the segment (in ``[0, count)``) of row ``rows[k]``.
    """
    return segment_logsumexp(logits.index_select(0, rows), segment, count)


# --- the merge ---------------------------------------------------------------


def _canonical(first: Tensor, second: Tensor, first_is_own: Tensor | None):
    """(own, peer) -> (first, second) per row; its own inverse."""
    if first_is_own is None:
        return first, second
    own_first = first_is_own[:, None]
    return torch.where(own_first, first, second), torch.where(own_first, second, first)


def _split(part: Tensor, width: int):
    """``[rows, 2 * width * groups]`` -> (lse, y), each ``[rows, groups, width]``."""
    lse, y = part.unflatten(1, (-1, 2, width)).unbind(2)
    return lse, y


def _scales(first_lse: Tensor, second_lse: Tensor):
    """exp(lse - shift) of both parts and their guarded sum (>= 1 unless both empty)."""
    with torch.no_grad():
        shift = torch.maximum(first_lse, second_lse)
        shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    first_scale = torch.exp(first_lse - shift)
    second_scale = torch.exp(second_lse - shift)
    total = first_scale + second_scale
    return first_scale, second_scale, torch.where(total > 0, total, torch.ones_like(total))


def merge_lse_partials_torch(first: Tensor, second: Tensor, width: int, *,
                             first_is_own: Tensor | None = None) -> Tensor:
    """Reference and fallback of :func:`merge_lse_partials`. Separate eager
    kernels: no FMA contraction reorders the two terms."""
    first, second = _canonical(first, second, first_is_own)
    (first_lse, first_y), (second_lse, second_y) = _split(first, width), _split(second, width)
    first_scale, second_scale, total = _scales(first_lse, second_lse)
    return ((first_y * first_scale + second_y * second_scale) / total).flatten(1)


def _merge_vjp_torch(first, second, first_is_own, grad, width):
    """The fused backward's formula in differentiable torch ops (create_graph)."""
    first, second = _canonical(first, second, first_is_own)
    (first_lse, first_y), (second_lse, second_y) = _split(first, width), _split(second, width)
    first_scale, second_scale, total = _scales(first_lse, second_lse)
    first_weight, second_weight = first_scale / total, second_scale / total
    grad = grad.unflatten(1, (-1, width))
    grad_lse = grad * (first_weight * second_weight * (first_y - second_y))
    grad_first = torch.stack((grad_lse, grad * first_weight), 2).flatten(1)
    grad_second = torch.stack((-grad_lse, grad * second_weight), 2).flatten(1)
    return _canonical(grad_first, grad_second, first_is_own)


@triton.jit
def _merge_scales(lse_1, lse_2):
    # _scales: stop-gradient shift, 0 when both parts are empty (-inf).
    shift = tl.maximum(lse_1, lse_2, propagate_nan=tl.PropagateNan.ALL)
    shift = tl.where((shift > -float("inf")) & (shift < float("inf")), shift, 0.0)
    scale_1 = libdevice.exp(lse_1 - shift)
    scale_2 = libdevice.exp(lse_2 - shift)
    total = scale_1 + scale_2
    return scale_1, scale_2, tl.where(total > 0.0, total, 1.0)


@triton.jit(do_not_specialize=["elements"])
def _merge_lse_partials_fwd(
    first,
    second,
    first_is_own,
    out,
    elements,
    FIRST_S0: tl.constexpr,
    SECOND_S0: tl.constexpr,
    W: tl.constexpr,
    GW: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ORDER: tl.constexpr,
):
    # One element per (row, group, channel), 64-bit like the attention kernels.
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < elements
    row = idx // GW
    col = idx % GW
    lse_col = (col // W) * (2 * W) + col % W
    lse_f = tl.load(first + row * FIRST_S0 + lse_col, mask=mask, other=0.0)
    y_f = tl.load(first + row * FIRST_S0 + lse_col + W, mask=mask, other=0.0)
    lse_s = tl.load(second + row * SECOND_S0 + lse_col, mask=mask, other=0.0)
    y_s = tl.load(second + row * SECOND_S0 + lse_col + W, mask=mask, other=0.0)
    if HAS_ORDER:
        own = tl.load(first_is_own + row, mask=mask, other=1) != 0
        lse_1 = tl.where(own, lse_f, lse_s)
        y_1 = tl.where(own, y_f, y_s)
        lse_2 = tl.where(own, lse_s, lse_f)
        y_2 = tl.where(own, y_s, y_f)
    else:
        lse_1, y_1, lse_2, y_2 = lse_f, y_f, lse_s, y_s
    scale_1, scale_2, total = _merge_scales(lse_1, lse_2)
    # First operand's term first; launched with fp fusion off, so neither
    # product is contracted into the sum (an FMA would be order-dependent).
    tl.store(out + idx, tl.div_rn(y_1 * scale_1 + y_2 * scale_2, total), mask=mask)


@triton.jit(do_not_specialize=["elements"])
def _merge_lse_partials_bwd(
    first,
    second,
    first_is_own,
    gout,
    grad_first,
    grad_second,
    elements,
    FIRST_S0: tl.constexpr,
    SECOND_S0: tl.constexpr,
    W: tl.constexpr,
    GW: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ORDER: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < elements
    row = idx // GW
    col = idx % GW
    lse_col = (col // W) * (2 * W) + col % W
    lse_f = tl.load(first + row * FIRST_S0 + lse_col, mask=mask, other=0.0)
    y_f = tl.load(first + row * FIRST_S0 + lse_col + W, mask=mask, other=0.0)
    lse_s = tl.load(second + row * SECOND_S0 + lse_col, mask=mask, other=0.0)
    y_s = tl.load(second + row * SECOND_S0 + lse_col + W, mask=mask, other=0.0)
    if HAS_ORDER:
        own = tl.load(first_is_own + row, mask=mask, other=1) != 0
        lse_1 = tl.where(own, lse_f, lse_s)
        y_1 = tl.where(own, y_f, y_s)
        lse_2 = tl.where(own, lse_s, lse_f)
        y_2 = tl.where(own, y_s, y_f)
    else:
        lse_1, y_1, lse_2, y_2 = lse_f, y_f, lse_s, y_s
    scale_1, scale_2, total = _merge_scales(lse_1, lse_2)
    weight_1 = tl.div_rn(scale_1, total)
    weight_2 = tl.div_rn(scale_2, total)
    g = tl.load(gout + idx, mask=mask, other=0.0)
    # An empty part has weight 0: its (lse, y) and the other part's lse get 0.
    g_lse_1 = g * (weight_1 * weight_2 * (y_1 - y_2))
    g_lse_2 = -g_lse_1
    g_y_1 = g * weight_1
    g_y_2 = g * weight_2
    if HAS_ORDER:  # back to (own, peer)
        g_lse_f = tl.where(own, g_lse_1, g_lse_2)
        g_y_f = tl.where(own, g_y_1, g_y_2)
        g_lse_s = tl.where(own, g_lse_2, g_lse_1)
        g_y_s = tl.where(own, g_y_2, g_y_1)
    else:
        g_lse_f, g_y_f, g_lse_s, g_y_s = g_lse_1, g_y_1, g_lse_2, g_y_2
    offs = row * (2 * GW) + lse_col
    tl.store(grad_first + offs, g_lse_f, mask=mask)
    tl.store(grad_first + offs + W, g_y_f, mask=mask)
    tl.store(grad_second + offs, g_lse_s, mask=mask)
    tl.store(grad_second + offs + W, g_y_s, mask=mask)


def _launch(kernel, first, second, first_is_own, width, *tensors):
    rows, packed = first.shape
    groups_width = packed // 2
    elements = rows * groups_width
    if elements:
        kernel[(triton.cdiv(elements, _MERGE_BLOCK),)](
            first, second, None if first_is_own is None else first_is_own.view(torch.uint8),
            *tensors, elements,
            FIRST_S0=first.stride(0), SECOND_S0=second.stride(0), W=width, GW=groups_width,
            BLOCK=_MERGE_BLOCK, HAS_ORDER=first_is_own is not None,
            num_warps=_MERGE_NUM_WARPS, enable_fp_fusion=False)


def _params(first, width, first_is_own):
    return {"rows": first.shape[0], "width": width, "groups": first.shape[1] // (2 * width),
            "first_is_own": first_is_own is not None}


class _MergeLsePartials(Function):
    @staticmethod
    def forward(ctx, first, second, width, first_is_own):
        first = first if first.stride(1) == 1 else first.contiguous()
        second = second if second.stride(1) == 1 else second.contiguous()
        first_is_own = None if first_is_own is None else first_is_own.contiguous()
        out = torch.empty((first.shape[0], first.shape[1] // 2), device=first.device, dtype=first.dtype)
        _launch(_merge_lse_partials_fwd, first, second, first_is_own, width, out)
        ctx.save_for_backward(first, second, first_is_own)
        ctx.width = width
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("merge_lse_partials", "triton", {
                **_params(first, width, first_is_own), "BLOCK": _MERGE_BLOCK,
                "num_warps": _MERGE_NUM_WARPS, "dtype": first.dtype})
        return out

    @staticmethod
    def backward(ctx, gout):
        first, second, first_is_own = ctx.saved_tensors
        width = ctx.width
        if torch.is_grad_enabled():
            # create_graph: the Triton VJP is not differentiable, this is.
            grad_first, grad_second = _merge_vjp_torch(first, second, first_is_own, gout, width)
            return grad_first, grad_second, None, None
        gout = gout.contiguous()
        grad_first = torch.empty(first.shape, device=first.device, dtype=first.dtype)
        grad_second = torch.empty_like(grad_first)
        _launch(_merge_lse_partials_bwd, first, second, first_is_own, width, gout, grad_first, grad_second)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("merge_lse_partials", "triton", {
                **_params(first, width, first_is_own), "BLOCK": _MERGE_BLOCK,
                "num_warps": _MERGE_NUM_WARPS}, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return grad_first, grad_second, None, None


def merge_lse_partials(first: Tensor, second: Tensor, width: int, *,
                       first_is_own: Tensor | None = None) -> Tensor:
    """Softmax-weighted means from two disjoint parts ``[rows, 2 * width * groups]``.

    Without ``first_is_own``, ``first`` must be the canonical first operand at
    both ends (the lower rank's part), so that both evaluate the identical
    expression. With ``first_is_own`` (bool ``[rows]``), ``first``/``second``
    are the own/peer parts and row ``r`` is merged as ``(own, peer)`` where
    ``first_is_own[r]``, else as ``(peer, own)``; gradients return to
    own/peer. Returns ``[rows, width * groups]``.
    """
    width = int(width)
    if (first.dim() != 2 or first.shape != second.shape or first.shape[1] % (2 * width)
            or (first_is_own is not None and (first_is_own.shape != (first.shape[0],)
                                              or first_is_own.dtype != torch.bool))):
        raise ValueError(f"merge operands must be [rows, 2 * {width} * groups] (and first_is_own "
                         f"bool [rows]), got {tuple(first.shape)}, {tuple(second.shape)}")
    if first.is_cuda and second.is_cuda and first.dtype == second.dtype == torch.float32:
        return _MergeLsePartials.apply(first, second, width, first_is_own)
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("merge_lse_partials", "torch", _params(first, width, first_is_own))
    return merge_lse_partials_torch(first, second, width, first_is_own=first_is_own)
