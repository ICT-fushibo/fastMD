"""Merged frozen projections: one wide GEMM feeding several consumers.

Several attention-layer reads of the same feature tensor are frozen no-bias
linear projections (the decomposed-cat3 p0 block plus the source/target
attention weight linears). Executing them as one wide GEMM and returning
narrow views collapses N autograd consumers of the input into one Function,
removing N-1 gradient-accumulation ``add_`` kernels per layer in backward
(the InputBuffer sums that dominate the measured [rows,128] add family).

Constraints honored (see md_perf_analysis/nsys/s3_20260716/S3_PLAN.md):
- single wide mm (bmm/expand historically regressed the graphed path);
- outputs are row-strided views with stride(1)==1 — every consumer must
  accept that (segment attention does; the cat3 epilogue takes SP0);
- frozen weights only; the caller gates on requires_grad and caches the
  concatenated weight outside CUDA-graph capture;
- dead consumers (e.g. layer-4 branches) yield grad=None, which backward
  must skip without materializing zeros.

Single-GEMM backward (C3, ``MATRIS_MERGED_PROJECTION_SINGLE_BWD``):
the outputs carry a ``GradSlot``. A consumer that supports it (the cat3
epilogue for p0, the segment attention for the alpha views) claims its
output's column range in the slot's ``[rows, sum(splits)]`` buffer from its
backward and writes its gradient there (row stride ``sum(splits)``) instead of
into a tensor of its own; the buffer is used only if consumers of two adjacent
outputs registered in forward. The backward then runs one GEMM per contiguous range
of such in-buffer gradients, i.e. one ``mm`` over the whole buffer when every
consumer wrote into it, instead of one ``mm`` plus an ``addmm_`` per further
split (each re-reading and re-writing grad_x). A gradient that is not a claimed
column of the buffer (a dead consumer, a consumer without slot support, a sum
of several consumers, a stage boundary, torch.compile) takes the per-split
path; unwritten columns are never read. Claims are host-side metadata (no
sync), so CUDA-graph capture replays the same kernels.

The switch is "on", "off" (one GEMM per output) or "auto" (default): on while
the memory admission predicts that the configuration fits with it at the
saved_tensor_recompute level it chose without it (the buffer holds columns
longer), decided outside the step. The last interaction block never uses the
buffer (``grad_buffer=False``, see ``Graph_Attention_Layer``).

Row GEMMs (C4, ``MATRIS_TRITON_ROW_GEMM``): the forward GEMMs of these
projections, and the other K <= 32 frozen projections that go through
``project`` (the cat3/cat4 node and edge parts), run ``triton_row_gemm``'s
row GEMM where it applies (FP32 CUDA rows, K = 32, the widths it has
tiles for; not inside torch.compile) at ``InferenceConfig.triton_dot_precision``
(by default 3xTF32 where cuBLAS runs IEEE FP32; cuBLAS where it runs TF32). The
projection-with-alias gradient ``grad_alias + grad_p0 @ W`` reads
``grad_alias`` in the kernel (FP32 FMA). The recompute recipes of these
outputs call ``project`` too, so a rebuilt output is bitwise the forward's.
"""

from __future__ import annotations

import torch
from torch.autograd import Function

from fastmd.config import env_value
from fastmd.engine import recorder as _recorder
from fastmd.precision import triton_dot_input_precision

from fastmd.engine.kernels import triton_wide_gemm as _wide

# "on", "off" or "auto" (the memory admission's decision, ``saved_recompute.auto_single_bwd``).
_SINGLE_BWD = env_value("MATRIS_MERGED_PROJECTION_SINGLE_BWD") or "auto"
# C4: the Triton row GEMM for the K <= 32 frozen projections (see the module docstring).
_TRITON_ROW_GEMM = env_value("MATRIS_TRITON_ROW_GEMM") == "1"


def row_gemm_decline(x: torch.Tensor, weight: torch.Tensor) -> str | None:
    """None if ``x @ weight^T`` runs the Triton row GEMM now, else why not. Not called inside
    torch.compile (callers test ``is_compiling`` first: the compiled program stays today's)."""
    if not _TRITON_ROW_GEMM:
        return "switch"
    from fastmd.engine.kernels.triton_row_gemm import row_linear_supported
    reason = row_linear_supported(x, weight)
    if reason is None and triton_dot_input_precision("row_gemm") is None:
        # cuBLAS's TF32 GEMMs outrun the TF32 row GEMM at these widths (e.g. 0.74 vs 0.92 ms at N = 128).
        return "cublas_precision"
    return reason


def _record_row_gemm_decline(reason: str, x: torch.Tensor, weight: torch.Tensor) -> None:
    rec = _recorder._ACTIVE
    if rec is not None and reason not in ("switch", "compiled") and x.dim() == 2 and x.shape[1] in (16, 32):
        rec.fallback("row_gemm", "triton", reason, {"rows": x.shape[0], "K": x.shape[1], "N": weight.shape[0]})


def merged_row_gemm_decline(x: torch.Tensor, w_cat: torch.Tensor, splits) -> str | None:
    """``row_gemm_decline`` of a merged product whose column ranges (p0, the alphas) recompute recipes may
    rebuild alone: those widths need row GEMM tiles too (bitwise the wide product's columns), whatever the
    recompute level, so every level runs the same forward."""
    reason = row_gemm_decline(x, w_cat)
    if reason is None:
        from fastmd.engine.kernels.triton_row_gemm import _TILES
        if splits[0] not in _TILES or splits[1] + splits[2] not in _TILES:
            return "unsupported_shape"
    return reason


def row_gemm_precision(x: torch.Tensor, weight: torch.Tensor, splits=None) -> str | None:
    """The row GEMM's dot precision if ``x @ weight^T`` (a merged product with ``splits``) runs it now, else
    None (``F.linear``): what a recompute recipe of the product binds (see ``project``)."""
    reason = row_gemm_decline(x, weight) if splits is None else merged_row_gemm_decline(x, weight, splits)
    if reason is not None:
        return None
    from fastmd.engine.kernels.triton_row_gemm import precision
    return precision()


_DECIDE = "decide"


def project(x: torch.Tensor, weight: torch.Tensor, precision: str | None = _DECIDE,
            wide: str | None = None) -> torch.Tensor:
    """``x @ weight^T`` of a frozen projection: the row GEMM where it applies, else ``F.linear``.

    Differentiable in ``x`` (the row GEMM's backward is ``dY @ W``); also the recompute recipe of such
    an output. ``precision``: what the forward took (``row_gemm_precision``: the row GEMM at that dot
    precision, None ``F.linear``) for a recipe of a column range of a wider product, so the rebuilt tensor
    is bitwise the forward's; omitted decides here. ``wide``: with ``precision`` None, the wide GEMM the
    product took (``wide_precision``; None cuBLAS)."""
    if precision != _DECIDE:
        if precision is None:
            return _wide.linear(x, weight, precision=wide)
        from fastmd.engine.kernels.triton_row_gemm import row_linear
        from fastmd.engine.kernels.triton_row_gemm import row_linear_supported
        if x.stride(1) != 1:  # a rebuilt input in another layout: row results do not depend on it
            x = x.contiguous()
        reason = row_linear_supported(x, weight)
        if reason is not None:  # the forward ran the row GEMM on these tensors: an invariant
            raise RuntimeError(f"recompute recipe cannot rebuild the row GEMM's output bitwise: {reason}")
        return row_linear(x, weight, precision)
    reason = "compiled" if torch.compiler.is_compiling() else row_gemm_decline(x, weight)
    if reason is None:
        from fastmd.engine.kernels.triton_row_gemm import row_linear
        return row_linear(x, weight)
    _record_row_gemm_decline(reason, x, weight)
    return _wide.linear(x, weight)


def wide_precision(x: torch.Tensor, w_cat: torch.Tensor, splits) -> str | None:
    """The wide GEMM ``merged_frozen_projections(x, w_cat, splits)`` takes now when the row GEMM does not (a
    precision, None cuBLAS): what a recipe of its column ranges (``project``) binds."""
    return _wide.linear_precision(x, w_cat, parts=((0, splits[0]), (splits[0], splits[1] + splits[2])))


def single_bwd_enabled() -> bool:
    """Whether a merged projection forward attaches a gradient slot now."""
    if _SINGLE_BWD == "auto":
        from fastmd.engine import saved_recompute
        return saved_recompute.auto_single_bwd()
    return _SINGLE_BWD == "on"


class GradSlot:
    """The shared gradient buffer of one merged projection's outputs.

    Consumers register their output in forward (``grad_claim``). The buffer
    is used only if two adjacent outputs registered, i.e. if it can save a
    GEMM; otherwise every claim falls back to a tensor of the consumer's own
    (e.g. the P>1 atom attention, whose alphas leave the stage). It is
    allocated by the first claim of a backward and handed to the merged
    backward by ``take``; a column range is claimed at most once per buffer,
    so two consumers of one output never write the same columns.
    """

    __slots__ = ("rows", "total", "offsets", "device", "dtype", "expected", "buffer", "claimed")

    def __init__(self, rows: int, total: int, offsets: tuple[int, int, int], device, dtype) -> None:
        self.rows, self.total, self.offsets, self.device, self.dtype = rows, total, offsets, device, dtype
        self.expected = set()
        self.buffer = None
        self.claimed = set()

    def claim(self, offset: int, width: int) -> torch.Tensor | None:
        """The ``[rows, width]`` column view to write a gradient into, or None."""
        if self.buffer is None:
            if not any(self.offsets[k] in self.expected and self.offsets[k + 1] in self.expected
                       for k in range(len(self.offsets) - 1)):
                return None
            self.buffer = torch.empty((self.rows, self.total), device=self.device, dtype=self.dtype)
            self.claimed = set()
        if offset in self.claimed:
            return None
        self.claimed.add(offset)
        return self.buffer.narrow(1, offset, width)

    def take(self) -> torch.Tensor | None:
        buffer, self.buffer = self.buffer, None
        return buffer


def grad_claim(tensor: torch.Tensor):
    """For a consumer's forward: ``(slot, offset, width)`` if ``tensor`` is a merged
    projection output whose gradient may go into the shared buffer, else None."""
    handle = getattr(tensor, "_matris_grad_slot", None)
    if handle is None or tuple(tensor.shape) != (handle[0].rows, handle[2]):
        return None
    handle[0].expected.add(handle[1])
    return handle


def claim_grad(handle, shape, device, dtype) -> torch.Tensor:
    """For a consumer's backward: the buffer view of ``handle`` (a ``grad_claim``
    result, possibly None), else a new ``torch.empty`` gradient."""
    if handle is not None:
        view = handle[0].claim(handle[1], handle[2])
        if view is not None:
            return view
    return torch.empty(shape, device=device, dtype=dtype)


def _in_buffer(grad, buffer, offset, size) -> bool:
    return (grad is not None and buffer is not None and grad.dtype == buffer.dtype
            and grad.device == buffer.device and tuple(grad.shape) == (buffer.shape[0], size)
            and grad.stride() == buffer.stride()
            and grad.data_ptr() == buffer.data_ptr() + offset * buffer.element_size())


class _MergedFrozenProjections(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_cat: torch.Tensor,
        split0: int,
        split1: int,
        split2: int,
        grad_buffer: bool = True,
    ):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(w_cat)
        ctx.splits = (split0, split1, split2)
        ctx.wide = _wide.dispatch_on()  # the backward follows the forward's GEMM dispatch
        if torch.compiler.is_compiling():
            reason = "compiled"
            wide = torch.nn.functional.linear(x, w_cat)
        else:
            reason = merged_row_gemm_decline(x, w_cat, (split0, split1, split2))
            if reason is None:
                from fastmd.engine.kernels.triton_row_gemm import precision
                from fastmd.engine.kernels.triton_row_gemm import row_linear_forward
                prec = precision()
                wide = row_linear_forward(x, w_cat, prec)
            else:
                _record_row_gemm_decline(reason, x, w_cat)
                # The column ranges [:split0] and [split0:] are rebuilt alone by recipes (``wide_precision``).
                wide = _wide.linear(x, w_cat, parts=((0, split0), (split0, split1 + split2)))
                if ctx.wide is not None:  # under a capture: the backward's weight row ranges
                    ends = (0, split0, split0 + split1, split0 + split1 + split2)
                    _wide.retain_transposed(w_cat, ranges=[(ends[i], ends[j] - ends[i])
                                                           for i in range(3) for j in range(i + 1, 4)])
        out0 = wide.narrow(1, 0, split0)
        out1 = wide.narrow(1, split0, split1)
        out2 = wide.narrow(1, split0 + split1, split2)
        slot = None
        if grad_buffer and not torch.compiler.is_compiling() and single_bwd_enabled():
            ctx._grad_slot = slot = GradSlot(wide.shape[0], wide.shape[1], (0, split0, split0 + split1),
                                             wide.device, wide.dtype)
            for out, offset, size in ((out0, 0, split0), (out1, split0, split1), (out2, split0 + split1, split2)):
                out._matris_grad_slot = (slot, offset, size)
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
            params = {
                "rows": x.shape[0], "in_width": x.shape[1], "splits": [split0, split1, split2],
                "out_stride0": wide.stride(0), "input_contiguous": x.is_contiguous(),
                "dtype": x.dtype,
            }
            if slot is not None:
                params["grad_slot"] = True
            if not reason:
                params["PRECISION"] = prec
            rec.record("merged_frozen_projections", "torch_linear" if reason else "triton_row_gemm", params)
        return out0, out1, out2

    @staticmethod
    def backward(ctx, *grads):
        (w_cat,) = ctx.saved_tensors
        slot = getattr(ctx, "_grad_slot", None)
        buffer = slot.take() if slot is not None else None
        offsets = (0, ctx.splits[0], ctx.splits[0] + ctx.splits[1])
        in_buffer = [_in_buffer(grad, buffer, offset, size)
                     for grad, offset, size in zip(grads, offsets, ctx.splits)]
        # (gradient, weight rows) per GEMM: one per contiguous run of in-buffer
        # gradients, one per other live gradient.
        terms, k = [], 0
        while k < 3:
            if grads[k] is None:
                k += 1
                continue
            last = k
            while in_buffer[k] and last + 1 < 3 and in_buffer[last + 1]:
                last += 1
            start, width = offsets[k], offsets[last] + ctx.splits[last] - offsets[k]
            grad = buffer.narrow(1, start, width) if in_buffer[k] else grads[k]
            terms.append((grad, w_cat.narrow(0, start, width)))
            k = last + 1
        rec = _recorder._ACTIVE
        if rec is not None:
            params = {"splits": list(ctx.splits), "live_grads": [grad is not None for grad in grads]}
            if slot is not None:
                params["in_buffer"] = in_buffer
                params["gemms"] = len(terms)
            rec.record("merged_frozen_projections", "torch_linear", params,
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        grad_x = None
        for grad, w_part in terms:
            grad_x = _wide.matmul(grad, w_part, grad_x, accumulate=grad_x is not None, enabled=ctx.wide)
        return grad_x, None, None, None, None, None


def merged_frozen_projections(
    x: torch.Tensor,
    w_cat: torch.Tensor,
    splits: tuple[int, int, int],
    grad_buffer: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute [x@W0.T | x@W1.T | x@W2.T] with one GEMM; views returned.

    ``grad_buffer=False``: never the shared gradient buffer (the last block's
    merged projections, whose backward runs first: see ``Graph_Attention_Layer``)."""
    return _MergedFrozenProjections.apply(x, w_cat, *splits, grad_buffer)


class _ProjectionWithAlias(Function):
    """One frozen projection plus an identity alias of the input.

    Routing a second consumer (e.g. a residual read) through the alias makes
    this Function the input's single autograd consumer, so the two gradient
    contributions meet here and fold into one addmm instead of a separate
    InputBuffer add_ kernel.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w0: torch.Tensor):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(w0)
        compiling = torch.compiler.is_compiling()
        reason = "compiled" if compiling else row_gemm_decline(x, w0)
        # The input gradient's addmm follows the switch in both FP32 GEMM modes (FP32 FMA; faster than
        # cuBLAS's addmm, which copies grad_alias into its output first, in both); torch.addmm under a
        # graph-building (create_graph) backward.
        ctx.row_addmm = not compiling and _TRITON_ROW_GEMM
        ctx.wide = _wide.dispatch_on()
        if reason is None:
            from fastmd.engine.kernels.triton_row_gemm import precision
            prec = precision()
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
            params = {"rows": x.shape[0], "in_width": x.shape[1], "out_width": w0.shape[0],
                      "input_contiguous": x.is_contiguous(), "dtype": x.dtype}
            if reason is None:
                params["PRECISION"] = prec
            rec.record("projection_with_alias", "torch_linear" if reason else "triton_row_gemm", params)
        if reason is None:
            from fastmd.engine.kernels.triton_row_gemm import row_linear_forward
            return row_linear_forward(x, w0, prec), x.view_as(x)
        if not compiling:
            _record_row_gemm_decline(reason, x, w0)
            if ctx.wide is not None:
                _wide.retain_transposed(w0)
            return _wide.linear(x, w0), x.view_as(x)
        return torch.nn.functional.linear(x, w0), x.view_as(x)

    @staticmethod
    def backward(ctx, grad_p0, grad_alias):
        (w0,) = ctx.saved_tensors
        triton_addmm = False
        if grad_p0 is not None and grad_alias is not None and ctx.row_addmm and not torch.is_grad_enabled():
            from fastmd.engine.kernels.triton_row_gemm import row_addmm_supported
            triton_addmm = row_addmm_supported(grad_alias, grad_p0, w0)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("projection_with_alias", "triton_row_addmm" if triton_addmm else "torch_linear", {
                "grad_p0": grad_p0 is not None, "grad_alias": grad_alias is not None,
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        if grad_p0 is None:
            return grad_alias, None
        if grad_alias is None:
            return _wide.matmul(grad_p0, w0, enabled=ctx.wide), None
        if triton_addmm:
            from fastmd.engine.kernels.triton_row_gemm import row_addmm
            return row_addmm(grad_alias, grad_p0, w0), None
        return _wide.matmul(grad_p0, w0, grad_alias, enabled=ctx.wide), None


def projection_with_alias(
    x: torch.Tensor,
    w0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x @ w0.T, alias-of-x) with a fused-accumulation backward."""
    return _ProjectionWithAlias.apply(x, w0)
