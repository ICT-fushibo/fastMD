"""Exact row SUM/gather semantics with zero-valued atomic writes omitted.

Only dense CUDA FP32 matrices and contiguous int64 routing vectors qualify.
Index *values* must already have passed the caller's topology/route admission;
these internal lowerings never download routing data or repair invalid indices.
There is no sink-row special case: nonzero, NaN and infinite values still SUM.
Widths that are a multiple of 4 skip all-zero row blocks and add the remaining
zeros as -0.0 (the exact additive identity; vector atomics), which yields the
same destination values as the per-element skip used for other widths.
Qualified kernel errors propagate; callers decide unsupported-shape fallbacks.
Both kernels work on 2-D tiles of BLOCK_R rows x BLOCK_W columns: one index load
per row, 64-bit addresses. The row count and the index/row pointer alignment
(exchange blocks are slices at arbitrary offsets) are excluded from Triton's
value/alignment specialization, so eager steps with new counts or offsets reuse
the compiled kernel; the row width and the tile stay compile-time constants.
The gathers (select forward, sum VJP) are exact copies through a Triton row
gather when the source has unit column stride; otherwise they use ATen.

In deterministic mode (``fastmd.precision.deterministic_mode``) the SUM runs
``ordered_index_add_`` instead of the atomic kernel: torch's ordered index_add_,
with all-zero rows sent to discarded scratch rows. It is also the ordered VJP
of the other atomic MatRIS kernels. Zero entries of live rows are then added,
so a -0.0 destination may become +0.0.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import Function
import triton
import triton.language as tl

from fastmd.engine import recorder as _recorder
from fastmd.precision import deterministic_mode
from fastmd.engine.kernels.row_count import static_under_compile

# Scratch rows of ordered_index_add_ that the all-zero rows are spread over.
_SCRATCH_ROWS = 1024


def _matrix_index_reason(matrix, indices):
    if not isinstance(matrix, Tensor) or not isinstance(indices, Tensor):
        return "not_tensor"
    if matrix.device.type != "cuda" or indices.device.type != "cuda":
        return "device_not_cuda"
    if matrix.device != indices.device:
        return "device_mismatch"
    if matrix.dtype != torch.float32:
        return "dtype_not_float32"
    if matrix.layout != torch.strided or matrix.ndim != 2 or not matrix.is_contiguous():
        return "matrix_shape_or_layout"
    if (indices.layout != torch.strided or indices.ndim != 1
            or indices.dtype != torch.int64 or not indices.is_contiguous()):
        return "index_shape_dtype_or_layout"
    return None


def zero_skip_index_sum_reason(rows, indices, count):
    reason = _matrix_index_reason(rows, indices)
    if reason is not None:
        return reason
    if type(count) is not int or count < 0:
        return "invalid_output_count"
    if indices.numel() != rows.shape[0]:
        return "index_row_count"
    if count == 0 and rows.shape[0] != 0:
        return "nonempty_input_for_empty_output"
    return None


def zero_skip_index_select_reason(source, indices):
    reason = _matrix_index_reason(source, indices)
    if reason is not None:
        return reason
    if source.shape[0] == 0 and indices.numel() != 0:
        return "nonempty_request_for_empty_source"
    return None


def _require(reason):
    if reason is not None:
        raise ValueError(f"unqualified zero-skipping index operation: {reason}")


def _gather_tile(width):
    """(BLOCK_R, BLOCK_W, num_warps) of the row gather (GB300 microbenchmark)."""
    block_w = min(128, triton.next_power_of_2(width))
    return max(1, 1024 // block_w), block_w, 2


def _add_tile(width):
    """(BLOCK_R, BLOCK_W, num_warps, ROW_SKIP) of the SUM (GB300 microbenchmark).

    Rows of a multiple of 4 floats take the row-granular skip, whose uniform
    mask lets Triton emit v4 atomics (~1.5x); other widths skip per element.
    """
    block_w = min(32, triton.next_power_of_2(width))
    if width % 4 == 0:
        return max(1, 2048 // block_w), block_w, 2, True
    if block_w < 32:
        return 128, block_w, 4, False
    return 16, block_w, 8, False


@triton.jit(do_not_specialize=["ROW_COUNT"], do_not_specialize_on_alignment=["INDEX", "ROWS"])
def _zero_skip_index_add_kernel(OUT, INDEX, ROWS, ROW_COUNT, WIDTH: tl.constexpr,
                                BLOCK_R: tl.constexpr, BLOCK_W: tl.constexpr,
                                ROW_SKIP: tl.constexpr):
    # Use 64-bit address arithmetic even when individual row IDs fit int32.
    row = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    column = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    valid_row = row < ROW_COUNT
    destination = tl.load(INDEX + row, mask=valid_row, other=0)
    valid = valid_row[:, None] & (column < WIDTH)[None, :]
    value = tl.load(ROWS + row[:, None] * WIDTH + column[None, :], mask=valid, other=0.)
    # In particular, do NOT mask by a special destination or by isfinite().
    # NaN != 0, +/-Inf != 0; both retain the ordinary atomic SUM behavior.
    if ROW_SKIP:
        # A row-uniform mask lets the atomics vectorize. Only all-zero row
        # blocks are skipped; a zero inside a live row is added as -0.0,
        # the exact additive identity (d + -0.0 == d, also for d == -0.0), so
        # every destination equals the per-element skip's result. (A literal
        # -0. would be +0.: Triton lowers negation to 0 - x; use the bits.)
        live = tl.max((value != 0.).to(tl.int32), axis=1) != 0
        negative_zero = tl.full(value.shape, 0x80000000, tl.uint32).to(tl.float32, bitcast=True)
        value = tl.where(value == 0., negative_zero, value)
        valid = (valid_row & live)[:, None] & (column < WIDTH)[None, :]
    else:
        valid = valid & (value != 0.)
    tl.atomic_add(OUT + destination[:, None] * WIDTH + column[None, :], value,
                  mask=valid, sem="relaxed")


@triton.jit(do_not_specialize=["ROW_COUNT"], do_not_specialize_on_alignment=["INDEX"])
def _row_gather_kernel(OUT, SOURCE, INDEX, ROW_COUNT, source_stride, WIDTH: tl.constexpr,
                       BLOCK_R: tl.constexpr, BLOCK_W: tl.constexpr):
    # OUT[i] = SOURCE[INDEX[i]]: an exact copy. The source row stride stays
    # specializable (divisibility only) so aligned rows vectorize.
    row = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    column = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    valid_row = row < ROW_COUNT
    source_row = tl.load(INDEX + row, mask=valid_row, other=0)
    valid = valid_row[:, None] & (column < WIDTH)[None, :]
    value = tl.load(SOURCE + source_row[:, None] * source_stride + column[None, :], mask=valid)
    tl.store(OUT + row[:, None] * WIDTH + column[None, :], value, mask=valid)


def _row_gather_qualifies(source, indices):
    # Metadata only; index values were admitted by the caller.
    return (source.device.type == "cuda" and indices.device == source.device
            and source.dtype == torch.float32 and source.layout == torch.strided
            and source.ndim == 2 and (source.stride(1) == 1 or source.shape[1] <= 1)
            and indices.layout == torch.strided and indices.ndim == 1
            and indices.dtype == torch.int64 and indices.is_contiguous())


def _gather_rows(source, indices):
    """``source.index_select(0, indices)`` into a fresh contiguous matrix.

    Returns ``(rows, impl, params)``; no host synchronization either way.
    """
    if not _row_gather_qualifies(source, indices):
        return source.index_select(0, indices), "torch_index_select", {}
    rows, width = indices.numel(), source.shape[1]
    output = source.new_empty((rows, width))
    block_r, block_w, num_warps = _gather_tile(max(width, 1))
    if rows != 0 and width != 0:
        _row_gather_kernel[(triton.cdiv(rows, block_r), triton.cdiv(width, block_w))](
            output, source, indices, ROW_COUNT=static_under_compile(rows),
            source_stride=static_under_compile(source.stride(0)), WIDTH=width,
            BLOCK_R=block_r, BLOCK_W=block_w, num_warps=num_warps)
    return output, "triton_row_gather", {
        "BLOCK_R": block_r, "BLOCK_W": block_w, "num_warps": num_warps,
        "launched": rows != 0 and width != 0}


@torch.no_grad()
def ordered_index_add_(output: Tensor, indices: Tensor, rows: Tensor, *, skip_zero: bool = True) -> Tensor:
    """``output.index_add_(0, indices, rows)`` in deterministic mode, without the hot sink.

    torch orders an index_add_ under its deterministic flag by sorting the
    destinations and adding each destination's rows sequentially, so the
    padding rows that all target one private sink serialize it (GB200,
    262,144 rows of width 32, 162,144 of them zero rows on one sink: 34 ms,
    0.27 ms here). With ``skip_zero`` (the private sink's layouts) all-zero
    rows, whose addition changes nothing up to the sign of a zero, are
    therefore spread over ``_SCRATCH_ROWS`` discarded rows (NaN counts as
    nonzero); without it this is the plain index_add_. ``output`` [C, W]
    accumulates with any strides; ``indices`` may be int32 or int64. No
    synchronization (capturable).
    """
    if not skip_zero:
        return output.index_add_(0, indices, rows)
    count = output.shape[0]
    live = (rows != 0).any(1)
    spread = torch.arange(rows.shape[0], device=rows.device) % _SCRATCH_ROWS + count
    scratch = rows.new_zeros((count + _SCRATCH_ROWS, *rows.shape[1:]))
    scratch.index_add_(0, torch.where(live, indices, spread), rows)
    return output.add_(scratch[:count])


@torch.no_grad()
def zero_skip_index_add_(output: Tensor, indices: Tensor, rows: Tensor) -> Tensor:
    """Accumulate into independent existing output; caller owns initialization.

    The separate zero-fill/allocation must precede this kernel on the current
    stream. Fusing a zero-fill into another CTA of this grid would race atomics.
    This low-level entry intentionally does not supply an autograd contract.
    """
    if not isinstance(output, Tensor) or output.ndim != 2:
        raise ValueError("zero-skipping index output must be a matrix")
    _require(zero_skip_index_sum_reason(rows, indices, output.shape[0]))
    if (output.layout != torch.strided or output.dtype != rows.dtype
            or output.device != rows.device or output.shape[1] != rows.shape[1]
            or not output.is_contiguous()):
        raise ValueError("zero-skipping index output shape/dtype/device/layout mismatch")
    if output.untyped_storage()._cdata in (
            rows.untyped_storage()._cdata, indices.untyped_storage()._cdata):
        raise ValueError("zero-skipping index output must not alias rows or indices")
    count, width = rows.shape
    if deterministic_mode():
        ordered_index_add_(output, indices, rows)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("zero_skip_index_add", "ordered_index_add", {
                "elements": rows.numel(), "width": width, "count": output.shape[0],
                "dtype": rows.dtype, "index_dtype": indices.dtype,
            }, phase="bwd" if torch._C._current_autograd_node() is not None else "fwd")
        return output
    block_r, block_w, num_warps, row_skip = _add_tile(max(width, 1))
    if rows.numel() != 0:
        _zero_skip_index_add_kernel[(triton.cdiv(count, block_r), triton.cdiv(width, block_w))](
            output, indices, rows, ROW_COUNT=static_under_compile(count), WIDTH=width,
            BLOCK_R=block_r, BLOCK_W=block_w, ROW_SKIP=row_skip, num_warps=num_warps,
            enable_fp_fusion=False)
    rec = _recorder._ACTIVE
    if rec is not None:
        # A launcher, not a Function: inside a VJP (e.g. _IndexSelect's) it
        # is backward work, attributed to the executing autograd node.
        in_backward = torch._C._current_autograd_node() is not None
        rec.record("zero_skip_index_add", "triton", {
            "elements": rows.numel(), "width": width, "count": output.shape[0],
            "BLOCK_R": block_r, "BLOCK_W": block_w, "num_warps": num_warps, "ROW_SKIP": row_skip,
            "launched": rows.numel() != 0, "dtype": rows.dtype, "index_dtype": indices.dtype,
        }, phase="bwd" if in_backward else "fwd")
    return output


class _IndexSum(Function):
    @staticmethod
    def forward(ctx, rows, indices, count):
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
        ctx.save_for_backward(indices)
        output = rows.new_zeros((count, rows.shape[1]))
        return zero_skip_index_add_(output, indices, rows)

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        # SUM is linear even at rows == 0. Reusing the forward value predicate
        # here would incorrectly erase a nonzero cotangent of a zero input.
        # Row-strided cotangents are read in place (no contiguous copy).
        derivative, impl, params = _gather_rows(gradient, indices)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("zero_skip_index_sum", impl, {
                "rows": indices.numel(), "width": gradient.shape[1], **params,
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return derivative, None, None


class _IndexSelect(Function):
    @staticmethod
    def forward(ctx, source, indices):
        ctx.save_for_backward(indices)
        ctx.source_rows = source.shape[0]
        gathered, impl, params = _gather_rows(source, indices)
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("zero_skip_index_select", impl, {
                "rows": indices.numel(), "width": source.shape[1],
                "source_rows": source.shape[0], **params,
            })
        return gathered

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("zero_skip_index_select", "zero_skip_index_sum", {
                "rows": indices.numel(), "width": gradient.shape[1],
                "source_rows": ctx.source_rows, "contiguous_copy": not gradient.is_contiguous(),
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        # Grad outputs may be expanded/strided, independently of primal layout.
        return zero_skip_index_sum(gradient.contiguous(), indices, ctx.source_rows), None


def zero_skip_index_sum(rows: Tensor, indices: Tensor, count: int) -> Tensor:
    """Fresh row-SUM output; derivative gathers every row, including zero rows."""
    _require(zero_skip_index_sum_reason(rows, indices, count))
    return _IndexSum.apply(rows, indices, count)


def zero_skip_index_select(source: Tensor, indices: Tensor) -> Tensor:
    """Ordinary row gather, whose VJP uses the same zero-skipping SUM kernel."""
    _require(zero_skip_index_select_reason(source, indices))
    return _IndexSelect.apply(source, indices)
