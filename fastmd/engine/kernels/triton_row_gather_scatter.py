"""Fused row gather-scatter for exchange self rows (one kernel, no temporary).

``out[dst[i]] = src[src_rows[i]]`` (COPY) or ``out[dst[i]] += src[src_rows[i]]``
(SUM, atomic) for ``i < n``. Only CUDA FP32 matrices with unit column stride and
contiguous int64 row vectors qualify; callers keep an ATen lowering for everything
else, including deterministic mode for SUM (atomics are order-nondeterministic).
Index values must already be admitted by the caller; nothing is downloaded or
repaired here. The row count and the index-vector alignment (self blocks are
slices at arbitrary offsets) are excluded from Triton's value/alignment
specialization, so eager steps with new counts or offsets reuse the compiled
kernel. Compile-time keys are the column width, the mode flags, and the
alignment/divisibility of the matrices and their row strides (vectorization).
Row/column addresses are 64-bit.
"""
from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl

from fastmd.engine import recorder as _recorder


def row_gather_scatter_reason(output, source, source_rows, output_rows):
    for value in (output, source, source_rows, output_rows):
        if not isinstance(value, Tensor):
            return "not_tensor"
        if value.device.type != "cuda" or value.device != output.device:
            return "device"
        if value.layout != torch.strided:
            return "layout"
    if output.dtype != torch.float32 or source.dtype != torch.float32:
        return "dtype_not_float32"
    if (output.ndim != 2 or source.ndim != 2 or output.shape[1] != source.shape[1]
            or output.stride(1) != 1 or source.stride(1) != 1):
        return "matrix_shape_or_column_stride"
    for index in (source_rows, output_rows):
        if index.ndim != 1 or index.dtype != torch.int64 or not index.is_contiguous():
            return "index_shape_dtype_or_layout"
    if source_rows.numel() != output_rows.numel():
        return "index_length"
    if output.untyped_storage()._cdata in (source.untyped_storage()._cdata,
                                           source_rows.untyped_storage()._cdata,
                                           output_rows.untyped_storage()._cdata):
        return "output_aliases_input"
    return None


@triton.jit(do_not_specialize=["count"], do_not_specialize_on_alignment=["SRC_ROWS", "OUT_ROWS"])
def _row_gather_scatter_kernel(OUT, SRC, SRC_ROWS, OUT_ROWS, count, src_stride, out_stride,
                               WIDTH: tl.constexpr, ACCUMULATE: tl.constexpr,
                               SKIP_ZERO: tl.constexpr, BLOCK_R: tl.constexpr,
                               BLOCK_W: tl.constexpr):
    item = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    column = tl.program_id(1).to(tl.int64) * BLOCK_W + tl.arange(0, BLOCK_W).to(tl.int64)
    valid_item = item < count
    source_row = tl.load(SRC_ROWS + item, mask=valid_item, other=0)
    output_row = tl.load(OUT_ROWS + item, mask=valid_item, other=0)
    mask = valid_item[:, None] & (column < WIDTH)[None, :]
    value = tl.load(SRC + source_row[:, None] * src_stride + column[None, :], mask=mask, other=0.)
    target = OUT + output_row[:, None] * out_stride + column[None, :]
    if ACCUMULATE:
        if SKIP_ZERO:
            # Same element predicate as zero_skip_index_add_: NaN/Inf still SUM.
            mask = mask & (value != 0.)
        tl.atomic_add(target, value, mask=mask, sem="relaxed")
    else:
        tl.store(target, value, mask=mask)


@torch.no_grad()
def row_gather_scatter_(output: Tensor, source: Tensor, source_rows: Tensor,
                        output_rows: Tensor, *, accumulate: bool,
                        skip_zero: bool = False) -> Tensor:
    """Write (or atomically add) gathered source rows into existing ``output``.

    For ``accumulate`` the caller owns the zero initialization, which must
    precede this kernel on the current stream.
    """
    reason = row_gather_scatter_reason(output, source, source_rows, output_rows)
    if reason is not None:
        raise ValueError(f"unqualified row gather-scatter: {reason}")
    if skip_zero and not accumulate:
        raise ValueError("zero skipping applies only to SUM")
    count, width = source_rows.numel(), output.shape[1]
    if count == 0 or width == 0:
        return output
    block_w = min(128, triton.next_power_of_2(width))
    block_r = max(16, min(256, 4096 // block_w))
    grid = (triton.cdiv(count, block_r), triton.cdiv(width, block_w))
    _row_gather_scatter_kernel[grid](
        output, source, source_rows, output_rows, count, source.stride(0), output.stride(0),
        WIDTH=width, ACCUMULATE=accumulate, SKIP_ZERO=skip_zero,
        BLOCK_R=block_r, BLOCK_W=block_w, num_warps=4)
    rec = _recorder._ACTIVE
    if rec is not None:
        # A launcher, not a Function: the reverse (SUM) runs inside a VJP.
        in_backward = torch._C._current_autograd_node() is not None
        rec.record("row_gather_scatter", "triton", {
            "rows": count, "width": width, "ACCUMULATE": accumulate, "SKIP_ZERO": skip_zero,
            "BLOCK_R": block_r, "BLOCK_W": block_w, "num_warps": 4,
            "src_stride": source.stride(0), "out_stride": output.stride(0),
        }, phase="bwd" if in_backward else "fwd")
    return output
