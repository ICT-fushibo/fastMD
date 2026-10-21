"""Row-chunked ``linear(cat([aligned, value_k[index_k], ...], 1))`` for frozen linears (saved-tensor recompute).

The concatenation is the widest transient of a line-graph first layer
([triplets, 224] on the refinement); this op builds and projects it a row
chunk at a time into preallocated outputs, forward and backward, so that it
never exists whole. Each chunk runs ``F.linear``'s GEMM (the 3xTF32 wide GEMM,
or cuBLAS's addmm) straight into its output rows; with the balanced chunk sizes
of ``saved_recompute.chunk_rows`` the rows equal those of one full GEMM
(tests/test_saved_recompute.py sweeps the sizes). Nothing per row is saved: the weights are frozen, so the input gradient needs only the weights.
The backward writes the aligned gradient in place and ``index_add_``s each
gathered part once per chunk into one buffer per input.
"""
from __future__ import annotations

import torch
from torch.autograd import Function

from fastmd.engine.kernels import triton_wide_gemm as _wide


class _ChunkedGatherLinear(Function):
    @staticmethod
    def forward(ctx, chunk, count, aligned, *rest):
        pairs = [(rest[2 * k], rest[2 * k + 1]) for k in range(count)]
        linears = rest[2 * count:]
        rows = aligned.shape[0]
        outputs = [aligned.new_empty((rows, weight.shape[0])) for weight, _ in zip(linears[::2], linears[1::2])]
        for start in range(0, rows, chunk):
            part = slice(start, min(rows, start + chunk))
            feas = torch.cat([aligned[part], *(torch.index_select(value, 0, index[part]) for value, index in pairs)],
                             dim=1)
            for output, weight, bias in zip(outputs, linears[::2], linears[1::2]):
                # F.linear's GEMM (the 3xTF32 wide GEMM or cuBLAS's addmm/mm, decided on all rows), written
                # straight into the rows.
                _wide.linear_into(output[part], feas, weight, bias, rows=rows)
            del feas
        ctx.chunk, ctx.count = chunk, count
        ctx.wide = _wide.dispatch_on()
        if ctx.wide is not None:
            _wide.retain_transposed(*linears[::2])
        ctx.widths = [aligned.shape[1]] + [value.shape[1] for value, _ in pairs]
        ctx.shapes = [value.shape for value, _ in pairs]
        ctx.aligned_shape, ctx.dtype = aligned.shape, aligned.dtype
        ctx.save_for_backward(*(index for _, index in pairs), *linears[::2])
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        saved = ctx.saved_tensors
        indices, weights = saved[:ctx.count], saved[ctx.count:]
        needs = ctx.needs_input_grad
        device = weights[0].device
        grad_aligned = (torch.empty(ctx.aligned_shape, device=device, dtype=ctx.dtype)
                        if needs[2] else None)
        grad_values = [torch.zeros(shape, device=device, dtype=ctx.dtype) if needs[3 + 2 * k] else None
                       for k, shape in enumerate(ctx.shapes)]
        rows = ctx.aligned_shape[0]
        for start in range(0, rows, ctx.chunk):
            part = slice(start, min(rows, start + ctx.chunk))
            grad_cat = None
            for grad, weight in zip(grads, weights):
                if grad is None:
                    continue
                grad_cat = _wide.matmul(grad[part], weight, grad_cat, rows=rows, accumulate=grad_cat is not None,
                                        enabled=ctx.wide)
            if grad_cat is None:
                continue
            offset = ctx.widths[0]
            if grad_aligned is not None:
                grad_aligned[part].copy_(grad_cat[:, :offset])
            for k, index in enumerate(indices):
                width = ctx.widths[k + 1]
                if grad_values[k] is not None:
                    grad_values[k].index_add_(0, index[part], grad_cat[:, offset:offset + width])
                offset += width
        if grad_aligned is not None and all(grad is None for grad in grads):
            grad_aligned.zero_()
        result = [None, None, grad_aligned]
        for grad in grad_values:
            result += [grad, None]
        return (*result, *([None] * (2 * len(weights))))


def chunked_gather_linear(aligned, gathered, linears, chunk: int) -> tuple[torch.Tensor, ...]:
    """``tuple(F.linear(cat([aligned, v[i] for v, i in gathered], 1), W, b) for W, b in linears)``
    computed ``chunk`` rows at a time; the linears must be frozen."""
    flat = [tensor for pair in gathered for tensor in pair]
    params = [tensor for weight, bias in linears for tensor in (weight, bias)]
    return _ChunkedGatherLinear.apply(int(chunk), len(gathered), aligned, *flat, *params)
