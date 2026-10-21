"""Fused weighted segment sum for graph aggregation.

Computes ``out[segment[i], d] += weight[i, d] * value[i, d]``. This replaces
the pair ``weight * value`` followed by ``index_add_`` where the product feeds
only a segment sum. The row count is a runtime argument excluded from Triton's
value specialization, so new row counts reuse the compiled kernels.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function

from fastmd.engine import recorder as _recorder
from fastmd.engine import saved_recompute as _saved_recompute
from fastmd.engine.kernels.row_count import static_under_compile


@triton.jit(do_not_specialize=["N"])
def _weighted_segsum_fwd(
    weight,
    value,
    segment,
    out,
    N,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_d * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    seg = tl.load(segment + rows, mask=rows < N, other=0).to(tl.int64)
    offs = rows[:, None] * D + cols[None, :]
    w = tl.load(weight + offs, mask=mask, other=0.0)
    v = tl.load(value + offs, mask=mask, other=0.0)
    tl.atomic_add(out + seg[:, None] * D + cols[None, :], w * v, mask=mask, sem="relaxed")


@triton.jit(do_not_specialize=["N"])
def _weighted_segsum_bwd(
    gout,
    weight,
    value,
    segment,
    gweight,
    gvalue,
    N,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_d * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    seg = tl.load(segment + rows, mask=rows < N, other=0).to(tl.int64)
    offs = rows[:, None] * D + cols[None, :]
    go = tl.load(gout + seg[:, None] * D + cols[None, :], mask=mask, other=0.0)
    w = tl.load(weight + offs, mask=mask, other=0.0)
    v = tl.load(value + offs, mask=mask, other=0.0)
    tl.store(gweight + offs, go * v, mask=mask)
    tl.store(gvalue + offs, go * w, mask=mask)


class _WeightedSegmentSum(Function):
    @staticmethod
    def forward(ctx, weight, value, segment, num_segment):
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = [not weight.is_contiguous(), not value.is_contiguous()]
            index_dtype = segment.dtype
        weight = weight.contiguous()
        value = value.contiguous()
        source = segment
        segment = segment.to(torch.int64).contiguous()
        _saved_recompute.register_copy(segment, source, int64=True)
        N, D = weight.shape
        BM = 4
        BD = min(triton.next_power_of_2(D), 128)
        out = value.new_zeros((num_segment, D))
        grid = (triton.cdiv(N, BM), triton.cdiv(D, BD))
        _weighted_segsum_fwd[grid](
            weight, value, segment, out, N=static_under_compile(N), D=D, BM=BM, BD=BD, num_warps=4
        )
        ctx.save_for_backward(weight, value, segment)
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("weighted_segment_sum", "triton", {
                "rows": N, "width": D, "segments": num_segment, "BM": BM, "BD": BD,
                "num_warps": 4, "dtype": value.dtype, "index_dtype": index_dtype,
                "contiguous_copy": copied,
            })
        return out

    @staticmethod
    def backward(ctx, gout):
        weight, value, segment = ctx.saved_tensors
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = not gout.is_contiguous()
        gout = gout.contiguous()
        N, D = weight.shape
        BM = 4
        BD = min(triton.next_power_of_2(D), 128)
        gweight = torch.empty_like(weight)
        gvalue = torch.empty_like(value)
        grid = (triton.cdiv(N, BM), triton.cdiv(D, BD))
        _weighted_segsum_bwd[grid](
            gout,
            weight,
            value,
            segment,
            gweight,
            gvalue,
            N=static_under_compile(N),
            D=D,
            BM=BM,
            BD=BD,
            num_warps=4,
        )
        if rec is not None:
            rec.record("weighted_segment_sum", "triton", {
                "rows": N, "width": D, "BM": BM, "BD": BD, "num_warps": 4,
                "dtype": value.dtype, "contiguous_copy": copied,
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return gweight, gvalue, None, None


def weighted_segment_sum(
    weight: torch.Tensor,
    value: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int,
) -> torch.Tensor:
    if weight.is_cuda and value.is_cuda:
        return _WeightedSegmentSum.apply(weight, value, segment, int(num_segment))
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("weighted_segment_sum", "torch", {
            "rows": weight.shape[0], "width": value.shape[1], "segments": int(num_segment),
        })
    out = (weight * value).new_zeros((int(num_segment), value.shape[1]))
    return out.index_add_(0, segment, weight * value)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for N, D, S in [(26174, 128, 1165), (152446, 128, 13087), (5000, 128, 512)]:
        seg = torch.randint(0, S, (N,), device=dev)
        w = torch.randn(N, D, device=dev, requires_grad=True)
        v = torch.randn(N, D, device=dev, requires_grad=True)
        w2 = w.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        ref = torch.zeros(S, D, device=dev).index_add_(0, seg, w * v)
        tri = weighted_segment_sum(w2, v2, seg, S)
        fwd_err = (ref - tri).abs().max().item()
        g = torch.randn_like(ref)
        ref.backward(g)
        tri.backward(g)
        bwd_err = max((w.grad - w2.grad).abs().max().item(), (v.grad - v2.grad).abs().max().item())
        print(f"N={N} S={S}: fwd max|d|={fwd_err:.2e}  bwd max|d|={bwd_err:.2e}")
