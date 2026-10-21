"""Deterministic CSR (weighted) segment sum (step 2.5a, D6).

Computes, for every segment ``s`` of a CSR over the rows,

    out[s, d] = sum_{j in [off[s], off[s + 1])} weight[r_j, d] * value[r_j, d]

(``value`` alone when ``weight`` is None), with ``r_j = j`` for an identity
CSR (rows sorted by segment) and ``r_j = perm[j]`` otherwise. Unlike the
atomic ``triton_weighted_segment_sum`` there are no atomics: a program owns
``BS`` consecutive segments, walks each run in ``[BR, BD]`` tiles, accumulates
a ``[BR, BD]`` register tile (row ``i`` of a run goes to slot ``i % BR``) and
reduces it with ``tl.sum`` at the end. The summation order is a function of
the CSR and the tile shape only, so results are bitwise reproducible.

Contract: the CSR partitions all ``N`` rows (``off[num_segment] == N``,
``perm`` a permutation of ``range(N)``, ``segment[perm[j]] == s`` for ``j`` in
the run of ``s``), as built by ``segment_attention_csr_metadata`` or by the
P>1 owner-CSR preparer. It is not checked (that would synchronize).

VJP (run merge): ``gvalue[r] = gout[s] * weight[r]`` (``gout[s]``
unweighted) and ``gweight[r] = gout[s] * value[r]``. The run-merge kernel walks
the same runs and loads ``gout[s]`` once per run; the row-parallel kernel reads
``segment[r]`` per row and gathers ``gout`` (also deterministic, no atomics).
Which one runs is a module constant per CSR kind (``_BWD_ROWS_*``, chosen by
the step 2.5a microbenchmark, see there; read at forward time). ``segment`` is
never copied: only the row VJP reads it (strided) and only then is it saved.

A skipped segment (``skip_segment``, e.g. the private capacity sink of a P>1
layout) is selected by the ``HAS_SKIP`` constexpr; its index is a runtime
value. Its output row is 0, its run is never walked and its rows get exactly
zero gradient (selected, not multiplied, so NaN/inf padding does not leak).

Row and segment counts are runtime arguments (``do_not_specialize``); the
launch passes ``static_under_compile`` like the other K5 kernels. Row offsets
are int64 (``perm`` may be int32 or int64); value/weight may be row-strided
views (strides are compile-time keys, no copy). Accumulation is FP32 (FP64
for FP64 inputs). Inference only: training runs plain torch.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function

from fastmd.engine import recorder as _recorder
from fastmd.engine.kernels.row_count import static_under_compile

# Tiles (step 2.5a microbenchmark on GB300, width 32, real RDX 1,344 / 10,752
# refinement CSRs; g_kernels_2p5a/scripts/bench_d6.py --tune, d6/tune1.json).
# Short runs (line graph, ~16 rows per segment): one [16, BD] tile per run on
# one warp. Long runs (atom graph, ~90 rows per segment): [32, BD] tiles on
# four warps. More segments per program (BS > 1) measured slower or equal.
# Warps scale with BD / 32 for wider rows.
_SHORT_RUN_ROWS = 24          # mean run length at or below which the short tiles apply
_SHORT_BS, _SHORT_BR, _SHORT_WARPS = 1, 16, 1
_LONG_BS, _LONG_BR, _LONG_WARPS = 1, 32, 4
_MAX_BD = 128
_MAX_WARPS = 8
# Backward: row-parallel gather VJP (True) or run-merge VJP (False), per CSR
# kind. Measured: equal within 2% on the identity (atom) CSR, so the run merge
# stays there; the row-parallel VJP is 1.5-4% faster on permutation CSRs
# (coalesced rows instead of permuted ones) and zeroes the skipped rows itself.
_BWD_ROWS_IDENTITY = False
_BWD_ROWS_PERM = True
_BWD_BM = 32
_BWD_WARPS = 4


@triton.jit(do_not_specialize=["num_segment", "skip_segment"])
def _csr_segsum_fwd(
    value,
    weight,
    perm,
    off,
    out,
    num_segment,
    skip_segment,
    D: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    W_S0: tl.constexpr,
    W_S1: tl.constexpr,
    BS: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
    IDENTITY_PERM: tl.constexpr,
    WEIGHTED: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    ACC_FP64: tl.constexpr,
):
    tl.assume(num_segment >= 0)
    acc_dtype = tl.float64 if ACC_FP64 else tl.float32
    base = tl.program_id(0).to(tl.int64) * BS
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    cmask = cols < D
    lanes = tl.arange(0, BR).to(tl.int64)
    for k in tl.static_range(BS):
        s = base + k
        live = s < num_segment
        start = tl.load(off + s, mask=live, other=0).to(tl.int64)
        end = tl.load(off + s + 1, mask=live, other=0).to(tl.int64)
        if HAS_SKIP:
            end = tl.where(s == skip_segment, start, end)
        acc = tl.zeros((BR, BD), acc_dtype)
        for j0 in range(start, end, BR):
            j = j0 + lanes
            rmask = j < end
            if IDENTITY_PERM:
                row = j
            else:
                row = tl.load(perm + j, mask=rmask, other=0).to(tl.int64)
            mask = rmask[:, None] & cmask[None, :]
            v = tl.load(value + row[:, None] * VAL_S0 + cols[None, :] * VAL_S1, mask=mask, other=0.0)
            if WEIGHTED:
                w = tl.load(weight + row[:, None] * W_S0 + cols[None, :] * W_S1, mask=mask, other=0.0)
                v = w * v
            acc += v.to(acc_dtype)
        total = tl.sum(acc, axis=0)
        tl.store(out + s * D + cols, total.to(out.dtype.element_ty), mask=cmask & live)


@triton.jit(do_not_specialize=["num_segment", "skip_segment"])
def _csr_segsum_bwd_runs(
    gout,
    value,
    weight,
    perm,
    off,
    gvalue,
    gweight,
    num_segment,
    skip_segment,
    D: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    W_S0: tl.constexpr,
    W_S1: tl.constexpr,
    BS: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
    IDENTITY_PERM: tl.constexpr,
    WEIGHTED: tl.constexpr,
    NEED_GV: tl.constexpr,
    NEED_GW: tl.constexpr,
    HAS_SKIP: tl.constexpr,
):
    # The skipped segment's rows are zero-initialized by the host (a parallel
    # memset) instead of walking a possibly huge sink run in one program.
    tl.assume(num_segment >= 0)
    base = tl.program_id(0).to(tl.int64) * BS
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    cmask = cols < D
    lanes = tl.arange(0, BR).to(tl.int64)
    for k in tl.static_range(BS):
        s = base + k
        live = s < num_segment
        start = tl.load(off + s, mask=live, other=0).to(tl.int64)
        end = tl.load(off + s + 1, mask=live, other=0).to(tl.int64)
        if HAS_SKIP:
            end = tl.where(s == skip_segment, start, end)
        go = tl.load(gout + s * D + cols, mask=cmask & live, other=0.0)
        for j0 in range(start, end, BR):
            j = j0 + lanes
            rmask = j < end
            if IDENTITY_PERM:
                row = j
            else:
                row = tl.load(perm + j, mask=rmask, other=0).to(tl.int64)
            mask = rmask[:, None] & cmask[None, :]
            dst = row[:, None] * D + cols[None, :]
            if NEED_GV:
                if WEIGHTED:
                    w = tl.load(weight + row[:, None] * W_S0 + cols[None, :] * W_S1, mask=mask, other=0.0)
                    tl.store(gvalue + dst, go[None, :] * w, mask=mask)
                else:
                    tl.store(gvalue + dst, tl.broadcast_to(go[None, :], (BR, BD)), mask=mask)
            if NEED_GW:
                v = tl.load(value + row[:, None] * VAL_S0 + cols[None, :] * VAL_S1, mask=mask, other=0.0)
                tl.store(gweight + dst, go[None, :] * v, mask=mask)


@triton.jit(do_not_specialize=["N", "skip_segment"])
def _csr_segsum_bwd_rows(
    gout,
    value,
    weight,
    segment,
    gvalue,
    gweight,
    N,
    skip_segment,
    D: tl.constexpr,
    SEG_S0: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    W_S0: tl.constexpr,
    W_S1: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    WEIGHTED: tl.constexpr,
    NEED_GV: tl.constexpr,
    NEED_GW: tl.constexpr,
    HAS_SKIP: tl.constexpr,
):
    tl.assume(N >= 0)
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    # segment may be a column view of the edge-index matrix (stride 2 or 5): no copy.
    seg = tl.load(segment + rows * SEG_S0, mask=rows < N, other=0).to(tl.int64)
    go = tl.load(gout + seg[:, None] * D + cols[None, :], mask=mask, other=0.0)
    dst = rows[:, None] * D + cols[None, :]
    if HAS_SKIP:
        live = (seg != skip_segment)[:, None]
    if NEED_GV:
        if WEIGHTED:
            w = tl.load(weight + rows[:, None] * W_S0 + cols[None, :] * W_S1, mask=mask, other=0.0)
            gv = go * w
        else:
            gv = go
        if HAS_SKIP:
            gv = tl.where(live, gv, 0.0)
        tl.store(gvalue + dst, gv, mask=mask)
    if NEED_GW:
        v = tl.load(value + rows[:, None] * VAL_S0 + cols[None, :] * VAL_S1, mask=mask, other=0.0)
        gw = go * v
        if HAS_SKIP:
            gw = tl.where(live, gw, 0.0)
        tl.store(gweight + dst, gw, mask=mask)


def _tiles(n_rows: int, num_segment: int, dim: int, *, forward: bool = False) -> tuple[int, int, int, int]:
    """(BS, BR, BD, num_warps) of the forward (``forward``) / run-merge kernels.

    The forward always takes the short-run tile: in GPU time (CUDA-graph
    replay) it is also the fastest forward on the long atom-graph runs
    (step 2.5a review, d6_fix). The run-merge VJP keeps the run-length split.
    The width is specialized under torch.compile: ``num_warps`` is a launch
    option, which dynamo needs as a constant (not a symbolic size).
    """
    bd = min(triton.next_power_of_2(static_under_compile(dim)), _MAX_BD)
    scale = max(bd // 32, 1)
    if forward or n_rows <= _SHORT_RUN_ROWS * max(num_segment, 1):
        return _SHORT_BS, _SHORT_BR, bd, min(_SHORT_WARPS * scale, _MAX_WARPS)
    return _LONG_BS, _LONG_BR, bd, min(_LONG_WARPS * scale, _MAX_WARPS)


def _op_name(weight) -> str:
    return "segment_sum" if weight is None else "weighted_segment_sum"


class _CsrSegmentSum(Function):
    @staticmethod
    def forward(ctx, value, weight, segment, num_segment, perm, off, identity_perm, skip_segment):
        if value.dim() != 2:
            raise ValueError(f"value must be 2D, got shape {tuple(value.shape)}")
        if weight is not None and (weight.shape != value.shape or weight.dtype != value.dtype):
            raise ValueError(
                f"weight must match value, got {tuple(weight.shape)}/{weight.dtype} and "
                f"{tuple(value.shape)}/{value.dtype}"
            )
        n_rows, dim = value.shape
        if segment.dim() != 1 or segment.numel() != n_rows:
            raise ValueError(f"segment must have one entry per row ({n_rows}), got {tuple(segment.shape)}")
        if any(t is not None and t.device != value.device for t in (weight, segment, perm, off)):
            raise ValueError("value, weight, segment and the CSR must be on the same device")
        num_segment = int(num_segment)
        skip_segment = int(skip_segment)
        if skip_segment < -1 or skip_segment >= num_segment:
            raise ValueError("skip_segment must be -1 or an existing private segment")
        identity_perm = bool(identity_perm)
        if identity_perm and off is None:
            raise ValueError("identity CSR metadata requires offsets")
        rec = _recorder._ACTIVE
        if rec is not None:
            index_dtype = segment.dtype
        if not identity_perm and (perm is None or off is None):
            from fastmd.models.matris.op.triton_segment_attention import segment_attention_csr_metadata

            perm, off = segment_attention_csr_metadata(segment, num_segment)
        # segment is never copied: the forward does not read it and the row
        # VJP loads it strided; only a CSR kind whose VJP is row-parallel keeps it.
        copied = False
        if off.dtype != torch.int64 or not off.is_contiguous():
            off, copied = off.to(torch.int64).contiguous(), True
        if not identity_perm and not perm.is_contiguous():
            perm, copied = perm.contiguous(), True
        if identity_perm:
            perm = off  # unused pointer argument
        rows_vjp = _BWD_ROWS_IDENTITY if identity_perm else _BWD_ROWS_PERM
        bs, br, bd, warps = _tiles(n_rows, num_segment, dim, forward=True)
        val_s0, val_s1 = value.stride()
        w_s0, w_s1 = (val_s0, val_s1) if weight is None else weight.stride()
        out = torch.empty((num_segment, dim), device=value.device, dtype=value.dtype)
        if num_segment > 0:
            grid = (triton.cdiv(num_segment, bs), triton.cdiv(dim, bd))
            _csr_segsum_fwd[grid](
                value,
                value if weight is None else weight,
                perm,
                off,
                out,
                static_under_compile(num_segment),
                skip_segment,
                D=dim,
                VAL_S0=val_s0,
                VAL_S1=val_s1,
                W_S0=w_s0,
                W_S1=w_s1,
                BS=bs,
                BR=br,
                BD=bd,
                IDENTITY_PERM=identity_perm,
                WEIGHTED=weight is not None,
                HAS_SKIP=skip_segment >= 0,
                ACC_FP64=value.dtype == torch.float64,
                num_warps=warps,
            )
        ctx.save_for_backward(value, weight, segment if rows_vjp else None, perm, off)
        ctx.rows_vjp = rows_vjp
        ctx.identity_perm = identity_perm
        ctx.skip_segment = skip_segment
        ctx.num_segment = num_segment
        if rec is not None:
            ctx._opsel_site = rec.site()
            params = {
                "rows": n_rows, "width": dim, "segments": num_segment, "tiles": [bs, br, bd],
                "num_warps": warps, "dtype": value.dtype, "index_dtype": index_dtype,
                "offsets_dtype": off.dtype, "IDENTITY": identity_perm, "contiguous_copy": copied,
            }
            if skip_segment != -1:
                params["SKIP_SEGMENT"] = skip_segment
            rec.record(_op_name(weight), "triton_csr", params)
        return out

    @staticmethod
    def backward(ctx, gout):
        value, weight, segment, perm, off = ctx.saved_tensors
        need_gv, need_gw = ctx.needs_input_grad[0], ctx.needs_input_grad[1] and weight is not None
        if not (need_gv or need_gw):
            return None, None, None, None, None, None, None, None
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = not gout.is_contiguous()
        gout = gout.contiguous()
        n_rows, dim = value.shape
        num_segment, skip_segment = ctx.num_segment, ctx.skip_segment
        has_skip = skip_segment >= 0
        weighted = weight is not None
        val_s0, val_s1 = value.stride()
        w_s0, w_s1 = (val_s0, val_s1) if weight is None else weight.stride()
        rows_vjp = ctx.rows_vjp
        # Run merge never visits the skipped run: zero its rows up front.
        allocate = torch.zeros if has_skip and not rows_vjp else torch.empty
        gvalue = allocate((n_rows, dim), device=value.device, dtype=value.dtype) if need_gv else None
        gweight = allocate((n_rows, dim), device=value.device, dtype=value.dtype) if need_gw else None
        # Unused pointer arguments alias an existing tensor.
        gv_arg = gvalue if need_gv else gweight
        gw_arg = gweight if need_gw else gvalue
        weight_arg = value if weight is None else weight
        if rows_vjp:
            bm = _BWD_BM
            bd = min(triton.next_power_of_2(dim), _MAX_BD)
            warps = _BWD_WARPS
            tiles = [bm, bd]
            if n_rows > 0:
                _csr_segsum_bwd_rows[(triton.cdiv(n_rows, bm), triton.cdiv(dim, bd))](
                    gout,
                    value,
                    weight_arg,
                    segment,
                    gv_arg,
                    gw_arg,
                    static_under_compile(n_rows),
                    skip_segment,
                    D=dim,
                    SEG_S0=segment.stride(0),
                    VAL_S0=val_s0,
                    VAL_S1=val_s1,
                    W_S0=w_s0,
                    W_S1=w_s1,
                    BM=bm,
                    BD=bd,
                    WEIGHTED=weighted,
                    NEED_GV=need_gv,
                    NEED_GW=need_gw,
                    HAS_SKIP=has_skip,
                    num_warps=warps,
                )
        else:
            bs, br, bd, warps = _tiles(n_rows, num_segment, dim)
            tiles = [bs, br, bd]
            if num_segment > 0:
                _csr_segsum_bwd_runs[(triton.cdiv(num_segment, bs), triton.cdiv(dim, bd))](
                    gout,
                    value,
                    weight_arg,
                    perm,
                    off,
                    gv_arg,
                    gw_arg,
                    static_under_compile(num_segment),
                    skip_segment,
                    D=dim,
                    VAL_S0=val_s0,
                    VAL_S1=val_s1,
                    W_S0=w_s0,
                    W_S1=w_s1,
                    BS=bs,
                    BR=br,
                    BD=bd,
                    IDENTITY_PERM=ctx.identity_perm,
                    WEIGHTED=weighted,
                    NEED_GV=need_gv,
                    NEED_GW=need_gw,
                    HAS_SKIP=has_skip,
                    num_warps=warps,
                )
        if rec is not None:
            params = {
                "rows": n_rows, "width": dim, "segments": num_segment, "tiles": tiles,
                "num_warps": warps, "vjp": "rows" if rows_vjp else "runs",
                "IDENTITY": ctx.identity_perm, "dtype": value.dtype, "contiguous_copy": copied,
            }
            if skip_segment != -1:
                params["SKIP_SEGMENT"] = skip_segment
            rec.record(_op_name(weight), "triton_csr", params,
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return gvalue, gweight, None, None, None, None, None, None


def csr_segment_sum(
    value: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int,
    perm: torch.Tensor | None = None,
    off: torch.Tensor | None = None,
    identity_perm: bool = False,
    *,
    weight: torch.Tensor | None = None,
    skip_segment: int = -1,
) -> torch.Tensor:
    """``out[s] = sum_{segment[i] == s} [weight[i] *] value[i]`` over a CSR, deterministic.

    ``perm``/``off``: the CSR of ``segment`` (``off`` int64 [num_segment + 1];
    ``perm`` int32/int64 [N], unused if ``identity_perm``); built from
    ``segment`` when missing. ``skip_segment``: -1 or a segment whose output
    is 0 and whose rows get zero gradient. CPU tensors use the torch
    reference (``index_add``).
    """
    if value.is_cuda and (weight is None or weight.is_cuda):
        return _CsrSegmentSum.apply(
            value,
            weight,
            segment,
            int(num_segment),
            perm,
            off,
            bool(identity_perm),
            int(skip_segment),
        )
    num_segment = int(num_segment)
    skip_segment = int(skip_segment)
    if skip_segment < -1 or skip_segment >= num_segment:
        raise ValueError("skip_segment must be -1 or an existing private segment")
    rec = _recorder._ACTIVE
    if rec is not None:
        params = {"rows": value.shape[0], "width": value.shape[-1], "segments": num_segment}
        if skip_segment != -1:
            params["SKIP_SEGMENT"] = skip_segment
        rec.record(_op_name(weight), "torch", params)
    if skip_segment != -1:
        # Select the operands (not the product): NaN padding gets no gradient path.
        skipped = (segment == skip_segment)[:, None]
        value = torch.where(skipped, 0.0, value)
        weight = None if weight is None else torch.where(skipped, 0.0, weight)
    rows = value if weight is None else weight * value
    return rows.new_zeros((num_segment, rows.shape[1])).index_add_(0, segment, rows)
