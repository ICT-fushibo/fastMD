"""Fused segmented attention aggregation.

Computes:

    out[s, d] = sum_i softmax(logits[i, d] over rows with segment[i] == s)
                      * value[i, d]

This fuses the existing ``Dimwise_softmax(logits, segment)`` plus
``weighted_segment_sum(alpha, value, segment)`` chain and avoids materializing
the row-wise attention weights.

Row and segment counts are runtime arguments excluded from Triton's integer
value specialization (``do_not_specialize``), so eager steps with new counts
reuse the compiled kernels: the paired backward row count ``N`` and the
skipped segment index. A skipped segment (``skip_segment``, e.g. a private
capacity sink at ``count - 1``) is selected by the ``HAS_SKIP`` constexpr; its
index is a runtime value. ``HAS_SKIP=False`` removes every skip branch, so the
default path compiles to the same code as before the argument existed.
Compile-time keys are the width, strides, tiles and mode flags.

``return_lse=True`` also returns the per-segment, per-channel log-sum-exp
``lse[s, d] = max + log(denom)`` of the same forward pass (-inf for an empty
or skipped segment), the ``(lse, y)`` partial of
:mod:`fastmd.models.matris.op.segment_partials`. Its cotangent adds
``alpha * grad_lse`` to the logits gradient (``d lse / d logit = alpha``).
The ``STORE_LSE``/``HAS_GLSE`` constexprs select it and the lse pointers are
``None`` (a constexpr) otherwise, so the default kernels are unchanged.
"""
from __future__ import annotations

from fastmd.config import env_value

import torch
import triton
import triton.language as tl
from torch.autograd import Function

from fastmd.engine import recorder as _recorder
from fastmd.engine import saved_recompute as _saved_recompute
from fastmd.engine.kernels.merged_frozen_projections import claim_grad, grad_claim
from fastmd.engine.kernels.row_count import static_under_compile
from fastmd.models.matris.op.segment_partials import segment_logsumexp

# None: by width (C10), one warp per 32 columns of the padded row, at most 4.
# A program walks its segment's rows over one row vector; with more warps than
# the row has 32-column chunks, the extra warps recompute the same vector and
# only halve the resident programs (32-wide line-graph rows take one warp,
# 128-wide rows keep four).
_NUM_WARPS = env_value("MATRIS_FUSED_SEGMENT_ATTENTION_NUM_WARPS")
_NUM_WARPS = None if _NUM_WARPS is None else int(_NUM_WARPS)
_PAIRED_BWD_BM = int(env_value("MATRIS_FUSED_PAIRED_SEGMENT_ATTENTION_BWD_BM", "8"))
_PAIRED_BWD_BD = int(env_value("MATRIS_FUSED_PAIRED_SEGMENT_ATTENTION_BWD_BD", "128"))


def _num_warps(blk: int) -> int:
    """Warps per program for a padded row of ``blk`` columns (a launch
    parameter: specialized when traced by dynamo)."""
    return _NUM_WARPS or max(1, min(4, static_under_compile(blk) // 32))


@triton.jit(do_not_specialize=["skip_segment"])
def _csr_segment_attention_fwd(
    logits,
    value,
    perm,
    off,
    out,
    lse,
    skip_segment,
    D: tl.constexpr,
    LOG_S0: tl.constexpr,
    LOG_S1: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    BLK: tl.constexpr,
    IDENTITY_PERM: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    # 64-bit rows in every kernel here: an identity-CSR row is the loop index
    # itself, and merged alpha views have LOG_S0 == 128 (wraps at 2**24 rows).
    s = tl.program_id(0).to(tl.int64)
    start = tl.load(off + s).to(tl.int64)
    end = tl.load(off + s + 1).to(tl.int64)
    if HAS_SKIP:
        # Private capacity sinks can contain many thousands of padding rows.
        # They are disconnected from all real atoms, so do not serialize that
        # dummy segment through the CSR recurrence.
        end = tl.where(s == skip_segment, start, end)
    cols = tl.arange(0, BLK)
    mask = cols < D
    dtype = logits.dtype.element_ty

    m = tl.full((BLK,), -float("inf"), dtype)
    denom = tl.zeros((BLK,), dtype)
    acc = tl.zeros((BLK,), dtype)

    for j in range(start, end):
        if IDENTITY_PERM:
            row = j
        else:
            row = tl.load(perm + j).to(tl.int64)
        logit = tl.load(
            logits + row * LOG_S0 + cols * LOG_S1,
            mask=mask,
            other=-float("inf"),
        )
        val = tl.load(
            value + row * VAL_S0 + cols * VAL_S1,
            mask=mask,
            other=0.0,
        ).to(dtype)
        m_new = tl.maximum(m, logit)
        old_scale = tl.exp(m - m_new)
        new_scale = tl.exp(logit - m_new)
        denom = denom * old_scale + new_scale
        acc = acc * old_scale + new_scale * val
        m = m_new

    result = tl.where(denom > 0.0, acc / denom, 0.0)
    tl.store(out + s * D + cols, result, mask=mask)
    if STORE_LSE:
        # An empty (or skipped) segment has m = -inf and denom = 0: -inf.
        tl.store(lse + s * D + cols, tl.where(denom > 0.0, m + tl.log(denom), -float("inf")), mask=mask)


@triton.jit(do_not_specialize=["skip_segment"])
def _csr_segment_attention_bwd(
    logits,
    value,
    out,
    gout,
    perm,
    off,
    grad_logits,
    grad_value,
    glse,
    skip_segment,
    D: tl.constexpr,
    LOG_S0: tl.constexpr,
    LOG_S1: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    GL_S0: tl.constexpr,
    BLK: tl.constexpr,
    IDENTITY_PERM: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    HAS_GLSE: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    start = tl.load(off + s).to(tl.int64)
    end = tl.load(off + s + 1).to(tl.int64)
    if HAS_SKIP:
        end = tl.where(s == skip_segment, start, end)
    cols = tl.arange(0, BLK)
    mask = cols < D
    dtype = logits.dtype.element_ty

    m = tl.full((BLK,), -float("inf"), dtype)
    denom = tl.zeros((BLK,), dtype)
    for j in range(start, end):
        if IDENTITY_PERM:
            row = j
        else:
            row = tl.load(perm + j).to(tl.int64)
        logit = tl.load(
            logits + row * LOG_S0 + cols * LOG_S1,
            mask=mask,
            other=-float("inf"),
        )
        m_new = tl.maximum(m, logit)
        denom = denom * tl.exp(m - m_new) + tl.exp(logit - m_new)
        m = m_new

    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    go = tl.load(gout + s * D + cols, mask=mask, other=0.0).to(dtype)
    out_s = tl.load(out + s * D + cols, mask=mask, other=0.0).to(dtype)
    if HAS_GLSE:
        # Only rows of this segment read it: the -inf lse of an empty or
        # skipped segment never meets a row.
        gl = tl.load(glse + s * D + cols, mask=mask, other=0.0).to(dtype)

    for j in range(start, end):
        if IDENTITY_PERM:
            row = j
        else:
            row = tl.load(perm + j).to(tl.int64)
        logit = tl.load(
            logits + row * LOG_S0 + cols * LOG_S1,
            mask=mask,
            other=-float("inf"),
        )
        val = tl.load(
            value + row * VAL_S0 + cols * VAL_S1,
            mask=mask,
            other=0.0,
        ).to(dtype)
        alpha = tl.exp(logit - m) * inv_denom
        offs = row * D + cols
        # GL_S0: D, or the row stride of a merged projection's gradient buffer.
        offs_g = row * GL_S0 + cols
        tl.store(grad_value + offs, alpha * go, mask=mask)
        if HAS_GLSE:
            tl.store(grad_logits + offs_g, alpha * go * (val - out_s) + alpha * gl, mask=mask)
        else:
            tl.store(grad_logits + offs_g, alpha * go * (val - out_s), mask=mask)


@triton.jit(do_not_specialize=["skip_segment"])
def _paired_csr_segment_attention_fwd(
    logits_a,
    logits_b,
    value,
    perm_a,
    off_a,
    perm_b,
    off_b,
    out_a,
    out_b,
    max_a,
    denom_a,
    max_b,
    denom_b,
    lse_a,
    lse_b,
    skip_segment,
    D: tl.constexpr,
    LOGA_S0: tl.constexpr,
    LOGA_S1: tl.constexpr,
    LOGB_S0: tl.constexpr,
    LOGB_S1: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    BLK: tl.constexpr,
    IDENTITY_A: tl.constexpr,
    IDENTITY_B: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    # A skipped segment runs empty loops on both sides, exactly like an empty
    # segment: out 0, max -inf, denom 0 (the backward masks its rows).
    s = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLK)
    mask = cols < D
    dtype = logits_a.dtype.element_ty

    start = tl.load(off_a + s).to(tl.int64)
    end = tl.load(off_a + s + 1).to(tl.int64)
    if HAS_SKIP:
        end = tl.where(s == skip_segment, start, end)
    m = tl.full((BLK,), -float("inf"), dtype)
    denom = tl.zeros((BLK,), dtype)
    acc = tl.zeros((BLK,), dtype)
    for j in range(start, end):
        if IDENTITY_A:
            row = j
        else:
            row = tl.load(perm_a + j).to(tl.int64)
        logit = tl.load(
            logits_a + row * LOGA_S0 + cols * LOGA_S1,
            mask=mask,
            other=-float("inf"),
        )
        val = tl.load(
            value + row * VAL_S0 + cols * VAL_S1,
            mask=mask,
            other=0.0,
        ).to(dtype)
        m_new = tl.maximum(m, logit)
        old_scale = tl.exp(m - m_new)
        new_scale = tl.exp(logit - m_new)
        denom = denom * old_scale + new_scale
        acc = acc * old_scale + new_scale * val
        m = m_new
    result = tl.where(denom > 0.0, acc / denom, 0.0)
    tl.store(out_a + s * D + cols, result, mask=mask)
    tl.store(max_a + s * D + cols, m, mask=mask)
    tl.store(denom_a + s * D + cols, denom, mask=mask)
    if STORE_LSE:
        tl.store(lse_a + s * D + cols, tl.where(denom > 0.0, m + tl.log(denom), -float("inf")), mask=mask)

    start = tl.load(off_b + s).to(tl.int64)
    end = tl.load(off_b + s + 1).to(tl.int64)
    if HAS_SKIP:
        end = tl.where(s == skip_segment, start, end)
    m = tl.full((BLK,), -float("inf"), dtype)
    denom = tl.zeros((BLK,), dtype)
    acc = tl.zeros((BLK,), dtype)
    for j in range(start, end):
        if IDENTITY_B:
            row = j
        else:
            row = tl.load(perm_b + j).to(tl.int64)
        logit = tl.load(
            logits_b + row * LOGB_S0 + cols * LOGB_S1,
            mask=mask,
            other=-float("inf"),
        )
        val = tl.load(
            value + row * VAL_S0 + cols * VAL_S1,
            mask=mask,
            other=0.0,
        ).to(dtype)
        m_new = tl.maximum(m, logit)
        old_scale = tl.exp(m - m_new)
        new_scale = tl.exp(logit - m_new)
        denom = denom * old_scale + new_scale
        acc = acc * old_scale + new_scale * val
        m = m_new
    result = tl.where(denom > 0.0, acc / denom, 0.0)
    tl.store(out_b + s * D + cols, result, mask=mask)
    tl.store(max_b + s * D + cols, m, mask=mask)
    tl.store(denom_b + s * D + cols, denom, mask=mask)
    if STORE_LSE:
        tl.store(lse_b + s * D + cols, tl.where(denom > 0.0, m + tl.log(denom), -float("inf")), mask=mask)


@triton.jit(do_not_specialize=["N", "skip_segment"])
def _paired_segment_attention_bwd_rows(
    logits_a,
    logits_b,
    value,
    gout_a,
    gout_b,
    out_a,
    out_b,
    max_a,
    denom_a,
    max_b,
    denom_b,
    segment_a,
    segment_b,
    grad_logits_a,
    grad_logits_b,
    grad_value,
    glse_a,
    glse_b,
    N,
    skip_segment,
    D: tl.constexpr,
    LOGA_S0: tl.constexpr,
    LOGA_S1: tl.constexpr,
    LOGB_S0: tl.constexpr,
    LOGB_S1: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    GLA_S0: tl.constexpr,
    GLB_S0: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    HAS_GLSE: tl.constexpr,
):
    # N is a runtime count (K5). A known non-negative N keeps the row mask an
    # unsigned 64-bit compare like the former constexpr N; without the hint
    # the sign-extended compare delays the first loads (+1.6% at 121k rows).
    tl.assume(N >= 0)
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    seg_a = tl.load(segment_a + rows, mask=rows < N, other=0).to(tl.int64)
    seg_b = tl.load(segment_b + rows, mask=rows < N, other=0).to(tl.int64)

    logit_a = tl.load(
        logits_a + rows[:, None] * LOGA_S0 + cols[None, :] * LOGA_S1,
        mask=mask,
        other=-float("inf"),
    )
    logit_b = tl.load(
        logits_b + rows[:, None] * LOGB_S0 + cols[None, :] * LOGB_S1,
        mask=mask,
        other=-float("inf"),
    )
    val = tl.load(
        value + rows[:, None] * VAL_S0 + cols[None, :] * VAL_S1,
        mask=mask,
        other=0.0,
    )

    stat_a = seg_a[:, None] * D + cols[None, :]
    stat_b = seg_b[:, None] * D + cols[None, :]
    ma = tl.load(max_a + stat_a, mask=mask, other=-float("inf"))
    da = tl.load(denom_a + stat_a, mask=mask, other=0.0)
    mb = tl.load(max_b + stat_b, mask=mask, other=-float("inf"))
    db = tl.load(denom_b + stat_b, mask=mask, other=0.0)

    alpha_a = tl.exp(logit_a - ma) * tl.where(da > 0.0, 1.0 / da, 0.0)
    alpha_b = tl.exp(logit_b - mb) * tl.where(db > 0.0, 1.0 / db, 0.0)
    go_a = tl.load(gout_a + stat_a, mask=mask, other=0.0)
    go_b = tl.load(gout_b + stat_b, mask=mask, other=0.0)
    out_a_row = tl.load(out_a + stat_a, mask=mask, other=0.0)
    out_b_row = tl.load(out_b + stat_b, mask=mask, other=0.0)

    offs = rows[:, None] * D + cols[None, :]
    # The logit gradients may be columns of a merged projection's gradient
    # buffer (GL*_S0 its row stride); D when they are tensors of their own.
    offs_ga = rows[:, None] * GLA_S0 + cols[None, :]
    offs_gb = rows[:, None] * GLB_S0 + cols[None, :]
    if HAS_GLSE:
        gl_a = tl.load(glse_a + stat_a, mask=mask, other=0.0)
        gl_b = tl.load(glse_b + stat_b, mask=mask, other=0.0)
    if HAS_SKIP:
        # Rows of the skipped segment see its empty-loop stats (max -inf,
        # denom 0): alpha = exp(logit + inf) * 0 = NaN. Their side contributes
        # exactly zero, as in segment_attention, whose backward never visits
        # them; select (not multiply) so no gout/value/NaN leaks through.
        live_a = (seg_a != skip_segment)[:, None]
        live_b = (seg_b != skip_segment)[:, None]
        term_a = tl.where(live_a, alpha_a * go_a, 0.0)
        term_b = tl.where(live_b, alpha_b * go_b, 0.0)
        # The else branches are the HAS_GLSE-free program, statement for
        # statement (a reordered statement reschedules the compiled code).
        if HAS_GLSE:  # the NaN alpha of a skipped row is selected away
            tl.store(grad_logits_a + offs_ga,
                     tl.where(live_a, term_a * (val - out_a_row) + alpha_a * gl_a, 0.0), mask=mask)
            tl.store(grad_logits_b + offs_gb,
                     tl.where(live_b, term_b * (val - out_b_row) + alpha_b * gl_b, 0.0), mask=mask)
        else:
            tl.store(grad_logits_a + offs_ga, tl.where(live_a, term_a * (val - out_a_row), 0.0), mask=mask)
            tl.store(grad_logits_b + offs_gb, tl.where(live_b, term_b * (val - out_b_row), 0.0), mask=mask)
        tl.store(grad_value + offs, term_a + term_b, mask=mask)
    else:
        if HAS_GLSE:
            tl.store(grad_logits_a + offs_ga, alpha_a * go_a * (val - out_a_row) + alpha_a * gl_a, mask=mask)
            tl.store(grad_logits_b + offs_gb, alpha_b * go_b * (val - out_b_row) + alpha_b * gl_b, mask=mask)
        else:
            tl.store(grad_logits_a + offs_ga, alpha_a * go_a * (val - out_a_row), mask=mask)
            tl.store(grad_logits_b + offs_gb, alpha_b * go_b * (val - out_b_row), mask=mask)
        tl.store(grad_value + offs, alpha_a * go_a + alpha_b * go_b, mask=mask)


class _SegmentAttention(Function):
    @staticmethod
    def forward(ctx, logits, value, segment, num_segment, perm, off, identity_perm, skip_segment,
                return_lse):
        if logits.shape != value.shape:
            raise ValueError(
                f"logits and value must have the same shape, got {logits.shape} and {value.shape}"
            )
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got shape {logits.shape}")
        if not torch.compiler.is_compiling():
            ctx._grad_claim = grad_claim(logits)  # a merged projection's logits (C3)
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = [logits.stride(1) != 1, value.stride(1) != 1]
            index_dtype = segment.dtype
            csr = ("identity" if identity_perm else
                   "provided" if perm is not None and off is not None else "computed")
        logits = logits.contiguous() if logits.stride(1) != 1 else logits
        value = value.contiguous() if value.stride(1) != 1 else value
        segment = segment.to(torch.int64)
        n_rows, dim = logits.shape
        num_segment = int(num_segment)
        skip_segment = int(skip_segment)
        if skip_segment < -1 or skip_segment >= num_segment:
            raise ValueError("skip_segment must be -1 or an existing private segment")
        log_s0, log_s1 = logits.stride()
        val_s0, val_s1 = value.stride()
        blk = triton.next_power_of_2(dim)

        identity_perm = bool(identity_perm)
        if identity_perm and off is None:
            raise ValueError("identity CSR metadata requires offsets")
        if not identity_perm and (perm is None or off is None):
            perm, off = segment_attention_csr_metadata(segment, num_segment)
        if identity_perm:
            perm = segment
        out = torch.empty((num_segment, dim), device=logits.device, dtype=logits.dtype)
        lse = torch.empty_like(out) if return_lse else None
        _csr_segment_attention_fwd[(num_segment,)](
            logits,
            value,
            perm,
            off,
            out,
            lse,
            skip_segment,
            D=dim,
            LOG_S0=log_s0,
            LOG_S1=log_s1,
            VAL_S0=val_s0,
            VAL_S1=val_s1,
            BLK=blk,
            IDENTITY_PERM=identity_perm,
            HAS_SKIP=skip_segment >= 0,
            STORE_LSE=return_lse,
            num_warps=_num_warps(blk),
        )
        ctx.save_for_backward(logits, value, perm, off, out)
        ctx.identity_perm = identity_perm
        ctx.skip_segment = skip_segment
        ctx.return_lse = return_lse
        # A provided CSR may cover a subset of the rows (a folded row
        # selection, perm' = rows[perm]); the backward never visits the rest.
        ctx.partial_rows = not identity_perm and perm.numel() != n_rows
        if rec is not None:
            ctx._opsel_site = rec.site()
            params = {
                "rows": n_rows, "width": dim, "segments": num_segment, "BLK": blk,
                "num_warps": _num_warps(blk), "IDENTITY_PERM": identity_perm,
                "SKIP_SEGMENT": skip_segment, "LOG_S": [log_s0, log_s1],
                "VAL_S": [val_s0, val_s1], "csr": csr, "dtype": logits.dtype,
                "index_dtype": index_dtype, "offsets_dtype": off.dtype,
                "contiguous_copy": copied,
            }
            if return_lse:  # new key only when set: default snapshots are unchanged
                params["LSE"] = True
            rec.record("segment_attention", "triton_csr", params)
        if return_lse:
            return out, lse
        return out

    @staticmethod
    def backward(ctx, gout, glse=None):
        logits, value, perm, off, out = ctx.saved_tensors
        identity_perm = ctx.identity_perm
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = not gout.is_contiguous()
        gout = gout.contiguous()
        if glse is not None:
            glse = glse.contiguous()
        n_rows, dim = logits.shape
        log_s0, log_s1 = logits.stride()
        val_s0, val_s1 = value.stride()
        blk = triton.next_power_of_2(dim)
        # The skipped sink's rows are never visited by the backward kernel.
        # Explicit zeros avoid uninitialized padding adjoints; initialization
        # is parallel and does not loop over a giant sink in one CTA.
        zero_init = ctx.skip_segment >= 0 or ctx.partial_rows
        allocate = torch.zeros if zero_init else torch.empty
        if zero_init:  # unvisited rows: a merged projection's buffer would leave them unwritten
            grad_logits = allocate((n_rows, dim), device=logits.device, dtype=logits.dtype)
        else:
            grad_logits = claim_grad(getattr(ctx, "_grad_claim", None), (n_rows, dim), logits.device, logits.dtype)
        grad_value = allocate((n_rows, dim), device=value.device, dtype=value.dtype)
        _csr_segment_attention_bwd[(out.shape[0],)](
            logits,
            value,
            out,
            gout,
            perm,
            off,
            grad_logits,
            grad_value,
            glse,
            ctx.skip_segment,
            D=dim,
            LOG_S0=log_s0,
            LOG_S1=log_s1,
            VAL_S0=val_s0,
            VAL_S1=val_s1,
            GL_S0=grad_logits.stride(0),
            BLK=blk,
            IDENTITY_PERM=identity_perm,
            HAS_SKIP=ctx.skip_segment >= 0,
            HAS_GLSE=glse is not None,
            num_warps=_num_warps(blk),
        )
        if rec is not None:
            params = {
                "rows": n_rows, "width": dim, "segments": out.shape[0], "BLK": blk,
                "num_warps": _num_warps(blk), "IDENTITY_PERM": identity_perm,
                "SKIP_SEGMENT": ctx.skip_segment, "LOG_S": [log_s0, log_s1],
                "VAL_S": [val_s0, val_s1], "zero_init": ctx.skip_segment >= 0,
                "contiguous_copy": copied,
            }
            if grad_logits.stride(0) != dim:
                params["GL_S0"] = grad_logits.stride(0)
            if ctx.return_lse:
                params["LSE"] = True
            rec.record("segment_attention", "triton_csr", params,
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return grad_logits, grad_value, None, None, None, None, None, None, None


class _PairedSegmentAttention(Function):
    @staticmethod
    def forward(
        ctx,
        logits_a,
        logits_b,
        value,
        segment_a,
        segment_b,
        num_segment,
        perm_a,
        off_a,
        perm_b,
        off_b,
        identity_a,
        identity_b,
        skip_segment,
        return_lse,
    ):
        if logits_a.shape != logits_b.shape or logits_a.shape != value.shape:
            raise ValueError(
                "logits_a, logits_b, and value must have the same shape, got "
                f"{logits_a.shape}, {logits_b.shape}, {value.shape}"
            )
        if logits_a.dim() != 2:
            raise ValueError(f"logits must be 2D, got {logits_a.shape}")
        if not torch.compiler.is_compiling():
            # Merged-projection logits: their gradients go into its buffer (C3).
            ctx._grad_claims = (grad_claim(logits_a), grad_claim(logits_b))
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = [logits_a.stride(1) != 1, logits_b.stride(1) != 1, value.stride(1) != 1]
            index_dtype = [segment_a.dtype, segment_b.dtype]
            csr = [("identity" if identity else
                    "provided" if perm is not None and off is not None else "computed")
                   for identity, perm, off in ((identity_a, perm_a, off_a),
                                               (identity_b, perm_b, off_b))]
        logits_a = logits_a.contiguous() if logits_a.stride(1) != 1 else logits_a
        logits_b = logits_b.contiguous() if logits_b.stride(1) != 1 else logits_b
        value = value.contiguous() if value.stride(1) != 1 else value
        source_a, source_b = segment_a, segment_b
        segment_a = segment_a.to(torch.int64).contiguous()
        segment_b = segment_b.to(torch.int64).contiguous()
        _saved_recompute.register_copy(segment_a, source_a, int64=True)
        _saved_recompute.register_copy(segment_b, source_b, int64=True)
        n_rows, dim = logits_a.shape
        num_segment = int(num_segment)
        skip_segment = int(skip_segment)
        if skip_segment < -1 or skip_segment >= num_segment:
            raise ValueError("skip_segment must be -1 or an existing private segment")
        loga_s0, loga_s1 = logits_a.stride()
        logb_s0, logb_s1 = logits_b.stride()
        val_s0, val_s1 = value.stride()
        blk = triton.next_power_of_2(dim)

        identity_a = bool(identity_a)
        identity_b = bool(identity_b)
        if identity_a and off_a is None:
            raise ValueError("identity CSR metadata for segment_a requires offsets")
        if identity_b and off_b is None:
            raise ValueError("identity CSR metadata for segment_b requires offsets")
        if not identity_a and (perm_a is None or off_a is None):
            perm_a, off_a = segment_attention_csr_metadata(segment_a, num_segment)
        if not identity_b and (perm_b is None or off_b is None):
            perm_b, off_b = segment_attention_csr_metadata(segment_b, num_segment)
        if identity_a:
            perm_a = segment_a
        if identity_b:
            perm_b = segment_b

        out_a = torch.empty((num_segment, dim), device=logits_a.device, dtype=logits_a.dtype)
        out_b = torch.empty_like(out_a)
        max_a = torch.empty_like(out_a)
        denom_a = torch.empty_like(out_a)
        max_b = torch.empty_like(out_a)
        denom_b = torch.empty_like(out_a)
        lse_a = torch.empty_like(out_a) if return_lse else None
        lse_b = torch.empty_like(out_a) if return_lse else None
        _paired_csr_segment_attention_fwd[(num_segment,)](
            logits_a,
            logits_b,
            value,
            perm_a,
            off_a,
            perm_b,
            off_b,
            out_a,
            out_b,
            max_a,
            denom_a,
            max_b,
            denom_b,
            lse_a,
            lse_b,
            skip_segment,
            D=dim,
            LOGA_S0=loga_s0,
            LOGA_S1=loga_s1,
            LOGB_S0=logb_s0,
            LOGB_S1=logb_s1,
            VAL_S0=val_s0,
            VAL_S1=val_s1,
            BLK=blk,
            IDENTITY_A=identity_a,
            IDENTITY_B=identity_b,
            HAS_SKIP=skip_segment >= 0,
            STORE_LSE=return_lse,
            num_warps=_num_warps(blk),
        )
        ctx.save_for_backward(
            logits_a,
            logits_b,
            value,
            segment_a,
            segment_b,
            out_a,
            out_b,
            max_a,
            denom_a,
            max_b,
            denom_b,
        )
        ctx.skip_segment = skip_segment
        ctx.return_lse = return_lse
        if rec is not None:
            ctx._opsel_site = rec.site()
            params = {
                "rows": n_rows, "width": dim, "segments": num_segment, "BLK": blk,
                "num_warps": _num_warps(blk), "IDENTITY_A": identity_a, "IDENTITY_B": identity_b,
                "LOGA_S": [loga_s0, loga_s1], "LOGB_S": [logb_s0, logb_s1],
                "VAL_S": [val_s0, val_s1], "csr": csr, "dtype": logits_a.dtype,
                "index_dtype": index_dtype, "offsets_dtype": [off_a.dtype, off_b.dtype],
                "contiguous_copy": copied,
            }
            if skip_segment != -1:  # new key only when set: -1 snapshots are unchanged
                params["SKIP_SEGMENT"] = skip_segment
            if return_lse:
                params["LSE"] = True
            rec.record("paired_segment_attention", "triton_csr", params)
        if return_lse:
            return out_a, out_b, lse_a, lse_b
        return out_a, out_b

    @staticmethod
    def backward(ctx, gout_a, gout_b, glse_a=None, glse_b=None):
        (
            logits_a,
            logits_b,
            value,
            segment_a,
            segment_b,
            out_a,
            out_b,
            max_a,
            denom_a,
            max_b,
            denom_b,
        ) = ctx.saved_tensors
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = [not gout_a.is_contiguous(), not gout_b.is_contiguous()]
        gout_a = gout_a.contiguous()
        gout_b = gout_b.contiguous()
        has_glse = glse_a is not None
        if has_glse:
            glse_a = glse_a.contiguous()
            glse_b = glse_b.contiguous()
        n_rows, dim = logits_a.shape
        loga_s0, loga_s1 = logits_a.stride()
        logb_s0, logb_s1 = logits_b.stride()
        val_s0, val_s1 = value.stride()
        bd = min(triton.next_power_of_2(dim), _PAIRED_BWD_BD)
        claims = getattr(ctx, "_grad_claims", (None, None))
        grad_logits_a = claim_grad(claims[0], (n_rows, dim), logits_a.device, logits_a.dtype)
        grad_logits_b = claim_grad(claims[1], (n_rows, dim), logits_b.device, logits_b.dtype)
        grad_value = torch.empty((n_rows, dim), device=value.device, dtype=value.dtype)
        _paired_segment_attention_bwd_rows[
            (triton.cdiv(n_rows, _PAIRED_BWD_BM), triton.cdiv(dim, bd))
        ](
            logits_a,
            logits_b,
            value,
            gout_a,
            gout_b,
            out_a,
            out_b,
            max_a,
            denom_a,
            max_b,
            denom_b,
            segment_a,
            segment_b,
            grad_logits_a,
            grad_logits_b,
            grad_value,
            glse_a,
            glse_b,
            static_under_compile(n_rows),
            ctx.skip_segment,
            D=dim,
            LOGA_S0=loga_s0,
            LOGA_S1=loga_s1,
            LOGB_S0=logb_s0,
            LOGB_S1=logb_s1,
            VAL_S0=val_s0,
            VAL_S1=val_s1,
            GLA_S0=grad_logits_a.stride(0),
            GLB_S0=grad_logits_b.stride(0),
            BM=_PAIRED_BWD_BM,
            BD=bd,
            HAS_SKIP=ctx.skip_segment >= 0,
            HAS_GLSE=has_glse,
            num_warps=4,
        )
        if rec is not None:
            params = {
                "rows": n_rows, "width": dim, "BM": _PAIRED_BWD_BM, "BD": bd, "num_warps": 4,
                "LOGA_S": [loga_s0, loga_s1], "LOGB_S": [logb_s0, logb_s1],
                "VAL_S": [val_s0, val_s1], "contiguous_copy": copied,
            }
            if (grad_logits_a.stride(0), grad_logits_b.stride(0)) != (dim, dim):
                params["GL_S0"] = [grad_logits_a.stride(0), grad_logits_b.stride(0)]
            if ctx.skip_segment != -1:
                params["SKIP_SEGMENT"] = ctx.skip_segment
            if ctx.return_lse:
                params["LSE"] = True
            rec.record("paired_segment_attention", "triton_csr", params,
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return (
            grad_logits_a,
            grad_logits_b,
            grad_value,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def segment_attention_csr_metadata(
    segment: torch.Tensor,
    num_segment: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    segment = segment.to(torch.int64)
    num_segment = int(num_segment)
    perm = torch.argsort(segment)
    sorted_segment = segment[perm]
    # int64 offsets, like every CSR these kernels consume (no int32 downcast).
    off = torch.searchsorted(
        sorted_segment,
        torch.arange(num_segment + 1, device=segment.device, dtype=segment.dtype),
    )
    return perm, off


def _eager_segment_attention(
    logits: torch.Tensor,
    value: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int,
) -> torch.Tensor:
    dim = logits.shape[1]
    segment_expanded = segment.unsqueeze(1).expand(-1, dim)
    max_per_segment = logits.new_full((num_segment, dim), float("-inf"))
    max_per_segment = max_per_segment.scatter_reduce(
        0,
        segment_expanded,
        logits,
        reduce="amax",
        include_self=False,
    )
    centered = logits - max_per_segment[segment]
    weight = centered.exp()
    denom = logits.new_zeros((num_segment, dim)).scatter_reduce(
        0,
        segment_expanded,
        weight,
        reduce="sum",
        include_self=False,
    )
    alpha = weight / denom[segment]
    out = logits.new_zeros((num_segment, dim))
    return out.index_add_(0, segment, alpha * value)


def segment_attention(
    logits: torch.Tensor,
    value: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int,
    perm: torch.Tensor | None = None,
    off: torch.Tensor | None = None,
    identity_perm: bool = False,
    *,
    skip_segment: int = -1,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """``return_lse=True`` returns ``(out, lse)``, ``lse`` [num_segment, D]
    the per-segment log-sum-exp of the logits: -inf for an empty segment and
    for ``skip_segment`` (whose ``out`` is 0)."""
    if logits.is_cuda and value.is_cuda:
        return _SegmentAttention.apply(
            logits,
            value,
            segment,
            int(num_segment),
            perm,
            off,
            bool(identity_perm),
            int(skip_segment),
            bool(return_lse),
        )
    rec = _recorder._ACTIVE
    if rec is not None:
        params = {
            "rows": logits.shape[0], "width": logits.shape[1], "segments": int(num_segment),
            "SKIP_SEGMENT": int(skip_segment),
        }
        if return_lse:
            params["LSE"] = True
        rec.record("segment_attention", "torch", params)
    result = _eager_segment_attention(logits, value, segment, int(num_segment))
    lse = segment_logsumexp(logits, segment, int(num_segment)) if return_lse else None
    if skip_segment != -1:
        if skip_segment < 0 or skip_segment >= int(num_segment):
            raise ValueError("skip_segment must be -1 or an existing private segment")
        mask = torch.arange(int(num_segment), device=result.device) != skip_segment
        result = torch.where(mask[:, None], result, 0)
        if return_lse:
            lse = torch.where(mask[:, None], lse, float("-inf"))
    return (result, lse) if return_lse else result


def paired_segment_attention(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    value: torch.Tensor,
    segment_a: torch.Tensor,
    segment_b: torch.Tensor,
    num_segment: int,
    perm_a: torch.Tensor | None = None,
    off_a: torch.Tensor | None = None,
    perm_b: torch.Tensor | None = None,
    off_b: torch.Tensor | None = None,
    identity_a: bool = False,
    identity_b: bool = False,
    *,
    skip_segment: int = -1,
    return_lse: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Two segment attentions sharing ``value``; ``skip_segment`` as in
    :func:`segment_attention` (both outputs 0 for it, its rows get no
    gradient from either side on which they belong to it).

    ``return_lse=True`` returns ``(out_a, out_b, lse_a, lse_b)``, the lse as
    in :func:`segment_attention`."""
    if logits_a.is_cuda and logits_b.is_cuda and value.is_cuda:
        return _PairedSegmentAttention.apply(
            logits_a,
            logits_b,
            value,
            segment_a,
            segment_b,
            int(num_segment),
            perm_a,
            off_a,
            perm_b,
            off_b,
            bool(identity_a),
            bool(identity_b),
            int(skip_segment),
            bool(return_lse),
        )
    rec = _recorder._ACTIVE
    if rec is not None:
        params = {
            "rows": logits_a.shape[0], "width": logits_a.shape[1], "segments": int(num_segment),
        }
        if skip_segment != -1:
            params["SKIP_SEGMENT"] = int(skip_segment)
        if return_lse:
            params["LSE"] = True
        rec.record("paired_segment_attention", "torch", params)
    result_a = _eager_segment_attention(logits_a, value, segment_a, int(num_segment))
    result_b = _eager_segment_attention(logits_b, value, segment_b, int(num_segment))
    lse = ([segment_logsumexp(logits, segment, int(num_segment))
            for logits, segment in ((logits_a, segment_a), (logits_b, segment_b))]
           if return_lse else [])
    if skip_segment != -1:
        if skip_segment < 0 or skip_segment >= int(num_segment):
            raise ValueError("skip_segment must be -1 or an existing private segment")
        mask = (torch.arange(int(num_segment), device=result_a.device) != skip_segment)[:, None]
        result_a = torch.where(mask, result_a, 0)
        result_b = torch.where(mask, result_b, 0)
        lse = [torch.where(mask, part, float("-inf")) for part in lse]
    return (result_a, result_b, *lse)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for n_rows, dim, num_segment in [(26174, 128, 1165), (152446, 128, 13087), (257, 32, 31)]:
        segment = torch.randint(0, num_segment, (n_rows,), device=dev)
        logits = (torch.randn(n_rows, dim, device=dev) * 3).requires_grad_(True)
        value = torch.randn(n_rows, dim, device=dev, requires_grad=True)
        logits_tri = logits.detach().clone().requires_grad_(True)
        value_tri = value.detach().clone().requires_grad_(True)
        ref = _eager_segment_attention(logits, value, segment, num_segment)
        tri = segment_attention(logits_tri, value_tri, segment, num_segment)
        fwd_err = (ref - tri).abs().max().item()
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        bwd_err = max(
            (logits.grad - logits_tri.grad).abs().max().item(),
            (value.grad - value_tri.grad).abs().max().item(),
        )
        print(
            f"N={n_rows} D={dim} S={num_segment}: "
            f"fwd max|d|={fwd_err:.2e}  bwd max|d|={bwd_err:.2e}"
        )
