"""Fused line-graph envelope gather product.

Computes ``out[row, d] = base[source[row], d] * base[target[row], d]`` and
fuses the two index_select calls plus the multiply in the line-graph refinement
envelope path.

``base`` needs unit column stride; any row stride is addressed directly
(``S0``, e.g. ``packed[:, :D]`` of a ``[R, 3D]`` buffer), so only
``stride(1) != 1`` is copied.  The row count ``N`` is a runtime argument
excluded from Triton's value specialization, so eager steps with new triplet
counts reuse the compiled kernels.  Compile-time keys are the width ``D``, the
row strides and the tiles.  The backward accumulates with atomics (order
nondeterministic); under ``private_sink_atomics`` (sink_atomics.py) they skip
rows whose addends are all zero (``SKIP_ZERO``), outside it they are unchanged.
In deterministic mode (read in the forward) the backward runs in torch with
ordered sums (``ordered_index_add_``).
"""
from __future__ import annotations

from fastmd.config import env_value


import torch
import triton
import triton.language as tl
from torch.autograd import Function

from fastmd.engine import recorder as _recorder
from fastmd.engine import saved_recompute as _saved_recompute
from fastmd.precision import deterministic_mode
from fastmd.engine.kernels.row_count import static_under_compile
from fastmd.engine.kernels.row_tiles import row_tile
from fastmd.engine.kernels.sink_atomics import zero_skip_atomics
from fastmd.engine.kernels.triton_zero_skip_index import ordered_index_add_


def _env_int(name: str, default: int | None) -> int | None:
    value = env_value(name)
    if value is None:
        return default
    return int(value)


# None: by width (row_tiles).
_BM = _env_int("MATRIS_LINE_ENVELOPE_BM", None)
_BD = _env_int("MATRIS_LINE_ENVELOPE_BD", 128)


@triton.jit(do_not_specialize=["N"])
def _line_envelope_fwd(
    base,
    source,
    target,
    out,
    N,
    D: tl.constexpr,
    S0: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    # N is a runtime count (K5); a known non-negative N keeps the row mask the
    # unsigned 64-bit compare the former constexpr N compiled to.
    tl.assume(N >= 0)
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    src = tl.load(source + rows, mask=rows < N, other=0).to(tl.int64)
    tgt = tl.load(target + rows, mask=rows < N, other=0).to(tl.int64)
    a = tl.load(base + src[:, None] * S0 + cols[None, :], mask=mask, other=0.0)
    b = tl.load(base + tgt[:, None] * S0 + cols[None, :], mask=mask, other=0.0)
    tl.store(out + rows[:, None] * D + cols[None, :], a * b, mask=mask)


@triton.jit
def _atomic_rows(mask, value, SKIP_ZERO: tl.constexpr):
    # The atomic mask of ``value``. With SKIP_ZERO (design D3) rows whose
    # addends are all zero, e.g. padding triplets of a private sink, are not
    # added. The mask stays uniform along a row, so the atomics keep their
    # vector width; a zero inside a live row is still added (as without
    # SKIP_ZERO), and NaN/Inf compare != 0, so the sums are unchanged.
    if SKIP_ZERO:
        mask = mask & (tl.max((value != 0.0).to(tl.int32), axis=1) != 0)[:, None]
    return mask


@triton.jit(do_not_specialize=["N"])
def _line_envelope_bwd(
    gout,
    base,
    source,
    target,
    grad_base,
    N,
    D: tl.constexpr,
    S0: tl.constexpr,
    GS0: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    SKIP_ZERO: tl.constexpr = False,
):
    # N is a runtime count (K5); a known non-negative N keeps the row mask the
    # unsigned 64-bit compare the former constexpr N compiled to.
    tl.assume(N >= 0)
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    src = tl.load(source + rows, mask=rows < N, other=0).to(tl.int64)
    tgt = tl.load(target + rows, mask=rows < N, other=0).to(tl.int64)
    offs = rows[:, None] * D + cols[None, :]
    go = tl.load(gout + offs, mask=mask, other=0.0)
    a = tl.load(base + src[:, None] * S0 + cols[None, :], mask=mask, other=0.0)
    b = tl.load(base + tgt[:, None] * S0 + cols[None, :], mask=mask, other=0.0)
    src_grad = grad_base + src[:, None] * GS0 + cols[None, :]
    add_src = go * b
    tl.atomic_add(src_grad, add_src, mask=_atomic_rows(mask, add_src, SKIP_ZERO), sem="relaxed")
    tgt_grad = grad_base + tgt[:, None] * GS0 + cols[None, :]
    add_tgt = go * a
    tl.atomic_add(tgt_grad, add_tgt, mask=_atomic_rows(mask, add_tgt, SKIP_ZERO), sem="relaxed")


class _LineEnvelopeProduct(Function):
    @staticmethod
    def forward(ctx, base: torch.Tensor, source: torch.Tensor, target: torch.Tensor):
        if base.dim() != 2 or source.numel() != target.numel():
            raise ValueError(
                f"expected base [S,D] and equal-length source/target, got {tuple(base.shape)}, "
                f"{source.numel()} source and {target.numel()} target rows"
            )
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = [base.stride(1) != 1, not source.is_contiguous(), not target.is_contiguous()]
        base = base.contiguous() if base.stride(1) != 1 else base
        source_index, target_index = source, target
        source = source.contiguous()
        target = target.contiguous()
        _saved_recompute.register_copy(source, source_index)
        _saved_recompute.register_copy(target, target_index)
        n = source.numel()
        d = base.shape[1]
        out = torch.empty((n, d), device=base.device, dtype=base.dtype)
        bd = triton.next_power_of_2(min(_BD, d))
        bm = row_tile("line_envelope", bd, _BM)
        grid = (triton.cdiv(n, bm), triton.cdiv(d, _BD))
        # Positional launch: keyword binding costs ~0.2 us per argument.
        _line_envelope_fwd[grid](
            base, source, target, out, static_under_compile(n),
            d, base.stride(0), bm, bd,  # constexprs
            num_warps=4,
        )
        ctx.save_for_backward(base, source, target)
        ctx.skip_zero = zero_skip_atomics()
        ctx.ordered = deterministic_mode()
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("line_envelope", "triton", {
                "rows": n, "width": d, "BM": bm, "BD": bd, "grid_BD": _BD, "num_warps": 4,
                "dtype": base.dtype, "index_dtype": [source.dtype, target.dtype],
                "base_stride0": base.stride(0), "contiguous_copy": copied,
            })
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        base, source, target = ctx.saved_tensors
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = not gout.is_contiguous()
        gout = gout.contiguous()
        n, d = gout.shape
        # zeros_like keeps a dense base's strides and is contiguous for a
        # row-strided view, so grad_base gets its own row stride (GS0).
        grad_base = torch.zeros_like(base)
        if ctx.ordered:
            ordered_index_add_(grad_base, source, gout * base.index_select(0, target), skip_zero=ctx.skip_zero)
            ordered_index_add_(grad_base, target, gout * base.index_select(0, source), skip_zero=ctx.skip_zero)
            if rec is not None:
                rec.record("line_envelope", "ordered_index_add", {"rows": n, "width": d},
                           phase="bwd", site=getattr(ctx, "_opsel_site", None))
            return grad_base, None, None
        bd = triton.next_power_of_2(min(_BD, d))
        bm = row_tile("line_envelope", bd, _BM)
        grid = (triton.cdiv(n, bm), triton.cdiv(d, _BD))
        _line_envelope_bwd[grid](
            gout, base, source, target, grad_base, static_under_compile(n),
            d, base.stride(0), grad_base.stride(0), bm, bd, ctx.skip_zero,  # constexprs
            num_warps=4,
        )
        if rec is not None:
            params = {
                "rows": n, "width": d, "BM": bm, "BD": bd, "grid_BD": _BD, "num_warps": 4,
                "dtype": gout.dtype, "base_stride0": base.stride(0), "contiguous_copy": copied,
            }
            if ctx.skip_zero:  # new key only under the private sink: P=1 records as before
                params["SKIP_ZERO"] = True
            rec.record("line_envelope", "triton", params,
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return grad_base, None, None


def line_envelope_product(
    base: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if base.is_cuda and source.is_cuda and target.is_cuda:
        return _LineEnvelopeProduct.apply(base, source, target)
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("line_envelope", "torch", {"rows": source.numel(), "width": base.shape[1]})
    return torch.index_select(base, 0, source) * torch.index_select(base, 0, target)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for n, s, d in [(1024, 113, 128), (152446, 13087, 128)]:
        source = torch.randint(0, s, (n,), device=dev)
        target = torch.randint(0, s, (n,), device=dev)
        base = torch.randn(s, d, device=dev, requires_grad=True)
        base_tri = base.detach().clone().requires_grad_(True)
        ref = torch.index_select(base, 0, source) * torch.index_select(base, 0, target)
        tri = line_envelope_product(base_tri, source, target)
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        print(
            f"N={n} S={s} D={d}: "
            f"fwd={(ref - tri).abs().max().item():.2e} "
            f"bwd={(base.grad - base_tri.grad).abs().max().item():.2e}"
        )
