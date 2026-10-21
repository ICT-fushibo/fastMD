"""Triton specialization for directed-edge -> undirected-edge averaging.

The atom graph stores exactly two directed edges for each undirected edge.  The
generic path uses ``index_add_(directed2undirected) / bincount``.  This operator
uses a precomputed ``pair_index[U, 2]`` and performs:

    out[u, d] = 0.5 * (data[pair_index[u, 0], d] + data[pair_index[u, 1], d])

and the reverse expansion ``out[pair_index[u, k], d] = data[u, d]``.

Contract (checked only where it is cheap host metadata):

* ``pair_index`` must be a bijection between the ``U`` pairs and the
  ``E == 2U`` directed rows (including padding rows): every directed row
  appears exactly once.  The kernels ``tl.store`` into ``torch.empty``
  (average backward, expand forward), so an uncovered row would stay
  uninitialized and a duplicated row would race.  The wrappers reject
  ``E != 2U`` (``ValueError``); the value-level check that needs a device
  read is ``pair_index_bijection_reason`` (admission, debug and tests only,
  never on the hot path).
* Inputs need unit column stride; any row stride is addressed directly
  (``S0``, e.g. ``packed[:, :D]`` of a ``[R, 3D]`` buffer), so only
  ``stride(1) != 1`` is copied.  Outputs are fresh contiguous tensors.
* The pair count ``U`` is a runtime argument excluded from Triton's value
  specialization, so eager steps with new edge counts reuse the compiled
  kernels.  Compile-time keys are the width ``D``, the input row stride
  ``S0`` and the tiles.
"""
from __future__ import annotations

from fastmd.config import env_value


import torch
import triton
import triton.language as tl
from torch.autograd import Function

from fastmd.engine import recorder as _recorder
from fastmd.engine.kernels.row_count import static_under_compile
from fastmd.engine.kernels.row_tiles import row_tile


def _env_int(name: str, default: int | None) -> int | None:
    value = env_value(name)
    if value is None:
        return default
    return int(value)


# None: by width (row_tiles).
_PAIR_AGG_BM = _env_int("MATRIS_DIRECTED_PAIR_AGG_BM", None)
_PAIR_AGG_BD = _env_int("MATRIS_DIRECTED_PAIR_AGG_BD", 128)

# _RUNTIME_COUNT_NOTE: the pair count U is a runtime int (K5).  Every kernel
# states ``tl.assume(U >= 0)`` so the row mask stays the unsigned 64-bit
# compare the former constexpr U compiled to; without it the sign-extended
# compare delays the first index loads (+1.5% at 60889 pairs).

@triton.jit(do_not_specialize=["U"])
def _directed_pair_average_fwd(
    data,
    pair_index,
    out,
    U,
    D: tl.constexpr,
    S0: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    tl.assume(U >= 0)  # see _RUNTIME_COUNT_NOTE
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < U) & (cols[None, :] < D)
    idx0 = tl.load(pair_index + rows * 2, mask=rows < U, other=0).to(tl.int64)
    idx1 = tl.load(pair_index + rows * 2 + 1, mask=rows < U, other=0).to(tl.int64)
    v0 = tl.load(data + idx0[:, None] * S0 + cols[None, :], mask=mask, other=0.0)
    v1 = tl.load(data + idx1[:, None] * S0 + cols[None, :], mask=mask, other=0.0)
    tl.store(out + rows[:, None] * D + cols[None, :], (v0 + v1) * 0.5, mask=mask)


@triton.jit(do_not_specialize=["U"])
def _directed_pair_average_bwd(
    gout,
    pair_index,
    grad_data,
    U,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    tl.assume(U >= 0)  # see _RUNTIME_COUNT_NOTE
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < U) & (cols[None, :] < D)
    idx0 = tl.load(pair_index + rows * 2, mask=rows < U, other=0).to(tl.int64)
    idx1 = tl.load(pair_index + rows * 2 + 1, mask=rows < U, other=0).to(tl.int64)
    grad = tl.load(gout + rows[:, None] * D + cols[None, :], mask=mask, other=0.0)
    grad *= 0.5
    tl.store(grad_data + idx0[:, None] * D + cols[None, :], grad, mask=mask)
    tl.store(grad_data + idx1[:, None] * D + cols[None, :], grad, mask=mask)


@triton.jit(do_not_specialize=["U"])
def _undirected_pair_expand_fwd(
    data,
    pair_index,
    out,
    U,
    D: tl.constexpr,
    S0: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    tl.assume(U >= 0)  # see _RUNTIME_COUNT_NOTE
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < U) & (cols[None, :] < D)
    idx0 = tl.load(pair_index + rows * 2, mask=rows < U, other=0).to(tl.int64)
    idx1 = tl.load(pair_index + rows * 2 + 1, mask=rows < U, other=0).to(tl.int64)
    vals = tl.load(data + rows[:, None] * S0 + cols[None, :], mask=mask, other=0.0)
    tl.store(out + idx0[:, None] * D + cols[None, :], vals, mask=mask)
    tl.store(out + idx1[:, None] * D + cols[None, :], vals, mask=mask)


@triton.jit(do_not_specialize=["U"])
def _undirected_pair_expand_bwd(
    gout,
    pair_index,
    grad_data,
    U,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    tl.assume(U >= 0)  # see _RUNTIME_COUNT_NOTE
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < U) & (cols[None, :] < D)
    idx0 = tl.load(pair_index + rows * 2, mask=rows < U, other=0).to(tl.int64)
    idx1 = tl.load(pair_index + rows * 2 + 1, mask=rows < U, other=0).to(tl.int64)
    g0 = tl.load(gout + idx0[:, None] * D + cols[None, :], mask=mask, other=0.0)
    g1 = tl.load(gout + idx1[:, None] * D + cols[None, :], mask=mask, other=0.0)
    tl.store(grad_data + rows[:, None] * D + cols[None, :], g0 + g1, mask=mask)


class _DirectedPairAverage(Function):
    @staticmethod
    def forward(ctx, data: torch.Tensor, pair_index: torch.Tensor):
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = data.stride(1) != 1
        data = data.contiguous() if data.stride(1) != 1 else data
        pair_index = pair_index.contiguous()
        u, d = pair_index.shape[0], data.shape[1]
        out = torch.empty((u, d), device=data.device, dtype=data.dtype)
        bd = triton.next_power_of_2(min(_PAIR_AGG_BD, d))
        bm = row_tile("directed_pair", bd, _PAIR_AGG_BM)
        # Positional launch: keyword binding costs ~0.2 us per argument.
        _directed_pair_average_fwd[
            (triton.cdiv(u, bm), triton.cdiv(d, _PAIR_AGG_BD))
        ](
            data, pair_index, out, static_under_compile(u),
            d, data.stride(0), bm, bd,  # constexprs
            num_warps=4,
        )
        ctx.save_for_backward(pair_index)
        ctx.data_shape = data.shape
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("pair_average", "triton", {
                "pairs": u, "width": d, "BM": bm, "BD": bd, "num_warps": 4,
                "dtype": data.dtype, "index_dtype": pair_index.dtype,
                "data_stride0": data.stride(0), "contiguous_copy": copied,
            })
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        (pair_index,) = ctx.saved_tensors
        e, d = ctx.data_shape
        u = pair_index.shape[0]
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = not gout.is_contiguous()
        gout = gout.contiguous()
        grad_data = torch.empty((e, d), device=gout.device, dtype=gout.dtype)
        bd = triton.next_power_of_2(min(_PAIR_AGG_BD, d))
        bm = row_tile("directed_pair", bd, _PAIR_AGG_BM)
        _directed_pair_average_bwd[
            (triton.cdiv(u, bm), triton.cdiv(d, _PAIR_AGG_BD))
        ](
            gout, pair_index, grad_data, static_under_compile(u),
            d, bm, bd,  # constexprs
            num_warps=4,
        )
        if rec is not None:
            rec.record("pair_average", "triton", {
                "pairs": u, "width": d, "BM": bm, "BD": bd, "num_warps": 4,
                "dtype": gout.dtype, "contiguous_copy": copied,
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return grad_data, None


class _UndirectedPairExpand(Function):
    @staticmethod
    def forward(ctx, data: torch.Tensor, pair_index: torch.Tensor, num_directed: int):
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = data.stride(1) != 1
        data = data.contiguous() if data.stride(1) != 1 else data
        pair_index = pair_index.contiguous()
        u, d = data.shape
        out = torch.empty((num_directed, d), device=data.device, dtype=data.dtype)
        bd = triton.next_power_of_2(min(_PAIR_AGG_BD, d))
        bm = row_tile("directed_pair", bd, _PAIR_AGG_BM)
        _undirected_pair_expand_fwd[
            (triton.cdiv(u, bm), triton.cdiv(d, _PAIR_AGG_BD))
        ](
            data, pair_index, out, static_under_compile(u),
            d, data.stride(0), bm, bd,  # constexprs
            num_warps=4,
        )
        ctx.save_for_backward(pair_index)
        ctx.data_shape = data.shape
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("pair_expand", "triton", {
                "pairs": u, "width": d, "rows": num_directed, "BM": bm, "BD": bd, "num_warps": 4,
                "dtype": data.dtype, "index_dtype": pair_index.dtype,
                "data_stride0": data.stride(0), "contiguous_copy": copied,
            })
        return out

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        (pair_index,) = ctx.saved_tensors
        u, d = ctx.data_shape
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = not gout.is_contiguous()
        gout = gout.contiguous()
        grad_data = torch.empty((u, d), device=gout.device, dtype=gout.dtype)
        bd = triton.next_power_of_2(min(_PAIR_AGG_BD, d))
        bm = row_tile("directed_pair", bd, _PAIR_AGG_BM)
        _undirected_pair_expand_bwd[
            (triton.cdiv(u, bm), triton.cdiv(d, _PAIR_AGG_BD))
        ](
            gout, pair_index, grad_data, static_under_compile(u),
            d, bm, bd,  # constexprs
            num_warps=4,
        )
        if rec is not None:
            rec.record("pair_expand", "triton", {
                "pairs": u, "width": d, "BM": bm, "BD": bd, "num_warps": 4,
                "dtype": gout.dtype, "contiguous_copy": copied,
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return grad_data, None, None


def pair_index_bijection_reason(pair_index, num_directed) -> str | None:
    """Why ``pair_index`` is not a pair bijection onto ``num_directed`` rows.

    Returns ``None`` when ``pair_index`` is an integer ``[U, 2]`` tensor with
    ``num_directed == 2U`` whose entries cover every directed row
    ``0 .. num_directed-1`` exactly once, else a short reason.  Host-side
    admission/debug check: it reads index values (a device sync for CUDA
    tensors), so it is never called on the hot path.
    """
    if not isinstance(pair_index, torch.Tensor):
        return "not_tensor"
    if pair_index.dim() != 2 or pair_index.shape[1] != 2:
        return f"pair_index_shape: expected [U, 2], got {list(pair_index.shape)}"
    if pair_index.dtype not in (torch.int32, torch.int64):
        return f"index_dtype: {pair_index.dtype}"
    num_directed = int(num_directed)
    num_pairs = pair_index.shape[0]
    if num_directed != 2 * num_pairs:
        return f"num_directed: {num_directed} != 2U = {2 * num_pairs}"
    if num_pairs == 0:
        return None
    flat = pair_index.reshape(-1).long()
    low, high = torch.stack((flat.min(), flat.max())).tolist()
    if low < 0 or high >= num_directed:
        return f"index_out_of_range: [{low}, {high}] not within [0, {num_directed})"
    # 2U in-range values cover all 2U rows iff none repeats.
    if bool((torch.bincount(flat, minlength=num_directed) != 1).any()):
        return "duplicate_directed_row"
    return None


def _raise_pair_shapes(data: torch.Tensor, pair_index: torch.Tensor, num_directed: int | None, what: str):
    if data.dim() != 2 or pair_index.dim() != 2 or pair_index.shape[1] != 2:
        raise ValueError(
            f"{what}: expected data [rows, D] and pair_index [U, 2], "
            f"got {tuple(data.shape)}, {tuple(pair_index.shape)}"
        )
    u = pair_index.shape[0]
    if num_directed is not None and num_directed != 2 * u:
        raise ValueError(
            f"{what}: num_directed must be 2U (a pair bijection); "
            f"got num_directed = {num_directed} for U = {u}"
        )
    rows_per_pair = 1 if num_directed is not None else 2
    raise ValueError(
        f"{what}: expected {rows_per_pair} data row(s) per pair (E == 2U directed rows), "
        f"got {data.shape[0]} rows for U = {u}"
    )


def directed_pair_average(
    data: torch.Tensor,
    pair_index: torch.Tensor,
) -> torch.Tensor:
    """``(data[pair_index[:, 0]] + data[pair_index[:, 1]]) / 2`` for data ``[2U, D]``."""
    # Host metadata only (shapes), never index values; one fused test.
    ds, ps = data.shape, pair_index.shape
    if len(ds) != 2 or len(ps) != 2 or ps[1] != 2 or ds[0] != 2 * ps[0]:
        _raise_pair_shapes(data, pair_index, None, "directed_pair_average")
    if data.is_cuda and pair_index.is_cuda:
        return _DirectedPairAverage.apply(data, pair_index)
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("pair_average", "torch", {"pairs": pair_index.shape[0], "width": data.shape[1]})
    return (data[pair_index[:, 0]] + data[pair_index[:, 1]]) * 0.5


def undirected_pair_expand(
    data: torch.Tensor,
    pair_index: torch.Tensor,
    num_directed: int,
) -> torch.Tensor:
    """``out[pair_index[u, k]] = data[u]`` for data ``[U, D]``; ``num_directed`` must be 2U."""
    num_directed = int(num_directed)
    # Host metadata only (shapes), never index values; one fused test.
    ds, ps = data.shape, pair_index.shape
    if len(ds) != 2 or len(ps) != 2 or ps[1] != 2 or ds[0] != ps[0] or num_directed != 2 * ps[0]:
        _raise_pair_shapes(data, pair_index, num_directed, "undirected_pair_expand")
    if data.is_cuda and pair_index.is_cuda:
        return _UndirectedPairExpand.apply(data, pair_index, num_directed)
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("pair_expand", "torch", {
            "pairs": pair_index.shape[0], "width": data.shape[1], "rows": num_directed,
        })
    out = data.new_empty((num_directed, data.shape[1]))
    out[pair_index[:, 0]] = data
    out[pair_index[:, 1]] = data
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for e, d in [(1024, 128), (26174, 128), (27648, 128)]:
        u = e // 2
        seg = torch.arange(u, device=dev, dtype=torch.long).repeat_interleave(2)
        perm = torch.randperm(e, device=dev)
        seg = seg[perm]
        pair = torch.argsort(seg).reshape(u, 2).contiguous()
        x = torch.randn(e, d, device=dev, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        ref = x_ref.new_zeros((u, d)).index_add_(0, seg, x_ref) * 0.5
        tri = directed_pair_average(x, pair)
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        torch.cuda.synchronize()
        print(
            f"E={e} D={d}: fwd={float((ref - tri).abs().max().detach()):.2e} "
            f"bwd={float((x_ref.grad - x.grad).abs().max().detach()):.2e}"
        )

        x = torch.randn(u, d, device=dev, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        ref = torch.index_select(x_ref, 0, seg)
        tri = undirected_pair_expand(x, pair, e)
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        torch.cuda.synchronize()
        print(
            f"E={e} D={d} expand: fwd={float((ref - tri).abs().max().detach()):.2e} "
            f"bwd={float((x_ref.grad - x.grad).abs().max().detach()):.2e}"
        )
