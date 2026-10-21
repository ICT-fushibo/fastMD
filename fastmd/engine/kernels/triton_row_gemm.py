"""Row GEMMs for the narrow (K <= 32) frozen projections (C4, residual round 2).

``y = x @ W^T`` with ``x [rows, K = 32]`` (unit column stride, any row stride) and a frozen ``W [N, K]``: each program
holds a [BM, K] row tile and a [K, BN] block of ``W`` and runs one dot per tile. At K = 32 cuBLAS picks
generic 64x64 or 128x64 SIMT tiles that stay at 2.3-2.6 TB/s on these row-major streams; the row GEMM reads
``x`` once per column block and writes ``y`` once. Two layouts, chosen per (K, N): column blocks in the grid
(N >= 128), or persistent programs that stride over the row tiles (N <= 64, where the per-tile work is
small). ``row_addmm`` (``c + g @ w``, e.g. a projection's input gradient plus its alias gradient) runs the
persistent layout with ``c`` read in the kernel: no copy of ``c`` into the output first.

The forward dots' input precision is ``InferenceConfig.triton_dot_precision``
(``fastmd.precision.triton_dot_input_precision("row_gemm")``): by default 3xTF32 where cuBLAS runs IEEE FP32
(FP32-level accuracy; the results differ from cuBLAS within rounding), and where cuBLAS runs TF32 the
projections stay cuBLAS (faster there than a TF32 row GEMM). ``row_addmm`` accumulates in plain FP32 FMA
(``input_precision="ieee"``) in every mode (faster than cuBLAS's addmm). Every element's reduction over K is the same instruction
sequence whatever the tile layout, so a narrower column range of the same product (a recompute recipe) is
bitwise the columns of the wide one (measured on GB200 for the model's widths, leading and trailing column
ranges; the recipes also take the forward's backend, see ``merged_frozen_projections.project``). Each row's
result depends on its own inputs only (not on the row count, its tile or the grid), so bond rows replicated on
two ranks get the same values. The rows are a runtime argument excluded from Triton's value specialization
(K5); K, N and the tiles are fixed per call site.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from fastmd.precision import triton_dot_input_precision


@triton.jit(do_not_specialize=["M"])
def _row_gemm_tiles(x, w, y, M, SX, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr, NB: tl.constexpr,
                    BM: tl.constexpr, PRECISION: tl.constexpr):
    # One [BM, BN] output tile per program: column block pid % NB of row tile pid // NB.
    # The block indices in int32 (int64 division here: +30% at N = 128), widened at the row offsets.
    pid = tl.program_id(0)  # int32 block index
    rows = (pid // NB).to(tl.int64) * BM + tl.arange(0, BM)
    cols = (pid % NB) * BN + tl.arange(0, BN)
    k = tl.arange(0, K)
    live = rows < M
    live_n = cols < N
    xt = tl.load(x + rows[:, None] * SX + k[None, :], mask=live[:, None], other=0.0)
    wt = tl.load(w + cols[None, :] * K + k[:, None], mask=live_n[None, :], other=0.0)
    acc = tl.dot(xt, wt, input_precision=PRECISION)
    tl.store(y + rows[:, None] * N + cols[None, :], acc, mask=live[:, None] & live_n[None, :])


@triton.jit(do_not_specialize=["M", "TILES"])
def _row_gemm_persistent(x, w, c, y, M, SX, SC, SWN, SWK, TILES, K: tl.constexpr, N: tl.constexpr,
                         KC: tl.constexpr, BM: tl.constexpr, HAS_C: tl.constexpr, STAGES: tl.constexpr,
                         PRECISION: tl.constexpr):
    # Programs stride over the row tiles; K is reduced in KC-wide chunks. ``w`` element (n, k) is at
    # w + n * SWN + k * SWK (forward: W [N, K]; addmm: w [K, N]); ``c`` (HAS_C) is added from its own row stride.
    pid = tl.program_id(0)  # int32 block index
    programs = tl.num_programs(0)
    n = tl.arange(0, N)
    kc = tl.arange(0, KC)
    # An int32 tile loop (an int64 induction variable: +8..13%), widened at the row offsets.
    for tile in tl.range(pid, TILES, programs, num_stages=STAGES):
        rows = tile.to(tl.int64) * BM + tl.arange(0, BM)
        live = rows < M
        acc = tl.zeros((BM, N), dtype=tl.float32)
        for k0 in tl.static_range(0, K, KC):
            xt = tl.load(x + rows[:, None] * SX + (k0 + kc)[None, :], mask=live[:, None], other=0.0)
            wt = tl.load(w + n[None, :] * SWN + (k0 + kc)[:, None] * SWK)
            acc = tl.dot(xt, wt, acc, input_precision=PRECISION)
        if HAS_C:
            acc += tl.load(c + rows[:, None] * SC + n[None, :], mask=live[:, None], other=0.0)
        tl.store(y + rows[:, None] * N + n[None, :], acc, mask=live[:, None])


# Forward tiles per output width N (K = 32; GB200, 3xTF32, 7.6M / 974k rows; vs cuBLAS IEEE FP32: N = 64
# 0.69 / 1.26 ms, 128 1.15 / 2.32, 256 0.30 / 0.59, 320 0.45 / 0.88): column blocks in the grid for wide
# outputs, persistent programs (two per SM) for N <= 64. (layout, BM, BN or KC, num_warps, stages)
_TILES = {128: ("tiles", 128, 64, 4, 0), 256: ("tiles", 128, 64, 4, 0), 320: ("tiles", 128, 128, 4, 0),
          64: ("persistent", 128, 32, 4, 3), 32: ("persistent", 128, 32, 4, 3)}
# row_addmm: (BM, KC, num_warps, stages, programs per SM), g [rows, 64] @ w [64, 32] + c.
_ADDMM_TILE = (64, 32, 4, 3, 4)
_SMS: dict[int, int] = {}


def _programs(device: torch.device, per_sm: int, tiles: int) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    sms = _SMS.get(index)
    if sms is None:
        sms = _SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return max(1, min(tiles, per_sm * sms))


def precision() -> str | None:
    """The forward dots' input precision now, None: the projections take cuBLAS (see the module docstring)."""
    return triton_dot_input_precision("row_gemm")


def row_linear_supported(x: torch.Tensor, weight: torch.Tensor) -> str | None:
    """None if ``x @ weight^T`` runs here, else the reason it does not."""
    if not (x.is_cuda and weight.is_cuda and x.device == weight.device) or x.device.index != torch.cuda.current_device():
        return "device"
    if x.dtype != torch.float32 or weight.dtype != torch.float32 or torch.is_autocast_enabled("cuda"):
        return "dtype"
    if x.dim() != 2 or weight.dim() != 2 or x.shape[1] != weight.shape[1]:
        return "shape"
    if x.shape[1] != 32 or weight.shape[0] not in _TILES:
        return "unsupported_shape"
    if x.stride(1) != 1 or not weight.is_contiguous():
        return "layout"
    if weight.requires_grad:
        return "trainable_params"
    return None


def row_linear_forward(x: torch.Tensor, weight: torch.Tensor, prec: str | None = None) -> torch.Tensor:
    """``x @ weight^T`` as a new contiguous [rows, N] tensor (the caller checked ``row_linear_supported``);
    ``prec`` the dots' input precision, None: ``precision()``."""
    rows, k = x.shape
    n = weight.shape[0]
    y = torch.empty((rows, n), device=x.device, dtype=torch.float32)
    if rows == 0:
        return y
    prec = precision() if prec is None else prec
    if prec is None:
        raise ValueError("the row GEMM site takes cuBLAS at this dot precision (triton_dot_input_precision)")
    layout, bm, block, warps, stages = _TILES[n]
    if layout == "tiles":
        nb = triton.cdiv(n, block)
        _row_gemm_tiles[(triton.cdiv(rows, bm) * nb,)](x, weight, y, rows, x.stride(0), K=k, N=n, BN=block, NB=nb,
                                                       BM=bm, PRECISION=prec, num_warps=warps)
    else:
        tiles = triton.cdiv(rows, bm)
        _row_gemm_persistent[(_programs(x.device, 2, tiles),)](
            x, weight, y, y, rows, x.stride(0), 0, k, 1, tiles, K=k, N=n, KC=min(block, k), BM=bm, HAS_C=False,
            STAGES=stages, PRECISION=prec, num_warps=warps)
    return y


def row_addmm_supported(c: torch.Tensor, g: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether ``c + g @ w`` runs here: g [rows, 64], w [64, 32] contiguous, c [rows, 32], FP32, unit column strides."""
    return (g.is_cuda and c.is_cuda and w.is_cuda and g.device.index == torch.cuda.current_device()
            and g.dtype == c.dtype == w.dtype == torch.float32
            and g.dim() == 2 and c.dim() == 2 and tuple(w.shape) == (64, 32) and g.shape[1] == 64
            and tuple(c.shape) == (g.shape[0], 32) and g.stride(1) == 1 and c.stride(1) == 1
            and w.is_contiguous() and not w.requires_grad)


def row_addmm(c: torch.Tensor, g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``c + g @ w`` (FP32 FMA) as a new contiguous tensor; the caller checked ``row_addmm_supported``."""
    rows = g.shape[0]
    y = torch.empty((rows, 32), device=g.device, dtype=torch.float32)
    if rows == 0:
        return y
    bm, kc, warps, stages, per_sm = _ADDMM_TILE
    tiles = triton.cdiv(rows, bm)
    # w [K = 64, N = 32]: element (n, k) at w + n + k * 32.
    _row_gemm_persistent[(_programs(g.device, per_sm, tiles),)](
        g, w, c, y, rows, g.stride(0), c.stride(0), 1, 32, tiles, K=64, N=32, KC=kc, BM=bm, HAS_C=True, STAGES=stages,
        PRECISION="ieee", num_warps=warps)
    return y


class _RowLinear(torch.autograd.Function):
    """``x @ weight^T`` with the row GEMM forward and a cuBLAS ``dX = dY @ W`` backward (frozen weight)."""

    @staticmethod
    def forward(ctx, x, weight, prec):
        ctx.save_for_backward(weight)
        return row_linear_forward(x, weight, prec)

    @staticmethod
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return torch.mm(grad, weight), None, None


def row_linear(x: torch.Tensor, weight: torch.Tensor, prec: str | None = None) -> torch.Tensor:
    """``x @ weight^T``, differentiable in ``x`` (the caller checked ``row_linear_supported``)."""
    if torch.is_grad_enabled() and x.requires_grad:
        return _RowLinear.apply(x, weight, prec)
    return row_linear_forward(x, weight, prec)
