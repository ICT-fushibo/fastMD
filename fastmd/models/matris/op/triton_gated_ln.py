"""Fused GatedMLP epilogue for frozen-parameter inference.

Computes:
    silu(layer_norm(core)) * sigmoid(layer_norm(gate))

The backward returns gradients for the core/gate inputs only. The public wrapper
falls back when LayerNorm parameters require gradients, preserving training
semantics.

D13a (step 2.5b): with ``core_lb``/``gate_lb`` the kernels also add the biases
of the linears that produced ``core_x``/``gate_x`` (called without them): the
saved inputs are the bias-free GEMM outputs, the backward adds the biases again
and returns their gradients (column sums) when they require them. The FP32
sum is the one the GEMM's bias epilogue rounds.

Two kernel variants, chosen by ``_TILE_2D`` when the forward launches (the
backward follows the forward's choice): [BM rows, BLK columns] tiles (D7,
the default since step 2.5a), or one row per single-warp program
(``MATRIS_GATED_LN_2D=0``, kept for one version). The 2-D tiles apply the same
FP32 formulas and save the same per-row mean/rstd, but reduce each row in a
different order, so results differ from the 1-row kernels in the last bits;
both are deterministic (no atomics). The 2-D row count is a runtime argument
excluded from Triton's value specialization. Under torch.compile it stays a
symbolic size like the 1-row grid: ``static_under_compile`` would guard every
row count and add _finish_gate recompiles that the 1-row kernels do not have.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function

from fastmd.config import env_value
from fastmd.engine import recorder as _recorder

# Step 2.5a (D7): 2-D row tiles, on by default (user decision); off restores the 1-row kernels.
_TILE_2D = env_value("MATRIS_GATED_LN_2D", "1") == "1"
# 2-D tiles (BM rows, num_warps) of the forward and backward kernels by BLK,
# measured on GB300 at widths 32 and 128 (bench_d7.py): the forward runs best
# with about 512 tile elements per warp, the backward with about 128. Other
# widths take BM = 1024 // BLK rows and warps by those ratios.
_TILE_2D_FWD = {32: (16, 1), 128: (8, 2)}
_TILE_2D_BWD = {32: (16, 4), 128: (8, 8)}


def _tile_2d(block: int, backward: bool) -> tuple[int, int]:
    """(BM, num_warps) of a 2-D kernel for a power-of-two row width."""
    tile = (_TILE_2D_BWD if backward else _TILE_2D_FWD).get(block)
    if tile is None:
        bm = max(1, 1024 // block)
        tile = bm, max(1, min(8, bm * block // (128 if backward else 512)))
    return tile


@triton.jit
def _gated_ln_fwd(
    core_x,
    gate_x,
    core_lb,
    gate_lb,
    core_w,
    core_b,
    gate_w,
    gate_b,
    residual,
    res_weight,
    out,
    core_mean,
    core_rstd,
    gate_mean,
    gate_rstd,
    D: tl.constexpr,
    BLK: tl.constexpr,
    EPS_CORE: tl.constexpr,
    EPS_GATE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLK)
    mask = c < D

    core_val = tl.load(core_x + row * D + c, mask=mask, other=0.0).to(tl.float32)
    gate_val = tl.load(gate_x + row * D + c, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        # D13a: the second linears' bias, added here instead of by their GEMM epilogue.
        core_val += tl.load(core_lb + c, mask=mask, other=0.0).to(tl.float32)
        gate_val += tl.load(gate_lb + c, mask=mask, other=0.0).to(tl.float32)

    c_mu = tl.sum(core_val, axis=0) / D
    c_centered = tl.where(mask, core_val - c_mu, 0.0)
    c_rs = 1.0 / tl.sqrt(tl.sum(c_centered * c_centered, axis=0) / D + EPS_CORE)
    c_y = (
        c_centered
        * c_rs
        * tl.load(core_w + c, mask=mask, other=0.0)
        + tl.load(core_b + c, mask=mask, other=0.0)
    )
    c_sig = tl.sigmoid(c_y)
    core = c_y * c_sig

    g_mu = tl.sum(gate_val, axis=0) / D
    g_centered = tl.where(mask, gate_val - g_mu, 0.0)
    g_rs = 1.0 / tl.sqrt(tl.sum(g_centered * g_centered, axis=0) / D + EPS_GATE)
    g_y = (
        g_centered
        * g_rs
        * tl.load(gate_w + c, mask=mask, other=0.0)
        + tl.load(gate_b + c, mask=mask, other=0.0)
    )
    gate = tl.sigmoid(g_y)

    result = core * gate
    if HAS_RESIDUAL:
        r = tl.load(residual + row * D + c, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(res_weight + c, mask=mask, other=0.0).to(tl.float32)
        result += r * w
    tl.store(out + row * D + c, result, mask=mask)
    tl.store(core_mean + row, c_mu)
    tl.store(core_rstd + row, c_rs)
    tl.store(gate_mean + row, g_mu)
    tl.store(gate_rstd + row, g_rs)


@triton.jit
def _gated_ln_bwd(
    core_x,
    gate_x,
    core_lb,
    gate_lb,
    core_w,
    core_b,
    gate_w,
    gate_b,
    res_weight,
    grad_out,
    core_mean,
    core_rstd,
    gate_mean,
    gate_rstd,
    grad_core_x,
    grad_gate_x,
    grad_residual,
    D: tl.constexpr,
    BLK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLK)
    mask = c < D

    core_val = tl.load(core_x + row * D + c, mask=mask, other=0.0).to(tl.float32)
    gate_val = tl.load(gate_x + row * D + c, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        # D13a: the second linears' bias, added here instead of by their GEMM epilogue.
        core_val += tl.load(core_lb + c, mask=mask, other=0.0).to(tl.float32)
        gate_val += tl.load(gate_lb + c, mask=mask, other=0.0).to(tl.float32)
    core_w_val = tl.load(core_w + c, mask=mask, other=0.0)
    gate_w_val = tl.load(gate_w + c, mask=mask, other=0.0)
    core_b_val = tl.load(core_b + c, mask=mask, other=0.0)
    gate_b_val = tl.load(gate_b + c, mask=mask, other=0.0)
    grad = tl.load(grad_out + row * D + c, mask=mask, other=0.0).to(tl.float32)

    c_mu = tl.load(core_mean + row)
    c_rs = tl.load(core_rstd + row)
    c_xhat = (core_val - c_mu) * c_rs
    c_y = c_xhat * core_w_val + core_b_val
    c_sig = tl.sigmoid(c_y)
    core = c_y * c_sig

    g_mu = tl.load(gate_mean + row)
    g_rs = tl.load(gate_rstd + row)
    g_xhat = (gate_val - g_mu) * g_rs
    g_y = g_xhat * gate_w_val + gate_b_val
    gate = tl.sigmoid(g_y)

    d_core_y = grad * gate * (c_sig * (1.0 + c_y * (1.0 - c_sig)))
    d_gate_y = grad * core * (gate * (1.0 - gate))

    d_core_xhat = d_core_y * core_w_val
    c_mean_dxhat = tl.sum(tl.where(mask, d_core_xhat, 0.0), axis=0) / D
    c_mean_dxhat_xhat = tl.sum(tl.where(mask, d_core_xhat * c_xhat, 0.0), axis=0) / D
    tl.store(
        grad_core_x + row * D + c,
        (d_core_xhat - c_mean_dxhat - c_xhat * c_mean_dxhat_xhat) * c_rs,
        mask=mask,
    )

    d_gate_xhat = d_gate_y * gate_w_val
    g_mean_dxhat = tl.sum(tl.where(mask, d_gate_xhat, 0.0), axis=0) / D
    g_mean_dxhat_xhat = tl.sum(tl.where(mask, d_gate_xhat * g_xhat, 0.0), axis=0) / D
    tl.store(
        grad_gate_x + row * D + c,
        (d_gate_xhat - g_mean_dxhat - g_xhat * g_mean_dxhat_xhat) * g_rs,
        mask=mask,
    )
    if HAS_RESIDUAL:
        w = tl.load(res_weight + c, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_residual + row * D + c, grad * w, mask=mask)


@triton.jit(do_not_specialize=["N"])
def _gated_ln_fwd_2d(
    core_x,
    gate_x,
    core_lb,
    gate_lb,
    core_w,
    core_b,
    gate_w,
    gate_b,
    residual,
    res_weight,
    out,
    core_mean,
    core_rstd,
    gate_mean,
    gate_rstd,
    N,
    D: tl.constexpr,
    BLK: tl.constexpr,
    BM: tl.constexpr,
    EPS_CORE: tl.constexpr,
    EPS_GATE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # _gated_ln_fwd on a [BM, BLK] tile: the same formulas, per-row sums along axis 1.
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    c = tl.arange(0, BLK)
    row_mask = rows < N
    col_mask = c < D
    mask = row_mask[:, None] & col_mask[None, :]
    offs = rows[:, None] * D + c[None, :]

    core_val = tl.load(core_x + offs, mask=mask, other=0.0).to(tl.float32)
    gate_val = tl.load(gate_x + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        core_val += tl.load(core_lb + c, mask=col_mask, other=0.0).to(tl.float32)[None, :]
        gate_val += tl.load(gate_lb + c, mask=col_mask, other=0.0).to(tl.float32)[None, :]

    c_mu = tl.sum(core_val, axis=1) / D
    c_centered = tl.where(mask, core_val - c_mu[:, None], 0.0)
    c_rs = 1.0 / tl.sqrt(tl.sum(c_centered * c_centered, axis=1) / D + EPS_CORE)
    c_y = (
        c_centered
        * c_rs[:, None]
        * tl.load(core_w + c, mask=col_mask, other=0.0)[None, :]
        + tl.load(core_b + c, mask=col_mask, other=0.0)[None, :]
    )
    c_sig = tl.sigmoid(c_y)
    core = c_y * c_sig

    g_mu = tl.sum(gate_val, axis=1) / D
    g_centered = tl.where(mask, gate_val - g_mu[:, None], 0.0)
    g_rs = 1.0 / tl.sqrt(tl.sum(g_centered * g_centered, axis=1) / D + EPS_GATE)
    g_y = (
        g_centered
        * g_rs[:, None]
        * tl.load(gate_w + c, mask=col_mask, other=0.0)[None, :]
        + tl.load(gate_b + c, mask=col_mask, other=0.0)[None, :]
    )
    gate = tl.sigmoid(g_y)

    result = core * gate
    if HAS_RESIDUAL:
        r = tl.load(residual + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(res_weight + c, mask=col_mask, other=0.0).to(tl.float32)
        result += r * w[None, :]
    tl.store(out + offs, result, mask=mask)
    tl.store(core_mean + rows, c_mu, mask=row_mask)
    tl.store(core_rstd + rows, c_rs, mask=row_mask)
    tl.store(gate_mean + rows, g_mu, mask=row_mask)
    tl.store(gate_rstd + rows, g_rs, mask=row_mask)


@triton.jit(do_not_specialize=["N"])
def _gated_ln_bwd_2d(
    core_x,
    gate_x,
    core_lb,
    gate_lb,
    core_w,
    core_b,
    gate_w,
    gate_b,
    res_weight,
    grad_out,
    core_mean,
    core_rstd,
    gate_mean,
    gate_rstd,
    grad_core_x,
    grad_gate_x,
    grad_residual,
    N,
    D: tl.constexpr,
    BLK: tl.constexpr,
    BM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # _gated_ln_bwd on a [BM, BLK] tile: the same formulas, per-row sums along axis 1.
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    c = tl.arange(0, BLK)
    row_mask = rows < N
    col_mask = c < D
    mask = row_mask[:, None] & col_mask[None, :]
    offs = rows[:, None] * D + c[None, :]

    core_val = tl.load(core_x + offs, mask=mask, other=0.0).to(tl.float32)
    gate_val = tl.load(gate_x + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        core_val += tl.load(core_lb + c, mask=col_mask, other=0.0).to(tl.float32)[None, :]
        gate_val += tl.load(gate_lb + c, mask=col_mask, other=0.0).to(tl.float32)[None, :]
    core_w_val = tl.load(core_w + c, mask=col_mask, other=0.0)[None, :]
    gate_w_val = tl.load(gate_w + c, mask=col_mask, other=0.0)[None, :]
    core_b_val = tl.load(core_b + c, mask=col_mask, other=0.0)[None, :]
    gate_b_val = tl.load(gate_b + c, mask=col_mask, other=0.0)[None, :]
    grad = tl.load(grad_out + offs, mask=mask, other=0.0).to(tl.float32)

    c_mu = tl.load(core_mean + rows, mask=row_mask, other=0.0)[:, None]
    c_rs = tl.load(core_rstd + rows, mask=row_mask, other=0.0)[:, None]
    c_xhat = (core_val - c_mu) * c_rs
    c_y = c_xhat * core_w_val + core_b_val
    c_sig = tl.sigmoid(c_y)
    core = c_y * c_sig

    g_mu = tl.load(gate_mean + rows, mask=row_mask, other=0.0)[:, None]
    g_rs = tl.load(gate_rstd + rows, mask=row_mask, other=0.0)[:, None]
    g_xhat = (gate_val - g_mu) * g_rs
    g_y = g_xhat * gate_w_val + gate_b_val
    gate = tl.sigmoid(g_y)

    d_core_y = grad * gate * (c_sig * (1.0 + c_y * (1.0 - c_sig)))
    d_gate_y = grad * core * (gate * (1.0 - gate))

    d_core_xhat = d_core_y * core_w_val
    c_mean_dxhat = tl.sum(tl.where(mask, d_core_xhat, 0.0), axis=1)[:, None] / D
    c_mean_dxhat_xhat = tl.sum(tl.where(mask, d_core_xhat * c_xhat, 0.0), axis=1)[:, None] / D
    tl.store(
        grad_core_x + offs,
        (d_core_xhat - c_mean_dxhat - c_xhat * c_mean_dxhat_xhat) * c_rs,
        mask=mask,
    )

    d_gate_xhat = d_gate_y * gate_w_val
    g_mean_dxhat = tl.sum(tl.where(mask, d_gate_xhat, 0.0), axis=1)[:, None] / D
    g_mean_dxhat_xhat = tl.sum(tl.where(mask, d_gate_xhat * g_xhat, 0.0), axis=1)[:, None] / D
    tl.store(
        grad_gate_x + offs,
        (d_gate_xhat - g_mean_dxhat - g_xhat * g_mean_dxhat_xhat) * g_rs,
        mask=mask,
    )
    if HAS_RESIDUAL:
        w = tl.load(res_weight + c, mask=col_mask, other=0.0).to(tl.float32)
        tl.store(grad_residual + offs, grad * w[None, :], mask=mask)


def _linear_biases(core_lb, gate_lb, placeholder):
    """(core bias, gate bias, HAS_BIAS) kernel arguments; a placeholder pointer without biases."""
    if core_lb is None:
        return placeholder, placeholder, False
    return core_lb.contiguous(), gate_lb.contiguous(), True


def _linear_bias_grads(ctx, first, grad_core_x, grad_gate_x):
    """Gradients of the folded biases (inputs ``first``, ``first + 1``): the column sums of
    the inputs' gradients, where required (the biases of a frozen model are not)."""
    return (grad_core_x.sum(0) if ctx.needs_input_grad[first] else None,
            grad_gate_x.sum(0) if ctx.needs_input_grad[first + 1] else None)


class _GatedLN(Function):
    @staticmethod
    def forward(ctx, core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate, core_lb, gate_lb):
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = [not core_x.is_contiguous(), not gate_x.is_contiguous()]
        core_x = core_x.contiguous()
        gate_x = gate_x.contiguous()
        core_lb, gate_lb, has_bias = _linear_biases(core_lb, gate_lb, core_w)
        rows, dim = core_x.shape
        block = triton.next_power_of_2(dim)
        out = torch.empty_like(core_x)
        core_mean = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        core_rstd = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        gate_mean = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        gate_rstd = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        tile = _tile_2d(block, False) if _TILE_2D else None
        if tile is None:
            _gated_ln_fwd[(rows,)](
                core_x,
                gate_x,
                core_lb,
                gate_lb,
                core_w.contiguous(),
                core_b.contiguous(),
                gate_w.contiguous(),
                gate_b.contiguous(),
                core_x,
                core_w,
                out,
                core_mean,
                core_rstd,
                gate_mean,
                gate_rstd,
                D=dim,
                BLK=block,
                EPS_CORE=eps_core,
                EPS_GATE=eps_gate,
                HAS_RESIDUAL=False,
                HAS_BIAS=has_bias,
                num_warps=1,
            )
        else:
            _gated_ln_fwd_2d[(triton.cdiv(rows, tile[0]),)](
                core_x, gate_x, core_lb, gate_lb, core_w.contiguous(), core_b.contiguous(),
                gate_w.contiguous(), gate_b.contiguous(), core_x, core_w, out, core_mean, core_rstd,
                gate_mean, gate_rstd, rows, D=dim, BLK=block, BM=tile[0],
                EPS_CORE=eps_core, EPS_GATE=eps_gate, HAS_RESIDUAL=False, HAS_BIAS=has_bias,
                num_warps=tile[1],
            )
        ctx.save_for_backward(
            core_x,
            gate_x,
            core_w,
            core_b,
            gate_w,
            gate_b,
            core_lb,
            gate_lb,
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
        )
        ctx._tile_2d = tile is not None
        ctx._has_bias = has_bias
        if rec is not None:
            ctx._opsel_site = rec.site()
            params = {
                "rows": rows, "width": dim, "BLK": block, "num_warps": 1,
                "HAS_RESIDUAL": False, "EPS": [eps_core, eps_gate], "dtype": core_x.dtype,
                "contiguous_copy": copied,
            }
            if has_bias:
                params["linear_bias"] = True
            if tile is not None:
                params.update(BM=tile[0], num_warps=tile[1], tile="2d")
            rec.record("gated_ln", "triton", params)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            core_x,
            gate_x,
            core_w,
            core_b,
            gate_w,
            gate_b,
            core_lb,
            gate_lb,
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
        ) = ctx.saved_tensors
        rows, dim = core_x.shape
        block = triton.next_power_of_2(dim)
        grad_core_x = torch.empty_like(core_x)
        grad_gate_x = torch.empty_like(gate_x)
        tile = _tile_2d(block, True) if ctx._tile_2d else None
        if tile is None:
            _gated_ln_bwd[(rows,)](
                core_x,
                gate_x,
                core_lb,
                gate_lb,
                core_w,
                core_b,
                gate_w,
                gate_b,
                core_w,
                grad_out.contiguous(),
                core_mean,
                core_rstd,
                gate_mean,
                gate_rstd,
                grad_core_x,
                grad_gate_x,
                grad_core_x,
                D=dim,
                BLK=block,
                HAS_RESIDUAL=False,
                HAS_BIAS=ctx._has_bias,
                num_warps=1,
            )
        else:
            _gated_ln_bwd_2d[(triton.cdiv(rows, tile[0]),)](
                core_x, gate_x, core_lb, gate_lb, core_w, core_b, gate_w, gate_b, core_w,
                grad_out.contiguous(), core_mean, core_rstd, gate_mean, gate_rstd, grad_core_x,
                grad_gate_x, grad_core_x, rows, D=dim, BLK=block, BM=tile[0],
                HAS_RESIDUAL=False, HAS_BIAS=ctx._has_bias, num_warps=tile[1],
            )
        rec = _recorder._ACTIVE
        if rec is not None:
            params = {
                "rows": rows, "width": dim, "BLK": block, "num_warps": 1, "HAS_RESIDUAL": False,
                "contiguous_copy": not grad_out.is_contiguous(),
            }
            if tile is not None:
                params.update(BM=tile[0], num_warps=tile[1], tile="2d")
            rec.record("gated_ln", "triton", params,
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return (grad_core_x, grad_gate_x, None, None, None, None, None, None,
                *_linear_bias_grads(ctx, 8, grad_core_x, grad_gate_x))


class _GatedLNResidual(Function):
    @staticmethod
    def forward(
        ctx,
        core_x,
        gate_x,
        core_w,
        core_b,
        gate_w,
        gate_b,
        residual,
        res_weight,
        eps_core,
        eps_gate,
        core_lb,
        gate_lb,
    ):
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = [not core_x.is_contiguous(), not gate_x.is_contiguous(),
                      not residual.is_contiguous()]
        core_x = core_x.contiguous()
        gate_x = gate_x.contiguous()
        residual = residual.contiguous()
        res_weight = res_weight.reshape(-1).contiguous()
        core_lb, gate_lb, has_bias = _linear_biases(core_lb, gate_lb, core_w)
        rows, dim = core_x.shape
        block = triton.next_power_of_2(dim)
        out = torch.empty_like(core_x)
        core_mean = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        core_rstd = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        gate_mean = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        gate_rstd = torch.empty(rows, device=core_x.device, dtype=torch.float32)
        tile = _tile_2d(block, False) if _TILE_2D else None
        if tile is None:
            _gated_ln_fwd[(rows,)](
                core_x,
                gate_x,
                core_lb,
                gate_lb,
                core_w.contiguous(),
                core_b.contiguous(),
                gate_w.contiguous(),
                gate_b.contiguous(),
                residual,
                res_weight,
                out,
                core_mean,
                core_rstd,
                gate_mean,
                gate_rstd,
                D=dim,
                BLK=block,
                EPS_CORE=eps_core,
                EPS_GATE=eps_gate,
                HAS_RESIDUAL=True,
                HAS_BIAS=has_bias,
                num_warps=1,
            )
        else:
            _gated_ln_fwd_2d[(triton.cdiv(rows, tile[0]),)](
                core_x, gate_x, core_lb, gate_lb, core_w.contiguous(), core_b.contiguous(),
                gate_w.contiguous(), gate_b.contiguous(), residual, res_weight, out, core_mean,
                core_rstd, gate_mean, gate_rstd, rows, D=dim, BLK=block, BM=tile[0],
                EPS_CORE=eps_core, EPS_GATE=eps_gate, HAS_RESIDUAL=True, HAS_BIAS=has_bias,
                num_warps=tile[1],
            )
        ctx.save_for_backward(
            core_x,
            gate_x,
            core_w,
            core_b,
            gate_w,
            gate_b,
            res_weight,
            core_lb,
            gate_lb,
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
        )
        ctx._tile_2d = tile is not None
        ctx._has_bias = has_bias
        if rec is not None:
            ctx._opsel_site = rec.site()
            params = {
                "rows": rows, "width": dim, "BLK": block, "num_warps": 1,
                "HAS_RESIDUAL": True, "EPS": [eps_core, eps_gate], "dtype": core_x.dtype,
                "contiguous_copy": copied,
            }
            if has_bias:
                params["linear_bias"] = True
            if tile is not None:
                params.update(BM=tile[0], num_warps=tile[1], tile="2d")
            rec.record("gated_ln_residual", "triton", params)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            core_x,
            gate_x,
            core_w,
            core_b,
            gate_w,
            gate_b,
            res_weight,
            core_lb,
            gate_lb,
            core_mean,
            core_rstd,
            gate_mean,
            gate_rstd,
        ) = ctx.saved_tensors
        rows, dim = core_x.shape
        block = triton.next_power_of_2(dim)
        grad_core_x = torch.empty_like(core_x)
        grad_gate_x = torch.empty_like(gate_x)
        grad_residual = torch.empty_like(core_x)
        tile = _tile_2d(block, True) if ctx._tile_2d else None
        if tile is None:
            _gated_ln_bwd[(rows,)](
                core_x,
                gate_x,
                core_lb,
                gate_lb,
                core_w,
                core_b,
                gate_w,
                gate_b,
                res_weight,
                grad_out.contiguous(),
                core_mean,
                core_rstd,
                gate_mean,
                gate_rstd,
                grad_core_x,
                grad_gate_x,
                grad_residual,
                D=dim,
                BLK=block,
                HAS_RESIDUAL=True,
                HAS_BIAS=ctx._has_bias,
                num_warps=1,
            )
        else:
            _gated_ln_bwd_2d[(triton.cdiv(rows, tile[0]),)](
                core_x, gate_x, core_lb, gate_lb, core_w, core_b, gate_w, gate_b, res_weight,
                grad_out.contiguous(), core_mean, core_rstd, gate_mean, gate_rstd, grad_core_x,
                grad_gate_x, grad_residual, rows, D=dim, BLK=block, BM=tile[0],
                HAS_RESIDUAL=True, HAS_BIAS=ctx._has_bias, num_warps=tile[1],
            )
        rec = _recorder._ACTIVE
        if rec is not None:
            params = {
                "rows": rows, "width": dim, "BLK": block, "num_warps": 1, "HAS_RESIDUAL": True,
                "contiguous_copy": not grad_out.is_contiguous(),
            }
            if tile is not None:
                params.update(BM=tile[0], num_warps=tile[1], tile="2d")
            rec.record("gated_ln_residual", "triton", params,
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return (
            grad_core_x,
            grad_gate_x,
            None,
            None,
            None,
            None,
            grad_residual,
            None,
            None,
            None,
            *_linear_bias_grads(ctx, 10, grad_core_x, grad_gate_x),
        )


def _record_fallback(rec, op, reason, core_x):
    rec.fallback(op, "triton", reason, {"rows": core_x.shape[0], "width": core_x.shape[-1]})


def _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate, core_lb=None, gate_lb=None):
    from fastmd.models.matris.op.triton_layernorm_act import ACT_SIGMOID
    from fastmd.models.matris.op.triton_layernorm_act import ACT_SILU
    from fastmd.models.matris.op.triton_layernorm_act import fused_ln_act

    if core_lb is not None:
        core_x, gate_x = core_x + core_lb, gate_x + gate_lb

    core = fused_ln_act(core_x, core_w, core_b, eps_core, ACT_SILU)
    gate = fused_ln_act(gate_x, gate_w, gate_b, eps_gate, ACT_SIGMOID)
    return core * gate


def fused_gated_ln(
    core_x: torch.Tensor,
    gate_x: torch.Tensor,
    core_w: torch.Tensor,
    core_b: torch.Tensor,
    gate_w: torch.Tensor,
    gate_b: torch.Tensor,
    eps_core: float = 1e-5,
    eps_gate: float = 1e-5,
    core_lb: torch.Tensor | None = None,
    gate_lb: torch.Tensor | None = None,
) -> torch.Tensor:
    """``silu(LN(core_x + core_lb)) * sigmoid(LN(gate_x + gate_lb))``; ``core_lb``/``gate_lb``
    (both or neither): the preceding linears' biases, added in the kernel (D13a)."""
    if not (core_x.is_cuda and gate_x.is_cuda):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_fallback(rec, "gated_ln", "not_cuda", core_x)
        return _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate, core_lb, gate_lb)
    if core_x.shape != gate_x.shape:
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_fallback(rec, "gated_ln", "shape_mismatch", core_x)
        return _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate, core_lb, gate_lb)
    if core_w.requires_grad or core_b.requires_grad or gate_w.requires_grad or gate_b.requires_grad:
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_fallback(rec, "gated_ln", "trainable_norm", core_x)
        return _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate, core_lb, gate_lb)
    return _GatedLN.apply(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate, core_lb, gate_lb)


def fused_gated_ln_residual(
    core_x: torch.Tensor,
    gate_x: torch.Tensor,
    core_w: torch.Tensor,
    core_b: torch.Tensor,
    gate_w: torch.Tensor,
    gate_b: torch.Tensor,
    residual: torch.Tensor,
    res_weight: torch.Tensor,
    eps_core: float = 1e-5,
    eps_gate: float = 1e-5,
    core_lb: torch.Tensor | None = None,
    gate_lb: torch.Tensor | None = None,
) -> torch.Tensor:
    """``fused_gated_ln(...) + res_weight * residual``."""
    if not (core_x.is_cuda and gate_x.is_cuda and residual.is_cuda and res_weight.is_cuda):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_fallback(rec, "gated_ln_residual", "not_cuda", core_x)
        return (
            _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate, core_lb, gate_lb)
            + res_weight * residual
        )
    if core_x.shape != gate_x.shape or residual.shape != core_x.shape:
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_fallback(rec, "gated_ln_residual", "shape_mismatch", core_x)
        return (
            _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate, core_lb, gate_lb)
            + res_weight * residual
        )
    if core_x.shape[1] != res_weight.numel():
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_fallback(rec, "gated_ln_residual", "width_mismatch", core_x)
        return (
            _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate, core_lb, gate_lb)
            + res_weight * residual
        )
    if (
        core_w.requires_grad
        or core_b.requires_grad
        or gate_w.requires_grad
        or gate_b.requires_grad
        or res_weight.requires_grad
    ):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_fallback(rec, "gated_ln_residual", "trainable_parameter", core_x)
        return (
            _fallback(core_x, gate_x, core_w, core_b, gate_w, gate_b, eps_core, eps_gate, core_lb, gate_lb)
            + res_weight * residual
        )
    return _GatedLNResidual.apply(
        core_x,
        gate_x,
        core_w,
        core_b,
        gate_w,
        gate_b,
        residual,
        res_weight,
        eps_core,
        eps_gate,
        core_lb,
        gate_lb,
    )
