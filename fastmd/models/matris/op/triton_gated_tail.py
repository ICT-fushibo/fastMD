"""A GatedMLP's narrow tail in one kernel per direction (C2 = step 2.5b item D13b).

Forward, per row tile: ``core = core_h @ Wc^T + bc`` and ``gate = gate_h @ Wg^T + bg`` (both second linears,
K -> D with K <= 32), then the gated-LayerNorm finish of ``triton_gated_ln``:
``silu(LN(core)) * sigmoid(LN(gate))``. The kernel also stores ``core``/``gate``, the tensors the gated
LayerNorm saves today; the two GEMMs' write and re-read of them, and their separate launches, are what it
saves. Backward, per row tile: the gated-LayerNorm backward from the saved ``core``/``gate`` and
``dX = dY @ W`` for both inputs in the same kernel (the full-recompute variant, four dots per tile, measured
slower than the unfused path).

Frozen weights, biases and norms only: no parameter gradients (GatedMLP declines otherwise; training runs
plain torch). The dots' input precision is ``InferenceConfig.triton_dot_precision``
(``fastmd.precision.triton_dot_input_precision("gated_tail")``): by default 3xTF32 where cuBLAS's FP32 GEMMs run
IEEE FP32 (``fastmd.precision.fp32_gemm_mode``; FP32-level accuracy, the results differ from cuBLAS within
rounding) and TF32 where they run TF32; ieee FP32 FMA dots (faster than the unfused cuBLAS IEEE path). The rows
are a runtime argument excluded from Triton's value specialization, like the 2-D gated-LN tiles.

Saved-tensor recompute (on/deep): the forward registers its saved linear outputs and its output with one
recipe, the same forward kernel on the (kept or rebuilt) inputs, so the rebuilt tensors are bitwise the
forward's and the forward is the one of the off mode.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd import Function

from fastmd.engine import recorder as _recorder
from fastmd.engine import saved_recompute as _saved_recompute
from fastmd.precision import triton_dot_input_precision

# (BM rows, num_warps), measured on GB200 at K = D = 32 (7.6M rows): the forward holds two [BM, 32]
# accumulators and their LayerNorms, the backward also the gradients of both.
_FWD_TILE = (32, 2)
_BWD_TILE = (16, 1)


@triton.jit(do_not_specialize=["M"])
def _gated_tail_fwd(core_h, gate_h, core_w, core_lb, gate_w, gate_lb, core_nw, core_nb, gate_nw, gate_nb,
                    core_x, gate_x, out, M, SC, SG,
                    K: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, EPS_CORE: tl.constexpr,
                    EPS_GATE: tl.constexpr, HAS_BIAS: tl.constexpr, PRECISION: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    live = rows < M
    k = tl.arange(0, K)
    n = tl.arange(0, D)
    a = tl.load(core_h + rows[:, None] * SC + k[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    core = tl.dot(a, tl.trans(tl.load(core_w + n[:, None] * K + k[None, :])), input_precision=PRECISION)
    a = tl.load(gate_h + rows[:, None] * SG + k[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    gate = tl.dot(a, tl.trans(tl.load(gate_w + n[:, None] * K + k[None, :])), input_precision=PRECISION)
    if HAS_BIAS:
        core += tl.load(core_lb + n)[None, :]
        gate += tl.load(gate_lb + n)[None, :]
    offs = rows[:, None] * D + n[None, :]
    tl.store(core_x + offs, core, mask=live[:, None])
    tl.store(gate_x + offs, gate, mask=live[:, None])

    # _gated_ln_fwd_2d on the tile (D is a power of two: no column mask).
    c_mu = tl.sum(core, axis=1) / D
    c_centered = core - c_mu[:, None]
    c_rs = 1.0 / tl.sqrt(tl.sum(c_centered * c_centered, axis=1) / D + EPS_CORE)
    c_y = c_centered * c_rs[:, None] * tl.load(core_nw + n)[None, :] + tl.load(core_nb + n)[None, :]
    g_mu = tl.sum(gate, axis=1) / D
    g_centered = gate - g_mu[:, None]
    g_rs = 1.0 / tl.sqrt(tl.sum(g_centered * g_centered, axis=1) / D + EPS_GATE)
    g_y = g_centered * g_rs[:, None] * tl.load(gate_nw + n)[None, :] + tl.load(gate_nb + n)[None, :]
    tl.store(out + offs, c_y * tl.sigmoid(c_y) * tl.sigmoid(g_y), mask=live[:, None])


@triton.jit(do_not_specialize=["M"])
def _gated_tail_bwd(core_x, gate_x, core_w, gate_w, core_nw, core_nb, gate_nw, gate_nb, grad_out,
                    grad_core_h, grad_gate_h, M,
                    K: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, EPS_CORE: tl.constexpr,
                    EPS_GATE: tl.constexpr, PRECISION: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    live = rows < M
    k = tl.arange(0, K)
    n = tl.arange(0, D)
    offs = rows[:, None] * D + n[None, :]
    core = tl.load(core_x + offs, mask=live[:, None], other=0.0)
    gate = tl.load(gate_x + offs, mask=live[:, None], other=0.0)
    grad = tl.load(grad_out + offs, mask=live[:, None], other=0.0).to(tl.float32)
    core_w_val = tl.load(core_nw + n)[None, :]
    gate_w_val = tl.load(gate_nw + n)[None, :]

    # _gated_ln_bwd_2d on the tile, with the row statistics recomputed from the saved inputs.
    c_mu = tl.sum(core, axis=1) / D
    c_centered = core - c_mu[:, None]
    c_rs = (1.0 / tl.sqrt(tl.sum(c_centered * c_centered, axis=1) / D + EPS_CORE))[:, None]
    c_xhat = c_centered * c_rs
    c_y = c_xhat * core_w_val + tl.load(core_nb + n)[None, :]
    c_sig = tl.sigmoid(c_y)
    g_mu = tl.sum(gate, axis=1) / D
    g_centered = gate - g_mu[:, None]
    g_rs = (1.0 / tl.sqrt(tl.sum(g_centered * g_centered, axis=1) / D + EPS_GATE))[:, None]
    g_xhat = g_centered * g_rs
    gate_act = tl.sigmoid(g_xhat * gate_w_val + tl.load(gate_nb + n)[None, :])
    d_core_y = grad * gate_act * (c_sig * (1.0 + c_y * (1.0 - c_sig)))
    d_gate_y = grad * (c_y * c_sig) * (gate_act * (1.0 - gate_act))
    d_core_xhat = d_core_y * core_w_val
    d_core = (d_core_xhat - (tl.sum(d_core_xhat, axis=1) / D)[:, None]
              - c_xhat * (tl.sum(d_core_xhat * c_xhat, axis=1) / D)[:, None]) * c_rs
    d_gate_xhat = d_gate_y * gate_w_val
    d_gate = (d_gate_xhat - (tl.sum(d_gate_xhat, axis=1) / D)[:, None]
              - g_xhat * (tl.sum(d_gate_xhat * g_xhat, axis=1) / D)[:, None]) * g_rs

    # dX = dY @ W for both second linears.
    hoffs = rows[:, None] * K + k[None, :]
    tl.store(grad_core_h + hoffs,
             tl.dot(d_core, tl.load(core_w + n[:, None] * K + k[None, :]), input_precision=PRECISION),
             mask=live[:, None])
    tl.store(grad_gate_h + hoffs,
             tl.dot(d_gate, tl.load(gate_w + n[:, None] * K + k[None, :]), input_precision=PRECISION),
             mask=live[:, None])


def _precision() -> str | None:
    return triton_dot_input_precision("gated_tail")


def _saved_tensor_hooks_active() -> bool:
    """Whether saved-tensor hooks (a non-reentrant checkpoint, saved-tensor recompute) may drop saved tensors."""
    top = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
    return top is not None and top(False) is not None


def _launch_forward(core_h, gate_h, core_w, core_lb, gate_w, gate_lb, core_nw, core_nb, gate_nw, gate_nb,
                    eps_core, eps_gate, precision, store_x=True, shared=True, inplace=False):
    """(core_x, gate_x, out): the saved linear outputs and the finish (contiguous weights and norms).

    ``shared``: the three results in one allocation (and the backward's two gradients in one): with separate
    ones the caching allocator's segments fragmented, up to +3.9 GiB reserved at 21,504 atoms. Only while the
    saved tensors are kept as they are: where saved-tensor hooks may drop the linear outputs, a shared out
    would keep them alive (+5.4 GiB reserved under a layer checkpoint at 21,504 atoms). ``store_x`` False
    (nothing to save): the linear outputs go to a scratch buffer freed on return, and core_x/gate_x are
    None (the same kernel: out is bitwise the one of a forward with a backward). ``inplace``: the linear
    outputs overwrite ``core_h``/``gate_h`` (each tile reads its rows before writing them), and only out is
    allocated; see ``fused_gated_tail``."""
    rows = core_h.shape[0]
    d, k = core_w.shape
    if inplace:
        core_x, gate_x = core_h, gate_h
        out = torch.empty((rows, d), device=core_h.device, dtype=torch.float32)
    elif store_x and shared:
        core_x, gate_x, out = torch.empty((3, rows, d), device=core_h.device, dtype=torch.float32)
    elif store_x:
        core_x = torch.empty((rows, d), device=core_h.device, dtype=torch.float32)
        gate_x = torch.empty_like(core_x)
        out = torch.empty_like(core_x)
    else:
        out = torch.empty((rows, d), device=core_h.device, dtype=torch.float32)
        core_x, gate_x = torch.empty((2, rows, d), device=core_h.device, dtype=torch.float32)
    has_bias = core_lb is not None
    bm, warps = _FWD_TILE
    _gated_tail_fwd[(triton.cdiv(rows, bm),)](
        core_h, gate_h, core_w, core_lb if has_bias else core_w, gate_w, gate_lb if has_bias else gate_w,
        core_nw, core_nb, gate_nw, gate_nb, core_x, gate_x, out, rows, core_h.stride(0), gate_h.stride(0),
        K=k, D=d, BM=bm, EPS_CORE=eps_core, EPS_GATE=eps_gate, HAS_BIAS=has_bias, PRECISION=precision,
        num_warps=warps,
    )
    if inplace:
        torch.autograd.graph.increment_version(core_h)
        torch.autograd.graph.increment_version(gate_h)
    return (core_x, gate_x, out) if store_x else (None, None, out)


def _inplace_ok(core_h, gate_h, k, d) -> bool:
    """Whether the linear outputs can overwrite the inputs: same [rows, K = D] FP32 layout, each the whole of
    its own storage (a recompute recipe rebuilds that layout as a standalone tensor)."""
    return k == d and all(
        x.dtype == torch.float32 and x.is_contiguous() and x.storage_offset() == 0
        and x.untyped_storage().nbytes() == x.numel() * x.element_size() for x in (core_h, gate_h)
    ) and core_h.untyped_storage().data_ptr() != gate_h.untyped_storage().data_ptr()


def gated_tail_supported(core_h: torch.Tensor, gate_h: torch.Tensor, core_w: torch.Tensor,
                         gate_w: torch.Tensor) -> bool:
    """FP32 CUDA rows with unit column stride, and K -> D second linears with power-of-two K in [16, 32] and
    D in [16, 64] (the line-graph GatedMLPs; wider K spills the tile and cuBLAS wins, as for act_linear_pair)."""
    d, k = core_w.shape
    for x in (core_h, gate_h):
        if not (x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.shape[1] == k
                and x.stride(1) == 1):
            return False
    return (core_h.shape[0] == gate_h.shape[0] and gate_w.shape == core_w.shape
            and core_w.dtype == gate_w.dtype == torch.float32
            and 16 <= k <= 32 and 16 <= d <= 64 and k & (k - 1) == 0 and d & (d - 1) == 0)


class _GatedTail(Function):
    @staticmethod
    def forward(ctx, core_h, gate_h, core_w, core_lb, gate_w, gate_lb, core_nw, core_nb, gate_nw, gate_nb,
                eps_core, eps_gate, inplace):
        rows = core_h.shape[0]
        d, k = core_w.shape
        params = [None if t is None else t.contiguous()
                  for t in (core_w, core_lb, gate_w, gate_lb, core_nw, core_nb, gate_nw, gate_nb)]
        core_w, core_lb, gate_w, gate_lb, core_nw, core_nb, gate_nw, gate_nb = params
        has_bias = core_lb is not None
        precision = _precision()
        bm, warps = _FWD_TILE
        backward = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        shared = not _saved_tensor_hooks_active()
        scope = _saved_recompute.active()
        # Not under other saved-tensor hooks (a layer checkpoint's): there the in-place forward raised the
        # reserved peak (+1.8 / +3.6 GiB at 10,752 / 21,504 atoms), and the separate outputs do not.
        inplace = inplace and (shared or scope is not None) and _inplace_ok(core_h, gate_h, k, d)
        # Taken before an in-place launch changes the inputs' versions (a Ref then rebuilds them).
        refs = (scope.ref(core_h), scope.ref(gate_h)) if scope is not None and backward else None
        core_x, gate_x, out = _launch_forward(core_h, gate_h, *params, eps_core, eps_gate, precision, backward,
                                              shared, inplace)
        if refs is not None:
            # Before anything saves them: the saved linear outputs and the output are rebuilt together (as
            # standalone tensors where they overwrote the inputs: same layout, the rebuilt inputs untouched).
            scope.register((core_x, gate_x, out), _launch_forward, *refs, *params, eps_core, eps_gate,
                           precision, True, shared and not inplace)
        ctx.save_for_backward(core_x, gate_x, core_w, gate_w, core_nw, core_nb, gate_nw, gate_nb)
        ctx.eps = (eps_core, eps_gate)
        ctx.precision = precision
        ctx.shared = shared
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("gated_tail", "triton", {
                "rows": rows, "K": k, "D": d, "BM": bm, "num_warps": warps, "HAS_BIAS": has_bias,
                "PRECISION": precision, "EPS": [eps_core, eps_gate], "dtype": core_h.dtype,
                "strides": [core_h.stride(0), gate_h.stride(0)],
            })
        return out

    @staticmethod
    def backward(ctx, grad_out):
        core_x, gate_x, core_w, gate_w, core_nw, core_nb, gate_nw, gate_nb = ctx.saved_tensors
        rows = core_x.shape[0]
        d, k = core_w.shape
        if getattr(ctx, "consumed", False):
            raise RuntimeError("fused_gated_tail: a second backward through the same graph (its saved linear "
                               "outputs were overwritten by the first; inference runs one backward)")
        if k == d and core_x.is_contiguous() and gate_x.is_contiguous():
            # The saved linear outputs are dead after this backward: dX overwrites them (each tile reads its
            # rows before writing them), and the backward allocates nothing.
            grad_core_h, grad_gate_h = core_x, gate_x
            ctx.consumed = True
        elif ctx.shared:
            grad_core_h, grad_gate_h = torch.empty((2, rows, k), device=core_x.device, dtype=torch.float32)
        else:
            grad_core_h = torch.empty((rows, k), device=core_x.device, dtype=torch.float32)
            grad_gate_h = torch.empty_like(grad_core_h)
        bm, warps = _BWD_TILE
        _gated_tail_bwd[(triton.cdiv(rows, bm),)](
            core_x, gate_x, core_w, gate_w, core_nw, core_nb, gate_nw, gate_nb, grad_out.contiguous(),
            grad_core_h, grad_gate_h, rows, K=k, D=d, BM=bm, EPS_CORE=ctx.eps[0], EPS_GATE=ctx.eps[1],
            PRECISION=ctx.precision, num_warps=warps,
        )
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("gated_tail", "triton", {
                "rows": rows, "K": k, "D": d, "BM": bm, "num_warps": warps, "PRECISION": ctx.precision,
                "contiguous_copy": not grad_out.is_contiguous(),
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return (grad_core_h, grad_gate_h) + (None,) * 11


def fused_gated_tail(core_h: torch.Tensor, gate_h: torch.Tensor, core_w: torch.Tensor,
                     core_lb: torch.Tensor | None, gate_w: torch.Tensor, gate_lb: torch.Tensor | None,
                     core_nw: torch.Tensor, core_nb: torch.Tensor, gate_nw: torch.Tensor,
                     gate_nb: torch.Tensor, eps_core: float = 1e-5, eps_gate: float = 1e-5,
                     inplace: bool = False) -> torch.Tensor:
    """``silu(LN(core_h @ core_w^T + core_lb)) * sigmoid(LN(gate_h @ gate_w^T + gate_lb))`` (norm weights/biases
    ``core_nw``/``core_nb``, ``gate_nw``/``gate_nb``). The caller checks ``gated_tail_supported`` and that no
    weight, bias or norm parameter requires a gradient.

    ``inplace``: the caller no longer uses ``core_h``/``gate_h`` (their autograd producers must not save them),
    so the linear outputs overwrite them where the layouts allow (K == D, standalone contiguous FP32) and the
    tail allocates only its output; their versions are bumped. Not under saved-tensor hooks other than the
    saved-tensor recompute scope's (a layer checkpoint). Without it, C2 allocated two [rows, D] tensors
    while the inputs were still alive, and the caching allocator fragmented (reserved peaks up to +5.4 GiB
    under saved-tensor recompute "on" at 21,504 atoms)."""
    return _GatedTail.apply(core_h, gate_h, core_w, core_lb, gate_w, gate_lb, core_nw, core_nb, gate_nw,
                            gate_nb, eps_core, eps_gate, inplace)
