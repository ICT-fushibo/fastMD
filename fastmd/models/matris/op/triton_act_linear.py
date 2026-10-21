"""``linear(silu(x))`` of both GatedMLP halves in one kernel (saved-tensor recompute).

The saved-activation recompute (``fastmd.engine.saved_recompute``) rebuilds a
GatedMLP's gate inputs ``core_x = W2c silu(pre_c) + b2c`` and
``gate_x = W2g silu(pre_g) + b2g`` from the first-layer pre-activations in the
backward. Through torch that is a SiLU, two cuBLAS GEMMs and their bias
epilogues per half, each a full pass over the rows; this kernel reads the
pre-activations once and writes both outputs. Only the backward consumes the
result, so it need not be bitwise the forward's: the dot products use
``InferenceConfig.triton_dot_precision`` (by default 3xTF32, FP32-level
accuracy, ~1e-6 relative on this data, bound at registration where cuBLAS
runs IEEE FP32; where it runs TF32 the forward's cuBLAS layers rebuild the
values; IEEE FP32 dots have no tensor-core path). Line-graph widths only
(K <= 32): at the atom layers' K=128 the tile spills and cuBLAS is faster.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from fastmd.engine.kernels.row_count import static_under_compile

_BM = 64


@triton.jit(do_not_specialize=["M"])
def _act_linear_pair_fwd(xc, xg, wc, bc, wg, bg, oc, og, M, SXC, SXG,
                         K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
                         SILU: tl.constexpr, HAS_BIAS: tl.constexpr, PRECISION: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    live = rows < M
    k = tl.arange(0, K)
    n = tl.arange(0, N)
    for half in tl.static_range(2):
        x_ptr = xc if half == 0 else xg
        stride = SXC if half == 0 else SXG
        w_ptr = wc if half == 0 else wg
        b_ptr = bc if half == 0 else bg
        o_ptr = oc if half == 0 else og
        a = tl.load(x_ptr + rows[:, None] * stride + k[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        if SILU:
            a = a * tl.sigmoid(a)
        w = tl.load(w_ptr + n[:, None] * K + k[None, :]).to(tl.float32)
        acc = tl.dot(a, tl.trans(w), input_precision=PRECISION)
        if HAS_BIAS:
            acc += tl.load(b_ptr + n).to(tl.float32)[None, :]
        tl.store(o_ptr + rows[:, None] * N + n[None, :], acc, mask=live[:, None])


def act_linear_pair_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    k, n = weight.shape[1], weight.shape[0]
    return (x.is_cuda and x.dtype == torch.float32 and weight.dtype == torch.float32
            and x.dim() == 2 and x.shape[1] == k and x.stride(1) == 1
            and 16 <= k <= 32 and 16 <= n <= 64 and k & (k - 1) == 0 and n & (n - 1) == 0)


def act_linear_pair(core_in, gate_in, core_linear, gate_linear, *, silu: bool, bias: bool = True,
                    precision: str = "tf32x3"):
    """``(core_linear(silu(core_in)), gate_linear(silu(gate_in)))`` (``silu=False``: no activation;
    ``bias=False``: without the linears' biases, D13a; ``precision``: the dots' input precision)."""
    rows = core_in.shape[0]
    n = core_linear.weight.shape[0]
    core_x = torch.empty((rows, n), device=core_in.device, dtype=core_in.dtype)
    gate_x = torch.empty_like(core_x)
    has_bias = bias and core_linear.bias is not None
    wc, wg = core_linear.weight.contiguous(), gate_linear.weight.contiguous()
    _act_linear_pair_fwd[(triton.cdiv(rows, _BM),)](
        core_in, gate_in, wc, core_linear.bias if has_bias else wc, wg,
        gate_linear.bias if has_bias else wg, core_x, gate_x, static_under_compile(rows),
        core_in.stride(0), gate_in.stride(0), K=wc.shape[1], N=n, BM=_BM, SILU=silu,
        HAS_BIAS=has_bias, PRECISION=precision, num_warps=4,
    )
    return core_x, gate_x
