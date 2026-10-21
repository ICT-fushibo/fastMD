"""Decomposed indexed concat first Linear with fused gather-add+SiLU.

This path exploits reuse in the line-graph indices:

    cat3: A@W0.T + B[index_a]@W1.T + B[index_b]@W2.T
    cat4: A@W0.T + X[index_a]@W1.T + B[index_b]@W2.T + B[index_c]@W3.T

The dense projections stay as high-throughput GEMMs over the source tensors;
one Triton epilogue then gathers projected rows, adds bias, applies SiLU, and
returns contiguous core/gate activations for the rest of GatedMLP.

Under ``private_sink_atomics`` (sink_atomics.py) the VJP's atomics skip rows
whose addends are all zero (``SKIP_ZERO``); outside it the kernels are unchanged.
In deterministic mode (read in the forward) the VJP runs in torch with ordered
sums (``ordered_index_add_``) instead of the atomic kernel.
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
from fastmd.engine.kernels.merged_frozen_projections import claim_grad
from fastmd.engine.kernels.merged_frozen_projections import grad_claim
from fastmd.engine.kernels.row_count import static_under_compile
from fastmd.engine.kernels.row_tiles import row_tile
from fastmd.engine.kernels.sink_atomics import zero_skip_atomics
from fastmd.engine.kernels.triton_zero_skip_index import ordered_index_add_


_LAST_ERROR: str | None = None
_DEBUG_PRINTED: set[str] = set()


def _env_int(name: str, default: int | None) -> int | None:
    value = env_value(name)
    if value is None:
        return default
    return int(value)


def _env_flag(name: str, default: bool) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "NO"}


# None: by width (row_tiles).
_EPILOGUE_BM = _env_int("MATRIS_DECOMPOSED_INDEXED_SILU_BM", None)
_EPILOGUE_BWD_BM = _env_int("MATRIS_DECOMPOSED_INDEXED_SILU_BWD_BM", 8)
_SORTED_A_PAIR_BWD = _env_flag("MATRIS_DECOMPOSED_INDEXED_SORTED_A_PAIR_BWD", True)
_GEMM_BACKEND = env_value("MATRIS_DECOMPOSED_INDEXED_GEMM_BACKEND", "torch").lower()


def _debug_check_index(name: str, index: torch.Tensor, limit: int) -> None:
    if not _env_flag("MATRIS_DECOMPOSED_INDEXED_DEBUG_CHECKS", False):
        return
    if index.numel() == 0:
        return
    lo = int(index.min().item())
    hi = int(index.max().item())
    if lo < 0 or hi >= limit:
        raise RuntimeError(f"{name} index out of bounds: min={lo} max={hi} limit={limit}")


def _debug_print_once(name: str, *parts: object) -> None:
    if not _env_flag("MATRIS_DECOMPOSED_INDEXED_DEBUG_CHECKS", False):
        return
    if name in _DEBUG_PRINTED:
        return
    _DEBUG_PRINTED.add(name)
    print("[decomposed_indexed_debug]", name, *parts, flush=True)


def _project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not torch.compiler.is_compiling():
        # The Triton row GEMM for the K <= 32 parts (C4), else F.linear; also the deep recipes of p1/p2.
        from fastmd.engine.kernels.merged_frozen_projections import project
        return project(x, weight)
    return torch.nn.functional.linear(x, weight, None)


# The row count M is a runtime argument excluded from value specialization
# (K5): a plain runtime int still specializes on ==1 and %16==0, and a ragged
# graph changes M every step, so each new count would otherwise compile a new
# kernel. Widths, tiles and index/row strides stay constexpr.
@triton.jit(do_not_specialize=["M"])
def _cat3_gather_add_silu_fwd(
    p0,
    p1,
    p2,
    index_a,
    index_b,
    bias,
    preact,
    core_act,
    gate_act,
    SIA: tl.constexpr,
    SIB: tl.constexpr,
    SP0: tl.constexpr,
    M,
    H: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BLK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # All row and gathered-index offsets are 64-bit (here and in the other
    # epilogues): p0 may be a stride-128 view of the merged projection, so a
    # 32-bit rows * SP0 wraps past 2**24 rows.
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.arange(0, BLK)
    mask = (rows[:, None] < M) & (cols[None, :] < H)
    idx_a = tl.load(index_a + rows * SIA, mask=rows < M, other=0).to(tl.int64)
    idx_b = tl.load(index_b + rows * SIB, mask=rows < M, other=0).to(tl.int64)

    base_n = rows[:, None] * N
    base_p0 = rows[:, None] * SP0
    core = tl.load(p0 + base_p0 + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(p0 + base_p0 + H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    core += tl.load(
        p1 + idx_a[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate += tl.load(
        p1 + idx_a[:, None] * N + H + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    core += tl.load(
        p2 + idx_b[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate += tl.load(
        p2 + idx_b[:, None] * N + H + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_BIAS:
        core += tl.load(bias + cols, mask=cols < H, other=0.0).to(tl.float32)[None, :]
        gate += tl.load(bias + H + cols, mask=cols < H, other=0.0).to(tl.float32)[
            None, :
        ]

    out_base = rows[:, None] * H
    tl.store(preact + base_n + cols[None, :], core, mask=mask)
    tl.store(preact + base_n + H + cols[None, :], gate, mask=mask)
    tl.store(core_act + out_base + cols[None, :], core * tl.sigmoid(core), mask=mask)
    tl.store(gate_act + out_base + cols[None, :], gate * tl.sigmoid(gate), mask=mask)


@triton.jit(do_not_specialize=["M"])
def _cat4_gather_add_silu_fwd(
    p0,
    p1,
    p2,
    p3,
    index_a,
    index_b,
    index_c,
    bias,
    preact,
    core_act,
    gate_act,
    SIA: tl.constexpr,
    SIB: tl.constexpr,
    SIC: tl.constexpr,
    SP0: tl.constexpr,
    M,
    H: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BLK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.arange(0, BLK)
    mask = (rows[:, None] < M) & (cols[None, :] < H)
    idx_a = tl.load(index_a + rows * SIA, mask=rows < M, other=0).to(tl.int64)
    idx_b = tl.load(index_b + rows * SIB, mask=rows < M, other=0).to(tl.int64)
    idx_c = tl.load(index_c + rows * SIC, mask=rows < M, other=0).to(tl.int64)

    base_n = rows[:, None] * N
    base_p0 = rows[:, None] * SP0
    core = tl.load(p0 + base_p0 + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(p0 + base_p0 + H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    core += tl.load(
        p1 + idx_a[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate += tl.load(
        p1 + idx_a[:, None] * N + H + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    core += tl.load(
        p2 + idx_b[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate += tl.load(
        p2 + idx_b[:, None] * N + H + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    core += tl.load(
        p3 + idx_c[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate += tl.load(
        p3 + idx_c[:, None] * N + H + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_BIAS:
        core += tl.load(bias + cols, mask=cols < H, other=0.0).to(tl.float32)[None, :]
        gate += tl.load(bias + H + cols, mask=cols < H, other=0.0).to(tl.float32)[
            None, :
        ]

    out_base = rows[:, None] * H
    tl.store(preact + base_n + cols[None, :], core, mask=mask)
    tl.store(preact + base_n + H + cols[None, :], gate, mask=mask)
    tl.store(core_act + out_base + cols[None, :], core * tl.sigmoid(core), mask=mask)
    tl.store(gate_act + out_base + cols[None, :], gate * tl.sigmoid(gate), mask=mask)


@triton.jit
def _atomic_rows(mask, value, SKIP_ZERO: tl.constexpr):
    # The atomic mask of ``value``. With SKIP_ZERO (design D3) rows whose
    # addends are all zero, e.g. the padding rows of a private sink, are not
    # added. The mask stays uniform along a row, so the atomics keep their
    # vector width; a zero inside a live row is still added (as without
    # SKIP_ZERO), and NaN/Inf compare != 0, so the sums are unchanged.
    if SKIP_ZERO:
        mask = mask & (tl.max((value != 0.0).to(tl.int32), axis=1) != 0)[:, None]
    return mask


@triton.jit(do_not_specialize=["M"])
def _cat3_gather_add_silu_bwd(
    grad_core,
    grad_gate,
    preact,
    index_a,
    index_b,
    grad_p0,
    grad_p1,
    grad_p2,
    SIA: tl.constexpr,
    SIB: tl.constexpr,
    M,
    H: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BLK: tl.constexpr,
    SORTED_A: tl.constexpr,
    SKIP_ZERO: tl.constexpr = False,
    GP0_S0: tl.constexpr = 0,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.arange(0, BLK)
    mask = (rows[:, None] < M) & (cols[None, :] < H)
    idx_a = tl.load(index_a + rows * SIA, mask=rows < M, other=0).to(tl.int64)
    idx_b = tl.load(index_b + rows * SIB, mask=rows < M, other=0).to(tl.int64)

    in_base = rows[:, None] * N
    out_base = rows[:, None] * H
    core = tl.load(preact + in_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(preact + in_base + H + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    gcore = tl.load(grad_core + out_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    ggate = tl.load(grad_gate + out_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )

    sig_core = tl.sigmoid(core)
    sig_gate = tl.sigmoid(gate)
    dcore = gcore * sig_core * (1.0 + core * (1.0 - sig_core))
    dgate = ggate * sig_gate * (1.0 + gate * (1.0 - sig_gate))

    # grad_p0 may be the first columns of a merged projection's gradient
    # buffer (GP0_S0 its row stride); 0 means a tensor of its own (stride N).
    if GP0_S0 == 0:
        gp0_base = in_base
    else:
        gp0_base = rows[:, None] * GP0_S0
    tl.store(grad_p0 + gp0_base + cols[None, :], dcore, mask=mask)
    tl.store(grad_p0 + gp0_base + H + cols[None, :], dgate, mask=mask)
    # grad_p0 (empty_like) is stored for every row; SKIP_ZERO masks atomics only.
    core_mask = _atomic_rows(mask, dcore, SKIP_ZERO)
    gate_mask = _atomic_rows(mask, dgate, SKIP_ZERO)
    if SORTED_A:
        # Rows pair by global parity (2k, 2k + 1), so with an odd BM a pair
        # straddles two tiles: the even row adds both adjoints if the pair
        # shares index_a, the odd row adds its own only if it does not.
        even = (rows % 2) == 0
        next_rows = rows + 1
        next_valid = even & (next_rows < M)
        idx_next = tl.load(index_a + next_rows * SIA, mask=next_valid, other=-1)
        next_mask = next_valid[:, None] & (cols[None, :] < H)
        next_in_base = next_rows[:, None] * N
        next_out_base = next_rows[:, None] * H
        core_next = tl.load(
            preact + next_in_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        gate_next = tl.load(
            preact + next_in_base + H + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        gcore_next = tl.load(
            grad_core + next_out_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        ggate_next = tl.load(
            grad_gate + next_out_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        sig_core_next = tl.sigmoid(core_next)
        sig_gate_next = tl.sigmoid(gate_next)
        dcore_next = gcore_next * sig_core_next * (
            1.0 + core_next * (1.0 - sig_core_next)
        )
        dgate_next = ggate_next * sig_gate_next * (
            1.0 + gate_next * (1.0 - sig_gate_next)
        )
        same_next = next_valid & (idx_next == idx_a)
        pair_dcore = dcore + tl.where(same_next[:, None], dcore_next, 0.0)
        pair_dgate = dgate + tl.where(same_next[:, None], dgate_next, 0.0)
        even_mask = mask & even[:, None]
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            pair_dcore,
            mask=_atomic_rows(even_mask, pair_dcore, SKIP_ZERO),
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            pair_dgate,
            mask=_atomic_rows(even_mask, pair_dgate, SKIP_ZERO),
            sem="relaxed",
        )

        prev_rows = rows - 1
        odd = ~even
        # rows < M: the tail tile's odd rows past M must not read index_a.
        prev_valid = odd & (rows > 0) & (rows < M)
        idx_prev = tl.load(index_a + prev_rows * SIA, mask=prev_valid, other=-2)
        same_prev = prev_valid & (idx_prev == idx_a)
        odd_mask = mask & odd[:, None] & (~same_prev[:, None])
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            dcore,
            mask=_atomic_rows(odd_mask, dcore, SKIP_ZERO),
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            dgate,
            mask=_atomic_rows(odd_mask, dgate, SKIP_ZERO),
            sem="relaxed",
        )
    else:
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            dcore,
            mask=core_mask,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            dgate,
            mask=gate_mask,
            sem="relaxed",
        )
    tl.atomic_add(
        grad_p2 + idx_b[:, None] * N + cols[None, :], dcore, mask=core_mask, sem="relaxed"
    )
    tl.atomic_add(
        grad_p2 + idx_b[:, None] * N + H + cols[None, :],
        dgate,
        mask=gate_mask,
        sem="relaxed",
    )


@triton.jit(do_not_specialize=["M"])
def _cat4_gather_add_silu_bwd(
    grad_core,
    grad_gate,
    preact,
    index_a,
    index_b,
    index_c,
    grad_p0,
    grad_p1,
    grad_p2,
    grad_p3,
    SIA: tl.constexpr,
    SIB: tl.constexpr,
    SIC: tl.constexpr,
    M,
    H: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BLK: tl.constexpr,
    SORTED_A: tl.constexpr,
    SKIP_ZERO: tl.constexpr = False,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.arange(0, BLK)
    mask = (rows[:, None] < M) & (cols[None, :] < H)
    idx_a = tl.load(index_a + rows * SIA, mask=rows < M, other=0).to(tl.int64)
    idx_b = tl.load(index_b + rows * SIB, mask=rows < M, other=0).to(tl.int64)
    idx_c = tl.load(index_c + rows * SIC, mask=rows < M, other=0).to(tl.int64)

    in_base = rows[:, None] * N
    out_base = rows[:, None] * H
    core = tl.load(preact + in_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(preact + in_base + H + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    gcore = tl.load(grad_core + out_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    ggate = tl.load(grad_gate + out_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )

    sig_core = tl.sigmoid(core)
    sig_gate = tl.sigmoid(gate)
    dcore = gcore * sig_core * (1.0 + core * (1.0 - sig_core))
    dgate = ggate * sig_gate * (1.0 + gate * (1.0 - sig_gate))

    tl.store(grad_p0 + in_base + cols[None, :], dcore, mask=mask)
    tl.store(grad_p0 + in_base + H + cols[None, :], dgate, mask=mask)
    # grad_p0 (empty_like) is stored for every row; SKIP_ZERO masks atomics only.
    core_mask = _atomic_rows(mask, dcore, SKIP_ZERO)
    gate_mask = _atomic_rows(mask, dgate, SKIP_ZERO)
    if SORTED_A:
        # Rows pair by global parity (2k, 2k + 1), so with an odd BM a pair
        # straddles two tiles: the even row adds both adjoints if the pair
        # shares index_a, the odd row adds its own only if it does not.
        even = (rows % 2) == 0
        next_rows = rows + 1
        next_valid = even & (next_rows < M)
        idx_next = tl.load(index_a + next_rows * SIA, mask=next_valid, other=-1)
        next_mask = next_valid[:, None] & (cols[None, :] < H)
        next_in_base = next_rows[:, None] * N
        next_out_base = next_rows[:, None] * H
        core_next = tl.load(
            preact + next_in_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        gate_next = tl.load(
            preact + next_in_base + H + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        gcore_next = tl.load(
            grad_core + next_out_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        ggate_next = tl.load(
            grad_gate + next_out_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        sig_core_next = tl.sigmoid(core_next)
        sig_gate_next = tl.sigmoid(gate_next)
        dcore_next = gcore_next * sig_core_next * (
            1.0 + core_next * (1.0 - sig_core_next)
        )
        dgate_next = ggate_next * sig_gate_next * (
            1.0 + gate_next * (1.0 - sig_gate_next)
        )
        same_next = next_valid & (idx_next == idx_a)
        pair_dcore = dcore + tl.where(same_next[:, None], dcore_next, 0.0)
        pair_dgate = dgate + tl.where(same_next[:, None], dgate_next, 0.0)
        even_mask = mask & even[:, None]
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            pair_dcore,
            mask=_atomic_rows(even_mask, pair_dcore, SKIP_ZERO),
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            pair_dgate,
            mask=_atomic_rows(even_mask, pair_dgate, SKIP_ZERO),
            sem="relaxed",
        )

        prev_rows = rows - 1
        odd = ~even
        # rows < M: the tail tile's odd rows past M must not read index_a.
        prev_valid = odd & (rows > 0) & (rows < M)
        idx_prev = tl.load(index_a + prev_rows * SIA, mask=prev_valid, other=-2)
        same_prev = prev_valid & (idx_prev == idx_a)
        odd_mask = mask & odd[:, None] & (~same_prev[:, None])
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            dcore,
            mask=_atomic_rows(odd_mask, dcore, SKIP_ZERO),
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            dgate,
            mask=_atomic_rows(odd_mask, dgate, SKIP_ZERO),
            sem="relaxed",
        )
    else:
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            dcore,
            mask=core_mask,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            dgate,
            mask=gate_mask,
            sem="relaxed",
        )
    tl.atomic_add(
        grad_p2 + idx_b[:, None] * N + cols[None, :], dcore, mask=core_mask, sem="relaxed"
    )
    tl.atomic_add(
        grad_p2 + idx_b[:, None] * N + H + cols[None, :],
        dgate,
        mask=gate_mask,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_p3 + idx_c[:, None] * N + cols[None, :], dcore, mask=core_mask, sem="relaxed"
    )
    tl.atomic_add(
        grad_p3 + idx_c[:, None] * N + H + cols[None, :],
        dgate,
        mask=gate_mask,
        sem="relaxed",
    )


def _launch_cat3_gather_add_silu(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    index_a: torch.Tensor,
    index_b: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = p0.shape
    h = n // 2
    # p0 may be a row-strided narrow view of a wider merged projection
    # (stride(1) must stay 1); preact is always contiguous.
    preact = torch.empty((m, n), device=p0.device, dtype=p0.dtype)
    core_act = torch.empty((m, h), device=p0.device, dtype=p0.dtype)
    gate_act = torch.empty((m, h), device=p0.device, dtype=p0.dtype)
    block = triton.next_power_of_2(h)
    bm = row_tile("indexed_silu", block, _EPILOGUE_BM)
    _cat3_gather_add_silu_fwd[(triton.cdiv(m, bm),)](
        p0,
        p1,
        p2,
        index_a,
        index_b,
        bias if bias is not None else p0,
        preact,
        core_act,
        gate_act,
        SIA=index_a.stride(0),
        SIB=index_b.stride(0),
        SP0=p0.stride(0),
        M=static_under_compile(m),
        H=h,
        N=n,
        BM=bm,
        BLK=block,
        HAS_BIAS=bias is not None,
        num_warps=4 if block >= 128 else 1,
    )
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("cat3_gather_add_silu", "triton", {
            "M": m, "H": h, "N": n, "BM": bm, "BLK": block,
            "num_warps": 4 if block >= 128 else 1, "HAS_BIAS": bias is not None,
            "SIA": index_a.stride(0), "SIB": index_b.stride(0), "SP0": p0.stride(0),
            "dtype": p0.dtype, "index_dtype": [index_a.dtype, index_b.dtype],
        })
    return preact, core_act, gate_act


def _launch_cat4_gather_add_silu(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    index_a: torch.Tensor,
    index_b: torch.Tensor,
    index_c: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = p0.shape
    h = n // 2
    # p0 may be a row-strided narrow view (stride(1)==1); preact stays
    # contiguous.
    preact = torch.empty((m, n), device=p0.device, dtype=p0.dtype)
    core_act = torch.empty((m, h), device=p0.device, dtype=p0.dtype)
    gate_act = torch.empty((m, h), device=p0.device, dtype=p0.dtype)
    block = triton.next_power_of_2(h)
    bm = row_tile("indexed_silu", block, _EPILOGUE_BM)
    _cat4_gather_add_silu_fwd[(triton.cdiv(m, bm),)](
        p0,
        p1,
        p2,
        p3,
        index_a,
        index_b,
        index_c,
        bias if bias is not None else p0,
        preact,
        core_act,
        gate_act,
        SIA=index_a.stride(0),
        SIB=index_b.stride(0),
        SIC=index_c.stride(0),
        SP0=p0.stride(0),
        M=static_under_compile(m),
        H=h,
        N=n,
        BM=bm,
        BLK=block,
        HAS_BIAS=bias is not None,
        num_warps=4 if block >= 128 else 1,
    )
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("cat4_gather_add_silu", "triton", {
            "M": m, "H": h, "N": n, "BM": bm, "BLK": block,
            "num_warps": 4 if block >= 128 else 1, "HAS_BIAS": bias is not None,
            "SIA": index_a.stride(0), "SIB": index_b.stride(0), "SIC": index_c.stride(0),
            "SP0": p0.stride(0), "dtype": p0.dtype,
            "index_dtype": [index_a.dtype, index_b.dtype, index_c.dtype],
        })
    return preact, core_act, gate_act


@triton.jit(do_not_specialize=["M"])
def _silu_halves_fwd(preact, core_act, gate_act, M, H: tl.constexpr, N: tl.constexpr,
                     BM: tl.constexpr, BLK: tl.constexpr):
    # The epilogue's activation (``x * tl.sigmoid(x)``) of a saved preact.
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.arange(0, BLK)
    mask = (rows[:, None] < M) & (cols[None, :] < H)
    core = tl.load(preact + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(preact + rows[:, None] * N + H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    out = rows[:, None] * H + cols[None, :]
    tl.store(core_act + out, core * tl.sigmoid(core), mask=mask)
    tl.store(gate_act + out, gate * tl.sigmoid(gate), mask=mask)


def _silu_halves(preact: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recipe of the epilogue outputs (saved-tensor recompute): bitwise the
    forward's core/gate activations, from the preact the VJP saves anyway."""
    m, n = preact.shape
    h = n // 2
    core_act = torch.empty((m, h), device=preact.device, dtype=preact.dtype)
    gate_act = torch.empty((m, h), device=preact.device, dtype=preact.dtype)
    block = triton.next_power_of_2(h)
    # Elementwise: a wide row tile (the epilogue's narrower tile, row_tiles, is for its gathers).
    rows = max(1, 4096 // block)
    _silu_halves_fwd[(triton.cdiv(m, rows),)](
        preact, core_act, gate_act, static_under_compile(m), H=h, N=n, BM=rows, BLK=block,
        num_warps=4,
    )
    return core_act, gate_act


def _cat3_preact(p0, p1, p2, index_a, index_b, bias):
    return _launch_cat3_gather_add_silu(p0, p1, p2, index_a, index_b, bias)[0]


def _register_activations(preact, core_act, gate_act, cat3=None) -> None:
    """Before the preact is saved. ``cat3``: the Cat3 epilogue's inputs
    (p0, p1, p2, index_a, index_b, bias), from which deep rebuilds the preact
    when p0 is rebuilt itself (re-launching the epilogue)."""
    scope = _saved_recompute.active()
    if scope is not None:
        if cat3 is not None and scope.mode == "deep":
            p0, p1, p2, index_a, index_b, bias = cat3
            source = scope.ref(p0)
            if source is not p0:
                scope.register((preact,), _cat3_preact, source, scope.ref(p1), scope.ref(p2), index_a, index_b, bias)
        scope.register((core_act, gate_act), _silu_halves, scope.ref(preact))


def _ordered_vjp(ctx, op, grad_core, grad_gate, preact, grad_p0, sums):
    """Deterministic mode: the epilogue VJP without atomics. ``grad_p0`` receives
    d(preact) = grad * silu'(preact); per ``(shape, index)`` of ``sums`` a zero
    gradient gets its ordered index_add_ of those rows."""
    h = preact.shape[1] // 2
    slope = torch.sigmoid(preact)
    slope.mul_((1.0 - slope).mul_(preact).add_(1.0))  # silu' = s (1 + x (1 - s))
    torch.mul(grad_core, slope[:, :h], out=grad_p0[:, :h])
    torch.mul(grad_gate, slope[:, h:], out=grad_p0[:, h:])
    grads = [ordered_index_add_(grad_p0.new_zeros(shape), index, grad_p0, skip_zero=ctx.skip_zero)
             for shape, index in sums]
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record(op, "ordered_index_add", {"M": preact.shape[0], "H": h, "N": preact.shape[1]},
                   phase="bwd", site=getattr(ctx, "_opsel_site", None))
    return (grad_p0, *grads)


class _Cat3GatherAddSilu(Function):
    @staticmethod
    def forward(
        ctx,
        p0: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        index_a: torch.Tensor,
        index_b: torch.Tensor,
        bias: torch.Tensor | None,
        index_a_sorted: bool,
    ):
        if not torch.compiler.is_compiling():
            ctx._grad_claim = grad_claim(p0)  # a merged projection's p0 (C3)
        preact, core_act, gate_act = _launch_cat3_gather_add_silu(
            p0, p1, p2, index_a, index_b, bias
        )
        _register_activations(preact, core_act, gate_act, (p0, p1, p2, index_a, index_b, bias))
        ctx.save_for_backward(preact, index_a, index_b)
        ctx.p1_shape = p1.shape
        ctx.p2_shape = p2.shape
        ctx.index_strides = (index_a.stride(0), index_b.stride(0))
        ctx.index_a_sorted = bool(index_a_sorted and _SORTED_A_PAIR_BWD)
        ctx.skip_zero = zero_skip_atomics()
        ctx.ordered = deterministic_mode()
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
        return core_act, gate_act

    @staticmethod
    def backward(ctx, grad_core: torch.Tensor, grad_gate: torch.Tensor):
        preact, index_a, index_b = ctx.saved_tensors
        m, n = preact.shape
        h = n // 2
        block = triton.next_power_of_2(h)
        grad_p0 = claim_grad(getattr(ctx, "_grad_claim", None), preact.shape, preact.device, preact.dtype)
        if ctx.ordered:
            return (*_ordered_vjp(ctx, "cat3_gather_add_silu", grad_core, grad_gate, preact, grad_p0,
                                  ((ctx.p1_shape, index_a), (ctx.p2_shape, index_b))), None, None, None, None)
        gp0_s0 = grad_p0.stride(0) if grad_p0.stride(0) != n else 0
        grad_p1 = torch.zeros(ctx.p1_shape, device=preact.device, dtype=preact.dtype)
        grad_p2 = torch.zeros(ctx.p2_shape, device=preact.device, dtype=preact.dtype)
        _cat3_gather_add_silu_bwd[(triton.cdiv(m, _EPILOGUE_BWD_BM),)](
            grad_core.contiguous(),
            grad_gate.contiguous(),
            preact,
            index_a,
            index_b,
            grad_p0,
            grad_p1,
            grad_p2,
            SIA=ctx.index_strides[0],
            SIB=ctx.index_strides[1],
            M=static_under_compile(m),
            H=h,
            N=n,
            BM=_EPILOGUE_BWD_BM,
            BLK=block,
            SORTED_A=ctx.index_a_sorted,
            SKIP_ZERO=ctx.skip_zero,
            GP0_S0=gp0_s0,
            num_warps=4 if block >= 128 else 1,
        )
        rec = _recorder._ACTIVE
        if rec is not None:
            params = {
                "M": m, "H": h, "N": n, "BM": _EPILOGUE_BWD_BM, "BLK": block,
                "num_warps": 4 if block >= 128 else 1, "SORTED_A": ctx.index_a_sorted,
                "SIA": ctx.index_strides[0], "SIB": ctx.index_strides[1],
                "contiguous_copy": [not grad_core.is_contiguous(), not grad_gate.is_contiguous()],
            }
            if ctx.skip_zero:  # new key only under the private sink: P=1 records as before
                params["SKIP_ZERO"] = True
            if gp0_s0:
                params["GP0_S0"] = gp0_s0
            rec.record("cat3_gather_add_silu", "triton", params,
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return grad_p0, grad_p1, grad_p2, None, None, None, None


class _Cat4GatherAddSilu(Function):
    @staticmethod
    def forward(
        ctx,
        p0: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        index_a: torch.Tensor,
        index_b: torch.Tensor,
        index_c: torch.Tensor,
        bias: torch.Tensor | None,
        index_a_sorted: bool,
    ):
        preact, core_act, gate_act = _launch_cat4_gather_add_silu(
            p0, p1, p2, p3, index_a, index_b, index_c, bias
        )
        ctx.save_for_backward(preact, index_a, index_b, index_c)
        _register_activations(preact, core_act, gate_act)
        ctx.p1_shape = p1.shape
        ctx.p2_shape = p2.shape
        ctx.p3_shape = p3.shape
        ctx.index_strides = (index_a.stride(0), index_b.stride(0), index_c.stride(0))
        ctx.index_a_sorted = bool(index_a_sorted and _SORTED_A_PAIR_BWD)
        ctx.skip_zero = zero_skip_atomics()
        ctx.ordered = deterministic_mode()
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
        return core_act, gate_act

    @staticmethod
    def backward(ctx, grad_core: torch.Tensor, grad_gate: torch.Tensor):
        preact, index_a, index_b, index_c = ctx.saved_tensors
        m, n = preact.shape
        h = n // 2
        block = triton.next_power_of_2(h)
        grad_p0 = torch.empty_like(preact)
        if ctx.ordered:
            return (*_ordered_vjp(ctx, "cat4_gather_add_silu", grad_core, grad_gate, preact, grad_p0, (
                (ctx.p1_shape, index_a), (ctx.p2_shape, index_b), (ctx.p3_shape, index_c))),
                None, None, None, None, None)
        grad_p1 = torch.zeros(ctx.p1_shape, device=preact.device, dtype=preact.dtype)
        grad_p2 = torch.zeros(ctx.p2_shape, device=preact.device, dtype=preact.dtype)
        grad_p3 = torch.zeros(ctx.p3_shape, device=preact.device, dtype=preact.dtype)
        _cat4_gather_add_silu_bwd[(triton.cdiv(m, _EPILOGUE_BWD_BM),)](
            grad_core.contiguous(),
            grad_gate.contiguous(),
            preact,
            index_a,
            index_b,
            index_c,
            grad_p0,
            grad_p1,
            grad_p2,
            grad_p3,
            SIA=ctx.index_strides[0],
            SIB=ctx.index_strides[1],
            SIC=ctx.index_strides[2],
            M=static_under_compile(m),
            H=h,
            N=n,
            BM=_EPILOGUE_BWD_BM,
            BLK=block,
            SORTED_A=ctx.index_a_sorted,
            SKIP_ZERO=ctx.skip_zero,
            num_warps=4 if block >= 128 else 1,
        )
        rec = _recorder._ACTIVE
        if rec is not None:
            params = {
                "M": m, "H": h, "N": n, "BM": _EPILOGUE_BWD_BM, "BLK": block,
                "num_warps": 4 if block >= 128 else 1, "SORTED_A": ctx.index_a_sorted,
                "SIA": ctx.index_strides[0], "SIB": ctx.index_strides[1],
                "SIC": ctx.index_strides[2],
                "contiguous_copy": [not grad_core.is_contiguous(), not grad_gate.is_contiguous()],
            }
            if ctx.skip_zero:  # new key only under the private sink: P=1 records as before
                params["SKIP_ZERO"] = True
            rec.record("cat4_gather_add_silu", "triton", params,
                       phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return grad_p0, grad_p1, grad_p2, grad_p3, None, None, None, None, None


def _record_decline(rec, op, reason, aligned) -> None:
    rec.fallback(op, "decomposed_indexed_silu", reason,
                 {"rows": aligned.shape[0], "width": aligned.shape[-1]})


def _record_selection(rec, op, aligned, gathered, indices, weights, bias,
                      index_a_sorted, precomputed_p0, p0, strict_errors) -> None:
    rec.record(op, "decomposed_indexed_silu", {
        "rows": aligned.shape[0], "width": aligned.shape[1],
        "projection_width": weights[0].shape[0],
        "gathered_rows": [value.shape[0] for value in gathered],
        "gemm_backend": _GEMM_BACKEND, "precomputed_p0": precomputed_p0 is not None,
        "p0_stride0": p0.stride(0), "index_a_sorted": bool(index_a_sorted),
        "SORTED_A": bool(index_a_sorted and _SORTED_A_PAIR_BWD), "HAS_BIAS": bias is not None,
        "index_dtype": [index.dtype for index in indices], "dtype": aligned.dtype,
        "strict_errors": bool(strict_errors),
    })


def decomposed_indexed_cat3_silu_linear(
    aligned: torch.Tensor,
    gathered: torch.Tensor,
    index_a: torch.Tensor,
    index_b: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bias: torch.Tensor | None = None,
    index_a_sorted: bool = False,
    precomputed_p0: torch.Tensor | None = None,
    *,
    strict_errors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _LAST_ERROR
    if precomputed_p0 is not None:
        # aligned is consumed only by the p0 projection; a caller that merged
        # that projection into a wider GEMM supplies p0 directly (possibly as
        # a row-strided narrow view: stride(1) must be 1).
        if not (
            precomputed_p0.is_cuda
            and precomputed_p0.stride(1) == 1
            and precomputed_p0.dim() == 2
            and precomputed_p0.shape[0] == aligned.shape[0]
            and precomputed_p0.shape[1] == weights[0].shape[0]
            and precomputed_p0.dtype == aligned.dtype
        ):
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_decline(rec, "indexed_cat3", "precomputed_p0_guard", aligned)
            return None
    elif not (aligned.is_cuda and aligned.is_contiguous()):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_decline(rec, "indexed_cat3", "aligned_guard", aligned)
        return None
    if not (
        gathered.is_cuda
        and gathered.is_contiguous()
        and len(weights) == 3
        and all(w.is_cuda and w.is_contiguous() and w.dtype == aligned.dtype for w in weights)
    ):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_decline(rec, "indexed_cat3", "gathered_or_weight_guard", aligned)
        return None
    if bias is not None and (
        not bias.is_cuda
        or not bias.is_contiguous()
        or bias.dtype != aligned.dtype
        or bias.dim() != 1
    ):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_decline(rec, "indexed_cat3", "bias_guard", aligned)
        return None
    try:
        _debug_check_index("cat3 index_a", index_a, gathered.shape[0])
        _debug_check_index("cat3 index_b", index_b, gathered.shape[0])
        _debug_print_once(
            "cat3",
            "aligned",
            tuple(aligned.shape),
            aligned.stride(),
            "gathered",
            tuple(gathered.shape),
            gathered.stride(),
            "weights",
            [tuple(w.shape) for w in weights],
            "bias",
            None if bias is None else tuple(bias.shape),
        )
        p0 = (
            precomputed_p0
            if precomputed_p0 is not None
            else _project(aligned, weights[0])
        )
        p1 = _project(gathered, weights[1])
        p2 = _project(gathered, weights[2])
        # Not active(): under torch.compile this function runs eagerly, where a
        # new call would be a new dynamo frame (a scope is never open there).
        scope = _saved_recompute._ACTIVE
        if scope is not None and scope.mode == "deep":
            # deep: the preact recipe holds these node projections weakly
            # (rebuilt from the narrower gathered rows), not [nodes, 2H] each,
            # with the forward's GEMM (C4 row GEMM precision, else the wide GEMM's or F.linear).
            from fastmd.engine.kernels import triton_wide_gemm as _wide
            from fastmd.engine.kernels.merged_frozen_projections import project, row_gemm_precision
            for p, weight in ((p1, weights[1]), (p2, weights[2])):
                precision = row_gemm_precision(gathered, weight)
                wide = None if precision is not None else _wide.linear_precision(gathered, weight)
                scope.register((p,), project, gathered, weight, precision, wide)
        result = _Cat3GatherAddSilu.apply(
            p0, p1, p2, index_a, index_b, bias, bool(index_a_sorted)
        )
    except Exception as exc:
        _LAST_ERROR = repr(exc)
        if strict_errors or _env_flag("MATRIS_DECOMPOSED_INDEXED_DEBUG_CHECKS", False):
            raise
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_decline(rec, "indexed_cat3", "exception:" + type(exc).__name__, aligned)
        return None
    # Recorded outside the try: a recorder error must not turn into a fallback.
    rec = _recorder._ACTIVE
    if rec is not None:
        _record_selection(rec, "indexed_cat3", aligned, (gathered,), (index_a, index_b),
                          weights, bias, index_a_sorted, precomputed_p0, p0,
                          strict_errors)
    return result


def decomposed_indexed_cat4_silu_linear(
    aligned: torch.Tensor,
    gathered_a: torch.Tensor,
    index_a: torch.Tensor,
    gathered_b: torch.Tensor,
    index_b: torch.Tensor,
    index_c: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    bias: torch.Tensor | None = None,
    index_a_sorted: bool = False,
    precomputed_p0: torch.Tensor | None = None,
    *,
    strict_errors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _LAST_ERROR
    if precomputed_p0 is not None:
        if not (
            precomputed_p0.is_cuda
            and precomputed_p0.stride(1) == 1
            and precomputed_p0.dim() == 2
            and precomputed_p0.shape[0] == aligned.shape[0]
            and precomputed_p0.shape[1] == weights[0].shape[0]
            and precomputed_p0.dtype == aligned.dtype
        ):
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_decline(rec, "indexed_cat4", "precomputed_p0_guard", aligned)
            return None
    elif not (aligned.is_cuda and aligned.is_contiguous()):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_decline(rec, "indexed_cat4", "aligned_guard", aligned)
        return None
    if not (
        gathered_a.is_cuda
        and gathered_b.is_cuda
        and gathered_a.is_contiguous()
        and gathered_b.is_contiguous()
        and len(weights) == 4
        and all(w.is_cuda and w.is_contiguous() and w.dtype == aligned.dtype for w in weights)
    ):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_decline(rec, "indexed_cat4", "gathered_or_weight_guard", aligned)
        return None
    if bias is not None and (
        not bias.is_cuda
        or not bias.is_contiguous()
        or bias.dtype != aligned.dtype
        or bias.dim() != 1
    ):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_decline(rec, "indexed_cat4", "bias_guard", aligned)
        return None
    try:
        _debug_check_index("cat4 index_a", index_a, gathered_a.shape[0])
        _debug_check_index("cat4 index_b", index_b, gathered_b.shape[0])
        _debug_check_index("cat4 index_c", index_c, gathered_b.shape[0])
        _debug_print_once(
            "cat4",
            "aligned",
            tuple(aligned.shape),
            aligned.stride(),
            "gathered_a",
            tuple(gathered_a.shape),
            gathered_a.stride(),
            "gathered_b",
            tuple(gathered_b.shape),
            gathered_b.stride(),
            "weights",
            [tuple(w.shape) for w in weights],
            "bias",
            None if bias is None else tuple(bias.shape),
        )
        p0 = (
            precomputed_p0
            if precomputed_p0 is not None
            else _project(aligned, weights[0])
        )
        p1 = _project(gathered_a, weights[1])
        p2 = _project(gathered_b, weights[2])
        p3 = _project(gathered_b, weights[3])
        result = _Cat4GatherAddSilu.apply(
            p0,
            p1,
            p2,
            p3,
            index_a,
            index_b,
            index_c,
            bias,
            bool(index_a_sorted),
        )
    except Exception as exc:
        _LAST_ERROR = repr(exc)
        if strict_errors or _env_flag("MATRIS_DECOMPOSED_INDEXED_DEBUG_CHECKS", False):
            raise
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_decline(rec, "indexed_cat4", "exception:" + type(exc).__name__, aligned)
        return None
    # Recorded outside the try: a recorder error must not turn into a fallback.
    rec = _recorder._ACTIVE
    if rec is not None:
        _record_selection(rec, "indexed_cat4", aligned, (gathered_a, gathered_b),
                          (index_a, index_b, index_c), weights, bias, index_a_sorted,
                          precomputed_p0, p0, strict_errors)
    return result


def last_decomposed_indexed_cat_silu_linear_error() -> str | None:
    return _LAST_ERROR
