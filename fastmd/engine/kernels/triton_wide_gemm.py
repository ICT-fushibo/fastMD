"""3xTF32 GEMMs for the wide frozen-weight projections of the 128-wide models.

``y = x @ W^T (+ b)`` and ``dX = dY @ W (+ c)`` with ``K >= 64`` and ``N >= 64`` and a frozen ``W``: the 128-wide models
(4M, 10M) spend about two thirds of their GPU time in these GEMMs, which cuBLAS runs as IEEE FP32 SIMT kernels. Here
they run on TF32 tensor cores at FP32 accuracy (3xTF32):

* ``A = A_hi + A_lo`` and ``B = B_hi + B_lo`` with ``*_hi`` rounded to nearest TF32; ``A B ~ A_lo B_hi + A_hi B_lo +
  A_hi B_hi`` (the ``A_lo B_lo`` term, below FP32 rounding, is dropped), the small cross terms first.
* The tensor-core accumulator is not IEEE FP32: chained over the whole K, the truncation of its additions biases the
  result in proportion to K. Each K chunk (BK = 64) therefore starts a fresh accumulator, and the chunks are summed
  in FP32 registers (round to nearest): the error is flat in K and below cuBLAS FP32's (rms 1.4-2e-7 vs 2-4e-7).
* ``B`` is split once per weight version (``_split``): ``W`` for the forward and ``W^T`` for the input
  gradient, each as a K-major ``hi``/``lo`` pair (TMA + tcgen05 want K-major FP32 operands). The pairs are derived
  caches of the weight (see ``_split``: weak per-tensor entries keyed on the view and the weight's version, retained
  by a CUDA graph capture that reads them, like the projection caches of ``fastmd.engine.cache_provenance``).
* Only ``A`` is split in the kernel. Persistent programs walk the output tiles (128 x 128, N fastest) for K <= 256,
  one program per tile for K >= 384 (more K chunks to overlap). TMA tile loads; the M tail is zero-filled on load
  and clipped on store. The rows are a runtime argument (no re-JIT per row count); a row's result depends on its own
  inputs only (not on the row count, its tile or the grid), so replicated bond rows on two ranks agree.

``triton_dot_precision`` "tf32" runs the same kernels with one TF32 dot of the rounded operands (where cuBLAS runs
IEEE FP32; under TF32 cuBLAS, cuBLAS is faster). The callers (``linear``, ``matmul``) take cuBLAS where this module
declines (``wide_gemm_decline``).
"""
from __future__ import annotations

from contextlib import contextmanager
import weakref

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function
from torch.utils.weak import WeakIdKeyDictionary

from fastmd.config import env_value
from fastmd.engine import recorder as _recorder
from fastmd.precision import triton_dot_input_precision

from fastmd.engine.cache_provenance import CAPTURE as _CAPTURE
from fastmd.engine.cache_provenance import retain_under_capture

# torch.cuda.is_current_stream_capturing without its Python frame (the eager dispatch's host cost).
_capturing = getattr(torch._C, "_cuda_isCurrentStreamCapturing", None) or torch.cuda.is_current_stream_capturing

# Bound by ResolvedConfig.scope (fastmd.config): the switch (topology profile) and the row threshold (0 in the
# P>1 scope, where replicated bond rows must take the same kernel on every rank; see ``rank_uniform``).
_TRITON_WIDE_GEMM = env_value("MATRIS_TRITON_WIDE_GEMM") == "1"
_MIN_ROWS = int(env_value("MATRIS_TRITON_WIDE_GEMM_MIN_ROWS"))
# Inside ``rank_uniform`` (a P>1 forward on the topology epoch's layout): its (captured, eager) row thresholds.
_BAND: tuple[int, int] | None = None
# > 0 inside a torch.compile'd block (``compiled_region``): Inductor owns its GEMMs, also in frames that run eagerly.
_COMPILED_REGION = 0
# Eager (outside a CUDA graph capture) the threshold is this many times higher: the Python launch path (~30 us per
# call, cuBLAS ~10 us) cost more than the GEMMs saved on CPU-bound small systems (GB200, 10M Si 1,000 eager +8%
# and 4M box60 +7% at 240k / 152k triplet rows with the captured threshold; from 935k rows -32..-40%).
_EAGER_FACTOR = 4
_BK = 64


@triton.jit
def _rna_tf32(x):
    # round to nearest (ties away) to TF32: 10 explicit mantissa bits, the low 13 bits zero
    return tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _k_chunk(a, b_hi, b_lo, PREC: tl.constexpr):
    # One K chunk on a fresh accumulator: 3xTF32 (cross terms first) or one TF32 dot of the rounded operands.
    a_hi = _rna_tf32(a)
    if PREC == "tf32x3":
        a_lo = a - a_hi
        part = tl.dot(a_lo, b_hi, input_precision="tf32")
        part = tl.dot(a_hi, b_lo, part, input_precision="tf32")
        part = tl.dot(a_hi, b_hi, part, input_precision="tf32")
    else:
        part = tl.dot(a_hi, b_hi, input_precision="tf32")
    return part


@triton.jit
def _tile(a_desc, bh_desc, bl_desc, c_ptr, bias_ptr, M, off_m, off_n, K: tl.constexpr, N: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr, HAS_BIAS: tl.constexpr,
          HAS_C: tl.constexpr):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = a_desc.load([off_m, k0])
        b_hi = bh_desc.load([off_n, k0]).T
        if PREC == "tf32x3":
            b_lo = bl_desc.load([off_n, k0]).T
        else:
            b_lo = b_hi
        acc = acc + _k_chunk(a, b_hi, b_lo, PREC)
    if HAS_BIAS:
        acc = acc + tl.load(bias_ptr + off_n + tl.arange(0, BN))[None, :]
    if HAS_C:  # c [M, N] contiguous (a plain pointer: one TMA descriptor less to encode per launch)
        rows = (off_m // BM).to(tl.int64) * BM + tl.arange(0, BM)
        acc = acc + tl.load(c_ptr + rows[:, None] * N + (off_n + tl.arange(0, BN))[None, :], mask=(rows < M)[:, None],
                            other=0.0)
    return acc


@triton.jit(do_not_specialize=["M"])
def _wide_gemm_persistent(a_desc, bh_desc, bl_desc, c_ptr, y_desc, bias_ptr, M, K: tl.constexpr, NT_N: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr,
                          HAS_BIAS: tl.constexpr, HAS_C: tl.constexpr):
    # Programs walk the output tiles (N fastest, so the column tiles of a row block run together).
    pid = tl.program_id(0)  # int32 block index
    tiles = tl.cdiv(M, BM) * NT_N
    for t in tl.range(pid, tiles, tl.num_programs(0), flatten=True):
        off_m = (t // NT_N) * BM
        off_n = (t % NT_N) * BN
        acc = _tile(a_desc, bh_desc, bl_desc, c_ptr, bias_ptr, M, off_m, off_n, K, NT_N * BN, BM, BN, BK, PREC,
                    HAS_BIAS, HAS_C)
        y_desc.store([off_m, off_n], acc)


@triton.jit(do_not_specialize=["M"])
def _wide_gemm_tiles(a_desc, bh_desc, bl_desc, c_ptr, y_desc, bias_ptr, M, K: tl.constexpr, NT_N: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr,
                     HAS_BIAS: tl.constexpr, HAS_C: tl.constexpr):
    pid = tl.program_id(0)  # int32 block index
    off_m = (pid // NT_N) * BM
    off_n = (pid % NT_N) * BN
    acc = _tile(a_desc, bh_desc, bl_desc, c_ptr, bias_ptr, M, off_m, off_n, K, NT_N * BN, BM, BN, BK, PREC,
                HAS_BIAS, HAS_C)
    y_desc.store([off_m, off_n], acc)


@triton.jit
def _split_kernel(src, hi, lo, total, COLS: tl.constexpr, S_ROW: tl.constexpr, S_COL: tl.constexpr,
                  BLOCK: tl.constexpr):
    # hi/lo [ROWS, COLS] contiguous from src[r, c] = src + r * S_ROW + c * S_COL: hi = rna_tf32(src), lo = src - hi.
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = o < total
    x = tl.load(src + (o // COLS) * S_ROW + (o % COLS) * S_COL, mask=live)
    h = _rna_tf32(x)
    tl.store(hi + o, h, mask=live)
    tl.store(lo + o, x - h, mask=live)


class _Split:
    """A weight's K-major (hi, lo) and their TMA descriptors (per block shape). ``storage`` keeps the weight
    storage the split was taken from alive while the entry exists, so its address cannot be reused by another
    tensor: (version, address) then identify the weight's state."""
    __slots__ = ("version", "pointer", "storage", "hi", "lo", "descriptors")

    def __init__(self, w, hi, lo):
        self.version, self.pointer, self.storage = w._version, w.data_ptr(), w.untyped_storage()
        self.hi, self.lo, self.descriptors = hi, lo, {}


# Per weight tensor (the view's base, weakly): {(view geometry, transposed): _Split}. A backward sees the weight
# through autograd's saved tensors, which (also through saved-tensor hooks that detach) may be another tensor object
# on the same storage: ``_ALIASES`` maps a storage (device, address) to the base it was first seen with (weakly; the
# address cannot be reused while that base is alive), so those aliases find the same entries.
_SPLITS = WeakIdKeyDictionary()
_ALIASES: dict = {}


def _owner(w: torch.Tensor) -> torch.Tensor:
    """The tensor whose entries in ``_SPLITS`` hold the splits of ``w``: its base, or an earlier base on the same
    storage (a saved-tensor alias)."""
    base = w if w._base is None else w._base
    if base in _SPLITS:
        return base
    key = (w.device.index, w.untyped_storage().data_ptr())
    known = _ALIASES.get(key)
    owner = None if known is None else known()
    if owner is not None and owner.untyped_storage().data_ptr() == key[1] and owner.device == w.device:
        return owner
    _ALIASES[key] = weakref.ref(base, lambda ref, key=key: _ALIASES.pop(key) if _ALIASES.get(key) is ref else None)
    return base


_SMS: dict[int, int] = {}


def _programs(device: torch.device, tiles: int) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    sms = _SMS.get(index)
    if sms is None:
        sms = _SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return max(1, min(tiles, sms))


def _split(w: torch.Tensor, transposed: bool, build: bool = True) -> _Split | None:
    """The K-major split of the GEMM operand ``B`` that ``w`` stands for: ``B = w^T`` (``x @ w^T``, ``transposed``
    False: K-major is ``w`` itself) or ``B = w`` (``g @ w``, ``transposed`` True: K-major is ``w^T``). A derived
    cache of ``w`` (frozen weights): rebuilt when its version or storage changes (``build`` False: None then),
    retained by a CUDA graph capture that reads it (``cache_provenance``). Never built under a capture: the
    callers decline a cold split there (the split kernel would only be recorded)."""
    base = _owner(w)
    key = (tuple(w.shape), tuple(w.stride()), w.storage_offset(), transposed)
    entries = _SPLITS.get(base)
    if entries is None:
        entries = _SPLITS[base] = {}
    split = entries.get(key)
    if split is None or split.version != w._version or split.pointer != w.data_ptr():
        if not build:
            return None
        src = w.t() if transposed else w
        rows, cols = src.shape
        hi = torch.empty((rows, cols), device=w.device, dtype=torch.float32)
        lo = torch.empty_like(hi)
        total = rows * cols
        _split_kernel[(triton.cdiv(total, 1024),)](src, hi, lo, total, COLS=cols, S_ROW=src.stride(0),
                                                    S_COL=src.stride(1), BLOCK=1024, num_warps=4)
        split = entries[key] = _Split(w, hi, lo)
    if _CAPTURE.holder is not None:  # a capture keeps what it reads
        retain_under_capture((split.hi, split.lo), (base,))
    return split


def _b_descriptors(split: _Split, bn: int, both: bool):
    """The TMA descriptors of a split's (hi, lo), created once per block shape (kept in the split)."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    pair = split.descriptors.get((bn, both))
    if pair is None:
        bh = TensorDescriptor.from_tensor(split.hi, [bn, _BK])
        pair = split.descriptors[(bn, both)] = (bh, TensorDescriptor.from_tensor(split.lo, [bn, _BK]) if both else bh)
    return pair


def wide_gemm_decline(a: torch.Tensor, w: torch.Tensor, transposed: bool, bias=None, c=None, out=None) -> str | None:
    """None if ``a @ B`` (``B = w^T``, or ``w`` when ``transposed``) runs here, else why not. Callers test the
    switch, torch.compile and the row threshold first."""
    k = a.shape[-1] if a.dim() == 2 else -1
    n = w.shape[1] if transposed else w.shape[0]
    if a.dim() != 2 or w.dim() != 2 or (w.shape[0] if transposed else w.shape[1]) != k:
        return "shape"
    # N: 128-wide tiles, or one power-of-two tile below 128 (Triton block shapes).
    if k < 64 or n < 64 or k % 16 or (n % 128 if n > 128 else n & (n - 1)):
        return "unsupported_shape"
    if not (a.is_cuda and w.device == a.device) or a.device.index != torch.cuda.current_device():
        return "device"
    if a.dtype != torch.float32 or w.dtype != torch.float32 or torch.is_autocast_enabled("cuda"):
        return "dtype"
    if w.requires_grad or (bias is not None and bias.requires_grad):
        return "trainable_params"
    # 16-byte aligned operands: the TMA descriptors, and the kernels are specialized on aligned pointers.
    if (a.stride(1) != 1 or a.stride(0) % 4 or a.stride(0) < k or a.data_ptr() % 16
            or (c is not None and (not c.is_contiguous() or c.shape != (a.shape[0], n) or c.data_ptr() % 16))
            or (out is not None and (not out.is_contiguous() or out.data_ptr() % 16))):
        return "layout"
    if bias is not None and (bias.dtype != torch.float32 or not bias.is_contiguous() or bias.data_ptr() % 16):
        return "layout"
    return None


# Compiled kernels by (kernel, constexprs, warps, stages, device): after the first launch of a configuration the
# launch skips Triton's argument binding and cache lookup and calls the compiled kernel directly (~30 us host cost
# per call instead of ~41 us). M is the only runtime scalar and is excluded from value specialization
# (do_not_specialize); the pointer arguments (bias, c) are specialized on 16-byte alignment, which
# ``wide_gemm_decline`` requires of every call, and the tensor descriptors are not specialized on, so the compiled
# kernel is valid for every call of its key.
_COMPILED: dict = {}


def _run(fn, grid0, args, consts, num_warps, num_stages):
    key = (fn, consts, num_warps, num_stages, args[0].base.device.index)
    kernel = _COMPILED.get(key)
    hooks = triton.knobs.runtime
    if kernel is None or hooks.launch_enter_hook.calls or hooks.launch_exit_hook.calls:
        # First launch (compiles), or launch hooks installed (e.g. a profiler): Triton's own path.
        kernel = fn[(grid0,)](*args, **dict(consts), num_warps=num_warps, num_stages=num_stages)
        _COMPILED[key] = kernel
        return
    kernel.run(grid0, 1, 1, torch._C._cuda_getCurrentRawStream(key[-1]), kernel.function,
               kernel.packed_metadata, None, None, None, *args, *(value for _, value in consts))


def _launch(a, split, prec, bias=None, c=None, out=None):
    from triton.tools.tensor_descriptor import TensorDescriptor

    m, k = a.shape
    n = split.hi.shape[0]
    y = torch.empty((m, n), device=a.device, dtype=torch.float32) if out is None else out
    if m == 0:
        return y
    bn = min(128, n)
    bm = 128
    nt_n = n // bn
    a_d = TensorDescriptor.from_tensor(a, [bm, _BK])
    bh_d, bl_d = _b_descriptors(split, bn, prec == "tf32x3")
    y_d = TensorDescriptor.from_tensor(y, [bm, bn])
    bias_arg = bias if bias is not None else split.hi
    tiles = triton.cdiv(m, bm) * nt_n
    consts = (("K", k), ("NT_N", nt_n), ("BM", bm), ("BN", bn), ("BK", _BK), ("PREC", prec),
              ("HAS_BIAS", bias is not None), ("HAS_C", c is not None))
    args = (a_d, bh_d, bl_d, c if c is not None else y, y_d, bias_arg, m)
    if k >= 384:
        _run(_wide_gemm_tiles, tiles, args, consts, 8, 3)
    else:
        _run(_wide_gemm_persistent, _programs(a.device, tiles), args, consts, 8, 2)
    return y


def _record(name, a, n, prec, phase="fwd", site=None, **extra):
    rec = _recorder._ACTIVE
    if rec is not None:
        params = {"rows": a.shape[0], "K": a.shape[1], "N": n, "PRECISION": prec, **extra}
        if phase == "fwd":
            rec.record("wide_gemm", name, params)
        else:
            rec.record("wide_gemm", name, params, phase=phase, site=site)


def _warm(w: torch.Tensor, orientations) -> bool:
    """Whether the splits of ``w`` in every orientation of ``orientations`` are built for its current state."""
    entries = _SPLITS.get(_owner(w))
    if entries is None:
        return not orientations
    for transposed in orientations:
        split = entries.get((tuple(w.shape), tuple(w.stride()), w.storage_offset(), transposed))
        if split is None or split.version != w._version or split.pointer != w.data_ptr():
            return False
    return True


# Eager-band memo: (address, shape, strides, orientations) of a weight -> its version when its splits were last found
# built, so that later eager calls skip ``_warm``'s lookups (CPU-bound sizes). A stale hit (a new tensor at the same
# address, shape and version) at worst leaves a split cold, which a later capture declines; it never selects a kernel.
_WARMED: dict = {}


def _warmed(w: torch.Tensor, orientations, parts) -> None:
    if len(_WARMED) > 4096:
        _WARMED.clear()
    _WARMED[(w.data_ptr(), w.shape, w.stride(), orientations, parts)] = w._version


def _thresholds() -> tuple[int, int]:
    """The (captured, eager) row thresholds here."""
    return _BAND if _BAND is not None else (_MIN_ROWS, _EAGER_FACTOR * _MIN_ROWS)


def _decide(a, w, transposed, bias=None, c=None, out=None, rows=None, warm=(), record=True, parts=(),
            band=None) -> str | None:
    """The Triton GEMM's dot precision for ``a @ B`` (``B = w^T``, or ``w`` when ``transposed``), or None: the
    caller's cuBLAS path (declines recorded unless ``record`` is False). ``rows``: the row count that decides
    against the threshold (a row chunk's whole GEMM). Eager below the eager threshold, cuBLAS, but the splits of
    the orientations in ``warm`` are built (once per weight version), so that a CUDA graph capture of the same
    GEMM later finds them; under a capture a cold split declines. ``parts``: (start, rows) row ranges of ``w`` (column
    ranges of the product) that a recompute recipe rebuilds alone with the bound choice: each a supported width,
    their splits warmed and retained like ``w``'s (the recipe may run inside a capture). ``band``: the
    (captured, eager) thresholds (a Function's forward state, ``dispatch_on``), default ``_thresholds()``."""
    rows = a.shape[0] if rows is None else rows
    if band is None:
        band = _BAND if _BAND is not None else (_MIN_ROWS, _EAGER_FACTOR * _MIN_ROWS)
    captured, eager = band
    if rows < captured or not a.is_cuda:
        return None
    shape = w.shape
    k, n = a.shape[-1], (shape[-1] if transposed else shape[0])
    if len(shape) != 2 or k < 64 or n < 64 or k % 16 or (n % 128 if n > 128 else n & (n - 1)):
        return None  # a width without a kernel (not recorded), before the other checks (CPU-bound sizes)
    if rows < eager and not _capturing():
        # Eager below its threshold, the splits already built: no further checks (CPU-bound sizes).
        if _WARMED.get((w.data_ptr(), w.shape, w.stride(), warm, parts)) == w._version:
            return None
        if _warm(w, warm) and all(_warm(w.narrow(0, start, size), (False,)) for start, size in parts):
            _warmed(w, warm, parts)
            return None
    reason = wide_gemm_decline(a, w, transposed, bias, c, out)
    if reason is None and any(n < 64 or (n % 128 if n > 128 else n & (n - 1)) for _, n in parts):
        reason = "unsupported_shape"
    prec = None
    if reason is None:
        prec = triton_dot_input_precision("wide_gemm")
        if prec is None:
            reason = "cublas_precision"
    if reason is None:
        if _capturing():
            if _split(w, transposed, build=False) is None or any(
                    _split(w.narrow(0, start, size), False, build=False) is None for start, size in parts):
                reason = "cold_split_under_capture"
        else:
            if rows < eager:
                for orientation in warm:
                    _split(w, orientation)
            for start, size in parts:  # also above the eager threshold: a later capture requires them
                _split(w.narrow(0, start, size), False)
            if rows < eager:
                _warmed(w, warm, parts)
                return None
    if reason is None:
        return prec
    rec = _recorder._ACTIVE
    if record and rec is not None and reason not in ("shape", "unsupported_shape"):
        rec.fallback("wide_gemm", "triton", reason, {"rows": a.shape[0], "width": a.shape[-1]})
    return None


class _WideLinear(Function):
    """``x @ W^T (+ b)`` with the 3xTF32 GEMM forward and input gradient (frozen ``W`` and ``b``)."""

    @staticmethod
    def forward(ctx, x, weight, bias, prec):
        y = _launch(x, _split(weight, False), prec, bias)
        ctx.save_for_backward(weight)
        ctx.prec = prec
        if _CAPTURE.holder is not None:
            # The backward runs on autograd's device thread, outside this thread's capture retention: retain
            # the input gradient's split here (a cold one is declined there).
            _split(weight, True, build=False)
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
        _record("triton", x, weight.shape[0], prec, bias=bias is not None)
        return y

    @staticmethod
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        if torch.is_grad_enabled():  # a graph-building backward (create_graph): torch's formula
            return grad @ weight, None, None, None
        split = None
        if wide_gemm_decline(grad, weight, True) is None:
            split = _split(weight, True, build=not torch.cuda.is_current_stream_capturing())
        if split is None:
            return torch.mm(grad, weight), None, None, None
        _record("triton", grad, weight.shape[1], ctx.prec, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        return _launch(grad, split, ctx.prec), None, None, None


def _enabled() -> bool:
    return _TRITON_WIDE_GEMM and not _COMPILED_REGION


def dispatch_on() -> tuple[int, int] | None:
    """The wide GEMM dispatch state here, for a Function's backward (``matmul(..., enabled=...)``): its row
    thresholds (``_thresholds``) when on, else None."""
    return _thresholds() if not torch.compiler.is_compiling() and _enabled() else None


def retain_transposed(*weights: torch.Tensor, ranges=None) -> None:
    """In a Function's forward under a CUDA graph capture: retain the transposed splits its backward's ``matmul``
    takes (the backward runs on autograd's device thread, outside this thread's capture retention; a cold split is
    declined there). ``ranges``: (start, rows) row ranges of the one weight the backward takes instead."""
    if _CAPTURE.holder is not None:
        if ranges is not None:
            (weight,) = weights
            weights = [weight.narrow(0, start, rows) for start, rows in ranges]
        for weight in weights:
            _split(weight, True, build=False)


@contextmanager
def rank_uniform(node_rows: tuple[int, int], min_rows: int):
    """A P>1 forward on the topology epoch's padded layout: the configured row threshold ``min_rows`` (which the
    P>1 scope sets to 0) and its eager multiple apply again, each moved above ``node_rows`` = the (lowest, highest)
    local atom rows (owned + ghost blocks) over the ranks when it falls inside that range. Every other row count of the layout (owned atoms,
    bonds, incidences, triplets: the agreed capacities) is the same on every rank, so every GEMM takes the same
    kernel on every rank (bitwise boundary replicas) and the small atom-row GEMMs keep cuBLAS."""
    global _BAND
    low, high = node_rows
    saved = _BAND
    _BAND = tuple(high + 1 if low < t <= high else t for t in (min_rows, _EAGER_FACTOR * min_rows))
    try:
        yield
    finally:
        _BAND = saved


DECIDE = "decide"


def linear_precision(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                     parts=()) -> str | None:
    """What ``linear(x, weight, bias)`` runs now: the Triton GEMM's precision, or None (``F.linear``). A recompute
    recipe binds it (``linear(..., precision=...)``), so the rebuild takes the forward's GEMM. ``parts``: as
    ``_decide``'s (a recipe of column ranges of a merged product)."""
    if torch.compiler.is_compiling() or not _enabled() or x.dim() != 2:
        return None
    grad = torch.is_grad_enabled() and x.requires_grad
    return _decide(x, weight, False, bias, record=False, warm=(False, True) if grad else (False,), parts=parts)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
           precision: str | None = DECIDE, parts=()) -> torch.Tensor:
    """``F.linear(x, weight, bias)`` of a frozen layer: the 3xTF32 GEMM (forward and input gradient) where it
    applies, else ``F.linear``. ``x`` 2-D (other ranks take ``F.linear``). ``precision``: a recompute recipe's
    binding of the forward's choice (``linear_precision``; None: ``F.linear``); omitted decides here. ``parts``: as
    ``_decide``'s."""
    if torch.compiler.is_compiling() or not _enabled():
        return F.linear(x, weight, bias)
    return _linear(x, weight, bias, precision, parts)


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, precision=DECIDE,
            parts=()) -> torch.Tensor:
    """``linear`` once the dispatch is known to be on and not compiling (``functions._apply_layer``)."""
    if x.dim() != 2 or (precision is DECIDE and x.shape[0] < (_MIN_ROWS if _BAND is None else _BAND[0])):
        return F.linear(x, weight, bias)
    grad = torch.is_grad_enabled() and x.requires_grad
    if precision is DECIDE:
        prec = _decide(x, weight, False, bias, warm=(False, True) if grad else (False,), parts=parts)
    else:
        prec = precision
        if prec is not None:
            if wide_gemm_decline(x, weight, False, bias) == "layout":
                x = x.contiguous()  # a rebuilt input in another layout: a row's result does not depend on it
            if _capturing() and _split(weight, False, build=False) is None:
                # The forward's decision warmed and retained it (``parts`` for a column range): an invariant.
                raise RuntimeError("recompute recipe cannot rebuild the wide GEMM's output bitwise: its weight split "
                                   "is cold inside a CUDA graph capture")
    if prec is None:
        return F.linear(x, weight, bias)
    if grad:
        return _WideLinear.apply(x, weight, bias, prec)
    _record("triton", x, weight.shape[0], prec, bias=bias is not None)
    return _launch(x, _split(weight, False), prec, bias)


def linear_into(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                rows: int | None = None):
    """``out[:] = x @ weight^T (+ bias)`` (no autograd): the 3xTF32 GEMM written into ``out`` where it applies,
    else cuBLAS's mm/addmm into ``out``. ``rows``: the whole GEMM's rows when ``x`` is a row chunk of it (every
    chunk takes the same kernel, so the rows equal those of one GEMM)."""
    prec = None
    if not torch.compiler.is_compiling() and _enabled():
        prec = _decide(x, weight, False, bias, out=out, rows=rows, warm=(False,))
    if prec is None:
        if bias is None:
            return torch.mm(x, weight.t(), out=out)
        return torch.addmm(bias, x, weight.t(), out=out)
    _record("triton", x, weight.shape[0], prec, bias=bias is not None, out=True)
    return _launch(x, _split(weight, False), prec, bias, out=out)


def matmul(grad: torch.Tensor, weight: torch.Tensor, c: torch.Tensor | None = None, rows: int | None = None,
           accumulate: bool = False, enabled: bool | None = None) -> torch.Tensor:
    """``grad @ weight (+ c)`` for a frozen ``weight`` (an input gradient ``dY @ W``, no autograd): the 3xTF32 GEMM
    where it applies, else ``torch.mm`` / ``torch.addmm``. ``accumulate``: written into ``c`` (``c.addmm_``), else a
    new tensor. ``rows`` as in ``linear_into``. ``enabled``: the dispatch state of the forward (``dispatch_on``,
    saved by a Function for its backward, which runs outside the forward's context: its thresholds apply); None:
    now."""
    prec = None
    if enabled is None:
        enabled = dispatch_on()
    if enabled and not torch.is_grad_enabled():
        prec = _decide(grad, weight, True, c=c, rows=rows, warm=(True,),
                       band=enabled if isinstance(enabled, tuple) else None)
    if prec is None:
        if c is None:
            return torch.mm(grad, weight)
        return c.addmm_(grad, weight) if accumulate else torch.addmm(c, grad, weight)
    _record("triton", grad, weight.shape[1], prec, phase="bwd", accumulate=c is not None)
    # In place: each program reads its tile of c before it stores the same tile.
    return _launch(grad, _split(weight, True), prec, c=c, out=c if accumulate else None)


@contextmanager
def compiled_region():
    """Around a torch.compile'd block: its GEMMs stay cuBLAS (Inductor's), also in frames that run eagerly after a
    graph break."""
    global _COMPILED_REGION
    _COMPILED_REGION += 1
    try:
        yield
    finally:
        _COMPILED_REGION -= 1
