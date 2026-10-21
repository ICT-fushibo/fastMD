"""Float32 GEMM precision: effective state, scoped changes, run records, oracle gates.

Measured with torch 2.12.0a0 (NGC 26.05), cuBLAS 13.4 on GB300, comparing
cuBLAS float32 GEMMs against float64 (TF32 ~3e-4 relative, IEEE ~2e-7):

* cuBLAS dispatches on the ``fp32_precision`` entry ``(cuda, matmul)``. Its
  value ``tf32`` means TF32; ``ieee`` or an unset ``none`` means IEEE FP32.
  ``set_float32_matmul_precision`` and ``allow_tf32`` write this entry.
* ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`` (exported by NGC containers) only
  changes the start-up default to ``high``/TF32; an explicit ``highest`` or
  ``allow_tf32 = False`` still disables TF32. Before the fp32_precision API
  (torch < 2.9, from source, not re-measured) ATen ORed the override into
  ``allow_tf32``, forcing TF32 under ``highest``.
* ``torch.get_float32_matmul_precision()`` is not the effective state: it
  reports ``high`` after ``fp32_precision = "ieee"`` and raises after a
  new-API-only ``tf32``. ``NVIDIA_TF32_OVERRIDE=0`` disables TF32 inside
  cuBLAS while every torch flag still reports TF32.

Hence gates use :func:`fp32_gemm_mode`, never the legacy precision string.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
from typing import Callable, Iterator, Mapping
import warnings

import torch
from torch import Tensor

from fastmd.config import _SCOPE_LOCK
from fastmd.config import OPTION_REGISTRY
from fastmd.config import TRITON_DOT_PRECISIONS
from fastmd.config import ConfigurationWarning
from fastmd.config import MATMUL_PRECISIONS


# Process environment that selects, forces or disables cuBLAS FP32 GEMM math.
PRECISION_ENVIRONMENT = (
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE",
    "CUBLAS_EMULATE_SINGLE_PRECISION", "CUBLAS_EMULATION_STRATEGY",
)
# fp32_precision entries, parents first: setting a parent also rewrites its
# unset children, so restoration must assign parents before children.
_FP32_KEYS = (
    ("generic", "all"), ("cuda", "all"), ("mkldnn", "all"),
    ("cuda", "matmul"), ("cuda", "conv"), ("cuda", "rnn"),
    ("mkldnn", "matmul"), ("mkldnn", "conv"), ("mkldnn", "rnn"),
)
# Same-precision implementation gates (tested path vs generic reference).
# IEEE: measured distributed-vs-reference force agreement is ~1e-6 eV/A
# (see the k3-precision README for the per-size margin). Energy stays at
# 1e-4 eV/atom: FP32 total-energy resolution is ~2e-5 eV/atom. Two different
# TF32 implementations disagree by ~1e-3 eV/A, so TF32 keeps the historic
# loose gate; its precision error against 'highest' is reported separately.
IMPLEMENTATION_TOLERANCES = {
    "ieee": {"energy_error_eV_per_atom": 1e-4, "force_RMSE_eV_A": 1e-5, "force_max_error_eV_A": 5e-5},
    "tf32": {"energy_error_eV_per_atom": 1e-4, "force_RMSE_eV_A": 5e-3, "force_max_error_eV_A": 5e-2},
}
_WARNED: set[str] = set()
# InferenceConfig.triton_dot_precision of the active scope (triton_dot_precision_scope); None: the
# MATRIS_TRITON_DOT_PRECISION environment value, default auto.
_TRITON_DOT_PRECISION: str | None = None
# The Triton FP32 dot sites (tl.dot on FP32 operands) and, per requested precision, the sites whose cuBLAS
# path runs instead where cuBLAS computes that precision itself (TF32 for tf32, IEEE FP32 for ieee). GB200,
# 10,752 atoms, ms (Triton / cuBLAS path): row_gemm (C4 projections) TF32 0.93 / 0.74 at [7.6M, 32] -> 128
# (0.49 / 0.55 at -> 64), FP32 FMA 2.65 / 2.15; edge_ffn FMA 7.18 / 3.83 (forward + backward);
# act_linear (the saved-tensor recompute's gate-input rebuild): wherever cuBLAS runs TF32 the forward's own
# cuBLAS layers rebuild its values bitwise (triton_dot_input_precision). FMA stays Triton where it is faster:
# gated_tail 6.02 / 6.29, act_linear 3.90 / 4.17. wide_gemm (the 3xTF32 GEMMs of the frozen projections with
# K, N >= 64): cuBLAS for ieee (FMA dots far slower) and under TF32 cuBLAS for tf32 (cuBLAS TF32 is faster than a
# Triton TF32 GEMM: 3.1 vs 4.1 ms over the six 2M-row shapes of the 10M model). Not a site: the projection
# gradient's row_addmm (FP32 FMA
# in every mode, faster than cuBLAS's addmm).
TRITON_DOT_SITES = ("row_gemm", "edge_ffn", "gated_tail", "act_linear", "wide_gemm")
_CUBLAS_PATH = frozenset({("row_gemm", "tf32"), ("row_gemm", "ieee"), ("edge_ffn", "ieee"), ("wide_gemm", "tf32")})


def _fp32_getter():
    return getattr(torch._C, "_get_fp32_precision_getter", None)


def fp32_gemm_mode() -> str:
    """Effective cuBLAS float32 GEMM math of this process: ``tf32`` or ``ieee``."""
    getter = _fp32_getter()
    if getter is not None:
        mode = "none"
        # An unset entry inherits from its parent backend, then generic.
        for backend, op in (("cuda", "matmul"), ("cuda", "all"), ("generic", "all")):
            mode = getter(backend, op)
            if mode != "none":
                break
        tf32 = mode == "tf32"
    else:
        # Before the fp32_precision API this getter includes the env override.
        tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    if tf32 and os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
        tf32 = False
    return "tf32" if tf32 else "ieee"


def _read(function):
    try:
        return function()
    except RuntimeError as exc:  # mixed legacy/new API state
        return "error: " + str(exc).split(",")[0][:120]


def precision_cache_key() -> tuple:
    """Cache-key component: torch's reported precision and the effective math."""
    return _read(torch.get_float32_matmul_precision), fp32_gemm_mode()


def precision_state() -> dict:
    """JSON-safe torch flags as reported, plus the effective GEMM mode."""
    getter = _fp32_getter()
    return {
        "fp32_gemm": fp32_gemm_mode(),
        "float32_matmul_precision": _read(torch.get_float32_matmul_precision),
        "cuda_matmul_allow_tf32": _read(lambda: bool(torch.backends.cuda.matmul.allow_tf32)),
        "cudnn_allow_tf32": _read(lambda: bool(torch.backends.cudnn.allow_tf32)),
        "fp32_precision": None if getter is None else {
            f"{backend}.{op}": _read(lambda: getter(backend, op)) for backend, op in _FP32_KEYS},
        "environment": {name: os.environ.get(name) for name in PRECISION_ENVIRONMENT},
        "triton_dot": {"requested": triton_dot_precision(),
                       **{site: triton_dot_input_precision(site) or "cublas" for site in TRITON_DOT_SITES}},
    }


def _snapshot():
    try:
        legacy = torch.get_float32_matmul_precision()
    except RuntimeError:
        legacy = None
    getter = _fp32_getter()
    entries = None if getter is None else tuple((key, getter(*key)) for key in _FP32_KEYS)
    return legacy, entries


def _restore(saved) -> None:
    legacy, entries = saved
    # The legacy setter also writes (cuda|mkldnn, matmul); entries win after.
    if legacy is not None:
        torch.set_float32_matmul_precision(legacy)
    for (backend, op), value in entries or ():
        torch._C._set_fp32_precision_setter(backend, op, value)


@contextmanager
def matmul_precision_scope(precision: str | None) -> Iterator[None]:
    """Temporarily ``torch.set_float32_matmul_precision``; None changes nothing.

    Restores the legacy precision and every fp32_precision entry, including
    new-API settings made by the caller. A legacy value that cannot be read
    in a mixed-API state is left as set here; GEMM math is still restored.
    The flags are process-wide, so the scope holds fastmd.config's reentrant
    scope lock: managed calls in other threads cannot interleave a snapshot
    or restore with it, and wait for it to exit.
    """
    if precision is None:
        yield
        return
    if precision not in MATMUL_PRECISIONS:
        raise ValueError(f"matmul precision must be one of {MATMUL_PRECISIONS!r}, got {precision!r}")
    with _SCOPE_LOCK:
        saved = _snapshot()
        torch.set_float32_matmul_precision(precision)
        try:
            yield
        finally:
            _restore(saved)


@contextmanager
def restore_fp32_state(changes: list | None = None) -> Iterator[None]:
    """Undo, on exit, any change of the FP32 GEMM flags (the legacy precision and every fp32_precision
    entry) made inside, e.g. by upstream model code calling ``torch.set_float32_matmul_precision``.
    Each distinct change is recorded once in ``changes`` as ``{"inside": ..., "restored": ...,
    "count": n}`` (the effective GEMM mode, the legacy precision and the fp32_precision entries).
    Holds fastmd.config's scope lock like the other scopes."""
    with _SCOPE_LOCK:
        saved = _snapshot()
        try:
            yield
        finally:
            if _snapshot() != saved:
                inside = _fp32_flags() if changes is not None else None
                _restore(saved)
                if changes is not None:
                    restored = _fp32_flags()
                    for change in changes:
                        if change["inside"] == inside and change["restored"] == restored:
                            change["count"] += 1
                            break
                    else:
                        changes.append({"inside": inside, "restored": restored, "count": 1})


def _fp32_flags() -> dict:
    getter = _fp32_getter()
    return {"fp32_gemm": fp32_gemm_mode(), "float32_matmul_precision": _read(torch.get_float32_matmul_precision),
            "fp32_precision": None if getter is None else {
                f"{backend}.{op}": _read(lambda: getter(backend, op)) for backend, op in _FP32_KEYS}}


def triton_dot_precision() -> str:
    """The requested precision of the Triton FP32 dots: auto, ieee, tf32x3 or tf32.

    The active configuration's ``triton_dot_precision``, else ``MATRIS_TRITON_DOT_PRECISION`` (every
    optimization profile), else auto."""
    if _TRITON_DOT_PRECISION is not None:
        return _TRITON_DOT_PRECISION
    value = os.environ.get("MATRIS_TRITON_DOT_PRECISION")
    return "auto" if value is None else OPTION_REGISTRY["MATRIS_TRITON_DOT_PRECISION"].parse(value)


def triton_dot_input_precision(site: str) -> str | None:
    """``tl.dot`` input precision of a Triton FP32 dot site (see ``TRITON_DOT_SITES``), or None: the site
    runs its cuBLAS path.

    auto: 3xTF32 where cuBLAS runs IEEE FP32 (:func:`fp32_gemm_mode`), TF32 where it runs TF32. ieee: FP32
    FMA dots; tf32x3 / tf32: 3xTF32 / TF32 whatever cuBLAS runs. A site takes its cuBLAS path only where
    cuBLAS computes the requested precision itself and is preferred there (``_CUBLAS_PATH``); act_linear
    (a rebuild of the forward's layers) also wherever cuBLAS runs TF32."""
    requested = triton_dot_precision()
    mode = fp32_gemm_mode()
    if site == "act_linear" and mode == "tf32":
        return None  # a rebuild of TF32 cuBLAS layers: bitwise through them, whatever the request
    if site == "wide_gemm" and requested == "ieee":
        return None  # the opt-out: cuBLAS FP32 (FMA dots at these widths are far slower)
    if requested == "auto":
        requested = "tf32" if mode == "tf32" else "tf32x3"
    if (site, requested) in _CUBLAS_PATH and mode == ("tf32" if requested == "tf32" else "ieee"):
        return None
    return requested


def deterministic_mode() -> bool:
    """The deterministic switch of MatRIS at every rank count: torch's own flag
    (``torch.use_deterministic_algorithms(True)``, ``warn_only`` included).

    Where it is on, every MatRIS sum that adds floats with atomics takes an
    ordered lowering (``triton_zero_skip_index.ordered_index_add_``, torch's
    deterministic index_add_): the zero-skipping row SUM of the P>1 private sink
    and of its reverse exchange, and the VJPs of the gather-cat, indexed
    projection, line envelope and three-body basis, whose forward kernels stay
    selected; the atomic weighted segment sum declines. Repeated runs are then
    bitwise at a fixed GPU type, rank count, software and memory decisions. Read
    on the host per kernel call, where the lowering is chosen (a VJP follows its
    forward): a CUDA graph keeps the lowering it was captured with. cuBLAS also
    needs ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (or ``:16:8``) before its first use."""
    return torch.are_deterministic_algorithms_enabled()


@contextmanager
def order_independent() -> Iterator[None]:
    """Ops inside give the same kept values in any order: integer sums, and writes
    whose only duplicate destinations are dropped trash rows. Under the
    deterministic flag torch would sort them on the device and serialize on the
    trash row (10,752 RDX atoms at P=4 on GB200: 57 ms of a 213 ms eager step), so
    they keep torch's default kernels; the flag (and ``warn_only``) is restored
    on exit. Host-side only, so capturable."""
    if not torch.are_deterministic_algorithms_enabled():
        yield
        return
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


def gemm_precision_key() -> tuple[str, str]:
    """This process's FP32 GEMM math for rank agreement and capture keys: cuBLAS's mode and the requested
    Triton dot precision (the per-site choice follows from both)."""
    return fp32_gemm_mode(), triton_dot_precision()


@contextmanager
def triton_dot_precision_scope(precision: str | None) -> Iterator[None]:
    """Temporarily set the requested Triton FP32 dot precision; None changes nothing. Process-wide like
    the torch flags: holds fastmd.config's scope lock."""
    if precision is None:
        yield
        return
    if precision not in TRITON_DOT_PRECISIONS:
        raise ValueError(f"triton dot precision must be one of {TRITON_DOT_PRECISIONS!r}, got {precision!r}")
    global _TRITON_DOT_PRECISION
    with _SCOPE_LOCK:
        saved, _TRITON_DOT_PRECISION = _TRITON_DOT_PRECISION, precision
        try:
            yield
        finally:
            _TRITON_DOT_PRECISION = saved


def matris_git_state() -> dict | None:
    """Commit and tracked-file dirty flag of the imported matris checkout."""
    root = Path(__file__).resolve().parents[1]
    try:
        run = lambda *args: subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True,
            timeout=30, check=True).stdout
        top, commit = run("rev-parse", "--show-toplevel", "HEAD").split()
        if Path(top).resolve() != root:
            return None
        dirty = bool(run("status", "--porcelain", "--untracked-files=no").strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return {"root": str(root), "commit": commit, "dirty": dirty}


def precision_record(requested: str | None, device=None, kernel_options: Mapping | None = None,
                     triton_dot_precision: str | None = None) -> dict:
    """Everything an output JSON needs to state its FP32 GEMM numerics.

    ``fp32_gemm`` describes cuBLAS, ``triton_dot`` the Triton FP32 dots (per site;
    whether a site's kernel runs at all is the kernel options' choice); ``device``
    names the GPU only for CUDA runs (``triton_dot_precision``: the caller's
    requested one, None the current). ``kernel_options`` is accepted and ignored
    (existing harnesses still pass it). ``deterministic_algorithms`` is
    :func:`deterministic_mode`, with the cuBLAS workspace setting it needs.
    """
    device = None if device is None else torch.device(device)
    with triton_dot_precision_scope(triton_dot_precision):
        state = precision_state()
    return {
        "requested": requested, **state,
        "deterministic_algorithms": deterministic_mode(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": (torch.cuda.get_device_name(device)
                   if device is not None and device.type == "cuda" else None),
        "matris_git": matris_git_state(),
    }


def warn_implicit_tf32(requested: str | None, where: str) -> bool:
    """Warn once per process when cuBLAS runs TF32 that nobody requested.

    ``requested`` is the caller's explicit precision, None if unspecified.
    Call it under the precision the caller will actually run with.
    """
    if requested in ("high", "medium") or fp32_gemm_mode() != "tf32" or "cublas" in _WARNED:
        return False
    _WARNED.add("cublas")
    source = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 made TF32 the process default"
              if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1" else
              "torch float32 matmul flags select TF32")
    wanted = ("matmul_precision='highest' did not disable it" if requested == "highest"
              else "no matmul precision was requested")
    warnings.warn(
        f"*** {where}: float32 GEMMs run in TF32 (10-bit mantissa) but {wanted}; "
        f"{source}. Forces can differ from FP32 by ~1e-3 eV/A or more. Use "
        "InferenceConfig(matmul_precision='highest') for FP32, or "
        "matmul_precision='high' to accept TF32 explicitly. ***",
        ConfigurationWarning, stacklevel=3)
    return True


def compute_references(predict: Callable[[], tuple[float, Tensor]],
                       run_precision: str, dot_precision: str | None = None) -> dict:
    """Reference energy/atom and CPU forces for an oracle comparison.

    ``predict`` must run the independent reference path (the caller supplies
    the generic kernel scope). It always runs under 'highest' (Triton dots
    auto: 3xTF32); when the tested run's effective GEMM math is not IEEE (cuBLAS
    TF32, or a Triton dot site at TF32 under ``dot_precision``, None:
    the current one) it runs again at ``run_precision`` and that dot
    precision, separating implementation from precision error.
    """
    with matmul_precision_scope(run_precision), triton_dot_precision_scope(dot_precision):
        run_mode = fp32_gemm_mode()
        if "tf32" in (triton_dot_input_precision(site) for site in TRITON_DOT_SITES):
            run_mode = "tf32"
        run_dot = triton_dot_precision()
    with matmul_precision_scope("highest"), triton_dot_precision_scope("auto"):
        highest_mode = fp32_gemm_mode()
        highest = predict()
    same = None
    if run_mode != "ieee":
        with matmul_precision_scope(run_precision), triton_dot_precision_scope(dot_precision):
            same = predict()
    return {"run_precision": run_precision, "run_fp32_gemm": run_mode, "run_triton_dot": run_dot,
            "highest_fp32_gemm": highest_mode, "highest": highest, "same_precision": same}


def ef_errors(energy_per_atom: float, forces: Tensor, reference) -> dict:
    reference_energy, reference_forces = reference
    delta = forces.detach().double().cpu() - reference_forces.detach().double().cpu()
    return {"energy_error_eV_per_atom": abs(float(energy_per_atom) - float(reference_energy)),
            "force_RMSE_eV_A": float(delta.square().mean().sqrt()),
            "force_max_error_eV_A": float(delta.abs().max())}


def oracle_report(energy_per_atom: float, forces: Tensor, references: dict, *,
                  precision_force_rmse_tolerance: float | None = None,
                  implementation_tolerance: dict | None = None,
                  reference_profile: str = "generic") -> dict:
    """Gate implementation error; report (optionally gate) precision error.

    Top-level error keys repeat the gated implementation comparison for
    readers of the historical oracle schema. ``implementation_tolerance``
    maps an fp32 GEMM mode to replacement gates (other modes keep
    IMPLEMENTATION_TOLERANCES). ``reference_profile`` only records which
    optimization profile the caller's eager reference ran.
    """
    mode = references["run_fp32_gemm"]
    same = references["same_precision"]
    implementation = ef_errors(energy_per_atom, forces, same if same is not None else references["highest"])
    tolerance = {**IMPLEMENTATION_TOLERANCES, **(implementation_tolerance or {})}[mode]
    implementation.update(reference="same_precision" if same is not None else "highest",
                          tolerance=dict(tolerance),
                          passed=all(implementation[key] <= limit for key, limit in tolerance.items()))
    precision = None
    if same is not None:
        precision = {
            "tested_vs_highest": ef_errors(energy_per_atom, forces, references["highest"]),
            "reference_vs_highest": ef_errors(*same, references["highest"]),
            "force_RMSE_tolerance_eV_A": precision_force_rmse_tolerance,
            "passed": None,
        }
        if precision_force_rmse_tolerance is not None:
            precision["passed"] = (precision["tested_vs_highest"]["force_RMSE_eV_A"]
                                   <= precision_force_rmse_tolerance)
    report = {
        "reference": {"path": f"original model, {reference_profile} eager profile", "precision": "highest",
                      "profile": reference_profile, "fp32_gemm": references["highest_fp32_gemm"]},
        "run_precision": references["run_precision"], "run_fp32_gemm": mode,
        "implementation": implementation, "precision": precision,
        **{key: implementation[key] for key in (
            "energy_error_eV_per_atom", "force_RMSE_eV_A", "force_max_error_eV_A")},
        # NaN errors compare False and therefore fail every gate.
        "passed": implementation["passed"] and (precision is None or precision["passed"] is not False),
    }
    if precision is None and precision_force_rmse_tolerance is not None:
        report["precision_note"] = (f"force RMSE tolerance {precision_force_rmse_tolerance} not applied: "
                                    "the tested GEMMs are IEEE FP32, as is the gated reference")
    return report
