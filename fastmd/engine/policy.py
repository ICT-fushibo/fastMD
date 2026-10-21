"""Kernel-selection policy and the fusion scope (docs/unified_engine_design.md §4.2, §4.3).

``enabled(op) = ops[op].enabled_eager or (fusion_scope and ops[op].enabled_fusion_scope)``
reproduces today's ``ENABLED and (*_EAGER or fusion scope)`` switches; ops without an
EAGER/fusion-scope pair have ``enabled_eager == enabled_fusion_scope``. ``min_rows``
is inclusive at every site: eligible when ``rows >= min_rows``.

The fusion scope is what runners open around capture, warmup and eager
fallback (formerly ``set_fused_graph_optimizations``); it is not "capturing".
It is a process-wide module global, not a ContextVar or thread-local:
activation-checkpoint recomputation re-evaluates the selection on the autograd
device thread and must see the forward's scope. Model gates read it at call
time through ``fusion_scope_active()``, never a by-value import, so dynamo
guards the live value. It is runtime state, not part of the policy or its
fingerprint.

Transition (step 2.2): ``KernelPolicy.from_globals()`` reads the module
globals the kernels read today (bound by ``ResolvedConfig.scope()``, legacy
env at import, or patched by tests); ``from_resolved()`` is the target
constructor and equals it inside ``resolved.scope()``. Neither is consulted by
a dispatch site yet. This module must not import ``fastmd.model``.

The policy covers the module globals the shared model code reads. The P>1
executor's ``IndexedProjectionConfig`` is outside it, so at P>1 the
fingerprint does not distinguish local backends (the dispatcher cache key
does). ``tiles`` holds every
registry "kernel" option of the op, tuning and diagnostic knobs included:
the fingerprint is deliberately over-inclusive. It identifies the selection
table (``ops`` and ``flags``); ``strict_errors`` changes error handling, not
selection, and is reported next to it.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
import sys
from types import MappingProxyType
from typing import Any

from fastmd.config import OPTION_REGISTRY
from fastmd.config import env_value

_FUSION_SCOPE = False

_INTERACTION_BLOCK = "fastmd.models.matris.interaction_block"
_FUNCTIONS = "fastmd.models.matris.functions"
# interaction_block.THRESHOLD_VALUE: gather-cat and line-envelope min rows (not a registry option).
_THRESHOLD_VALUE = 60000
# Registry options consumed as enabled/backend rather than as kernel parameters.
_NOT_TILES = frozenset({"MATRIS_COMPILED_LOWERINGS", "MATRIS_COMPILED_LOWERINGS_MODE"})
# Owner op of the "kernel" options bound in these modules; unbound DECOMPOSED_INDEXED_*
# options tune indexed_projection; any other lands in ``flags``.
_TILE_OWNERS = MappingProxyType({
    "fastmd.models.matris.op.triton_segment_attention": ("segment_attention",),
    "fastmd.models.matris.op.triton_line_envelope": ("line_envelope",),
    "fastmd.models.matris.op.triton_threebody_fourier": ("threebody_basis",),
    "fastmd.models.matris.op.triton_directed_pair_aggregate": ("directed_pair_aggregate", "directed_pair_expand"),
    "fastmd.models.matris.op.triton_residual_add": ("residual_add",),
    "fastmd.models.matris.op.triton_gated_ln": ("gated_ln", "gated_ln_residual"),
    "fastmd.engine.kernels.decomposed_indexed_cat_silu_linear": ("indexed_projection",),
    "fastmd.engine.kernels.row_tiles": ("line_envelope", "directed_pair_aggregate", "directed_pair_expand",
                                        "residual_add", "indexed_projection"),
})
_INDEXED_PREFIXES = ("MATRIS_DECOMPOSED_INDEXED_",)


def fusion_scope_active() -> bool:
    return _FUSION_SCOPE


def set_fusion_scope(enabled: bool) -> bool:
    """Set (not OR) the process-wide fusion scope; returns the previous value."""
    global _FUSION_SCOPE
    previous, _FUSION_SCOPE = _FUSION_SCOPE, bool(enabled)
    return previous


@contextmanager
def fusion_scope(enabled: bool = True) -> Iterator[None]:
    """Set the fusion scope for the block; the outer value is restored on exit."""
    previous = set_fusion_scope(enabled)
    try:
        yield
    finally:
        set_fusion_scope(previous)


def _json_value(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


@dataclass(frozen=True)
class OpPolicy:
    enabled_eager: bool
    enabled_fusion_scope: bool
    min_rows: int | None = None
    backend: str | None = None
    # Sorted (registry option name, value) kernel parameters and variant switches.
    tiles: tuple[tuple[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"enabled_eager": self.enabled_eager, "enabled_fusion_scope": self.enabled_fusion_scope,
                "min_rows": self.min_rows, "backend": self.backend,
                "tiles": {name: _json_value(value) for name, value in self.tiles}}


@dataclass(frozen=True)
class KernelPolicy:
    ops: Mapping[str, OpPolicy]
    strict_errors: bool = False
    # Fingerprinted switches without fusion-scope semantics (read at their sites).
    flags: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ops", MappingProxyType(dict(sorted(self.ops.items()))))
        object.__setattr__(self, "flags", MappingProxyType(dict(sorted(self.flags.items()))))

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def enabled(self, op: str, fusion_scope: bool | None = None) -> bool:
        """Selection switch of ``op``; ``fusion_scope`` defaults to the live scope."""
        policy = self.ops[op]
        scope = _FUSION_SCOPE if fusion_scope is None else fusion_scope
        return policy.enabled_eager or (bool(scope) and policy.enabled_fusion_scope)

    def as_dict(self) -> dict[str, Any]:
        return {"ops": {name: policy.as_dict() for name, policy in self.ops.items()},
                "flags": {name: _json_value(value) for name, value in self.flags.items()},
                "strict_errors": self.strict_errors}

    @cached_property
    def fingerprint(self) -> str:
        """Of the selection table; ``strict_errors`` is excluded."""
        table = {key: value for key, value in self.as_dict().items() if key != "strict_errors"}
        payload = json.dumps(table, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def from_globals(cls, *, strict_errors: bool = False) -> KernelPolicy:
        """The policy the kernels read now: loaded modules' globals, else their import value."""
        def option(name: str) -> Any:
            spec = OPTION_REGISTRY[name]
            for module_name, attribute in spec.bindings:
                module = sys.modules.get(module_name)
                if module is not None and hasattr(module, attribute):
                    return getattr(module, attribute)
            # Unloaded module (the value its import would bind) or read at call time.
            return spec.parse(env_value(name))

        block, functions = sys.modules.get(_INTERACTION_BLOCK), sys.modules.get(_FUNCTIONS)
        return cls._build(option, getattr(block, "THRESHOLD_VALUE", _THRESHOLD_VALUE),
                          getattr(functions, "_FUSED_ACTIVATIONS", True), strict_errors)

    @classmethod
    def from_resolved(cls, resolved, *, strict_errors: bool = False) -> KernelPolicy:
        """The same table from ``resolved.kernel_options`` (fused activations on)."""
        return cls._build(resolved.kernel_options.__getitem__, _THRESHOLD_VALUE, True, strict_errors)

    @classmethod
    def _build(cls, read: Callable[[str], Any], threshold: int, activations: bool,
               strict_errors: bool) -> KernelPolicy:
        consumed: set[str] = set()

        def option(name: str) -> Any:
            consumed.add(name)
            return read(name)

        def value(suffix: str) -> Any:
            return option("MATRIS_" + suffix)

        tiles: dict[str, dict[str, Any]] = {}
        flags: dict[str, Any] = {
            "merged_attention_projections": value("MERGED_ATTENTION_PROJECTIONS"),
            "merged_refinement_projection": value("MERGED_REFINEMENT_PROJECTION"),
            "skip_dead_threebody_tail": value("SKIP_DEAD_THREEBODY_TAIL"),
            "undirected_edge_init": value("UNDIRECTED_EDGE_INIT"),
        }
        for name, spec in OPTION_REGISTRY.items():
            if spec.category != "kernel" or name in _NOT_TILES:
                continue
            owners = _TILE_OWNERS.get(spec.bindings[0][0], ()) if spec.bindings else (
                ("indexed_projection",) if name.startswith(_INDEXED_PREFIXES) else ())
            for owner in owners:
                tiles.setdefault(owner, {})[name] = option(name)
            if not owners:
                flags[name] = option(name)
        for suffix in ("FUSED_PAIRED_SEGMENT_ATTENTION", "FUSED_SEGMENT_ATTENTION_REUSE_CSR",
                       "FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR"):
            tiles.setdefault("segment_attention", {})["MATRIS_" + suffix] = value(suffix)

        def op(name: str, enabled: bool, eager: bool | None = None, min_rows: int | None = None,
               backend: str | None = None) -> tuple[str, OpPolicy]:
            # eager is None: no EAGER/fusion-scope pair, the switch applies everywhere.
            return name, OpPolicy(bool(enabled if eager is None else enabled and eager), bool(enabled),
                                  min_rows, backend, tuple(sorted(tiles.get(name, {}).items())))

        ln_rows = value("GATED_LN_FUSE_MIN_ROWS")
        ops = dict([
            # EAGER-or-fusion-scope ops; weighted_segment_sum and gather_cat have no ENABLED switch.
            op("weighted_segment_sum", True, value("FUSED_WEIGHTED_SEGMENT_EAGER")),
            op("gather_cat", True, value("FUSED_GATHER_CAT_EAGER"), threshold),
            op("segment_attention", value("FUSED_SEGMENT_ATTENTION"), value("FUSED_SEGMENT_ATTENTION_EAGER"),
               value("FUSED_SEGMENT_ATTENTION_MIN_ROWS")),
            op("line_envelope", value("FUSED_LINE_ENVELOPE"), value("FUSED_LINE_ENVELOPE_EAGER"), threshold),
            op("threebody_linear", value("FUSED_THREEBODY_LINEAR"), value("FUSED_THREEBODY_LINEAR_EAGER"),
               value("FUSED_THREEBODY_LINEAR_MIN_ROWS")),
            op("threebody_basis", value("FUSED_THREEBODY_BASIS"), value("FUSED_THREEBODY_BASIS_EAGER"),
               value("FUSED_THREEBODY_BASIS_MIN_ROWS")),
            # Plain switches.
            op("directed_pair_aggregate", value("FUSED_DIRECTED_PAIR_AGGREGATE")),
            op("directed_pair_expand", value("FUSED_DIRECTED_PAIR_EXPAND")),
            op("residual_add", value("FUSED_RESIDUAL_ADD"), min_rows=value("FUSED_RESIDUAL_ADD_MIN_ROWS")),
            op("segment_softmax", value("FUSED_SEGMENT_SOFTMAX")),
            # Step 2.5a (D6): deterministic CSR segment sums (summation order); a profile default since C10.
            op("segment_sum_csr", value("CSR_SEGMENT_SUM")),
            # Also requires the module's structural GatedMLP._fuse_ln.
            op("gated_ln", True, min_rows=ln_rows),
            op("gated_ln_residual", value("FUSED_GATED_LN_RESIDUAL"), min_rows=ln_rows),
            op("indexed_projection", value("FUSED_INDEXED_CAT_LINEAR"),
               min_rows=value("INDEXED_CAT_LINEAR_MIN_ROWS"), backend=value("FUSED_INDEXED_CAT_LINEAR_BACKEND")),
            # Step 2.5b (D8): indexed projections over parts of different widths.
            op("mixed_width_cat", value("MIXED_WIDTH_CAT")),
            # Step 2.5b (D13a): the second linears' bias in the fused gated LayerNorm (where gated_ln fuses).
            op("gated_ln_bias_fold", value("GATED_LN_BIAS_FOLD")),
            # C2 (step 2.5b D13b): the fused narrow tail, where gated_ln fuses after an indexed projection.
            op("gated_tail", value("FUSED_GATED_TAIL"), min_rows=ln_rows),
            # C4 (residual round 2): row GEMMs for the K <= 32 frozen projections; the fused line-graph edge FFN.
            op("row_gemm", value("TRITON_ROW_GEMM")),
            op("edge_ffn", value("FUSED_EDGE_FFN")),
            # TX: 3xTF32 GEMMs for the wide (K, N >= 64) frozen projections, forward and input gradient.
            op("wide_gemm", value("TRITON_WIDE_GEMM"), min_rows=value("TRITON_WIDE_GEMM_MIN_ROWS")),
            op("compiled_lowerings", value("COMPILED_LOWERINGS"), backend=value("COMPILED_LOWERINGS_MODE")),
            op("fused_activations", activations),
        ])
        # A model/kernel option not consumed above (e.g. MATRIS_MERGED_PROJECTION_SINGLE_BWD) is still fingerprinted.
        for name, spec in OPTION_REGISTRY.items():
            if spec.category in ("model", "kernel") and name not in consumed:
                flags[name] = option(name)
        return cls(ops, bool(strict_errors), flags)


def policy_summary(*, strict_errors: bool = False) -> dict[str, Any]:
    """``config_summary``/recorder entry for the live globals; never raises."""
    try:
        fingerprint = KernelPolicy.from_globals().fingerprint
    except Exception as exc:  # e.g. a malformed MATRIS_* value no kernel has read yet
        fingerprint = f"<{type(exc).__name__}>"
    return {"fingerprint": fingerprint, "strict_errors": bool(strict_errors)}
