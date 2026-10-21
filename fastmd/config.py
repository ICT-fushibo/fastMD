"""Typed, scoped inference configuration, without importing torch or CUDA.

``legacy`` preserves the historical defaults and environment overrides.
``topology`` and ``generic`` are reproducible profiles: registered environment
variables are ignored in favour of registry defaults plus the selected profile.
Use ``expert_overrides`` for intentional per-operator changes. Profile names
describe requested optimizations, not a promise of kernel eligibility.

Scopes serialize managed calls while temporarily binding existing module
constants and, when requested, torch's float32 matmul precision. They do not
mutate ``os.environ``. Unmanaged raw model calls from other threads must not
run concurrently with a configured inference scope; a training step
(MatRIS.forward(is_training=True) and its backward) counts as such a call.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
import hashlib
import json
import math
import os
import re
import sys
import threading
from types import MappingProxyType
from typing import Any
import warnings


class ConfigurationWarning(UserWarning):
    """A legacy or ineffective configuration needs the caller's attention."""


class ConfigurationDeprecationWarning(ConfigurationWarning, FutureWarning):
    """A deprecated spelling that still works for one version (shown by default)."""


# Runtime and diagnostic options (memory policy, debug checks) that are not
# kernel choices: the environment sets them under every profile.
RUNTIME_ENVIRONMENT = frozenset({
    "MATRIS_SAVED_TENSOR_RECOMPUTE", "MATRIS_TOPOLOGY_REUSE_CHECK",
    "MATRIS_DECOMPOSED_INDEXED_DEBUG_CHECKS", "MATRIS_TOPOLOGY_VERIFY",
})


def parse_bool(value: Any, *, name: str = "boolean") -> bool:
    """Accept common explicit boolean spellings; never use string truthiness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "on", "yes"}:
            return True
        if normalized in {"0", "false", "off", "no"}:
            return False
    raise ValueError(f"{name}: expected true/false, 1/0, on/off or yes/no; got {value!r}")


@dataclass(frozen=True)
class OptionSpec:
    name: str
    kind: str
    default: Any
    category: str
    description: str
    bindings: tuple[tuple[str, str], ...] = ()
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    # Integer tiles used directly as tl.arange extents must be powers of two.
    power_of_two: bool = False

    def parse(self, value: Any) -> Any:
        if value is None:
            if self.default is None:
                return None
            raise ValueError(f"{self.name}: None is not supported")
        if self.kind == "bool":
            result = parse_bool(value, name=self.name)
        elif self.kind == "int":
            if isinstance(value, bool) or not re.fullmatch(r"[+-]?\d+", str(value).strip()):
                raise ValueError(f"{self.name}: expected an integer, got {value!r}")
            result = int(value)
        elif self.kind == "float":
            if isinstance(value, bool):
                raise ValueError(f"{self.name}: expected a finite number, got {value!r}")
            try:
                result = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.name}: expected a finite number, got {value!r}") from exc
            if not math.isfinite(result):
                raise ValueError(f"{self.name}: expected a finite number, got {value!r}")
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{self.name}: expected a nonempty string")
            result = value.strip().lower()
        if self.minimum is not None and result < self.minimum:
            raise ValueError(f"{self.name}: must be >= {self.minimum:g}, got {result!r}")
        if self.power_of_two and (result < 1 or result & (result - 1)):
            raise ValueError(f"{self.name}: must be a power of two, got {result!r}")
        if result in _REMOVED_CHOICES.get(self.name, ()):
            raise ValueError(f"{self.name}={result!r}: the CuTeDSL backends were removed on 2026-09-28; "
                             f"use {' or '.join(map(repr, self.choices))}")
        if self.choices and result not in self.choices:
            raise ValueError(f"{self.name}: expected one of {self.choices!r}, got {result!r}")
        return result

    def format(self, value: Any) -> str | None:
        if value is None:
            return None
        if self.kind == "bool":
            return "1" if value else "0"
        return str(value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _option(
    suffix: str, kind: str, default: Any, category: str, description: str,
    module: str = "", attribute: str = "", *, choices: tuple[Any, ...] = (),
    minimum: float | None = None, power_of_two: bool = False,
) -> OptionSpec:
    binding = (("fastmd." + module, attribute),) if module else ()
    return OptionSpec("MATRIS_" + suffix, kind, default, category, description, binding, choices, minimum,
                      power_of_two)


# This registry is the source of truth for defaults, validation, documentation
# and loaded-module bindings. Optional tuning values use None to mean auto.
_OPTIONS = [
    _option("GATED_LN_FUSE_MIN_ROWS", "int", 10000, "model", "Minimum rows for fused gated LayerNorm.", "models.matris.functions", "_LN_FUSE_THRESH", minimum=0),
    _option("FUSED_GATED_LN_RESIDUAL", "bool", True, "model", "Fuse gated LayerNorm with residual when eligible.", "models.matris.functions", "_FUSED_GATED_LN_RESIDUAL"),
    _option("FUSED_SEGMENT_SOFTMAX", "bool", True, "model", "Use fused segmented softmax.", "engine.kernels.segment", "_FUSED_SEGMENT_SOFTMAX"),
    _option("FUSED_INDEXED_CAT_LINEAR", "bool", False, "model", "Enable indexed first-linear projection; required by merged projections.", "models.matris.functions", "_INDEXED_CAT_LINEAR_FUSE"),
    _option("FUSED_INDEXED_CAT_LINEAR_BACKEND", "str", "decomposed_indexed_silu", "model", "Indexed projection implementation.", "models.matris.functions", "_INDEXED_CAT_LINEAR_BACKEND", choices=("decomposed_indexed_silu",)),
    _option("INDEXED_CAT_LINEAR_MIN_ROWS", "int", 20000, "model", "Minimum rows for indexed projections.", "models.matris.functions", "_INDEXED_CAT_LINEAR_MIN_ROWS", minimum=0),
    # Step 2.5b (D8): also read by the OpDispatcher's policy (the cat4 gate and the P>1 projection config).
    _option("MIXED_WIDTH_CAT", "bool", True, "model", "Indexed projections over parts of different widths (e.g. [32|128|32|32]); changes the summation order.", "models.matris.functions", "_MIXED_WIDTH_CAT"),
    # Step 2.5b (D13a).
    _option("GATED_LN_BIAS_FOLD", "bool", True, "model", "The fused gated LayerNorm adds the GatedMLP second linears' bias (no GEMM bias epilogue).", "models.matris.functions", "_GATED_LN_BIAS_FOLD"),
    # C2 (step 2.5b D13b).
    _option("FUSED_GATED_TAIL", "bool", True, "model", "A GatedMLP's narrow tail (both second linears, K <= 32, and the gated-LN finish) in one kernel per direction; changes the GEMM rounding.", "models.matris.functions", "_FUSED_GATED_TAIL"),
    # C4 (residual round 2): topology-profile switches (see _PROFILE_SWITCHES).
    _option("TRITON_ROW_GEMM", "bool", False, "model", "Triton row GEMMs for the K = 32 frozen projections (dots at MATRIS_TRITON_DOT_PRECISION; auto: 3xTF32 under IEEE FP32 GEMMs, cuBLAS in TF32 mode) and the projection-with-alias gradient (FP32 FMA); changes the GEMM rounding.", "engine.kernels.merged_frozen_projections", "_TRITON_ROW_GEMM"),
    _option("TRITON_WIDE_GEMM", "bool", False, "model", "3xTF32 Triton GEMMs for the wide frozen-weight projections (K, N >= 64; forward and input gradient) at MATRIS_TRITON_DOT_PRECISION (auto: 3xTF32 under IEEE FP32 GEMMs, cuBLAS TF32 in TF32 mode; ieee: cuBLAS FP32); changes the GEMM rounding.", "engine.kernels.triton_wide_gemm", "_TRITON_WIDE_GEMM"),
    # GB200: the 3xTF32 GEMM's ~30 us launch path loses to cuBLAS FP32 below ~60k rows; 2.1-4.4x faster from 120k.
    _option("TRITON_WIDE_GEMM_MIN_ROWS", "int", 65536, "model", "Minimum rows for the 3xTF32 wide GEMMs inside a CUDA graph capture (below: cuBLAS FP32); eager 4x this; at P>1 the topology epoch moves both above the ranks' atom-row range when inside it, other P>1 layouts 0 (rank-uniform kernel choice).", "engine.kernels.triton_wide_gemm", "_MIN_ROWS", minimum=0),
    _option("FUSED_EDGE_FFN", "bool", False, "model", "The line-graph refinement's edge FFN (Linear-SiLU-Linear, no bias) in one kernel per direction (dots at MATRIS_TRITON_DOT_PRECISION); changes the GEMM rounding.", "models.matris.interaction_block", "_FUSED_EDGE_FFN"),
    _option("FUSED_WEIGHTED_SEGMENT_EAGER", "bool", False, "model", "Allow fused weighted reduction outside graph capture.", "models.matris.interaction_block", "_FUSED_WEIGHTED_SEGMENT_EAGER"),
    # Step 2.5a (D6): read only by the OpDispatcher's policy, not a module global.
    _option("CSR_SEGMENT_SUM", "bool", True, "model", "Deterministic CSR segment sums (no atomics; changes the summation order). Default on in the topology and legacy profiles (C10), off in generic; 0 restores the atomic sums (kept for one version)."),
    _option("FUSED_GATHER_CAT_EAGER", "bool", False, "model", "Allow fused gather/concatenate outside graph capture.", "models.matris.interaction_block", "_FUSED_GATHER_CAT_EAGER"),
    _option("FUSED_DIRECTED_PAIR_AGGREGATE", "bool", True, "model", "Fuse directed-pair aggregation.", "models.matris.interaction_block", "_FUSED_DIRECTED_PAIR_AGGREGATE"),
    _option("FUSED_DIRECTED_PAIR_EXPAND", "bool", True, "model", "Fuse directed-pair expansion.", "models.matris.interaction_block", "_FUSED_DIRECTED_PAIR_EXPAND"),
    _option("FUSED_SEGMENT_ATTENTION", "bool", True, "model", "Enable segmented attention fusion.", "models.matris.interaction_block", "_FUSED_SEGMENT_ATTENTION_ENABLED"),
    _option("FUSED_SEGMENT_ATTENTION_EAGER", "bool", False, "model", "Allow segmented attention fusion outside graph capture.", "models.matris.interaction_block", "_FUSED_SEGMENT_ATTENTION_EAGER"),
    _option("FUSED_SEGMENT_ATTENTION_MIN_ROWS", "int", 0, "model", "Minimum rows for fused segmented attention.", "models.matris.interaction_block", "_FUSED_SEGMENT_ATTENTION_MIN_ROWS", minimum=0),
    _option("FUSED_PAIRED_SEGMENT_ATTENTION", "bool", True, "model", "Share paired attention work when eligible.", "models.matris.interaction_block", "_FUSED_PAIRED_SEGMENT_ATTENTION"),
    _option("FUSED_SEGMENT_ATTENTION_REUSE_CSR", "bool", True, "model", "Reuse verified CSR for segmented attention.", "models.matris.interaction_block", "_FUSED_SEGMENT_ATTENTION_REUSE_CSR"),
    _option("FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR", "bool", True, "model", "Use sorted-target CSR attention.", "models.matris.interaction_block", "_FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR"),
    _option("FUSED_LINE_ENVELOPE", "bool", True, "model", "Enable fused line-graph envelope.", "models.matris.interaction_block", "_FUSED_LINE_ENVELOPE_ENABLED"),
    _option("FUSED_LINE_ENVELOPE_EAGER", "bool", False, "model", "Allow fused line envelope outside graph capture.", "models.matris.interaction_block", "_FUSED_LINE_ENVELOPE_EAGER"),
    _option("FUSED_RESIDUAL_ADD", "bool", True, "model", "Fuse frozen-weight residual addition.", "models.matris.interaction_block", "_FUSED_RESIDUAL_ADD_ENABLED"),
    _option("FUSED_RESIDUAL_ADD_MIN_ROWS", "int", 0, "model", "Minimum rows for residual-add fusion.", "models.matris.interaction_block", "_FUSED_RESIDUAL_ADD_MIN_ROWS", minimum=0),
    _option("MERGED_ATTENTION_PROJECTIONS", "bool", True, "model", "Merge compatible attention projections; requires indexed-linear fusion and equal input widths.", "models.matris.interaction_block", "_MERGED_ATTN_PROJECTIONS"),
    _option("MERGED_REFINEMENT_PROJECTION", "bool", True, "model", "Merge compatible refinement projections; requires indexed-linear fusion.", "models.matris.interaction_block", "_MERGED_REFINEMENT_PROJECTION"),
    # C3: the merged projection's backward as one GEMM over a shared gradient buffer.
    _option("MERGED_PROJECTION_SINGLE_BWD", "str", "auto", "model", "Consumers of a merged projection write their gradients into one shared buffer and its backward is one GEMM (changes the summation order): on, off (one GEMM per output, kept for one version) or auto (on while the memory admission predicts that it fits, before saved_tensor_recompute=auto climbs).", "engine.kernels.merged_frozen_projections", "_SINGLE_BWD", choices=("off", "on", "auto")),
    _option("SKIP_DEAD_THREEBODY_TAIL", "bool", True, "model", "Skip edge updates proven unused by the output: the last interaction block's three-body update; at P=1 also its bond update under a conservative force/stress head, and the atom attention's bond update of a graph without triplets (P>1 always prunes these).", "models.matris.interaction_block", "_SKIP_DEAD_THREEBODY_TAIL"),
    _option("FUSED_THREEBODY_LINEAR", "bool", True, "model", "Enable fused three-body basis projection.", "models.matris.feature_embed", "_FUSED_THREEBODY_LINEAR_ENABLED"),
    _option("FUSED_THREEBODY_LINEAR_EAGER", "bool", False, "model", "Allow three-body projection fusion outside graph capture.", "models.matris.feature_embed", "_FUSED_THREEBODY_LINEAR_EAGER"),
    _option("FUSED_THREEBODY_LINEAR_MIN_ROWS", "int", 60000, "model", "Minimum rows for three-body projection fusion.", "models.matris.feature_embed", "_FUSED_THREEBODY_LINEAR_MIN_ROWS", minimum=0),
    _option("FUSED_THREEBODY_BASIS", "bool", True, "model", "Enable fused three-body Fourier basis.", "models.matris.feature_embed", "_FUSED_THREEBODY_BASIS_ENABLED"),
    _option("FUSED_THREEBODY_BASIS_EAGER", "bool", False, "model", "Allow three-body basis fusion outside graph capture.", "models.matris.feature_embed", "_FUSED_THREEBODY_BASIS_EAGER"),
    _option("FUSED_THREEBODY_BASIS_MIN_ROWS", "int", 60000, "model", "Minimum rows for three-body basis fusion.", "models.matris.feature_embed", "_FUSED_THREEBODY_BASIS_MIN_ROWS", minimum=0),
    _option("UNDIRECTED_EDGE_INIT", "bool", True, "model", "Initialize equivalent edge pairs only once.", "models.matris.feature_embed", "_UNDIRECTED_EDGE_INIT"),
    # Default of InferenceConfig.saved_tensor_recompute (what autograd keeps, not a kernel choice).
    # Default of InferenceConfig.triton_dot_precision (fastmd.precision.triton_dot_input_precision); read in every profile.
    _option("TRITON_DOT_PRECISION", "str", "auto", "model", "Default of InferenceConfig.triton_dot_precision, the input precision of the Triton FP32 dots (the wide 3xTF32 GEMMs, C4 row GEMMs and edge FFN, C2 gated tail, the recompute's act_linear): auto (3xTF32 where cuBLAS runs IEEE FP32, TF32 where it runs TF32), ieee (cuBLAS for the wide GEMMs, FP32 FMA dots elsewhere), tf32x3 or tf32. Read in every optimization profile.", choices=("auto", "ieee", "tf32x3", "tf32")),
    _option("SAVED_TENSOR_RECOMPUTE", "str", "auto", "memory", "Default of InferenceConfig.saved_tensor_recompute: off, on, deep or auto (the cheapest level the admission model predicts to fit the device budget).", choices=("off", "on", "deep", "auto")),
    _option("COMPILED_LOWERINGS", "bool", False, "kernel", "Opt into torch.compile operator lowerings.", "models.matris.op.compiled_lowerings", "COMPILED_LOWERINGS_ENABLED"),
    _option("COMPILED_LOWERINGS_MODE", "str", "max-autotune-no-cudagraphs", "kernel", "torch.compile mode for operator lowerings.", "models.matris.op.compiled_lowerings", "_COMPILE_MODE", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")),
    # Row tiles of the memory-bound row kernels (C1): unset, by feature width (engine.kernels.row_tiles);
    # a value pins that tile for every width.
    _option("WIDTH_ROW_TILES", "bool", True, "kernel", "Row tiles of the residual-add, line-envelope, directed-pair and indexed-SiLU forward kernels by feature width; 0 restores the former fixed tiles (kept for one version).", "engine.kernels.row_tiles", "_WIDTH_ROW_TILES"),
    _option("DECOMPOSED_INDEXED_SILU_BM", "int", None, "kernel", "Indexed SiLU forward row tile; unset: by width.", "engine.kernels.decomposed_indexed_cat_silu_linear", "_EPILOGUE_BM", minimum=1, power_of_two=True),
    _option("DECOMPOSED_INDEXED_SILU_BWD_BM", "int", 8, "kernel", "Indexed SiLU backward row tile.", "engine.kernels.decomposed_indexed_cat_silu_linear", "_EPILOGUE_BWD_BM", minimum=1, power_of_two=True),
    _option("DECOMPOSED_INDEXED_SORTED_A_PAIR_BWD", "bool", True, "kernel", "Use sorted paired-index backward.", "engine.kernels.decomposed_indexed_cat_silu_linear", "_SORTED_A_PAIR_BWD"),
    _option("DECOMPOSED_INDEXED_GEMM_BACKEND", "str", "torch", "kernel", "Dense GEMM backend for decomposed projections.", "engine.kernels.decomposed_indexed_cat_silu_linear", "_GEMM_BACKEND", choices=("torch",)),
    _option("DECOMPOSED_INDEXED_DEBUG_CHECKS", "bool", False, "kernel", "Expensive index checks and debug messages."),
    _option("GATED_LN_2D", "bool", True, "kernel", "Gated LayerNorm with 2-D row tiles; 0 restores the 1-row kernels (kept for one version).", "models.matris.op.triton_gated_ln", "_TILE_2D"),
    _option("FUSED_RESIDUAL_ADD_BM", "int", None, "kernel", "Residual-add row tile; unset: by width.", "models.matris.op.triton_residual_add", "_BM", minimum=1, power_of_two=True),
    _option("FUSED_RESIDUAL_ADD_BD", "int", 128, "kernel", "Residual-add feature tile.", "models.matris.op.triton_residual_add", "_BD", minimum=1),
    _option("THREEBODY_FOURIER_BM", "int", 128, "kernel", "Three-body Fourier row tile.", "models.matris.op.triton_threebody_fourier", "_BM", minimum=1, power_of_two=True),
    _option("LINE_ENVELOPE_BM", "int", None, "kernel", "Line-envelope row tile; unset: by width.", "models.matris.op.triton_line_envelope", "_BM", minimum=1, power_of_two=True),
    _option("LINE_ENVELOPE_BD", "int", 128, "kernel", "Line-envelope feature tile.", "models.matris.op.triton_line_envelope", "_BD", minimum=1),
    _option("DIRECTED_PAIR_AGG_BM", "int", None, "kernel", "Directed-pair aggregation row tile; unset: by width.", "models.matris.op.triton_directed_pair_aggregate", "_PAIR_AGG_BM", minimum=1, power_of_two=True),
    _option("DIRECTED_PAIR_AGG_BD", "int", 128, "kernel", "Directed-pair aggregation feature tile.", "models.matris.op.triton_directed_pair_aggregate", "_PAIR_AGG_BD", minimum=1),
    _option("FUSED_SEGMENT_ATTENTION_NUM_WARPS", "int", None, "kernel", "Warps per segmented-attention program; unset: by width, one warp per 32 columns of the padded row, at most 4 (C10: a 32-wide row takes one warp; 4 for every width, the default before, kept for one version).", "models.matris.op.triton_segment_attention", "_NUM_WARPS", choices=(1, 2, 4, 8, 16, 32)),
    _option("FUSED_PAIRED_SEGMENT_ATTENTION_BWD_BM", "int", 8, "kernel", "Paired attention backward row tile.", "models.matris.op.triton_segment_attention", "_PAIRED_BWD_BM", minimum=1, power_of_two=True),
    _option("FUSED_PAIRED_SEGMENT_ATTENTION_BWD_BD", "int", 128, "kernel", "Paired attention backward feature tile.", "models.matris.op.triton_segment_attention", "_PAIRED_BWD_BD", minimum=1),
    _option("WHOLE_STEP_TIER_CACHE", "bool", False, "capacity", "Keep multiple whole-step capacity tiers.", "interfaces.gpu_md", "_WHOLE_STEP_TIER_CACHE"),
    _option("WHOLE_STEP_MAX_TIERS", "int", 3, "capacity", "Maximum cached whole-step capacity tiers.", "interfaces.gpu_md", "_WHOLE_STEP_MAX_TIERS", minimum=1),
    _option("TIER_HYSTERESIS_WINDOW", "int", 2000, "capacity", "Demand history length for tier hysteresis (GPU MD and the P>1 TieredCapacity).", "interfaces.gpu_md", "_TIER_HYSTERESIS_WINDOW", minimum=1),
    _option("TIER_HYSTERESIS_SIGMA", "float", 3.0, "capacity", "Demand standard-deviation margin for tier sizing (GPU MD and the P>1 TieredCapacity).", "interfaces.gpu_md", "_TIER_HYSTERESIS_SIGMA", minimum=0),
    _option("TIER_CHECK_INTERVAL", "int", 256, "capacity", "Replays between tier shrink checks (GPU MD and the P>1 TieredCapacity).", "interfaces.gpu_md", "_TIER_CHECK_INTERVAL", minimum=1),
    _option("TIER_DOWN_COOLDOWN", "int", 4000, "capacity", "Shrink cooldown; legacy unset default is twice hysteresis window (GPU MD and the P>1 TieredCapacity).", "interfaces.gpu_md", "_TIER_DOWN_COOLDOWN", minimum=0),
    _option("TIER_GROWTH_HEADROOM", "float", 0.03125, "capacity", "P>1 TieredCapacity: an overflow grows each bucket to the tier of its window high-water mark plus max(sigma x std, this fraction of it); the eager topology epoch uses 1/128.", minimum=0),
    _option("TIER_MIN_FREE_GIB", "float", 6.0, "capacity", "Free-memory floor for capacity-tier captures in GiB.", "interfaces.gpu_md", "_TIER_MIN_FREE_GIB", minimum=0),
    _option("CAPACITY_THERMAL_FACTOR", "float", 1.0, "capacity", "Initial capacity multiplier for thermal expansion.", "interfaces.gpu_md", "_CAPACITY_THERMAL_FACTOR", minimum=1),
    _option("LINE_GRAPH_TRITON", "bool", True, "graph", "Use Triton line-graph construction.", "graph.gpu_graph_builder", "_LINE_GRAPH_TRITON"),
    _option("LINE_GRAPH_TRITON_BLOCK", "int", 256, "graph", "Line-graph builder block size.", "graph.gpu_graph_builder", "_LINE_GRAPH_TRITON_BLOCK", minimum=1),
    _option("PRESORTED_BUILD", "bool", True, "graph", "Use presorted graph construction.", "graph.gpu_graph_builder", "_PRESORTED_BUILD"),
    _option("NARROW_SORT_KEYS", "bool", True, "graph", "Use narrow sort keys when bounds permit."),
    _option("TOPOLOGY_VERIFY", "str", "builder", "graph", "Topology verification strength.", choices=("builder", "sampled", "exhaustive")),
    # Step C7 debug: InferenceConfig.topology_reuse="skin" checker (0 = off).
    _option("TOPOLOGY_REUSE_CHECK", "int", 0, "graph", "Every N-th skin-reuse evaluation also rebuilds its topology and raises on a difference (0: off).", minimum=0),
    _option("M3GNET_INDEXED_AFFINE", "bool", True, "integration", "Enable M3GNet indexed-affine optimization."),
    _option("M3GNET_INDEXED_EDGE", "bool", True, "integration", "Enable M3GNet edge indexed-affine path."),
    _option("M3GNET_INDEXED_NODE", "bool", True, "integration", "Enable M3GNet node indexed-affine path."),
]
OPTION_REGISTRY: Mapping[str, OptionSpec] = MappingProxyType({spec.name: spec for spec in _OPTIONS})
# The CuTeDSL GEMM backends were removed on 2026-09-28. Selecting one is an
# error (in every profile); their former tuning knobs warn and are ignored.
_REMOVED_CHOICES = MappingProxyType({
    "MATRIS_FUSED_INDEXED_CAT_LINEAR_BACKEND": ("cutedsl_dense_silu",),
    "MATRIS_DECOMPOSED_INDEXED_GEMM_BACKEND": ("cutedsl",),
})
REMOVED_ENVIRONMENT = frozenset({
    "MATRIS_DECOMPOSED_INDEXED_CUTEDSL_FALLBACK",
    "MATRIS_CUTEDSL_SILU_SPLIT_BM", "MATRIS_CUTEDSL_SILU_TUNE_WARMUP", "MATRIS_CUTEDSL_SILU_TUNE_REPEAT",
    "MATRIS_CUTEDSL_SILU_EPILOGUE", "MATRIS_CUTEDSL_USE_TVM_FFI", "MATRIS_CUTEDSL_MMA_TILER_MN",
    "MATRIS_CUTEDSL_CLUSTER_SHAPE_MN", "MATRIS_CUTEDSL_USE_2CTA", "MATRIS_CUTEDSL_SWIZZLE_SIZE",
    "MATRIS_CUTEDSL_RASTER_ALONG", "MATRIS_CUTEDSL_AUTOTUNE_SCHEDULER", "MATRIS_CUTEDSL_AUTOTUNE",
    "MATRIS_CUTEDSL_AUTOTUNE_EXHAUSTIVE", "MATRIS_CUTEDSL_TUNE_WARMUP", "MATRIS_CUTEDSL_TUNE_REPEAT",
    "MATRIS_CUTEDSL_TUNE_TIMER", "MATRIS_CUTEDSL_TUNE_CUPTI_STRICT", "MATRIS_CUTEDSL_TUNE_CUPTI_TIMEOUT_S",
})
OBSOLETE_ENVIRONMENT = MappingProxyType({
    "MATRIS_CUDA_GRAPH_U_STEP": "CapacityConfig(u_step=...)",
    "MATRIS_CUDA_GRAPH_T_STEP": "CapacityConfig(t_step=...)",
    "MATRIS_WHOLE_STEP_CUDA_GRAPH": "InferenceConfig(execution='whole_step')",
})
# External benchmark metadata/control, not model configuration.
_EXTERNAL_ENVIRONMENT = frozenset({
    "MATRIS_COMPILE_CPP_GUARDS", "MATRIS_COMPILE_GUARD_NN_MODULES",
    "MATRIS_COMPILE_SKIP_PROCESS_GRAPHS", "MATRIS_COMPILE_PLAIN_ACTIVATIONS",
    "MATRIS_COMPILE_SCOPE",
    "MATRIS_COMPILE_FULLGRAPH_PROBE", "MATRIS_BENCHMARK_PHYSICAL_GPU", "MATRIS_BENCHMARK_NUMA_NODE",
    "MATRIS_REPOSITORY", "MATRIS_MODEL_PATH", "MATRIS_BASELINE_REPOSITORY",
    # optional test assets (tests/matris_test_assets.py) and reference tree
    # (tests/test_gpu_nhc.py)
    "MATRIS_TEST_CHECKPOINT", "MATRIS_TEST_S01_FRAMES", "MATRIS_TEST_REFERENCE_TREE",
    "MATRIS_TEST_ASSETS",
    # the ORB checkpoint of the ORB adapter tests and CI lane (tests/adapters/test_orb_adapter.py)
    "MATRIS_ORB_CHECKPOINT",
    # distributed launcher CPU placement (fastmd/distributed/affinity.py)
    "MATRIS_DISTRIBUTED_CPU_AFFINITY", "MATRIS_DISTRIBUTED_CPU_AFFINITY_REEXEC",
    # test-only fault injection (fastmd/distributed/watchdog.py)
    "MATRIS_DIST_FAULT_INJECT",
    # distributed device-memory admission (matris/engine/memory_admission.py) and the default of
    # InferenceConfig.recompute_policy (not a kernel option: configuration fingerprints stay)
    "MATRIS_MEMORY_ADMISSION", "MATRIS_RECOMPUTE_POLICY",
    # capture policy knob outside the kernel options (matris/engine/capacity_policy.py): the
    # warmup count of a runner's later captures (the P>1 growth band is the registered
    # MATRIS_TIER_GROWTH_HEADROOM)
    "MATRIS_RECAPTURE_WARMUP",
    # removed operator dispatch engine switch (fastmd/engine/dispatch.py, step 2.6):
    # "unified" is a no-op, any other value is rejected at import
    "MATRIS_DISPATCH_ENGINE",
    # removed forward runtime switch (fastmd/engine/runtime.py, step 4.2):
    # "unified" is a no-op, any other value is rejected at import
    "MATRIS_RUNTIME_ENGINE",
    # bond_partial boundary-replica checker (matris/models/matris/distributed/bond_partial.py); diagnostic
    "MATRIS_A2_CHECK_REPLICAS",
    # opt-in test tiers (tests/matris_test_tiers.py); MATRIS_RUN_PERF_TESTS is
    # the step-3.1a spelling of MATRIS_RUN_PERF_TIER
    "MATRIS_RUN_MULTI_GPU_TESTS", "MATRIS_RUN_LARGE_ROW_TESTS", "MATRIS_RUN_PERF_TIER",
    "MATRIS_RUN_PERF_TESTS", "MATRIS_RUN_OP_SELECTION_SNAPSHOTS",
})

_PROFILE_SWITCHES = (
    "FUSED_WEIGHTED_SEGMENT_EAGER", "FUSED_GATHER_CAT_EAGER", "FUSED_SEGMENT_SOFTMAX",
    "FUSED_INDEXED_CAT_LINEAR", "FUSED_DIRECTED_PAIR_AGGREGATE", "FUSED_DIRECTED_PAIR_EXPAND",
    "FUSED_SEGMENT_ATTENTION", "FUSED_SEGMENT_ATTENTION_EAGER", "FUSED_PAIRED_SEGMENT_ATTENTION",
    "FUSED_LINE_ENVELOPE", "FUSED_LINE_ENVELOPE_EAGER", "FUSED_THREEBODY_LINEAR",
    "FUSED_THREEBODY_LINEAR_EAGER", "FUSED_THREEBODY_BASIS", "FUSED_THREEBODY_BASIS_EAGER",
    "UNDIRECTED_EDGE_INIT", "FUSED_RESIDUAL_ADD", "FUSED_GATED_LN_RESIDUAL", "CSR_SEGMENT_SUM",
    "TRITON_ROW_GEMM", "FUSED_EDGE_FFN", "TRITON_WIDE_GEMM",
)
# Every option that selects a fused or custom kernel, plus the profiles' other
# optimization switches (undirected edge init), at the reference value, i.e.
# the generic profile. Training (MatRIS.forward(is_training=True)) runs in this
# state: fused kernels are inference-only; several have a non-differentiable backward.
PLAIN_TORCH_OPTIONS = MappingProxyType({
    **{"MATRIS_" + suffix: False for suffix in _PROFILE_SWITCHES},
    "MATRIS_GATED_LN_FUSE_MIN_ROWS": 1_000_000_000,
    "MATRIS_FUSED_INDEXED_CAT_LINEAR": False,
    "MATRIS_MIXED_WIDTH_CAT": False,
    "MATRIS_GATED_LN_BIAS_FOLD": False,
    "MATRIS_FUSED_GATED_TAIL": False,
    "MATRIS_MERGED_ATTENTION_PROJECTIONS": False,
    "MATRIS_MERGED_REFINEMENT_PROJECTION": False,
    "MATRIS_COMPILED_LOWERINGS": False,
})
# torch.set_float32_matmul_precision values; see fastmd.precision.
MATMUL_PRECISIONS = ("highest", "high", "medium")
# InferenceConfig.saved_tensor_recompute (fastmd.engine.saved_recompute).
SAVED_TENSOR_RECOMPUTE = ("off", "on", "deep", "auto")
# InferenceConfig.recompute_policy (fastmd.engine.saved_recompute.policy_threshold).
RECOMPUTE_POLICY = ("safe", "fast")
TOPOLOGY_REUSE = ("off", "skin", "auto")
# InferenceConfig.triton_dot_precision (fastmd.precision.triton_dot_input_precision).
TRITON_DOT_PRECISIONS = ("auto", "ieee", "tf32x3", "tf32")
# InferenceConfig.optimization_profile=None (2026-09-30; "legacy" before).
DEFAULT_PROFILE = "topology"
_ACTIVE: ContextVar[ResolvedConfig | None] = ContextVar("matris_inference_config", default=None)
_SCOPE_LOCK = threading.RLock()


def current_config() -> ResolvedConfig | None:
    """Return the active configuration without importing an inference backend."""
    return _ACTIVE.get()


def current_options_environment() -> dict[str, str]:
    """Export active options for a child process, or legacy environment defaults.

    Callers should first remove all OPTION_REGISTRY keys from the inherited
    child environment: optional None-valued automatic choices are omitted.
    """
    active = _ACTIVE.get()
    options = active.kernel_options if active is not None else _kernel_options("legacy", {})
    return {
        name: formatted for name, value in options.items()
        if (formatted := OPTION_REGISTRY[name].format(value)) is not None
    }


def env_value(name: str, default: Any = None) -> str | None:
    """Read a validated normalized option, respecting the active inference scope.

    Registered options always use their registry default, not a duplicated
    caller default. Unknown names retain ``os.getenv`` fallback semantics.
    The cooldown's default tracks twice the selected hysteresis window.
    """
    spec = OPTION_REGISTRY.get(name)
    active = _ACTIVE.get()
    if active is not None and name in active.kernel_options:
        value = active.kernel_options[name]
    elif spec is not None and name not in os.environ:
        if name == "MATRIS_TIER_DOWN_COOLDOWN":
            value = 2 * int(env_value("MATRIS_TIER_HYSTERESIS_WINDOW"))
        else:
            value = spec.default
    else:
        value = os.environ.get(name, default)
    if value is None:
        return None
    if spec is None:
        return str(value)
    return spec.format(spec.parse(value))


def _warn_environment(profile: str) -> None:
    for name in sorted(os.environ):
        if name in OBSOLETE_ENVIRONMENT:
            warnings.warn(f"{name} is obsolete and has no effect; use {OBSOLETE_ENVIRONMENT[name]}.", ConfigurationWarning, stacklevel=3)
        elif name in REMOVED_ENVIRONMENT:
            warnings.warn(f"{name} was removed with the CuTeDSL backends and is ignored.", ConfigurationWarning, stacklevel=3)
        elif name.startswith("MATRIS_") and name not in OPTION_REGISTRY and name not in _EXTERNAL_ENVIRONMENT:
            warnings.warn(f"Unknown MatRIS environment option {name}; it has no registered effect.", ConfigurationWarning, stacklevel=3)
    if profile != "legacy":
        ignored = sorted(name for name in OPTION_REGISTRY if name in os.environ
                         and name not in RUNTIME_ENVIRONMENT and name != "MATRIS_TRITON_DOT_PRECISION")
        if ignored:
            warnings.warn(
                f"optimization_profile={profile!r} ignores registered environment options: "
                + ", ".join(ignored) + "; use expert_overrides for intentional changes.",
                ConfigurationWarning, stacklevel=3,
            )


def _kernel_options(profile: str, overrides: Mapping[str, Any]) -> dict[str, Any]:
    if profile not in {"legacy", "topology", "generic"}:
        raise ValueError("optimization_profile must be 'legacy', 'topology' or 'generic'")
    for name, removed in _REMOVED_CHOICES.items():
        # Raised in every profile (also for profile_env), although topology/generic ignore the variable.
        if os.environ.get(name, "").strip().lower() in removed:
            OPTION_REGISTRY[name].parse(os.environ[name])
    options = {name: spec.default for name, spec in OPTION_REGISTRY.items()}
    if profile == "legacy":
        for name, spec in OPTION_REGISTRY.items():
            if name in os.environ:
                options[name] = spec.parse(os.environ[name])
    else:
        options.update({"MATRIS_" + suffix: profile == "topology" for suffix in _PROFILE_SWITCHES})
        options["MATRIS_GATED_LN_FUSE_MIN_ROWS"] = 10000 if profile == "topology" else 1_000_000_000
        for name in RUNTIME_ENVIRONMENT:
            if name in os.environ:
                options[name] = OPTION_REGISTRY[name].parse(os.environ[name])
    for name, value in overrides.items():
        if name not in OPTION_REGISTRY:
            replacement = OBSOLETE_ENVIRONMENT.get(name)
            hint = f"; use {replacement}" if replacement else ""
            raise ValueError(f"Unknown expert override {name!r}{hint}")
        options[name] = OPTION_REGISTRY[name].parse(value)
    cooldown = "MATRIS_TIER_DOWN_COOLDOWN"
    if cooldown not in overrides and not (profile == "legacy" and cooldown in os.environ):
        options[cooldown] = 2 * options["MATRIS_TIER_HYSTERESIS_WINDOW"]
    return options


def profile_env(profile: str) -> dict[str, str]:
    """Normalized profile values for subprocesses and legacy benchmark scripts.

    Optional auto-valued settings are omitted; callers configuring a fresh
    subprocess should remove inherited registered MATRIS variables first.
    ``legacy`` intentionally reflects the current process environment.
    """
    return {
        name: formatted for name, value in _kernel_options(profile, {}).items()
        if (formatted := OPTION_REGISTRY[name].format(value)) is not None
    }


@dataclass(frozen=True)
class CapacityConfig:
    """Graph capacity and capture setup (counts, except skin in angstrom).

    ``warmup``: side-stream warmups before a graph runner's first capture; its
    later captures run ``MATRIS_RECAPTURE_WARMUP`` (1;
    ``fastmd.engine.capacity_policy.CapturePolicy``).
    """
    u_step: int = 512
    t_step: int = 8192
    n_dummy: int = 64
    min_pad_u: int = 256
    min_pad_t: int = 4096
    warmup: int = 3
    candidate_skin: float = 2.0

    def __post_init__(self) -> None:
        for name in ("u_step", "t_step", "n_dummy", "min_pad_u", "min_pad_t", "warmup"):
            minimum = 1 if name in {"u_step", "t_step", "n_dummy"} else 0
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"capacity.{name} must be an integer >= {minimum}")
        skin = self.candidate_skin
        if isinstance(skin, bool) or not isinstance(skin, (int, float)) or not math.isfinite(skin) or skin < 0:
            raise ValueError("capacity.candidate_skin must be finite and >= 0 angstrom")


# Layer checkpointing is superseded for inference by saved_tensor_recompute
# (less memory at less time, sized by the memory admission model). The model's
# MatRIS(enable_checkpoint=...) stays the training / fine-tuning lever.
CHECKPOINT_DEPRECATION = (
    "InferenceConfig(checkpoint={!r}) is deprecated and will be removed in the next version: use "
    "saved_tensor_recompute ('on', 'deep' or 'auto', the default) to reduce inference memory. Layer "
    "checkpointing stays available for training and fine-tuning through MatRIS(enable_checkpoint=...)."
)
_checkpoint_deprecation_warned = False


def _warn_checkpoint_deprecated(checkpoint: str) -> None:
    """Warn once per process that ``checkpoint`` 'on'/'auto' is deprecated."""
    global _checkpoint_deprecation_warned
    if checkpoint == "off" or _checkpoint_deprecation_warned:
        return
    _checkpoint_deprecation_warned = True
    warnings.warn(CHECKPOINT_DEPRECATION.format(checkpoint), ConfigurationWarning, stacklevel=4)


@dataclass(frozen=True)
class InferenceConfig:
    """Public inference controls; use expert overrides only for kernel research.

    ``optimization_profile`` selects the kernels everywhere (single GPU and
    P>1): "topology" (default; None) = the fused kernels (the P>1 "native" local
    operators), "generic" = plain torch reference operators (P>1
    "torch_reference"), "legacy" = registry defaults plus MATRIS_* environment
    options (single GPU only; the pre-2026-09-30 default).
    ``execution`` "auto" (the default of the single-GPU entry points, also
    ``None``) takes the fastest mode that applies: MLIPCalculator (and the
    ASE MD / optimizer drivers) model_graph, GPU MD whole_step, falling back to
    eager on CPU, with checkpointing, and after a CUDA out-of-memory error in a
    graph capture; TorchSim eager. Explicit values are kept as given.
    ``checkpoint`` 'on'/'auto' is deprecated for inference (a ConfigurationWarning,
    once per process) and will be removed in the next version: use
    ``saved_tensor_recompute``. ``checkpoint='auto'`` uses the model's existing
    row-count threshold policy; it is not a memory-budget-based automatic selection.
    ``saved_tensor_recompute`` (inference only) rebuilds the cheap saved
    activations of the interaction blocks in the backward instead of keeping
    them: "on" about halves the peak memory for about +10% time; "deep" also
    rebuilds the first-layer outputs (about a third of "off" for about +20%).
    Energies are bitwise those of "off", forces within the FP32 noise floor.
    "auto" (the default; None reads ``MATRIS_SAVED_TENSOR_RECOMPUTE``) picks
    the cheapest level (off, on, deep) whose memory admission prediction
    stays within a share of the device budget, deciding at the first evaluation
    and at capture/growth boundaries; "on"/"deep" exclude checkpointing.
    ``recompute_policy`` sets that share for the memory decisions (the
    recompute ladder, C3 and C2 auto): "safe" 85%; "fast" (the default; None
    reads ``MATRIS_RECOMPUTE_POLICY``, else "fast") 98% at P=1 eager, 95% at
    P=1 model_graph and on the P>1 topology epoch. Where
    "fast" keeps a cheaper level than "safe" would at P=1, it switches the
    process's CUDA caching allocator to expandable segments (unless
    ``PYTORCH_CUDA_ALLOC_CONF`` configures the allocator; without them
    "fast" decides as "safe" there), which removes the fragmentation growth
    of the reserved memory in long MD runs, and it falls back to "safe" if
    an evaluation leaves less than the margin free. Elsewhere the allocator
    stays unchanged: ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
    removes the fragmentation growth of long eager MD runs at any level for
    about +1% eager step time. "fast" applies only to a
    decision whose predictions all come from a measured table of the memory
    admission model (``memory_admission.memory_model`` and
    ``entry_calibrated``); a "derived" or "uncalibrated" model, or an entry
    its architecture was not measured with, decides as "safe".
    ``topology_reuse`` (the single-GPU calculator): "skin" keeps the graph
    topology across calls within a skin epoch (candidates within cutoff +
    ``capacity.candidate_skin``, filtered exactly on the device every call,
    refreshed before a call in which an atom moved 0.45 of the skin), so a call has no host
    synchronization before its single result readback; "off" rebuilds the
    graph every call; "auto" (default) is "skin" for execution="model_graph"
    where it applies (CUDA, periodic in all directions, an unchanged cell)
    and "off" elsewhere (eager gains 1-2%).
    ``matmul_precision`` applies ``torch.set_float32_matmul_precision`` only
    inside the entry point's calls, restoring the caller's flags afterwards;
    None leaves torch's process state (e.g. an NGC TF32 default) unchanged.
    ``triton_dot_precision`` is the input precision of the Triton FP32 dots
    (the wide 3xTF32 GEMMs of the frozen projections with K, N >= 64, the C4
    row GEMMs and edge FFN, the C2 gated tail, the recompute's gate input
    rebuild), inside the entry point's calls: "auto" 3xTF32 where cuBLAS runs
    IEEE FP32 (FP32-level accuracy) and TF32 where it runs TF32 (the wide
    GEMMs then stay cuBLAS TF32); "ieee" cuBLAS for the wide GEMMs and FP32
    FMA dots elsewhere; "tf32x3" or "tf32" whatever cuBLAS runs (see
    ``fastmd.precision.triton_dot_input_precision``: where cuBLAS computes
    the requested precision itself and is faster, or for the rebuild under
    TF32 cuBLAS, a site keeps its cuBLAS path). None: the
    ``MATRIS_TRITON_DOT_PRECISION`` expert override, else the value of an
    outer configured scope, else the environment variable of that name
    (every profile), else "auto". The projection gradient's row addmm
    always accumulates in FP32 FMA.
    No explicit configuration changes any process-wide environment variable.
    """
    execution: str | None = None
    optimization_profile: str | None = None
    checkpoint: str = "off"
    isolated_atoms: str = "reference"
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    expert_overrides: Mapping[str, Any] = field(default_factory=dict)
    report: bool = True
    matmul_precision: str | None = None
    saved_tensor_recompute: str | None = None
    topology_reuse: str = "auto"
    triton_dot_precision: str | None = None
    recompute_policy: str | None = None

    def __post_init__(self) -> None:
        if self.triton_dot_precision is not None and self.triton_dot_precision not in TRITON_DOT_PRECISIONS:
            raise ValueError(f"triton_dot_precision must be None or one of {TRITON_DOT_PRECISIONS}")
        if self.topology_reuse not in TOPOLOGY_REUSE:
            raise ValueError(f"topology_reuse must be one of {TOPOLOGY_REUSE}")
        if self.execution not in {None, "auto", "eager", "model_graph", "whole_step"}:
            raise ValueError("execution must be None, 'auto', 'eager', 'model_graph' or 'whole_step'")
        if self.optimization_profile not in {None, "legacy", "topology", "generic"}:
            raise ValueError("optimization_profile must be None (topology), 'legacy', 'topology' or 'generic'")
        if self.checkpoint not in {"off", "on", "auto"}:
            raise ValueError("checkpoint must be 'off', 'on' or 'auto'")
        _warn_checkpoint_deprecated(self.checkpoint)
        if self.saved_tensor_recompute is not None and self.saved_tensor_recompute not in SAVED_TENSOR_RECOMPUTE:
            raise ValueError(f"saved_tensor_recompute must be None or one of {SAVED_TENSOR_RECOMPUTE}")
        if self.recompute_policy is not None and self.recompute_policy not in RECOMPUTE_POLICY:
            raise ValueError(f"recompute_policy must be None or one of {RECOMPUTE_POLICY}")
        if self.saved_tensor_recompute in ("on", "deep") and self.checkpoint != "off":
            raise ValueError(f"saved_tensor_recompute={self.saved_tensor_recompute!r} excludes checkpoint: "
                             "use checkpoint='off'")
        if self.isolated_atoms not in {"reference", "error"}:
            raise ValueError("isolated_atoms must be 'reference' or 'error'")
        if not isinstance(self.capacity, CapacityConfig):
            raise TypeError("capacity must be a CapacityConfig")
        if not isinstance(self.report, bool):
            raise TypeError("report must be a bool")
        if self.matmul_precision is not None and self.matmul_precision not in MATMUL_PRECISIONS:
            raise ValueError("matmul_precision must be None, 'highest', 'high' or 'medium'")
        parsed = {}
        for name, value in self.expert_overrides.items():
            if name in REMOVED_ENVIRONMENT:
                warnings.warn(f"{name} was removed with the CuTeDSL backends and is ignored.", ConfigurationWarning, stacklevel=3)
                continue
            if name not in OPTION_REGISTRY:
                replacement = OBSOLETE_ENVIRONMENT.get(name)
                hint = f"; use {replacement}" if replacement else ""
                raise ValueError(f"Unknown expert override {name!r}{hint}")
            parsed[name] = OPTION_REGISTRY[name].parse(value)
        object.__setattr__(self, "expert_overrides", MappingProxyType(parsed))


@dataclass(frozen=True)
class ResolvedConfig:
    execution: str
    optimization_profile: str
    checkpoint_enabled: bool | None
    handle_isolated_atoms: bool
    capacity: CapacityConfig
    kernel_options: Mapping[str, Any]
    report: bool = True
    matmul_precision: str | None = None
    saved_tensor_recompute: str = "off"
    topology_reuse: str = "auto"
    triton_dot_precision: str | None = None
    recompute_policy: str = "fast"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel_options", MappingProxyType(dict(self.kernel_options)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution": self.execution,
            "optimization_profile": self.optimization_profile,
            "checkpoint": "auto" if self.checkpoint_enabled is None else ("on" if self.checkpoint_enabled else "off"),
            "isolated_atoms": "reference" if self.handle_isolated_atoms else "error",
            "capacity": asdict(self.capacity),
            "kernel_options": dict(self.kernel_options),
            "report": self.report,
            "matmul_precision": self.matmul_precision,
            "saved_tensor_recompute": self.saved_tensor_recompute,
            "topology_reuse": self.topology_reuse,
            "triton_dot_precision": self.triton_dot_precision,
            # Only when not the default: configuration fingerprints (and records) of the default stay.
            **({} if self.recompute_policy == "fast" else {"recompute_policy": self.recompute_policy}),
        }

    @cached_property
    def fingerprint(self) -> str:
        payload = self.as_dict()
        payload.pop("report")
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    @contextmanager
    def scope(self) -> Iterator[ResolvedConfig]:
        """Temporarily bind options, restoring both old and lazy-imported modules.

        The lock is deliberately held over the complete inference operation:
        module globals are shared even though dynamic readers use ContextVar.
        Nested reuse of the identical resolved configuration is inexpensive.
        A requested matmul precision and Triton dot precision are applied and
        restored the same way.
        """
        with _SCOPE_LOCK:
            outer = _ACTIVE.get()
            if outer is self:
                yield self
                return
            # Baseline values for modules imported lazily during this scope.
            # Ignored malformed ENV in an explicit reproducible profile must
            # not make cleanup fail or prevent already-loaded globals restoring.
            fallback = dict(outer.kernel_options) if outer is not None else {
                name: spec.default for name, spec in OPTION_REGISTRY.items()
            }
            if outer is None:
                for name, spec in OPTION_REGISTRY.items():
                    if name in os.environ:
                        try:
                            fallback[name] = spec.parse(os.environ[name])
                        except ValueError:
                            pass
                if "MATRIS_TIER_DOWN_COOLDOWN" not in os.environ:
                    fallback["MATRIS_TIER_DOWN_COOLDOWN"] = 2 * fallback["MATRIS_TIER_HYSTERESIS_WINDOW"]
            saved: dict[tuple[str, str], Any] = {}
            for name, spec in OPTION_REGISTRY.items():
                for module_name, attribute in spec.bindings:
                    module = sys.modules.get(module_name)
                    if module is not None and hasattr(module, attribute):
                        saved[module_name, attribute] = getattr(module, attribute)
                        setattr(module, attribute, self.kernel_options[name])
            token = _ACTIVE.set(self)
            try:
                if self.matmul_precision is None and self.triton_dot_precision is None:
                    yield self
                else:
                    # Only a configured precision needs torch; keep imports lazy.
                    from fastmd.precision import matmul_precision_scope
                    from fastmd.precision import triton_dot_precision_scope
                    with matmul_precision_scope(self.matmul_precision), \
                            triton_dot_precision_scope(self.triton_dot_precision):
                        yield self
            finally:
                _ACTIVE.reset(token)
                for name, spec in OPTION_REGISTRY.items():
                    for module_name, attribute in spec.bindings:
                        module = sys.modules.get(module_name)
                        if module is not None and hasattr(module, attribute):
                            setattr(module, attribute, saved.get((module_name, attribute), fallback[name]))


def resolve_config(
    config: InferenceConfig | None,
    *,
    default_execution: str = "eager",
    default_isolated_atoms: str = "reference",
) -> ResolvedConfig:
    """Resolve once at entry-point construction; values are then immutable.

    Entry points declare execution and isolated-atom defaults here. Omitted
    configuration disables checkpointing and uses those defaults. An outer
    scope supplies only its kernel profile/options and matmul precision to an
    unconfigured entry; explicit InferenceConfig is the sole interface for
    changing public controls.
    """
    if default_execution not in {"auto", "eager", "model_graph", "whole_step"}:
        raise ValueError("invalid default_execution")
    if default_isolated_atoms not in {"reference", "error"}:
        raise ValueError("default_isolated_atoms must be 'reference' or 'error'")
    if config is not None and not isinstance(config, InferenceConfig):
        raise TypeError("config must be an InferenceConfig or None")
    inherited = _ACTIVE.get() if config is None else None
    profile = ((inherited.optimization_profile if inherited else DEFAULT_PROFILE) if config is None
               else config.optimization_profile or DEFAULT_PROFILE)
    if inherited is None:
        _warn_environment(profile)
    recompute = (config.saved_tensor_recompute if config is not None and config.saved_tensor_recompute
                 is not None else env_value("MATRIS_SAVED_TENSOR_RECOMPUTE"))
    warmup = CapacityConfig().warmup if config is None else config.capacity.warmup
    if recompute in ("on", "deep") and config is not None and config.checkpoint != "off":
        # (an explicit value is rejected by InferenceConfig itself)
        raise ValueError(f"MATRIS_SAVED_TENSOR_RECOMPUTE={recompute!r} excludes checkpoint: use checkpoint='off'")
    if recompute in ("on", "deep") and warmup == 0:
        # A capture must not be the first run of the recompute-only kernels.
        raise ValueError(f"saved_tensor_recompute={recompute!r} requires capacity.warmup >= 1")
    if recompute == "auto" and (warmup == 0 or (config is not None and config.checkpoint != "off")):
        recompute = "off"  # auto never turns on without a warmup or under checkpointing
    return ResolvedConfig(
        execution=default_execution if config is None else (config.execution or default_execution),
        optimization_profile=profile,
        checkpoint_enabled=False if config is None else {"off": False, "on": True, "auto": None}[config.checkpoint],
        handle_isolated_atoms=(default_isolated_atoms if config is None else config.isolated_atoms) == "reference",
        capacity=CapacityConfig() if config is None else config.capacity,
        kernel_options=inherited.kernel_options if inherited is not None else _kernel_options(profile, {} if config is None else config.expert_overrides),
        report=False if config is None else config.report,
        # Kept like kernel_options for calls after the outer scope exits;
        # re-applying it inside that scope is idempotent.
        matmul_precision=getattr(inherited, "matmul_precision", None) if config is None else config.matmul_precision,
        saved_tensor_recompute=recompute,
        topology_reuse="auto" if config is None else config.topology_reuse,
        # Like matmul_precision; None leaves the outer (or process) value.
        triton_dot_precision=getattr(inherited, "triton_dot_precision", None) if config is None else (
            config.triton_dot_precision if config.triton_dot_precision is not None
            else config.expert_overrides.get("MATRIS_TRITON_DOT_PRECISION")),
        recompute_policy=(config.recompute_policy if config is not None and config.recompute_policy is not None
                          else getattr(inherited, "recompute_policy", None) or recompute_policy_default()),
    )


def recompute_policy_default() -> str:
    """``MATRIS_RECOMPUTE_POLICY`` (safe or fast), else "fast": the default of InferenceConfig.recompute_policy."""
    value = os.environ.get("MATRIS_RECOMPUTE_POLICY", "fast").strip().lower()
    if value not in RECOMPUTE_POLICY:
        raise ValueError(f"MATRIS_RECOMPUTE_POLICY must be one of {RECOMPUTE_POLICY}, got {value!r}")
    return value


def plain_torch_config() -> ResolvedConfig:
    """The active (or legacy) configuration with every fused kernel disabled.

    Only kernel options change; the precision of an outer scope is kept.
    """
    active = _ACTIVE.get()
    if active is None:
        # Legacy values, tolerating malformed environment entries like scope().
        options = {name: spec.default for name, spec in OPTION_REGISTRY.items()}
        for name, spec in OPTION_REGISTRY.items():
            if name in os.environ:
                try:
                    options[name] = spec.parse(os.environ[name])
                except ValueError:
                    pass
        if "MATRIS_TIER_DOWN_COOLDOWN" not in os.environ:
            options["MATRIS_TIER_DOWN_COOLDOWN"] = 2 * options["MATRIS_TIER_HYSTERESIS_WINDOW"]
        active = ResolvedConfig(execution="eager", optimization_profile="legacy",
                                checkpoint_enabled=False, handle_isolated_atoms=True,
                                capacity=CapacityConfig(), kernel_options=options)
    return replace(active, kernel_options={**active.kernel_options, **PLAIN_TORCH_OPTIONS}, report=False)


@contextmanager
def bind_options(options: Mapping[str, Any]) -> Iterator[None]:
    """Bind option values in loaded modules without taking the scope lock.

    Only for callers already serialized with every scope, e.g. activation
    checkpoint recomputation on an autograd thread while the scope owner waits.
    """
    saved = []
    for name, value in options.items():
        for module_name, attribute in OPTION_REGISTRY[name].bindings:
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, attribute):
                saved.append((module, attribute, getattr(module, attribute)))
                setattr(module, attribute, value)
    try:
        yield
    finally:
        for module, attribute, value in reversed(saved):
            setattr(module, attribute, value)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Print the typed expert-option registry as JSON")
    group.add_argument("--profile", choices=("legacy", "topology", "generic"), help="Print a normalized environment profile as JSON")
    args = parser.parse_args()
    data = [spec.as_dict() for spec in OPTION_REGISTRY.values()] if args.list else profile_env(args.profile)
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
