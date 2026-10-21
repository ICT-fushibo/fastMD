"""Single-device MatRIS memory estimates retained from the source calibration.

Other MLIPs use adapters.memory.MeasuredMemoryModel. No collective admission lives here.
"""
from __future__ import annotations

from dataclasses import astuple, dataclass, replace
import hashlib
import os

import torch

ENVIRONMENT = "MATRIS_MEMORY_ADMISSION"
_GIB = 1 << 30
# Relative slack on every prediction; the margin kept free is the larger of
# 2 GiB and 2% of the device (43,008 atoms on 2 GPUs eager peaked 1.5% under
# the device and OOMed at its third step from fragmentation).
_SLACK = 0.02
_MARGIN_BYTES = 2 * _GIB
_MARGIN_FRACTION = 0.02


@dataclass(frozen=True)
class Coefficients:
    base_gib: float
    gib_per_million_rows: float
    gib_per_million_line_rows: float = 0.0

    def predict(self, rows: int, line_rows: int = 0) -> int:
        gib = (self.base_gib + self.gib_per_million_rows * rows / 1e6
               + self.gib_per_million_line_rows * line_rows / 1e6)
        return int(max(gib, 0.0) * (1 + _SLACK) * _GIB)


# (engine, profile, execution[, recompute]) -> peak reserved; see the module docstring.
# Distributed: profile None = the native local backend, else the backend.
CALIBRATION = {
    ("single", "topology", "eager"): Coefficients(3.289, 10.913),
    ("single", "topology", "model_graph"): Coefficients(12.638, 11.122),
    ("single", "legacy", "eager"): Coefficients(0.212, 14.007),
    ("single", "legacy", "model_graph"): Coefficients(-2.423, 29.170),
    # The bond_partial exchange (no routed rows; rows: the installed capacity's triplet rows); see the docstring.
    # saved_tensor_recompute "on" (internal "narrow"); see the docstring.
    ("single", "topology", "eager", "narrow"): Coefficients(3.132, 5.497),
    ("single", "topology", "model_graph", "narrow"): Coefficients(6.967, 5.876),
    ("single", "legacy", "eager", "narrow"): Coefficients(2.074, 5.976),
    ("single", "legacy", "model_graph", "narrow"): Coefficients(10.767, 6.203),
    # saved_tensor_recompute "deep"; see the docstring.
    ("single", "topology", "eager", "deep"): Coefficients(0.239, 4.196),
    ("single", "topology", "model_graph", "deep"): Coefficients(5.348, 4.527),
    ("single", "legacy", "eager", "deep"): Coefficients(1.226, 4.420),
    ("single", "legacy", "model_graph", "deep"): Coefficients(9.756, 4.140),
}


# The entries whose peaks depend on the mixed-width indexed projections (step 2.5b D8), with D8
# inactive (the parent tree, GB200, 2026-09-29; same probe and fit, peak reserved GiB, triplet
# rows as in the tables above):
#   single topology  eager off 12.40 / 49.12 / 97.56 / OOM, model_graph off 25.49 / 49.99 / 99.84 / OOM,
#                    eager on 7.42 / 21.09 / 41.29 / 82.17 / 165.47, model_graph on 16.09 / 21.97 / 45.58 /
#                    90.96 / 181.70, eager deep 5.38 / 14.50 / 29.79 / 59.54 / 115.23, model_graph deep 11.06 /
#                    14.92 / 30.54 / 60.92 / 121.66 (the fifth: 43,008 atoms, 30,443,712 rows)
#   bond_partial     eager off 26.63 / 51.51 / 104.60 / OOM, model_graph off 28.16 / 54.26 / 109.99 / OOM,
#                    eager on 13.97 / 25.03 / 51.56 / 99.08, model_graph on 18.75 / 36.51 / 72.93 / 143.91,
#                    eager deep 10.32 / 19.04 / 37.58 / 74.93, model_graph deep 14.97 / 29.01 / 57.94 / 117.68
CALIBRATION_MIXED_WIDTH_OFF = {
    ("single", "topology", "eager"): Coefficients(6.913, 11.910),
    ("single", "topology", "model_graph"): Coefficients(14.768, 11.269),
    ("single", "topology", "eager", "narrow"): Coefficients(2.294, 5.384),
    ("single", "topology", "model_graph", "narrow"): Coefficients(10.594, 5.779),
    ("single", "topology", "eager", "deep"): Coefficients(2.380, 3.755),
    ("single", "topology", "model_graph", "deep"): Coefficients(7.388, 3.861),
}


# C3 (``MATRIS_MERGED_PROJECTION_SINGLE_BWD``, default auto): the merged attention projection's
# consumers write their gradients into one [rows, sum(splits)] buffer, allocated by the first of
# them in the backward (the segment attention: the alphas), so the p0 columns are held until the
# cat3 epilogue backward. The last block (whose backward runs first, before the caching allocator
# holds a free block of that size) keeps the per-output backward. Measured on GB200 (2026-09-29,
# same probe and sweep as above, peak reserved GiB, C3 on minus off; recompute =
# saved_tensor_recompute; every other point of the sweep: +0.00):
#   single topology eager, recompute deep         7,610,928 rows: +2.32 (0 at 1.0M, 3.8M, 15.2M)
#   single topology model_graph, recompute deep     951,366 rows: +0.46;  15,221,856 rows: +7.27
#   bond_partial eager, recompute on and deep     1.9M-15.2M rows: +0.48 GiB per M rows (P=2 and 4)
#   bond_partial model_graph, recompute on        1,902,732 rows: +0.94 (P=4); P=2 up to +1.80
#   bond_partial model_graph, recompute deep    1.9M-15.2M rows: +0.63 .. +7.49
# With the fused narrow tail (C2, ``MATRIS_FUSED_GATED_TAIL``) on, which changes the GatedMLP
# backward's allocations inside the buffer's claim window (C3 on minus C2 alone, same sweep):
#   single topology eager, recompute deep     1.0M / 3.8M / 7.6M rows: +0.46 / +1.82 / +1.81
#   bond_partial model_graph, recompute deep       3.8M / 15.2M rows: +1.88 / +7.49 (P=2)
# The increments below cover each such C3 peak, or at least C3's own share of it, plus 0.02; on the
# recalibrated coefficients (see the module docstring) every C3 peak of that sweep, P=2 and P=4 on the
# automatic and 4x1x1 grids, is within the prediction with them. They are added to the slope where C3 runs: always
# under "on", and under "auto" (the default) for its decision, which keeps C3 only while the
# prediction with it fits (``saved_recompute.AUTO_THRESHOLD``); the recompute ladder and the
# execution modes read the predictions without it.
SINGLE_BWD_GIB_PER_MILLION_ROWS = {
    ('single', 'topology', 'eager', 'deep'): 0.20,
    ('single', 'topology', 'model_graph', 'deep'): 0.23,
}


def single_bwd_mode(kernel_options) -> str:
    """C3 of a resolved configuration's kernel options: "off" without the merged attention
    projections and the indexed projection they need, else its switch ("on", "off" or "auto":
    on only while the prediction with it fits, see ``saved_recompute.AUTO_THRESHOLD``)."""
    if not (kernel_options.get("MATRIS_MERGED_ATTENTION_PROJECTIONS", True)
            and kernel_options.get("MATRIS_FUSED_INDEXED_CAT_LINEAR", False)):
        return "off"
    return kernel_options.get("MATRIS_MERGED_PROJECTION_SINGLE_BWD", "auto")


# C2 (``MATRIS_FUSED_GATED_TAIL``, default on): the fused GatedMLP narrow tail writes its linear outputs
# and gradients over dead tensors and lowers the allocated peak at every point of the sweep (up to
# -5.4 GiB), and the reserved peak at almost every point (up to -7.3 GiB). With less memory cached
# after the first evaluations, the caching allocator adds segments later at two points, where the
# reserved peak ends above the entry's prediction (GB200 2026-09-30, same probe and sweep; C2 on /
# off, GiB; off = main's peak):
#   single topology eager, recompute on         15,221,856 rows: 87.01 / 85.17 (predicted 86.88)
#   bond_partial model_graph, recompute on (P=4) 1,902,732 rows: 15.64 / 15.55 (predicted 15.58)
# (bond_partial eager on the epoch topology exceeded its prediction without C2, by up to 5.5%.) C2
# runs only while the prediction with its increment fits AUTO_THRESHOLD of the budget at the
# recompute level chosen without it (the P=1 entry point's state, auto or explicit level, or the
# evaluator's admission: ``gated_tail_fits``); the admission, the recompute ladder and the execution
# modes read the predictions without it, so the largest system per level is unchanged.
# Entries without an increment always run C2. The increments were 0.03 and 0.10 (the second on C7 and
# C3); the recalibration fitted every entry with C2 on (its peaks, with C3 off and on, are within the
# predictions without an increment), so none is needed and C2 always runs.
GATED_TAIL_GIB_PER_MILLION_ROWS = {}


def gated_tail_active(kernel_options) -> bool:
    """Whether a resolved configuration's kernel options run C2 (its switch and the indexed
    projection whose Cat3/Cat4 outputs it fuses)."""
    return bool(kernel_options.get("MATRIS_FUSED_GATED_TAIL", True)
                and kernel_options.get("MATRIS_FUSED_INDEXED_CAT_LINEAR", False))


def gated_tail_fits(engine: str, profile: str | None, execution: str, rows: int, line_rows: int, *,
                    recompute: str, mixed_width: bool, budget: int, single_bwd: bool = True,
                    memory: "MemoryModel | None" = None, threshold: float | None = None) -> bool:
    """C2's memory decision for one rank at a recompute level: no increment for the entry, or the
    prediction with it (and with C3's where ``single_bwd``) fits ``threshold`` (default
    ``saved_recompute.AUTO_THRESHOLD``, see ``saved_recompute.policy_threshold``) of ``budget``
    (uncalibrated: no)."""
    if not _gated_tail_increment(engine, profile, execution, recompute, memory):
        return True
    if threshold is None:
        from fastmd.engine.saved_recompute import AUTO_THRESHOLD as threshold
    peak = predict(engine, profile, execution, rows, line_rows, recompute=recompute, mixed_width=mixed_width,
                   single_bwd=single_bwd, gated_tail=True, memory=memory)
    return peak is not None and peak <= threshold * budget


def _gated_tail_increment(engine, profile, execution, recompute, memory=None) -> float:
    if memory is not None and memory.calibration != "rdx_1m":
        return 0.0  # the other tables were fitted with C2's path (no increment)
    if engine.startswith("distributed") and profile != "torch_reference":
        profile = None
    key = (engine, profile, execution) if recompute == "off" else (engine, profile, execution, recompute)
    return GATED_TAIL_GIB_PER_MILLION_ROWS.get(key, 0.0)


def mixed_width_active(kernel_options) -> bool:
    """Whether a resolved configuration's kernel options run D8 (the mixed-width switch
    and the indexed projection it needs): selects the calibration of the twelve D8 entries."""
    return bool(kernel_options.get("MATRIS_MIXED_WIDTH_CAT", True)
                and kernel_options.get("MATRIS_FUSED_INDEXED_CAT_LINEAR", False))


@dataclass(frozen=True)
class Architecture:
    """The hyperparameters of a MatRIS model that set its bytes per triplet row (``architecture``)."""
    num_layers: int
    node_feat_dim: int
    edge_feat_dim: int
    three_body_feat_dim: int
    mlp_hidden: int  # the widest MLP hidden layer
    num_radial: int
    num_angular: int
    max_n: int
    max_l: int
    pairwise_cutoff: float
    three_body_cutoff: float
    kinds: tuple = ("layer", "fourier", "gatemlp", False)  # norm, three-body expansion, conv MLP, smoothed delta
    dtype_bytes: int = 4

    def fingerprint(self) -> str:
        return hashlib.sha256(repr(astuple(self)).encode()).hexdigest()[:12]

    def within(self, other: "Architecture") -> bool:
        """Whether ``other``'s calibration bounds this model: no deeper or wider, no more pair edges per
        triplet row (pair cutoff at most, three-body cutoff at least), the same layer kinds."""
        sizes = ("num_layers", "node_feat_dim", "edge_feat_dim", "three_body_feat_dim", "mlp_hidden",
                 "num_radial", "num_angular", "max_n", "max_l", "dtype_bytes")
        return (all(getattr(self, size) <= getattr(other, size) for size in sizes) and self.kinds == other.kinds
                and self.pairwise_cutoff <= other.pairwise_cutoff and self.three_body_cutoff >= other.three_body_cutoff)


def architecture(model) -> Architecture | None:
    """A model's ``Architecture`` from its ``config``; None without one (``memory_model``: uncalibrated)."""
    config = getattr(model, "config", None)
    if not isinstance(config, dict):
        return None
    try:
        hidden = config["mlp_hidden_dims"]
        # As the energy head reads it: one width, or any iterable of widths (a list, tuple, range or array; a
        # string is one value).
        hidden = max([hidden] if isinstance(hidden, (int, str)) else list(hidden), default=0)
        kinds = tuple(value.lower() if isinstance(value, str) else bool(value) for value in (
            config["norm_type"], config["three_body_expansion"], config["graph_conv_mlp"],
            config["use_smoothed_for_delta_edge"]))
        parameter = next(iter(model.parameters()), None) if hasattr(model, "parameters") else None
        return Architecture(
            int(config["num_layers"]), int(config["node_feat_dim"]), int(config["edge_feat_dim"]),
            int(config["three_body_feat_dim"]), int(hidden), int(config["num_radial"]), int(config["num_angular"]),
            int(config["max_n"]), int(config["max_l"]), float(config["pairwise_cutoff"]),
            float(config["three_body_cutoff"]), kinds, 4 if parameter is None else int(parameter.element_size()))
    except (KeyError, TypeError, ValueError):
        return None


# The measured architectures: the RDX 1M fine-tune (the tables above), the 4M FeCOH fine-tune and the
# library default 10M OAM model (``matris_10m_oam``).
RDX_1M = Architecture(4, 128, 32, 32, 256, 7, 7, 4, 4, 6.0, 4.0)
FECOH_4M = Architecture(4, 128, 128, 128, 256, 7, 7, 4, 4, 6.0, 4.0)
OAM_10M = Architecture(10, 128, 128, 128, 256, 7, 7, 4, 4, 6.0, 4.5)

# Per-architecture tables (GB200, 2026-09-30; see the module docstring). Peak reserved GiB, max rank:
#   4M single topology, triplet rows 152,262 / 951,366 / 1,218,096 / 3,805,464 / 7,610,928 (FeCOH 1,165,
#   RDX 1,344, FeCOH 9,320, RDX 5,376, RDX 10,752 atoms)
#     eager off 6.60 / 38.36 / 51.46 / 152.76 / OOM, on 3.15 / 17.35 / 23.42 / 67.09 / 133.98,
#     deep 2.41 / 13.71 / 18.77 / 61.63 / 104.93; model_graph off 14.13 / 78.71 / 104.35 / - / -,
#     on 6.96 / 35.34 / 49.03 / 69.24 / -, deep 5.68 / 31.62 / 43.19 / 65.60 / -
#   10M single topology (cutoff 4.5 A), eager off: Si 1,000 / 1,728 / 4,096 (240,000 / 414,720 / 983,040 rows)
#     24.74 / 42.34 / 99.71, SiO2 1,125 (935,250) 83.36, LiFePO4 1,008 (1,099,440) 97.93, RDX 1,344
#     (1,951,244) 170.27, RDX 2,688 OOM; on Si 1,000 / 4,096 10.07 / 40.59, RDX 1,344 / 2,688 65.01 / 127.83,
#     Cu 4,096 (7,053,312) OOM; deep Si 1,000 / 4,096 7.54 / 28.87, RDX 1,344 / 2,688 49.18 / 98.04,
#     Cu 4,096 159.67, RDX 5,376 OOM; model_graph off Si 1,000 / 1,728 51.83 / 86.76 (Si 4,096 and RDX 1,344
#     OOM), on Si 1,000 / 4,096 21.39 / 83.38, RDX 1,344 132.01 (2,688 OOM), deep 15.81 / 67.71 / 101.10
#   bond_partial native P=4 eager, capacity triplet rows: 4M RDX 10,752 (1,925,120) off 79.76, on 37.33;
#     10M RDX 1,344 (524,288) off 47.23 (model_graph 48.11), 2,688 (1,024,000) on 35.39, 5,376 (1,974,272)
#     deep 52.98, 10,752 (3,940,352) on 135.67, deep 99.91
_TABLE_4M = {
    ("single", "topology", "eager"): Coefficients(2.991, 39.791),
    ("single", "topology", "model_graph"): Coefficients(3.578, 83.764),
    ("single", "topology", "eager", "narrow"): Coefficients(2.149, 17.463),
    ("single", "topology", "model_graph", "narrow"): Coefficients(30.870, 14.909),
    ("single", "topology", "eager", "deep"): Coefficients(8.672, 13.916),
    ("single", "topology", "model_graph", "deep"): Coefficients(25.439, 14.573),
}
_TABLE_10M = {
    ("single", "topology", "eager"): Coefficients(17.173, 83.960),
    ("single", "topology", "model_graph"): Coefficients(3.849, 199.920),
    ("single", "topology", "eager", "narrow"): Coefficients(9.664, 31.460),
    ("single", "topology", "model_graph", "narrow"): Coefficients(20.533, 63.932),
    ("single", "topology", "eager", "deep"): Coefficients(11.210, 22.250),
    ("single", "topology", "model_graph", "deep"): Coefficients(19.459, 49.083),
}
# name -> (architecture, table); the RDX table is CALIBRATION (with its D8 and C3 variants).
CALIBRATED_ARCHITECTURES = {"rdx_1m": (RDX_1M, None), "fecoh_4m": (FECOH_4M, _TABLE_4M),
                            "oam_10m": (OAM_10M, _TABLE_10M)}
ARCHITECTURE_CALIBRATION = {name: table for name, (_, table) in CALIBRATED_ARCHITECTURES.items() if table}
# P=1 model_graph runners of these calibrations hold their envelope bucket only above these triplet rows
# (the graph-gain limit of applications.capacity_policy, which reads this table); there their single
# topology model_graph peaks are ENVELOPE_CALIBRATION's (main after #39, GB200, peak reserved GiB with C3 on,
# one bucket each):
#   4M off  Si 1,728 / LiFePO4 1,008 / RDX 1,344 / Si 4,096 / FeCOH 9,320 / RDX 2,688 / RDX 5,376 (414,720 /
#     617,472 / 951,366 / 983,040 / 1,218,096 / 1,902,732 / 3,805,464 rows) 18.50 / 25.83 / 38.99 / 43.47 /
#     51.91 / 76.24 / 155.33; on the same and RDX 10,752 (7,610,928) 9.07 / 11.99 / 17.85 / 20.92 / 24.49 /
#     34.15 / 69.44 / 142.39; deep 7.45 / 11.39 / 15.06 / 15.68 / 20.85 / 28.61 / 54.87 / 113.30
#   10M off Si 1,728 / SiO2 1,125 / Si 4,096 / LiFePO4 1,008 / Cu 1,000 / RDX 1,344 (414,720 / 943,388 /
#     983,040 / 1,099,440 / 1,722,000 / 1,951,244 rows) 43.76 / 85.96 / 101.95 / 99.89 / 149.92 / 171.79; on the
#     same and RDX 2,688 (3,902,488) 18.00 / 33.22 / 41.54 / 38.52 / 57.13 / 65.81 / 132.43; deep the same and
#     Cu 4,096 (7,053,312) 13.74 / 24.09 / 33.44 / 28.43 / 43.05 / 49.85 / 98.82 / 174.23
# About the eager entries (one working set); Si 4,096 binds every level.
GRAPH_GAIN_MAX_ROWS = {"fecoh_4m": 400_000, "oam_10m": 400_000}
ENVELOPE_CALIBRATION = {
    "fecoh_4m": {("single", "topology", "model_graph"): Coefficients(3.921, 40.232),
                 ("single", "topology", "model_graph", "narrow"): Coefficients(2.770, 18.464),
                 ("single", "topology", "model_graph", "deep"): Coefficients(3.117, 14.558)},
    "oam_10m": {("single", "topology", "model_graph"): Coefficients(22.006, 81.324),
                ("single", "topology", "model_graph", "narrow"): Coefficients(9.557, 32.535),
                ("single", "topology", "model_graph", "deep"): Coefficients(9.659, 24.192)},
}

# Derived entries: the RDX entry times DERIVED_SAFETY x the analytic per-row ratio (derived_bytes_per_row at
# derived_edges_per_row, at least _DERIVED_EDGES_PER_ROW, over RDX's at _RDX_EDGES_PER_ROW); the ratio alone under-predicts five measured 4M/10M
# model_graph points by up to 9% (10M deep, 1,344 atoms).
DERIVED_SAFETY = 1.25
_DERIVED_EDGES_PER_ROW = 0.25
_RDX_EDGES_PER_ROW = 0.128  # RDX: 121,778 directed edges per 951,366 triplet rows (6 / 4.0 A)
# (tau, fixed bytes) per level: the next layer's inputs held while one layer is rebuilt, and per-row constants
_DERIVED_OVERLAP = {"off": (0.60, 50), "narrow": (0.72, 100), "deep": (0.81, 160)}


def derived_bytes_per_row(arch: Architecture, recompute: str, edges_per_row: float) -> float:
    """Saved bytes per triplet row of one evaluation at a recompute level: per layer the line-graph row's
    tensors plus ``edges_per_row`` directed edges' (FP32 widths from the kept activations of each level),
    plus the overlap of the next layer and a fixed part (fitted on RDX and 4M; the analytic model of the
    task that measured the 4M and 10M peaks)."""
    n, e, t = arch.node_feat_dim, arch.edge_feat_dim, arch.three_body_feat_dim
    line = {"off": 6 * e + 10 * t + 8, "narrow": t + 4 * e + 8, "deep": t + 2 * e + 4}
    edge = {"off": 6 * n + 11 * e + 8, "narrow": 4 * n + 2 * e + 8, "deep": 2 * n + e + 4}
    tau, fixed = _DERIVED_OVERLAP[recompute]
    per_layer = arch.dtype_bytes * (line[recompute] + edges_per_row * edge[recompute])
    return arch.num_layers * per_layer + tau * arch.dtype_bytes * (line["off"] + edges_per_row * edge["off"]) + fixed


def derived_edges_per_row(arch: Architecture) -> float:
    """Directed pair edges per triplet row for the derived bound: RDX's at 6 / 4.0 A scaled by the pair
    edges per atom ((pair cutoff / 6)^3) over the triplets per atom ((three-body cutoff / 4)^6) at the same
    density, and at least ``_DERIVED_EDGES_PER_ROW``."""
    ratio = (arch.pairwise_cutoff / RDX_1M.pairwise_cutoff) ** 3 / max(
        arch.three_body_cutoff / RDX_1M.three_body_cutoff, 1e-3) ** 6
    return max(_DERIVED_EDGES_PER_ROW, _RDX_EDGES_PER_ROW * ratio)


def derived_scale(arch: Architecture, recompute: str) -> float:
    """The factor on an RDX entry that bounds ``arch`` (``DERIVED_SAFETY`` included)."""
    return DERIVED_SAFETY * (derived_bytes_per_row(arch, recompute, derived_edges_per_row(arch))
                             / derived_bytes_per_row(RDX_1M, recompute, _RDX_EDGES_PER_ROW))


@dataclass(frozen=True)
class MemoryModel:
    """Which calibration predicts a model's peaks (``memory_model``): ``calibration`` is a measured
    architecture's name, "derived" or "uncalibrated" (no architecture, no prediction); for the last two
    ``calibrated`` is False: no automatic choice may relax memory."""
    architecture: Architecture | None = None
    calibration: str = "rdx_1m"

    @property
    def calibrated(self) -> bool:
        return self.calibration in CALIBRATED_ARCHITECTURES

    @property
    def strict(self) -> bool:
        """An unpredicted (None) peak does not count as fitting: every model but the RDX calibration's."""
        return self.calibration != "rdx_1m"

    def summary(self) -> dict:
        return {"calibration": self.calibration,
                "architecture": None if self.architecture is None else self.architecture.fingerprint()}


RDX_MEMORY_MODEL = MemoryModel()
# A model without a MatRIS ``Architecture`` (``memory_model``): every prediction None, strict.
UNCALIBRATED_MEMORY_MODEL = MemoryModel(None, "uncalibrated")


def no_bound_reason(memory: MemoryModel, outside: str) -> str:
    """Why K6 bounds no capture of a model whose automatic choices may not rely on it: no MatRIS
    architecture (``UNCALIBRATED_MEMORY_MODEL``), else ``outside`` (the caller's wording for a derived model
    or an entry its architecture was not measured with)."""
    return "no MatRIS architecture (K6 uncalibrated)" if memory.calibration == "uncalibrated" else outside


def entry_calibrated(memory: MemoryModel | None, engine: str, profile: str | None, execution: str,
                     recompute: str = "off") -> bool:
    """Whether a prediction comes from a measured table: the RDX model's always (its uncalibrated entries
    predict None), another measured architecture's where its table has the entry, a derived model's or
    ``UNCALIBRATED_MEMORY_MODEL``'s never."""
    if memory is None or memory.calibration == "rdx_1m":
        return True
    return _key(engine, profile, execution, recompute) in ARCHITECTURE_CALIBRATION.get(memory.calibration, {})


def memory_model(model) -> MemoryModel:
    """The calibration for a model (or an ``Architecture``): the measured architecture with the smallest
    per-row bytes whose envelope holds it; else the RDX tables where the derived bound is at most RDX's at
    every level (``derived_scale`` <= 1: e.g. the small test models with other cutoffs or norms); else
    "derived". A model without a MatRIS ``Architecture`` (``architecture`` None, e.g. another potential's
    module, a config lacking a key, or None) is ``UNCALIBRATED_MEMORY_MODEL``."""
    arch = model if isinstance(model, Architecture) else architecture(model)
    if arch is None:
        return UNCALIBRATED_MEMORY_MODEL
    covering = [name for name, (calibrated, _) in CALIBRATED_ARCHITECTURES.items() if arch.within(calibrated)]
    if covering:
        return MemoryModel(arch, min(covering, key=lambda name: derived_bytes_per_row(
            CALIBRATED_ARCHITECTURES[name][0], "off", _RDX_EDGES_PER_ROW)))
    if all(derived_scale(arch, level) <= 1.0 for level in _DERIVED_OVERLAP):
        return MemoryModel(arch, "rdx_1m")
    return MemoryModel(arch, "derived")


class MemoryAdmissionError(RuntimeError):
    """A requested mode, or every mode under auto, exceeds the device budget."""


def enabled(value: str | None = None) -> bool:
    value = os.environ.get(ENVIRONMENT, "on") if value is None else value
    if value not in ("on", "off", "1", "0"):
        raise ValueError(f"{ENVIRONMENT} must be on/off/1/0, got {value!r}")
    return value in ("on", "1")


def predict(engine: str, profile: str | None, execution: str, rows: int, line_rows: int = 0, *,
            checkpoint: bool = False, recompute: str = "off", mixed_width: bool = True,
            single_bwd: bool = True, gated_tail: bool = False, memory: MemoryModel | None = None) -> int | None:
    """Predicted peak bytes, or None where the table has no calibration.

    ``recompute`` ("narrow" = saved_tensor_recompute "on", "deep") other than "off" reads the
    ``(engine, profile, execution, recompute)`` entries. ``mixed_width``: whether D8 runs
    (``mixed_width_active``); False reads ``CALIBRATION_MIXED_WIDTH_OFF`` where it has the key.
    ``single_bwd``: whether C3 runs (``single_bwd_mode``); True adds its slope increment.
    ``gated_tail``: with C2's slope increment (``GATED_TAIL_GIB_PER_MILLION_ROWS``).
    ``memory``: the model's calibration (``memory_model``; None: the RDX tables); ``UNCALIBRATED_MEMORY_MODEL``
    predicts None everywhere."""
    if checkpoint:
        return None
    if memory is not None and memory.calibration != "rdx_1m":
        table = ARCHITECTURE_CALIBRATION.get(memory.calibration, {})
        key = _key(engine, profile, execution, recompute)
        entry = table.get(key)
        if engine == "single" and execution == "model_graph" and rows > GRAPH_GAIN_MAX_ROWS.get(memory.calibration, rows):
            entry = ENVELOPE_CALIBRATION.get(memory.calibration, {}).get(key, entry)  # an envelope-only runner
        if entry is not None:
            return entry.predict(rows, line_rows)
        # Not measured for this architecture: the RDX entry scaled by the derived per-row ratio.
        arch = CALIBRATED_ARCHITECTURES[memory.calibration][0] if memory.calibrated else memory.architecture
        if arch is None:
            return None  # uncalibrated: no architecture to derive a bound from
        rdx = predict(engine, profile, execution, rows, line_rows, recompute=recompute, mixed_width=mixed_width,
                      single_bwd=single_bwd, gated_tail=gated_tail)
        return None if rdx is None else int(rdx * derived_scale(arch, recompute))
    key = _key(engine, profile, execution, recompute)
    coefficients = None if mixed_width else CALIBRATION_MIXED_WIDTH_OFF.get(key)
    coefficients = CALIBRATION.get(key) if coefficients is None else coefficients
    if coefficients is not None and single_bwd and key in SINGLE_BWD_GIB_PER_MILLION_ROWS:
        coefficients = replace(coefficients, gib_per_million_rows=coefficients.gib_per_million_rows
                               + SINGLE_BWD_GIB_PER_MILLION_ROWS[key])
    if coefficients is None:
        return None
    increment = GATED_TAIL_GIB_PER_MILLION_ROWS.get(key, 0.0) if gated_tail else 0.0
    return coefficients.predict(rows, line_rows) + int(increment * rows / 1e6 * (1 + _SLACK) * _GIB)


def _key(engine, profile, execution, recompute):
    return (engine, profile, execution) if recompute == "off" else (engine, profile, execution, recompute)


def margin_bytes(total: int) -> int:
    return max(_MARGIN_BYTES, int(_MARGIN_FRACTION * total))

