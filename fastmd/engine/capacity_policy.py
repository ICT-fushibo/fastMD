"""Joint edge/triplet capacity planning for fixed-shape CUDA Graphs.

``CapturePolicy`` holds the capacity and capture parameters and pure rules that
every CUDA-graph runner reads (P=1: BucketedGraphRunner, SkinTopologyRunner, the
GPU-MD whole step; P>1: the topology epoch's TieredCapacity and
DistributedGraphRunner). The stateful parts stay with their owners: the P=1
pool valve and tight-bucket set, the P>1 global-MAX agreement and its single
resident entry. ``precapture_fits`` sizes P=1 neighbor buckets by the model's
predicted bytes (K6, ``fastmd.engine.memory_admission.memory_model``) and
its ``graph_gain_max_rows``; ``graph_execution`` is the P=1 execution-auto rule.
"""
from __future__ import annotations

from dataclasses import astuple, dataclass
import hashlib
import os


def tier(value: int, step: int, min_pad: int) -> int:
    """The capacity of one bucket: the next multiple of ``step`` above ``value``, one more step if that
    leaves less than ``min_pad`` slack (``JointCapacityPolicy.key_for``; P=1 model-graph triplets use 0)."""
    capacity = (int(value) // step + 1) * step
    return capacity + step if capacity - int(value) < min_pad else capacity


@dataclass(frozen=True)
class CapturePolicy:
    """Capacity and capture parameters shared by the P=1 and P>1 graph runners (see the module docstring).

    Quantization comes from ``CapacityConfig``; the hysteresis knobs from the registered
    ``MATRIS_TIER_*`` options (``from_config``). Defaults are the values each runner used before.
    """

    u_step: int = 512
    t_step: int = 8192
    min_pad_u: int = 256
    min_pad_t: int = 4096
    n_dummy: int = 64
    # Side-stream warmups before a runner's first capture (CapacityConfig.warmup), and before every
    # later capture of the same runner: the program and its shapes' kernels (Triton JIT, cuBLAS
    # workspace, frozen-weight caches) are warm after one uncaptured run; a later capture differs only
    # in its capacities (MATRIS_RECAPTURE_WARMUP).
    warmup: int = 3
    recapture_warmup: int = 1
    # Growth headroom above a grown demand: P=1 buckets and the P>1 eager epoch (one step or 1/128 of
    # the capacity; see cuda_graph.py), and the P>1 model_graph band (a recapture is collective;
    # MATRIS_TIER_GROWTH_HEADROOM). A P=1 shrink must save twice the headroom.
    growth_headroom: float = 1 / 128
    graph_growth_headroom: float = 1 / 32
    shrink_min_saving: float = 1 / 64
    # Tier hysteresis (MATRIS_TIER_*): demand window, sigma band, check interval, shrink cooldown.
    hysteresis_window: int = 2000
    hysteresis_sigma: float = 3.0
    check_interval: int = 256
    down_cooldown: int = 4000
    # P=1 shared pool: a miss releases the pool above this fraction of the device reserved.
    release_above: float = 1 / 8
    # P=1 precapture: a dimension is small where one step is at least this fraction of its demand
    # (its tight neighbor is then precaptured, if ``precapture_fits``).
    precapture_min_step: float = 1 / 256

    @classmethod
    def from_config(cls, capacity=None, *, distributed=False, options=None, **overrides) -> "CapturePolicy":
        """The policy of a ``CapacityConfig`` and the registered ``MATRIS_TIER_*`` options: ``options``
        (a resolved configuration's kernel options; the P>1 evaluator passes its forward profile's,
        where the environment is ignored as for GPU MD under that profile), else ``env_value`` (in an
        inference scope its resolved options). P>1 (``distributed``, which alone reads its model_graph
        band) builds it once per evaluator in its collectively checked construction; the evaluator's
        rank contract and the tiers' fingerprint compare it."""
        from fastmd.config import CapacityConfig
        from fastmd.config import env_value
        option = env_value if options is None else options.__getitem__
        c = CapacityConfig() if capacity is None else capacity
        values = dict(
            u_step=int(c.u_step), t_step=int(c.t_step), min_pad_u=int(c.min_pad_u), min_pad_t=int(c.min_pad_t),
            n_dummy=int(c.n_dummy), warmup=int(c.warmup),
            # Not kernel options (configuration fingerprints stay): read from the process environment.
            recapture_warmup=int(os.environ.get("MATRIS_RECAPTURE_WARMUP", "1")),
            graph_growth_headroom=(float(option("MATRIS_TIER_GROWTH_HEADROOM")) if distributed
                                   else cls.graph_growth_headroom),
            hysteresis_window=int(option("MATRIS_TIER_HYSTERESIS_WINDOW")),
            hysteresis_sigma=float(option("MATRIS_TIER_HYSTERESIS_SIGMA")),
            check_interval=int(option("MATRIS_TIER_CHECK_INTERVAL")),
            down_cooldown=int(option("MATRIS_TIER_DOWN_COOLDOWN")))
        values.update(overrides)
        return cls(**values)

    def __post_init__(self):
        if self.warmup < 0 or self.recapture_warmup < 1:
            raise ValueError("warmup must be nonnegative and recapture_warmup (MATRIS_RECAPTURE_WARMUP) positive")
        if not all(0 <= value < 1 for value in (self.growth_headroom, self.graph_growth_headroom,
                                                self.shrink_min_saving, self.precapture_min_step)):
            raise ValueError("growth headrooms and fractions must be finite and in [0, 1)")
        if not 0 <= self.release_above <= 1 or not 0 <= self.hysteresis_sigma < float("inf"):
            raise ValueError("release_above must be in [0, 1] (0: always release) and hysteresis_sigma finite, >= 0")
        if self.hysteresis_window < 1 or self.check_interval < 1 or self.down_cooldown < 0:
            raise ValueError("hysteresis window and check interval must be positive, the cooldown nonnegative")

    @property
    def fingerprint(self) -> str:
        """Every parameter (the P>1 capture agreement compares it across ranks)."""
        return hashlib.sha256(repr(astuple(self)).encode()).hexdigest()[:16]

    def warmups(self, first: bool, *, distributed: bool = False) -> int:
        """Side-stream warmup runs before a capture: the configured count for a runner's first capture,
        ``recapture_warmup`` (at most the configured count) for a later one. A P>1 capture always needs
        one (its program initializes collectives and the builder's buffers outside capture)."""
        count = self.warmup if first else min(self.warmup, self.recapture_warmup)
        return max(1, count) if distributed else count

    def bucket_key(self, u: int, t: int) -> tuple[int, int]:
        """A P=1 model-graph bucket for (undirected edges, triplets): u with ``min_pad_u``, t without."""
        return tier(u, self.u_step, self.min_pad_u), tier(t, self.t_step, 0)

    def headroom(self, capacity: int, step: int) -> int:
        """A P=1 growth's headroom: one step, or whole steps up to ``growth_headroom`` of the capacity."""
        return max(step, int(capacity * self.growth_headroom) // step * step)

    def small(self, demand: int, step: int) -> bool:
        """Whether one ``step`` is at least ``precapture_min_step`` of ``demand`` (P=1 precapture)."""
        return step >= demand * self.precapture_min_step


# Measured model_graph gains over eager on the 128-wide calibrations (P=1 MD steps, GB200, task MG):
# 10M Si 1,000 / 240k triplet rows -15.5%, Si 1,728 / 415k -1.3%, RDX 1,344 / 1.95M -1.5%;
# 4M box60 / 152k -44.8%, RDX 1,344 / 951k -2.7%: below 3% above ~400k rows, where these GPU-bound
# models leave launch overhead nothing to hide. There a runner's neighbor buckets, which only save the
# padding to the envelope bucket (under 1% of the rows), are not worth a second working set in the
# pool (10M, RDX 1,344, explicit model_graph: 4 buckets 132 GiB, the envelope alone 66 GiB, +0.4%). One
# working set costs about the eager peak, so execution auto keeps the graph (10M, RDX 1,344, 600
# Nose-Hoover chain steps: whole_step 67.6 GiB 281.7 ms vs eager 65.7 GiB 284.8 ms). The RDX model's
# neighbors pay at every measured size (to 43,008 atoms): no limit.
# The limits live with the K6 tables (memory_admission.GRAPH_GAIN_MAX_ROWS), whose model_graph entries above
# them are fitted on envelope-only runners.
from fastmd.engine.memory_admission import GRAPH_GAIN_MAX_ROWS  # noqa: E402


def graph_gain_max_rows(model) -> int | None:
    """P=1 precapture: the triplet rows above which a model_graph runner of this model holds its
    envelope bucket only (its K6 calibration, ``memory_model``); None: no limit (every model without a
    measured one: the RDX model, the models its tables cover, a derived or an uncalibrated model)."""
    from fastmd.engine import memory_admission
    return GRAPH_GAIN_MAX_ROWS.get(memory_admission.memory_model(model).calibration)


def graph_execution(model, config, rows: int, device, *, admission_off_trusts_any_model=False) -> tuple[bool, str]:
    """P=1 execution auto (calculator, GPU-resident MD): (whether a CUDA graph may run, reason) for
    ``rows`` triplet rows. Eager outside every measured K6 calibration (``memory_model``), without a
    prediction, or where K6's model_graph peak at the recompute level eager takes exceeds
    AUTO_THRESHOLD of the budget. Admission off: a graph for a calibrated model, or for any model with
    ``admission_off_trusts_any_model`` (the calculator; the callers' OOM safety nets remain)."""
    from fastmd.engine import memory_admission
    from fastmd.engine.saved_recompute import AUTO_THRESHOLD, LADDER, _PUBLIC
    memory = memory_admission.memory_model(model)
    if not memory_admission.enabled() and (memory.calibrated or admission_off_trusts_any_model):
        return True, "memory admission off"
    if not memory.calibrated:
        return False, memory_admission.no_bound_reason(
            memory, "architecture outside every measured K6 calibration") + ": no memory bound for a capture"
    budget = graph_budget(model, device)
    options = config.kernel_options
    predict = lambda execution, level: memory_admission.predict(
        "single", config.optimization_profile, execution, rows, recompute=level,
        mixed_width=memory_admission.mixed_width_active(options),
        single_bwd=memory_admission.single_bwd_mode(options) == "on", memory=memory)
    requested = config.saved_tensor_recompute
    levels = LADDER if requested == "auto" else (dict((v, k) for k, v in _PUBLIC.items())[requested],)
    level = next((lv for lv in levels if (predict("eager", lv) or 0) <= AUTO_THRESHOLD * budget), levels[-1])
    peak = predict("model_graph", level)
    if peak is None:
        return False, f"no K6 prediction for optimization_profile={config.optimization_profile!r}"
    fits = peak <= AUTO_THRESHOLD * budget
    return fits, (f"K6: model_graph at recompute {_PUBLIC[level]} predicted {peak / 2**30:.1f} GiB "
                  f"{'<=' if fits else '>'} {AUTO_THRESHOLD:.0%} of {budget / 2**30:.1f} GiB "
                  f"({rows} triplet rows; calibration {memory.calibration})")


def declared_capacity(adapter, config):
    """``config`` with the capacity step ``adapter`` declares (``CapacityDeclaration``): its pair tiers where
    the configuration keeps the default pair step (``CapacityConfig.u_step`` and ``min_pad_u``), its triplet
    tiers (if it declares them) where it keeps the default triplet step (``t_step`` and ``min_pad_t``); else
    ``config``."""
    from dataclasses import replace

    from fastmd.adapters.protocol import CapacityDeclaration
    from fastmd.config import CapacityConfig
    if not isinstance(adapter, CapacityDeclaration):
        return config
    capacity, default = config.capacity, CapacityConfig()
    step, declared = adapter.capacity_step(), {}
    if (capacity.u_step, capacity.min_pad_u) == (default.u_step, default.min_pad_u):
        declared.update(u_step=step.u_step, min_pad_u=step.min_pad_u)
    if step.t_step is not None and (capacity.t_step, capacity.min_pad_t) == (default.t_step, default.min_pad_t):
        declared.update(t_step=step.t_step, min_pad_t=step.min_pad_t)
    return replace(config, capacity=replace(capacity, **declared)) if declared else config


def thermal_factors(adapter, config) -> tuple[float, float]:
    """The (pair, triplet) thermal factors of ``adapter``'s declared capacity step (``CapacityStep``
    ``u_thermal_factor``, ``t_thermal_factor``), each where the resolved ``config`` runs the declared tiers of
    its dimension; else 1 (MatRIS, an adapter without a declaration, an explicit configuration of other tiers)."""
    from fastmd.adapters.protocol import CapacityDeclaration
    if config is None or not isinstance(adapter, CapacityDeclaration):
        return 1.0, 1.0
    step, capacity = adapter.capacity_step(), config.capacity
    pairs = (capacity.u_step, capacity.min_pad_u) == (step.u_step, step.min_pad_u)
    triplets = step.t_step is not None and (capacity.t_step, capacity.min_pad_t) == (step.t_step, step.min_pad_t)
    return float(step.u_thermal_factor) if pairs else 1.0, float(step.t_thermal_factor) if triplets else 1.0


def held_demand(demand: int, headroom: int, factor: float) -> int:
    """What a capture holds for ``demand`` with the runtime's ``headroom``: the demand plus the headroom, or for
    a declared thermal factor (``thermal_factors``) the larger of that and the factor times the demand."""
    return demand + headroom if factor == 1.0 else max(demand + headroom, int(round(demand * factor)))


def graph_budget(model, device) -> int:
    """The P=1 memory budget of a capture decision: free + reserved, less the entry point's baseline
    and the admission margin (the recompute ladder's, so both decide on one budget)."""
    import torch
    from fastmd.engine import memory_admission
    from fastmd.engine.saved_recompute import entry_baseline
    free, total = torch.cuda.mem_get_info(device)
    return (free + torch.cuda.memory_reserved(device) - entry_baseline(model, device)
            - memory_admission.margin_bytes(total))


def precapture_fits(model, config, rows: int, level: str, budget: int) -> tuple[bool, str]:
    """Whether a P=1 runner may hold its neighbor buckets next to the envelope (precapture and the later
    tight recaptures): up to the model's ``graph_gain_max_rows``, and when the peak of several buckets
    in one pool, max(K6 model_graph, 2 x K6 eager) at the envelope's triplet rows and the running
    recompute level, fits the share of ``budget`` that the recompute policy gives model_graph.
    Admission off, or no prediction for the RDX model: as before."""
    from fastmd.engine import memory_admission
    from fastmd.engine.saved_recompute import policy_threshold
    memory = memory_admission.memory_model(model)
    limit = GRAPH_GAIN_MAX_ROWS.get(memory.calibration)
    if limit is not None and rows > limit:
        return False, f"{rows} triplet rows > {limit} ({memory.calibration}): GPU-bound, the envelope only"
    if not memory_admission.enabled():
        return True, "memory admission off"
    options = config.kernel_options
    predict = lambda execution: memory_admission.predict(
        "single", config.optimization_profile, execution, rows, recompute=level,
        mixed_width=memory_admission.mixed_width_active(options), single_bwd=True, memory=memory)
    graph, eager = predict("model_graph"), predict("eager")
    if graph is None or eager is None:
        if memory.strict:  # an unpredicted peak does not fit (every model but the RDX calibration's)
            return False, f"no K6 prediction ({memory.calibration}, recompute {level}): the envelope only"
        if graph is None and eager is None:
            return True, "no K6 prediction (the RDX tables): as before"
    multi = max(graph or 0, 2 * (eager or 0))
    share = policy_threshold(getattr(config, "recompute_policy", "safe"), "model_graph") if (
        memory.calibrated and memory_admission.entry_calibrated(
            memory, "single", config.optimization_profile, "model_graph", level)) else policy_threshold("safe", "")
    fits = multi <= share * budget
    return fits, (f"neighbors at {rows} triplet rows, recompute {level}: {multi / 2**30:.1f} GiB "
                  f"{'<=' if fits else '>'} {share:.0%} of {budget / 2**30:.1f} GiB")


@dataclass(frozen=True, order=True)
class CapacityKey:
    u_capacity: int
    t_capacity: int


class JointCapacityPolicy:
    """Track and plan correlated undirected-edge/triplet capacities.

    Runtime growth is monotonic to avoid graph churn. ``recommend`` performs a
    small offline greedy facility search over observed joint keys and is used by
    the paper artifact to quantify the padding-vs-capture tradeoff.
    ``triplets=False`` (a model that reads edges only): every triplet capacity is 0.
    """

    def __init__(
        self,
        *,
        u_step: int = 512,
        t_step: int = 8192,
        min_pad_u: int = 256,
        min_pad_t: int = 4096,
        u_cost_weight: float = 2.0,
        t_cost_weight: float = 1.0,
        triplets: bool = True,
    ) -> None:
        self.u_step = int(u_step)
        self.t_step = int(t_step)
        self.min_pad_u = int(min_pad_u)
        self.triplets = bool(triplets)
        self.min_pad_t = int(min_pad_t) if self.triplets else 0
        self.u_cost_weight = float(u_cost_weight)
        self.t_cost_weight = float(t_cost_weight)
        self.capacity_history: list[CapacityKey] = []
        self.observations: list[tuple[int, int, CapacityKey]] = []
        self.overflow_events: list[tuple[int, int]] = []

    def key_for(self, u: int, t: int) -> CapacityKey:
        if not self.triplets:
            return CapacityKey(tier(u, self.u_step, self.min_pad_u), 0)
        return CapacityKey(tier(u, self.u_step, self.min_pad_u), tier(t, self.t_step, self.min_pad_t))

    def start(self, u: int, t: int) -> CapacityKey:
        key = self.key_for(u, t)
        self.capacity_history.append(key)
        return key

    def grow(self, current: CapacityKey, u: int, t: int) -> CapacityKey:
        required = self.key_for(u, t)
        grown = CapacityKey(
            max(current.u_capacity, required.u_capacity),
            max(current.t_capacity, required.t_capacity),
        )
        if grown != current:
            self.capacity_history.append(grown)
        return grown

    def observe(self, u: int, t: int, capacity: CapacityKey) -> None:
        self.observations.append((int(u), int(t), capacity))

    def record_overflow(self, u: int, t: int) -> None:
        self.overflow_events.append((int(u), int(t)))

    def reset_observations(self) -> None:
        self.observations.clear()
        self.overflow_events.clear()

    def _assignment_cost(
        self,
        point: tuple[int, int],
        buckets: list[CapacityKey],
    ) -> float:
        u, t = point
        costs = [
            self.u_cost_weight * (key.u_capacity - u)
            + self.t_cost_weight * (key.t_capacity - t)
            for key in buckets
            if u <= key.u_capacity and t <= key.t_capacity
        ]
        return min(costs) if costs else float("inf")

    def recommend(self, max_buckets: int = 4) -> list[CapacityKey]:
        if not self.observations or max_buckets <= 0:
            return []
        points = [(u, t) for u, t, _ in self.observations]
        full = self.key_for(
            max(u for u, _ in points),
            max(t for _, t in points),
        )
        candidates = sorted({self.key_for(u, t) for u, t in points})
        selected = [full]
        while len(selected) < max_buckets:
            current_cost = sum(self._assignment_cost(point, selected) for point in points)
            best_key = None
            best_cost = current_cost
            for candidate in candidates:
                if candidate in selected:
                    continue
                trial = selected + [candidate]
                cost = sum(self._assignment_cost(point, trial) for point in points)
                if cost < best_cost:
                    best_key = candidate
                    best_cost = cost
            if best_key is None:
                break
            selected.append(best_key)
        return sorted(selected)

    def stats(self) -> dict[str, object]:
        if not self.observations:
            return {
                "observations": 0,
                "overflow_events": len(self.overflow_events),
                "capacity_history": [
                    (key.u_capacity, key.t_capacity) for key in self.capacity_history
                ],
                "recommended_keys": [],
            }
        u_padding = [
            (key.u_capacity - u) / max(u, 1)
            for u, _, key in self.observations
        ]
        t_padding = [
            (key.t_capacity - t) / max(t, 1)
            for _, t, key in self.observations
        ]
        return {
            "observations": len(self.observations),
            "overflow_events": len(self.overflow_events),
            "capacity_history": [
                (key.u_capacity, key.t_capacity) for key in self.capacity_history
            ],
            "u_padding_mean": sum(u_padding) / len(u_padding),
            "u_padding_max": max(u_padding),
            "t_padding_mean": sum(t_padding) / len(t_padding),
            "t_padding_max": max(t_padding),
            "recommended_keys": [
                (key.u_capacity, key.t_capacity) for key in self.recommend()
            ],
        }
