"""Measured memory model of a model without a K6 calibration.

K6 (``fastmd.engine.memory_admission``) predicts MatRIS peaks from tables measured per
architecture; any other model is "uncalibrated" there, so every automatic memory choice takes its safe
side and execution auto stays eager. A :class:`MeasuredMemoryModel` measures the model instead: the
peak CUDA bytes of one eager step and of one capture at three or more capacity points, fitted per kind
by least squares to

    bytes = a + b * edges            (+ c * triplets when the topology has triplets)

The fit is valid with at least three distinct capacity points (and one more than its unknowns), with
triplet/edge ratios spanning at least ``min_ratio_span`` when triplets are fitted (else the two costs
cannot be told apart), with costs that are not negative beyond the noise (a negative per-edge or
per-triplet cost whose effect over the measured range stays within the residual bound is taken as 0),
and with every residual within the residual bound ``max(residual_floor, residual_fraction * largest
peak)``. Its bound at a capacity is the fit plus twice the largest residual (at least
``headroom_floor``): the headroom comes from the measured fit, not from a fixed factor. It predicts
nothing for an invalid fit or beyond ``max_extrapolation`` times the largest measured edge (or
triplet) capacity, so auto then keeps the safe side. MatRIS stays on K6.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from typing import Callable, Iterable, Mapping, Sequence

KINDS = ("eager", "capture")
MIB = 1 << 20


@dataclass(frozen=True)
class MemoryPoint:
    """One measurement: the peak bytes above the allocated baseline of one ``kind`` run at a capacity."""

    kind: str
    edges: int
    triplets: int
    peak_bytes: int


@dataclass(frozen=True)
class MemoryFit:
    kind: str
    a: float  # bytes
    b: float  # bytes per edge
    c: float  # bytes per triplet (0 without triplets)
    max_residual: float | None  # bytes, over the fitted points (None: not fitted)
    residual_bound: float  # bytes
    max_edges: int  # the largest measured edge capacity
    max_triplets: int  # the largest measured triplet capacity
    valid: bool
    reason: str

    def predict(self, edges: int, triplets: int = 0) -> float:
        return max(self.a + self.b * edges + self.c * triplets, 0.0)


class MeasuredMemoryModel:
    """Peaks of one model measured at capacity points and fitted linearly per kind (see the module doc).

    ``triplets``: whether the topology has triplets (a ``c * triplets`` term). Points come from
    :meth:`measure` (CUDA) or :meth:`from_points`.
    """

    calibration = "measured"

    def __init__(self, points: Iterable[MemoryPoint], *, triplets: bool, residual_floor: float = 4 * MIB,
                 residual_fraction: float = 0.02, max_extrapolation: float = 2.0, min_ratio_span: float = 1.5,
                 headroom_floor: float = 2 * MIB) -> None:
        self.points = tuple(points)
        self.triplets = bool(triplets)
        self.residual_floor = float(residual_floor)
        self.residual_fraction = float(residual_fraction)
        self.max_extrapolation = float(max_extrapolation)
        self.min_ratio_span = float(min_ratio_span)
        self.headroom_floor = float(headroom_floor)
        self._failure = None  # why the measurement failed (``failed``)
        unknown = {point.kind for point in self.points} - set(KINDS)
        if unknown:
            raise ValueError(f"memory point kinds must be in {KINDS}, got {sorted(unknown)}")
        self.fits = {kind: self._fit(kind) for kind in KINDS if any(p.kind == kind for p in self.points)}

    @classmethod
    def from_points(cls, points: Iterable[MemoryPoint], *, triplets: bool, **bounds) -> "MeasuredMemoryModel":
        return cls(points, triplets=triplets, **bounds)

    @classmethod
    def failed(cls, reason: str, *, triplets: bool) -> "MeasuredMemoryModel":
        """A measurement that failed (``reason``, e.g. a CUDA OOM): no points, invalid, predicts nothing."""
        model = cls((), triplets=triplets)
        model._failure = str(reason)
        return model

    @classmethod
    def measure(cls, runs: Mapping[str, Callable], capacities: Sequence, device, *, triplets: bool,
                warmup: bool = True, metric: str = "allocated", **bounds) -> "MeasuredMemoryModel":
        """Measure ``runs`` (kind -> a callable that runs one step of that kind at a capacity and returns
        the (edges, triplets) capacity it held) at each capacity (an edge count, or an (edges, triplets)
        pair for a topology with triplets, whose ratios must vary): the peak allocated bytes above
        the allocated baseline before the call, after a collection and ``empty_cache`` (``metric="reserved"``:
        the peak reserved bytes above the reserved baseline, what runs out on the device). Every point must
        start from the same state: a capture's run captures its graph and releases it before returning
        (a captured graph that stays alive keeps state later captures share, e.g. the first capture's
        cuBLAS workspace, which then counts in one point only: the peaks are not linear and the fit is
        invalid). ``warmup``: each run first runs once, unmeasured, at the smallest capacity (its
        one-time state then exists before the measurement)."""
        import torch
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("MeasuredMemoryModel.measure needs a CUDA device")
        if metric not in ("allocated", "reserved"):
            raise ValueError(f"metric must be 'allocated' or 'reserved', got {metric!r}")
        current, peak = ((torch.cuda.memory_allocated, torch.cuda.max_memory_allocated) if metric == "allocated"
                         else (torch.cuda.memory_reserved, torch.cuda.max_memory_reserved))
        points = []
        convert = lambda capacity: (tuple(int(value) for value in capacity) if isinstance(capacity, (tuple, list))
                                    else int(capacity))
        for kind, run in runs.items():
            if warmup:
                run(convert(min(capacities, key=lambda c: c[0] if isinstance(c, (tuple, list)) else c)))
            for capacity in capacities:
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                baseline = current(device)
                torch.cuda.reset_peak_memory_stats(device)
                edges, held_triplets = run(convert(capacity))
                torch.cuda.synchronize(device)
                points.append(MemoryPoint(kind, int(edges), int(held_triplets), int(peak(device) - baseline)))
        return cls(points, triplets=triplets, **bounds)

    @staticmethod
    def _ratios(points) -> list[float]:
        return [p.triplets / p.edges for p in points if p.edges > 0]

    def _fit(self, kind: str) -> MemoryFit:
        import numpy as np
        points = [point for point in self.points if point.kind == kind]
        columns = 3 if self.triplets else 2
        x = np.array([[1.0, p.edges] + ([p.triplets] if self.triplets else []) for p in points], dtype=np.float64)
        y = np.array([p.peak_bytes for p in points], dtype=np.float64)
        distinct = len({(p.edges, p.triplets if self.triplets else 0) for p in points})
        max_edges = max((p.edges for p in points), default=0)
        max_triplets = max((p.triplets for p in points), default=0) if self.triplets else 0
        bound = max(self.residual_floor, self.residual_fraction * (float(y.max()) if len(y) else 0.0))
        need = max(3, columns + 1)
        invalid = lambda why: MemoryFit(kind, 0.0, 0.0, 0.0, None, bound, max_edges, max_triplets, False,
                                        f"{kind}: {why}")
        if distinct < need:
            return invalid(f"{distinct} distinct capacity points, at least {need} needed")
        ratios = self._ratios(points)
        if self.triplets and not (ratios and max(ratios) > 0 and (
                min(ratios) == 0 or max(ratios) >= self.min_ratio_span * min(ratios))):
            return invalid(f"triplet/edge ratios span less than {self.min_ratio_span}x: the per-edge and "
                           "per-triplet costs cannot be told apart")
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        if np.linalg.matrix_rank(x) < columns:
            return invalid("edges and triplets do not vary independently")
        residual = float(np.abs(x @ coefficients - y).max())
        a, b = float(coefficients[0]), float(coefficients[1])
        c = float(coefficients[2]) if self.triplets else 0.0
        # A negative cost whose effect over the measured range stays within the residual bound is noise
        # (taken as 0, which only raises the predictions); a larger one means the peaks are not linear.
        problems = [why for ok, why in ((b * max_edges >= -bound, f"negative bytes per edge {b:.1f}"),
                                        (c * max_triplets >= -bound, f"negative bytes per triplet {c:.1f}"),
                                        (residual <= bound, f"max residual {residual / MIB:.2f} MiB > bound "
                                                            f"{bound / MIB:.2f} MiB (not linear)")) if not ok]
        reason = f"{kind}: " + ("; ".join(problems) if problems else
                                f"{len(points)} points, max residual {residual / MIB:.2f} MiB <= {bound / MIB:.2f} MiB")
        return MemoryFit(kind, a, max(b, 0.0), max(c, 0.0), residual, bound, max_edges, max_triplets, not problems,
                         reason)

    @property
    def valid(self) -> bool:
        """Every measured kind fitted, and the capture kind was measured."""
        return "capture" in self.fits and all(fit.valid for fit in self.fits.values())

    @property
    def reason(self) -> str:
        if self._failure is not None:
            return self._failure
        if "capture" not in self.fits:
            return "no capture points measured"
        return "; ".join(fit.reason for fit in self.fits.values())

    def predict(self, kind: str, edges: int, triplets: int = 0) -> int | None:
        """The fitted peak bytes, or None (no valid fit of ``kind``, or beyond the extrapolation range)."""
        fit = self.fits.get(kind)
        if fit is None or not fit.valid or edges > self.max_extrapolation * max(fit.max_edges, 1):
            return None
        if self.triplets:
            # Two costs fitted on a few ratios: predict only within the measured triplet/edge ratios.
            ratios = self._ratios([p for p in self.points if p.kind == kind])
            if triplets > self.max_extrapolation * max(fit.max_triplets, 1) or not (
                    min(ratios) <= (triplets / edges if edges > 0 else float("inf")) <= max(ratios)):
                return None
        return int(round(fit.predict(edges, triplets)))

    def headroom(self, kind: str) -> int:
        """Twice the fit's largest residual, at least ``headroom_floor`` bytes."""
        if kind not in self.fits:
            raise ValueError(f"no {kind!r} points were measured (measured: {sorted(self.fits)})")
        return int(max(2 * (self.fits[kind].max_residual or 0.0), self.headroom_floor))

    def bound(self, kind: str, edges: int, triplets: int = 0) -> int | None:
        """The prediction plus the measured headroom (:meth:`headroom`), or None."""
        predicted = self.predict(kind, edges, triplets)
        if predicted is None:
            return None
        return predicted + self.headroom(kind)

    def max_edges(self, kind: str, budget: float, triplets_per_edge: float = 0.0) -> int | None:
        """The largest edge capacity whose bound fits ``budget`` bytes (triplets growing with the
        edges at ``triplets_per_edge``), within the extrapolation range; None without a valid fit."""
        fit = self.fits.get(kind)
        if fit is None or not fit.valid:
            return None
        per_edge = fit.b + fit.c * triplets_per_edge
        limit = int(self.max_extrapolation * max(fit.max_edges, 1))
        if self.triplets and triplets_per_edge > 0:
            limit = min(limit, int(self.max_extrapolation * max(fit.max_triplets, 1) / triplets_per_edge))
        if budget < self.headroom(kind):
            return 0  # even an empty capture's bound (the headroom) does not fit
        fixed = fit.a + self.headroom(kind)
        if per_edge <= 0:
            return limit if fixed <= budget else 0
        return max(0, min(int((budget - fixed) // per_edge), limit))

    def summary(self) -> dict:
        return {"calibration": self.calibration, "valid": self.valid, "reason": self.reason,
                "triplets": self.triplets,
                "fits": {kind: {**asdict(fit), "max_residual_mib": None if fit.max_residual is None
                                else fit.max_residual / MIB} for kind, fit in self.fits.items()},
                "points": [asdict(point) for point in self.points]}
