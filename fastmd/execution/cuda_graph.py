"""Bucketed CUDA-graph execution for MatRIS MD (static-shape forward+backward).

The model forward is D2H-sync-free, so a fixed-shape forward+backward can be
captured with torch.cuda.graph and replayed each MD step (collapsing ~3300
kernel launches into one replay -> removes launch-gap GPU idle).

MD edge/triplet counts fluctuate, so shapes are made static by padding to fixed
CAPACITY BUCKETS: padding edges/triplets are routed to dummy SINK atoms (index
>= N; a model whose rule is far self-loops: loops of real atoms that add exactly
nothing) that are masked out of the energy/reference via the model's n_real arg, so
results are numerically identical to the ragged model on the real atoms. One
graph is captured per (U_cap, T_cap) bucket and selected at replay time from the
current real counts (host-known .shape, no sync); MD counts cluster into a few
buckets, so after warmup every step is a cache hit.

Note: this does NOT make the ragged neighbor-graph BUILD sync-free (the sparse
three-body line graph cannot be densified without exploding compute); it removes
the forward's launch-gap idle, which is the dominant, addressable component.

A model that reads edges only gets ragged ``GraphBatch`` builds; its buckets
pad them with the adapter's padding rule (the fixed-capacity builder's
layout) and capture the model's call over ``adapter.prepare`` of the padded
batch.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from fastmd._nvtx import nvtx_range
from fastmd.adapters import as_adapter
from fastmd.adapters.protocol import GraphBatch
from fastmd.adapters.protocol import PaddingRule
from fastmd.engine.policy import fusion_scope
from fastmd.graph.fixed_capacity_builder import line_graph_batch
from fastmd.graph.fixed_capacity_builder import self_loop_padding
from fastmd.graph.geometry import small_matmul
from fastmd.graph.radiusgraph import RadiusGraph
from fastmd.engine.entry_config import scoped_call
from fastmd.engine.entry_config import scoped_init
from fastmd.engine._graph_capture import capture_graph
from fastmd.engine.capacity_policy import CapturePolicy, held_demand, precapture_fits, thermal_factors, tier
from fastmd.engine.energy_derivative import engine_derivative


def pad_radius_graph(g: RadiusGraph, u_max: int, t_max: int, n_dummy: int = 64):
    """Pad a single ragged RadiusGraph to fixed capacity (e_max=2*u_max, t_max)
    with n_dummy dummy SINK atoms. Returns (padded_graph, n_real).

    Padding self-loops use a large periodic-image shift so lengths are finite
    (radial basis -> 0) and unit vectors well-defined. Padding edges/triplets are
    spread across the dummy atoms and padding undirected edges to avoid atomic-add
    scatter contention.
    """
    with nvtx_range("pad_radius_graph"):
        dev = g.lattice.device
        n = g.atomic_number.shape[0]
        e = g.atom_graph.shape[0]
        u = g.undirected2directed.shape[0]
        t = g.line_graph.shape[0]
        e_max = 2 * u_max
        n_pad_u = u_max - u
        assert e <= e_max and n_pad_u >= 1 and t <= t_max, (e, u, t, e_max, u_max, t_max)
        D = n_dummy  # fixed -> total atom count (N+D) is static across steps

        atomic_number = torch.cat([g.atomic_number, g.atomic_number.new_ones(D)])
        atom_frac_coord = torch.cat([g.atom_frac_coord, g.atom_frac_coord.new_zeros(D, 3)])

        e_pad = e_max - e
        pad_edge_pair = torch.arange(e_pad, device=dev) // 2
        pad_edge_sink = n + (pad_edge_pair * D // n_pad_u)
        atom_graph = g.atom_graph.new_zeros(e_max, 2)
        atom_graph[:e] = g.atom_graph
        atom_graph[e:, 0] = pad_edge_sink
        atom_graph[e:, 1] = pad_edge_sink

        neighbor_image = g.neighbor_image.new_zeros(e_max, 3)
        neighbor_image[:e] = g.neighbor_image
        neighbor_image[e:] = 3.0

        pad_u_ids = torch.arange(u, u_max, device=dev)
        directed2undirected = g.directed2undirected.new_zeros(e_max)
        directed2undirected[:e] = g.directed2undirected
        directed2undirected[e:] = pad_u_ids.repeat_interleave(2)
        undirected2directed = g.undirected2directed.new_zeros(u_max)
        undirected2directed[:u] = g.undirected2directed
        undirected2directed[u:] = torch.arange(e, e_max, 2, device=dev)

        line_graph = g.line_graph.new_zeros(t_max, 5)
        if t > 0:
            line_graph[:t] = g.line_graph
        t_pad = t_max - t
        pad_triplet_atom = torch.arange(t_pad, device=dev) * D // max(t_pad, 1)
        pu = torch.arange(t_pad, device=dev) % n_pad_u
        line_graph[t:, 0] = n + pad_triplet_atom
        line_graph[t:, 1] = u + pu
        line_graph[t:, 2] = e + 2 * pu
        line_graph[t:, 3] = u + pu
        line_graph[t:, 4] = e + 2 * pu

        pg = RadiusGraph(
            atomic_number=atomic_number, atom_frac_coord=atom_frac_coord,
            atom_graph=atom_graph, neighbor_image=neighbor_image,
            directed2undirected=directed2undirected, undirected2directed=undirected2directed,
            line_graph=line_graph, lattice=g.lattice, graph_id=None, mp_id=None,
            composition=g.composition, atom_graph_cutoff=g.atom_graph_cutoff,
            line_graph_cutoff=g.line_graph_cutoff,
            atom_target_sorted=getattr(g, "atom_target_sorted", False),
            line_atom_sorted=getattr(g, "line_atom_sorted", False),
            isolated_atom_count=g.isolated_atom_count,
        )
        return pg, n


@dataclass
class _PaddedGraphWorkspace:
    pg: RadiusGraph
    n_real: int
    u_max: int
    t_max: int
    n_dummy: int
    edge_pair: torch.Tensor
    u_ids: torch.Tensor
    twice_u_ids: torch.Tensor
    triplet_ids: torch.Tensor

    @classmethod
    def create(
        cls,
        pg: RadiusGraph,
        n_real: int,
        u_max: int,
        t_max: int,
        n_dummy: int,
    ) -> "_PaddedGraphWorkspace":
        dev = pg.lattice.device
        e_max = 2 * u_max
        edge_pair = torch.arange(e_max, device=dev, dtype=pg.atom_graph.dtype) // 2
        u_ids = torch.arange(u_max, device=dev, dtype=pg.undirected2directed.dtype)
        return cls(
            pg=pg,
            n_real=n_real,
            u_max=u_max,
            t_max=t_max,
            n_dummy=n_dummy,
            edge_pair=edge_pair,
            u_ids=u_ids,
            twice_u_ids=2 * u_ids,
            triplet_ids=torch.arange(t_max, device=dev, dtype=pg.line_graph.dtype),
        )

    def update(self, g: RadiusGraph) -> None:
        with nvtx_range("static input copy"):
            pg = self.pg
            n = g.atomic_number.shape[0]
            e = g.atom_graph.shape[0]
            u = g.undirected2directed.shape[0]
            t = g.line_graph.shape[0]
            e_max = 2 * self.u_max
            n_pad_u = self.u_max - u
            assert n == self.n_real, (n, self.n_real)
            assert e <= e_max and n_pad_u >= 1 and t <= self.t_max, (
                e,
                u,
                t,
                e_max,
                self.u_max,
                self.t_max,
            )

            pg.atomic_number[:n].copy_(g.atomic_number)
            pg.atom_frac_coord[:n].copy_(g.atom_frac_coord)
            pg.lattice.copy_(g.lattice)
            pg.atom_graph[:e].copy_(g.atom_graph)
            pg.neighbor_image[:e].copy_(g.neighbor_image)
            pg.directed2undirected[:e].copy_(g.directed2undirected)
            pg.undirected2directed[:u].copy_(g.undirected2directed)
            if t > 0:
                pg.line_graph[:t].copy_(g.line_graph)

            if e < e_max:
                e_pad = e_max - e
                pad_edge_sink = n + (
                    self.edge_pair[:e_pad] * self.n_dummy // n_pad_u
                )
                pg.atom_graph[e:, 0].copy_(pad_edge_sink)
                pg.atom_graph[e:, 1].copy_(pad_edge_sink)
                pg.neighbor_image[e:].fill_(3.0)
                pg.directed2undirected[e:e_max:2].copy_(self.u_ids[u:self.u_max])
                pg.directed2undirected[e + 1 : e_max : 2].copy_(self.u_ids[u:self.u_max])
                pg.undirected2directed[u:self.u_max].copy_(self.twice_u_ids[u:self.u_max])

            if t < self.t_max:
                t_pad = self.t_max - t
                triplet_ids = self.triplet_ids[:t_pad]
                pu = triplet_ids % n_pad_u
                pg.line_graph[t:, 0].copy_(
                    n + (triplet_ids * self.n_dummy // max(t_pad, 1))
                )
                pg.line_graph[t:, 1].copy_(u + pu)
                pg.line_graph[t:, 2].copy_(e + 2 * pu)
                pg.line_graph[t:, 3].copy_(u + pu)
                pg.line_graph[t:, 4].copy_(e + 2 * pu)


@dataclass
class _PaddedEdgesWorkspace:
    """The persistent batch of an edges-only bucket: the real edges of each ragged build, then
    padding rows as the fixed-capacity builder writes them (self-loop row k on real atom k mod n,
    or MatRIS's sink rows)."""

    batch: GraphBatch
    n_dummy: int
    e_max: int
    edge_pair: torch.Tensor | None  # sink padding: the padding pair of each row
    loop_atoms: torch.Tensor | None  # self-loop padding: the atom of padding row k
    loop_image: torch.Tensor | None  # far self-loops: the image row, rewritten from each update's cell
    loop_length: float  # far self-loops: the shortest padded edge
    pad_vector: torch.Tensor | None

    @classmethod
    def create(cls, sample: GraphBatch, u_max: int, rule: PaddingRule, n_dummy: int) -> "_PaddedEdgesWorkspace":
        dev = sample.edge_index.device
        n, e_max = sample.n_real, 2 * u_max
        sinks = n_dummy if rule.kind == "sink_atoms" else 0
        edge_index = sample.edge_index.new_zeros(e_max, 2)
        edge_image = sample.edge_image.new_zeros(e_max, 3)
        edge_pair = loop_atoms = loop_image = pad_vector = override = None
        loop_length = 0.0
        if sinks:
            edge_pair = torch.arange(e_max, device=dev, dtype=edge_index.dtype) // 2
        else:
            image, _, vector = self_loop_padding(rule, sample.cell, sample.cutoff)
            loop_atoms = (torch.arange(e_max, device=dev) % n).to(edge_index.dtype)
            loop_image = torch.tensor(image, dtype=edge_image.dtype, device=dev)
            if vector is not None:
                override = edge_image.new_zeros(e_max, 3)
                pad_vector = torch.tensor(vector, dtype=edge_image.dtype, device=dev)
            else:
                loop_length = max(rule.min_length, 2.0 * sample.cutoff)
        batch = GraphBatch(
            positions=sample.positions.new_zeros(n + sinks, 3),
            frac_coords=sample.frac_coords.new_zeros(n + sinks, 3),
            cell=sample.cell.clone(),
            species=torch.cat([sample.species, sample.species.new_ones(sinks)]),
            n_real=n,
            edge_index=edge_index,
            edge_image=edge_image,
            n_edges=torch.zeros(1, dtype=torch.int64, device=dev),
            cutoff=sample.cutoff,
            padding=rule,
            edge_vec_override=override,
            composition=sample.composition,
            evidence=sample.evidence,
        )
        return cls(batch, sinks, e_max, edge_pair, loop_atoms, loop_image, loop_length, pad_vector)

    def update(self, g: GraphBatch) -> None:
        with nvtx_range("static input copy"):
            pb = self.batch
            n, e, e_max = g.n_real, g.edge_index.shape[0], self.e_max
            assert n == pb.n_real and e <= e_max, (n, e, pb.n_real, e_max)
            pb.species[:n].copy_(g.species)
            pb.frac_coords[:n].copy_(g.frac_coords)
            pb.positions[:n].copy_(g.positions)
            pb.cell.copy_(g.cell)
            pb.edge_index[:e].copy_(g.edge_index)
            pb.edge_image[:e].copy_(g.edge_image)
            pb.n_edges.copy_(g.n_edges)
            if e < e_max and self.edge_pair is not None:
                pad_edge_sink = n + self.edge_pair[: e_max - e] * self.n_dummy // (e_max // 2 - e // 2)
                pb.edge_index[e:, 0].copy_(pad_edge_sink)
                pb.edge_index[e:, 1].copy_(pad_edge_sink)
                pb.edge_image[e:].fill_(3.0)
            elif e < e_max:
                if self.loop_length:
                    # Far self-loops: the fixed builder's image, from this cell (a calculator's
                    # bucket outlives a cell change): a whole number of its longest lattice vector.
                    lengths = torch.linalg.vector_norm(g.cell.double(), dim=1)
                    longest = lengths.max()
                    multiple = torch.ceil(self.loop_length / longest).clamp_min(1.0)
                    self.loop_image.copy_(torch.where(lengths == longest, multiple, 0.0)
                                          * (torch.cumsum(lengths == longest, 0) == 1))
                pb.edge_index[e:, 0].copy_(self.loop_atoms[: e_max - e])
                pb.edge_index[e:, 1].copy_(self.loop_atoms[: e_max - e])
                pb.edge_image[e:].copy_(self.loop_image.expand(e_max - e, 3))
            if self.pad_vector is not None:
                pb.edge_vec_override[:e].zero_()
                pb.edge_vec_override[e:].copy_(self.pad_vector.expand(e_max - e, 3))


@dataclass
class _PaddedLineGraphWorkspace:
    """A line-graph bucket of a model the engine differentiates: MatRIS's padded graph (sink rows) as a
    graph batch whose positions each update writes (``frac @ cell`` as the builders compute it; the sink
    rows stay at the origin)."""

    graph: _PaddedGraphWorkspace
    batch: GraphBatch

    @classmethod
    def create(cls, graph: _PaddedGraphWorkspace, sample: RadiusGraph) -> "_PaddedLineGraphWorkspace":
        positions = graph.pg.atom_frac_coord.new_zeros(graph.pg.atom_frac_coord.shape)
        positions[: graph.n_real].copy_(small_matmul(sample.atom_frac_coord, sample.lattice))
        return cls(graph, line_graph_batch(graph.pg, positions=positions, n_real=graph.n_real))

    def update(self, g: RadiusGraph) -> None:
        self.graph.update(g)
        with nvtx_range("static input copy"):
            self.batch.positions[: self.graph.n_real].copy_(small_matmul(g.atom_frac_coord, g.lattice))


@dataclass
class _LoopPaddedLineGraphWorkspace:
    """A line-graph bucket of a model the engine differentiates whose padding rule is far self-loops: the
    real rows of each ragged build, then padding as the fixed-capacity builder writes it (padded pair p:
    the two opposite self-loops of real atom p mod n along the longest lattice vector of the update's cell;
    a padded triplet joins a padded pair to itself at that atom). No sink rows."""

    batch: GraphBatch
    u_max: int
    t_max: int
    loop_length: float  # the shortest padded edge
    loop_image: torch.Tensor  # [3] the padded pairs' image, rewritten from each update's cell
    loop_atoms: torch.Tensor  # [2 u_max] the atom of padding row r (pair r // 2)
    loop_sign: torch.Tensor  # [2 u_max, 1] +1 on even padding rows, -1 on odd ones
    edge_pair: torch.Tensor  # [2 u_max] the padding pair of padding row r
    pair_ids: torch.Tensor  # [max(u_max, t_max)]

    @classmethod
    def create(cls, sample: RadiusGraph, u_max: int, t_max: int, rule: PaddingRule) -> "_LoopPaddedLineGraphWorkspace":
        dev = sample.lattice.device
        n, e_max = sample.atomic_number.shape[0], 2 * u_max
        rows = torch.arange(e_max, device=dev)
        batch = GraphBatch(
            positions=sample.atom_frac_coord.new_zeros(n, 3),
            frac_coords=sample.atom_frac_coord.new_zeros(n, 3),
            cell=sample.lattice.clone(),
            species=sample.atomic_number.new_zeros(n),
            n_real=n,
            edge_index=sample.atom_graph.new_zeros(e_max, 2),
            edge_image=sample.neighbor_image.new_zeros(e_max, 3),
            n_edges=None,
            cutoff=sample.atom_graph_cutoff,
            padding=rule,
            triplet_cutoff=sample.line_graph_cutoff,
            pair_d2u=sample.directed2undirected.new_zeros(e_max),
            pair_u2d=sample.undirected2directed.new_zeros(u_max),
            triplets=sample.line_graph.new_zeros(t_max, 5),
            composition=sample.composition or "",
            evidence=getattr(sample, "topology_evidence", None),
        )
        return cls(batch, u_max, t_max, max(rule.min_length, 2.0 * sample.atom_graph_cutoff),
                   sample.neighbor_image.new_zeros(3), (rows // 2 % n).to(batch.edge_index.dtype),
                   torch.where(rows % 2 == 0, 1.0, -1.0).to(batch.edge_image.dtype)[:, None],
                   rows // 2, torch.arange(max(u_max, t_max), device=dev))

    def update(self, g: RadiusGraph) -> None:
        with nvtx_range("static input copy"):
            pb = self.batch
            n, e = g.atomic_number.shape[0], g.atom_graph.shape[0]
            u, t = g.undirected2directed.shape[0], g.line_graph.shape[0]
            e_max = 2 * self.u_max
            assert n == pb.n_real and e == 2 * u and u < self.u_max and t <= self.t_max, (
                n, e, u, t, pb.n_real, self.u_max, self.t_max)
            pb.species.copy_(g.atomic_number)
            pb.frac_coords.copy_(g.atom_frac_coord)
            pb.cell.copy_(g.lattice)
            pb.positions.copy_(small_matmul(g.atom_frac_coord, g.lattice))
            pb.edge_index[:e].copy_(g.atom_graph)
            pb.edge_image[:e].copy_(g.neighbor_image)
            pb.pair_d2u[:e].copy_(g.directed2undirected)
            pb.pair_u2d[:u].copy_(g.undirected2directed)
            if t > 0:
                pb.triplets[:t].copy_(g.line_graph)
            # The fixed builder's image, from this cell (a calculator's bucket outlives a cell change).
            lengths = torch.linalg.vector_norm(g.lattice.double(), dim=1)
            longest = lengths.max()
            multiple = torch.ceil(self.loop_length / longest).clamp_min(1.0)
            self.loop_image.copy_(torch.where(lengths == longest, multiple, 0.0)
                                  * (torch.cumsum(lengths == longest, 0) == 1))
            pb.edge_index[e:, 0].copy_(self.loop_atoms[: e_max - e])
            pb.edge_index[e:, 1].copy_(self.loop_atoms[: e_max - e])
            pb.edge_image[e:].copy_(self.loop_sign[: e_max - e] * self.loop_image)
            pb.pair_d2u[e:].copy_(u + self.edge_pair[: e_max - e])
            pb.pair_u2d[u:].copy_(e + 2 * self.pair_ids[: self.u_max - u])
            if t < self.t_max:
                pair = self.pair_ids[: self.t_max - t] % (self.u_max - u)
                pb.triplets[t:, 0].copy_(pair % n)
                pb.triplets[t:, 1].copy_(u + pair)
                pb.triplets[t:, 2].copy_(e + 2 * pair)
                pb.triplets[t:, 3].copy_(u + pair)
                pb.triplets[t:, 4].copy_(e + 2 * pair)


def _demand(g) -> tuple[int, int, int]:
    """(atoms, undirected pairs, triplets) of a ragged build: MatRIS's graph, or an edges-only batch."""
    if isinstance(g, GraphBatch):
        return g.n_real, g.edge_index.shape[0] // 2, 0
    return g.atomic_number.shape[0], g.undirected2directed.shape[0], g.line_graph.shape[0]


def _cap_for(u, t, u_step, t_step, min_pad_u):
    """A model-graph bucket (``CapturePolicy.bucket_key``): u with ``min_pad_u``, t without a pad."""
    return tier(u, u_step, min_pad_u), tier(t, t_step, 0)


# The parameters below are CapturePolicy's (capacity_policy.py), shared with
# the whole step and the P>1 tiers. Growth headroom and minimum shrink saving
# (growth_headroom 1/128, shrink_min_saving 1/64), as fractions of a capacity: a
# grown bucket adds one step or whole steps up to 1/128 of it (1 step at 1,344
# RDX atoms, 7 triplet / 7 edge steps at 10,752). A growth (pool release, side-
# stream warmup, capture: ~2.7 s at 10,752 atoms, ~9 replays) then comes less
# often for <= 0.8% more padded rows, which cost ~1.2x their share of a replay
# (padding rows land on the few sink atoms). Measured on 1,000-step heating
# (300 -> 3000 K, 10,752 atoms) and replays of its demand traces 10x and 100x
# slower: 1/128 was within 1.1% of the best of 1/256..1/16 in every case but
# the md_pyrolysis-style test ramp (2.5%; 1/256 lost 7.5% there). A shrink must save
# 1/64 (or one step) in some dimension: twice the headroom, so it repays its
# capture within a window and does not ping-pong with growth.
# A system is small where one triplet step is at least precapture_min_step
# (1/256) of its capacity (up to ~2,900 RDX atoms): precapture keeps the tight
# buckets there, and the runner recaptures them after a growth (see _select),
# when their bytes fit (precapture_fits). Large systems precapture and keep
# only the envelope.
# A miss releases the pool only above release_above (1/8) of the device
# reserved; below it the requested bucket is captured into the live pool as
# before the fix (1,344 atoms: ~12-25 GiB of 277, so stationary and ramp MD
# keep all their buckets and pay no release; 10,752 atoms: ~100 GiB, every
# growth releases). The pool can then outgrow its working set by at most this much.


def _smallest_fitting_key(keys, u, t, u_step, t_step):
    fitting = [key for key in keys if key[0] > u and key[1] >= t]
    if not fitting:
        return None
    return min(
        fitting,
        key=lambda key: (
            (key[0] - u) / u_step + (key[1] - t) / t_step,
            key[0],
            key[1],
        ),
    )


class BucketedGraphRunner:
    """Captures one CUDA graph per capacity bucket; replays the matching one."""

    @scoped_init
    def __init__(self, model, task="efsm", u_step=512, t_step=8192,
                 n_dummy=64, min_pad_u=256, warmup=3,
                 enable_model_fusions=True, *, config=None, shrink_window=None,
                 release_above=None):
        self.adapter = as_adapter(model)
        self.model = self.adapter.module
        # None: the model differentiates itself (adapter.evaluate); else the engine-owned derivative.
        self._derivative = engine_derivative(self.adapter)
        self.task = task
        self.u_step, self.t_step = u_step, t_step
        self.n_dummy, self.min_pad_u, self.warmup = n_dummy, min_pad_u, warmup
        self.policy = CapturePolicy.from_config(
            None, u_step=u_step, t_step=t_step, n_dummy=n_dummy, min_pad_u=min_pad_u, warmup=warmup,
            **({} if shrink_window is None else {"hysteresis_window": int(shrink_window)}),
            **({} if release_above is None else {"release_above": float(release_above)}))
        self.enable_model_fusions = bool(enable_model_fusions)
        if config is not None and config.optimization_profile != "legacy":
            self.enable_model_fusions = config.optimization_profile != "generic"
        # The (pair, triplet) demand factors every bucket holds (the adapter's declared ones; 1: MatRIS).
        self._thermal_factors = thermal_factors(self.adapter, config)
        # Bucket graphs are mutually exclusive and replayed on one stream, so
        # their private allocations can safely reuse the same graph pool.
        self.pool = torch.cuda.graph_pool_handle()
        # Own capture stream (created at the first capture): torch keeps a
        # cuBLAS workspace per (handle, stream) in the first capturer's pool
        # and graph.reset() frees it, so on the shared default capture stream
        # a pool release here would free the workspace another runner's
        # graphs still use. torch.cuda.Stream() draws from torch's 32-stream
        # pool, so two live graph owners can still alias (rarely) one stream.
        self._capture_stream = None
        self.cache = {}
        self.captures = 0
        self.warmup_runs = 0  # side-stream warmups of every capture (CapturePolicy.warmups)
        self._bound_captures = 0  # captures since the runner was bound to its atom count
        self.cache_fit_reuses = 0
        self.pool_resets = 0
        self.shrinks = 0
        # Replays of a single cached bucket per shrink decision.
        self._shrink_window = self.policy.hysteresis_window
        self._window = [0, 0, 0]  # replays, max u, max t since the last pool reset
        self._n_real = None
        # Small systems: the tight buckets of precapture and of the first
        # capture, cached or released by a growth.
        self._tight = set()
        # The bucket a pressure release (grow/shrink) captured: the only one
        # the shrink window applies to; None before the first release.
        self._envelope = None
        # Fraction of device memory reserved above which a miss releases the
        # pool (0: always); the device total is read at the first miss.
        self._release_above = self.policy.release_above
        self._release_bytes = None
        self._capture_exact_capacity = False
        # Whether the neighbor buckets fit next to an envelope (precapture_fits),
        # per (atoms, envelope, recompute level): decided once, outside capture.
        self._neighbor_fits = {}
        self.precapture_note = None

    @contextmanager
    def exact_capacity_capture(self):
        previous = self._capture_exact_capacity
        self._capture_exact_capacity = True
        try:
            yield
        finally:
            self._capture_exact_capacity = previous

    @scoped_call
    def _capture(self, u_cap, t_cap, sample_g):
        if isinstance(sample_g, GraphBatch):
            # Edges only: the triplet capacity (that of an empty line graph) is unused.
            workspace = _PaddedEdgesWorkspace.create(sample_g, u_cap, self.adapter.padding_rule(), self.n_dummy)
            workspace.update(sample_g)
            pg, n_real = self.adapter.prepare(workspace.batch), sample_g.n_real
        elif self._derivative is not None and self.adapter.padding_rule().kind == "far_self_loops":
            # A line graph the engine differentiates, padded with far self-loops (no sink rows).
            workspace = _LoopPaddedLineGraphWorkspace.create(sample_g, u_cap, t_cap, self.adapter.padding_rule())
            workspace.update(sample_g)
            pg, n_real = self.adapter.prepare(workspace.batch), workspace.batch.n_real
        else:
            pg, n_real = pad_radius_graph(sample_g, u_cap, t_cap, self.n_dummy)
            workspace = _PaddedGraphWorkspace.create(
                pg=pg,
                n_real=n_real,
                u_max=u_cap,
                t_max=t_cap,
                n_dummy=self.n_dummy,
            )
            if self._derivative is not None:
                # A line graph the engine differentiates: the padded graph as a batch with positions.
                workspace = _PaddedLineGraphWorkspace.create(workspace, sample_g)
                pg = self.adapter.prepare(workspace.batch)
        with fusion_scope(self.enable_model_fusions):
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                if self._derivative is None:
                    self._warm(lambda: self.adapter.evaluate([pg], task=self.task, is_training=False, n_real=n_real))
                else:
                    self._warm(lambda: self._derivative(workspace.batch, pg, task=self.task, n_real=n_real))
            torch.cuda.current_stream().wait_stream(side)
            gr = torch.cuda.CUDAGraph()
            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream()
            with capture_graph(gr, pool=self.pool, stream=self._capture_stream):
                if self._derivative is None:
                    out = self.adapter.evaluate([pg], task=self.task, is_training=False, n_real=n_real)
                else:
                    out = self._derivative(workspace.batch, pg, task=self.task, n_real=n_real)
        self._n_real = n_real
        self.cache[(u_cap, t_cap)] = dict(
            pg=pg,
            graph=gr,
            out=out,
            n_real=n_real,
            workspace=workspace,
        )
        self.captures += 1

    def _warm(self, run):
        """The side-stream warmups of a capture (CapturePolicy.warmups): the configured count for
        this runner's first capture, one for a later one, plus one where recompute_policy "fast"
        still has to check its decision on an uncaptured run (saved_recompute.capture_check_pending)."""
        from fastmd.engine.saved_recompute import capture_check_pending
        first = self._bound_captures == 0  # also the first capture after a rebind (a new system)
        for _ in range(self.policy.warmups(first)):
            run()
            self.warmup_runs += 1
        if not first and self.policy.warmups(False) < self.warmup and capture_check_pending(self.model, self._scope()):
            run()
            self.warmup_runs += 1
        self._bound_captures += 1

    def _scope(self):
        """The configuration this runner's calls run under (its own, which GPU-resident MD
        replaces on a switch to model_graph, else the active one)."""
        from fastmd.config import current_config
        return getattr(self, "_inference_config", None) or current_config()

    def _neighbors_fit(self, n, envelope):
        """Whether neighbor buckets may join ``envelope`` in the pool (CapturePolicy precapture_fits),
        decided once per (atoms, envelope, recompute level) outside capture; True without a
        configuration (a bare runner) or CUDA."""
        from fastmd.engine.saved_recompute import current_mode
        config = self._scope()
        if config is None or not torch.cuda.is_available():
            return True
        level = current_mode(self.model, config)
        key = (n, envelope, level)
        if key not in self._neighbor_fits:
            from fastmd.engine.capacity_policy import graph_budget
            parameter = next(iter(self.model.parameters()), None)
            device = torch.device("cuda", torch.cuda.current_device()) if parameter is None else parameter.device
            fits, note = precapture_fits(self.model, config, envelope[1], level, graph_budget(self.model, device))
            self._neighbor_fits[key] = fits
            self.precapture_note = note
        return self._neighbor_fits[key]

    def _grown_key(self, requested_key):
        """Growth past every cached bucket: the envelope of the cached keys
        and ``requested_key`` plus the policy's headroom in each dimension that grew.
        It holds every demand a cached bucket held."""
        u_cap = max(key[0] for key in self.cache)
        t_cap = max(key[1] for key in self.cache)
        if requested_key[0] > u_cap:
            u_cap = requested_key[0] + self.policy.headroom(requested_key[0], self.u_step)
        if requested_key[1] > t_cap:
            t_cap = requested_key[1] + self.policy.headroom(requested_key[1], self.t_step)
        return u_cap, t_cap

    def _shrunk_key(self, key, u, t):
        """Hysteresis for a single cached bucket, which otherwise keeps a
        growth's size for the rest of the run: when this replay completes a
        window, the window's demand tier plus the policy's headroom, if that
        saves at least its ``shrink_min_saving`` (or one step) in a dimension."""
        replays, max_u, max_t = self._window
        if replays + 1 < self._shrink_window:
            return key
        demand_u, demand_t = max(max_u, u), max(max_t, t)
        tight_u, tight_t = self._held_key(demand_u, demand_t, _cap_for(demand_u, demand_t, self.u_step, self.t_step,
                                                                      self.min_pad_u))
        policy = self.policy
        shrunk = (min(key[0], tight_u + policy.headroom(tight_u, self.u_step)),
                  min(key[1], tight_t + policy.headroom(tight_t, self.t_step)))
        if (key[0] - shrunk[0] < max(self.u_step, key[0] * policy.shrink_min_saving)
                and key[1] - shrunk[1] < max(self.t_step, key[1] * policy.shrink_min_saving)):
            return key
        return shrunk

    def _under_pressure(self):
        """True when a miss should release the pool: reserved memory is at
        least ``_release_above`` of the device, or the device has less free
        memory than this process reserves (other tenants: a capture on top of
        the live pool needs about another working set plus its warmup, and
        memory admission counts the pool as reclaimable). Host-side queries,
        only on a miss."""
        if self._release_bytes is None:
            self._release_bytes = int(torch.cuda.get_device_properties(
                torch.cuda.current_device()).total_memory * self._release_above)
        reserved = torch.cuda.memory_reserved()
        return reserved >= self._release_bytes or torch.cuda.mem_get_info()[0] < reserved

    def _release_pool(self):
        """Drop every captured bucket and return the shared pool to the device.

        Shared-pool reuse only helps smaller tensors: a grown bucket's larger
        ones do not fit the blocks the cached buckets freed, so each growth
        capture added a whole new working set to the pool (10,752 atoms,
        legacy: ~13 GiB per triplet tier, 171 GiB of mostly free pool blocks
        by step 300 of a heating ramp -> OOM in the next warmup). Blocks
        return to the device only once every graph of the pool is gone.
        Outputs a caller still holds keep their blocks (never reused) until
        dropped. No gc.collect(): nothing of a released entry is held in a
        cycle (1,344 atoms: the pool falls from 11.6 GiB to the held forces'
        small segment without it), and it costs ~0.1 s per growth.
        """
        # A replay in flight still uses the pool.
        torch.cuda.synchronize()
        for ent in self.cache.values():
            ent["graph"].reset()
        self.cache.clear()
        self.pool = torch.cuda.graph_pool_handle()
        torch.cuda.empty_cache()
        self.pool_resets += 1
        self._window = [0, 0, 0]

    def _select(self, n, u, t):
        """(bucket, action) of run() for n atoms and demand (u, t), without
        changing state.

        action: None (replay), "capture" (the pool's other buckets stay),
        "grow"/"shrink" (release the pool, then capture: a bucket that is not
        smaller than every cached one would add its own working set), or
        "rebind" (graphs replay a fixed atom count: release, then capture).
        """
        requested_key = _cap_for(u, t, self.u_step, self.t_step, self.min_pad_u)
        if self.cache and n != self._n_real:
            return self._held_key(u, t, requested_key), "rebind"
        if self._capture_exact_capacity:
            if requested_key in self.cache:
                return requested_key, None
            if self.cache and (requested_key[0] > max(key[0] for key in self.cache)
                               or requested_key[1] > max(key[1] for key in self.cache)):
                if self._under_pressure():
                    return requested_key, "grow"  # the exact key, after a pool release
            return requested_key, "capture"
        if not self.cache:
            return self._held_key(u, t, requested_key), "capture"
        key = _smallest_fitting_key(self.cache, u, t, self.u_step, self.t_step)
        if key is None:
            if not self._under_pressure():
                return self._held_key(u, t, requested_key), "capture"
            return self._grown_key(self._held_key(u, t, requested_key)), "grow"
        if self._tight:
            # Small systems: a released tight bucket inside the envelope that
            # fits better is captured again into the live pool (its tensors
            # fit the envelope's blocks). After a tail event in stationary MD
            # the padding returns to the precapture level (1,344 atoms: 1.8%
            # -> ~0.5% of triplet rows). The pool is small at these sizes, so
            # its warmup on top of the pool is not a memory concern.
            u_cap = max(k[0] for k in self.cache)
            t_cap = max(k[1] for k in self.cache)
            best = _smallest_fitting_key(
                set(self.cache) | {k for k in self._tight if k[0] <= u_cap and k[1] <= t_cap},
                u, t, self.u_step, self.t_step)
            # Into the live pool next to the envelope: only while the bytes of
            # several buckets fit (a heating growth raises the envelope's rows).
            if best not in self.cache and self._neighbors_fit(n, (u_cap, t_cap)):
                return best, "capture"
        if key == self._envelope and set(self.cache) - self._tight == {key}:
            # The released pool's envelope, alone or next to recaptured tight
            # buckets (below the pressure valve nothing shrinks, as before).
            shrunk = self._shrunk_key(key, u, t)
            if shrunk != key:
                return shrunk, "shrink"
        return key, None

    def captures_next(self, g: RadiusGraph) -> bool:
        """True if ``run(g)`` would capture (run model code), not only replay."""
        return self._select(*_demand(g))[1] is not None

    def _bucket(self, g: RadiusGraph):
        _, u, t = _demand(g)
        return self._held_key(u, t, _cap_for(u, t, self.u_step, self.t_step, self.min_pad_u))

    def _held_key(self, u, t, key):
        """The bucket a capture holds for demand (u, t): ``key`` (its tier), or for an adapter's declared
        thermal factors the bucket of those factors times (u, t) (the counts of a heating system grow; fine
        tiers would capture at every step of that growth)."""
        u_factor, t_factor = self._thermal_factors
        if u_factor == 1.0 and t_factor == 1.0:
            return key
        return _cap_for(held_demand(u, 0, u_factor), held_demand(t, 0, t_factor), self.u_step, self.t_step,
                        self.min_pad_u)

    def ensure(self, sample_g: RadiusGraph, key):
        """Capture the (u_cap, t_cap) bucket if not already cached. sample_g must
        fit (its real counts <= capacity)."""
        if key not in self.cache:
            self._capture(key[0], key[1], sample_g)

    @scoped_call
    def precapture(self, sample_g: RadiusGraph, proactive: bool = True):
        """Pre-capture the bucket for sample_g and (optionally) the next-larger U/T
        neighbor buckets, without replaying — so later run() calls are pure replay.
        Larger buckets always fit sample_g (just more padding)."""
        self._precapture(_demand(sample_g)[0], *self._bucket(sample_g), sample_g, proactive)

    def _precapture(self, n, u0, t0, sample_g, proactive):
        """precapture() for n atoms and the demand bucket (u0, t0)."""
        keys = [(u0, t0)]
        if proactive:
            # The next-larger neighbors; the tight value of a dimension only
            # where one step is noticeable padding. At 10,752 RDX atoms a step
            # is ~0.1% of either capacity: only the largest bucket is captured,
            # which cuts the precapture peak (pool + one more warmup) from 189
            # to ~101 GiB at the same ms/step. Small systems keep all four.
            us = [u0 + self.u_step] + ([u0] if self.policy.small(u0, self.u_step) else [])
            ts = [t0 + self.t_step] + ([t0] if self.policy.small(t0, self.t_step) else [])
            if isinstance(sample_g, GraphBatch):
                ts = [t0]  # edges only: the triplet capacity does not grow
            keys = [(u, t) for u in us for t in ts]
        # Largest (triplet-sized tensors first) to smallest: the smaller
        # buckets then fit the shared pool's freed blocks instead of adding
        # their own (10,752 atoms, legacy: 90 -> 120 GiB in the other order).
        keys.sort(key=lambda key: (key[1], key[0]), reverse=True)
        if self.cache and self._n_real != n:
            self._release_pool()
            self._tight.clear()
            self._envelope = None
            self._bound_captures = 0
        elif self.cache and any(k[0] > max(c[0] for c in self.cache) or k[1] > max(c[1] for c in self.cache)
                                for k in keys) and self._under_pressure():
            # A key beyond the cached envelope would grow the pool (run()).
            self._release_pool()
            self._envelope = None
        elif self.cache and keys[0] not in self.cache and not self._neighbors_fit(
                n, (max([c[0] for c in self.cache] + [keys[0][0]]), max([c[1] for c in self.cache] + [keys[0][1]]))):
            # The envelope would join a live pool (e.g. after a calculator's
            # probe run) whose buckets and it do not fit together: one at a time.
            self._release_pool()
            self._envelope = None
        # The envelope first; the neighbors only if their bytes fit next to it
        # (decided after its capture: the recompute level is then the running
        # one). Each further bucket costs about one more working set in the
        # pool (10M model, 2,688 RDX atoms, recompute on: 130 GiB, then an OOM
        # in the second bucket's warmup).
        self.ensure(sample_g, keys[0])
        denied = len(keys) > 1 and not self._neighbors_fit(n, keys[0])
        for k in keys[1:] if not denied else ():
            self.ensure(sample_g, k)
        if self.policy.small(t0, self.t_step) and not denied:
            self._tight.update(keys)

    @scoped_call
    def run(self, g: RadiusGraph):
        """Return (out_dict, n_real). out_dict tensors are valid until the next run().

        NB: grad must stay enabled — the model computes conservative forces via an
        internal torch.autograd.grad; the copy_ of inputs is wrapped in no_grad.
        """
        with nvtx_range("BucketedGraphRunner.run"):
            ent = self._bucket_entry(*_demand(g), g)
            with torch.no_grad():
                ent["workspace"].update(g)
            with nvtx_range("cudaGraphLaunch"):
                ent["graph"].replay()
            return ent["out"], ent["n_real"]

    def _bucket_entry(self, n, u, t, sample_g):
        """The cache entry run() replays for n atoms and demand (u, t): selects
        the bucket and captures (after a pool release if needed) on a miss."""
        key, action = self._select(n, u, t)
        if action in ("grow", "shrink", "rebind"):
            self._release_pool()
            self.shrinks += action == "shrink"
            if action == "rebind":
                self._tight.clear()
                self._bound_captures = 0
        if action is not None:
            self._capture(*key, sample_g)
            if action in ("grow", "shrink", "rebind"):
                self._envelope = key if action != "rebind" else None
            if action in ("capture", "rebind") and not self._tight and self.policy.small(key[1], self.t_step):
                self._tight.add(key)
        else:
            if key != _cap_for(u, t, self.u_step, self.t_step, self.min_pad_u):
                self.cache_fit_reuses += 1
            if (self._envelope is not None and set(self.cache) - self._tight == {self._envelope}
                    and not self._capture_exact_capacity):
                window = self._window
                window[0] += 1
                window[1] = max(window[1], u)
                window[2] = max(window[2], t)
                if window[0] >= self._shrink_window:
                    self._window = [0, 0, 0]
        return self.cache[key]

    def stats(self) -> dict[str, object]:
        return {
            "captures": self.captures,
            "warmup_runs": self.warmup_runs,
            "cache_fit_reuses": self.cache_fit_reuses,
            "pool_resets": self.pool_resets,
            "shrinks": self.shrinks,
            "cached_buckets": [list(key) for key in sorted(self.cache)],
            "enable_model_fusions": self.enable_model_fusions,
            "shared_graph_pool": True,
        }
