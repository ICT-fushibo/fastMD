"""Fixed-capacity graph builder at the edge-capacity boundary, and its edges-only mode.

When the real directed-edge count equals ``e_capacity`` no padding undirected
pair is left, yet the padding triplets are routed to one.  Before the fix they
indexed ``u == u_capacity`` (one past the end) and the overflow flag stayed
clear, so a captured whole step hit a CUDA device-side assert in the model's
line-graph segment reductions.

The edges-only mode (no line graph) routes nothing to padding rows: its edges
fit up to ``e_capacity`` inclusive. Its edges equal a dense FP64 reference and
MatRIS mode's, every padding rule adds exactly nothing with an exact-cutoff
potential, and a reused candidate list gives the rows of a fresh one.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fastmd.adapters import MatRISAdapter
from fastmd.adapters.protocol import PaddingRule
from fastmd.graph import FixedCapacityGraphBuilder
from fastmd.graph import validate_staticization_invariants
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.fixed_capacity_builder import fixed_capacity_exceeded
from fastmd.graph.geometry import small_matmul
from pair_potential_adapter import SmoothPairPotential
from helpers import deterministic_algorithms

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
CUTOFF, LINE_CUTOFF, CANDIDATE_CUTOFF = 3.0, 2.2, 3.5
N_DUMMY = 4


def _crystal() -> tuple[torch.Tensor, torch.Tensor]:
    """Perturbed 3x3x3 simple-cubic crystal in a slightly triclinic cell."""
    rng = np.random.default_rng(1234)
    cell = np.array([[6.0, 0.0, 0.0], [0.3, 6.1, 0.0], [0.2, -0.25, 5.9]])
    grid = np.stack(np.meshgrid(*(np.arange(3),) * 3, indexing="ij"), -1).reshape(-1, 3)
    frac = (grid + 0.5 + rng.uniform(-0.08, 0.08, grid.shape)) / 3.0
    positions = frac @ cell
    return (
        torch.tensor(cell, dtype=torch.float32),
        torch.tensor(positions, dtype=torch.float32, device="cuda"),
    )


def _builder(cell: torch.Tensor, n_atoms: int, u_capacity: int, t_capacity: int):
    return FixedCapacityGraphBuilder(
        cell=cell,
        atomic_numbers=torch.full((n_atoms,), 14, dtype=torch.int32),
        u_capacity=u_capacity,
        t_capacity=t_capacity,
        atom_graph_cutoff=CUTOFF,
        line_graph_cutoff=LINE_CUTOFF,
        candidate_cutoff=CANDIDATE_CUTOFF,
        n_dummy=N_DUMMY,
        device="cuda",
    )


def _edges_only_builder(cell, n_atoms, u_capacity, padding=None, **kwargs):
    return FixedCapacityGraphBuilder(
        cell=cell,
        atomic_numbers=torch.full((n_atoms,), 14, dtype=torch.int32),
        u_capacity=u_capacity,
        t_capacity=0,
        atom_graph_cutoff=CUTOFF,
        line_graph_cutoff=None,
        candidate_cutoff=CANDIDATE_CUTOFF,
        n_dummy=N_DUMMY,
        device="cuda",
        padding=padding,
        **kwargs,
    )


def _dense_pairs(frac: torch.Tensor, cell: torch.Tensor, cutoff: float):
    """FP64 O(N^2 x images) reference of the directed edges (i, j, image) of the wrapped
    fractional coordinates: {(i, j, nx, ny, nz): distance}."""
    frac64, cell64 = frac.double().cpu(), cell.double().cpu()
    positions = frac64 @ cell64
    heights = cell64.det().abs() / torch.stack(
        [torch.linalg.cross(cell64[(k + 1) % 3], cell64[(k + 2) % 3]).norm() for k in range(3)])
    reach = [int(np.ceil(cutoff / float(h))) + 1 for h in heights]
    images = torch.cartesian_prod(*(torch.arange(-r, r + 1, dtype=torch.float64) for r in reach))
    pairs = {}
    for image in images:
        vectors = positions[None, :, :] + (image @ cell64)[None, None, :] - positions[:, None, :]
        distances = vectors.norm(dim=-1)
        for i, j in (distances < cutoff).nonzero().tolist():
            if i != j or bool(image.abs().sum() > 0):
                pairs[(i, j, *map(int, image.tolist()))] = float(distances[i, j])
    return pairs


def _batch_pairs(batch):
    edges = int(batch.n_edges.item())
    rows = torch.cat([batch.edge_index[:edges].long(), batch.edge_image[:edges].round().long()], 1)
    return [tuple(row) for row in rows.tolist()]


class FixedCapacityPredicateTest(unittest.TestCase):
    def test_edges_only_fits_up_to_capacity(self) -> None:
        builder = SimpleNamespace(e_capacity=32, t_capacity=0, edges_only=True)
        for edges, invalid, expected in ((31, False, False), (32, False, False), (33, False, True),
                                         (34, False, True), (30, True, True)):
            status = SimpleNamespace(edge_count=torch.tensor([edges]), triplet_count=torch.tensor([0]),
                                     invalid_pairs=torch.tensor([invalid]))
            with self.subTest(edges=edges, invalid=invalid):
                self.assertEqual(bool(fixed_capacity_exceeded(status, builder)), expected)

    def test_padding_rule_rejects_invalid_fields(self) -> None:
        for kwargs, message in (({"kind": "zero_loops"}, "unknown kind"), ({"min_length": -1.0}, "min_length"),
                                ({"min_length": float("nan")}, "min_length"),
                                ({"kind": "sink_atoms", "min_length": 1.0}, "far_self_loops only"),
                                ({"kind": "far_self_loops", "pad_vector": (1.0, 0.0, 0.0)}, "exact_rc_loops only"),
                                ({"kind": "exact_rc_loops", "pad_vector": (1.0, 0.0)}, "three finite"),
                                ({"placement": "one_row"}, "unknown placement")):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, message):
                PaddingRule(**kwargs)

    def test_edge_count_equal_to_capacity_is_an_overflow(self) -> None:
        builder = SimpleNamespace(e_capacity=32, t_capacity=64)

        def status(edges, triplets=0, invalid=False):
            return SimpleNamespace(
                edge_count=torch.tensor([edges]),
                triplet_count=torch.tensor([triplets]),
                invalid_pairs=torch.tensor([invalid]),
            )

        for edges, triplets, invalid, expected in (
            (30, 64, False, False),  # one padding pair, full triplet capacity
            (32, 0, False, True),  # no padding pair left
            (34, 0, False, True),
            (30, 65, False, True),
            (30, 0, True, True),
        ):
            with self.subTest(edges=edges, triplets=triplets, invalid=invalid):
                self.assertEqual(
                    bool(fixed_capacity_exceeded(status(edges, triplets, invalid), builder)),
                    expected,
                )


@unittest.skipUnless(
    torch.cuda.is_available() and _fixed_builder_available,
    "CUDA fixed-capacity graph builder is unavailable",
)
class FixedCapacityBoundaryTest(unittest.TestCase):
    def test_capacity_boundary_flags_overflow_and_keeps_indices_in_range(self) -> None:
        cell, positions = _crystal()
        n_real = positions.shape[0]
        reference_builder = _builder(cell, n_real, 4096, 65536)
        reference, reference_status = reference_builder.build(positions)
        edges = int(reference_status.edge_count.item())
        triplets = int(reference_status.triplet_count.item())
        self.assertFalse(bool(reference_status.overflow.item()))
        self.assertEqual(edges % 2, 0)
        self.assertGreater(triplets, 0)
        real_u = edges // 2
        # leave padding triplets in every case so they must be routed to a pair
        t_capacity = triplets + 32

        # A tiny open cluster far above capacity (u_capacity=1, two dummies).
        cluster = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device="cuda",
        )
        cluster_builder = FixedCapacityGraphBuilder(
            cell=torch.eye(3) * 20.0,
            atomic_numbers=torch.ones(4, dtype=torch.int32),
            pbc=torch.zeros(3, dtype=torch.bool),
            u_capacity=1,
            t_capacity=4,
            atom_graph_cutoff=2.0,
            line_graph_cutoff=1.5,
            candidate_cutoff=2.5,
            n_dummy=2,
            device="cuda",
        )
        cases = [
            ("crystal", _builder(cell, n_real, u_capacity, t_capacity), positions, expected_overflow)
            for u_capacity, expected_overflow in (
                (real_u - 1, True),  # edges > e_capacity
                (real_u, True),  # edges == e_capacity: no padding pair left
                (real_u + 1, False),  # exactly one padding pair
            )
        ]
        cases.append(("cluster", cluster_builder, cluster, True))
        for name, builder, coordinates, expected_overflow in cases:
            u_capacity = builder.u_capacity
            with self.subTest(system=name, u_capacity=u_capacity):
                graph, status = builder.build(coordinates)
                torch.cuda.synchronize()
                if name == "crystal":
                    self.assertEqual(int(status.edge_count.item()), edges)
                else:
                    self.assertGreater(int(status.edge_count.item()), builder.e_capacity)
                self.assertEqual(bool(status.overflow.item()), expected_overflow)
                self.assertEqual(
                    bool(fixed_capacity_exceeded(status, builder).item()),
                    expected_overflow,
                )
                # every index the model gathers/scatters with must be in range
                n_all = builder.n_real + builder.n_dummy
                line = graph.line_graph.long()
                self.assertTrue(bool((graph.atom_graph.long() < n_all).all()))
                self.assertTrue(bool((graph.atom_graph.long() >= 0).all()))
                self.assertTrue(bool((graph.directed2undirected < u_capacity).all()))
                self.assertTrue(bool((graph.undirected2directed < 2 * u_capacity).all()))
                self.assertTrue(bool((line[:, 0] < n_all).all()))
                self.assertLess(int(line[:, [1, 3]].max()), u_capacity)
                self.assertLess(int(line[:, [2, 4]].max()), 2 * u_capacity)
                self.assertGreaterEqual(int(line.min()), 0)
                if expected_overflow:
                    # the overflowed topology is a bounded all-sink graph
                    self.assertTrue(bool((graph.atom_graph >= builder.n_real).all()))
                    self.assertTrue(bool((line[:, 0] >= builder.n_real).all()))
                else:
                    report = validate_staticization_invariants(
                        graph, status, n_real=n_real
                    )
                    self.assertEqual(report.edge_count, edges)
                    self.assertEqual(report.triplet_count, triplets)
                    # all padding triplets use the single padding pair
                    self.assertTrue(bool((line[triplets:, [1, 3]] == real_u).all()))
                    self.assertTrue(bool((line[triplets:, [2, 4]] == edges).all()))

    def test_minimum_image_reuse_matches_fresh_build(self) -> None:
        # A shared candidate list reused after small moves and lattice jumps
        # (ASE wrap) gives bitwise the graph of a fresh candidate build.
        cell, positions = _crystal()
        n_real = positions.shape[0]
        common = dict(cell=cell, atomic_numbers=torch.full((n_real,), 14, dtype=torch.int32),
                      atom_graph_cutoff=CUTOFF, line_graph_cutoff=LINE_CUTOFF,
                      candidate_cutoff=CANDIDATE_CUTOFF, n_dummy=N_DUMMY, device="cuda", minimum_image=True)
        owner = FixedCapacityGraphBuilder(u_capacity=1, t_capacity=1, **common)
        owner.refresh_candidates(positions)
        reused = FixedCapacityGraphBuilder(u_capacity=4096, t_capacity=65536, candidates_from=owner, **common)
        fresh = FixedCapacityGraphBuilder(u_capacity=4096, t_capacity=65536, **common)
        generator = torch.Generator(device="cuda").manual_seed(5)
        skin = CANDIDATE_CUTOFF - CUTOFF
        lattice = cell.to("cuda")
        for step in range(4):
            moved = positions + 0.4 * skin * (torch.rand(positions.shape, device="cuda", generator=generator) - 0.5)
            moved[step::5] += lattice[step % 3] * (1 if step % 2 else -2)  # wrapped by the caller
            graph, status = reused.build(moved, rebuild_candidates=False)
            self.assertFalse(bool(status.overflow.item()))
            expected, _ = fresh.build(moved, rebuild_candidates=True)
            report = validate_staticization_invariants(graph, status, n_real=n_real, reference=expected)
            self.assertTrue(report.reference_exact)
        # Moving an atom beyond half the skin invalidates the reused list.
        jumped = positions.clone()
        jumped[0, 0] += 0.6 * skin
        _, status = reused.build(jumped, rebuild_candidates=False)
        self.assertTrue(bool(status.overflow.item()))
        self.assertFalse(bool(status.candidate_valid.item()))

    def test_matris_adapter_prepare_is_the_builders_graph(self) -> None:
        # GPU-resident MD and the skin runner hand MatRIS's adapter.prepare to the
        # builder: the RadiusGraph over the published batch, the same tensor objects.
        cell, positions = _crystal()
        builder = _builder(cell, positions.shape[0], 4096, 65536)
        graph, _ = builder.build(positions)
        view = MatRISAdapter(None).prepare(builder.batch)
        self.assertIsNot(view, graph)
        for field in ("atomic_number", "atom_frac_coord", "lattice", "neighbor_image", "atom_graph",
                      "line_graph", "directed2undirected", "undirected2directed"):
            self.assertIs(getattr(view, field), getattr(graph, field), field)
        for field in ("atom_graph_cutoff", "line_graph_cutoff", "atom_target_sorted", "line_atom_sorted",
                      "composition", "topology_evidence"):
            self.assertEqual(getattr(view, field), getattr(graph, field), field)
        self.assertIsNone(builder.batch.positions)
        self.assertEqual(MatRISAdapter(None).padding_rule().kind, "sink_atoms")

    def test_whole_step_replay_at_capacity_boundary_grows(self) -> None:
        # Separate process: the pre-fix failure is a device-side assert that
        # would poison the CUDA context of this test process. The worker also
        # checks the initial evaluation (single and batched runner), bitwise
        # rollback of an overflowing window (single runner with 1 and 3 steps
        # per launch, batched runner) and GPU MD growing before its first
        # capture.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO), str(TESTS), env.get("PYTHONPATH", "")]
        )
        proc = subprocess.run(
            [sys.executable, str(TESTS / "fixed_capacity_boundary_worker.py")],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
        self.assertEqual(
            proc.returncode,
            0,
            f"worker failed\nstdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}",
        )
        self.assertEqual(len(lines), 1, proc.stdout)
        result = json.loads(lines[0][len("RESULT "):])
        self.assertTrue(result["ok"])
        self.assertEqual(result["runner_overflow_edges"], 2 * (result["u_base"] + 1))
        self.assertEqual(
            result["md_capacity_history"],
            [[result["u_base"] + 1, 64], [result["u_dense"] + 1, 64]],
        )
        self.assertEqual(result["initial_overflow_edges"], 2 * result["u_dense"])
        self.assertEqual(result["rollback_steps_per_launch"], [1, 3])
        self.assertEqual(result["batched_overflow_edges"], 2 * (result["u_base"] + 1))
        self.assertEqual(
            result["md_initial_capacity_history"],
            [[result["u_dense"], 64], [result["u_dense"] + 1, 64]],
        )


@unittest.skipUnless(
    torch.cuda.is_available() and _fixed_builder_available,
    "CUDA fixed-capacity graph builder is unavailable",
)
class EdgesOnlyBuilderTest(unittest.TestCase):
    RULES = (PaddingRule("far_self_loops"), PaddingRule("exact_rc_loops"), PaddingRule("sink_atoms"))

    def setUp(self) -> None:
        self.cell, self.positions = _crystal()
        self.n = self.positions.shape[0]
        reference = _edges_only_builder(self.cell, self.n, 4096)
        _, status = reference.build(self.positions)
        self.edges = int(status.edge_count.item())

    def test_edges_equal_dense_reference_and_matris_mode(self) -> None:
        builder = _edges_only_builder(self.cell, self.n, 4096)
        graph, status = builder.build(self.positions)
        batch = builder.batch
        self.assertIs(graph, batch)  # default prepare: the batch itself
        self.assertTrue(torch.equal(batch.edge_dst, batch.edge_index[:, 0]))
        self.assertTrue(torch.equal(batch.edge_src, batch.edge_index[:, 1]))
        self.assertFalse(bool(status.overflow.item()))
        self.assertIsNone(batch.triplets)
        self.assertIsNone(batch.pair_d2u)
        self.assertEqual(int(status.triplet_count.item()), 0)
        n, edges = self.n, self.edges
        # The rows of MatRIS mode's atom graph, in the same order.
        reference, _ = _builder(self.cell, n, 4096, 65536).build(self.positions)
        self.assertTrue(torch.equal(batch.edge_index[:edges], reference.atom_graph[:edges]))
        self.assertTrue(torch.equal(batch.edge_image[:edges], reference.neighbor_image[:edges]))
        self.assertTrue(torch.equal(batch.frac_coords[:n], reference.atom_frac_coord[:n]))
        # The pair set of a dense FP64 search of the same wrapped coordinates (triclinic cell),
        # up to pairs within float32 rounding of the cutoff.
        dense = _dense_pairs(batch.frac_coords[:n], batch.cell, CUTOFF + 1.0e-3)
        mine = set(_batch_pairs(batch))
        self.assertEqual(len(mine), edges)
        theirs = {key for key, distance in dense.items() if distance < CUTOFF}
        for key in mine ^ theirs:
            self.assertLess(abs(dense[key] - CUTOFF), 1.0e-5, key)
        # The published wrapped coordinates (the builder's own arithmetic, no TF32 GEMM).
        self.assertTrue(torch.equal(batch.positions[:n], small_matmul(batch.frac_coords[:n], batch.cell)))

    def test_capacity_boundary_and_padding_layout(self) -> None:
        n, edges, real_u = self.n, self.edges, self.edges // 2
        for vector in ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)):  # inside the cutoff: not inert
            with self.subTest(pad_vector=vector), self.assertRaisesRegex(ValueError, "shorter than the cutoff"):
                _edges_only_builder(self.cell, n, real_u, padding=PaddingRule("exact_rc_loops", pad_vector=vector))
        for rule in self.RULES:
            # The last capacity pads more rows than there are atoms: the round robin wraps.
            for u_capacity, expected_overflow in ((real_u - 1, True), (real_u, False), (real_u + 40, False)):
                with self.subTest(kind=rule.kind, u_capacity=u_capacity):
                    builder = _edges_only_builder(self.cell, n, u_capacity, padding=rule)
                    batch = builder.batch
                    _, status = builder.build(self.positions)
                    self.assertEqual(int(status.edge_count.item()), edges)
                    self.assertEqual(bool(status.overflow.item()), expected_overflow)
                    self.assertEqual(bool(fixed_capacity_exceeded(status, builder).item()), expected_overflow)
                    rows = batch.edge_index.long()
                    self.assertTrue(bool(((rows >= 0) & (rows < n + builder.n_dummy)).all()))
                    if expected_overflow:
                        continue
                    pad = builder.e_capacity - edges
                    center, neighbor = rows[edges:, 0], rows[edges:, 1]
                    self.assertTrue(torch.equal(center, neighbor))
                    if rule.kind == "sink_atoms":
                        self.assertTrue(bool((center >= n).all()))
                        self.assertTrue(bool((batch.edge_image[edges:] == 3.0).all()))
                        continue
                    # Round robin over the real atoms.
                    self.assertTrue(torch.equal(center, torch.arange(pad, device="cuda") % n))
                    images = batch.edge_image[edges:]
                    shift = small_matmul(images, batch.cell)
                    if rule.kind == "far_self_loops":
                        self.assertTrue(bool((shift.norm(dim=1) >= 2 * CUTOFF).all()))
                    else:
                        self.assertTrue(bool((images == 0).all()))
                        override = batch.edge_vec_override
                        self.assertTrue(bool((override[:edges] == 0).all()))
                        self.assertTrue(bool((override[edges:] == torch.tensor([CUTOFF, 0.0, 0.0],
                                                                                 device="cuda")).all()))

    def test_padding_rules_add_nothing(self) -> None:
        # The pair potential has an exact cutoff: every padded row's energy and
        # force are exactly 0, and energy and forces equal those of the real
        # rows alone bit for bit (deterministic scatters).
        import dataclasses

        potential = SmoothPairPotential(cutoff=CUTOFF).cuda()
        with deterministic_algorithms():
            for rule in self.RULES:
                with self.subTest(kind=rule.kind):
                    builder = _edges_only_builder(self.cell, self.n, self.edges // 2 + 37, padding=rule)
                    builder.build(self.positions)
                    batch = builder.batch
                    energy, forces, _ = potential(batch, n_real=self.n)
                    edges = self.edges
                    override = batch.edge_vec_override
                    real = dataclasses.replace(
                        batch, edge_index=batch.edge_index[:edges], edge_image=batch.edge_image[:edges],
                        edge_vec_override=None if override is None else override[:edges])
                    real_energy, real_forces, _ = potential(real, n_real=self.n)
                    self.assertTrue(torch.equal(energy, real_energy))
                    self.assertTrue(torch.equal(forces[: self.n], real_forces[: self.n]))
                    # Each padded row alone: energy and gradient exactly 0.
                    pad = dataclasses.replace(
                        batch, edge_index=batch.edge_index[edges:], edge_image=batch.edge_image[edges:],
                        n_edges=torch.zeros_like(batch.n_edges),
                        edge_vec_override=None if override is None else override[edges:])
                    pad_energy, pad_forces, _ = potential(pad, n_real=self.n)
                    self.assertEqual(float(pad_energy), 0.0)
                    self.assertEqual(float(pad_forces.abs().max()), 0.0)

    def test_reused_candidates_give_the_rows_of_a_fresh_build(self) -> None:
        common = dict(minimum_image=True)
        owner = _edges_only_builder(self.cell, self.n, 1, **common)
        owner.refresh_candidates(self.positions)
        reused = _edges_only_builder(self.cell, self.n, 4096, candidates_from=owner, **common)
        fresh = _edges_only_builder(self.cell, self.n, 4096, **common)
        generator = torch.Generator(device="cuda").manual_seed(7)
        skin = CANDIDATE_CUTOFF - CUTOFF
        lattice = self.cell.to("cuda")
        for step in range(4):
            moved = self.positions + 0.4 * skin * (torch.rand(self.positions.shape, device="cuda",
                                                                generator=generator) - 0.5)
            moved[step::5] += lattice[step % 3] * (1 if step % 2 else -2)  # wrapped by the caller
            _, status = reused.build(moved, rebuild_candidates=False)
            self.assertFalse(bool(status.overflow.item()))
            fresh.build(moved, rebuild_candidates=True)
            for field in ("edge_index", "edge_image", "positions", "frac_coords", "n_edges"):
                self.assertTrue(torch.equal(getattr(reused.batch, field), getattr(fresh.batch, field)), field)
        jumped = self.positions.clone()
        jumped[0, 0] += 0.6 * skin
        _, status = reused.build(jumped, rebuild_candidates=False)
        self.assertTrue(bool(status.overflow.item()))
        self.assertFalse(bool(status.candidate_valid.item()))

    def test_float64_unsorted_kernels_equal_the_general_sequence(self) -> None:
        # A float64 build with unsorted edges and self-loop padding writes its rows, padding and flags in three
        # kernels: bit for bit the general sequence's (the second builder shares the first one's candidates and
        # runs the general sequence), with more edges than rows and with a move past the skin too.
        cell64, positions = self.cell.double(), self.positions.double()
        common = dict(padding=PaddingRule("far_self_loops"), positions_dtype=torch.float64, sorted_edges=False)
        skin = CANDIDATE_CUTOFF - CUTOFF
        for u_capacity in (4096, self.edges // 4):  # the second: about half the rows the edges need
            fused = _edges_only_builder(cell64, self.n, u_capacity, **common)
            general = _edges_only_builder(cell64, self.n, u_capacity, candidates_from=fused, **common)
            self.assertTrue(fused._unsorted_edges)
            general._unsorted_edges = False
            fused.refresh_candidates(positions)
            generator = torch.Generator(device="cuda").manual_seed(11)
            for step in range(4):
                scale = 1.6 if step == 2 else 0.4
                moved = positions + scale * skin * (torch.rand(positions.shape, device="cuda", dtype=torch.float64,
                                                               generator=generator) - 0.5)
                for builder in (fused, general):  # stale values that a missing write would leave
                    builder.batch.edge_index.fill_(-7)
                    builder.batch.edge_image.fill_(-7.0)
                    builder._overflow.fill_(True)
                    builder._isolated_atom_count.fill_(99)
                _, status = fused.build(moved, rebuild_candidates=False)
                general.build(moved, rebuild_candidates=False)
                self.assertEqual(bool(status.overflow.item()), step == 2 or u_capacity < 4096)
                for name in ("center", "neighbor", "images", "edge_offsets", "edge_counts", "_overflow",
                             "_isolated_atom_count", "_candidate_valid", "_max_displacement", "positions"):
                    self.assertTrue(torch.equal(getattr(fused, name), getattr(general, name)), name)
                rows = min(int(status.edge_count.item()), fused.e_capacity)  # no path writes a padding distance
                self.assertTrue(torch.equal(fused.distances[:rows], general.distances[:rows]))
                for name in ("edge_index", "edge_image"):
                    self.assertTrue(torch.equal(getattr(fused.batch, name), getattr(general.batch, name)), name)

    def test_gpu_md_modes_agree_through_the_adapter(self) -> None:
        # GPU-resident MD with an edges-only adapter: eager (a ragged build per
        # step), model_graph (padded buckets) and whole_step (skin reuse across
        # several candidate refreshes) give the same trajectory bit for bit.
        from ase import Atoms

        from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
        from fastmd.execution.gpu_nhc import maxwell_boltzmann_momenta
        from fastmd.config import CapacityConfig
        from fastmd.config import InferenceConfig
        from pair_potential_adapter import PairPotentialAdapter

        atoms = Atoms("Si27", positions=self.positions.double().cpu().numpy(), cell=self.cell.double().numpy(),
                      pbc=True)
        atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), 3000.0, seed=11))
        capacity = CapacityConfig(u_step=8, t_step=32, min_pad_u=4, min_pad_t=8, n_dummy=N_DUMMY, warmup=1,
                                  candidate_skin=CANDIDATE_CUTOFF - CUTOFF)
        results = {}
        with deterministic_algorithms():
            for padding in self.RULES:
                for execution in ("eager", "model_graph", "whole_step"):
                    adapter = PairPotentialAdapter(SmoothPairPotential(cutoff=CUTOFF), padding=padding)
                    config = InferenceConfig(execution=execution, report=False, capacity=capacity,
                                             matmul_precision="highest")
                    with GPUResidentMolecularDynamics(atoms.copy(), model=adapter, ensemble="nve", timestep=1.0,
                                                      config=config) as md:
                        md.run(40)
                        if execution == "whole_step":
                            self.assertGreaterEqual(md.whole_step_stats()["candidate_refreshes"], 3)
                            state = md.whole_step_runner.state
                        else:
                            state = md.state
                        results[padding.kind, execution] = (state.positions.clone(), state.forces.clone(),
                                                            state.potential_energy.reshape(()).clone())
        reference = results["far_self_loops", "eager"]
        self.assertGreater(float(reference[1].abs().max()), 0.0)
        for key, value in results.items():
            with self.subTest(padding=key[0], execution=key[1]):
                for got, expected in zip(value, reference):
                    self.assertTrue(torch.equal(got, expected))


if __name__ == "__main__":
    unittest.main()
