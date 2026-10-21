"""Tiny, download-free CUDA regressions for the public inference config API,
and for the replica entry point (ReplicaMolecularDynamics) with the test pair
potential.

Run only on an idle device, e.g. CUDA_VISIBLE_DEVICES=0 python -m unittest
discover -s tests -p test_config_cuda.py -v. The CPU suite skips this module.
"""
from __future__ import annotations

from contextlib import nullcontext
import dataclasses
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
import ase.io

# Deliberately import model modules before constructing the profile. This
# catches regressions where configuration still relies on import-time ENV.
from fastmd.models.matris import functions as model_functions
from fastmd.adapters.protocol import PaddingRule
from fastmd.interfaces.calculator import MLIPCalculator
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.interfaces.multi_system_md import MultiSystemMolecularDynamics
from fastmd.interfaces import multi_system_relax
from fastmd.interfaces.multi_system_relax import MultiSystemRelaxation
from fastmd.interfaces.replica_md import ReplicaMolecularDynamics
from fastmd.execution.gpu_nhc import maxwell_boltzmann_momenta
from fastmd.execution.replica_batch import ReplicaHealthError
from fastmd.execution.whole_step_cuda_graph import FixedCapacityOverflow
from fastmd.config import CapacityConfig, ConfigurationWarning, InferenceConfig, resolve_config
from fastmd.engine import capacity_policy
from fastmd.engine import memory_admission
from fastmd.graph import atoms_to_graph_gpu
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.gpu_graph_builder import TensorGraphBuilder, op_available

from pair_potential_adapter import PairPotentialAdapter, SmoothPairPotential, SystemTapPairAdapter, UnionPairAdapter
from test_isolated_atom_cuda import make_atoms
from test_isolated_atom_handling import make_model
from test_runtime_config import assert_adapter_matches_model, deterministic_algorithms, fake_adapter
from fastmd.adapters.memory import MeasuredMemoryModel
from fastmd.adapters.negotiate import dry_run, negotiate
from fastmd.adapters.protocol import CapacityStep, CaptureContract, NegotiationError

# A calibration outside every measured K6 envelope (12 layers at 128/128/128).
DERIVED = memory_admission.MemoryModel(memory_admission.Architecture(12, 128, 128, 128, 256, 7, 7, 4, 4, 6.0, 4.5),
                                       "derived")


@unittest.skipUnless(
    torch.cuda.is_available() and op_available and _fixed_builder_available,
    "CUDA graph builders are unavailable",
)
class InferenceConfigCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    @classmethod
    def tearDownClass(cls):
        torch.set_float32_matmul_precision(cls.previous_precision)

    def capacity(self):
        return CapacityConfig(
            u_step=8, t_step=32, min_pad_u=4, min_pad_t=8,
            n_dummy=4, warmup=1, candidate_skin=1.0,
        )

    def model(self, *, reference_energy="demo", num_layers=2):
        # Fresh, identically initialized models; no mutable model/cache sharing.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(7301)
            return make_model(num_layers=num_layers, norm_type="layer", reference_energy=reference_energy).cuda()

    def config(self, execution, *, isolated_atoms="reference", force_indexed=False, topology_reuse="auto"):
        overrides = {"MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS": 0} if force_indexed else {}
        if topology_reuse != "off":
            # Every skin-reuse call also checks its topology against fresh builds.
            overrides["MATRIS_TOPOLOGY_REUSE_CHECK"] = 1
        return InferenceConfig(
            execution=execution, optimization_profile="topology",
            checkpoint="off", isolated_atoms=isolated_atoms,
            capacity=self.capacity(), report=False, expert_overrides=overrides,
            topology_reuse=topology_reuse,
        )

    def captures(self, calc):
        """CUDA graphs the calculator's last call path captured."""
        runner = calc._skin if calc.topology_reuse_summary()["last_call"] == "skin" else calc._graph_runner
        return runner.captures

    def reference(self, atoms, *, reference_energy="demo", num_layers=2):
        model = self.model(reference_energy=reference_energy, num_layers=num_layers)
        resolved = resolve_config(InferenceConfig(
            execution="eager", optimization_profile="generic", report=False,
        ))
        with resolved.scope():
            graph = atoms_to_graph_gpu(
                atoms, atom_graph_cutoff=2.0, line_graph_cutoff=1.5,
                check_isolated_atoms=False,
            )
            out = model([graph], task="ef", is_training=False, handle_isolated_atoms=True)
            energy = out["e"][0].detach().clone()
            if model.is_intensive:
                energy = energy * len(atoms)
            forces = out["f"][0].detach().clone()
        return energy.cpu(), forces.cpu()

    def assert_finite_close(self, energy, forces, expected, *, isolated_index=2):
        energy = torch.as_tensor(energy).detach().cpu().reshape(())
        forces = torch.as_tensor(forces).detach().cpu()
        self.assertTrue(bool(torch.isfinite(energy).all()))
        self.assertTrue(bool(torch.isfinite(forces).all()))
        torch.testing.assert_close(energy.to(expected[0].dtype), expected[0], rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(forces.to(expected[1].dtype), expected[1], rtol=2e-4, atol=2e-5)
        if isolated_index is not None:
            # An isolated atom retains reference energy and exactly zero force.
            torch.testing.assert_close(forces[isolated_index], torch.zeros_like(forces[isolated_index]), rtol=0, atol=0)

    def test_calculator_model_graph_profile_after_import(self):
        atoms = make_atoms("partial")
        expected = self.reference(atoms)
        previous_indexed = model_functions._INDEXED_CAT_LINEAR_FUSE
        for reuse in ("off", "skin"):
            with self.subTest(topology_reuse=reuse):
                calc = MLIPCalculator(
                    model=self.model(), device="cuda", task="ef",
                    config=self.config("model_graph", force_indexed=True, topology_reuse=reuse),
                )
                self.assertEqual(model_functions._INDEXED_CAT_LINEAR_FUSE, previous_indexed)
                self.assertEqual(calc.config_summary()["effective"]["execution"], "model_graph")
                for _ in range(2):
                    # Call directly to exercise replay, not ASE's property-result cache.
                    calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
                    self.assert_finite_close(calc.results["energy"], calc.results["forces"], expected)
                # The captured model graph ran, on the requested topology path.
                self.assertEqual(calc.topology_reuse_summary()["last_call"], reuse)
                self.assertGreaterEqual(self.captures(calc), 1)
                self.assertEqual(model_functions._INDEXED_CAT_LINEAR_FUSE, previous_indexed)
                summary = calc.config_summary()
                # Tiny equal-width, frozen layer-norm atom attention is eligible when
                # MIN_ROWS=0. Count materialized caches, not an assertion about every op.
                self.assertGreater(summary["effective"]["merged_projections"]["materialized_module_count"], 0)

    def test_calculator_no_reference_uses_graph_and_rejects_isolation(self):
        atoms = make_atoms("connected")
        expected = self.reference(atoms, reference_energy=None)
        for reuse in ("off", "skin"):
            # A requested reference policy becomes strict without a reference
            # table, but must not silently disable model_graph for connected atoms.
            calc = MLIPCalculator(
                model=self.model(reference_energy=None), device="cuda", task="ef",
                config=self.config("model_graph", topology_reuse=reuse),
            )
            for _ in range(2):
                calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
                self.assert_finite_close(calc.results["energy"], calc.results["forces"], expected, isolated_index=None)
            self.assertEqual(calc.topology_reuse_summary()["last_call"], reuse)
            self.assertGreaterEqual(self.captures(calc), 1)
            for state in ("partial", "isolated"):
                with self.subTest(topology_reuse=reuse, state=state), self.assertRaisesRegex(ValueError, "isolated atom"):
                    calc.calculate(make_atoms(state), properties=["energy", "forces"], system_changes=all_changes)

    def test_calculator_topology_reuse_matches_rebuild(self):
        """Skin reuse equals the per-call rebuild over MD-like steps, jumps,
        lattice wraps, a structure change A -> B -> A and a cell change."""
        rng = np.random.default_rng(3)
        base = Atoms("H6", positions=rng.uniform(0, 4.0, (6, 3)), cell=[4.0] * 3, pbc=True)
        other = Atoms("H5", positions=rng.uniform(0, 4.5, (5, 3)), cell=[4.5] * 3, pbc=True)
        # (atoms, a new system?) A new system's first eager call rebuilds.
        frames = [(base.copy(), True)]
        for step in range(3):  # small moves: the same epoch
            frames.append((frames[-1][0].copy(), False))
            frames[-1][0].positions += rng.normal(0, 0.02, (6, 3))
        wrapped = frames[-1][0].copy()
        wrapped.positions[0] += wrapped.cell[0]  # a lattice jump is no motion
        wrapped.positions[3] -= 2 * wrapped.cell[2]
        frames.append((wrapped, False))
        jumped = wrapped.copy()
        jumped.positions[1] += 0.6  # beyond 0.45 of the skin: refreshed
        swapped = jumped.copy()
        swapped.numbers[[1, 4]] = 6  # same atoms, new elements: set in place
        frames += [(jumped, False), (swapped, False), (other.copy(), True), (other.copy(), False),
                   (base.copy(), True), (base.copy(), False)]
        strained = base.copy()
        strained.set_cell(strained.cell * 1.03, scale_atoms=True)
        for execution in ("model_graph", "eager"):
            # A cell change rebuilds per call until the cell was stable for a few calls.
            hold = 4 if execution == "model_graph" else 2
            skin = MLIPCalculator(model=self.model(), device="cuda", task="efs",
                                    config=self.config(execution, topology_reuse="skin"))
            rebuild = MLIPCalculator(model=self.model(), device="cuda", task="efs",
                                       config=self.config(execution, topology_reuse="off"))
            sequence = [(atoms, "off" if new and execution == "eager" else "skin") for atoms, new in frames]
            sequence += [(strained, "off")] * hold + [(strained, "skin")] * 2
            for index, (atoms, path) in enumerate(sequence):
                with self.subTest(execution=execution, frame=index):
                    skin.calculate(atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
                    rebuild.calculate(atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
                    self.assertEqual(skin.topology_reuse_summary()["last_call"], path)
                    for name in ("energy", "forces", "stress"):
                        torch.testing.assert_close(torch.as_tensor(skin.results[name]),
                                                   torch.as_tensor(rebuild.results[name]),
                                                   rtol=2e-4, atol=2e-5)
            stats = skin.topology_reuse_summary()["runner"]
            self.assertGreaterEqual(stats["checks"], 2)
            self.assertEqual(stats["redo_skin"], 0)

    def test_calculator_skin_precapture_replays(self):
        # Precapture drives a probe trajectory on the skin path and captures
        # the neighbor buckets of its final demand; later calls only replay.
        rng = np.random.default_rng(4)
        atoms = Atoms("H8", positions=rng.uniform(0, 4.0, (8, 3)), cell=[4.0] * 3, pbc=True)
        calc = MLIPCalculator(model=self.model(), device="cuda", task="ef",
                                config=self.config("model_graph", topology_reuse="skin"))
        calc.precapture(atoms, n_steps=3)
        runner = calc._skin
        self.assertIsNotNone(runner)
        self.assertGreaterEqual(len(runner.cache), 2)
        captures = runner.captures
        calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
        self.assertEqual(calc.topology_reuse_summary()["last_call"], "skin")
        self.assertEqual(runner.captures, captures)

    def edges_only_config(self, execution, topology_reuse, isolated_atoms="reference"):
        """``config`` with buckets of 64 pairs: a few captures for the ~1,900 pairs of edges_only_frames."""
        config = self.config(execution, isolated_atoms=isolated_atoms, topology_reuse=topology_reuse)
        return dataclasses.replace(config, capacity=dataclasses.replace(config.capacity, u_step=64, min_pad_u=16))

    @staticmethod
    def edges_only_frames():
        """A perturbed rocksalt crystal in a sheared (triclinic) cell, 144 atoms, then calls with
        small moves, two lattice jumps (no motion), a 0.6 A jump (a candidate refresh at skin 1.0)
        and new elements of the same atoms."""
        rng = np.random.default_rng(13)
        crystal = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 2))
        crystal.set_cell(crystal.cell.array + [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.5, 0.3, 0.0]],
                         scale_atoms=True)
        crystal.positions += rng.normal(0, 0.05, crystal.positions.shape)

        def moved(atoms):
            atoms = atoms.copy()
            atoms.positions += rng.normal(0, 0.03, atoms.positions.shape)
            return atoms

        frames = [crystal]
        for _ in range(3):
            frames.append(moved(frames[-1]))
        wrapped = frames[-1].copy()
        wrapped.positions[0] += wrapped.cell[0]
        wrapped.positions[3] -= 2 * wrapped.cell[2]
        jumped = wrapped.copy()
        jumped.positions[1] += 0.6
        swapped = jumped.copy()
        swapped.numbers[[1, 4]] = 19
        strained = moved(swapped)
        strained.set_cell(strained.cell * 1.03, scale_atoms=True)  # a new cell: the calculator rebuilds
        return frames + [wrapped, jumped, swapped, moved(swapped), strained]

    @staticmethod
    def edges_only_direct(adapter, atoms):
        """Energy, forces and stress (eV/A^3) of ``adapter`` evaluated directly on a fresh ragged
        edges-only build of ``atoms`` (no calculator)."""
        builder = TensorGraphBuilder(
            cell=torch.tensor(atoms.cell.array, dtype=torch.float32), atomic_numbers=torch.tensor(atoms.numbers),
            pbc=torch.tensor(atoms.pbc), atom_graph_cutoff=adapter.topology_spec().cutoff, line_graph_cutoff=None,
            device="cuda", check_isolated_atoms=False)
        batch = builder.build(torch.tensor(atoms.positions, dtype=torch.float32))
        out = adapter.evaluate([adapter.prepare(batch)], task="efs", is_training=False)
        return float(out["e"][0]), out["f"][0].cpu().numpy(), out["s"][0].cpu().numpy() * units.GPa

    def test_calculator_edges_only_adapter_skin_matches_rebuild(self):
        """An edges-only adapter (the test pair potential, every padding rule): skin reuse, eager and
        model_graph, equals the per-call rebuild and the adapter evaluated directly on a fresh ragged
        build, over small moves, lattice jumps, a candidate refresh, new elements and two calls that
        overflow their bucket. Energy and forces bit for bit with deterministic scatters (an edge's
        arithmetic does not depend on the row count and padded rows add exact zeros); the stress at
        FP32 rounding (the strain gradient sums over all edge rows, which padding changes in number)."""
        frames = self.edges_only_frames()
        with deterministic_algorithms():
            for kind in ("far_self_loops", "exact_rc_loops", "sink_atoms"):
                for execution in ("eager", "model_graph"):
                    calcs = {reuse: MLIPCalculator(model=PairPotentialAdapter(padding=PaddingRule(kind)),
                                                     device="cuda", task="efs",
                                                     config=self.edges_only_config(execution, reuse))
                             for reuse in ("skin", "off")}
                    for index, atoms in enumerate(frames):
                        runner = calcs["skin"]._skin
                        if index in (3, 6) and runner is not None:
                            # An under-estimated demand selects a bucket that this call overflows: the
                            # overflowed evaluation is discarded and redone in a bucket that fits.
                            runner.close()
                            runner._demand = (100, 0)
                        with self.subTest(padding=kind, execution=execution, frame=index):
                            for calc in calcs.values():
                                calc.calculate(atoms, properties=["energy", "forces", "stress"],
                                               system_changes=all_changes)
                            # A new system's first eager call rebuilds, and so does a call after a cell change.
                            path = ("off" if execution == "eager" and index == 0 or index == len(frames) - 1
                                    else "skin")
                            self.assertEqual(calcs["skin"].topology_reuse_summary()["last_call"], path)
                            energy, forces, stress = self.edges_only_direct(calcs["off"].adapter, atoms)
                            self.assertGreater(np.abs(forces).max(), 0.0)
                            for reuse, calc in calcs.items():
                                self.assertEqual(calc.results["energy"], energy, reuse)
                                self.assertEqual(calc.results["ref_energy"], 0.0, reuse)
                                np.testing.assert_array_equal(calc.results["forces"], forces, err_msg=reuse)
                                np.testing.assert_allclose(calc.results["stress"], stress, rtol=0,
                                                           atol=1e-5 * np.abs(stress).max(), err_msg=reuse)
                        if calcs["skin"]._skin is not None:  # the cell change of the last frame drops it
                            stats = calcs["skin"].topology_reuse_summary()["runner"]
                            t_capacities = {entry["builder"].t_capacity
                                            for entry in calcs["skin"]._skin.cache.values()}
                    # The jump refreshed the candidates; every skin call was checked against fresh builds.
                    self.assertGreaterEqual(stats["refreshes"], 2)
                    self.assertEqual(stats["redo_skin"], 0)
                    self.assertGreaterEqual(stats["redo_capacity"], 2)
                    self.assertGreaterEqual(stats["checks"], len(frames) - 2)
                    self.assertEqual(t_capacities, {0})

    def test_calculator_edges_only_adapter_skin_precapture_replays(self):
        # Precapture on the skin path of an edges-only model (neighbor buckets allowed): neighbors
        # in the pair dimension only, builders without triplets; later calls only replay.
        atoms = self.edges_only_frames()[0]
        calc = MLIPCalculator(model=PairPotentialAdapter(), device="cuda", task="ef",
                                config=self.edges_only_config("model_graph", "skin"))
        with deterministic_algorithms():
            with mock.patch("fastmd.execution.cuda_graph.precapture_fits", return_value=(True, "test")):
                calc.precapture(atoms, n_steps=3)
            runner = calc._skin
            captures = runner.captures
            calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
            energy, forces, _ = self.edges_only_direct(calc.adapter, atoms)
        self.assertGreaterEqual(len(runner.cache), 2)
        self.assertEqual(len({key[1] for key in runner.cache}), 1)
        self.assertEqual({entry["builder"].t_capacity for entry in runner.cache.values()}, {0})
        self.assertEqual(calc.topology_reuse_summary()["last_call"], "skin")
        self.assertEqual(runner.captures, captures)
        self.assertEqual(calc.results["energy"], energy)
        np.testing.assert_array_equal(calc.results["forces"], forces)

    def test_calculator_edges_only_adapter_isolated_atoms_and_limits(self):
        """An edges-only adapter without reference energies: an isolated atom adds nothing (zero
        force, the other atoms' energy) on every path and is rejected under isolated_atoms="error";
        execution auto captures it after a memory measurement (K6 has no calibration for it); a
        non-periodic cell and a CPU device are rejected."""
        rng = np.random.default_rng(17)
        cluster = Atoms("Na4Cl4", positions=rng.uniform(4.0, 7.0, (8, 3)), cell=[20.0] * 3, pbc=True)
        lonely = cluster + Atoms("K", positions=[[15.0, 15.0, 15.0]])
        for execution in ("eager", "model_graph"):
            for reuse in ("skin", "off"):
                with self.subTest(execution=execution, topology_reuse=reuse):
                    calc = MLIPCalculator(model=PairPotentialAdapter(), device="cuda", task="efs",
                                            config=self.edges_only_config(execution, reuse))
                    with deterministic_algorithms():
                        for _ in range(2):  # an eager calculator's second call takes the skin path
                            calc.calculate(lonely, properties=["energy", "forces"], system_changes=all_changes)
                            energy, forces, _ = self.edges_only_direct(calc.adapter, lonely)
                            self.assertEqual(calc.results["energy"], energy)
                            np.testing.assert_array_equal(calc.results["forces"], forces)
                    self.assertEqual(calc.topology_reuse_summary()["last_call"], reuse)
                    np.testing.assert_array_equal(calc.results["forces"][-1], np.zeros(3))
                    alone, _, _ = self.edges_only_direct(calc.adapter, cluster)
                    self.assertAlmostEqual(calc.results["energy"], alone, delta=1e-6 * abs(alone))
                    strict = MLIPCalculator(model=PairPotentialAdapter(), device="cuda", task="ef",
                                              config=self.edges_only_config(execution, reuse, isolated_atoms="error"))
                    for _ in range(2):
                        with self.assertRaisesRegex(ValueError, "isolated atom"):
                            strict.calculate(lonely, properties=["energy", "forces"], system_changes=all_changes)
        with self.subTest("execution auto"):
            calc = MLIPCalculator(model=PairPotentialAdapter(), device="cuda", task="ef")
            for _ in range(3):
                calc.calculate(cluster, properties=["energy", "forces"], system_changes=all_changes)
            auto = calc.config_summary()["effective"]["auto_execution"]
            # K6 has no calibration for this model: a memory model measured on its own evaluation admits
            # the capture (test_calculator_auto_plans_an_edges_only_adapter).
            self.assertEqual(auto["mode"], "model_graph")
            self.assertIn("measured capture bound", auto["reason"])
            self.assertEqual(calc.topology_reuse_summary()["last_call"], "skin")
            with self.assertRaisesRegex(ValueError, "periodic in all three directions"):
                calc.calculate(Atoms("NaCl", positions=[[0, 0, 0], [2.5, 0, 0]], pbc=False),
                               properties=["energy"], system_changes=all_changes)
        with self.subTest("CPU device"), self.assertRaisesRegex(ValueError, "CUDA device"):
            MLIPCalculator(model=PairPotentialAdapter(), device="cpu", config=InferenceConfig(report=False))
        # New atomic numbers of the same atoms go through the adapter's own input check (prepare) on every path,
        # also where they enter a captured or skin bucket in place; a rejected call leaves the calculator usable.
        class Checked(PairPotentialAdapter):
            def prepare(self, batch):
                if bool((batch.species[:batch.n_real] > 20).any()):
                    raise ValueError("this pair potential knows atomic numbers 1-20")
                return batch
        crystal = self.edges_only_frames()[0]
        foreign = crystal.copy()
        foreign.numbers[1] = 21
        for execution in ("eager", "model_graph"):
            for reuse in ("skin", "off"):
                with self.subTest("elements", execution=execution, topology_reuse=reuse):
                    calc = MLIPCalculator(model=Checked(), device="cuda", task="ef",
                                            config=self.edges_only_config(execution, reuse))
                    for _ in range(2):  # an eager calculator's second call takes the skin path
                        calc.calculate(crystal, properties=["energy", "forces"], system_changes=all_changes)
                    self.assertEqual(calc.topology_reuse_summary()["last_call"], reuse)
                    energy = calc.results["energy"]
                    with self.assertRaisesRegex(ValueError, "atomic numbers 1-20"):
                        calc.calculate(foreign, properties=["energy", "forces"], system_changes=all_changes)
                    calc.calculate(crystal, properties=["energy", "forces"], system_changes=all_changes)
                    self.assertEqual(calc.results["energy"], energy)

    def test_calculator_execution_auto(self):
        """config=None: a new system's call runs eager, its repeat switches to
        model_graph with skin reuse when K6 (the model's calibration) says it fits,
        and stays eager when it does not or the architecture is outside every
        measured calibration; a loop over new systems never captures; a CUDA OOM
        on the graph path continues eager."""
        rng = np.random.default_rng(5)
        systems = [Atoms(f"H{n}", positions=rng.uniform(0, 4.0, (n, 3)), cell=[4.0] * 3, pbc=True) for n in (6, 7)]
        huge = lambda *args, **kwargs: 1 << 50
        for calibrated, predict, mode, rule in ((True, None, "model_graph", "K6: model_graph"),
                                                (False, None, "eager", "outside every measured K6 calibration"),
                                                (True, huge, "eager", "K6: model_graph at recompute deep")):
            with self.subTest(calibrated=calibrated, fits=mode), \
                    mock.patch.object(memory_admission, "memory_model", return_value=DERIVED) if not calibrated \
                    else nullcontext(), \
                    mock.patch.object(memory_admission, "predict", side_effect=predict or memory_admission.predict):
                calc = MLIPCalculator(model=self.model(), device="cuda", task="ef")
                summary = calc.config_summary()
                self.assertEqual(summary["requested"]["execution"], "auto")
                fresh = calc.plan_record().decision("topology_reuse")  # before the first call
                self.assertEqual((fresh.chosen, fresh.reason), ("rebuild", "no call yet"))
                self.assertEqual(summary["configuration"]["optimization_profile"], "topology")
                path = (mode, "skin" if mode == "model_graph" else "off")
                for atoms in systems:
                    expected = self.reference(atoms)
                    paths = []
                    for _ in range(3):
                        calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
                        self.assert_finite_close(calc.results["energy"], calc.results["forces"], expected,
                                                 isolated_index=None)
                        paths.append((calc.config_summary()["effective"]["execution"],
                                      calc.topology_reuse_summary()["last_call"]))
                    self.assertEqual(paths, [("eager", "off"), path, path])
                decisions = calc.config_summary()["effective"]["auto_execution"]["decisions"]
                self.assertEqual([d["mode"] for d in decisions],
                                 ["model_graph", "eager", "model_graph"] if mode == "model_graph" else ["eager", "eager"])
                self.assertIn(rule, decisions[0]["reason"])
                # The plan record reads the decision (MatRIS is recorded, not negotiated), and the dry run
                # (python -m fastmd.explain) makes the same ones.
                record = calc.plan_record()
                execution = record.decision("execution")
                self.assertEqual((execution.chosen, execution.reason), (decisions[-1]["mode"], decisions[-1]["reason"]))
                dry = dry_run(calc.adapter, None, entry="calculator", device="cuda", atoms=systems[-1])
                for decision in ("execution", "topology_reuse", "padding", "recompute"):
                    self.assertEqual(dry.decision(decision).chosen, record.decision(decision).chosen, decision)
        with self.subTest("rows above the model's graph-gain limit"), \
                mock.patch.dict(capacity_policy.GRAPH_GAIN_MAX_ROWS, {"rdx_1m": 0}):
            # The limit (128-wide calibrations) only drops neighbor buckets: auto keeps the graph.
            calc = MLIPCalculator(model=self.model(), device="cuda", task="ef")
            for _ in range(2):
                calc.calculate(systems[0], properties=["energy", "forces"], system_changes=all_changes)
            self.assertEqual(calc.config_summary()["effective"]["auto_execution"]["mode"], "model_graph")
        with self.subTest("new systems only"):
            calc = MLIPCalculator(model=self.model(), device="cuda", task="ef")
            for n in range(3, 8):
                atoms = Atoms(f"H{n}", positions=rng.uniform(0, 4.0, (n, 3)), cell=[4.0] * 3, pbc=True)
                calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
            self.assertIsNone(calc._graph_runner)
            self.assertEqual(calc.config_summary()["effective"]["auto_execution"]["decisions"], [])
        with self.subTest("out of memory"):
            atoms = systems[0]
            expected = self.reference(atoms)
            calc = MLIPCalculator(model=self.model(), device="cuda", task="ef")
            calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)

            def out_of_memory(*args, **kwargs):
                raise torch.OutOfMemoryError("CUDA out of memory (test)")
            calc._skin_predict = out_of_memory
            with self.assertWarnsRegex(RuntimeWarning, "model_graph failed"):
                calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
            self.assert_finite_close(calc.results["energy"], calc.results["forces"], expected, isolated_index=None)
            calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
            self.assertIsNone(calc._graph_runner)
            decisions = calc.config_summary()["effective"]["auto_execution"]["decisions"]
            self.assertEqual([d["mode"] for d in decisions], ["model_graph", "eager"])

    def test_gpu_md_execution_auto(self):
        """config=None: whole_step for nve, Berendsen nvt and nvt_nhc, eager
        when K6 says model_graph does not fit or the architecture is
        outside the K6 calibration; a CUDA OOM in a whole-step capture
        (construction, capacity growth, a later capture in a window) continues
        eager from the failed window's start: the trajectory equals eager."""
        atoms = make_atoms("partial")
        expected = self.reference(atoms)
        huge = lambda *args, **kwargs: 1 << 50
        for ensemble, calibrated, predict, mode, rule in (
                ("nve", True, None, "whole_step", "K6"), ("nvt_nhc", True, None, "whole_step", "K6"),
                ("nvt", True, None, "whole_step", "K6"), ("nve", True, huge, "eager", "K6"),
                ("nve", False, None, "eager", "outside every measured K6 calibration")):
            with self.subTest(ensemble=ensemble, calibrated=calibrated, mode=mode), \
                    mock.patch.object(memory_admission, "memory_model", return_value=DERIVED) if not calibrated \
                    else nullcontext(), \
                    mock.patch.object(memory_admission, "predict", side_effect=predict or memory_admission.predict), \
                    GPUResidentMolecularDynamics(atoms, model=self.model(), task="ef", device="cuda",
                                                 ensemble=ensemble, timestep=0.0, tdamp=10.0) as md:
                summary = md.config_summary()["effective"]
                self.assertEqual(summary["execution"], mode)
                self.assertIn(rule, summary["auto_execution"]["reason"])
                # The plan record reads the decision; the dry run (python -m fastmd.explain) makes the same.
                execution = md.plan_record().decision("execution")
                self.assertEqual((execution.chosen, execution.reason), (mode, summary["auto_execution"]["reason"]))
                dry = dry_run(md.adapter, None, entry="gpu_md", device="cuda", atoms=atoms)
                self.assertEqual(dry.decision("execution").chosen, mode)
                self.assertIn(rule, dry.decision("execution").reason)
                md.run(2)
                # The recompute level the forwards use, previewed by the dry run with the same rule.
                self.assertEqual(dry.decision("recompute").chosen, md.plan_record().decision("recompute").chosen)
                self.assert_finite_close(md.state.potential_energy, md.state.forces, expected)

        with self.subTest("rows above the model's graph-gain limit"):
            # The limit (128-wide calibrations) drops a model-graph runner's neighbor buckets.
            for limit in (None, 0):
                with mock.patch.dict(capacity_policy.GRAPH_GAIN_MAX_ROWS, {} if limit is None else {"rdx_1m": limit}), \
                        GPUResidentMolecularDynamics(make_atoms("partial"), model=self.model(), task="ef",
                                                     device="cuda", ensemble="nvt", timestep=0.0, tdamp=10.0,
                                                     config=InferenceConfig(execution="model_graph")) as md:
                    self.assertEqual(md.config_summary()["effective"]["execution"], "model_graph")
                    self.assertEqual(len(md.runner.stats()["cached_buckets"]) > 1, limit is None)
                    if limit is not None:
                        self.assertIn("GPU-bound, the envelope only", md.runner.precapture_note)

        moving = make_atoms("connected")
        oom = torch.OutOfMemoryError("CUDA out of memory (test)")

        def trajectory(ensemble, config, fail=None):
            with mock.patch.object(GPUResidentMolecularDynamics, "_make_whole_step_runner",
                                      side_effect=oom, autospec=True) if fail == "construction" else nullcontext(), \
                    self.assertWarnsRegex(RuntimeWarning, "whole_step ran out of CUDA memory") if fail \
                    else nullcontext():
                md = GPUResidentMolecularDynamics(moving, model=self.model(), task="ef", device="cuda",
                                                  ensemble=ensemble, timestep=0.5, tdamp=10.0, config=config)
                for step in range(4):
                    if step != 2 or fail in (None, "construction"):
                        md.step()
                        continue
                    runner = md.whole_step_runner
                    replay = runner.replay
                    if fail == "window":
                        def failing_replay(*args, **kwargs):
                            replay(*args, **kwargs)  # the window ran, then a capture in it failed
                            raise oom
                    else:  # "growth": the window overflows (rolled back), the larger capture fails
                        def failing_replay(*args, **kwargs):
                            replay(*args, **kwargs)
                            runner._restore_backup()
                            raise FixedCapacityOverflow(4 * md._whole_u_capacity, 4 * md._whole_t_capacity)
                        md._make_whole_step_runner = mock.Mock(side_effect=oom)
                    runner.replay = failing_replay
                    md.step()
            result = md.nsteps, md.state.positions.clone(), md.config_summary()["effective"]["execution"]
            md.close()
            return result

        for ensemble in ("nve", "nvt_nhc"):
            eager = trajectory(ensemble, InferenceConfig(execution="eager", report=False))
            for fail in ("construction", "window", "growth"):
                with self.subTest("whole_step out of memory", ensemble=ensemble, fail=fail):
                    fallback = trajectory(ensemble, None, fail)
                    self.assertEqual((fallback[0], fallback[2]), (4, "eager"))
                    torch.testing.assert_close(fallback[1], eager[1], rtol=2e-4, atol=2e-5)

    def test_whole_step_trial_against_model_graph(self):
        """select_whole_step_cuda_graph(baseline="model_graph") (a whole-step trial on request) keeps
        model_graph unless whole_step is min_speedup faster; MD continues on the kept path."""
        atoms = make_atoms("connected")
        expected = self.reference(atoms)
        eager_config = InferenceConfig(execution="eager", capacity=self.capacity(), report=False)
        with GPUResidentMolecularDynamics(atoms, model=self.model(), task="ef", device="cuda", ensemble="nve",
                                          timestep=0.5, config=eager_config) as md:
            md.run(5)
            reference = md.state.positions.clone()
        for min_speedup, paths in ((1e9, ("model_graph",)), (1.01, ("model_graph", "whole_step_cuda_graph"))):
            with self.subTest(min_speedup=min_speedup), GPUResidentMolecularDynamics(
                    atoms, model=self.model(), task="ef", device="cuda", ensemble="nve", timestep=0.5,
                    config=eager_config) as md:
                result = md.select_whole_step_cuda_graph(warmup_steps=3, min_speedup=min_speedup,
                                                         baseline="model_graph")
                self.assertEqual(result["baseline"], "model_graph")
                self.assertIn(result["selected_path"], paths)
                effective = md.config_summary()["effective"]["execution"]
                self.assertEqual(effective, "model_graph" if result["selected_path"] == "model_graph" else "whole_step")
                md.run(2)  # continues from the trial's 3 steps on the kept path
                self.assertEqual(md.nsteps, 5)
                torch.testing.assert_close(md.state.positions, reference, rtol=2e-4, atol=2e-5)
        del expected

    def test_measured_memory_model_predicts_a_held_out_capacity(self):
        """MeasuredMemoryModel on a test-only module whose memory grows linearly with its edges: fitted
        on three capacities (an eager step and a capture), it predicts a held-out capacity within the
        residual bound; a model other than MatRIS then captures under execution auto, and stays eager
        on an invalid fit."""
        torch.manual_seed(0)
        module = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).cuda()
        stream = torch.cuda.Stream()  # one capture stream, as a runner has

        def forces(x):
            return torch.autograd.grad(module(x).sum(), x)[0]

        def eager(capacity):
            forces(torch.randn(capacity, 16, device="cuda", requires_grad=True))
            return capacity, 0

        def capture(capacity):
            x = torch.randn(capacity, 16, device="cuda", requires_grad=True)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forces(x)  # warmup
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                forces(x)
            del graph  # every point starts from the same state
            return capacity, 0

        runs = {"eager": eager, "capture": capture}
        for run in runs.values():
            run(1 << 10)  # warm up before the measurement
        model = MeasuredMemoryModel.measure(runs, (1 << 16, 1 << 17, 1 << 18), "cuda", triplets=False)
        self.assertTrue(model.valid, model.reason)
        held_out = MeasuredMemoryModel.measure(runs, (3 << 16,), "cuda", triplets=False)
        for point in held_out.points:
            with self.subTest(kind=point.kind):
                fit = model.fits[point.kind]
                self.assertLessEqual(abs(model.predict(point.kind, point.edges) - point.peak_bytes), fit.residual_bound)
                self.assertLessEqual(point.peak_bytes, model.bound(point.kind, point.edges))
        contract = CaptureContract(sync_free=True, exact_cutoff=True)
        for memory, mode in ((model, "whole_step"), (MeasuredMemoryModel(model.points[:2] + model.points[3:5],
                                                                          triplets=False), "eager")):
            record = negotiate(fake_adapter(contract=contract), None, device="cuda", memory=memory,
                               capacity={"edges": 3 << 16})
            self.assertEqual(record.decision("execution").chosen, mode, record.decision("execution").reason)

    def edges_only_md(self, execution, adapter=None, steps=20):
        """GPU-resident MD (NVE, 1500 K momenta) of the first edges-only frame with the test pair
        potential under ``execution``, ``steps`` steps: (plan record, the negotiated plan as kept, config
        summary, the runtime's graph path read from its builders, final (positions, forces, energy)).
        Deterministic scatters."""
        from fastmd.execution.gpu_nhc import maxwell_boltzmann_momenta
        atoms = self.edges_only_frames()[0]
        atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), 1500.0, seed=5))
        config = InferenceConfig(execution=execution, report=False,
                                 capacity=CapacityConfig(u_step=64, min_pad_u=16, warmup=1, candidate_skin=1.0))
        with deterministic_algorithms(), GPUResidentMolecularDynamics(
                atoms, model=adapter or PairPotentialAdapter(), ensemble="nve", timestep=1.0, config=config) as md:
            md.run(steps)
            self.assertEqual(md.nsteps, steps)
            if md.whole_step_runner is not None:
                builder = md.whole_step_runner.builder
                path = ("fixed_capacity_builder", "exact_skin_refilter", builder.padding.kind, builder.edges_only)
                state = md.whole_step_runner.state
            else:
                padding = md.adapter.padding_rule().kind if md.runner is not None else "none"
                path = ("tensor_graph_builder", "rebuild", padding, md.graph_builder.line_graph_cutoff is None)
                state = md.state
            final = (state.positions.clone(), state.forces.clone(), state.potential_energy.reshape(()).clone())
            return md.plan_record(), md._engine_plan[1], md.config_summary(), path, final

    def assert_plan_is_path(self, plan, summary, path):
        """The plan's graph decisions name the runtime's path and its execution is the effective one."""
        self.assertEqual(tuple(plan.decision(name).chosen for name in ("neighbor_list", "topology_reuse", "padding")),
                         path[:3])
        self.assertEqual(plan.decision("execution").chosen, summary["effective"]["execution"])

    def test_gpu_md_auto_plans_an_edges_only_adapter(self):
        """execution auto for an edges-only model other than MatRIS (the test pair potential), under a TF32
        process default: the plan comes from its capabilities (edges-only builders, its padding rule, exact
        skin, a memory model measured on its own evaluation, the whole-step trial against model_graph); the
        negotiated plan names the path the runtime runs, the forwards run IEEE FP32, and the trajectory
        (after the trial) equals eager's bit for bit. Each trial outcome (mocked) selects its path."""
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")  # TF32 by default: the plan pins its forwards to IEEE
        try:
            _, _, _, _, reference = self.edges_only_md("eager")
            record, plan, summary, path, final = self.edges_only_md("auto")
        finally:
            torch.set_float32_matmul_precision(previous)
        effective = summary["effective"]
        execution = plan.decision("execution")
        self.assertIn(execution.chosen, ("whole_step", "model_graph"), execution.reason)
        self.assertEqual(effective["auto_execution"]["reason"], execution.reason)
        self.assertIn("measured capture bound", execution.reason)
        self.assertGreater(execution.evidence["whole_step_speedup_over_model_graph"], 0.0)
        for kept in (plan, record):
            self.assert_plan_is_path(kept, summary, path)
        self.assertEqual(path[2:], ("far_self_loops", True))  # the adapter's rule, edges only
        self.assertIn("edges only", plan.decision("neighbor_list").reason)
        self.assertEqual(plan.decision("topology_reuse").evidence["candidate_skin"], 1.0)
        self.assertTrue(plan.memory["valid"], plan.memory["reason"])
        self.assertEqual(plan.memory["calibration"], "measured")
        self.assertEqual(len({point["edges"] for point in plan.memory["points"]}), 3)
        self.assertEqual((plan.decision("precision").chosen, plan.matmul_precision, plan.accuracy_class),
                         ("ieee", "highest", "fp32_noise"))
        self.assertEqual(effective["matmul_precision"]["fp32_gemm"], "ieee")
        self.assertFalse(any("negotiated" in decision.reason for decision in record.decisions))
        for got, expected in zip(final, reference):
            self.assertTrue(torch.equal(got, expected))
        _, _, _, _, reference = self.edges_only_md("eager", steps=4)
        for speedup, chosen in ((2.0, "whole_step"), (1.0, "model_graph"), (float("nan"), "model_graph")):
            with self.subTest(trial=speedup), mock.patch.object(GPUResidentMolecularDynamics, "_whole_step_trial",
                                                                return_value=speedup):
                record, plan, summary, path, final = self.edges_only_md("auto", steps=4)
                self.assertEqual(plan.decision("execution").chosen, chosen)
                self.assert_plan_is_path(plan, summary, path)
                self.assertFalse(any("negotiated" in decision.reason for decision in record.decisions))
                for got, expected in zip(final, reference):
                    self.assertTrue(torch.equal(got, expected))

    def test_system_energy_tap_matches_the_model_owned_derivative(self):
        """A whole-system energy tap (the test pair potential's energy as one number, differentiated by the
        engine) runs like the model-owned derivative of the same potential: eager, model_graph, whole_step
        and auto (whose memory measurement evaluates through the engine's derivative), 20 NVE steps with
        deterministic scatters, positions and forces bit for bit (the same per-edge arithmetic and
        backward), the energy at FP32 rounding; on one build the engine's stress (eV/A^3) equals the
        model's own (GPa, converted) at FP32 rounding."""
        from pair_potential_adapter import SmoothPairPotential, SystemTapPairAdapter
        from fastmd.engine.energy_derivative import EnergyDerivative, engine_derivative
        for execution in ("eager", "model_graph", "whole_step", "auto"):
            with self.subTest(execution=execution):
                *_, reference = self.edges_only_md("eager" if execution == "auto" else execution)
                _, plan, *_, final = self.edges_only_md(execution, adapter=SystemTapPairAdapter())
                if execution == "auto":
                    self.assertTrue(plan.memory["valid"], plan.memory["reason"])
                    self.assertIn(plan.decision("execution").chosen, ("whole_step", "model_graph"))
                self.assertTrue(torch.equal(final[0], reference[0]))
                self.assertTrue(torch.equal(final[1], reference[1]))
                torch.testing.assert_close(final[2].double(), reference[2].double(), rtol=1e-6, atol=0)
        # The calculator returns the engine's FP64 energy on every path: the skin path's readback keeps FP64.
        frames = self.edges_only_frames()[:4]
        for execution in ("eager", "model_graph"):
            with self.subTest("calculator", execution=execution), deterministic_algorithms():
                calcs = {reuse: MLIPCalculator(model=SystemTapPairAdapter(), device="cuda", task="ef",
                                                 config=self.edges_only_config(execution, reuse))
                         for reuse in ("skin", "off")}
                for atoms in frames:
                    for calc in calcs.values():
                        calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
                    self.assertEqual(calcs["skin"].topology_reuse_summary()["last_call"],
                                     "off" if execution == "eager" and atoms is frames[0] else "skin")
                    energy = calcs["off"].results["energy"]
                    self.assertAlmostEqual(calcs["skin"].results["energy"], energy, delta=1e-12 * abs(energy))
                    for calc in calcs.values():
                        self.assertNotEqual(calc.results["energy"], float(np.float32(calc.results["energy"])))
        adapter = SystemTapPairAdapter(SmoothPairPotential().cuda())
        derivative = engine_derivative(adapter)
        self.assertIsInstance(derivative, EnergyDerivative)
        atoms = self.edges_only_frames()[1]
        builder = TensorGraphBuilder(
            cell=torch.tensor(atoms.cell.array, dtype=torch.float32), atomic_numbers=torch.tensor(atoms.numbers),
            pbc=torch.tensor(atoms.pbc), atom_graph_cutoff=adapter.topology_spec().cutoff, line_graph_cutoff=None,
            device="cuda", check_isolated_atoms=False)
        batch = builder.build(torch.tensor(atoms.positions, dtype=torch.float32))
        with deterministic_algorithms():
            out = derivative(batch, adapter.prepare(batch), task="efs")
            energy, forces, stress = self.edges_only_direct(PairPotentialAdapter(module=adapter.module), atoms)
        self.assertAlmostEqual(float(out["e"][0]), energy, delta=1e-6 * abs(energy))
        np.testing.assert_array_equal(out["f"][0].cpu().numpy(), forces)
        np.testing.assert_allclose(out["s"][0].cpu().numpy(), stress, rtol=0, atol=1e-5 * np.abs(stress).max())

    def test_entry_points_report_a_neighbor_cap_excess(self):
        """A model that keeps at most ``max_neighbors`` neighbours per atom (a kNN cap) gets every pair within
        the cutoff; GPU-resident MD reports the largest per-atom count of its builds (eager, model_graph and
        whole_step: a device maximum, read on request) and warns once when it exceeds the cap (on request, or
        when it closes); the calculator does the same on every path (eager and model_graph, rebuilt or
        reused), checked at each call."""
        import warnings
        atoms = self.edges_only_frames()[0]
        builder = TensorGraphBuilder(
            cell=torch.tensor(atoms.cell.array, dtype=torch.float32), atomic_numbers=torch.tensor(atoms.numbers),
            pbc=torch.tensor(atoms.pbc), atom_graph_cutoff=PairPotentialAdapter().topology_spec().cutoff,
            line_graph_cutoff=None, device="cuda", check_isolated_atoms=False)
        static = int(torch.bincount(builder.build(torch.tensor(atoms.positions, dtype=torch.float32))
                                    .edge_index[:, 0].long()).max())
        for cap in (None, 10_000, 4):
            for execution in ("eager", "model_graph"):
                for reuse in ("skin", "off"):
                    adapter = PairPotentialAdapter()
                    spec = dataclasses.replace(adapter.topology_spec(), max_neighbors=cap)
                    with self.subTest("calculator", cap=cap, execution=execution, topology_reuse=reuse), \
                            mock.patch.object(adapter, "topology_spec", return_value=spec), \
                            warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        calc = MLIPCalculator(model=adapter, device="cuda", task="ef",
                                                config=self.edges_only_config(execution, reuse))
                        for _ in range(3):  # an eager calculator's second call takes the skin path
                            calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
                        self.assertEqual(calc.topology_reuse_summary()["last_call"], reuse)
                        summary = calc.neighbor_cap_summary()
                        excess = [w for w in caught if "neighbours per atom" in str(w.message)]
                        self.assertEqual(calc.config_summary()["effective"].get("neighbor_cap"), summary)
                        if cap is None:
                            self.assertIsNone(summary)
                            self.assertEqual(excess, [])
                            continue
                        self.assertEqual(summary, {"max_neighbors": cap, "largest": static, "exceeded": static > cap})
                        self.assertEqual(len(excess), int(static > cap))
        demand = None
        for cap in (None, 10_000, 4):
            for execution in ("eager", "model_graph", "whole_step"):
                adapter = PairPotentialAdapter()
                spec = dataclasses.replace(adapter.topology_spec(), max_neighbors=cap)
                with self.subTest(cap=cap, execution=execution), \
                        mock.patch.object(adapter, "topology_spec", return_value=spec), \
                        warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    config = InferenceConfig(execution=execution, report=False,
                                             capacity=CapacityConfig(u_step=64, min_pad_u=16, warmup=1,
                                                                     candidate_skin=1.0))
                    with GPUResidentMolecularDynamics(atoms.copy(), model=adapter, ensemble="nve", timestep=1.0,
                                                      config=config) as md:
                        md.run(4)
                        if execution == "eager":  # elsewhere the first report comes from closing the MD
                            self.assertEqual(md.config_summary()["effective"].get("neighbor_cap"),
                                             md.neighbor_cap_summary())
                    summary = md.neighbor_cap_summary()
                    excess = [w for w in caught if "neighbours per atom" in str(w.message)]
                    if cap is None:
                        self.assertIsNone(summary)
                        self.assertEqual(excess, [])
                        continue
                    demand = summary["largest"] if demand is None else demand
                    self.assertEqual(summary["largest"], demand)  # the same structure on every path
                    self.assertGreater(demand, 4)
                    self.assertEqual(summary["exceeded"], demand > cap)
                    self.assertEqual(len(excess), int(demand > cap))

    def test_entry_points_take_a_declared_capacity_step(self):
        """A model that declares its capacity step (CapacityDeclaration) gets it where the configuration keeps
        the default step, the pair and the triplet tiers each: GPU-resident MD's whole-step capacity and the
        calculator's configuration and skin-runner buckets use the declared tiers, every capture holding the
        declared pair thermal factor times the demand; an explicit step is kept as given (factor 1)."""
        from fastmd.adapters.protocol import CapacityStep
        atoms = self.edges_only_frames()[0]
        demand = None
        for capacity, step, triplet_step in (
                (CapacityConfig(warmup=1, candidate_skin=1.0), (8, 4), (16, 8)),
                (CapacityConfig(u_step=64, min_pad_u=16, warmup=1, candidate_skin=1.0), (64, 16), (16, 8)),
                (CapacityConfig(t_step=32, min_pad_t=0, warmup=1, candidate_skin=1.0), (8, 4), (32, 0))):
            with self.subTest(capacity=capacity):
                factor = 1.25 if step == (8, 4) else 1.0  # where the declared pair tiers are in force
                adapter = PairPotentialAdapter()
                adapter.capacity_step = lambda: CapacityStep(u_step=8, min_pad_u=4, t_step=16, min_pad_t=8,
                                                             u_thermal_factor=1.25)
                config = InferenceConfig(execution="whole_step", report=False, capacity=capacity)
                with GPUResidentMolecularDynamics(atoms.copy(), model=adapter, ensemble="nve", timestep=1.0,
                                                  config=config) as md:
                    resolved = md._inference_config.capacity
                    self.assertEqual((resolved.u_step, resolved.min_pad_u), step)
                    self.assertEqual((resolved.t_step, resolved.min_pad_t), triplet_step)
                    self.assertEqual((resolved.warmup, resolved.candidate_skin), (1, 1.0))
                    builder = md.whole_step_runner.builder
                    demand = int(md.graph_builder.build(md.state.positions).edge_index.shape[0]) // 2
                    held = round(demand * factor)
                    # The first capture: whole steps holding the demand times the factor, the minimum padding and
                    # the whole-step headroom (at most 1/128 of it).
                    self.assertEqual(builder.u_capacity % step[0], 0)
                    self.assertGreaterEqual(builder.u_capacity, held + step[1])
                    self.assertLess(builder.u_capacity, held + step[1] + held // 128 + step[0])
                calc = MLIPCalculator(model=adapter, device="cuda", task="ef",
                                        config=InferenceConfig(execution="model_graph", report=False,
                                                               capacity=capacity))
                resolved = calc._inference_config.capacity
                self.assertEqual(((resolved.u_step, resolved.min_pad_u), (resolved.t_step, resolved.min_pad_t)),
                                 (step, triplet_step))
                probe = atoms.copy()
                probe.calc = calc
                for _ in range(2):  # the skin runner's bucket: the call's demand plus the minimum padding, held
                    probe.get_potential_energy()
                    calc.reset()
                runner = calc.topology_reuse_summary()["runner"]
                pairs = runner["last_edge_count"] // 2
                self.assertTrue(runner["cached_buckets"])
                for bucket in runner["cached_buckets"]:
                    self.assertGreaterEqual(bucket[0], round((pairs + step[1]) * factor) + step[1])

    def test_gpu_md_whole_step_trial_restores_the_state(self):
        """The whole-step trial (Nose-Hoover chain, 2-step windows, stress logged every window) leaves no
        graph, the initial state, step count and whole-step bookkeeping: MD after it equals MD without it
        bit for bit, thermostat included."""
        import tempfile
        from fastmd.execution.gpu_nhc import maxwell_boltzmann_momenta
        atoms = self.edges_only_frames()[0]
        atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), 1500.0, seed=7))
        config = InferenceConfig(execution="eager", report=False,
                                 capacity=CapacityConfig(u_step=64, min_pad_u=16, warmup=1, candidate_skin=1.0))
        results = []
        with tempfile.TemporaryDirectory() as directory, deterministic_algorithms():
            for trial in (False, True):
                with GPUResidentMolecularDynamics(
                        atoms.copy(), model=PairPotentialAdapter(), ensemble="nvt_nhc", temperature=1500.0,
                        timestep=1.0, steps_per_launch=2, log_stress=True, loginterval=2,
                        trajectory=f"{directory}/{trial}.traj", config=config) as md:
                    if trial:
                        history = list(md.whole_step_capacity_history)
                        self.assertTrue(md._stress_every_window)
                        self.assertGreater(md._whole_step_trial(windows=2), 0.0)
                        self.assertEqual((md.runner, md.whole_step_runner, md.nsteps), (None, None, 0))
                        self.assertEqual(md.whole_step_capacity_history, history)
                        self.assertEqual(md.config_summary()["effective"]["execution"], "eager")
                    md.run(6)
                    thermostat = md.integrator.state_dict()
                    results.append((md.state.positions.clone(), md.state.momenta.clone(), md.state.forces.clone(),
                                    md.state.potential_energy.clone(), torch.as_tensor(thermostat["eta"]),
                                    torch.as_tensor(thermostat["p_eta"])))
        for got, expected in zip(results[1], results[0]):
            self.assertTrue(torch.equal(got, expected))
        # Under auto with log_stress, model_graph's measured bound covers both of its runners' graph pools.
        with tempfile.TemporaryDirectory() as directory, GPUResidentMolecularDynamics(
                atoms.copy(), model=PairPotentialAdapter(), ensemble="nvt_nhc", temperature=1500.0, timestep=1.0,
                steps_per_launch=2, log_stress=True, loginterval=2, trajectory=f"{directory}/auto.traj",
                config=dataclasses.replace(config, execution="auto")) as md:
            memory = md._engine_plan[1].decision("execution").evidence["memory"]
            self.assertEqual(memory["model_graph"]["pools"], 2)
            self.assertAlmostEqual(memory["model_graph"]["capture_bound_gib"], 2 * memory["capture_bound_gib"],
                                   delta=0.002)

    def test_gpu_md_auto_without_a_memory_bound_stays_eager(self):
        """execution auto keeps an edges-only model eager, with the reason in its plan, when the measured
        bound does not fit, when the measurement runs out of memory, or when the model is not declared
        sync-free (an explicit model_graph then runs eager too, with a warning); a model_graph capture that
        runs out of memory falls back to eager, and the record says so."""
        _, _, _, _, reference = self.edges_only_md("eager", steps=3)
        unsafe = PairPotentialAdapter(contract=CaptureContract(sync_free=False, exact_cutoff=True))
        for case, patch, adapter, reason in (
                ("bound", mock.patch("fastmd.adapters.negotiate.capture_budget", return_value=1 << 20), None,
                 "GiB > "),
                ("probe", mock.patch.object(MeasuredMemoryModel, "measure",
                                            side_effect=torch.OutOfMemoryError("probe")), None,
                 "the memory measurement ran out of CUDA memory (probe)"),
                ("contract", nullcontext(), unsafe, "not declared sync-free")):
            for execution in ("auto", "model_graph") if case == "contract" else ("auto",):
                with self.subTest(case=case, execution=execution), patch:
                    record, plan, summary, path, final = self.edges_only_md(execution, adapter, steps=3)
                    decision = plan.decision("execution")
                    self.assertEqual((decision.chosen, summary["effective"]["execution"]), ("eager", "eager"))
                    self.assertIn(reason, decision.reason)
                    self.assertEqual(path, ("tensor_graph_builder", "rebuild", "none", True))
                    self.assert_plan_is_path(plan, summary, path)
                    self.assertFalse(any("negotiated" in d.reason for d in record.decisions))
                    for got, expected in zip(final, reference):
                        self.assertTrue(torch.equal(got, expected))
        with self.subTest("explicit model_graph warns"), self.assertWarnsRegex(ConfigurationWarning,
                                                                                "not declared sync-free"):
            GPUResidentMolecularDynamics(self.edges_only_frames()[0], model=unsafe, ensemble="nve", timestep=1.0,
                                         config=InferenceConfig(execution="model_graph", report=False)).close()
        with self.subTest("model_graph out of memory"), \
                mock.patch.object(GPUResidentMolecularDynamics, "_whole_step_trial", return_value=1.0), \
                mock.patch.object(GPUResidentMolecularDynamics, "enable_model_cuda_graph",
                                  side_effect=torch.OutOfMemoryError("model graph")):
            record, plan, summary, path, final = self.edges_only_md("auto", steps=3)
            self.assertEqual((plan.decision("execution").chosen, summary["effective"]["execution"]),
                             ("model_graph", "eager"))
            execution = record.decision("execution")
            self.assertEqual(execution.chosen, "eager")
            self.assertIn("negotiated model_graph; eager now: model_graph ran out of CUDA memory (model graph)",
                          execution.reason)
            self.assertEqual(record.decision("padding").chosen, "none")
            for got, expected in zip(final, reference):
                self.assertTrue(torch.equal(got, expected))

    def test_calculator_auto_plans_an_edges_only_adapter(self):
        """The calculator's execution auto for an edges-only model other than MatRIS: a system's first
        call runs eager, its first repeat call decides with a memory model measured on that system
        (model_graph with skin reuse when the bound fits, else eager with the reason); the negotiated plan
        names what every call ran (the skin path's builder read), except the rebuild after a cell change,
        which plan_record() notes; a new system starts from the construction plan. Results equal the
        adapter evaluated directly."""
        frames = self.edges_only_frames()
        frames = frames[:4] + frames[-1:]  # moves of one cell, then a new cell
        other = frames[0].repeat((1, 1, 2))
        for fits in (True, False):
            patch = nullcontext() if fits else mock.patch("fastmd.adapters.negotiate.capture_budget",
                                                          return_value=1 << 20)
            with self.subTest(fits=fits), patch, deterministic_algorithms():
                calc = MLIPCalculator(model=PairPotentialAdapter(), device="cuda", task="ef",
                                        config=InferenceConfig(report=False))
                self.assertIn("first repeat call decides", calc.plan_record().decision("execution").reason)
                for index, atoms in enumerate(frames + [other, other]):
                    calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
                    energy, forces, _ = self.edges_only_direct(calc.adapter, atoms)
                    self.assertEqual(calc.results["energy"], energy)
                    np.testing.assert_array_equal(calc.results["forces"], forces)
                    record, plan, effective = calc.plan_record(), calc._engine_plan[1], calc.config_summary()["effective"]
                    skin = effective["topology_reuse"]["last_call"] == "skin"
                    graph = effective["execution"] == "model_graph"
                    repeat = index in (1, 2, 3, 6)  # a system's repeat call (frame 4: a new cell, same atoms)
                    self.assertEqual((skin, graph), (fits and repeat, fits and index not in (0, 5)))
                    if skin:
                        self.assertEqual((calc._skin.owner.padding.kind, calc._skin.owner.edges_only),
                                         ("far_self_loops", True))
                    runtime = (effective["execution"], "exact_skin_refilter" if skin else "rebuild",
                               "fixed_capacity_builder" if skin else "gpu_graph_builder",
                               "far_self_loops" if skin or graph else "none")
                    names = ("execution", "topology_reuse", "neighbor_list", "padding")
                    cell_change = fits and index == 4
                    self.assertEqual(tuple(plan.decision(name).chosen for name in names),
                                     ("model_graph", "exact_skin_refilter", "fixed_capacity_builder",
                                      "far_self_loops") if cell_change else runtime)
                    self.assertEqual(tuple(record.decision(name).chosen for name in names), runtime)
                    noted = [d.id for d in record.decisions if "negotiated" in d.reason]
                    self.assertEqual(noted, ["neighbor_list", "topology_reuse"] if cell_change else [])
                    if index == 5:  # a new system's first call: the construction plan
                        self.assertIn("first repeat call decides", plan.decision("execution").reason)
                execution = plan.decision("execution")
                self.assertEqual(effective["auto_execution"]["reason"], execution.reason)
                self.assertIn("measured capture bound", execution.reason)
                self.assertIn("<=" if fits else ">", execution.reason)
                self.assertTrue(plan.memory["valid"], plan.memory["reason"])
        # The auto gate of a compiled model: measured and decided at a repeat call only where its contract
        # declares the compiled modules capture-safe; otherwise eager, as for dynamic compilation.
        from pair_potential_adapter import CONTRACT
        for compiled, decides in (("capture_safe", True), ("none", False)):
            with self.subTest(compiled=compiled):
                adapter = PairPotentialAdapter(contract=dataclasses.replace(CONTRACT, compiled=compiled))
                adapter.compiled = True
                calc = MLIPCalculator(model=adapter, device="cuda", task="ef", config=InferenceConfig(report=False))
                reason = calc.plan_record().decision("execution").reason
                self.assertEqual("first repeat call decides" in reason, decides, reason)
                self.assertEqual(calc.config_summary()["effective"]["auto_execution"]["mode"], "eager")
                if not decides:
                    self.assertIn("compiled model", reason)

    def test_gpu_md_graph_modes_match_eager_reference(self):
        atoms = make_atoms("partial")
        expected = self.reference(atoms)
        previous_indexed = model_functions._INDEXED_CAT_LINEAR_FUSE
        # model_graph precapture with and without room for the neighbor buckets
        # (CapturePolicy's byte rule): the envelope alone gives the same results.
        # The first capture warms up capacity.warmup times, a later one once
        # (twice where recompute_policy "fast" still has to check a decision).
        for execution, neighbors, warmup, pending in (
                ("model_graph", True, 1, False), ("model_graph", False, 1, False), ("model_graph", True, 3, False),
                ("model_graph", True, 3, True), ("whole_step", True, 1, False)):
            config = self.config(execution)
            config = dataclasses.replace(config, capacity=dataclasses.replace(config.capacity, warmup=warmup))
            with self.subTest(execution=execution, neighbors=neighbors, warmup=warmup, pending=pending), \
                    mock.patch("fastmd.execution.cuda_graph.precapture_fits",
                               side_effect=lambda *args: (neighbors, "test")), \
                    mock.patch("fastmd.engine.saved_recompute.capture_check_pending", return_value=pending), \
                    GPUResidentMolecularDynamics(atoms, model=self.model(), task="ef", device="cuda",
                                                 ensemble="nve", timestep=0.0, config=config) as md:
                self.assertFalse(md.validate_isolated_atoms)
                before_positions = md.state.positions.detach().clone()
                for step in range(1, 3):
                    md.run(1)
                    self.assertEqual(md.nsteps, step)
                    self.assert_finite_close(md.state.potential_energy, md.state.forces, expected)
                    torch.testing.assert_close(md.state.positions, before_positions, rtol=0, atol=0)
                # The requested graph mode ran.
                if execution == "model_graph":
                    self.assertIsNotNone(md.runner)
                    stats = md.runner.stats()
                    self.assertEqual(len(stats["cached_buckets"]) > 1, neighbors)
                    later = 1 + (pending and warmup > 1)
                    self.assertEqual(stats["warmup_runs"], warmup + later * (stats["captures"] - 1))
                else:
                    self.assertIsNotNone(md.whole_step_runner)
                    self.assertEqual(md.whole_step_runner.fallback_windows, 0)
            self.assertEqual(model_functions._INDEXED_CAT_LINEAR_FUSE, previous_indexed)

    def test_adapter_outputs_equal_the_direct_model_call(self):
        assert_adapter_matches_model(self, "cuda")

    def test_explicit_strict_isolation_rejected(self):
        atoms = make_atoms("partial")
        with self.subTest(entrypoint="calculator"):
            calc = MLIPCalculator(
                model=self.model(), device="cuda", task="ef",
                config=self.config("model_graph", isolated_atoms="error"),
            )
            with self.assertRaisesRegex(ValueError, "isolated atom"):
                calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
        for execution in ("model_graph", "whole_step"):
            with self.subTest(entrypoint="gpu_md", execution=execution):
                with self.assertRaisesRegex(ValueError, "isolated atom"):
                    GPUResidentMolecularDynamics(
                        atoms, model=self.model(), device="cuda", task="ef",
                        ensemble="nve", timestep=0.0,
                        config=self.config(execution, isolated_atoms="error"),
                    )


@unittest.skipUnless(
    torch.cuda.is_available() and op_available and _fixed_builder_available,
    "CUDA graph builders are unavailable",
)
class ReplicaMolecularDynamicsCudaTest(unittest.TestCase):
    """ReplicaMolecularDynamics (replicas of one system advanced together as one disjoint union in whole-step
    CUDA graphs) with the test pair potential behind a per-atom energy tap that declares "union"
    (UnionPairAdapter; a whole-system tap declaring "segments", SystemTapPairAdapter, where named), at ten
    times its default strength so that its forces move the atoms by about 0.2 A
    in 40 steps: 108 Cu atoms per replica, each replica moved by up to 0.05 A with momenta at 600 K, a 900 K
    target, 2 fs steps in windows of 5, deterministic scatters. Trajectories that must agree do within the
    float32 rounding of the state (:meth:`assert_same_state`); measured: bit for bit (positions, momenta,
    forces), against single-system GPU-resident MD and across the window paths alike (without deterministic
    scatters: up to 1e-6 A, and 2e-6 of the largest momentum or force)."""

    STEPS, WINDOW = 40, 5
    TOLERANCE = 1e-5  # positions in A; momenta and forces relative to their largest value

    @staticmethod
    def crystal() -> Atoms:
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(3)  # 108 atoms
        atoms.rattle(0.05, seed=3)
        return atoms

    @staticmethod
    def start(atoms: Atoms, replicas: int, seed: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Positions and momenta (``(replicas, n, 3)``) of replicas of ``atoms``: every atom moved by up to
        0.05 A, Maxwell-Boltzmann momenta at 600 K with seed ``seed + replica``."""
        rng = np.random.default_rng(11)
        positions = atoms.positions + rng.uniform(-0.05, 0.05, (replicas, len(atoms), 3))
        momenta = np.stack([maxwell_boltzmann_momenta(atoms.get_masses(), 600.0, seed=seed + replica)
                            for replica in range(replicas)])
        return positions, momenta

    @staticmethod
    def adapter() -> UnionPairAdapter:
        return UnionPairAdapter(SmoothPairPotential(strength=0.5).cuda())

    @staticmethod
    def config(candidate_skin: float = 1.0) -> InferenceConfig:
        return InferenceConfig(execution="whole_step", report=False,
                               capacity=CapacityConfig(warmup=1, candidate_skin=candidate_skin))

    def replica_md(self, adapter, atoms, positions, momenta, ensemble="nvt_nhc", **options):
        """The replicas' MD (the class's dynamics), without the self-check unless asked for."""
        options = {"config": self.config(), "self_check": False, "steps_per_launch": self.WINDOW, **options}
        return ReplicaMolecularDynamics(adapter, atoms, positions=positions, momenta=momenta, ensemble=ensemble,
                                        temperature=900.0, timestep=2.0, tdamp=40.0, taut=40.0, **options)

    @staticmethod
    def state(md) -> tuple[np.ndarray, ...]:
        """Every replica's positions, momenta and forces (``(N, n, 3)``) and potential energy (``(N,)``)."""
        return md.get_positions(), md.get_momenta(), md.get_forces(), md.potential_energies().cpu().numpy()

    def assert_same_state(self, actual, expected, what: str) -> None:
        """Positions within TOLERANCE A, momenta and forces within TOLERANCE of their largest value, energies
        relative 1e-6 (the float32 rounding of the state, and of one system's energy)."""
        for name, ours, theirs in zip(("positions", "momenta", "forces", "energy"), actual, expected):
            theirs = np.asarray(theirs)
            self.assertTrue(np.isfinite(ours).all(), f"{what}: {name}")
            if name == "energy":
                np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=0, err_msg=f"{what}: {name}")
                continue
            scale = 1.0 if name == "positions" else float(np.abs(theirs).max())
            np.testing.assert_allclose(ours, theirs, rtol=0, atol=self.TOLERANCE * scale, err_msg=f"{what}: {name}")

    def test_replicas_match_single_system_md(self):
        """Four replicas in one union, nvt_nhc, Berendsen nvt and nve (and nvt_nhc for the potential behind a
        whole-system tap that declares "segments", one energy per system, behind the per-atom tap declaring
        "segments", summed per system like "union", and with a three-body term over a line graph, whose
        triplets the union lists per replica): after 40 steps each replica's
        positions, momenta, forces and potential energy (and the Nose-Hoover chain's conserved energy, relative
        1e-6) equal single-system GPU-resident MD of that replica (whole_step); the plan batches the systems as
        one union of four (the model's declaration) after the self-check (two copies of the system as a union
        and alone) passed."""
        atoms = self.crystal()
        positions, momenta = self.start(atoms, 4)
        module = SmoothPairPotential(strength=0.5).cuda()
        cases = [(UnionPairAdapter(module), ensemble) for ensemble in ("nvt_nhc", "nvt", "nve")]
        cases += [(SystemTapPairAdapter(module, declare="segments"), "nvt_nhc"),
                  (UnionPairAdapter(module, declare="segments"), "nvt_nhc"),
                  (UnionPairAdapter(module, triplet_cutoff=3.3, angle=0.5), "nvt_nhc")]
        with deterministic_algorithms():
            for adapter, ensemble in cases:
                line_graph = adapter.topology_spec().triplet_cutoff is not None
                with self.subTest(batching=adapter.system_batching(), ensemble=ensemble, line_graph=line_graph):
                    with self.replica_md(adapter, atoms, positions, momenta, ensemble, self_check=True) as md:
                        md.run(self.STEPS)
                        ours, conserved = self.state(md), md.conserved_energies().cpu().numpy()
                        stats, plan = md.stats(), md.plan_record()
                    self.assertGreater(stats["launches"], 0)  # the captured windows ran
                    if line_graph:
                        self.assertGreater(stats["max_triplet_count"][0], 0)  # the three-body term had triplets
                    self.assertEqual(stats["unions"], [4])
                    self.assertEqual(plan.decision("union_size").chosen, "4")
                    batching = plan.decision("system_batching")
                    self.assertEqual(batching.chosen, adapter.system_batching())
                    self.assertTrue(batching.evidence["self_check"]["passed"])
                    for replica in range(4):
                        probe = atoms.copy()
                        probe.positions = positions[replica]
                        probe.set_momenta(momenta[replica])
                        with GPUResidentMolecularDynamics(probe, model=adapter, ensemble=ensemble, temperature=900.0,
                                                          timestep=2.0, tdamp=40.0, taut=40.0, config=self.config(),
                                                          steps_per_launch=self.WINDOW) as single:
                            single.run(self.STEPS)
                            final = single.snapshot_atoms()
                            theirs = (final.positions, final.get_momenta(), single.state.forces.cpu().numpy(),
                                      float(single.state.potential_energy.reshape(-1)[0]))
                            if ensemble == "nvt_nhc":
                                expected = float(single.integrator.conserved_energy())
                                np.testing.assert_allclose(conserved[replica], expected, rtol=1e-6, atol=0,
                                                           err_msg=f"replica {replica}: conserved energy")
                        self.assert_same_state([value[replica] for value in ours], theirs, f"replica {replica}")

    def test_replica_md_rejects_models_and_arguments(self):
        """NegotiationError for a model that differentiates its own energy (PairPotentialAdapter), one that does
        not declare its systems' batching (per-atom or whole-system tap, with or without a line graph) or declares
        an unknown one, a whole-system tap that declares "union" (one energy for the whole union), or "segments"
        but returns one energy for the whole batch, sink-atom padding, a line graph padded otherwise than with far
        self-loops (the union builder's self-loops; padded triplets on padded pairs), and two that
        declare "union" but are not local (atoms tied to the centroid of all rows: the self-check's union of two
        copies differs from the copies alone in force; an energy shared out over all rows: in energy only);
        ValueError for a cell that is not
        periodic in all three directions, momenta of the wrong shape or not finite, an unknown ensemble, nvt_nhc
        with fix_com, a loginterval that is not a multiple of steps_per_launch, a trajectory name without
        "{replica}", inconsistent replica counts, a union size below 1, and a run of steps that are not whole
        windows."""
        atoms = self.crystal()
        positions, momenta = self.start(atoms, 2)
        module = SmoothPairPotential(strength=0.5).cuda()
        sinks, exact_loops = UnionPairAdapter(module), UnionPairAdapter(module, triplet_cutoff=3.3, angle=0.5)
        sinks.padding_rule = lambda: PaddingRule("sink_atoms")
        exact_loops.padding_rule = lambda: PaddingRule("exact_rc_loops")
        for model, keyword in ((PairPotentialAdapter(module), "differentiates its own energy"),
                               (SystemTapPairAdapter(module), "does not declare"),
                               (SystemTapPairAdapter(module, declare="union"), "whole-system energy tap"),
                               (SystemTapPairAdapter(module, declare="segments", segments=False),
                                "returned 1 value"),
                               (UnionPairAdapter(module, declare=None), "does not declare"),
                               (UnionPairAdapter(module, declare="graphs"), "batching 'graphs'"),
                               (UnionPairAdapter(module, declare=None, triplet_cutoff=3.3, angle=0.5),
                                "does not declare"),
                               (exact_loops, "far self-loops"),
                               (sinks, "sink atoms"),
                               (UnionPairAdapter(module, tether=0.1), "alone in force"),
                               (UnionPairAdapter(module, batch_energy=1.0), "alone in energy")):
            with self.subTest(model=model.name, keyword=keyword), self.assertRaisesRegex(NegotiationError, keyword):
                self.replica_md(model, atoms, positions, momenta, self_check=True)
        flat = atoms.copy()
        flat.pbc = (True, True, False)
        poisoned = momenta.copy()
        poisoned[1, 3, 0] = np.nan
        adapter = UnionPairAdapter(module)
        cases = {
            "periodic in two directions": ({"atoms": flat}, "periodic"),
            "momenta (N, n - 1, 3)": ({"momenta": momenta[:, :-1]}, "momenta"),
            "momenta (n, 3)": ({"momenta": momenta[0]}, "momenta"),
            "NaN momenta": ({"momenta": poisoned}, "momenta"),
            "unknown ensemble": ({"ensemble": "npt"}, "ensemble"),
            "nvt_nhc with fix_com": ({"fix_com": True}, "fix_com"),
            "loginterval 7, windows of 5": ({"loginterval": 7}, "loginterval"),
            "trajectory without {replica}": ({"trajectory": "md.traj"}, "trajectory"),
            "3 replicas, momenta of 2": ({"replicas": 3}, "replicas"),
            "union size 0": ({"union_size": 0}, "union_size"),
        }
        for case, (options, keyword) in cases.items():
            options = {"atoms": atoms, "momenta": momenta, **options}
            with self.subTest(case), self.assertRaisesRegex(ValueError, keyword):
                self.replica_md(adapter, options.pop("atoms"), positions, options.pop("momenta"), **options)
        with self.replica_md(adapter, atoms, positions, momenta) as md:
            for steps in (7, -5):
                with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "steps"):
                    md.run(steps)

    def test_replica_md_capacity_growth_recaptures_the_union(self):
        """A model whose capacity tiers add no padding (CapacityStep of 8 pairs, no minimum padding): the
        union's first capture holds the demand of the start, which the pair count of the moving crystal
        outgrows: the window is rolled back, the capacity grows and the union is captured again (recaptures
        >= 1, the window counters kept over the captures), and the trajectory equals that of tiers padding
        4,096 pairs, which never overflow (three replicas, nvt_nhc and nve). With a line graph (nvt_nhc), tiers
        of 8 pairs and 8 triplets without padding on the triplets, the pairs or both: the capacities that
        overflow grow and the other stays, and the trajectory equals that of tiers padding 4,096 pairs and
        16,384 triplets."""
        atoms = self.crystal()
        positions, momenta = self.start(atoms, 3)
        module = SmoothPairPotential(strength=0.5).cuda()
        line_graph = dict(triplet_cutoff=3.3, angle=0.5)
        # (ensemble, triplet cutoff and angle, {padding: (min_pad_u, min_pad_t)}, the capacities that grow)
        cases = [("nvt_nhc", None, {"padded": (4096, None), "grown": (0, None)}, ("u_capacity",)),
                 ("nve", None, {"padded": (4096, None), "grown": (0, None)}, ("u_capacity",))]
        cases += [("nvt_nhc", line_graph, {"padded": (4096, 16384), "grown": grown}, capacities)
                  for grown, capacities in (((4096, 0), ("t_capacity",)), ((0, 16384), ("u_capacity",)),
                                            ((0, 0), ("u_capacity", "t_capacity")))]
        with deterministic_algorithms():
            for ensemble, options, paddings, growing in cases:
                with self.subTest(ensemble=ensemble, line_graph=options is not None, growing=growing):
                    runs = {}
                    for name, (pad_u, pad_t) in paddings.items():
                        adapter = UnionPairAdapter(module, **(options or {}))
                        triplet_tiers = {} if pad_t is None else {"t_step": 8, "min_pad_t": pad_t}
                        adapter.capacity_step = lambda pad_u=pad_u, tiers=triplet_tiers: CapacityStep(
                            u_step=8, min_pad_u=pad_u, **tiers)
                        with self.replica_md(adapter, atoms, positions, momenta, ensemble) as md:
                            captured = md.stats()
                            md.run(self.STEPS)
                            runs[name] = self.state(md), md.stats(), captured
                    (reference, padded, _), (grown, stats, captured) = runs["padded"], runs["grown"]
                    self.assertEqual(padded["recaptures"], 0)
                    self.assertGreaterEqual(stats["recaptures"], 1)
                    for capacity in ("u_capacity", "t_capacity") if options is not None else ("u_capacity",):
                        if capacity in growing:
                            self.assertGreater(stats[capacity][0], captured[capacity][0], capacity)
                        else:
                            self.assertEqual(stats[capacity], captured[capacity], capacity)
                    # every window counted, over both captures
                    self.assertGreaterEqual(stats["launches"], self.STEPS // self.WINDOW)
                    self.assert_same_state(grown, reference, f"{ensemble}: grown capacity")

    def test_replica_md_unions_share_one_graph_pool(self):
        """Five replicas in unions of at most two (sizes 2, 2 and 1, captured into one graph pool and replayed
        one after another) equal one union of five, replica by replica (state, energies and health in the
        replicas' order)."""
        atoms = self.crystal()
        positions, momenta = self.start(atoms, 5)
        adapter = self.adapter()
        runs = {}
        with deterministic_algorithms():
            for union_size in (None, 2):
                with self.replica_md(adapter, atoms, positions, momenta, union_size=union_size) as md:
                    md.run(self.STEPS)
                    runs[union_size] = self.state(md), md.stats(), md.health(), md.plan_record()
        (reference, one, _, _), (split, stats, health, plan) = runs[None], runs[2]
        self.assertEqual((one["unions"], stats["unions"]), ([5], [2, 2, 1]))
        self.assertEqual(plan.decision("union_size").chosen, "2")
        self.assertGreater(stats["launches"], 0)
        np.testing.assert_array_equal(health, [True] * 5)
        self.assert_same_state(split, reference, "unions of at most two")

    def test_replica_md_fallback_windows_keep_the_trajectory(self):
        """Windows of 10 steps on a 0.1 A candidate skin, less than twice what the atoms move in a window: the
        windows fall back (rolled back, replayed on a fresh candidate list, then run eagerly with a fresh list at
        every step: fallback_windows and recovery_replays > 0), and the trajectory equals the run on the default
        2 A skin, which never falls back (three replicas, nvt_nhc). The potential has power 2 here, so that the
        pairs near the cutoff carry force: a window kept on its stale candidate list would differ by about
        2e-3 A."""
        atoms = self.crystal()
        positions, momenta = self.start(atoms, 3)
        module = SmoothPairPotential(power=2, strength=0.5).cuda()
        runs = {}
        with deterministic_algorithms():
            for skin in (2.0, 0.1):
                with self.replica_md(UnionPairAdapter(module), atoms, positions, momenta, config=self.config(skin),
                                     steps_per_launch=10) as md:
                    md.run(self.STEPS)
                    runs[skin] = self.state(md), md.stats()
        (reference, plain), (fallen, stats) = runs[2.0], runs[0.1]
        self.assertEqual(plain["fallback_windows"], 0)
        self.assertGreater(stats["fallback_windows"], 0)
        self.assertGreater(stats["recovery_replays"], 0)
        self.assert_same_state(fallen, reference, "candidate fallback")

    def test_replica_md_restore_rewinds_and_takes_new_momenta(self):
        """restore(snapshot) rewinds every replica, step count included (the next 20 steps repeat), and
        restore(snapshot, momenta=P) continues like a fresh run from the snapshot's positions with momenta P
        (the snapshot taken at the start, where the Nose-Hoover chains are a fresh run's; nvt_nhc and Berendsen
        nvt)."""
        atoms = self.crystal()
        positions, momenta = self.start(atoms, 3)
        _, other = self.start(atoms, 3, seed=40)
        adapter = self.adapter()
        steps = 20
        with deterministic_algorithms():
            for ensemble in ("nvt_nhc", "nvt"):
                with self.subTest(ensemble=ensemble):
                    with self.replica_md(adapter, atoms, positions, other, ensemble) as fresh:
                        fresh.run(steps)
                        expected = self.state(fresh)
                    with self.replica_md(adapter, atoms, positions, momenta, ensemble) as md:
                        snapshot = md.snapshot()
                        md.run(steps)
                        first = self.state(md)
                        md.restore(snapshot)
                        self.assertEqual(md.nsteps, 0)
                        md.run(steps)
                        self.assert_same_state(self.state(md), first, f"{ensemble}: rewound")
                        md.restore(snapshot, momenta=other)
                        md.run(steps)
                        self.assert_same_state(self.state(md), expected, f"{ensemble}: new momenta")

    def test_replica_md_names_the_replica_that_turned_non_finite(self):
        """NaN momenta restored into one atom of replica 2 of four: the next window raises ReplicaHealthError
        naming replica 2 only (its index among all replicas, also where it is the first of a second union),
        health() marks it, and the other replicas' states stay finite (nvt_nhc and Berendsen nvt)."""
        atoms = self.crystal()
        positions, momenta = self.start(atoms, 4)
        adapter = self.adapter()
        for ensemble, union_size in (("nvt_nhc", None), ("nvt", None), ("nvt_nhc", 2)):
            with self.subTest(ensemble=ensemble, union_size=union_size), \
                    self.replica_md(adapter, atoms, positions, momenta, ensemble, union_size=union_size) as md:
                md.run(self.WINDOW)
                poisoned = md.get_momenta()
                poisoned[2, 7, 1] = np.nan
                md.restore(md.snapshot(), momenta=poisoned)
                with self.assertRaises(ReplicaHealthError) as caught:
                    md.run(self.WINDOW)
                self.assertEqual(caught.exception.replicas, [2])
                self.assertIn("[2]", str(caught.exception))
                np.testing.assert_array_equal(md.health(), [True, True, False, True])
                for values in (md.get_positions(), md.get_momenta()):
                    self.assertTrue(np.isfinite(values[[0, 1, 3]]).all())

    def test_replica_md_writes_a_trajectory_and_log_per_replica(self):
        """trajectory with "{replica}" and a logfile: each replica's file holds a frame at the start and every
        loginterval steps (three after 20 steps at loginterval 10) with that replica's state, bit for bit (the
        float64 positions and momenta, the forces and the energy of the getters; at the start the system's
        positions and the default momenta, ASE's Maxwell-Boltzmann draw with seed + replica), and the log one line
        per replica and logged step."""
        atoms = self.crystal()
        with tempfile.TemporaryDirectory() as directory:
            pattern = os.path.join(directory, "md_{replica}.traj")
            logfile = os.path.join(directory, "md.log")
            with ReplicaMolecularDynamics(self.adapter(), atoms, replicas=3, temperature=600.0, timestep=2.0, seed=7,
                                          steps_per_launch=self.WINDOW, trajectory=pattern, logfile=logfile,
                                          loginterval=10, config=self.config(), self_check=False) as md:
                md.run(20)
                final = self.state(md)
            frames = [ase.io.read(pattern.format(replica=replica), ":") for replica in range(3)]
            with open(logfile) as handle:
                lines = handle.read().splitlines()
        self.assertEqual([len(replica_frames) for replica_frames in frames], [3, 3, 3])
        for replica, replica_frames in enumerate(frames):
            first, last = replica_frames[0], replica_frames[-1]
            np.testing.assert_array_equal(first.positions, atoms.positions)
            np.testing.assert_array_equal(first.get_momenta(),
                                          maxwell_boltzmann_momenta(atoms.get_masses(), 600.0, seed=7 + replica))
            for name, value, expected in zip(("positions", "momenta", "forces", "energy"),
                                             (last.positions, last.get_momenta(), last.get_forces(),
                                              last.get_potential_energy()), final):
                np.testing.assert_array_equal(value, expected[replica], err_msg=f"replica {replica}: {name}")
        self.assertEqual(len(lines), 1 + 3 * 3)  # a header, then a line per replica and logged step
        self.assertEqual([int(line.split()[1]) for line in lines[1:]], [0, 1, 2] * 3)
        self.assertEqual([float(line.split()[0]) for line in lines[1::3]], [0.0, 0.02, 0.04])  # ps


@unittest.skipUnless(
    torch.cuda.is_available() and op_available and _fixed_builder_available,
    "CUDA graph builders are unavailable",
)
class MultiSystemMolecularDynamicsCudaTest(unittest.TestCase):
    """MultiSystemMolecularDynamics (different systems in the slots of one union, advanced together in whole-step
    CUDA graphs) with the test pair potential behind a per-atom energy tap that declares "union" and takes the
    engine's per-edge shifts (UnionPairAdapter), at ten times its default strength: four systems of different
    atoms and cells (108-atom cubic Cu, 64-atom triclinic Cu, 24-atom orthorhombic Al, 36-atom triclinic Si,
    each moved by up to 0.05 A) in slots of 108, 70, 24 and 40 atoms (two with vacant rows), momenta at 600 K, a
    900 K target, 2 fs steps in windows of 5, deterministic scatters. Each system is compared with single-system
    GPU-resident MD of it within the float32 rounding of the state (ReplicaMolecularDynamicsCudaTest's
    tolerances)."""

    STEPS, WINDOW, SLOTS = 40, 5, (108, 70, 24, 40)
    TOLERANCE = ReplicaMolecularDynamicsCudaTest.TOLERANCE

    @staticmethod
    def systems() -> list[Atoms]:
        systems = [bulk("Cu", "fcc", a=3.6, cubic=True).repeat(3), bulk("Cu", "fcc", a=3.6).repeat(4),
                   bulk("Al", "fcc", a=4.05, orthorhombic=True).repeat((2, 3, 2)),
                   bulk("Si", "diamond", a=5.43).repeat((3, 3, 2))]
        for seed, atoms in enumerate(systems):
            atoms.positions += np.random.default_rng(seed).uniform(-0.05, 0.05, atoms.positions.shape)
            atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), 600.0, seed=20 + seed))
        return systems

    def multi_md(self, adapter, systems, ensemble="nvt_nhc", **options):
        options = {"config": ReplicaMolecularDynamicsCudaTest.config(), "self_check": False,
                   "steps_per_launch": self.WINDOW, "slot_atoms": list(self.SLOTS), **options}
        return MultiSystemMolecularDynamics(adapter, systems, momenta=[atoms.get_momenta() for atoms in systems],
                                            ensemble=ensemble, temperature=900.0, timestep=2.0, tdamp=40.0,
                                            taut=40.0, **options)

    def single(self, adapter, atoms, ensemble, steps):
        """Single-system GPU-resident MD of ``atoms`` (its positions and momenta) for ``steps``: its positions,
        momenta, forces and potential energy."""
        with GPUResidentMolecularDynamics(atoms.copy(), model=adapter, ensemble=ensemble, temperature=900.0,
                                          timestep=2.0, tdamp=40.0, taut=40.0,
                                          config=ReplicaMolecularDynamicsCudaTest.config(),
                                          steps_per_launch=self.WINDOW) as single:
            single.run(steps)
            final = single.snapshot_atoms()
            return (final.positions, final.get_momenta(), single.state.forces.cpu().numpy(),
                    float(single.state.potential_energy.reshape(-1)[0]))

    @staticmethod
    def state(md, index: int) -> tuple:
        return (md.get_positions()[index], md.get_momenta()[index], md.get_forces()[index],
                float(md.potential_energies()[index]))

    def assert_same_state(self, actual, expected, what: str) -> None:
        ReplicaMolecularDynamicsCudaTest.assert_same_state(self, actual, expected, what)

    def test_systems_match_single_system_md(self):
        """nvt_nhc (with the self-check: the two smallest systems as a union and alone), Berendsen nvt and nve:
        after 40 steps each system's positions, momenta, forces and potential energy equal single-system
        GPU-resident MD of it (whole_step); the plan batches the four systems in one union of the slots' 242 rows;
        the vacant rows never move."""
        systems, adapter = self.systems(), ReplicaMolecularDynamicsCudaTest.adapter()
        with deterministic_algorithms():
            for ensemble in ("nvt_nhc", "nvt", "nve"):
                with self.subTest(ensemble=ensemble):
                    with self.multi_md(adapter, systems, ensemble, self_check=ensemble == "nvt_nhc") as md:
                        md.run(self.STEPS)
                        ours = [self.state(md, index) for index in range(len(systems))]
                        stats, plan = md.stats(), md.plan_record()
                        union = md._unions[0]
                        vacant = ~union.runner.builder.occupied
                        self.assertTrue(bool((union.runner.positions[vacant] == 0).all()), "a vacant row moved")
                    self.assertGreater(stats["launches"], 0)
                    self.assertEqual(stats["slot_atoms"], [list(self.SLOTS)])
                    self.assertEqual(plan.decision("union_size").evidence["unions"], [sum(self.SLOTS)])
                    batching = plan.decision("system_batching")
                    self.assertEqual(batching.chosen, "union")
                    if ensemble == "nvt_nhc":
                        self.assertTrue(batching.evidence["self_check"]["passed"])
                        self.assertEqual(batching.evidence["self_check"]["atoms"], [24, 36])
                    for index, atoms in enumerate(systems):
                        self.assert_same_state(ours[index], self.single(adapter, atoms, ensemble, self.STEPS),
                                               f"{ensemble}: system {index}")

    def test_write_system_equals_a_fresh_run(self):
        """After 20 steps, a 54-atom triclinic Cu written into the 70-atom slot of the 64-atom one and a 40-atom
        Si of another cell into the 40-atom slot of the 36-atom one (nvt_nhc and nve), without a recapture: over
        the next 40 steps each runs as a fresh single-system run from the written state (its fresh thermostat),
        and the other two systems as their own single-system runs of 60 steps; the slots' atom counts follow."""
        adapter = ReplicaMolecularDynamicsCudaTest.adapter()
        written = {1: bulk("Cu", "fcc", a=3.6).repeat((3, 3, 6)), 3: bulk("Si", "diamond", a=5.43).repeat((2, 2, 5))}
        for seed, atoms in written.items():
            atoms.positions += np.random.default_rng(10 + seed).uniform(-0.05, 0.05, atoms.positions.shape)
            atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), 600.0, seed=30 + seed))
        with deterministic_algorithms():
            for ensemble in ("nvt_nhc", "nve"):
                with self.subTest(ensemble=ensemble):
                    systems = self.systems()
                    with self.multi_md(adapter, systems, ensemble) as md:
                        md.run(20)
                        for index, atoms in written.items():
                            md.write_system(index, atoms, momenta=atoms.get_momenta())
                        md.run(self.STEPS)
                        ours = [self.state(md, index) for index in range(len(systems))]
                        stats = md.stats()
                    self.assertEqual(stats["recaptures"], 0)
                    self.assertEqual(stats["atoms"], [[108, 54, 24, 40]])
                    for index, atoms in enumerate(systems):
                        if index in written:
                            expected = self.single(adapter, written[index], ensemble, self.STEPS)
                        else:
                            expected = self.single(adapter, atoms, ensemble, 20 + self.STEPS)
                        self.assert_same_state(ours[index], expected, f"{ensemble}: system {index}")

    def test_capacity_growth_with_different_systems(self):
        """A model whose capacity tiers add no padding (CapacityStep of 8 pairs): the union's first capture holds
        the demand of the start, which the moving systems outgrow, and so do compressed Cu crystals written into
        the first and the last slot (more pairs within the cutoff: the written system's forces come from the
        system alone, not from the union's truncated edges): each time the window is rolled back, the pair
        capacity grows and the union is captured again (recaptures >= 2), and every system's trajectory equals
        that of tiers padding 4,096 pairs, which never overflow (nvt_nhc)."""
        module = SmoothPairPotential(strength=0.5).cuda()
        written = {0: bulk("Cu", "fcc", a=3.45, cubic=True).repeat(3),
                   3: bulk("Cu", "fcc", a=3.4, cubic=True).repeat(2)}
        momenta = {}
        for index, atoms in written.items():
            atoms.positions += np.random.default_rng(7 + index).uniform(-0.05, 0.05, atoms.positions.shape)
            momenta[index] = maxwell_boltzmann_momenta(atoms.get_masses(), 600.0, seed=40 + index)
        runs = {}
        with deterministic_algorithms():
            for name, pad in (("padded", 4096), ("grown", 0)):
                adapter = UnionPairAdapter(module)
                adapter.capacity_step = lambda pad=pad: CapacityStep(u_step=8, min_pad_u=pad)
                with self.multi_md(adapter, self.systems(), capacity_headroom=0.0) as md:
                    md.run(20)
                    for index, atoms in written.items():
                        md.write_system(index, atoms, momenta=momenta[index])
                    md.run(20)
                    runs[name] = [self.state(md, index) for index in range(4)], md.stats()
        (reference, padded), (grown, stats) = runs["padded"], runs["grown"]
        self.assertEqual(padded["recaptures"], 0)
        self.assertGreaterEqual(stats["recaptures"], 2)
        for index in range(4):
            self.assert_same_state(grown[index], reference[index], f"grown capacity: system {index}")

    def test_unions_share_one_graph_pool_and_write_files_per_system(self):
        """The four systems in two unions of consecutive systems (union_rows 150: 108 and 134 rows, captured into
        one graph pool and replayed one after another) equal one union of the four, system by system (state,
        energies and health in the systems' order); trajectory with "{system}" and a logfile: each system's file
        holds a frame at the start and every loginterval steps (three after 20 steps at loginterval 10) with that
        system's state, bit for bit, and the log one line per system and logged step."""
        adapter = ReplicaMolecularDynamicsCudaTest.adapter()
        systems = self.systems()
        runs = {}
        with deterministic_algorithms(), tempfile.TemporaryDirectory() as directory:
            pattern = os.path.join(directory, "md_{system}.traj")
            logfile = os.path.join(directory, "md.log")
            for name, rows in (("one", None), ("two", 150)):
                files = dict(trajectory=pattern, logfile=logfile, loginterval=10) if name == "two" else {}
                with self.multi_md(adapter, systems, union_rows=rows, **files) as md:
                    md.run(20)
                    runs[name] = [self.state(md, index) for index in range(4)], md.health().tolist(), md.stats()
            frames = [ase.io.read(pattern.format(system=index), ":") for index in range(4)]
            with open(logfile) as handle:
                lines = handle.read().splitlines()
        self.assertEqual(runs["one"][2]["unions"], [4])
        self.assertEqual(runs["two"][2]["unions"], [1, 3])
        self.assertEqual(runs["two"][1], [True] * 4)
        for index in range(4):
            self.assert_same_state(runs["two"][0][index], runs["one"][0][index], f"system {index}")
        self.assertEqual([len(system_frames) for system_frames in frames], [3] * 4)
        for index, system_frames in enumerate(frames):
            first, last = system_frames[0], system_frames[-1]
            np.testing.assert_array_equal(first.positions, systems[index].positions)
            np.testing.assert_array_equal(first.get_momenta(), systems[index].get_momenta())
            for name, value, expected in zip(("positions", "momenta", "forces", "energy"),
                                             (last.positions, last.get_momenta(), last.get_forces(),
                                              last.get_potential_energy()), runs["two"][0][index]):
                np.testing.assert_array_equal(value, expected, err_msg=f"system {index}: {name}")
        self.assertEqual(len(lines), 1 + 3 * 4)  # a header, then a line per system and logged step
        self.assertEqual([int(line.split()[1]) for line in lines[1:]], [0, 1, 2, 3] * 3)

    def test_names_the_system_that_turned_non_finite(self):
        """A NaN momentum written into one atom of system 1 (of four, in the state the integrator steps): the next
        window raises ReplicaHealthError naming system 1 only ("1 system(s) [1]"), health() marks it, and the
        other systems' states and energies stay finite (nvt_nhc and Berendsen nvt, whose per-system sums read
        each system's own rows)."""
        adapter = ReplicaMolecularDynamicsCudaTest.adapter()
        for ensemble in ("nvt_nhc", "nvt"):
            with self.subTest(ensemble=ensemble), self.multi_md(adapter, self.systems(), ensemble) as md:
                md.run(self.WINDOW)
                union = md._unions[0]
                if ensemble == "nvt_nhc":
                    union.integrator.system_view(union.integrator.state.p, 1)[3, 1] = float("nan")
                else:
                    union.runner.momenta[union.layout.starts[1] + 3, 1] = float("nan")
                with self.assertRaises(ReplicaHealthError) as caught:
                    md.run(self.WINDOW)
                self.assertEqual(caught.exception.replicas, [1])
                self.assertIn("1 system(s) [1]", str(caught.exception))
                np.testing.assert_array_equal(md.health(), [True, False, True, True])
                for values in (md.get_positions(), md.get_momenta()):
                    self.assertTrue(all(np.isfinite(values[index]).all() for index in (0, 2, 3)))
                self.assertTrue(bool(torch.isfinite(md.potential_energies()[[0, 2, 3]]).all()))

    def test_rejects_models_systems_and_swaps(self):
        """NegotiationError for a model that does not take the engine's per-edge shifts (a whole-system tap among
        them), one that declares "segments", one with a line graph, one that differentiates its own energy, and
        one that declares the shifts but reads the cell (the self-check: two systems in one cell, the union with
        twice the cell as the model's argument). ValueError for a system that is not periodic in all three
        directions, has a singular cell or non-finite coordinates, or more atoms than its slot's ceiling;
        write_system rejects the same, an index out of range and a cell thinner than the union's sort keys allow
        (a union whose keys are narrow), before changing anything: the union then runs as before. restore
        refuses a snapshot from before a write and momenta of another shape."""
        module = SmoothPairPotential(strength=0.5).cuda()
        systems = self.systems()
        for adapter, reason in ((UnionPairAdapter(module, edge_shifts=False), "per-edge shifts"),
                                (UnionPairAdapter(module, declare="segments"), "'segments'"),
                                (SystemTapPairAdapter(module, declare="segments"), "per-edge shifts"),
                                (UnionPairAdapter(module, triplet_cutoff=3.3, angle=0.5), "line graph"),
                                (PairPotentialAdapter(module), "differentiates its own energy")):
            with self.subTest(reason=reason), self.assertRaisesRegex(NegotiationError, reason):
                self.multi_md(adapter, systems)
        one_cell = [systems[0], systems[0].copy()]
        one_cell[1].positions += np.random.default_rng(5).uniform(-0.05, 0.05, one_cell[1].positions.shape)
        with self.subTest(reason="reads the cell"), self.assertRaisesRegex(NegotiationError, "reads the batch's cell"):
            self.multi_md(UnionPairAdapter(module, read_cell=True), one_cell, self_check=True, slot_atoms=None)
        adapter = ReplicaMolecularDynamicsCudaTest.adapter()
        slab = systems[0].copy()
        slab.pbc = (True, True, False)
        flat = systems[2].copy()
        flat.set_cell(np.diag([5.0, 5.0, 0.0]))
        bad = systems[3].copy()
        bad.positions[0, 0] = np.nan
        for name, index, replaced, pattern in (("slab", 0, slab, "periodic"), ("singular cell", 2, flat, "cell"),
                                               ("non-finite", 3, bad, "non-finite")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, pattern):
                self.multi_md(adapter, systems[:index] + [replaced] + systems[index + 1:])
        with self.assertRaisesRegex(ValueError, "slot_atoms"):
            self.multi_md(adapter, systems, slot_atoms=[108, 60, 24, 40])
        thin = bulk("Cu", "fcc", a=3.6)  # one atom: its cell's heights (2.1 A) need images beyond narrow keys
        with deterministic_algorithms(), self.multi_md(adapter, systems[:2], slot_atoms=None) as narrow:
            self.assertEqual(narrow._unions[0].runner.builder.sort_key_dtype, torch.int32)
            before = [self.state(narrow, index) for index in range(2)]
            with self.assertRaisesRegex(ValueError, "thinner than the union's sort keys"):
                narrow.write_system(1, thin)
            self.assertEqual(narrow.stats()["atoms"], [[108, 64]])
            for index in range(2):
                self.assert_same_state(self.state(narrow, index), before[index], f"after the thin cell: {index}")
        with deterministic_algorithms(), self.multi_md(adapter, systems) as md:
            md.run(10)
            snapshot = md.snapshot()
            before = [self.state(md, index) for index in range(4)]
            too_large = bulk("Cu", "fcc", a=3.6).repeat((3, 3, 9))  # 81 atoms > 70
            for index, atoms, error, pattern in ((1, too_large, ValueError, "ceiling"),
                                                 (0, slab, ValueError, "periodic"), (2, flat, ValueError, "cell"),
                                                 (3, bad, ValueError, "non-finite"),
                                                 (4, systems[0], IndexError, "out of range")):
                with self.subTest(write=pattern), self.assertRaisesRegex(error, pattern):
                    md.write_system(index, atoms)
            self.assertEqual(md.stats()["atoms"], [[108, 64, 24, 36]])
            for index in range(4):
                self.assert_same_state(self.state(md, index), before[index], f"after the rejected writes: {index}")
            with self.assertRaisesRegex(ValueError, "momenta of system 1"):
                md.restore(snapshot, momenta=[atoms.get_momenta() for atoms in systems[:1]] + [np.zeros(3)] * 3)
            same_size = systems[1].copy()
            same_size.positions += 0.02
            md.write_system(1, same_size)
            with self.assertRaisesRegex(ValueError, "systems it holds now"):
                md.restore(snapshot)
            md.run(10)  # the union runs on
            self.assertTrue(md.health().all())



@unittest.skipUnless(
    torch.cuda.is_available() and op_available and _fixed_builder_available,
    "CUDA graph builders are unavailable",
)
class MultiSystemRelaxationCudaTest(unittest.TestCase):
    """MultiSystemRelaxation (structures relaxed by FIRE in the slots of a union, in whole-step CUDA graphs) with the
    test pair potential behind UnionPairAdapter at ten times its default strength: six structures of 4 to 32 atoms
    (fcc Cu and Al, diamond Si; cubic, primitive and orthorhombic cells), each moved by up to 0.15 A, windows of 5
    steps, deterministic scatters. Each structure is compared with ASE's FIRE on it alone (fmax 0.02 eV/A) with
    MLIPCalculator of the same adapter, with a fixed cell and with ASE's FrechetCellFilter."""

    FMAX, MAX_STEPS, WINDOW = 0.02, 300, 5

    @staticmethod
    def structures() -> list[Atoms]:
        structures = [bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2), bulk("Cu", "fcc", a=3.6).repeat(3),
                      bulk("Al", "fcc", a=4.05, orthorhombic=True).repeat((2, 2, 1)),
                      bulk("Si", "diamond", a=5.43).repeat((2, 1, 1)), bulk("Cu", "fcc", a=3.5).repeat(2),
                      bulk("Al", "fcc", a=4.0).repeat((3, 2, 2))]
        for seed, atoms in enumerate(structures):
            atoms.positions += np.random.default_rng(seed).uniform(-0.15, 0.15, atoms.positions.shape)
        return structures

    def relax(self, adapter, structures, **options) -> MultiSystemRelaxation:
        options = {"config": ReplicaMolecularDynamicsCudaTest.config(), "self_check": False,
                   "steps_per_launch": self.WINDOW, "fmax": self.FMAX, "max_steps": self.MAX_STEPS, **options}
        return MultiSystemRelaxation(adapter, structures, **options)

    def alone(self, adapter, cell_filter: bool) -> list[tuple]:
        """ASE's FIRE on each structure alone: its steps, positions, cell, energy, forces and stress (eV/A^3)."""
        out = []
        calculator = MLIPCalculator(model=adapter, task="efs" if cell_filter else "ef", device="cuda",
                                      config=InferenceConfig(execution="eager", report=False))
        for atoms in self.structures():
            atoms.calc = calculator
            optimizer = FIRE(FrechetCellFilter(atoms) if cell_filter else atoms, logfile=None)
            self.assertTrue(optimizer.run(fmax=self.FMAX, steps=self.MAX_STEPS))
            out.append((optimizer.nsteps, atoms.positions.copy(), atoms.cell.array.copy(), atoms.get_potential_energy(),
                        atoms.get_forces(), atoms.get_stress(voigt=False) if cell_filter else None))
        calculator.close()
        return out

    def test_structures_relax_as_ase_fire_alone(self):
        """A fixed cell and FrechetCellFilter, the six structures in six slots, in two (four structures written
        into freed slots, no recapture), drained in six or in two (a union captured again with only its relaxing
        systems once they hold half its rows; in two, after the queue empties), with a capture again that fails (the
        union runs on as it was), or in three unions of at most 40 rows: each structure's FIRE steps equal ASE's (13
        to 96), its positions and cell
        within 1e-5 A, its energy and stress (cell filter) within 1e-5 of their largest value and its forces within
        1e-5 eV/A, its status converged; the results come back in input order with a SinglePointCalculator of the
        relaxed structure."""
        adapter = ReplicaMolecularDynamicsCudaTest.adapter()
        with deterministic_algorithms():
            for cell_filter in (False, True):
                expected = self.alone(adapter, cell_filter)
                for slots, drain, union_rows in ((None, False, None), (2, False, None), (None, True, None),
                                                 (2, True, None), (None, "fails", None), (6, False, 40)):
                    with self.subTest(cell_filter=cell_filter, slots=slots, drain=drain, union_rows=union_rows):
                        structures = self.structures()
                        draining = (mock.patch.multiple(multi_system_relax, DRAIN_FRACTION=0.5, DRAIN_MIN_ROWS=0,
                                                        DRAIN_PAYBACK=0.0) if drain else nullcontext())
                        failing = (mock.patch.object(multi_system_relax._SlotRelaxationEngine, "compact",
                                                     side_effect=torch.OutOfMemoryError("simulated"))
                                   if drain == "fails" else nullcontext())
                        with draining, failing, self.relax(adapter, structures, slots=slots, union_rows=union_rows,
                                                           cell_filter="frechet" if cell_filter else None) as relax:
                            results = relax.run()
                            stats = relax.stats()
                        self.assertEqual(stats["structure_writes"], 0 if slots in (None, 6) else len(structures) - 2)
                        self.assertEqual(stats["recaptures"], stats["compactions"])
                        self.assertEqual(stats["compactions"] > 0, drain is True)
                        self.assertEqual(len(stats["drain_failures"]), 1 if drain == "fails" else 0)
                        self.assertEqual(len(stats["unions"]), 1 if union_rows is None else 3)
                        self.assertEqual(stats["finished"], {"converged": 6, "max_steps": 0, "failed": 0})
                        for index, (result, (steps, positions, cell, energy, forces, stress)) in enumerate(
                                zip(results, expected)):
                            what = f"structure {index}"
                            self.assertEqual(result.index, index)
                            self.assertEqual((result.steps, result.status), (steps, "converged"), what)
                            self.assertLess(result.fmax, self.FMAX)
                            np.testing.assert_allclose(result.atoms.positions, positions, rtol=0, atol=1e-5,
                                                       err_msg=what)
                            np.testing.assert_allclose(result.atoms.cell.array, cell, rtol=0, atol=1e-5, err_msg=what)
                            self.assertAlmostEqual(result.atoms.get_potential_energy(), energy,
                                                   delta=1e-5 * abs(energy))
                            np.testing.assert_allclose(result.atoms.get_forces(), forces, rtol=0, atol=1e-5,
                                                       err_msg=what)
                            if cell_filter:
                                np.testing.assert_allclose(result.stress, stress, rtol=0,
                                                           atol=1e-5 * np.abs(stress).max(), err_msg=what)
                                np.testing.assert_allclose(result.atoms.get_stress(voigt=False), stress, rtol=0,
                                                           atol=1e-5 * np.abs(stress).max(), err_msg=what)
                            else:
                                self.assertIsNone(result.stress)

    def test_rejections_and_the_step_limit(self):
        """Invalid arguments and structures (a slab, constraints, a queued structure with an isolated atom where the
        configuration rejects them) are rejected at construction; with max_steps 0 every structure stops at its start
        (status "max_steps", 0 steps) with the forces there, and a structure whose energy is not finite fails without
        stopping the others."""
        adapter = ReplicaMolecularDynamicsCudaTest.adapter()
        structures = self.structures()
        slab = structures[0].copy()
        slab.pbc = (True, True, False)
        for options, pattern in (({"cell_filter": "exp"}, "cell_filter"), ({"fmax": 0.0}, "fmax"),
                                 ({"fmax": float("nan")}, "fmax"), ({"max_steps": -1}, "max_steps"),
                                 ({"steps_per_launch": 0}, "steps_per_launch"), ({"slots": 0}, "slots"),
                                 ({"fire": {"dt": 0.1}}, "FIREParameters")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, pattern):
                self.relax(adapter, structures, **options)
        fixed = structures[0].copy()
        fixed.set_constraint(FixAtoms(indices=[0]))
        for bad, pattern in (([], "one or more"), ([slab] + structures[1:], "periodic"),
                             (structures[1:] + [fixed], "constraints")):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                self.relax(adapter, bad)
        lone = structures[0].copy()  # queued behind structure 0 in its slot class: an atom far from the others
        lone.set_cell(lone.cell.array * [1.0, 1.0, 10.0], scale_atoms=False)
        lone.positions[0, 2] = 0.5 * lone.cell.array[2, 2]
        config = dataclasses.replace(ReplicaMolecularDynamicsCudaTest.config(), isolated_atoms="error")
        with self.subTest(pattern="neighbour"), self.assertRaisesRegex(ValueError, r"structures \[2\] have an atom"):
            self.relax(adapter, structures[:2] + [lone], slots=2, config=config)
        helium = structures[3].copy()
        helium.numbers[1] = 2

        class HeliumFails(UnionPairAdapter):  # a non-finite energy on helium rows
            def node_energy(self, inputs, positions, cell, edge_shifts=None):
                energies = super().node_energy(inputs, positions, cell, edge_shifts=edge_shifts)
                return torch.where(inputs.species == 2, torch.full_like(energies, float("nan")), energies)

        with self.relax(HeliumFails(adapter.module), structures + [helium], max_steps=0, slots=3) as relax:
            results = relax.run()
        self.assertEqual(results[-1].status, "failed")
        calculator = MLIPCalculator(model=adapter, task="ef", device="cuda",
                                      config=InferenceConfig(execution="eager", report=False))
        for atoms, result in zip(structures, results):
            self.assertEqual((result.status, result.steps), ("max_steps", 0))
            np.testing.assert_allclose(result.atoms.positions, atoms.positions, rtol=0, atol=0)
            atoms.calc = calculator
            np.testing.assert_allclose(result.forces, atoms.get_forces(), rtol=0,
                                       atol=1e-5 * np.abs(atoms.get_forces()).max())
        calculator.close()

if __name__ == "__main__":
    unittest.main()
