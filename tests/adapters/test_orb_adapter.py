"""ORB v3 behind the adapter protocol, checked against orb-models itself (accuracy only).

:mod:`fastmd.models.orb` runs upstream's ``ConservativeForcefieldRegressor``
(orb-v3-conservative-inf-mpa) through the engine: the edges-only graph
builders (ragged, and fixed capacity padded with far self-loops), the
engine-owned derivative of the adapter's whole-system energy tap
(:class:`~fastmd.engine.energy_derivative.EnergyDerivative`),
GPU-resident MD in the eager, model_graph and whole_step modes, and the ASE
calculator. The reference is orb-models' own ASE calculator (``ORBCalculator``:
its own graph, featurisation and autograd), built under IEEE FP32 like every
evaluation here (under TF32 its own neighbour search is not the radius
graph). ORB runs FP32 only and its GPU scatters are atomics, so no path
repeats bit for bit. A reference is therefore the mean of K=5 repeated
evaluations, and an evaluation agrees with it when its forces are within
``max(1e-6, 1.5 nu)`` eV/A, ``nu`` the K=5 eager floor (the largest max|dF|
between the 5 repeats; where both sides are repeated, the larger of the two),
its energy within 1e-6 eV/atom and its stress within ``max(1e-8, 1.5 nu_S)``
eV/A^3 (``nu_S`` the floor of the stress). Against a single other evaluation
a comparison would carry the scatter of both, which exceeds 1.5 nu now and
then.

The structures are perturbed CsSnI3 perovskites in cells whose lattice
vectors are binary fractions (the shifts ``image @ cell`` are exact in FP32
for either side) and a cluster with an isolated atom, all built here. ORB is
evaluated at the coordinates the engine evaluates: the engine's builders wrap
the atoms into the cell in FP32, upstream wraps them in FP64, and the ~1e-6 A
between the two moves the forces by several times the floor. Once an adapter
is loaded, its aggregation patch is installed process-wide, but it takes its
sync-free path only inside the adapter's own calls, so the reference
calculator runs upstream's function; the patch test also compares against an
unpatched upstream in a fresh process.

Needs the optional ORB extra (orb-models), a CUDA device and the
orb-v3-conservative-inf-mpa checkpoint: the file ``MATRIS_ORB_CHECKPOINT``
names, else orb-models' download cache (cached-path) must hold it; the tests
never download. They are skipped otherwise, except the checks that the extra
is optional, which run everywhere.
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import torch
from ase import Atoms, units
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress

from fastmd.adapters import as_adapter
from fastmd.adapters import load_adapter
from fastmd.adapters.negotiate import NegotiationError
from fastmd.adapters.negotiate import can_capture
from fastmd.models.orb import ORB_CHECKPOINTS
from fastmd.adapters.protocol import EnergyTap
from fastmd.adapters.protocol import ModelOwnedDerivative
from fastmd.adapters.protocol import SystemEnergyTap
from fastmd.interfaces.calculator import MLIPCalculator
from fastmd.engine.energy_derivative import EnergyDerivative
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.config import CapacityConfig
from fastmd.config import ConfigurationWarning
from fastmd.config import InferenceConfig
from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.gpu_graph_builder import atoms_to_edges_gpu
from fastmd.graph.gpu_graph_builder import op_available
from fastmd.precision import fp32_gemm_mode
from fastmd.precision import matmul_precision_scope
from fastmd.precision import restore_fp32_state

ORB = importlib.util.find_spec("orb_models") is not None  # the optional extra (found, not imported)
CUDA = torch.cuda.is_available() and op_available and _fixed_builder_available
MODEL = "orb_v3"
REPEATS = 5  # evaluations of one input that give an FP32 noise floor (10 pairs)
EXECUTIONS = ("eager", "model_graph", "whole_step")
REPOSITORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Loaded once per process: the reference calculator, the eager adapter and the compiled one (its first
# call compiles the backbone).
_LOADED: dict = {}


def tearDownModule():
    for key, value in list(_LOADED.items()):
        if key[0] == "adapter":
            value.close()  # reverts the adapters' patches
    _LOADED.clear()
    gc.collect()  # the captured graphs and their pools
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def orb_checkpoint() -> str:
    """The orb-v3-conservative-inf-mpa file: ``MATRIS_ORB_CHECKPOINT``, else orb-models' download cache;
    SkipTest without either (the tests never download)."""
    path = os.environ.get("MATRIS_ORB_CHECKPOINT")
    if path:
        if not os.path.isfile(path):
            raise unittest.SkipTest(f"MATRIS_ORB_CHECKPOINT={path!r} is not a file")
        return path
    from cached_path import find_latest_cached  # orb-models' downloader; looks into its cache only

    cached = find_latest_cached(ORB_CHECKPOINTS[MODEL][1])
    if cached is None:
        raise unittest.SkipTest("the ORB checkpoint is not in orb-models' download cache: set "
                                "MATRIS_ORB_CHECKPOINT to orb-v3-conservative-inf-mpa-20250404.ckpt "
                                "(the tests never download)")
    return str(cached)


def orb_adapter(checkpoint: str, profile: str | None = "generic"):
    """ORB through ``load_adapter`` ("generic": the eager backbone, the reference; None: compiled)."""
    key = ("adapter", profile)
    if key not in _LOADED:
        _LOADED[key] = load_adapter(MODEL, model_path=checkpoint, device="cuda", optimization_profile=profile)
    return _LOADED[key]


def orb_calculator(checkpoint: str):
    """orb-models' own ASE calculator of the checkpoint, IEEE FP32, eager (the reference). Upstream's loader
    sets the process's FP32 GEMM precision and default dtype: both restored."""
    key = ("calculator",)
    if key not in _LOADED:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        default = torch.get_default_dtype()
        try:
            with restore_fp32_state(), warnings.catch_warnings():
                warnings.simplefilter("ignore")  # upstream's notes on the default dtype and on TF32
                model = pretrained.orb_v3_conservative_inf_mpa(weights_path=checkpoint, device="cuda",
                                                               precision="float32-highest", compile=False)
        finally:
            torch.set_default_dtype(default)
        _LOADED[key] = ORBCalculator(model, device="cuda")
    return _LOADED[key]


def perovskite(repeat: int = 2, *, sheared: bool = True, seed: int = 0) -> Atoms:
    """CsSnI3, ``repeat``^3 cubic perovskite cells (a = 6.25 A; 40 atoms for 2, 135 for 3), sheared to a
    triclinic cell with binary-fraction lattice vectors unless ``sheared`` is False, positions perturbed by
    0.1 A."""
    a = 6.25
    atoms = Atoms("CsSnI3", cell=[a, a, a], pbc=True,
                  scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    atoms = atoms.repeat(repeat)
    if sheared:
        atoms.set_cell(atoms.cell.array + [[0.0, 0.0, 0.0], [1.25, 0.0, 0.0], [0.75, 0.5, 0.0]], scale_atoms=True)
    atoms.positions += np.random.default_rng(seed).normal(0.0, 0.1, atoms.positions.shape)
    return atoms


def thermalized(atoms: Atoms) -> Atoms:
    """``atoms`` with Maxwell-Boltzmann momenta at 500 K (fixed seed)."""
    kinetic = np.sqrt(atoms.get_masses()[:, None] * units.kB * 500.0)
    atoms.set_momenta(np.random.default_rng(2).normal(0.0, 1.0, atoms.positions.shape) * kinetic)
    return atoms


def cluster_with_lone_atom() -> Atoms:
    """A perturbed Cs4I4 cube (edge 3.2 A) and one more Cs over 11 A from every other atom and image, in a
    20 A box (ORB's cutoff is 6 A)."""
    corners = np.array(list(itertools.product((0.0, 3.2), repeat=3)))
    cube = Atoms(numbers=[55 if sum(index) % 2 else 53 for index in itertools.product((0, 1), repeat=3)],
                 positions=4.0 + corners + np.random.default_rng(3).normal(0.0, 0.05, corners.shape))
    atoms = cube + Atoms("Cs", positions=[[14.0, 14.0, 14.0]])
    atoms.set_cell([20.0] * 3)
    atoms.pbc = True
    return atoms


@contextlib.contextmanager
def ieee_fp32():
    """IEEE FP32 GEMMs (no TF32) inside; the caller's precision flags afterwards."""
    with matmul_precision_scope("highest"):
        cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cudnn.allow_tf32 = cudnn


def small_capacity(**changes) -> CapacityConfig:
    """Graph capacities for these small systems, given explicitly: tiers of 32 undirected pairs with at least
    12 of padding (the step the adapter declares) and one warm-up."""
    return CapacityConfig(**{"u_step": 32, "min_pad_u": 12, "warmup": 1, **changes})


def ragged(atoms: Atoms, adapter):
    """The engine's ragged edges-only graph batch of ``atoms`` (no padding)."""
    return atoms_to_edges_gpu(atoms, adapter.topology_spec().cutoff, "cuda", check_isolated_atoms=False)


def at_engine_positions(atoms: Atoms, adapter) -> Atoms:
    """``atoms`` at the coordinates the engine evaluates: its builders' FP32 wrap into the cell."""
    probe = atoms.copy()
    probe.positions = ragged(atoms, adapter).positions.double().cpu().numpy()
    return probe


def fixed_capacity_builder(atoms: Atoms, adapter, u_capacity: int) -> FixedCapacityGraphBuilder:
    """A fixed-capacity edges-only builder of ``atoms`` (``2 u_capacity`` edge rows) padded by the adapter's
    rule, its real rows sorted like the ragged batch's (the comparison then isolates the padded rows)."""
    return FixedCapacityGraphBuilder(
        cell=torch.tensor(atoms.cell.array, dtype=torch.float32), atomic_numbers=torch.tensor(atoms.numbers),
        u_capacity=u_capacity, t_capacity=0, atom_graph_cutoff=adapter.topology_spec().cutoff,
        line_graph_cutoff=None, padding=adapter.padding_rule(), prepare=adapter.prepare, device="cuda")


def engine_efs(adapter, batch, inputs=None) -> tuple[float, np.ndarray, np.ndarray]:
    """The engine-owned derivative of ``batch``: energy, forces of the real atoms and stress (eV/A^3),
    copied in float64."""
    out = EnergyDerivative(adapter)(batch, adapter.prepare(batch) if inputs is None else inputs, task="efs")
    return (float(out["e"][0]), out["f"][0][: batch.n_real].double().cpu().numpy(),
            out["s"][0].double().cpu().numpy())


def orb_efs(calculator, atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
    """A fresh evaluation of orb-models' calculator: energy, forces and stress (3x3, eV/A^3)."""
    calculator.calculate(atoms, ["energy", "forces", "stress"])
    results = calculator.results
    return (float(results["energy"]), np.asarray(results["forces"], dtype=np.float64),
            voigt_6_to_full_3x3_stress(np.asarray(results["stress"], dtype=np.float64)))


def calculator_efs(adapter, atoms: Atoms, execution: str = "eager") -> tuple[float, np.ndarray, np.ndarray]:
    """Energy, forces and stress (3x3, eV/A^3) of the calculator over the adapter."""
    probe = atoms.copy()
    probe.calc = MLIPCalculator(model=adapter, device="cuda", config=InferenceConfig(
        execution=execution, report=False, optimization_profile=profile_of(adapter), capacity=small_capacity()))
    return probe.get_potential_energy(), probe.get_forces(), probe.get_stress(voigt=False)


def profile_of(adapter) -> str | None:
    """The entry points' profile of an adapter: "generic" keeps the eager backbone (the default profile would
    install the compiled one in an adapter that is not in use yet), None the compiled one."""
    return None if adapter.compiled else "generic"


def gpu_md(atoms: Atoms, adapter, execution: str, *, capacity: CapacityConfig | None = None,
           matmul_precision: str | None = None, **kwargs):
    """GPU-resident MD of ``atoms``: a Nose-Hoover chain at 500 K, 1 fs steps."""
    config = InferenceConfig(execution=execution, report=False, optimization_profile=profile_of(adapter),
                             capacity=capacity or small_capacity(), matmul_precision=matmul_precision)
    return GPUResidentMolecularDynamics(atoms.copy(), model=adapter, ensemble="nvt_nhc", temperature=500.0,
                                        timestep=1.0, tdamp=25.0, config=config, **kwargs)


def largest(values) -> float:
    """The largest of ``values`` (floats), NaN if any is NaN (the built-in max drops a NaN after the first)."""
    return float(np.max(np.asarray(list(values), dtype=float)))


def max_pairwise(values) -> float:
    """The largest max|a - b| over the pairs of ``values`` (arrays); NaN if any difference is NaN."""
    pairs = [float(np.abs(np.asarray(a) - np.asarray(b)).max()) for index, a in enumerate(values)
             for b in values[:index]]
    return largest(pairs) if pairs else 0.0


def max_difference(actual, expected) -> float:
    return float(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)).max())


def mean_efs(runs) -> tuple[float, np.ndarray, np.ndarray]:
    """The mean (float64) of repeated evaluations ``(energy, forces, stress)``: a reference."""
    energy, forces, stress = (np.mean([np.asarray(run[index], dtype=float) for run in runs], axis=0)
                              for index in range(3))
    return float(energy), forces, stress


def floors(*repeats) -> tuple[float, float]:
    """The K=5 floors (forces, stress) of one or more sets of repeated evaluations: the largest."""
    return (largest([max_pairwise([run[1] for run in runs]) for runs in repeats]),
            largest([max_pairwise([run[2] for run in runs]) for runs in repeats]))


def patch_lifecycle(checkpoint: str) -> dict:
    """What ``ORBPatchTest`` checks, run in a fresh process (JSON-ready).

    A process-wide patch is shared by the adapters of a process (counted by record name; an installed record
    is not checked again), so declining one needs a process in which no adapter has applied it, and seeing
    it reverted one in which no other adapter holds it. First an adapter whose two synchronization-removing
    records decline (their source hashes do not match): its outcomes, contract and plan, its evaluations
    and orb-models' calculator's (unpatched) at the engine's positions, and GPU-resident MD asked for
    whole_step. Then two adapters that apply the records, closed one after the other.
    """
    from orb_models.forcefield import segment_ops

    import fastmd.models.orb as orb_module

    original = segment_ops.aggregate_nodes
    zbl_record = orb_module._zbl_power_record
    mismatched = "0" * 64
    report: dict = {}
    with ieee_fp32():
        with mock.patch.object(orb_module, "AGGREGATE_NODES",
                               dataclasses.replace(orb_module.AGGREGATE_NODES, source_sha256=mismatched)), \
                mock.patch.object(orb_module, "_zbl_power_record",
                                  lambda zbl: dataclasses.replace(zbl_record(zbl), source_sha256=mismatched)):
            adapter = load_adapter(MODEL, model_path=checkpoint, device="cuda", optimization_profile="generic")
        report["outcomes"] = {outcome.record.name: {"applied": outcome.applied, "reason": outcome.reason,
                                                    "pin": outcome.record.pin} for outcome in adapter.patches}
        report["sync_free"] = adapter.capture_contract().sync_free
        report["can_capture"] = can_capture(adapter)
        report["upstream_untouched"] = segment_ops.aggregate_nodes is original
        atoms = perovskite(3, sheared=False)
        reference = orb_calculator(checkpoint)
        probe = at_engine_positions(atoms, adapter)
        report["orb"] = [orb_efs(reference, probe) for _ in range(REPEATS)]
        batch = ragged(atoms, adapter)
        report["engine"] = [engine_efs(adapter, batch) for _ in range(REPEATS)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md = gpu_md(atoms, adapter, "whole_step")
        with md:
            plan = md.plan_record()
            report["execution"] = md.config_summary()["effective"]["execution"]
            report["execution_reason"] = plan.decision("execution").reason
            report["plan_declined"] = sorted(plan.patches["declined"])
            forces, energy = md.evaluate(torch.tensor(atoms.positions, device="cuda"))
            report["gpu_md"] = (float(energy), forces.double().cpu().numpy(), None)
            md.run(5)  # eager steps through upstream's own (synchronizing) code
            report["gpu_md_steps"] = md.nsteps
        report["warnings"] = [[warning.category.__name__, str(warning.message)] for warning in caught]
        zbl = adapter.module.pair_repulsion_fn
        report["declined_zbl_untouched"] = "_polynomial_cutoff_with_derivative" not in vars(zbl)
        adapter.close()
        report["declined_closed_untouched"] = segment_ops.aggregate_nodes is original

        first = load_adapter(MODEL, model_path=checkpoint, device="cuda", optimization_profile="generic")
        zbl = first.module.pair_repulsion_fn
        report["applied"] = {"aggregate": segment_ops.aggregate_nodes is orb_module._aggregate_nodes,
                             "zbl": "_polynomial_cutoff_with_derivative" in vars(zbl),
                             "sync_free": first.capture_contract().sync_free}
        second = load_adapter(MODEL, model_path=checkpoint, device="cuda", optimization_profile="generic")
        first.close()
        report["first_closed"] = {"aggregate": segment_ops.aggregate_nodes is orb_module._aggregate_nodes,
                                  "zbl_reverted": "_polynomial_cutoff_with_derivative" not in vars(zbl)}
        second.close()
        report["both_closed_original"] = segment_ops.aggregate_nodes is original

    def jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        return value

    return jsonable(report)


@unittest.skipUnless(ORB, "the optional ORB extra (orb-models) is not installed")
@unittest.skipUnless(CUDA, "CUDA and the GPU graph builders are unavailable")
class ORBTestCase(unittest.TestCase):
    """The checks shared by the ORB tests, under IEEE FP32 GEMMs; a class is skipped without the checkpoint."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = orb_checkpoint()
        cls._precision = contextlib.ExitStack()
        cls._precision.enter_context(ieee_fp32())

    @classmethod
    def tearDownClass(cls):
        cls._precision.close()

    def orb(self, profile: str | None = "generic"):
        return orb_adapter(self.checkpoint, profile)

    def upstream_calculator(self):
        return orb_calculator(self.checkpoint)

    def assert_within(self, error: float, bound: float, what: str) -> None:
        self.assertTrue(math.isfinite(error), f"{what}: {error}")
        self.assertLessEqual(error, bound, f"{what}: {error:.3e} > {bound:.3e}")

    def assert_efs(self, actual, expected, floor_f: float | None, floor_s: float | None, what: str,
                   multiplier: float = 1.5) -> None:
        """Against a reference ``expected``: |dE|/N within 1e-6 eV/atom, max|dF| within max(1e-6,
        ``multiplier`` floor_f) eV/A and max|dS| within max(1e-8, ``multiplier`` floor_s) eV/A^3; a None
        entry of ``actual`` (or a None floor) is not compared."""
        energy, forces, stress = actual
        expected_energy, expected_forces, expected_stress = expected
        if energy is not None:
            atoms = len(expected_forces) if expected_forces is not None else len(forces)
            self.assert_within(abs(energy - expected_energy) / atoms, 1e-6, f"{what}: |dE|/N (eV)")
        if forces is not None and floor_f is not None:
            self.assert_within(max_difference(forces, expected_forces), largest([1e-6, multiplier * floor_f]),
                               f"{what}: max|dF| (eV/A, floor {floor_f:.2e})")
        if stress is not None and floor_s is not None:
            self.assert_within(max_difference(stress, expected_stress), largest([1e-8, multiplier * floor_s]),
                               f"{what}: max|dS| (eV/A^3, floor {floor_s:.2e})")


class ORBAccuracyTest(ORBTestCase):
    def test_matches_the_orb_calculator(self):
        """135 atoms (cubic) and 40 atoms (triclinic) against orb-models' calculator at the engine's
        positions (the reference: the mean of its K=5 repeats; ``nu`` and ``nu_S`` the larger floor of its
        repeats and the engine-owned derivative's): the engine-owned derivative of a fresh ragged build,
        GPU-resident MD's eager evaluate and its logged first frame (energy, forces, stress), and the
        calculator (eager)."""
        adapter = self.orb()
        reference = self.upstream_calculator()
        self.assertEqual(fp32_gemm_mode(), "ieee")
        for name, atoms in (("135 atoms", perovskite(3, sheared=False)), ("40 atoms", perovskite())):
            with self.subTest(name):
                probe = at_engine_positions(atoms, adapter)
                upstream = [orb_efs(reference, probe) for _ in range(REPEATS)]
                self.assertGreater(np.abs(upstream[0][1]).max(), 0.5)  # the perturbed crystal has forces
                batch = ragged(atoms, adapter)
                engine = [engine_efs(adapter, batch) for _ in range(REPEATS)]
                floor_f, floor_s = floors(upstream, engine)
                results = {"engine-owned derivative": engine[0]}
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "md.traj")
                    with gpu_md(atoms, adapter, "eager", trajectory=path, loginterval=1, log_stress=True) as md:
                        forces, energy = md.evaluate(torch.tensor(atoms.positions, device="cuda"))
                        results["GPU MD evaluate"] = (float(energy), forces.double().cpu().numpy(), None)
                        md.run(1)
                    frame = read(path, index=0)
                results["GPU MD frame"] = (frame.get_potential_energy(), frame.get_forces(),
                                           frame.get_stress(voigt=False))
                results["calculator"] = calculator_efs(adapter, atoms)
                expected = mean_efs(upstream)
                for path_name, actual in results.items():
                    self.assert_efs(actual, expected, floor_f, floor_s, f"{name}, {path_name}")

    def test_padding_adds_exactly_nothing(self):
        """The adapter's padding (far self-loops) on 40 atoms: a fixed-capacity build with 128 padded rows,
        fresh and on its reused candidate list, equals the ragged batch (the mean of its K=5 repeats; energy
        within 2 ulp or 1.5 times its K=5 floor, forces within max(1e-6, 1.5 nu), stress within max(1e-8,
        1.5 nu_S)); a batch of padded rows only (two atoms farther apart than the cutoff) has the energy of
        the batch without edges bit for bit, and forces and stress exactly zero."""
        adapter = self.orb()
        self.assertEqual(adapter.padding_rule().kind, "far_self_loops")
        atoms = perovskite()
        batch = ragged(atoms, adapter)
        repeats = [engine_efs(adapter, batch) for _ in range(REPEATS)]
        expected = mean_efs(repeats)
        floor_e = max_pairwise([np.float64(run[0]) for run in repeats])
        floor_f, floor_s = floors(repeats)
        energy_bound = largest([2.0 * float(np.spacing(np.float32(abs(expected[0])))), 1.5 * floor_e])
        builder = fixed_capacity_builder(atoms, adapter, u_capacity=batch.edge_index.shape[0] // 2 + 64)
        positions = torch.tensor(atoms.positions, device="cuda")
        for reuse in (False, True):
            with self.subTest(candidates="reused" if reuse else "fresh"):
                inputs, status = builder.build(positions, rebuild_candidates=not reuse)
                edges = int(status.edge_count)
                self.assertEqual(edges, batch.edge_index.shape[0])
                self.assertEqual(builder.batch.edge_index.shape[0] - edges, 128)  # padded rows
                self.assertTrue(torch.equal(builder.batch.edge_index[:edges], batch.edge_index))
                energy, forces, stress = engine_efs(adapter, builder.batch, inputs)
                self.assert_within(abs(energy - expected[0]), energy_bound, f"padded: |dE| (eV, floor {floor_e:.2e})")
                self.assert_efs((None, forces, stress), expected, floor_f, floor_s, "padded")
        apart = Atoms("CsI", positions=[[2.0, 2.0, 2.0], [12.0, 12.0, 12.0]], cell=[20.0] * 3, pbc=True)
        empty = ragged(apart, adapter)
        self.assertEqual(empty.edge_index.shape[0], 0)
        with torch.no_grad():
            alone = adapter.system_energy(adapter.prepare(empty), empty.positions, empty.cell)
        builder = fixed_capacity_builder(apart, adapter, u_capacity=8)
        inputs, status = builder.build(torch.tensor(apart.positions, device="cuda"))
        padded = builder.batch
        self.assertEqual(int(status.edge_count), 0)
        self.assertEqual(padded.edge_index.shape[0], 16)
        with torch.no_grad():
            self.assertTrue(torch.equal(adapter.system_energy(inputs, padded.positions, padded.cell), alone))
        energy, forces, stress = engine_efs(adapter, padded, inputs)
        self.assertTrue(math.isfinite(energy))
        self.assertTrue(bool((forces == 0).all()) and bool((stress == 0).all()))

    def test_isolated_atom_is_evaluated_by_the_model(self):
        """Isolated-atom policy "model": an atom farther than the cutoff from every other enters ORB's
        pooled energy like any other atom. The engine-owned derivative and GPU-resident MD (eager, and
        whole_step's padded build) equal orb-models' calculator at the engine's positions (the mean of its
        K=5 repeats; energy within 1e-6 eV/atom, forces within max(1e-6, 1.5 nu)), with exactly no force on
        that atom, and the atom adds to the calculator's energy an amount over 1 eV away from its element's
        linear reference (no reference-energy substitution)."""
        adapter = self.orb()
        self.assertEqual(adapter.isolated_atom_policy(), "model")
        atoms = cluster_with_lone_atom()
        lone = len(atoms) - 1
        reference = self.upstream_calculator()
        probe = at_engine_positions(atoms, adapter)
        upstream = [orb_efs(reference, probe) for _ in range(REPEATS)]
        self.assertTrue(np.all(upstream[0][1][lone] == 0.0))
        batch = ragged(atoms, adapter)
        engine = [engine_efs(adapter, batch) for _ in range(REPEATS)]
        floor_f, _ = floors(upstream, engine)
        expected = mean_efs(upstream)
        results = {"engine-owned derivative": engine[0]}
        for execution in ("eager", "whole_step"):
            with gpu_md(atoms, adapter, execution) as md:
                self.assertEqual(md.plan_record().isolated_atoms["policy"], "model")
                if execution == "eager":
                    forces, energy = md.evaluate(torch.tensor(atoms.positions, device="cuda"))
                else:  # the whole-step runner's first evaluation: the padded fixed-capacity build
                    forces, energy = md.state.forces, md.state.potential_energy
                results[f"GPU MD {execution}"] = (float(energy.reshape(-1)[0]), forces.double().cpu().numpy(), None)
        for name, actual in results.items():
            with self.subTest(name):
                self.assert_efs(actual, expected, floor_f, None, name)
                self.assertTrue(np.all(actual[1][lone] == 0.0))
        added = expected[0] - orb_efs(reference, probe[:lone])[0]
        linear_reference = float(adapter.reference_energies()[0, atoms.numbers[lone] - 1])
        self.assertGreater(abs(added - linear_reference), 1.0)

    def test_tf32_is_a_reduced_accuracy_opt_in(self):
        """GPU-resident MD's plan: IEEE FP32 by default (precision "ieee", the plan's accuracy class
        "fp32_noise"); matmul_precision="high" runs TF32 ("tf32", accuracy class "reduced"), which moves the
        forces by more than 1e-5 eV/A (ten times the IEEE floor); the process's FP32 GEMM flags stay."""
        adapter = self.orb()
        atoms = perovskite()
        positions = torch.tensor(atoms.positions, device="cuda")
        forces = {}
        for requested, chosen, accuracy in ((None, "ieee", "fp32_noise"), ("high", "tf32", "reduced")):
            with self.subTest(matmul_precision=requested), gpu_md(atoms, adapter, "eager",
                                                                  matmul_precision=requested) as md:
                plan = md.plan_record()
                self.assertEqual(plan.decision("precision").chosen, chosen)
                self.assertEqual(plan.accuracy_class, accuracy)
                if requested is not None:
                    self.assertEqual(plan.decision("precision").accuracy_class, "reduced")
                forces[chosen] = md.evaluate(positions)[0].double().cpu().numpy()
            self.assertEqual(fp32_gemm_mode(), "ieee")
        self.assertGreater(max_difference(forces["tf32"], forces["ieee"]), 1e-5)


class ORBCompiledTest(ORBTestCase):
    def test_compiled_backbone_matches_eager(self):
        """The default kernels (the backbone under upstream's torch.compile switch; the first call compiles
        it) against the eager backbone (optimization_profile="generic", the reference: the mean of its K=5
        repeats) on 135 atoms: the engine-owned derivative (energy within 1e-6 eV/atom, forces within
        max(1e-6, 3 nu) and stress within max(1e-8, 3 nu_S), ``nu`` the larger K=5 floor of the two: another
        arithmetic, as for MACE's kernels), and the whole-step runner's first evaluation with the compiled
        backbone captured."""
        compiled, eager = self.orb(None), self.orb()
        self.assertEqual((compiled.kernel, eager.kernel), ("inductor", "eager"))
        self.assertEqual(compiled.capture_contract().compiled, "capture_safe")
        atoms = thermalized(perovskite(3, sheared=False))
        batch = ragged(atoms, eager)
        ours = [engine_efs(compiled, batch) for _ in range(REPEATS)]
        theirs = [engine_efs(eager, batch) for _ in range(REPEATS)]
        floor_f, floor_s = floors(ours, theirs)
        expected = mean_efs(theirs)
        self.assert_efs(ours[0], expected, floor_f, floor_s, "compiled", multiplier=3.0)
        with gpu_md(atoms, compiled, "whole_step") as md:
            self.assertEqual(md.config_summary()["effective"]["execution"], "whole_step")
            first = (float(md.state.potential_energy.reshape(-1)[0]), md.state.forces.double().cpu().numpy(), None)
            md.run(10)
            self.assertEqual(md.whole_step_stats(capacity_plan=False)["launches"], 10)
        self.assert_efs(first, expected, floor_f, None, "compiled, whole_step", multiplier=3.0)


class ORBDynamicsTest(ORBTestCase):
    def test_execution_modes_agree(self):
        """GPU-resident MD (nvt_nhc, 1 fs) of 40 atoms in eager, model_graph and whole_step, and the
        calculator in model_graph, against the engine-owned derivative of the ragged batch eager MD evaluates
        (the mean of its K=5 repeats, ``nu`` and ``nu_S`` their floors): the first evaluation (eager: the
        ragged batch; model_graph: a captured padded bucket, the calculator's on its skin topology;
        whole_step: the padded fixed-capacity build) and each mode's logged first frame with its stress, as
        in the module docstring. Over 20 steps, every logged frame's energy agrees with eager's within 1e-6
        eV/atom and the positions within (20 fs)^2 F_tol / m_min (F_tol = max(1e-6, 1.5 nu)): twice the
        displacement of the lightest atom under the force tolerance held over the 20 steps."""
        adapter = self.orb()
        atoms = thermalized(perovskite(seed=1))
        positions = torch.tensor(atoms.positions, device="cuda")
        steps = 20
        repeats = [engine_efs(adapter, ragged(atoms, adapter)) for _ in range(REPEATS)]
        expected = mean_efs(repeats)
        floor_f, floor_s = floors(repeats)
        first, frames, final = {}, {}, {}
        with tempfile.TemporaryDirectory() as directory:
            for execution in EXECUTIONS:
                path = os.path.join(directory, f"{execution}.traj")
                with gpu_md(atoms, adapter, execution, trajectory=path, loginterval=10, log_stress=True) as md:
                    self.assertEqual(md.config_summary()["effective"]["execution"], execution)
                    if execution == "whole_step":
                        forces, energy = md.state.forces, md.state.potential_energy
                    else:  # model_graph returns its graph's output buffers: copied before the next replay
                        forces, energy = md.evaluate(positions)
                    first[f"{execution}, first evaluation"] = (float(energy.reshape(-1)[0]),
                                                               forces.double().cpu().numpy(), None)
                    md.run(steps)
                    final[execution] = md.snapshot_atoms().positions
                    if execution == "model_graph":
                        self.assertGreater(md.model_graph_stats()["captures"], 0)
                    elif execution == "whole_step":
                        self.assertEqual(md.whole_step_stats(capacity_plan=False)["launches"], steps)
                frames[execution] = read(path, ":")
                frame = frames[execution][0]
                first[f"{execution}, logged frame 0"] = (frame.get_potential_energy(), frame.get_forces(),
                                                         frame.get_stress(voigt=False))
        first["calculator model_graph, first evaluation"] = calculator_efs(adapter, atoms, "model_graph")
        for name, actual in first.items():
            self.assert_efs(actual, expected, floor_f, floor_s, name)
        drift = (steps * units.fs) ** 2 * largest([1e-6, 1.5 * floor_f]) / atoms.get_masses().min()
        for execution in EXECUTIONS[1:]:
            with self.subTest(execution):
                self.assert_within(max_difference(final[execution], final["eager"]), drift,
                                   f"{execution}: max|dx| after {steps} steps (A)")
                self.assertEqual(len(frames[execution]), len(frames["eager"]))
                self.assertEqual(len(frames["eager"]), steps // 10 + 1)
                for index, (ours, theirs) in enumerate(zip(frames[execution], frames["eager"])):
                    self.assert_within(max_difference(ours.positions, theirs.positions), drift,
                                       f"{execution} frame {index}: max|dx| (A)")
                    self.assert_within(abs(ours.get_potential_energy() - theirs.get_potential_energy()) / len(atoms),
                                       1e-6, f"{execution} frame {index}: |dE|/N (eV)")

    def test_skin_reuse_equals_rebuild(self):
        """Whole-step MD of 40 atoms (40 steps) on one candidate list (the default 2 A skin) equals a 0.02 A
        skin (a fresh candidate list after every step) and eager MD (a graph rebuilt every step): positions
        within (40 fs)^2 F_tol / m_min (F_tol = max(1e-6, 1.5 nu), as in test_execution_modes_agree)."""
        adapter = self.orb()
        atoms = thermalized(perovskite(seed=1))
        steps = 40
        floor_f, _ = floors([engine_efs(adapter, ragged(atoms, adapter)) for _ in range(REPEATS)])
        drift = (steps * units.fs) ** 2 * largest([1e-6, 1.5 * floor_f]) / atoms.get_masses().min()
        positions, stats = {}, {}
        for execution, skin in (("whole_step", 2.0), ("whole_step", 0.02), ("eager", 2.0)):
            with gpu_md(atoms, adapter, execution, capacity=small_capacity(candidate_skin=skin)) as md:
                self.assertEqual(md.config_summary()["effective"]["execution"], execution)
                md.run(steps)
                positions[execution, skin] = md.snapshot_atoms().positions
                stats[execution, skin] = md.whole_step_stats(capacity_plan=False)
        self.assertLess(stats["whole_step", 2.0]["candidate_refreshes"], 5)
        self.assertGreaterEqual(stats["whole_step", 0.02]["candidate_refreshes"], steps // 2)
        for key in (("whole_step", 0.02), ("eager", 2.0)):
            with self.subTest(execution=key[0], candidate_skin=key[1]):
                self.assert_within(max_difference(positions[key], positions["whole_step", 2.0]), drift,
                                   f"{key[0]}, skin {key[1]} A: max|dx| after {steps} steps (A)")


class ORBPatchTest(ORBTestCase):
    """The patch records' lifecycle, in a fresh process (see :func:`patch_lifecycle`)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        code = ("import importlib.util, json, sys\n"
                "spec = importlib.util.spec_from_file_location('orb_patch_lifecycle', sys.argv[1])\n"
                "module = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(module)\n"
                "print(json.dumps(module.patch_lifecycle(sys.argv[2])))\n")
        try:
            result = subprocess.run([sys.executable, "-B", "-c", code, os.path.abspath(__file__), cls.checkpoint],
                                    capture_output=True, text=True, timeout=900, cwd=REPOSITORY)
        except BaseException:
            cls.tearDownClass()
            raise
        cls.failure = result.stderr[-4000:] if result.returncode != 0 else None
        cls.lifecycle = None if cls.failure else json.loads(result.stdout.strip().splitlines()[-1])

    def lifecycle_report(self) -> dict:
        self.assertIsNone(self.failure, self.failure)
        return self.lifecycle

    def test_declined_patches_keep_the_adapter_correct(self):
        """Both synchronization-removing records declined (source hash mismatch): each outcome says why, the
        pins still apply, upstream stays as it is, the contract is not sync-free and nothing is captured:
        GPU-resident MD asked for whole_step runs eager with a ConfigurationWarning. Its evaluation and the
        engine-owned derivative equal orb-models' (unpatched) calculator at the engine's positions, as in
        the module docstring, and the adapter's close leaves upstream as it is."""
        report = self.lifecycle_report()
        declined = {name: outcome["reason"] for name, outcome in report["outcomes"].items() if not outcome["applied"]}
        self.assertEqual(sorted(declined), ["orb.aggregate_nodes_sizes_on_host", "orb.zbl_cutoff_power_on_host"])
        for name, reason in declined.items():
            self.assertIn("changed", reason, name)
        self.assertTrue(all(outcome["applied"] for outcome in report["outcomes"].values() if outcome["pin"]))
        self.assertTrue(report["upstream_untouched"] and report["declined_zbl_untouched"])
        self.assertFalse(report["sync_free"])
        self.assertFalse(report["can_capture"])
        self.assertEqual(report["plan_declined"], sorted(declined))
        self.assertEqual(report["execution"], "eager")
        self.assertIn("sync-free", report["execution_reason"])
        self.assertTrue(any(category == ConfigurationWarning.__name__ and "sync-free" in message
                            for category, message in report["warnings"]), report["warnings"])
        self.assertEqual(report["gpu_md_steps"], 5)
        upstream, engine = report["orb"], report["engine"]
        floor_f, floor_s = floors(upstream, engine)
        expected = mean_efs(upstream)
        for name, actual in (("engine-owned derivative", engine[0]), ("GPU MD evaluate", report["gpu_md"])):
            self.assert_efs(actual, expected, floor_f, floor_s, f"declined patches, {name}")
        self.assertTrue(report["declined_closed_untouched"])

    def test_close_reverts_the_patches(self):
        """Applied records: the aggregation replacement installed process-wide and the ZBL one on the
        adapter's own module; two adapters share the process-wide one, closing the first removes only its
        own ZBL patch, closing the second puts upstream's aggregate_nodes back."""
        report = self.lifecycle_report()
        self.assertEqual(report["applied"], {"aggregate": True, "zbl": True, "sync_free": True})
        self.assertEqual(report["first_closed"], {"aggregate": True, "zbl_reverted": True})
        self.assertTrue(report["both_closed_original"])


class ORBLoadingTest(ORBTestCase):
    def test_load_records_the_checkpoint(self):
        """By path: the file's sha256 and the model name (the pinned file) in ``checkpoint``, an adapter with
        a whole-system energy tap returned as is by the resolvers. By name: orb-models' downloader (stubbed
        with the same file) and the pinned hash; the same weights; a hash that is not the pinned one is
        rejected. Loading leaves the process's FP32 GEMM precision and default dtype as they were."""
        adapter = self.orb()
        upstream, url, pinned = ORB_CHECKPOINTS[MODEL]
        digest = hashlib.sha256()
        with open(self.checkpoint, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        digest = digest.hexdigest()
        self.assertEqual(adapter.checkpoint, {"name": MODEL if digest == pinned else os.path.basename(self.checkpoint),
                                              "path": self.checkpoint, "sha256": digest})
        self.assertIsInstance(adapter, SystemEnergyTap)
        self.assertNotIsInstance(adapter, EnergyTap)
        self.assertNotIsInstance(adapter, ModelOwnedDerivative)
        self.assertIs(as_adapter(adapter), adapter)
        self.assertIs(load_adapter(adapter), adapter)
        if digest != pinned:
            return  # another file than the pinned one: loading it by name does not apply
        default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)  # upstream's loader sets float32
            with matmul_precision_scope("high"), mock.patch("cached_path.cached_path",
                                                            return_value=self.checkpoint) as fetch:
                named = load_adapter(MODEL, device="cuda", optimization_profile="generic")
                self.assertEqual(fp32_gemm_mode(), "tf32")  # upstream's loader asks for "float32-highest"
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        finally:
            torch.set_default_dtype(default)
        try:
            fetch.assert_called_once_with(url)
            self.assertEqual(named.checkpoint, {"name": MODEL, "path": self.checkpoint, "sha256": pinned})
            ours, theirs = named.module.state_dict(), adapter.module.state_dict()
            self.assertEqual(ours.keys(), theirs.keys())
            self.assertTrue(all(torch.equal(ours[key], theirs[key]) for key in ours))
        finally:
            named.close()
        with mock.patch.dict(ORB_CHECKPOINTS, {MODEL: (upstream, url, "0" * 64)}), \
                mock.patch("cached_path.cached_path", return_value=self.checkpoint), \
                self.assertRaisesRegex(ValueError, "sha256"):
            load_adapter(MODEL, device="cuda", optimization_profile="generic")

    def test_declarations(self):
        """What the adapter declares: edges within 6 A for five layers, no triplets, float32, edges grouped by
        centre, ORB's 120-neighbour cap; far self-loop padding; a sync-free contract with an exact cutoff
        held to upstream within the FP32 noise, the engine-owned derivative and the stress in eV/A^3; the
        backbone's kernel choice (eager here); every patch record applied; and the reference-energy row:
        [1, 118] float64, column Z - 1 upstream's linear reference weight of Z, NaN for Z = 118."""
        from importlib.metadata import version

        adapter = self.orb()
        spec = adapter.topology_spec()
        self.assertEqual((spec.cutoff, spec.num_interactions, spec.triplet_cutoff, spec.undirected_pairs,
                          spec.model_dtype, spec.edge_order, spec.max_neighbors),
                         (6.0, 5, None, False, "float32", "by_centre", 120))
        self.assertEqual(adapter.padding_rule().kind, "far_self_loops")
        self.assertEqual(adapter.upstream, f"orb-models {version('orb-models')}")
        contract = adapter.capture_contract()
        self.assertEqual((contract.sync_free, contract.exact_cutoff, contract.accuracy_class, contract.force_mode,
                          contract.stress_unit, contract.compiled),
                         (True, True, "fp32_noise", "engine_autograd", "eV/A^3", "none"))
        self.assertTrue(can_capture(adapter))
        self.assertEqual((adapter.kernel, adapter.compiled), ("eager", False))
        (choice,) = adapter.kernel_backends()
        self.assertEqual((choice.site, choice.options, choice.reference), ("backbone", ("inductor", "eager"), "eager"))
        self.assertTrue(all(outcome.applied for outcome in adapter.upstream_patches()))
        self.assertLessEqual({"orb.aggregate_nodes_sizes_on_host", "orb.zbl_cutoff_power_on_host"},
                             {outcome.record.name for outcome in adapter.upstream_patches()})
        row = adapter.reference_energies()
        weight = self.upstream_calculator().model.heads["energy"].reference.linear.weight.detach().reshape(-1)
        self.assertEqual((tuple(row.shape), row.dtype), ((1, 118), torch.float64))
        self.assertTrue(torch.equal(row[0, :117], weight[1:118].to(row)))
        self.assertTrue(bool(torch.isnan(row[0, 117])))


class ORBOptionalTest(unittest.TestCase):
    """The ORB extra is optional: matris imports without it, an ORB load then fails with an ImportError,
    and load_adapter rejects invalid options before importing anything of ORB."""

    def test_matris_imports_without_orb(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "orb.ckpt")  # any file: the import fails first
            with open(checkpoint, "wb") as handle:
                handle.write(b"not a checkpoint")
            if not ORB:  # this module imported fastmd.adapters (and fastmd.models.orb) without it
                with self.assertRaises(ImportError):
                    load_adapter(MODEL, model_path=checkpoint, device="cpu")
                return
            code = ("import sys\n"
                    "sys.modules['orb_models'] = None  # orb-models cannot be imported\n"
                    "import fastmd, fastmd.adapters, fastmd.models.orb\n"
                    "try:\n"
                    "    fastmd.adapters.load_adapter('orb_v3', model_path=sys.argv[1], device='cpu')\n"
                    "except ImportError:\n"
                    "    print('ImportError at load time')\n")
            result = subprocess.run([sys.executable, "-B", "-c", code, checkpoint], capture_output=True, text=True,
                                    timeout=300, cwd=REPOSITORY)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        self.assertIn("ImportError at load time", result.stdout)

    def test_invalid_load_options_are_rejected(self):
        for options, error, keyword in (({"model_dtype": "float64"}, ValueError, "float32"),
                                        ({"model_dtype": "float16"}, ValueError, "model_dtype"),
                                        ({"optimization_profile": "fastest"}, ValueError, "optimization_profile"),
                                        ({"optimization_profile": "legacy"}, NegotiationError, "legacy")):
            with self.subTest(**options), self.assertRaisesRegex(error, keyword):
                load_adapter(MODEL, device="cpu", **options)


if __name__ == "__main__":
    unittest.main()
