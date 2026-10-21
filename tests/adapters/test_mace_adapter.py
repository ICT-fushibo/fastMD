"""MACE behind the adapter protocol, checked against mace-torch itself (accuracy only).

:mod:`fastmd.models.mace` runs the upstream model (MACE-MP-0 medium here)
through the engine: the edges-only graph builders (ragged, and fixed capacity
padded with far self-loops), the engine-owned derivative
(:class:`~fastmd.engine.energy_derivative.EnergyDerivative`),
GPU-resident MD in the eager, model_graph and whole_step modes, and the ASE
calculator. The references are mace-torch's own ASE calculator
(``mace.calculators.mace_mp``) and its own force and stress autograd on the
same input. FP64 agrees to round-off (relative 1e-12). IEEE FP32 forces agree
within ``max(1e-6, 1.5 nu)`` eV/A (cuEquivariance against e3nn: ``max(1e-5,
3 nu)``), ``nu`` the largest difference between repeated evaluations of one
input, measured in the test (the GPU's scatter atomics do not repeat bit for
bit).

The structure is a perturbed CsSnI3 perovskite in a sheared cell whose lattice
vectors are binary fractions: an FP32 shift ``image @ cell`` is then exact in
any summation order, so the engine and mace-torch see the same FP32 edge
vectors.

Needs the optional MACE extra (mace-torch; cuequivariance-torch for the
cuEquivariance kernels, without which the e3nn reference runs and the kernel
comparison is skipped), a CUDA device and the MACE-MP-0 checkpoint in
mace-torch's cache (``$XDG_CACHE_HOME/mace``, default ``~/.cache/mace``): the
tests never download. They are skipped otherwise, except the checks that the
extra is optional, which run everywhere.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.util
import io
import itertools
import math
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from ase import Atoms, units
from ase.io import read

from fastmd.adapters import as_adapter
from fastmd.adapters import load_adapter
from fastmd.models.mace import MACE_CHECKPOINTS
from fastmd.adapters.protocol import EnergyTap
from fastmd.interfaces.calculator import MLIPCalculator
from fastmd.engine.energy_derivative import EnergyDerivative
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.execution.md_protocol import HeatingRamp
from fastmd.config import CapacityConfig
from fastmd.config import InferenceConfig
from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.gpu_graph_builder import atoms_to_edges_gpu
from fastmd.graph.gpu_graph_builder import op_available
from fastmd.precision import fp32_gemm_mode
from fastmd.precision import matmul_precision_scope

MACE = importlib.util.find_spec("mace") is not None  # the optional extra (found, not imported)
CUEQ = importlib.util.find_spec("cuequivariance_torch") is not None
CUDA = torch.cuda.is_available() and op_available and _fixed_builder_available
MODEL = "mace_mp_0"
FP64_TOLERANCE = 1e-12  # relative: the round-off of another summation order
REPEATS = 5  # evaluations of one input that give an FP32 noise floor (10 pairs)
EXECUTIONS = ("eager", "model_graph", "whole_step")

# Loaded once per process: the first cuEquivariance call of a dtype compiles its kernels (about 30 s).
_LOADED: dict = {}
_PROCESS: dict = {}


def setUpModule():
    # Importing mace-torch sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for the process (its own torch.load of
    # pickled models needs it); this module puts it back afterwards, and the default dtype.
    _PROCESS.update(weights_only_load=os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"),
                    default_dtype=torch.get_default_dtype())


def tearDownModule():
    _LOADED.clear()
    if not _PROCESS:
        return
    if _PROCESS["weights_only_load"] is None:
        os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    else:
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = _PROCESS["weights_only_load"]
    torch.set_default_dtype(_PROCESS["default_dtype"])
    gc.collect()  # the captured graphs and their pools
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mace_checkpoint() -> str:
    """The MACE-MP-0 file mace-torch loads (``download_mace_mp_checkpoint("medium")``); SkipTest when it
    is not in mace-torch's cache: the tests never download."""
    from mace.calculators.foundations_models import mace_mp_urls
    from mace.tools.utils import get_cache_dir

    # mace-torch's cache name of a download: the URL's file name, letters, digits and "_" only
    name = "".join(c for c in os.path.basename(mace_mp_urls["medium"]) if c.isalnum() or c == "_")
    path = os.path.join(get_cache_dir(), name)
    if not os.path.isfile(path):
        raise unittest.SkipTest(f"MACE-MP-0 is not in mace-torch's cache ({path}): set XDG_CACHE_HOME to a "
                                f"directory with mace/{name} (the tests never download)")
    return path


def mace(model_dtype: str, profile: str | None = None):
    """MACE-MP-0 through ``load_adapter`` (None: cuEquivariance kernels where installed; "generic": e3nn)."""
    key = ("adapter", model_dtype, profile)
    if key not in _LOADED:
        _LOADED[key] = load_adapter(MODEL, device="cuda", model_dtype=model_dtype, optimization_profile=profile)
    return _LOADED[key]


def mace_calculator(model_dtype: str, cueq: bool):
    """mace-torch's own ASE calculator of MACE-MP-0, the reference (its cuEquivariance conversion sets the
    process default dtype: restored)."""
    key = ("calculator", model_dtype, cueq)
    if key not in _LOADED:
        from mace.calculators import mace_mp

        default = torch.get_default_dtype()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _LOADED[key] = mace_mp(model="medium", device="cuda", default_dtype=model_dtype, enable_cueq=cueq)
        finally:
            torch.set_default_dtype(default)
    return _LOADED[key]


def perovskite(seed: int = 0) -> Atoms:
    """CsSnI3, 2x2x2 cubic perovskite cells (a = 6.25 A, 40 atoms) sheared to a triclinic cell with
    binary-fraction lattice vectors, positions perturbed by 0.1 A."""
    a = 6.25
    atoms = Atoms("CsSnI3", cell=[a, a, a], pbc=True,
                  scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    atoms = atoms.repeat(2)
    atoms.set_cell(atoms.cell.array + [[0.0, 0.0, 0.0], [1.25, 0.0, 0.0], [0.75, 0.5, 0.0]], scale_atoms=True)
    atoms.positions += np.random.default_rng(seed).normal(0.0, 0.1, atoms.positions.shape)
    return atoms


def cluster_with_lone_atom() -> Atoms:
    """A perturbed Cs4I4 cube (edge 3.2 A) and one more Cs over 11 A from every other atom and image, in a
    20 A box (MACE-MP-0's r_max is 6 A)."""
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


def precision(model_dtype: str):
    return ieee_fp32() if model_dtype == "float32" else contextlib.nullcontext()


def ragged(atoms: Atoms, adapter):
    """The engine's ragged edges-only graph batch of ``atoms`` (no padding), in the model's dtype."""
    spec = adapter.topology_spec()
    return atoms_to_edges_gpu(atoms, spec.cutoff, "cuda", check_isolated_atoms=False,
                              positions_dtype=getattr(torch, spec.model_dtype))


def fixed_capacity_builder(atoms: Atoms, adapter, u_capacity: int) -> FixedCapacityGraphBuilder:
    """A fixed-capacity edges-only builder of ``atoms`` (the whole-step builder's mode, ``2 u_capacity`` edge
    rows) padded by the adapter's rule."""
    spec = adapter.topology_spec()
    dtype = getattr(torch, spec.model_dtype)
    return FixedCapacityGraphBuilder(
        cell=torch.tensor(atoms.cell.array, dtype=dtype), atomic_numbers=torch.tensor(atoms.numbers),
        u_capacity=u_capacity, t_capacity=0, atom_graph_cutoff=spec.cutoff, line_graph_cutoff=None,
        padding=adapter.padding_rule(), prepare=adapter.prepare, positions_dtype=dtype, device="cuda")


def node_energies(adapter, batch, inputs=None) -> torch.Tensor:
    """The adapter's energy tap on ``batch`` (no gradient; a detached view of the positions, which MACE marks
    as requiring grad)."""
    with torch.no_grad():
        return adapter.node_energy(adapter.prepare(batch) if inputs is None else inputs, batch.positions.detach(),
                                   batch.cell)


def engine_efs(adapter, batch, inputs=None) -> tuple[float, torch.Tensor, torch.Tensor]:
    """The engine-owned derivative of ``batch``: energy, forces of the real atoms and stress (eV/A^3), copied
    in float64."""
    out = EnergyDerivative(adapter)(batch, adapter.prepare(batch) if inputs is None else inputs, task="efs")
    return float(out["e"][0]), out["f"][0][: batch.n_real].double().clone(), out["s"][0].double().clone()


def upstream_efs(adapter, batch) -> tuple[float, torch.Tensor, torch.Tensor, float]:
    """mace-torch's own forces and stress (its autograd, ``compute_force`` and ``compute_stress``) of the
    engine's ``batch`` in MACE's input dict, each (centre, neighbour, image) edge as sender = neighbour,
    receiver = centre, ``unit_shifts = -image``: (energy as the FP64 sum of its node energies, forces,
    stress, its total energy in the model's dtype)."""
    module = adapter.module
    dtype = next(module.parameters()).dtype
    device = batch.positions.device
    z_table = module.atomic_numbers.tolist()
    species = torch.tensor([z_table.index(int(z)) for z in batch.species.tolist()], device=device)
    unit_shifts = -batch.edge_image.to(dtype)
    rows = batch.positions.shape[0]
    data = {
        "positions": batch.positions.detach().clone(),
        "node_attrs": torch.nn.functional.one_hot(species, len(z_table)).to(dtype),
        "edge_index": torch.stack((batch.edge_index[:, 1], batch.edge_index[:, 0])).long(),
        "unit_shifts": unit_shifts,
        "shifts": unit_shifts @ batch.cell,
        "cell": batch.cell.detach().clone(),
        "batch": torch.zeros(rows, dtype=torch.long, device=device),
        "ptr": torch.tensor([0, rows], device=device),
        "head": torch.zeros(1, dtype=torch.long, device=device),  # MACE-MP-0 has one head
    }
    out = module(data, training=False, compute_force=True, compute_stress=True)
    return (float(out["node_energy"].detach().sum()), out["forces"].detach().double(),
            out["stress"][0].detach().double(), float(out["energy"].detach()[0]))


def calculator_efs(adapter, atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
    """Energy, forces and stress (3x3, eV/A^3) of the calculator in eager mode."""
    probe = atoms.copy()
    probe.calc = MLIPCalculator(model=adapter, device="cuda", config=InferenceConfig(execution="eager", report=False))
    return probe.get_potential_energy(), probe.get_forces(), probe.get_stress(voigt=False)


def gpu_md(atoms: Atoms, adapter, execution: str, *, capacity: CapacityConfig | None = None, **kwargs):
    """GPU-resident MD of ``atoms``: a Nose-Hoover chain at 500 K (the integrator with a float64 state, which
    a float64 model needs), 1 fs steps."""
    config = InferenceConfig(execution=execution, report=False, capacity=capacity or CapacityConfig(warmup=1))
    return GPUResidentMolecularDynamics(atoms.copy(), model=adapter, ensemble="nvt_nhc", temperature=500.0,
                                        timestep=1.0, tdamp=25.0, config=config, **kwargs)


class FrameSink(list):
    """A trajectory that keeps the logged frames (their ``info`` included) in memory."""

    def write(self, atoms: Atoms) -> None:
        self.append(atoms)

    def close(self) -> None:
        pass


def as_numpy(values) -> tuple:
    return tuple(value.cpu().numpy() if torch.is_tensor(value) else value for value in values)


def relative(actual, expected) -> float:
    """max|actual - expected| / max|expected|."""
    expected = np.asarray(expected)
    return float(np.abs(np.asarray(actual) - expected).max() / np.abs(expected).max())


def largest(values) -> float:
    """The largest of ``values`` (floats), NaN if any is NaN (the built-in max drops a NaN after the first)."""
    return float(np.max(np.asarray(list(values), dtype=float)))


def max_pairwise(values) -> float:
    """The largest max|a - b| over the pairs of ``values`` (tensors); NaN if any difference is NaN."""
    pairs = [float((a - b).abs().max()) for index, a in enumerate(values) for b in values[:index]]
    return largest(pairs) if pairs else 0.0


@unittest.skipUnless(MACE, "the optional MACE extra (mace-torch) is not installed")
@unittest.skipUnless(CUDA, "CUDA and the GPU graph builders are unavailable")
class MACETestCase(unittest.TestCase):
    """The checks shared by the MACE tests; a class is skipped without the cached MACE-MP-0 checkpoint."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = mace_checkpoint()

    def assert_within(self, error: float, bound: float, what: str) -> None:
        self.assertTrue(math.isfinite(error), f"{what}: {error}")
        self.assertLessEqual(error, bound, f"{what}: {error:.3e} > {bound:.3e}")

    def assert_efs(self, actual, expected, tolerance: float, what: str) -> None:
        """|dE|/N (eV), max|dF| / max|F| and max|dS| / max|S| of ``actual`` within ``tolerance`` (a None entry
        is not compared)."""
        energy, forces, stress = actual
        expected_energy, expected_forces, expected_stress = expected
        if energy is not None:
            self.assert_within(abs(energy - expected_energy) / len(expected_forces), tolerance, f"{what}: |dE|/N")
        if forces is not None:
            self.assert_within(relative(forces, expected_forces), tolerance, f"{what}: max|dF|/max|F|")
        if stress is not None:
            self.assert_within(relative(stress, expected_stress), tolerance, f"{what}: max|dS|/max|S|")

    def reference(self, atoms: Atoms, adapter) -> Atoms:
        """``atoms`` with mace-torch's calculator of the adapter's dtype and kernels attached."""
        probe = atoms.copy()
        probe.calc = mace_calculator(adapter.topology_spec().model_dtype, adapter.kernel == "cueq")
        return probe


class MACEAccuracyTest(MACETestCase):
    def test_fp64_matches_the_mace_calculator(self):
        """FP64 against mace-torch's calculator (relative 1e-12): GPU-resident MD's eager evaluate and its
        logged first frame (energy, forces, stress in eV/A^3), and the calculator."""
        adapter = mace("float64")
        atoms = perovskite()
        probe = self.reference(atoms, adapter)
        expected = probe.get_potential_energy(), probe.get_forces(), probe.get_stress(voigt=False)
        self.assertGreater(np.abs(expected[1]).max(), 0.1)  # the perturbed crystal has forces
        results = {}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "md.traj")
            with gpu_md(atoms, adapter, "eager", trajectory=path, loginterval=1, log_stress=True) as md:
                forces, energy = md.evaluate(torch.tensor(atoms.positions, device="cuda"), stress=True)
                self.assertEqual(forces.dtype, torch.float64)
                results["GPU MD evaluate"] = (float(energy), forces.cpu().numpy(), None)
                md.run(1)
            frame = read(path, index=0)
        results["GPU MD frame"] = frame.get_potential_energy(), frame.get_forces(), frame.get_stress(voigt=False)
        results["calculator"] = calculator_efs(adapter, atoms)
        for name, actual in results.items():
            with self.subTest(name):
                self.assert_efs(actual, expected, FP64_TOLERANCE, name)

    def test_engine_derivative_matches_mace_autograd(self):
        """The engine-owned derivative (one autograd.grad of the FP64 sum of MACE's node energies through the
        positions and a strain) against mace-torch's own force and stress autograd on the same input: FP64
        relative 1e-12 (energy: mace-torch's total); IEEE FP32 forces within max(1e-6, 1.5 nu) eV/A, stress
        within max(1e-6 max|S|, 1.5 nu_S) (nu: the larger floor of the two paths over repeated evaluations),
        energy within 1e-6 eV/atom of the FP64 sum of mace-torch's node energies."""
        atoms = perovskite()
        for model_dtype in ("float64", "float32"):
            adapter = mace(model_dtype)
            with self.subTest(model_dtype=model_dtype), precision(model_dtype):
                batch = ragged(atoms, adapter)
                engine = [engine_efs(adapter, batch) for _ in range(REPEATS)]
                upstream = [upstream_efs(adapter, batch) for _ in range(REPEATS)]
                self.assertGreater(float(upstream[0][1].abs().max()), 0.1)
                if model_dtype == "float64":
                    _, forces, stress, total = upstream[0]
                    self.assert_efs(as_numpy(engine[0]), as_numpy((total, forces, stress)), FP64_TOLERANCE,
                                    "FP64")
                    continue
                self.assertEqual(fp32_gemm_mode(), "ieee")
                for index, name, absolute in ((1, "forces", 1e-6),
                                              (2, "stress", 1e-6 * float(upstream[0][2].abs().max()))):
                    floor = largest([max_pairwise([run[index] for run in engine]),
                                     max_pairwise([run[index] for run in upstream])])
                    error = largest(float((ours[index] - theirs[index]).abs().max())
                                    for ours, theirs in zip(engine, upstream))
                    self.assert_within(error, largest([absolute, 1.5 * floor]), f"FP32 {name} (floor {floor:.2e})")
                error = largest(abs(float(ours[0] - theirs[0])) for ours, theirs in zip(engine, upstream)) / len(atoms)
                self.assert_within(error, 1e-6, "FP32 |dE|/N")

    def test_padding_adds_exactly_nothing(self):
        """The adapter's padding (far self-loops): a fixed-capacity build with padded rows, fresh and on its
        reused candidate list, equals the ragged batch at FP64 round-off (relative 1e-12); a batch of padded
        rows only (two atoms farther apart than r_max) gives node energies equal bit for bit to those of the
        batch without edges, and forces and stress exactly zero."""
        adapter = mace("float64")
        self.assertEqual(adapter.padding_rule().kind, "far_self_loops")
        atoms = perovskite()
        expected = as_numpy(engine_efs(adapter, ragged(atoms, adapter)))
        builder = fixed_capacity_builder(atoms, adapter, u_capacity=512)
        positions = torch.tensor(atoms.positions, device="cuda")
        for reuse in (False, True):
            with self.subTest(candidates="reused" if reuse else "fresh"):
                inputs, status = builder.build(positions, rebuild_candidates=not reuse)
                self.assertGreater(builder.batch.edge_index.shape[0], int(status.edge_count))  # padded rows
                self.assert_efs(as_numpy(engine_efs(adapter, builder.batch, inputs)), expected, FP64_TOLERANCE,
                                "padded")
        apart = Atoms("CsI", positions=[[2.0, 2.0, 2.0], [12.0, 12.0, 12.0]], cell=[20.0] * 3, pbc=True)
        empty = ragged(apart, adapter)
        self.assertEqual(empty.edge_index.shape[0], 0)
        alone = node_energies(adapter, empty)
        builder = fixed_capacity_builder(apart, adapter, u_capacity=8)
        inputs, status = builder.build(torch.tensor(apart.positions, device="cuda"))
        padded = builder.batch
        self.assertEqual(int(status.edge_count), 0)
        self.assertGreater(padded.edge_index.shape[0], 0)
        self.assertTrue(torch.equal(node_energies(adapter, padded, inputs), alone))
        _, forces, stress = engine_efs(adapter, padded, inputs)
        self.assertTrue(bool((forces == 0).all()) and bool((stress == 0).all()))

    def test_isolated_atom_keeps_the_model_energy(self):
        """Isolated-atom policy "model": an atom farther than r_max from every other gets MACE's own energy,
        which for MACE-MP-0 is not its E0 (reference energy); the calculator and GPU-resident MD (eager, and
        whole_step's padded build) equal mace-torch's calculator at FP64 (relative 1e-12), with no force on
        that atom."""
        adapter = mace("float64")
        self.assertEqual(adapter.isolated_atom_policy(), "model")
        atoms = cluster_with_lone_atom()
        lone = len(atoms) - 1
        probe = self.reference(atoms, adapter)
        expected = probe.get_potential_energy(), probe.get_forces(), probe.get_stress(voigt=False)
        node_energy = probe.get_potential_energies()[lone]  # E0 included
        e0 = float(adapter.reference_energies()[0, atoms.numbers[lone] - 1])
        # mace-torch's E0 of the atom: its node energy minus its node energy without E0
        self.assert_within(abs(node_energy - probe.calc.results["node_energy"][lone] - e0),
                           FP64_TOLERANCE * abs(e0), "reference energy (E0)")
        self.assertGreater(abs(node_energy - e0), 0.1)
        rows = node_energies(adapter, ragged(atoms, adapter))
        self.assert_within(abs(float(rows[lone]) - node_energy), FP64_TOLERANCE * abs(node_energy),
                           "the isolated atom's node energy")
        results = {"calculator": calculator_efs(adapter, atoms)}
        for execution in ("eager", "whole_step"):
            with gpu_md(atoms, adapter, execution) as md:
                if execution == "eager":
                    forces, energy = md.evaluate(torch.tensor(atoms.positions, device="cuda"))
                else:  # the whole-step runner's first evaluation: the padded fixed-capacity build
                    forces, energy = md.state.forces, md.state.potential_energy
                results[f"GPU MD {execution}"] = (float(energy.reshape(-1)[0]), forces.cpu().numpy(), None)
        for name, actual in results.items():
            with self.subTest(name):
                self.assert_efs(actual, expected, FP64_TOLERANCE, name)
                self.assertTrue(np.all(actual[1][lone] == 0.0))

    @unittest.skipUnless(CUEQ, "cuequivariance-torch is not installed")
    def test_cuequivariance_matches_e3nn(self):
        """The default kernels (cuEquivariance) against e3nn (optimization_profile="generic", the reference)
        through the engine-owned derivative: FP64 forces within 1e-10 eV/A, energy and stress relative
        1e-12; IEEE FP32 forces within max(1e-5, 3 nu)."""
        atoms = perovskite()
        for model_dtype in ("float64", "float32"):
            fast, reference = mace(model_dtype), mace(model_dtype, "generic")
            with self.subTest(model_dtype=model_dtype), precision(model_dtype):
                self.assertEqual((fast.kernel, reference.kernel), ("cueq", "e3nn"), fast.kernel_reason)
                batch = ragged(atoms, fast)
                ours = [engine_efs(fast, batch) for _ in range(REPEATS)]
                theirs = [engine_efs(reference, batch) for _ in range(REPEATS)]
                error = largest(float((a[1] - b[1]).abs().max()) for a, b in zip(ours, theirs))
                if model_dtype == "float64":
                    self.assert_within(error, 1e-10, "FP64 max|dF| (eV/A)")
                    self.assert_efs(as_numpy(ours[0]), as_numpy(theirs[0]), FP64_TOLERANCE, "FP64")
                    continue
                self.assertEqual(fp32_gemm_mode(), "ieee")
                floor = largest([max_pairwise([run[1] for run in ours]), max_pairwise([run[1] for run in theirs])])
                self.assert_within(error, largest([1e-5, 3.0 * floor]), f"FP32 max|dF| (floor {floor:.2e})")


class MACEDynamicsTest(MACETestCase):
    def thermalized(self) -> Atoms:
        """The perovskite with Maxwell-Boltzmann momenta at 500 K (fixed seed)."""
        atoms = perovskite(seed=1)
        kinetic = np.sqrt(atoms.get_masses()[:, None] * units.kB * 500.0)
        atoms.set_momenta(np.random.default_rng(2).normal(0.0, 1.0, atoms.positions.shape) * kinetic)
        return atoms

    def test_execution_modes_agree(self):
        """GPU-resident MD (nvt_nhc) in eager, model_graph and whole_step, and the calculator in model_graph:
        the first evaluation (eager: the ragged batch; model_graph: a captured padded bucket, the calculator's
        on its skin topology where it applies; whole_step: the padded fixed-capacity build) equals eager's
        (FP64 relative 1e-12; IEEE FP32 forces within max(1e-6, 1.5 nu) eV/A, energy 1e-6 eV/atom); FP64 over
        20 steps: positions within 1e-10 A, the logged first frame's energy, forces and stress (and the
        calculator's stress) relative 1e-12."""
        atoms = self.thermalized()
        positions = torch.tensor(atoms.positions, device="cuda")
        steps = 20
        for model_dtype in ("float64", "float32"):
            adapter = mace(model_dtype)
            first, frames, final = {}, {}, {}
            with self.subTest(model_dtype=model_dtype), precision(model_dtype), \
                    tempfile.TemporaryDirectory() as directory:
                for execution in EXECUTIONS:
                    path = os.path.join(directory, f"{execution}.traj")
                    with gpu_md(atoms, adapter, execution, trajectory=path, loginterval=10, log_stress=True) as md:
                        self.assertEqual(md.config_summary()["effective"]["execution"], execution)
                        if execution == "whole_step":
                            forces, energy = md.state.forces, md.state.potential_energy
                        else:  # model_graph returns its graph's output buffers: copied before the next replay
                            forces, energy = md.evaluate(positions)
                        first[execution] = (float(energy.reshape(-1)[0]), forces.double().cpu())
                        if execution == "eager":
                            repeats = [first["eager"][1]] + [md.evaluate(positions)[0].double().cpu()
                                                             for _ in range(REPEATS - 1)]
                        md.run(steps)
                        final[execution] = md.snapshot_atoms().positions
                        if execution == "model_graph":
                            self.assertGreater(md.model_graph_stats()["captures"], 0)
                        elif execution == "whole_step":
                            self.assertEqual(md.whole_step_stats(capacity_plan=False)["launches"], steps)
                    frames[execution] = read(path, ":")
                probe = atoms.copy()
                probe.calc = MLIPCalculator(model=adapter, device="cuda",
                                              config=InferenceConfig(execution="model_graph", report=False))
                first["calculator model_graph"] = (probe.get_potential_energy(), torch.as_tensor(probe.get_forces()))
                self.assertEqual(probe.calc.config_summary()["effective"]["execution"], "model_graph")
                if model_dtype == "float32":
                    self.assertEqual(fp32_gemm_mode(), "ieee")
                    floor = max_pairwise(repeats)
                for name, (energy, forces) in first.items():
                    if name == "eager":
                        continue
                    if model_dtype == "float32":
                        self.assert_within(float((forces - first["eager"][1]).abs().max()),
                                           largest([1e-6, 1.5 * floor]),
                                           f"FP32 {name}: max|dF| (floor {floor:.2e})")
                        self.assert_within(abs(energy - first["eager"][0]) / len(atoms), 1e-6, f"FP32 {name}: |dE|/N")
                        continue
                    self.assert_efs((energy, forces.numpy(), None),
                                    (first["eager"][0], first["eager"][1].numpy(), None), FP64_TOLERANCE,
                                    f"{name} first evaluation")
                if model_dtype == "float32":
                    continue
                eager = frames["eager"][0]
                eager = eager.get_potential_energy(), eager.get_forces(), eager.get_stress(voigt=False)
                self.assert_efs((None, None, probe.get_stress(voigt=False)), eager, FP64_TOLERANCE,
                                "calculator model_graph")
                for execution in EXECUTIONS[1:]:
                    self.assert_within(float(np.abs(final[execution] - final["eager"]).max()), 1e-10,
                                       f"{execution}: max|dx| after {steps} steps (A)")
                    self.assertEqual(len(frames[execution]), len(frames["eager"]))
                    # Every logged frame (steps 0, 10, 20): the whole-step frames are staged on the device and
                    # written after the next window is launched.
                    for index, (ours, theirs) in enumerate(zip(frames[execution], frames["eager"])):
                        self.assert_within(float(np.abs(ours.positions - theirs.positions).max()), 1e-10,
                                           f"{execution} frame {index}: max|dx| (A)")
                        self.assert_efs((ours.get_potential_energy(), ours.get_forces(), ours.get_stress(voigt=False)),
                                        (theirs.get_potential_energy(), theirs.get_forces(),
                                         theirs.get_stress(voigt=False)), FP64_TOLERANCE, f"{execution} frame {index}")

    def test_ten_step_windows_under_a_ramp(self):
        """Whole-step MD (FP64) in ten-step windows (inner steps without the float32 mirrors, the window's
        statistics in one read, frames staged and copied while the next window runs) under a heating ramp
        (per-step targets inside each window) equals eager MD over 30 steps: positions within 1e-10 A, and
        every logged frame's energy, forces, stress and conserved energy relative 1e-12 (a staged frame's
        thermostat sums use its own window's target, not the next one's)."""
        adapter = mace("float64")
        atoms = self.thermalized()
        steps, final, frames = 30, {}, {}
        ramp = HeatingRamp(start_K=500.0, end_K=1500.0, total_steps=steps)
        for execution in ("eager", "whole_step"):
            sink = FrameSink()
            with gpu_md(atoms, adapter, execution, trajectory=sink, loginterval=10, log_stress=True,
                        steps_per_launch=10, ramp=ramp) as md:
                md.run(steps)
                final[execution] = md.snapshot_atoms().positions
                if execution == "whole_step":
                    self.assertEqual(md.whole_step_stats(capacity_plan=False)["launches"], steps // 10)
                md.flush_log()
            frames[execution] = list(sink)
        self.assert_within(float(np.abs(final["whole_step"] - final["eager"]).max()), 1e-10,
                           f"whole_step: max|dx| after {steps} steps (A)")
        self.assertEqual(len(frames["whole_step"]), len(frames["eager"]))
        for index, (ours, theirs) in enumerate(zip(frames["whole_step"], frames["eager"])):
            self.assert_efs((ours.get_potential_energy(), ours.get_forces(), ours.get_stress(voigt=False)),
                            (theirs.get_potential_energy(), theirs.get_forces(), theirs.get_stress(voigt=False)),
                            FP64_TOLERANCE, f"frame {index}")
            conserved = float(np.asarray(ours.info["nhc_conserved_energy"]).reshape(-1)[0])
            expected = float(np.asarray(theirs.info["nhc_conserved_energy"]).reshape(-1)[0])
            self.assert_within(abs(conserved - expected) / abs(expected), FP64_TOLERANCE,
                               f"frame {index}: conserved energy (relative)")

    def test_skin_reuse_equals_rebuild(self):
        """Whole-step MD (FP64, 50 steps) on one candidate list (the default 2 A skin) equals a 0.02 A skin (a
        fresh candidate list after almost every step) and a skin of 0 (whole_step needs a skin: the plan
        runs model_graph, a rebuilt graph every step): positions within 1e-10 A."""
        adapter = mace("float64")
        atoms = self.thermalized()
        steps = 50
        positions, stats, executions = {}, {}, {}
        for skin in (2.0, 0.02, 0.0):
            with gpu_md(atoms, adapter, "whole_step", capacity=CapacityConfig(warmup=1, candidate_skin=skin)) as md:
                md.run(steps)
                positions[skin] = md.snapshot_atoms().positions
                stats[skin] = md.whole_step_stats(capacity_plan=False)
                executions[skin] = next(decision for decision in md.plan_record().as_dict()["decisions"]
                                        if decision["id"] == "execution")
                self.assertEqual(md.config_summary()["effective"]["execution"], executions[skin]["chosen"])
        self.assertLess(stats[2.0]["candidate_refreshes"], 5)
        self.assertGreaterEqual(stats[0.02]["candidate_refreshes"], steps // 2)
        self.assertEqual(executions[0.0]["chosen"], "model_graph")
        self.assertIn("needs a skin", executions[0.0]["reason"])
        self.assertIsNone(stats[0.0])
        for skin in (0.02, 0.0):
            with self.subTest(candidate_skin=skin):
                self.assert_within(float(np.abs(positions[skin] - positions[2.0]).max()), 1e-10,
                                   f"skin {skin} A: max|dx| after {steps} steps (A)")

    def test_auto_execution_is_negotiated(self):
        """execution="auto" (FP64): GPU-MD and the calculator decide through the negotiation with a memory
        model measured on MACE's own evaluation (the engine-owned derivative), and the plan record says what
        runs (the negotiated mode, the effective mode and the runner that ran agree)."""
        adapter = mace("float64")
        atoms = self.thermalized()
        config = InferenceConfig(execution="auto", report=False, capacity=CapacityConfig(warmup=1))
        with GPUResidentMolecularDynamics(atoms.copy(), model=adapter, ensemble="nvt_nhc", temperature=500.0,
                                          timestep=1.0, tdamp=25.0, config=config) as md:
            md.run(20)
            execution = next(decision for decision in md.plan_record().as_dict()["decisions"]
                             if decision["id"] == "execution")
            ran = ("whole_step" if md.whole_step_runner is not None and md.whole_step_runner.launches > 0
                   else "model_graph" if md.runner is not None else "eager")
            self.assertIn("measured capture bound", execution["reason"])
            self.assertEqual(execution["chosen"], md.config_summary()["effective"]["execution"])
            self.assertEqual(execution["chosen"], ran)
        calc = MLIPCalculator(model=adapter, device="cuda", config=config)
        for frame in range(3):  # a first call (eager), its repeat (decides), one call on the decided path
            probe = atoms.copy()
            probe.positions += 0.01 * frame
            probe.calc = calc
            probe.get_potential_energy()
        execution = next(decision for decision in calc.plan_record().as_dict()["decisions"]
                         if decision["id"] == "execution")
        self.assertIn("measured capture bound", execution["reason"])
        self.assertEqual(execution["chosen"], calc.config_summary()["effective"]["execution"])


class MACELoadingTest(MACETestCase):
    def test_load_by_name_and_by_path(self):
        """By name: mace-torch's cached file with the pinned sha256, an adapter (returned as is by the
        resolvers) with an energy tap; by a ".model" path or "mace" with model_path: the same weights."""
        adapter = mace("float64")
        pinned = MACE_CHECKPOINTS[MODEL][1]
        with open(self.checkpoint, "rb") as handle:
            self.assertEqual(hashlib.sha256(handle.read()).hexdigest(), pinned)
        self.assertEqual(adapter.checkpoint["sha256"], pinned)
        self.assertEqual(os.path.realpath(adapter.checkpoint["path"]), os.path.realpath(self.checkpoint))
        self.assertIsInstance(adapter, EnergyTap)
        self.assertIs(as_adapter(adapter), adapter)
        self.assertIs(load_adapter(adapter), adapter)
        spec = adapter.topology_spec()
        self.assertEqual((spec.cutoff, spec.num_interactions, spec.triplet_cutoff, spec.model_dtype),
                         (6.0, 2, None, "float64"))
        named = mace("float64", "generic").module.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "mace-mp-0-medium.model")
            os.symlink(self.checkpoint, path)
            for model, model_path in ((path, None), ("mace", self.checkpoint)):
                with self.subTest(model=os.path.basename(model)):
                    loaded = load_adapter(model, model_path=model_path, device="cuda", model_dtype="float64",
                                          optimization_profile="generic")
                    self.assertEqual((loaded.kernel, loaded.checkpoint["sha256"]), ("e3nn", pinned))
                    weights = loaded.module.state_dict()
                    self.assertEqual(weights.keys(), named.keys())
                    self.assertTrue(all(torch.equal(weights[key], named[key]) for key in named))

    def test_checkpoint_hash_mismatch_is_rejected(self):
        upstream, _ = MACE_CHECKPOINTS[MODEL]
        with mock.patch.dict(MACE_CHECKPOINTS, {MODEL: (upstream, "0" * 64)}), \
                self.assertRaisesRegex(ValueError, "sha256"):
            load_adapter(MODEL, device="cuda", model_dtype="float64", optimization_profile="generic")


class MACEOptionalTest(unittest.TestCase):
    """The MACE extra is optional: matris imports without it, a MACE load then fails with an ImportError,
    and load_adapter rejects invalid options before importing anything of MACE."""

    def test_matris_imports_without_mace(self):
        if not MACE:  # this module imported fastmd.adapters (and fastmd.models.mace) without it
            with self.assertRaises(ImportError):
                load_adapter(MODEL, device="cpu")
            return
        code = ("import sys\n"
                "sys.modules['mace'] = None  # mace-torch cannot be imported\n"
                "import fastmd, fastmd.adapters, fastmd.models.mace\n"
                "try:\n"
                "    fastmd.adapters.load_adapter('mace_mp_0', device='cpu')\n"
                "except ImportError:\n"
                "    print('ImportError at load time')\n")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        self.assertIn("ImportError at load time", result.stdout)

    def test_invalid_load_options_are_rejected(self):
        for options, keyword in (({"model_dtype": "float16"}, "model_dtype"),
                                 ({"optimization_profile": "fastest"}, "optimization_profile"),
                                 ({"optimization_profile": "legacy"}, "legacy")):
            with self.subTest(**options), self.assertRaisesRegex(ValueError, keyword):
                load_adapter(MODEL, device="cpu", **options)


if __name__ == "__main__":
    unittest.main()
