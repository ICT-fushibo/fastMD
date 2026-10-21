"""SevenNet behind the adapter protocol, checked against sevenn itself (accuracy only).

:mod:`fastmd.models.sevennet` runs the upstream model (SevenNet-0) through
the engine: the edges-only graph builders (ragged, and fixed capacity padded
with exact-rc self-loops), the engine-owned derivative
(:class:`~fastmd.engine.energy_derivative.EnergyDerivative`),
GPU-resident MD in the eager, model_graph and whole_step modes, and the ASE
calculator. The references are sevenn's own force head on the same input (its
autograd of the edge vectors) and its own ASE calculator. SevenNet-0 is a
float32 model: IEEE FP32 forces agree within ``max(1e-6, 1.5 nu)`` eV/A
(another kernel option: ``max(1e-5, 3 nu)``), ``nu`` the largest difference
between repeated evaluations of one input, measured in the test (the GPU's
scatter atomics do not repeat bit for bit). On the CPU (e3nn, one thread) the
padding is checked bit for bit.

The structure is a perturbed CsSnI3 perovskite in a sheared cell whose lattice
vectors are binary fractions: an FP32 shift ``image @ cell`` is then exact in
any summation order, so the engine and sevenn see the same FP32 edge vectors.

Needs the optional SevenNet extra (sevenn; cuequivariance-torch for the
cuEquivariance kernels, without which the e3nn reference runs and the kernel
checks are skipped) and a CUDA device; the 7net-0 checkpoint ships inside the
sevenn wheel. The tests are skipped otherwise, except the checks that the
extra is optional, which run everywhere.
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import importlib.util
import itertools
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

from fastmd.adapters import as_adapter
from fastmd.adapters import load_adapter
from fastmd.adapters.protocol import EnergyTap
from fastmd.interfaces.calculator import MLIPCalculator
from fastmd.engine.energy_derivative import EnergyDerivative
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.config import CapacityConfig
from fastmd.config import InferenceConfig
from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.gpu_graph_builder import atoms_to_edges_gpu
from fastmd.graph.gpu_graph_builder import op_available
from fastmd.precision import fp32_gemm_mode
from fastmd.precision import matmul_precision_scope

SEVENN = importlib.util.find_spec("sevenn") is not None  # the optional extra (found, not imported)
CUEQ = importlib.util.find_spec("cuequivariance_torch") is not None
CUDA = torch.cuda.is_available() and op_available and _fixed_builder_available
MODEL = "sevennet_0"
REPEATS = 5  # evaluations of one input that give an FP32 noise floor (10 pairs)
EXECUTIONS = ("eager", "model_graph", "whole_step")

# Loaded once per process: the first cuEquivariance call of a method compiles its kernels (about 20 s).
_LOADED: dict = {}


def tearDownModule():
    _LOADED.clear()
    gc.collect()  # the captured graphs and their pools
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sevennet(profile: str | None = None, device: str = "cuda"):
    """SevenNet-0 through ``load_adapter`` (None: cuEquivariance kernels where installed; "generic": e3nn)."""
    key = ("adapter", profile, device)
    if key not in _LOADED:
        _LOADED[key] = load_adapter(MODEL, device=device, optimization_profile=profile)
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
    """A perturbed Cs4I4 cube (edge 3.2 A) and one more Cs over 9 A from every other atom and image, in a
    20 A box (SevenNet-0's cutoff is 5 A)."""
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


def ragged(atoms: Atoms, adapter, device: str = "cuda"):
    """The engine's ragged edges-only graph batch of ``atoms`` (no padding), float32."""
    return atoms_to_edges_gpu(atoms, adapter.topology_spec().cutoff, device, check_isolated_atoms=False,
                              positions_dtype=torch.float32)


def fixed_capacity_builder(atoms: Atoms, adapter, u_capacity: int) -> FixedCapacityGraphBuilder:
    """A fixed-capacity edges-only builder of ``atoms`` (the whole-step builder's mode, ``2 u_capacity`` edge
    rows, candidates within the cutoff + 2 A) padded by the adapter's rule, its edges grouped by centre."""
    spec = adapter.topology_spec()
    return FixedCapacityGraphBuilder(
        cell=torch.tensor(atoms.cell.array, dtype=torch.float32), atomic_numbers=torch.tensor(atoms.numbers),
        u_capacity=u_capacity, t_capacity=0, atom_graph_cutoff=spec.cutoff, line_graph_cutoff=None,
        candidate_cutoff=spec.cutoff + 2.0, padding=adapter.padding_rule(), prepare=adapter.prepare,
        positions_dtype=torch.float32, sorted_edges=spec.edge_order == "sorted", device="cuda")


def node_energies(adapter, batch, inputs=None) -> torch.Tensor:
    with torch.no_grad():
        return adapter.node_energy(adapter.prepare(batch) if inputs is None else inputs, batch.positions.detach(),
                                   batch.cell)


def engine_efs(adapter, batch, inputs=None) -> tuple[float, torch.Tensor, torch.Tensor]:
    """The engine-owned derivative of ``batch``: energy, forces of the real atoms and stress (eV/A^3), copied
    in float64."""
    out = EnergyDerivative(adapter)(batch, adapter.prepare(batch) if inputs is None else inputs, task="efs")
    return float(out["e"][0]), out["f"][0][: batch.n_real].double().clone(), out["s"][0].double().clone()


def upstream_module(adapter):
    """sevenn's own module of the adapter's checkpoint and kernels, with its force head (the reference
    derivative), built through sevenn's switches: ``enable_cueq`` and its ``CUEQ_TP_METHOD``."""
    key = ("upstream", adapter.kernel)
    if key not in _LOADED:
        from sevenn.checkpoint import SevenNetCheckpoint

        method = {"cueq_uniform_1d": "uniform_1d", "cueq_fused_tp": "fused_tp"}.get(adapter.kernel)
        with mock.patch.dict(os.environ, {"CUEQ_TP_METHOD": method} if method else {}):
            module = SevenNetCheckpoint(adapter.checkpoint["path"]).build_model(enable_cueq=method is not None)
        module.set_is_batch_data(False)
        _LOADED[key] = module.to("cuda").eval()
    return _LOADED[key]


def upstream_efs(adapter, batch) -> tuple[float, torch.Tensor, torch.Tensor]:
    """sevenn's own energy (the FP64 sum of its atomic energies), forces and stress (its force head: the
    autograd of the edge vectors, ASE's sign and eV/A^3) on the engine's ``batch``: the same edges, (centre,
    neighbour) as (receiver, sender), and the same FP32 edge vectors."""
    from sevenn.atom_graph_data import AtomGraphData

    module = upstream_module(adapter)
    positions, cell, image = batch.positions.detach(), batch.cell, batch.edge_image.to(torch.float32)
    centre, neighbour = batch.edge_index[:, 0].long(), batch.edge_index[:, 1].long()
    shift = image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]
    vectors = positions[neighbour] + shift - positions[centre]
    numbers = batch.species.long()
    data = AtomGraphData(x=numbers.clone(), atomic_numbers=numbers, pos=positions.clone(),
                         edge_index=torch.stack((centre, neighbour)), edge_vec=vectors.clone(),
                         cell_volume=torch.linalg.det(cell.double()).abs().float(),
                         num_atoms=torch.tensor(numbers.shape[0], device=numbers.device))
    out = module(data)
    stress = -out["inferred_stress"].detach().double()  # sevenn's (xx, yy, zz, xy, yz, zx), the virial's sign
    xx, yy, zz, xy, yz, zx = stress.tolist()
    return (float(out["atomic_energy"].detach().double().sum()), out["inferred_force"].detach().double(),
            torch.tensor([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]], dtype=torch.float64, device=stress.device))


def gpu_md(atoms: Atoms, adapter, execution: str, *, capacity: CapacityConfig | None = None, **kwargs):
    """GPU-resident MD of ``atoms``: a Nose-Hoover chain at 500 K, 1 fs steps, 32-pair capacity tiers."""
    config = InferenceConfig(execution=execution, report=False,
                             capacity=capacity or CapacityConfig(warmup=1, u_step=32, min_pad_u=32))
    return GPUResidentMolecularDynamics(atoms.copy(), model=adapter, ensemble="nvt_nhc", temperature=500.0,
                                        timestep=1.0, tdamp=25.0, config=config, **kwargs)


def largest(values) -> float:
    """The largest of ``values`` (floats), NaN if any is NaN (the built-in max drops a NaN after the first)."""
    return float(np.max(np.asarray(list(values), dtype=float)))


def max_pairwise(values) -> float:
    """The largest max|a - b| over the pairs of ``values`` (tensors); NaN if any difference is NaN."""
    pairs = [float((a - b).abs().max()) for index, a in enumerate(values) for b in values[:index]]
    return largest(pairs) if pairs else 0.0


def cpu_copy(batch, rows: int | None = None):
    """A CPU copy of a graph batch; ``rows``: only its first edge rows, without padding."""
    tensors = {field.name: getattr(batch, field.name) for field in dataclasses.fields(batch)
               if torch.is_tensor(getattr(batch, field.name))}
    copied = {name: value.detach().cpu().clone() for name, value in tensors.items()}
    if rows is not None:
        copied.update(edge_index=copied["edge_index"][:rows], edge_image=copied["edge_image"][:rows],
                      edge_vec_override=None)
        return dataclasses.replace(batch, **copied, padding=None, status=None)
    return dataclasses.replace(batch, **copied, status=None)


@unittest.skipUnless(SEVENN, "the optional SevenNet extra (sevenn) is not installed")
@unittest.skipUnless(CUDA, "CUDA and the GPU graph builders are unavailable")
class SevenNetTestCase(unittest.TestCase):
    """The checks shared by the SevenNet tests."""

    def assert_within(self, error: float, bound: float, what: str) -> None:
        self.assertTrue(math.isfinite(error), f"{what}: {error}")
        self.assertLessEqual(error, bound, f"{what}: {error:.3e} > {bound:.3e}")


class SevenNetAccuracyTest(SevenNetTestCase):
    def test_engine_derivative_matches_the_force_head(self):
        """The engine-owned derivative (one autograd.grad of the FP64 sum of SevenNet's atomic energies through
        the positions and a strain) against sevenn's own force head on the same input, IEEE FP32: forces
        within max(1e-6, 1.5 nu) eV/A, stress within max(1e-6 max|S|, 1.5 nu_S) (nu: the larger floor of the
        two paths over repeated evaluations), energy within 1e-6 eV/atom."""
        adapter = sevennet()
        atoms = perovskite()
        with ieee_fp32():
            batch = ragged(atoms, adapter)
            engine = [engine_efs(adapter, batch) for _ in range(REPEATS)]
            upstream = [upstream_efs(adapter, batch) for _ in range(REPEATS)]
            self.assertEqual(fp32_gemm_mode(), "ieee")
        self.assertGreater(float(upstream[0][1].abs().max()), 0.1)  # the perturbed crystal has forces
        for index, name, absolute in ((1, "forces", 1e-6), (2, "stress", 1e-6 * float(upstream[0][2].abs().max()))):
            floor = largest([max_pairwise([run[index] for run in engine]),
                             max_pairwise([run[index] for run in upstream])])
            error = largest(float((ours[index] - theirs[index]).abs().max()) for ours, theirs in zip(engine, upstream))
            self.assert_within(error, largest([absolute, 1.5 * floor]), f"{name} (floor {floor:.2e})")
        error = largest(abs(ours[0] - theirs[0]) for ours, theirs in zip(engine, upstream)) / len(atoms)
        self.assert_within(error, 1e-6, "|dE|/N")

    def test_calculator_matches_the_sevennet_calculator(self):
        """The calculator (the engine on float32 positions) against sevenn's own ASE calculator (float64 host
        geometry), IEEE FP32: energy within 1e-6 eV/atom, forces within 5e-5 eV/A, stress within 1e-6
        eV/A^3."""
        from sevenn.calculator import SevenNetCalculator

        adapter = sevennet()
        atoms = perovskite()
        with ieee_fp32(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = atoms.copy()
            probe.calc = SevenNetCalculator("7net-0", device="cuda", enable_cueq=adapter.kernel != "e3nn")
            expected = probe.get_potential_energy(), probe.get_forces(), probe.get_stress(voigt=False)
            probe.calc = MLIPCalculator(model=adapter, device="cuda",
                                          config=InferenceConfig(execution="eager", report=False))
            actual = probe.get_potential_energy(), probe.get_forces(), probe.get_stress(voigt=False)
        self.assert_within(abs(actual[0] - expected[0]) / len(atoms), 1e-6, "|dE|/N")
        self.assert_within(float(np.abs(actual[1] - expected[1]).max()), 5e-5, "max|dF|")
        self.assert_within(float(np.abs(actual[2] - expected[2]).max()), 1e-6, "max|dS|")

    def test_padding_adds_exactly_nothing(self):
        """The adapter's padding (self-loops whose edge vector is (rc, 0, 0)): on the CPU (e3nn, one thread) a
        fixed-capacity build with padded rows gives the energy, forces and stress of its real rows bit for bit,
        and a batch of padded rows only (two atoms farther apart than rc) the node energies of the batch
        without edges with forces and stress exactly zero; on the GPU (cuEquivariance) the padded build is
        within max(1e-6, 1.5 nu) of the ragged batch."""
        reference, adapter = sevennet(device="cpu"), sevennet()
        rule = adapter.padding_rule()
        self.assertEqual((rule.kind, rule.pad_vector), ("exact_rc_loops", (5.0, 0.0, 0.0)))
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for name, atoms, capacity in (("perovskite", perovskite(), 512),
                                          ("apart", Atoms("CsI", positions=[[2.0, 2.0, 2.0], [12.0, 12.0, 12.0]],
                                                          cell=[20.0] * 3, pbc=True), 8)):
                with self.subTest(name):
                    builder = fixed_capacity_builder(atoms, adapter, u_capacity=capacity)
                    _, status = builder.build(torch.tensor(atoms.positions, device="cuda"))
                    real = int(status.edge_count)
                    self.assertGreater(builder.batch.edge_index.shape[0], real)  # padded rows
                    padded, unpadded = cpu_copy(builder.batch), cpu_copy(builder.batch, rows=real)
                    ours, theirs = engine_efs(reference, padded), engine_efs(reference, unpadded)
                    self.assertEqual(ours[0], theirs[0])
                    self.assertTrue(torch.equal(ours[1], theirs[1]) and torch.equal(ours[2], theirs[2]))
                    self.assertTrue(torch.equal(node_energies(reference, padded), node_energies(reference, unpadded)))
                    if real == 0:
                        self.assertTrue(bool((ours[1] == 0).all()) and bool((ours[2] == 0).all()))
        finally:
            torch.set_num_threads(threads)
        atoms = perovskite()
        with ieee_fp32():
            batch = ragged(atoms, adapter)
            expected = [engine_efs(adapter, batch)[1] for _ in range(REPEATS)]
            builder = fixed_capacity_builder(atoms, adapter, u_capacity=512)
            inputs, _ = builder.build(torch.tensor(atoms.positions, device="cuda"))
            actual = [engine_efs(adapter, builder.batch, inputs)[1] for _ in range(REPEATS)]
        floor = largest([max_pairwise(expected), max_pairwise(actual)])
        self.assert_within(largest(float((a - b).abs().max()) for a, b in zip(actual, expected)),
                           largest([1e-6, 1.5 * floor]), f"GPU padded vs ragged max|dF| (floor {floor:.2e})")

    def test_isolated_atom_keeps_the_model_energy(self):
        """Isolated-atom policy "model": an atom farther than rc from every other gets SevenNet's own energy (not
        its species shift); the calculator and GPU-resident MD (eager, and whole_step's padded build) give it
        no force and agree with sevenn's calculator (energy 1e-6 eV/atom)."""
        from sevenn.calculator import SevenNetCalculator

        adapter = sevennet()
        self.assertEqual(adapter.isolated_atom_policy(), "model")
        self.assertIsNone(adapter.reference_energies())
        atoms = cluster_with_lone_atom()
        lone = len(atoms) - 1
        with ieee_fp32(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = atoms.copy()
            probe.calc = SevenNetCalculator("7net-0", device="cuda", enable_cueq=adapter.kernel != "e3nn")
            expected = probe.get_potential_energy()
            rows = node_energies(adapter, ragged(atoms, adapter))
            shift = float(adapter.module._modules["rescale_atomic_energy"].shift[
                adapter.module.type_map[int(atoms.numbers[lone])]])
            self.assertGreater(abs(float(rows[lone]) - shift), 0.1)
            results = {}
            probe.calc = MLIPCalculator(model=adapter, device="cuda",
                                          config=InferenceConfig(execution="eager", report=False))
            results["calculator"] = probe.get_potential_energy(), probe.get_forces()
            for execution in ("eager", "whole_step"):
                with gpu_md(atoms, adapter, execution) as md:
                    if execution == "eager":
                        forces, energy = md.evaluate(torch.tensor(atoms.positions, device="cuda"))
                    else:  # the whole-step runner's first evaluation: the padded fixed-capacity build
                        forces, energy = md.state.forces, md.state.potential_energy
                    results[f"GPU MD {execution}"] = (float(energy.reshape(-1)[0]), forces.cpu().numpy())
        for name, (energy, forces) in results.items():
            with self.subTest(name):
                self.assert_within(abs(energy - expected) / len(atoms), 1e-6, f"{name}: |dE|/N")
                self.assertTrue(np.all(forces[lone] == 0.0))

    @unittest.skipUnless(CUEQ, "cuequivariance-torch is not installed")
    def test_kernel_options_match_e3nn(self):
        """cuEquivariance fused_tp and uniform_1d (switched in place on one module) against e3nn
        (optimization_profile="generic", the reference) through the engine-owned derivative, IEEE FP32: forces
        within max(1e-5, 3 nu); every convolution runs the installed method."""
        fast, reference = sevennet(), sevennet("generic")
        self.assertEqual(reference.kernel, "e3nn")
        atoms = perovskite()
        with ieee_fp32():
            batch = ragged(atoms, fast)
            theirs = [engine_efs(reference, batch)[1] for _ in range(REPEATS)]
            for option in ("cueq_fused_tp", "cueq_uniform_1d"):
                with self.subTest(option):
                    module = fast.module
                    fast.select_kernel("convolution", option)
                    self.assertIs(fast.module, module)  # in place: the same module object
                    self.assertEqual(fast.kernel, option, fast.kernel_reason)
                    ours = [engine_efs(fast, batch)[1] for _ in range(REPEATS)]
                    floor = largest([max_pairwise(ours), max_pairwise(theirs)])
                    error = largest(float((a - b).abs().max()) for a, b in zip(ours, theirs))
                    self.assert_within(error, largest([1e-5, 3.0 * floor]), f"{option} max|dF| (floor {floor:.2e})")
        with self.assertRaisesRegex(RuntimeError, "rebuilds the module"):
            fast.select_kernel("convolution", "e3nn")  # the e3nn module is chosen at load

    def test_one_evaluation_never_synchronizes(self):
        """The tap and its derivative on a fixed-capacity batch make no host synchronization (sync-free: a CUDA
        graph captures them), with the one-hot patch applied."""
        adapter = sevennet()
        self.assertTrue(adapter.capture_contract().sync_free)
        atoms = perovskite()
        builder = fixed_capacity_builder(atoms, adapter, u_capacity=512)
        inputs, _ = builder.build(torch.tensor(atoms.positions, device="cuda"))
        derivative = EnergyDerivative(adapter)
        derivative(builder.batch, inputs, task="efs", n_real=len(atoms))  # first call outside the check
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            derivative(builder.batch, inputs, task="efs", n_real=len(atoms))
        finally:
            torch.cuda.set_sync_debug_mode("default")


class SevenNetDynamicsTest(SevenNetTestCase):
    def thermalized(self) -> Atoms:
        """The perovskite with Maxwell-Boltzmann momenta at 500 K (fixed seed)."""
        atoms = perovskite(seed=1)
        kinetic = np.sqrt(atoms.get_masses()[:, None] * units.kB * 500.0)
        atoms.set_momenta(np.random.default_rng(2).normal(0.0, 1.0, atoms.positions.shape) * kinetic)
        return atoms

    def test_execution_modes_agree(self):
        """GPU-resident MD (nvt_nhc) in eager, model_graph and whole_step, and the calculator in model_graph:
        the first evaluation (eager: the ragged batch; model_graph: a captured padded bucket; whole_step: the
        padded fixed-capacity build) equals eager's within max(1e-6, 1.5 nu) eV/A (nu over repeated eager
        evaluations) and 1e-6 eV/atom; every mode runs the kernel its plan ranks first (uniform_1d); 20 steps
        stay finite."""
        adapter = sevennet()
        atoms = self.thermalized()
        positions = torch.tensor(atoms.positions, device="cuda")
        first, repeats = {}, []
        with ieee_fp32():
            for execution in EXECUTIONS:
                with self.subTest(execution), gpu_md(atoms, adapter, execution) as md:
                    self.assertEqual(md.config_summary()["effective"]["execution"], execution)
                    plan = md.plan_record()
                    kernel = plan.decision("kernels.convolution")
                    if CUEQ:
                        self.assertEqual((kernel.chosen, adapter.kernel), ("cueq_uniform_1d",) * 2, kernel.reason)
                    self.assertEqual(plan.decision("execution").chosen, execution)
                    if execution == "whole_step":
                        forces, energy = md.state.forces, md.state.potential_energy
                    else:  # model_graph returns its graph's output buffers: copied before the next replay
                        forces, energy = md.evaluate(positions)
                    first[execution] = (float(energy.reshape(-1)[0]), forces.double().cpu())
                    if execution == "eager":
                        repeats += [md.evaluate(positions)[0].double().cpu() for _ in range(REPEATS)]
                    md.run(20)
                    final = md.snapshot_atoms()
                    self.assertTrue(np.isfinite(final.positions).all() and np.isfinite(final.get_momenta()).all())
                    if execution == "model_graph":
                        self.assertGreater(md.model_graph_stats()["captures"], 0)
                    elif execution == "whole_step":
                        self.assertEqual(md.whole_step_stats(capacity_plan=False)["launches"], 20)
            probe = atoms.copy()
            probe.calc = MLIPCalculator(model=adapter, device="cuda",
                                          config=InferenceConfig(execution="model_graph", report=False))
            first["calculator model_graph"] = (probe.get_potential_energy(), torch.as_tensor(probe.get_forces()))
            self.assertEqual(probe.calc.config_summary()["effective"]["execution"], "model_graph")
            self.assertEqual(fp32_gemm_mode(), "ieee")
        floor = max_pairwise(repeats)
        for name, (energy, forces) in first.items():
            if name == "eager":
                continue
            with self.subTest(name):
                self.assert_within(float((forces - first["eager"][1]).abs().max()), largest([1e-6, 1.5 * floor]),
                                   f"{name}: max|dF| (floor {floor:.2e})")
                self.assert_within(abs(energy - first["eager"][0]) / len(atoms), 1e-6, f"{name}: |dE|/N")

    def test_skin_reuse_equals_rebuild(self):
        """The whole-step builder's reused candidates (within rc + skin, re-filtered exactly to r < rc at every
        build) give the edges of a fresh build, as sets (edges grouped by centre), over five displacements of
        up to 0.3 A, and forces within max(1e-6, 1.5 nu) of the fresh build's."""
        adapter = sevennet()
        atoms = perovskite()
        start = torch.tensor(atoms.positions, device="cuda")
        reused = fixed_capacity_builder(atoms, adapter, u_capacity=512)
        reused.build(start)
        rng = np.random.default_rng(5)
        with ieee_fp32():
            for step in range(1, 6):
                positions = start + torch.tensor(rng.uniform(-0.06 * step, 0.06 * step, atoms.positions.shape),
                                                 device="cuda", dtype=start.dtype)
                with self.subTest(step=step):
                    inputs, status = reused.build(positions, rebuild_candidates=False)
                    fresh = fixed_capacity_builder(atoms, adapter, u_capacity=512)
                    fresh_inputs, fresh_status = fresh.build(positions)
                    rows = {}
                    for name, builder, count in (("reused", reused, status.edge_count),
                                                 ("fresh", fresh, fresh_status.edge_count)):
                        count = int(count)
                        edges = torch.cat((builder.batch.edge_index[:count].double(),
                                           builder.batch.edge_image[:count].double()), dim=1)
                        rows[name] = sorted(map(tuple, edges.cpu().tolist()))
                    self.assertEqual(rows["reused"], rows["fresh"])
                    ours = [engine_efs(adapter, reused.batch, inputs)[1] for _ in range(REPEATS)]
                    theirs = [engine_efs(adapter, fresh.batch, fresh_inputs)[1] for _ in range(REPEATS)]
                    floor = largest([max_pairwise(ours), max_pairwise(theirs)])
                    self.assert_within(largest(float((a - b).abs().max()) for a, b in zip(ours, theirs)),
                                       largest([1e-6, 1.5 * floor]), f"max|dF| (floor {floor:.2e})")


class SevenNetPatchesTest(SevenNetTestCase):
    def test_declined_one_hot_patch_stays_correct(self):
        """A one-hot patch declined (its pinned source changed): the adapter loads, says why in its plan, is not
        declared sync-free, and a requested whole_step runs eager with the forces of the patched adapter
        (within max(1e-6, 1.5 nu)); closing an adapter reverts its patch (the one-hot table back on the CPU)."""
        import fastmd.models.sevennet as sevennet_module

        patched = sevennet("generic")
        with mock.patch.object(sevennet_module, "ONEHOT_SOURCE_SHA256", "0" * 64):
            declined = load_adapter(MODEL, device="cuda", optimization_profile="generic")
        outcome = {o.record.name: o for o in declined.upstream_patches()}["sevennet.onehot_table_on_device"]
        self.assertFalse(outcome.applied)
        self.assertIn("changed", outcome.reason)
        self.assertEqual(declined.module.z_to_onehot_tensor.device.type, "cpu")
        self.assertFalse(declined.capture_contract().sync_free)
        module = declined.module
        atoms = perovskite()
        positions = torch.tensor(atoms.positions, device="cuda")
        with ieee_fp32():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the requested whole_step runs eager (a ConfigurationWarning)
                md = gpu_md(atoms, declined, "whole_step")
            with md:
                self.assertEqual(md.config_summary()["effective"]["execution"], "eager")
                plan = md.plan_record()
                ours = [md.evaluate(positions)[0].double().cpu() for _ in range(REPEATS)]
            self.assertIn("not declared sync-free", plan.decision("execution").reason)
            # Loaded with e3nn, kept with e3nn: the default profile's cuEquivariance options would rebuild the module
            self.assertIs(declined.module, module)
            self.assertEqual(plan.decision("kernels.convolution").chosen, "e3nn")
            self.assertIn("sevennet.onehot_table_on_device", plan.patches["declined"])
            self.assertIn("sevennet.xplor_cutoff", [entry["name"] for entry in plan.patches["applied"]])
            with gpu_md(atoms, patched, "eager") as md:
                theirs = [md.evaluate(positions)[0].double().cpu() for _ in range(REPEATS)]
        floor = largest([max_pairwise(ours), max_pairwise(theirs)])
        self.assert_within(largest(float((a - b).abs().max()) for a, b in zip(ours, theirs)),
                           largest([1e-6, 1.5 * floor]), f"declined vs patched max|dF| (floor {floor:.2e})")
        closing = load_adapter(MODEL, device="cuda", optimization_profile="generic")
        self.assertEqual(closing.module.z_to_onehot_tensor.device.type, "cuda")
        closing.close()
        self.assertEqual(closing.module.z_to_onehot_tensor.device.type, "cpu")

    def test_changed_pinned_source_is_refused(self):
        """A pin whose source hash does not match (code the padding or the tap relies on changed) refuses the
        load, naming the symbol."""
        import fastmd.models.sevennet as sevennet_module

        broken = tuple(dataclasses.replace(record, source_sha256="0" * 64) if record.name == "sevennet.xplor_cutoff"
                       else record for record in sevennet_module.PINNED_RECORDS)
        with mock.patch.object(sevennet_module, "PINNED_RECORDS", broken), \
                self.assertRaisesRegex(RuntimeError, "XPLORCutoff.forward"):
            load_adapter(MODEL, device="cuda", optimization_profile="generic")


class SevenNetLoadingTest(SevenNetTestCase):
    def test_load_by_name_and_by_path(self):
        """By name: the checkpoint in the sevenn wheel with the pinned sha256, an adapter (returned as is by the
        resolvers) with an energy tap; "sevennet" with model_path: the same weights."""
        from fastmd.models.sevennet import SEVENNET_CHECKPOINTS

        adapter = sevennet("generic")
        pinned = SEVENNET_CHECKPOINTS[MODEL][1]
        self.assertEqual(adapter.checkpoint["sha256"], pinned)
        self.assertIsInstance(adapter, EnergyTap)
        self.assertIs(as_adapter(adapter), adapter)
        self.assertIs(load_adapter(adapter), adapter)
        spec = adapter.topology_spec()
        self.assertEqual((spec.cutoff, spec.num_interactions, spec.triplet_cutoff, spec.model_dtype, spec.edge_order),
                         (5.0, 5, None, "float32", "by_centre"))
        self.assertNotIn("force_output", adapter.module._modules)
        loaded = load_adapter("sevennet", model_path=adapter.checkpoint["path"], device="cuda",
                              optimization_profile="generic")
        self.assertEqual((loaded.kernel, loaded.checkpoint["sha256"]), ("e3nn", pinned))
        weights, named = loaded.module.state_dict(), adapter.module.state_dict()
        self.assertEqual(weights.keys(), named.keys())
        self.assertTrue(all(torch.equal(weights[key], named[key]) for key in named))

    @unittest.skipUnless(CUEQ, "cuequivariance-torch is not installed")
    def test_load_falls_back_along_the_kernel_options(self):
        """At load the cuEquivariance methods are tried in order, then e3nn: uniform_1d unavailable gives
        fused_tp; a cuEquivariance build without sevenn's fused convolution (CUEQ_USE_SCATTER_FUSION=0) gives
        e3nn's own build, not the unfused one under the e3nn label; the reasons are kept."""
        import fastmd.models.sevennet as sevennet_module

        with mock.patch.dict(sevennet_module._TP_CLASSES, {"uniform_1d": "unavailable"}):
            adapter = load_adapter(MODEL, device="cuda")
        self.assertEqual(adapter.kernel, "cueq_fused_tp")
        self.assertIn("cueq_uniform_1d", adapter.kernel_reason)
        with mock.patch.dict(os.environ, {"CUEQ_USE_SCATTER_FUSION": "0"}):
            adapter = load_adapter(MODEL, device="cuda")
        self.assertEqual(adapter.kernel, "e3nn")
        self.assertIn("fused scatter convolution", adapter.kernel_reason)
        self.assertTrue(all(type(layer.convolution).__module__.startswith("e3nn.")
                            for name, layer in adapter.module._modules.items() if name.endswith("_convolution")))

    def test_module_with_the_force_head_is_refused(self):
        """The adapter relies on a module whose force head is removed (the loader removes it)."""
        from sevenn.checkpoint import SevenNetCheckpoint

        from fastmd.models.sevennet import SevenNetAdapter

        module = SevenNetCheckpoint(sevennet("generic").checkpoint["path"]).build_model(enable_cueq=False)
        with self.assertRaisesRegex(ValueError, "force head"):
            SevenNetAdapter(module.to("cuda").eval())

    def test_checkpoint_hash_mismatch_is_rejected(self):
        from fastmd.models.sevennet import SEVENNET_CHECKPOINTS

        upstream, _ = SEVENNET_CHECKPOINTS[MODEL]
        with mock.patch.dict(SEVENNET_CHECKPOINTS, {MODEL: (upstream, "0" * 64)}), \
                self.assertRaisesRegex(ValueError, "sha256"):
            load_adapter(MODEL, device="cuda", optimization_profile="generic")


class SevenNetOptionalTest(unittest.TestCase):
    """The SevenNet extra is optional: matris imports without it, a SevenNet load then fails with an
    ImportError, and load_adapter rejects invalid options before importing anything of sevenn."""

    def test_matris_imports_without_sevenn(self):
        if not SEVENN:  # this module imported fastmd.adapters (and fastmd.models.sevennet) without it
            with self.assertRaises(ImportError):
                load_adapter(MODEL, device="cpu")
            return
        code = ("import sys\n"
                "sys.modules['sevenn'] = None  # sevenn cannot be imported\n"
                "import fastmd, fastmd.adapters, fastmd.models.sevennet\n"
                "try:\n"
                "    fastmd.adapters.load_adapter('sevennet_0', device='cpu')\n"
                "except ImportError:\n"
                "    print('ImportError at load time')\n")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        self.assertIn("ImportError at load time", result.stdout)

    def test_invalid_load_options_are_rejected(self):
        for options, keyword in (({"model_dtype": "float64"}, "float32"),
                                 ({"optimization_profile": "fastest"}, "optimization_profile"),
                                 ({"optimization_profile": "legacy"}, "legacy")):
            with self.subTest(**options), self.assertRaisesRegex(ValueError, keyword):
                load_adapter(MODEL, device="cpu", **options)


if __name__ == "__main__":
    unittest.main()
