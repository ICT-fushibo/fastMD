"""CHGNet behind the adapter protocol, checked against upstream chgnet itself (accuracy only).

:mod:`fastmd.models.chgnet` runs the upstream model (CHGNet 0.3.0) through the
engine: the line-graph builders (ragged, and fixed capacity padded with sink
atoms), the engine-owned derivative
(:class:`~fastmd.engine.energy_derivative.EnergyDerivative`), GPU-resident
MD in the eager, model_graph and whole_step modes, and the ASE calculator. The
references are upstream's own graph converter (``CrystalGraphConverter``), its
own force and stress autograd (``CHGNet._compute``) on the same input, and its
ASE calculator. FP64 (CPU) agrees to round-off (relative 1e-12). IEEE FP32
forces agree within ``max(1e-6, 1e-5 max|F|, k nu)`` eV/A: ``k nu`` where the
GPU's scatter atomics make repeated evaluations of one input differ (``k`` the
capture contract's tolerance multiplier, 2; ``nu`` the largest difference
between repeated evaluations, measured in the test), ``1e-5 max|F|`` the FP32
rounding of another summation order where they do not. Upstream's calculator converts the structure itself:
its FP32 fractional coordinates differ from the engine's by an ulp, which
moves CHGNet's forces by up to a few 1e-4 eV/A, so end-to-end comparisons with
it are sanity checks (5e-4 eV/A).

Needs the optional CHGNet extra (chgnet) and a CUDA device; skipped otherwise,
except the checks that the extra is optional, which run everywhere.
"""
from __future__ import annotations

import copy
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
from ase.io import read

from fastmd.adapters import as_adapter
from fastmd.adapters import load_adapter
from fastmd.adapters.protocol import EnergyTap
from fastmd.adapters.protocol import GraphBatch
from fastmd.adapters.protocol import UpstreamPatches
from fastmd.interfaces.calculator import MLIPCalculator
from fastmd.engine.energy_derivative import EnergyDerivative
from fastmd.engine.energy_derivative import cell_volume
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.config import CapacityConfig
from fastmd.config import ConfigurationWarning
from fastmd.config import InferenceConfig
from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.fixed_capacity_builder import line_graph_batch
from fastmd.graph.gpu_graph_builder import atoms_to_graph_gpu
from fastmd.graph.gpu_graph_builder import op_available
from fastmd.precision import fp32_gemm_mode
from fastmd.precision import matmul_precision_scope
from matris_test_assets import RDX_1344
from matris_test_assets import test_asset
from matris_test_assets import test_assets_available

CHGNET = importlib.util.find_spec("chgnet") is not None  # the optional extra (found, not imported)
CUDA = torch.cuda.is_available() and op_available and _fixed_builder_available
MODEL = "chgnet_0_3_0"
FP64_TOLERANCE = 1e-12  # relative: the round-off of another summation order
REPEATS = 5  # evaluations of one input that give an FP32 noise floor (10 pairs)
SANITY = 5e-4  # eV/A: against upstream's calculator, which converts the structure itself (see above)
ROUNDING = 1e-5  # relative: FP32 rounding of another summation order (the floor without atomics noise)
EXECUTIONS = ("eager", "model_graph", "whole_step")
# Small capacity tiers for the 40-atom test cells (the defaults pad them to thousands of rows).
SMALL = dict(u_step=64, min_pad_u=64, t_step=256, min_pad_t=128)

_LOADED: dict = {}


def tearDownModule():
    for adapter in _LOADED.values():
        adapter.close()
    _LOADED.clear()
    gc.collect()  # the captured graphs and their pools
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def chgnet():
    """CHGNet 0.3.0 through ``load_adapter`` (loaded once per process)."""
    if "adapter" not in _LOADED:
        _LOADED["adapter"] = load_adapter(MODEL, device="cuda")
    return _LOADED["adapter"]


def perovskite(seed: int = 0) -> Atoms:
    """CsSnI3, 2x2x2 cubic perovskite cells with a = 5.9 A (Sn-I bonds of 2.95 A, inside CHGNet's 3 A
    bond-graph cutoff, so the Sn and I centres carry angles) sheared to a triclinic cell, positions
    perturbed by 0.05 A and wrapped into the cell (40 atoms)."""
    a = 5.9
    atoms = Atoms("CsSnI3", cell=[a, a, a], pbc=True,
                  scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    atoms = atoms.repeat(2)
    atoms.set_cell(atoms.cell.array + [[0.0, 0.0, 0.0], [1.25, 0.0, 0.0], [0.75, 0.5, 0.0]], scale_atoms=True)
    atoms.positions += np.random.default_rng(seed).normal(0.0, 0.05, atoms.positions.shape)
    atoms.wrap()
    return atoms


def primitive_cells() -> dict:
    """Cells shorter than the cutoffs, whose atoms bond to their own images: fcc copper (one atom, lattice
    vectors of 2.55 A: bonds and angles between images of one atom) and diamond silicon (two atoms, 3.84 A),
    both perturbed."""
    from ase.build import bulk

    cells = {"Cu fcc primitive": bulk("Cu", "fcc", a=3.61), "Si diamond primitive": bulk("Si", "diamond", a=5.43)}
    for index, atoms in enumerate(cells.values()):
        atoms.positions += np.random.default_rng(10 + index).normal(0.0, 0.03, atoms.positions.shape)
        atoms.wrap()
    return cells


def thermalized(seed: int = 1) -> Atoms:
    """The perovskite with Maxwell-Boltzmann momenta at 500 K (fixed seed)."""
    atoms = perovskite(seed)
    kinetic = np.sqrt(atoms.get_masses()[:, None] * units.kB * 500.0)
    atoms.set_momenta(np.random.default_rng(seed + 1).normal(0.0, 1.0, atoms.positions.shape) * kinetic)
    return atoms


def cluster_with_lone_atom() -> Atoms:
    """A perturbed Cs4I4 cube (edge 3.2 A) and one more Cs over 11 A from every other atom and image, in a
    20 A box (CHGNet's atom-graph cutoff is 6 A)."""
    corners = np.array(list(itertools.product((0.0, 3.2), repeat=3)))
    cube = Atoms(numbers=[55 if sum(index) % 2 else 53 for index in itertools.product((0, 1), repeat=3)],
                 positions=4.0 + corners + np.random.default_rng(3).normal(0.0, 0.05, corners.shape))
    atoms = cube + Atoms("Cs", positions=[[14.0, 14.0, 14.0]])
    atoms.set_cell([20.0] * 3)
    atoms.pbc = True
    return atoms


def ieee_fp32():
    """IEEE FP32 GEMMs (no TF32) inside; the caller's precision afterwards."""
    return matmul_precision_scope("highest")


def ragged(atoms: Atoms, adapter) -> GraphBatch:
    """The engine's ragged line-graph batch of ``atoms`` (no padding) with the builder's positions."""
    spec = adapter.topology_spec()
    graph = atoms_to_graph_gpu(atoms, spec.cutoff, spec.triplet_cutoff, "cuda", check_isolated_atoms=False)
    return line_graph_batch(graph)


def fixed_capacity_builder(atoms: Atoms, adapter, u_capacity: int, t_capacity: int) -> FixedCapacityGraphBuilder:
    """A fixed-capacity line-graph builder of ``atoms`` (the whole-step builder's mode: sink padding and
    published positions)."""
    spec = adapter.topology_spec()
    return FixedCapacityGraphBuilder(
        cell=torch.tensor(atoms.cell.array, dtype=torch.float32), atomic_numbers=torch.tensor(atoms.numbers),
        u_capacity=u_capacity, t_capacity=t_capacity, atom_graph_cutoff=spec.cutoff,
        line_graph_cutoff=spec.triplet_cutoff, padding=adapter.padding_rule(), prepare=adapter.prepare,
        publish_positions=True, device="cuda")


def engine_efs(adapter, batch, inputs=None) -> tuple[float, torch.Tensor, torch.Tensor]:
    """The engine-owned derivative of ``batch``: energy, forces of the real atoms and stress (eV/A^3), copied
    in float64."""
    out = EnergyDerivative(adapter)(batch, adapter.prepare(batch) if inputs is None else inputs, task="efs",
                                    n_real=batch.n_real)
    return float(out["e"][0]), out["f"][0][: batch.n_real].double().clone(), out["s"][0].double().clone()


def crystal_graph(batch: GraphBatch, dtype: torch.dtype):
    """Upstream's ``CrystalGraph`` of the engine's ragged ``batch`` (the same edges, pairs and triplets, the
    fractional coordinates and cell in ``dtype``)."""
    from chgnet.graph.crystalgraph import CrystalGraph

    return CrystalGraph(
        atomic_number=batch.species.long(), atom_frac_coord=batch.frac_coords.to(dtype),
        atom_graph=batch.edge_index.long(), atom_graph_cutoff=batch.cutoff,
        neighbor_image=batch.edge_image.to(dtype), directed2undirected=batch.pair_d2u.long(),
        undirected2directed=batch.pair_u2d.long(), bond_graph=batch.triplets.long(),
        bond_graph_cutoff=batch.triplet_cutoff, lattice=batch.cell.to(dtype))


def upstream_efs(module, graph) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Upstream's own derivative (``CHGNet._compute`` with ``compute_force`` and ``compute_stress``, its
    ``create_graph`` autograd) of ``graph``: the energy as the FP64 sum of the readout's per-atom energies
    plus the AtomRef weights, forces, and the stress in eV/A^3 (upstream's GPa scale, whose volume is
    float32, undone with that same volume, then divided by the exact volume)."""
    from chgnet.model.model import BatchedGraph

    batched = BatchedGraph.from_graphs([graph], module.bond_basis_expansion, module.angle_basis_expansion,
                                       compute_stress=True)
    tapped = []
    hook = module.mlp.register_forward_hook(lambda _module, _args, output: tapped.append(output))
    try:
        with warnings.catch_warnings(), torch.enable_grad():
            warnings.simplefilter("ignore")
            out = module._compute(batched, compute_force=True, compute_stress=True)
    finally:
        hook.remove()
    weights = module.composition_model.fc.weight.detach()[0]
    energy = (tapped[-1].detach().squeeze(1).double().sum()
              + weights.index_select(0, graph.atomic_number - 1).double().sum())
    scale = 1 / batched.volumes * 160.21766208  # upstream's own GPa scale (model.py), undone below
    strain_gradient = out["s"][0].detach().double() / scale.double()
    lattice = graph.lattice.double()
    stress = strain_gradient / cell_volume(lattice)
    return float(energy), out["f"][0].detach().double(), stress


def calculator_efs(adapter, atoms: Atoms, execution: str = "eager", reuse: str = "off"):
    """Energy, forces and stress (3x3, eV/A^3) of the calculator."""
    probe = atoms.copy()
    probe.calc = MLIPCalculator(model=adapter, device="cuda", config=InferenceConfig(
        execution=execution, topology_reuse=reuse, report=False))
    return probe.get_potential_energy(), probe.get_forces(), probe.get_stress(voigt=False)


def gpu_md(atoms: Atoms, adapter, execution: str, *, capacity: CapacityConfig | None = None, **kwargs):
    """GPU-resident MD of ``atoms``: a Nose-Hoover chain at 500 K, 1 fs steps, small capacity tiers."""
    config = InferenceConfig(execution=execution, report=False, capacity=capacity or CapacityConfig(warmup=1, **SMALL))
    return GPUResidentMolecularDynamics(atoms.copy(), model=adapter, ensemble="nvt_nhc", temperature=500.0,
                                        timestep=1.0, tdamp=25.0, config=config, **kwargs)


def relative(actual, expected) -> float:
    """max|actual - expected| / max|expected|."""
    expected = np.asarray(expected)
    return float(np.abs(np.asarray(actual) - expected).max() / np.abs(expected).max())


def largest(values) -> float:
    """The largest of ``values`` (floats), NaN if any is NaN."""
    return float(np.max(np.asarray(list(values), dtype=float)))


def max_pairwise(values) -> float:
    """The largest max|a - b| over the pairs of ``values`` (tensors); NaN if any difference is NaN."""
    pairs = [float((a - b).abs().max()) for index, a in enumerate(values) for b in values[:index]]
    return largest(pairs) if pairs else 0.0


# --- topology sets ------------------------------------------------------------------------------------------

def topology_sets(atom_graph, image, d2u, u2d, bond_graph) -> dict:
    """The canonical sets of a CHGNet-layout graph: directed edges (centre, neighbour, image), undirected
    pairs (the two directed keys) and angles (centre, first bond's key, second bond's key); and whether
    the rows are consistent (no edge or angle row twice, d2u pairs reverse edges, u2d lies in its pair, an
    angle row's two distinct bonds start at its centre and belong to its pairs)."""
    edges = np.asarray(atom_graph, dtype=np.int64)
    images = np.rint(np.asarray(image, dtype=np.float64)).astype(np.int64)
    d2u = np.asarray(d2u, dtype=np.int64)
    u2d = np.asarray(u2d, dtype=np.int64)
    rows = np.asarray(bond_graph, dtype=np.int64).reshape(-1, 5)
    keys = [(int(c), int(n), *map(int, shift)) for (c, n), shift in zip(edges, images)]
    by_pair: dict[int, list[int]] = {}
    for edge, pair in enumerate(d2u):
        by_pair.setdefault(int(pair), []).append(edge)

    def reverse(key):
        return (key[1], key[0], -key[2], -key[3], -key[4])

    consistent = (len(by_pair) == u2d.shape[0]
                  and all(len(pair) == 2 and keys[pair[1]] == reverse(keys[pair[0]]) for pair in by_pair.values())
                  and all(int(d2u[int(u2d[pair])]) == pair for pair in range(u2d.shape[0]))
                  and all(d2u[e1] == p1 and d2u[e2] == p2 and keys[e1][0] == c == keys[e2][0] and e1 != e2
                          for c, p1, e1, p2, e2 in rows))
    angles = [(int(c), keys[e1], keys[e2]) for c, _, e1, _, e2 in rows]
    # no row twice: the sets below would hide a duplicated edge or angle row
    consistent = consistent and len(set(keys)) == len(keys) and len(set(angles)) == len(angles)
    return {"edges": set(keys), "pairs": {frozenset((keys[a], keys[b])) for a, b in by_pair.values()},
            "angles": set(angles), "consistent": consistent}


def borderline_keys(atoms: Atoms, keys, cutoff: float) -> set:
    """The edge keys whose FP64 length is within eps of ``cutoff`` (eps = 8 ulp(M) of FP32, M = max|r| +
    max|n cell| over the keys: where the builders' FP32 test and the converter's FP64 test may differ)."""
    keys = list(keys)
    edges = np.array([key[:2] for key in keys], dtype=np.int64)
    shifts = np.array([key[2:] for key in keys], dtype=np.float64) @ atoms.cell.array
    positions = atoms.positions
    lengths = np.linalg.norm(positions[edges[:, 1]] + shifts - positions[edges[:, 0]], axis=1)
    eps = 8 * np.spacing(np.float32(np.abs(positions).max() + np.abs(shifts).max()))
    return {key for key, length in zip(keys, lengths) if abs(length - cutoff) <= eps}


def stable_sets(sets: dict, near_atom_graph: set, near_bond_graph: set) -> dict:
    """``sets`` without the borderline keys: edges and pairs near the atom-graph cutoff, angles with a bond
    near either cutoff."""
    near = near_atom_graph | near_bond_graph
    return {"edges": {key for key in sets["edges"] if key not in near_atom_graph},
            "pairs": {pair for pair in sets["pairs"] if not pair & near_atom_graph},
            "angles": {angle for angle in sets["angles"] if angle[1] not in near and angle[2] not in near}}


@unittest.skipUnless(CHGNET, "the optional CHGNet extra (chgnet) is not installed")
@unittest.skipUnless(CUDA, "CUDA and the GPU graph builders are unavailable")
class CHGNetTestCase(unittest.TestCase):
    """The checks shared by the CHGNet tests."""

    def assert_within(self, error: float, bound: float, what: str) -> None:
        self.assertTrue(math.isfinite(error), f"{what}: {error}")
        self.assertLessEqual(error, bound, f"{what}: {error:.3e} > {bound:.3e}")

    def assert_efs(self, actual, expected, tolerance: float, what: str, *, forces: float | None = None) -> None:
        """|dE|/N (eV), max|dF| / max|F| (or ``forces``: max|dF| in eV/A) and max|dS| / max|S| of
        ``actual`` within ``tolerance`` (a None entry is not compared)."""
        energy, force, stress = actual
        expected_energy, expected_forces, expected_stress = expected
        if energy is not None:
            self.assert_within(abs(energy - expected_energy) / len(expected_forces), tolerance, f"{what}: |dE|/N")
        if force is not None:
            if forces is None:
                self.assert_within(relative(force, expected_forces), tolerance, f"{what}: max|dF|/max|F|")
            else:
                self.assert_within(float(np.abs(np.asarray(force) - expected_forces).max()), forces,
                                   f"{what}: max|dF| (eV/A)")
        if stress is not None:
            self.assert_within(relative(stress, expected_stress), tolerance, f"{what}: max|dS|/max|S|")

    def tolerance(self, floor: float, scale: float) -> float:
        """The FP32 force tolerance: the capture contract's max(1e-6, k nu), and at least the FP32 rounding of
        another summation order, ROUNDING times the largest force ``scale``."""
        return largest([1e-6, ROUNDING * scale, chgnet().capture_contract().tol_multiplier * floor])


class CHGNetTopologyTest(CHGNetTestCase):
    def test_line_graph_sets_equal_upstream(self):
        """The engine's line graph (the ragged builder and the fixed-capacity builder's real rows) equals
        upstream's CrystalGraphConverter (an FP64 neighbour list) as sets: directed edges, undirected pairs
        and angles, except pairs within eps (8 FP32 ulps of the coordinates) of the 6 A or 3 A cutoff, where an
        FP32 and an FP64 distance test may disagree; no row appears twice and the pair maps and angle rows are
        consistent. On the perturbed perovskite (two seeds), two primitive cells whose atoms bond to their own
        images, and the 1,344-atom RDX cell when its test asset is available."""
        from pymatgen.io.ase import AseAtomsAdaptor

        adapter = chgnet()
        spec = adapter.topology_spec()
        structures = {"perovskite seed 0": perovskite(0), "perovskite seed 5": perovskite(5), **primitive_cells()}
        if test_assets_available(RDX_1344):
            rdx = read(test_asset(RDX_1344))
            rdx.wrap()
            structures["rdx_1344"] = rdx
        for name, atoms in structures.items():
            with self.subTest(name):
                graph = adapter.module.graph_converter(AseAtomsAdaptor.get_structure(atoms))
                upstream = topology_sets(graph.atom_graph, graph.neighbor_image, graph.directed2undirected,
                                         graph.undirected2directed, graph.bond_graph)
                batch = ragged(atoms, adapter)
                engines = {"ragged": topology_sets(batch.edge_index.cpu(), batch.edge_image.cpu(),
                                                   batch.pair_d2u.cpu(), batch.pair_u2d.cpu(),
                                                   batch.triplets.cpu())}
                u, t = batch.pair_u2d.shape[0], batch.triplets.shape[0]
                builder = fixed_capacity_builder(atoms, adapter, u_capacity=u + 64, t_capacity=t + 128)
                _, status = builder.build(torch.tensor(atoms.positions, dtype=torch.float32, device="cuda"))
                e, t = int(status.edge_count), int(status.triplet_count)
                fixed = builder.batch
                engines["fixed capacity"] = topology_sets(
                    fixed.edge_index[:e].cpu(), fixed.edge_image[:e].cpu(), fixed.pair_d2u[:e].cpu(),
                    fixed.pair_u2d[: e // 2].cpu(), fixed.triplets[:t].cpu())
                self.assertTrue(upstream["consistent"])
                self.assertGreater(len(upstream["angles"]), 0)
                for builder_name, ours in engines.items():
                    self.assertTrue(ours["consistent"], builder_name)
                    keys = ours["edges"] | upstream["edges"]
                    near = (borderline_keys(atoms, keys, spec.cutoff), borderline_keys(atoms, keys, spec.triplet_cutoff))
                    expected, actual = stable_sets(upstream, *near), stable_sets(ours, *near)
                    for kind in ("edges", "pairs", "angles"):
                        self.assertEqual(actual[kind], expected[kind], f"{builder_name}: {kind}")


class CHGNetAccuracyTest(CHGNetTestCase):
    def test_fp64_engine_derivative_matches_upstream_autograd(self):
        """FP64 on the CPU, the same graph on both sides (upstream's converter's): the energy tap plus the
        engine-owned derivative (one autograd.grad through the positions and a symmetric strain) against
        upstream's own derivative: energy, forces and stress relative 1e-12."""
        from fastmd.models.chgnet import CHGNetAdapter

        module = copy.deepcopy(chgnet().module).cpu().double()
        adapter = CHGNetAdapter(module)
        try:
            from pymatgen.io.ase import AseAtomsAdaptor

            atoms = perovskite()
            graph = module.graph_converter(AseAtomsAdaptor.get_structure(atoms))
            frac, lattice = graph.atom_frac_coord.detach().double(), graph.lattice.detach().double()
            batch = GraphBatch(
                positions=frac @ lattice, frac_coords=frac, cell=lattice, species=graph.atomic_number.int(),
                n_real=len(atoms), edge_index=graph.atom_graph.int(), edge_image=graph.neighbor_image.double(),
                n_edges=None, cutoff=6.0, padding=None, triplet_cutoff=3.0,
                pair_d2u=graph.directed2undirected.int(), pair_u2d=graph.undirected2directed.int(),
                triplets=graph.bond_graph.int())
            ours = engine_efs(adapter, batch)
            graph.atom_frac_coord, graph.lattice, graph.neighbor_image = frac, lattice, graph.neighbor_image.double()
            theirs = upstream_efs(module, graph)
            self.assertGreater(float(theirs[1].abs().max()), 0.1)
            self.assert_efs(tuple(v.numpy() if torch.is_tensor(v) else v for v in ours),
                            tuple(v.numpy() if torch.is_tensor(v) else v for v in theirs), FP64_TOLERANCE, "FP64")
        finally:
            adapter.close()

    def test_fp32_engine_matches_upstream_on_the_same_input(self):
        """IEEE FP32 on the GPU, the same input on both sides (the engine's ragged line graph and fractional
        coordinates, the positions frac @ cell): the engine-owned derivative against upstream's own; forces
        within max(1e-6, 1e-5 max|F|, k nu) eV/A, stress within max(1e-9 eV/A^3, 1e-5 max|S|, k nu_S), energy
        1e-6 eV/atom."""
        adapter = chgnet()
        atoms = perovskite()
        with ieee_fp32():
            batch = ragged(atoms, adapter)
            batch = dataclasses.replace(batch, positions=batch.frac_coords @ batch.cell)  # upstream's arithmetic
            graph = crystal_graph(batch, torch.float32)
            engine = [engine_efs(adapter, batch) for _ in range(REPEATS)]
            upstream = [upstream_efs(adapter.module, graph) for _ in range(REPEATS)]
            self.assertEqual(fp32_gemm_mode(), "ieee")
        self.assertGreater(float(upstream[0][1].abs().max()), 0.1)
        for index, name, absolute in ((1, "forces", 1e-6), (2, "stress", 1e-9)):
            floor = largest([max_pairwise([run[index] for run in engine]),
                             max_pairwise([run[index] for run in upstream])])
            error = largest(float((ours[index] - theirs[index]).abs().max()) for ours, theirs in zip(engine, upstream))
            scale = float(upstream[0][index].abs().max())
            bound = largest([absolute, ROUNDING * scale, chgnet().capture_contract().tol_multiplier * floor])
            self.assert_within(error, bound, f"FP32 {name} (floor {floor:.2e})")
        error = largest(abs(ours[0] - theirs[0]) for ours, theirs in zip(engine, upstream)) / len(atoms)
        self.assert_within(error, 1e-6, "FP32 |dE|/N")

    def test_matches_the_chgnet_calculator(self):
        """End to end against upstream's CHGNetCalculator (sanity: it converts the structure itself): GPU-
        resident MD's logged first frame and the calculator (eager), energy 1e-5 eV/atom, forces 5e-4 eV/A,
        stress 1e-6 eV/A^3."""
        from chgnet.model.dynamics import CHGNetCalculator

        adapter = chgnet()
        atoms = perovskite()
        probe = atoms.copy()
        probe.calc = CHGNetCalculator(model=adapter.module, use_device="cuda")
        with ieee_fp32():
            expected = probe.get_potential_energy(), probe.get_forces(), probe.get_stress(voigt=False)
            results = {"calculator": calculator_efs(adapter, atoms)}
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "md.traj")
                with gpu_md(atoms, adapter, "eager", trajectory=path, loginterval=1, log_stress=True) as md:
                    md.run(1)
                frame = read(path, index=0)
            results["GPU MD frame"] = frame.get_potential_energy(), frame.get_forces(), frame.get_stress(voigt=False)
        for name, (energy, forces, stress) in results.items():
            with self.subTest(name):
                self.assert_within(abs(energy - expected[0]) / len(atoms), 1e-5, f"{name}: |dE|/N")
                self.assert_within(float(np.abs(forces - expected[1]).max()), SANITY, f"{name}: max|dF|")
                self.assert_within(float(np.abs(stress - expected[2]).max()), 1e-6, f"{name}: max|dS|")

    def test_padding_adds_exactly_nothing(self):
        """Sink padding: a fixed-capacity build (sink rows, padded edges, pairs and triplets), fresh and on
        its reused candidate list, equals the ragged batch at the same coordinates (forces within
        max(1e-6, 1e-5 max|F|, k nu), energy 1e-6 eV/atom, stress relative 1e-5); a batch whose rows are all
        padding (two atoms farther apart than 6 A) gives exactly zero forces and stress."""
        adapter = chgnet()
        self.assertEqual(adapter.padding_rule().kind, "sink_atoms")
        atoms = perovskite()
        with ieee_fp32():
            batch = ragged(atoms, adapter)
            expected = [engine_efs(adapter, batch) for _ in range(REPEATS)]
            floor = max_pairwise([run[1] for run in expected])
            u, t = batch.pair_u2d.shape[0], batch.triplets.shape[0]
            builder = fixed_capacity_builder(atoms, adapter, u_capacity=u + 64, t_capacity=t + 128)
            positions = torch.tensor(atoms.positions, dtype=torch.float32, device="cuda")
            for reuse in (False, True):
                with self.subTest(candidates="reused" if reuse else "fresh"):
                    inputs, status = builder.build(positions, rebuild_candidates=not reuse)
                    padded = builder.batch
                    self.assertGreater(padded.edge_index.shape[0], int(status.edge_count))
                    self.assertGreater(padded.triplets.shape[0], int(status.triplet_count))
                    self.assertGreater(padded.species.shape[0], padded.n_real)  # sink rows
                    self.assertTrue(torch.equal(padded.positions[: padded.n_real], batch.positions))
                    energy, forces, stress = engine_efs(adapter, padded, inputs)
                    self.assert_within(float((forces - expected[0][1]).abs().max()),
                                       self.tolerance(floor, float(expected[0][1].abs().max())),
                                       f"padded: max|dF| (floor {floor:.2e})")
                    self.assert_within(abs(energy - expected[0][0]) / len(atoms), 1e-6, "padded: |dE|/N")
                    self.assert_within(relative(stress.cpu(), expected[0][2].cpu()), 1e-5, "padded: max|dS|/max|S|")
            apart = Atoms("CsI", positions=[[2.0, 2.0, 2.0], [12.0, 12.0, 12.0]], cell=[20.0] * 3, pbc=True)
            builder = fixed_capacity_builder(apart, adapter, u_capacity=8, t_capacity=16)
            inputs, status = builder.build(torch.tensor(apart.positions, dtype=torch.float32, device="cuda"))
            self.assertEqual((int(status.edge_count), int(status.triplet_count)), (0, 0))
            _, forces, stress = engine_efs(adapter, builder.batch, inputs)
        self.assertTrue(bool((forces == 0).all()) and bool((stress == 0).all()))

    def test_one_evaluation_makes_no_host_synchronization(self):
        """With the patches applied, the energy tap and the engine-owned derivative (with the stress) of a
        padded fixed-capacity batch make no host synchronization (CUDA's sync debug mode "error")."""
        adapter = chgnet()
        self.assertTrue(adapter.capture_contract().sync_free)
        atoms = perovskite()
        builder = fixed_capacity_builder(atoms, adapter, u_capacity=512, t_capacity=1024)
        with ieee_fp32():
            inputs, _ = builder.build(torch.tensor(atoms.positions, dtype=torch.float32, device="cuda"))
            derivative = EnergyDerivative(adapter)
            derivative(builder.batch, inputs, task="efs", n_real=builder.n_real)  # warm (lazy initializations)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                derivative(builder.batch, inputs, task="efs", n_real=builder.n_real)
            finally:
                torch.cuda.set_sync_debug_mode("default")

    def test_isolated_atom_keeps_the_model_energy(self):
        """Isolated-atom policy "model": an atom farther than 6 A from every other gets CHGNet's own energy
        (not its AtomRef weight), with no force; GPU-resident MD (eager, whole_step's padded build) and the
        calculator against upstream's calculator (energy 1e-5 eV/atom, forces 5e-4 eV/A)."""
        from chgnet.model.dynamics import CHGNetCalculator

        adapter = chgnet()
        self.assertEqual(adapter.isolated_atom_policy(), "model")
        atoms = cluster_with_lone_atom()
        lone = len(atoms) - 1
        probe = atoms.copy()
        probe.calc = CHGNetCalculator(model=adapter.module, use_device="cuda")
        with ieee_fp32(), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # upstream warns about the isolated atom
            expected_energy, expected_forces = probe.get_potential_energy(), probe.get_forces()
            batch = ragged(atoms, adapter)
            with torch.no_grad():
                rows = adapter.node_energy(adapter.prepare(batch), batch.positions, batch.cell)
            weight = float(adapter.reference_energies()[0, atoms.numbers[lone] - 1])
            self.assertGreater(abs(float(rows[lone]) - weight), 0.1)
            results = {"calculator": calculator_efs(adapter, atoms)[:2]}
            for execution in ("eager", "whole_step"):
                with gpu_md(atoms, adapter, execution) as md:
                    if execution == "eager":
                        forces, energy = md.evaluate(torch.tensor(atoms.positions, device="cuda"))
                    else:  # the whole-step runner's first evaluation: the padded fixed-capacity build
                        forces, energy = md.state.forces, md.state.potential_energy
                    results[f"GPU MD {execution}"] = (float(energy.reshape(-1)[0]), forces.cpu().numpy())
        for name, (energy, forces) in results.items():
            with self.subTest(name):
                self.assert_within(abs(energy - expected_energy) / len(atoms), 1e-5, f"{name}: |dE|/N")
                self.assert_within(float(np.abs(forces - expected_forces).max()), SANITY, f"{name}: max|dF|")
                self.assertTrue(np.all(forces[lone] == 0.0))


class CHGNetDynamicsTest(CHGNetTestCase):
    def test_execution_modes_agree(self):
        """GPU-resident MD (nvt_nhc) in eager, model_graph and whole_step, and the calculator in model_graph
        (with and without topology reuse): the first evaluation (eager: the ragged batch; model_graph: a
        captured padded bucket; whole_step: the padded fixed-capacity build) equals eager's (forces within
        max(1e-6, 1e-5 max|F|, k nu) eV/A, energy 1e-6 eV/atom); after 20 steps the positions agree within
        1e-4 A and the logged frames' stress within 1e-4 relative."""
        adapter = chgnet()
        atoms = thermalized()
        positions = torch.tensor(atoms.positions, device="cuda")
        steps = 20
        first, frames, final = {}, {}, {}
        with ieee_fp32(), tempfile.TemporaryDirectory() as directory:
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
            for reuse in ("off", "skin"):
                probe = atoms.copy()
                probe.calc = MLIPCalculator(model=adapter, device="cuda", config=InferenceConfig(
                    execution="model_graph", topology_reuse=reuse, report=False))
                first[f"calculator model_graph, reuse {reuse}"] = (probe.get_potential_energy(),
                                                                   torch.as_tensor(probe.get_forces()))
                self.assertEqual(probe.calc.config_summary()["effective"]["execution"], "model_graph")
                self.assertEqual(probe.calc.topology_reuse_summary()["last_call"], reuse)
            self.assertEqual(fp32_gemm_mode(), "ieee")
        floor = max_pairwise(repeats)
        for name, (energy, forces) in first.items():
            if name == "eager":
                continue
            with self.subTest(name):
                self.assert_within(float((forces - first["eager"][1]).abs().max()),
                                   self.tolerance(floor, float(first["eager"][1].abs().max())),
                                   f"FP32 {name}: max|dF| (floor {floor:.2e})")
                self.assert_within(abs(energy - first["eager"][0]) / len(atoms), 1e-6, f"FP32 {name}: |dE|/N")
        for execution in EXECUTIONS[1:]:
            with self.subTest(execution):
                self.assert_within(float(np.abs(final[execution] - final["eager"]).max()), 1e-4,
                                   f"{execution}: max|dx| after {steps} steps (A)")
                self.assertEqual(len(frames[execution]), len(frames["eager"]))
                for index, (ours, theirs) in enumerate(zip(frames[execution], frames["eager"])):
                    self.assert_within(relative(ours.get_stress(voigt=False), theirs.get_stress(voigt=False)), 1e-4,
                                       f"{execution} frame {index}: max|dS|/max|S|")

    def test_execution_auto_captures(self):
        """Execution auto measures CHGNet's memory on its own evaluation with the line graph (a fit with a
        per-triplet cost) and captures it: GPU-resident MD runs the graph path its plan records (whole_step or
        model_graph, by the whole-step trial) and its positions after 20 steps equal eager MD's within 1e-4 A;
        the calculator switches to model_graph at a system's first repeat call as its plan records, and equals
        eager there (forces within max(1e-6, 1e-5 max|F|, k nu), energy 1e-6 eV/atom)."""
        adapter = chgnet()
        atoms = thermalized()
        positions = torch.tensor(atoms.positions, device="cuda")
        steps = 20
        with ieee_fp32():
            with gpu_md(atoms, adapter, "eager") as md:
                expected = [md.evaluate(positions) for _ in range(REPEATS)]
                expected = [(float(energy), forces.double().cpu()) for forces, energy in expected]
                md.run(steps)
                eager_final = md.snapshot_atoms().positions
            with gpu_md(atoms, adapter, "auto") as md:
                plan = md.plan_record()
                execution = plan.decision("execution")
                self.assertIn(execution.chosen, ("whole_step", "model_graph"), execution.reason)
                self.assertEqual(md.config_summary()["effective"]["execution"], execution.chosen)
                memory = plan.as_dict()["memory"]
                self.assertTrue(memory["valid"] and memory["triplets"], memory["reason"])
                md.run(steps)
                self.assert_within(float(np.abs(md.snapshot_atoms().positions - eager_final).max()), 1e-4,
                                   f"GPU MD auto ({execution.chosen}): max|dx| after {steps} steps (A)")
            probe = atoms.copy()
            probe.calc = MLIPCalculator(model=adapter, device="cuda", config=InferenceConfig(report=False))
            for _ in range(2):  # the first call of a system runs eager, its first repeat call decides
                probe.calc.reset()
                energy = probe.get_potential_energy()
            execution = probe.calc.plan_record().decision("execution")
            self.assertEqual(execution.chosen, "model_graph", execution.reason)
            self.assertEqual(probe.calc.config_summary()["effective"]["execution"], "model_graph")
            forces = torch.as_tensor(probe.get_forces()).double()
        floor = max_pairwise([run[1] for run in expected])
        self.assert_within(float((forces - expected[0][1]).abs().max()),
                           self.tolerance(floor, float(expected[0][1].abs().max())),
                           f"calculator auto: max|dF| (floor {floor:.2e})")
        self.assert_within(abs(energy - expected[0][0]) / len(atoms), 1e-6, "calculator auto: |dE|/N")

    def test_skin_reuse_equals_rebuild(self):
        """Whole-step MD (50 steps) on one candidate list (the default 2 A skin) equals a 0.02 A skin (a fresh
        candidate list after almost every step) and eager MD (the graph rebuilt from scratch at every step):
        positions within 1e-4 A (FP32 atomics noise only)."""
        adapter = chgnet()
        atoms = thermalized()
        steps = 50
        positions, stats = {}, {}
        with ieee_fp32():
            for name, execution, skin in (("skin 2 A", "whole_step", 2.0), ("skin 0.02 A", "whole_step", 0.02),
                                          ("rebuild", "eager", 2.0)):
                capacity = CapacityConfig(warmup=1, candidate_skin=skin, **SMALL)
                with gpu_md(atoms, adapter, execution, capacity=capacity) as md:
                    self.assertEqual(md.config_summary()["effective"]["execution"], execution)
                    md.run(steps)
                    positions[name] = md.snapshot_atoms().positions
                    stats[name] = md.whole_step_stats(capacity_plan=False) if execution == "whole_step" else None
        self.assertLess(stats["skin 2 A"]["candidate_refreshes"], 5)
        self.assertGreaterEqual(stats["skin 0.02 A"]["candidate_refreshes"], steps // 2)
        for name in ("skin 0.02 A", "rebuild"):
            with self.subTest(name):
                self.assert_within(float(np.abs(positions[name] - positions["skin 2 A"]).max()), 1e-4,
                                   f"{name}: max|dx| after {steps} steps (A)")


class CHGNetPatchTest(CHGNetTestCase):
    def test_declined_patches_leave_the_adapter_correct(self):
        """With the three synchronization patches declined (records for another chgnet version, upstream as
        installed), the adapter runs upstream's code: not declared sync-free, so whole_step falls back to
        eager with a warning and the plan lists the declined records; the evaluation equals the patched
        adapter's (forces within max(1e-6, 1e-5 max|F|, k nu), energy 1e-6 eV/atom). A declined pin refuses to
        load."""
        from chgnet.model import layers as upstream_layers
        from fastmd.models.chgnet import PATCH_RECORDS
        from fastmd.models.chgnet import SYNC_PATCHES
        from fastmd.models.chgnet import CHGNetAdapter

        patched = chgnet()
        module, atoms = patched.module, perovskite()
        with ieee_fp32():
            batch = ragged(atoms, patched)
            expected = [engine_efs(patched, batch) for _ in range(REPEATS)]
        floor = max_pairwise([run[1] for run in expected])
        bound = self.tolerance(floor, float(expected[0][1].abs().max()))
        # Records are installed process-wide by name: revert the shared adapter's first.
        _LOADED.pop("adapter").close()
        upstream_aggregate = upstream_layers.aggregate
        records = tuple(dataclasses.replace(record, versions=("0",)) if record.name in SYNC_PATCHES else record
                        for record in PATCH_RECORDS)
        declined = CHGNetAdapter(module, records=records)
        try:
            self.assertIs(upstream_layers.aggregate, upstream_aggregate)
            self.assertFalse(declined.capture_contract().sync_free)
            reasons = {o.record.name: o.reason for o in declined.upstream_patches() if not o.applied}
            self.assertEqual(set(reasons), set(SYNC_PATCHES))
            with ieee_fp32():
                energy, forces, _ = engine_efs(declined, batch)
                self.assert_within(float((forces - expected[0][1]).abs().max()), bound,
                                   f"declined: max|dF| (floor {floor:.2e})")
                self.assert_within(abs(energy - expected[0][0]) / len(atoms), 1e-6, "declined: |dE|/N")
                with self.assertWarns(ConfigurationWarning), gpu_md(thermalized(), declined, "whole_step") as md:
                    self.assertEqual(md.config_summary()["effective"]["execution"], "eager")
                    self.assertEqual(set(md.plan_record().patches["declined"]), set(SYNC_PATCHES))
                    md.run(2)
            pins = tuple(dataclasses.replace(record, versions=("0",)) if record.pin else record
                         for record in PATCH_RECORDS)
            with self.assertRaisesRegex(RuntimeError, "relies on"):
                CHGNetAdapter(module, records=pins)
        finally:
            declined.close()
        self.assertIs(upstream_layers.aggregate, upstream_aggregate)


class CHGNetLoadingTest(CHGNetTestCase):
    def test_load_by_name(self):
        """By name: upstream's bundled file with the pinned sha256, an adapter (returned as is by the
        resolvers) with an energy tap and patch records, CHGNet's topology."""
        from fastmd.models.chgnet import CHGNET_CHECKPOINTS

        adapter = chgnet()
        self.assertEqual(adapter.checkpoint["sha256"], CHGNET_CHECKPOINTS[MODEL][2])
        self.assertIsInstance(adapter, EnergyTap)
        self.assertIsInstance(adapter, UpstreamPatches)
        self.assertIs(as_adapter(adapter), adapter)
        self.assertIs(load_adapter(adapter), adapter)
        spec = adapter.topology_spec()
        self.assertEqual((spec.cutoff, spec.num_interactions, spec.triplet_cutoff, spec.undirected_pairs,
                          spec.model_dtype), (6.0, 4, 3.0, True, "float32"))
        self.assertTrue(all(outcome.applied for outcome in adapter.upstream_patches()))

    def test_moved_module(self):
        """An adapter loaded on the CPU runs on the GPU once an entry point moves its module (the entry points
        move a supplied model to their device): GPU-resident MD's first evaluation equals the GPU-loaded
        adapter's (forces within max(1e-6, 1e-5 max|F|, k nu)). Once closed, the adapter is no longer declared
        sync-free."""
        adapter = chgnet()
        moved = load_adapter(MODEL, device="cpu")
        self.assertTrue(moved.capture_contract().sync_free)
        try:
            atoms = perovskite()
            positions = torch.tensor(atoms.positions, device="cuda")
            with ieee_fp32():
                with gpu_md(atoms, adapter, "eager") as md:
                    expected = [md.evaluate(positions)[0].double().cpu() for _ in range(REPEATS)]
                with gpu_md(atoms, moved, "eager") as md:
                    self.assertEqual(next(moved.module.parameters()).device.type, "cuda")
                    forces = md.evaluate(positions)[0].double().cpu()
            floor = max_pairwise(expected)
            self.assert_within(float((forces - expected[0]).abs().max()),
                               self.tolerance(floor, float(expected[0].abs().max())), f"max|dF| (floor {floor:.2e})")
        finally:
            moved.close()
        self.assertFalse(moved.capture_contract().sync_free)

    def test_checkpoint_hash_mismatch_is_rejected(self):
        from fastmd.models.chgnet import CHGNET_CHECKPOINTS

        name, path, _ = CHGNET_CHECKPOINTS[MODEL]
        with mock.patch.dict(CHGNET_CHECKPOINTS, {MODEL: (name, path, "0" * 64)}), \
                self.assertRaisesRegex(ValueError, "sha256"):
            load_adapter(MODEL, device="cuda")


class CHGNetOptionalTest(unittest.TestCase):
    """The CHGNet extra is optional: matris imports without it, a CHGNet load then fails with an
    ImportError, and load_adapter rejects invalid options for a CHGNet name."""

    def test_matris_imports_without_chgnet(self):
        if not CHGNET:  # this module imported fastmd.adapters without it
            with self.assertRaises(ImportError):
                load_adapter(MODEL, device="cpu")
            return
        code = ("import sys\n"
                "sys.modules['chgnet'] = None  # chgnet cannot be imported\n"
                "import fastmd, fastmd.adapters, fastmd.models.chgnet\n"
                "try:\n"
                "    fastmd.adapters.load_adapter('chgnet_0_3_0', device='cpu')\n"
                "except ImportError:\n"
                "    print('ImportError at load time')\n")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        self.assertIn("ImportError at load time", result.stdout)

    def test_invalid_load_options_are_rejected(self):
        for options, keyword in (({"model_dtype": "float64"}, "float32"),
                                 ({"optimization_profile": "legacy"}, "legacy"),
                                 ({"model_path": "chgnet.pth.tar"}, "model_path")):
            with self.subTest(**options), self.assertRaisesRegex(ValueError, keyword):
                load_adapter(MODEL, device="cpu", **options)


if __name__ == "__main__":
    unittest.main()
