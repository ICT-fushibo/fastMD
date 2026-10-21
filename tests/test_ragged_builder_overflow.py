"""Ragged GPU builder neighbor capacity: exact overflow detection and rebuild.

nvalchemiops counts every neighbor candidate but stores only the first
``max_neighbors`` per atom, so ``num_neighbors.max() > max_neighbors`` detects
truncation exactly. The builders must rebuild with more slots instead of
silently dropping edges, and must be bitwise unchanged when nothing overflows.
Forced small capacities use the ``max_neighbors`` test hook; the dense fixture
overflows the default capacity with no hook at all.
"""
from __future__ import annotations

import itertools
import sys
import unittest
from importlib import metadata
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms

from fastmd.graph import gpu_graph_builder as gb
from fastmd.graph import neighbor_list_backend as backend
from fastmd.graph.gpu_graph_builder import TensorGraphBuilder
from fastmd.graph.gpu_graph_builder import atoms_to_graph_gpu
from fastmd.graph.gpu_graph_builder import neighbor_list_nvidia
from fastmd.graph.gpu_graph_builder import op_available
from fastmd.graph.gpu_graph_builder import tensors_to_graph_gpu

_CUDA = torch.cuda.is_available() and op_available


def _lattice(n_side: int, spacing: float, jitter: float, seed: int) -> Atoms:
    # Power-of-two cubic cells make pos -> frac -> pos exact, so the ragged
    # builder and the distributed builder see bitwise identical coordinates.
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*(np.arange(n_side),) * 3, indexing="ij"), -1)
    positions = (grid.reshape(-1, 3) + 0.5) * spacing
    positions = positions + rng.uniform(-jitter, jitter, positions.shape)
    numbers = rng.choice([1, 6, 7, 8], size=len(positions))
    return Atoms(numbers=numbers, positions=positions.astype(np.float32),
                 cell=np.eye(3) * n_side * spacing, pbc=True)


# 125 atoms, ~30 neighbors within 3 A: far below the default 80 slots.
def _sparse(seed: int = 0) -> Atoms:
    return _lattice(5, 1.6, 0.15, seed)


# 512 atoms, >100 neighbors within 3 A: overflows the default 80 slots.
def _dense(seed: int = 1) -> Atoms:
    return _lattice(8, 1.0, 0.1, seed)


SPARSE_CUTOFFS = (3.0, 2.0)
DENSE_CUTOFFS = (3.0, 1.6)


def _tensors(atoms: Atoms, device: torch.device):
    return (
        torch.tensor(atoms.positions, dtype=torch.float32, device=device),
        torch.tensor(atoms.cell.array, dtype=torch.float32, device=device),
        torch.tensor(atoms.numbers, device=device),
        torch.tensor(atoms.pbc, device=device),
    )


def _builder(atoms: Atoms, cutoffs, device, **kwargs) -> TensorGraphBuilder:
    _, cell, numbers, pbc = _tensors(atoms, device)
    return TensorGraphBuilder(cell=cell, atomic_numbers=numbers, pbc=pbc,
                              atom_graph_cutoff=cutoffs[0],
                              line_graph_cutoff=cutoffs[1], device=device,
                              **kwargs)


def _max_count(atoms: Atoms, cutoff: float, device) -> int:
    positions, cell, _, pbc = _tensors(atoms, device)
    counts = neighbor_list_nvidia(
        positions, cell.unsqueeze(0), cutoff,
        torch.zeros(len(atoms), dtype=torch.int32, device=device),
        pbc.unsqueeze(0), device=device, return_num_neighbors=True,
        max_neighbors=4096,
    )[4]
    return int(counts.max())


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() and tensor.numel():
        return tensor.contiguous().view(torch.int32 if tensor.element_size() == 4
                                        else torch.int64)
    return tensor


def assert_graphs_bitwise(case: unittest.TestCase, actual, expected) -> None:
    actual_fields, expected_fields = vars(actual), vars(expected)
    case.assertEqual(actual_fields.keys(), expected_fields.keys())
    for name, value in expected_fields.items():
        other = actual_fields[name]
        if torch.is_tensor(value):
            case.assertEqual((other.dtype, other.shape, other.device),
                             (value.dtype, value.shape, value.device), name)
            case.assertTrue(torch.equal(_bits(other), _bits(value)), name)
        else:
            case.assertEqual(other, value, name)


def _triclinic(numbers: str, scaled, length: float) -> Atoms:
    shear = np.array([[1.0, 0.0, 0.0], [0.18, 1.0, 0.0], [-0.11, 0.23, 1.0]])
    return Atoms(numbers, scaled_positions=scaled, cell=shear * length, pbc=True)


def _brute_force_pairs(atoms_list, cutoff: float, margin: float):
    """(center, neighbor, image) of every pair within ``cutoff`` in float64 from the float32
    coordinates, with global atom indices; pairs within ``margin`` of the cutoff are returned apart."""
    inside, near, offset = set(), set(), 0
    for atoms in atoms_list:
        positions = atoms.positions.astype(np.float32).astype(np.float64)
        cell = atoms.cell.array.astype(np.float32).astype(np.float64)
        reach = np.ceil(cutoff * np.linalg.norm(np.linalg.inv(cell), axis=0)).astype(int)
        for image in itertools.product(*(range(-r, r + 1) for r in reach)):
            distance = np.linalg.norm(positions[None, :] + np.array(image) @ cell - positions[:, None], axis=-1)
            for i, j in zip(*np.nonzero(distance < cutoff + margin)):
                if i == j and not any(image):
                    continue
                pair = (offset + int(i), offset + int(j), *image)
                (near if abs(distance[i, j] - cutoff) <= margin else inside).add(pair)
        offset += len(atoms)
    return inside, near


def _matrix_pairs(matrix: torch.Tensor, shifts: torch.Tensor, counts: torch.Tensor) -> set:
    slots = matrix.shape[1]
    valid = torch.arange(slots, device=matrix.device)[None, :] < counts.clamp(max=slots)[:, None]
    center = torch.nonzero(valid)[:, 0]
    return set(zip(center.tolist(), matrix[valid].tolist(), *shifts[valid].T.tolist()))


@unittest.skipUnless(_CUDA, "CUDA nvalchemiops graph builder is unavailable")
class RaggedBuilderOverflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda", torch.cuda.current_device())

    def test_status_kernel_and_fallback_agree(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(0)
        for n_atoms in (1, 1000, 65_536, 70_000):  # Triton up to 65,536 atoms.
            counts = torch.randint(0, 900, (n_atoms,), generator=generator,
                                   dtype=torch.int32)
            counts[::7] = 0
            short = torch.randint(0, 40, (n_atoms,), generator=generator)
            pairs = short * (short - 1)
            expected = [int(pairs.sum()), int((counts == 0).sum()),
                        int(counts.sum()), int(counts.max())]
            status = gb._line_graph_status(pairs.to(self.device),
                                           counts.to(self.device))
            self.assertEqual(status.tolist(), expected, n_atoms)

    def test_forced_capacity_is_bitwise_default_and_persists(self) -> None:
        atoms = _sparse()
        positions, cell, numbers, pbc = _tensors(atoms, self.device)
        maximum = _max_count(atoms, SPARSE_CUTOFFS[0], self.device)
        default = _builder(atoms, SPARSE_CUTOFFS, self.device)
        self.assertLessEqual(maximum, default.max_neighbors)
        reference = default.build(positions)
        self.assertGreater(reference.line_graph.shape[0], 0)
        # ASE input also records the chemical formula as the composition.
        atoms_reference = atoms_to_graph_gpu(atoms, *SPARSE_CUTOFFS, self.device)
        # Far too few slots, one slot short of the largest star, and exactly enough.
        for capacity in (1, 8, 16, maximum - 1, maximum):
            with self.subTest(capacity=capacity):
                builder = _builder(atoms, SPARSE_CUTOFFS, self.device,
                                   max_neighbors=capacity)
                assert_graphs_bitwise(self, builder.build(positions), reference)
                # Later builds with the grown capacity stay exact.
                assert_graphs_bitwise(self, builder(positions), reference)

                assert_graphs_bitwise(self, tensors_to_graph_gpu(
                    positions, cell, numbers, pbc, *SPARSE_CUTOFFS, self.device,
                    max_neighbors=capacity), reference)
                assert_graphs_bitwise(self, atoms_to_graph_gpu(
                    atoms, *SPARSE_CUTOFFS, self.device, max_neighbors=capacity),
                    atoms_reference)
        # Without counts the free function checks and reruns by itself.
        batch = torch.zeros(len(atoms), dtype=torch.int32, device=self.device)
        full = neighbor_list_nvidia(
            positions, cell.unsqueeze(0), SPARSE_CUTOFFS[0], batch, pbc.unsqueeze(0),
            device=self.device, max_neighbors=maximum - 1,
        )
        self.assertTrue(torch.equal(
            torch.stack(full[:2], dim=1).int(), reference.atom_graph))
        self.assertTrue(torch.equal(full[2].float(), reference.neighbor_image))
        with self.assertRaises(ValueError):
            _builder(atoms, SPARSE_CUTOFFS, self.device, max_neighbors=0)

    def test_default_capacity_overflow_is_rebuilt(self) -> None:
        atoms = _dense()
        positions, cell, numbers, pbc = _tensors(atoms, self.device)
        builder = _builder(atoms, DENSE_CUTOFFS, self.device)
        default_capacity = builder.max_neighbors
        maximum = _max_count(atoms, DENSE_CUTOFFS[0], self.device)
        self.assertGreater(maximum, default_capacity, "fixture must overflow")
        graph = builder.build(positions)
        roomy = _builder(atoms, DENSE_CUTOFFS, self.device, max_neighbors=4096)
        assert_graphs_bitwise(self, graph, roomy.build(positions))
        assert_graphs_bitwise(self, graph, tensors_to_graph_gpu(
            positions, cell, numbers, pbc, *DENSE_CUTOFFS, self.device))
        assert_graphs_bitwise(self, atoms_to_graph_gpu(
            atoms, *DENSE_CUTOFFS, self.device), atoms_to_graph_gpu(
            atoms, *DENSE_CUTOFFS, self.device, max_neighbors=4096))
        # Every atom's full star is present: degree == candidate count within
        # the cutoff (all neighbors here are well inside or outside 3 A).
        degree = torch.bincount(graph.atom_graph[:, 0].long(), minlength=len(atoms))
        self.assertEqual(int(degree.max()), maximum)

    def test_batched_path_rebuilds(self) -> None:
        batch = [_sparse(2), _dense(3), _sparse(4)]
        cutoffs = DENSE_CUTOFFS
        reference = atoms_to_graph_gpu(batch, *cutoffs, self.device,
                                       max_neighbors=4096)
        for capacity in (None, 8):
            graphs = atoms_to_graph_gpu(batch, *cutoffs, self.device,
                                        max_neighbors=capacity)
            self.assertEqual(len(graphs), len(reference))
            for actual, expected in zip(graphs, reference):
                assert_graphs_bitwise(self, actual, expected)

    def test_no_short_edge_branch_checks_capacity(self) -> None:
        # A tiny line cutoff leaves no short edges: the capacity check rides on
        # the exact isolated-count read of that branch instead of the status.
        atoms = _sparse()
        positions = _tensors(atoms, self.device)[0]
        cutoffs = (SPARSE_CUTOFFS[0], 0.05)
        reference = _builder(atoms, cutoffs, self.device).build(positions)
        self.assertEqual(reference.line_graph.shape[0], 0)
        builder = _builder(atoms, cutoffs, self.device, max_neighbors=4)
        assert_graphs_bitwise(self, builder.build(positions), reference)

    def test_isolated_atoms_with_small_capacity(self) -> None:
        # The connected atoms overflow one slot; the rebuild precedes the
        # isolated-atom verdict, which must still count only the far atom.
        atoms = Atoms("H4", positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.8, 0],
                                       [8, 8, 8]], cell=[20] * 3, pbc=True)
        with self.assertRaises(ValueError):
            atoms_to_graph_gpu(atoms, 2.0, 1.5, self.device, max_neighbors=1)
        graph = atoms_to_graph_gpu(atoms, 2.0, 1.5, self.device,
                                   check_isolated_atoms=False, max_neighbors=1)
        self.assertEqual(int(graph.isolated_atom_count.item()), 1)
        lone = Atoms("H", positions=[[0, 0, 0]], cell=[20] * 3, pbc=True)
        with self.assertRaises(ValueError):
            atoms_to_graph_gpu(lone, 2.0, 1.5, self.device, max_neighbors=1)
        empty = atoms_to_graph_gpu(lone, 2.0, 1.5, self.device,
                                   check_isolated_atoms=False, max_neighbors=1)
        self.assertEqual(tuple(empty.atom_graph.shape), (0, 2))


    def test_neighbor_list_backend_pair_set(self) -> None:
        # Whichever nvalchemiops API the backend resolved (the suite runs with each): the batch cell
        # list and the fixed-capacity builder's primitives find exactly the brute-force pairs, across
        # periodic images (a 4.1 A cell holds images of its own atoms) and systems of one batch, and a
        # truncated row still counts all of its pairs.
        import warp as wp

        rng = np.random.default_rng(5)
        systems = [_triclinic("C24", rng.random((24, 3)), 9.0),
                   _triclinic("CO", [[0.1, 0.2, 0.3], [0.6, 0.45, 0.8]], 4.1),
                   _triclinic("N40", rng.random((40, 3)), 11.5)]
        cutoff, slots = 4.5, 512
        payload = gb._extract_atoms(systems, self.device)
        positions, cells, pbc, batch = (payload[key] for key in ("positions", "cells", "pbc", "batch_idx"))
        # _extract_atoms wraps coordinates; the brute force uses the same cell, so images differ only
        # by the wrap: compare on the wrapped coordinates.
        wrapped = [Atoms(s.numbers, positions=positions[a:b].cpu().numpy(), cell=s.cell, pbc=True)
                   for s, a, b in zip(systems, payload["atom_offsets"][:-1], payload["atom_offsets"][1:])]
        expected, near = _brute_force_pairs(wrapped, cutoff, margin=1e-4)
        n_atoms = positions.shape[0]

        def buffers(width):
            return (torch.full((n_atoms, width), n_atoms, dtype=torch.int32, device=self.device),
                    torch.zeros((n_atoms, width, 3), dtype=torch.int32, device=self.device),
                    torch.zeros(n_atoms, dtype=torch.int32, device=self.device))

        matrix, shifts, counts = buffers(slots)
        backend.neighbor_list(positions=positions, cutoff=cutoff, cell=cells, pbc=pbc, batch_idx=batch,
                              method="batch_cell_list", neighbor_matrix=matrix, neighbor_matrix_shifts=shifts,
                              num_neighbors=counts, half_fill=False)
        listed = _matrix_pairs(matrix, shifts, counts)
        self.assertLessEqual(int(counts.max()), slots)
        self.assertEqual(listed - near, expected, backend.describe())

        total, radius = backend.estimate_batch_cell_list_sizes(cells, pbc, cutoff)
        cell_list = backend.allocate_cell_list(n_atoms, total, radius, self.device)
        primitive = buffers(slots)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            backend.batch_build_cell_list(positions, cutoff, cells, pbc, batch, *cell_list)
            backend.batch_query_cell_list(positions, cells, pbc, cutoff, batch, *cell_list, *primitive, False)
        self.assertEqual(_matrix_pairs(*primitive), listed)

        truncated = buffers(4)
        backend.neighbor_list(positions=positions, cutoff=cutoff, cell=cells, pbc=pbc, batch_idx=batch,
                              method="batch_cell_list", neighbor_matrix=truncated[0],
                              neighbor_matrix_shifts=truncated[1], num_neighbors=truncated[2])
        self.assertTrue(torch.equal(truncated[2], counts))
        self.assertLessEqual(_matrix_pairs(*truncated), listed)

        # A fixed-cell caller's cell list, sized once (current API) and used twice.
        fixed = backend.cell_list_for_fixed_cell(cells, pbc, cutoff, n_atoms, self.device)
        self.assertEqual(fixed is None, backend.API != "torch")
        for _ in range(2 if fixed is not None else 0):
            matrix, shifts, counts = buffers(slots)
            backend.neighbor_list(positions=positions, cutoff=cutoff, cell=cells, pbc=pbc, batch_idx=batch,
                                  method="batch_cell_list", neighbor_matrix=matrix, neighbor_matrix_shifts=shifts,
                                  num_neighbors=counts, half_fill=False, cell_list=fixed)
            self.assertEqual(_matrix_pairs(matrix, shifts, counts), listed)


class NeighborListBackendTest(unittest.TestCase):
    def test_resolved_backend_is_reported(self) -> None:
        record = backend.describe()
        self.assertEqual(record["api"], backend.API)
        self.assertIn(record["api"], ("torch", "neighborlist", None))
        try:
            installed = metadata.version("nvalchemi-toolkit-ops")
        except metadata.PackageNotFoundError:
            installed = None
        self.assertEqual(record["version"], installed)
        self.assertEqual(record["unavailable_reason"] is None, backend.available)

    def test_max_neighbor_estimate_is_the_old_api_estimate(self) -> None:
        # nvalchemiops 0.2: 0.35 atoms/A^3 x safety factor x sphere volume, up to a multiple of 16.
        for cutoff, expected in ((6.0 + 1e-6, 640), (5.0, 368), (3.0, 80), (0.0, 0)):
            self.assertEqual(backend.estimate_max_neighbors(cutoff=cutoff, safety_factor=2.0), expected)
        if backend.API == "neighborlist":
            from nvalchemiops.neighborlist.neighbor_utils import estimate_max_neighbors
            for cutoff in np.linspace(0.5, 12.0, 200):
                self.assertEqual(backend.estimate_max_neighbors(cutoff=float(cutoff), safety_factor=2.0),
                                 estimate_max_neighbors(cutoff=float(cutoff), safety_factor=2.0))

    def test_unavailable_backend_names_the_version(self) -> None:
        modules = [name for name in sys.modules if name.split(".")[0] == "nvalchemiops"]
        modules += ["nvalchemiops", "nvalchemiops.torch.neighbors.batch_cell_list",
                    "nvalchemiops.neighborlist.batch_cell_list"]
        no_metadata = patch.object(backend.metadata, "version", side_effect=metadata.PackageNotFoundError)
        if backend.available:
            # Package metadata is optional: the module's own version still selects the API.
            with no_metadata:
                self.assertEqual(backend.resolve()[0], backend.API)
        with patch.dict(sys.modules, dict.fromkeys(modules)):
            api, functions, reason = backend.resolve()
            self.assertEqual((api, functions), (None, {}))
            if backend.VERSION is not None:
                self.assertIn(f"nvalchemi-toolkit-ops {backend.VERSION}", reason)
            self.assertIn(backend.SUPPORTED_VERSIONS, reason)
            with no_metadata:
                api, _, reason = backend.resolve()
        self.assertIsNone(api)
        self.assertIn("nvalchemi-toolkit-ops is not installed", reason)


if __name__ == "__main__":
    unittest.main()
