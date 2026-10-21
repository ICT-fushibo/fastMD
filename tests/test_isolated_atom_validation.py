from __future__ import annotations

import unittest

import torch
from ase import Atoms

from fastmd.graph import FixedCapacityGraphBuilder
from fastmd.graph import atoms_to_graph_gpu
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.gpu_graph_builder import op_available
from fastmd.graph.validation import raise_if_graph_has_isolated_atoms
from fastmd.graph.validation import raise_if_isolated_atoms

from test_isolated_atom_handling import make_graph


def _isolated_atoms() -> Atoms:
    return Atoms(
        "H3",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [8.0, 8.0, 8.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )


class IsolatedAtomValidationTest(unittest.TestCase):
    def test_isolated_atom_error_contract(self) -> None:
        # The one wording check of the isolated-atom error; other tests match
        # only "isolated atom".
        for name, validate, connected, isolated in (
            ("count", raise_if_isolated_atoms, 0, torch.tensor([1], dtype=torch.int32)),
            (
                "graph",
                raise_if_graph_has_isolated_atoms,
                make_graph(2, connected_pair=True),
                make_graph(3, connected_pair=True),
            ),
        ):
            with self.subTest(input=name):
                validate(connected)
                with self.assertRaisesRegex(ValueError, "Detected 1 isolated atom"):
                    validate(isolated)

    @unittest.skipUnless(
        torch.cuda.is_available() and op_available,
        "CUDA nvalchemiops graph builder is unavailable",
    )
    def test_gpu_graph_builder_supports_immediate_and_deferred_validation(
        self,
    ) -> None:
        atoms = _isolated_atoms()
        with self.assertRaisesRegex(ValueError, "isolated atom"):
            atoms_to_graph_gpu(
                atoms,
                atom_graph_cutoff=2.0,
                line_graph_cutoff=1.5,
            )

        graph = atoms_to_graph_gpu(
            atoms,
            atom_graph_cutoff=2.0,
            line_graph_cutoff=1.5,
            check_isolated_atoms=False,
        )
        self.assertEqual(int(graph.isolated_atom_count.item()), 1)
        self.assertEqual(int(graph.to("cpu").isolated_atom_count.item()), 1)
        with self.assertRaisesRegex(ValueError, "isolated atom"):
            raise_if_graph_has_isolated_atoms(graph)

        all_isolated = Atoms(
            "H",
            positions=[[0.0, 0.0, 0.0]],
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )
        with self.assertRaisesRegex(ValueError, "isolated atom"):
            atoms_to_graph_gpu(
                all_isolated,
                atom_graph_cutoff=2.0,
                line_graph_cutoff=1.5,
            )
        empty_graph = atoms_to_graph_gpu(
            all_isolated,
            atom_graph_cutoff=2.0,
            line_graph_cutoff=1.5,
            check_isolated_atoms=False,
        )
        self.assertEqual(tuple(empty_graph.atom_graph.shape), (0, 2))
        self.assertEqual(tuple(empty_graph.line_graph.shape), (0, 5))
        self.assertEqual(empty_graph.directed2undirected.numel(), 0)
        self.assertEqual(empty_graph.undirected2directed.numel(), 0)
        self.assertEqual(int(empty_graph.isolated_atom_count.item()), 1)

        connected = Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )
        graphs = atoms_to_graph_gpu(
            [connected, atoms],
            atom_graph_cutoff=2.0,
            line_graph_cutoff=1.5,
            check_isolated_atoms=False,
        )
        self.assertEqual(
            [int(graph.isolated_atom_count.item()) for graph in graphs],
            [0, 1],
        )
        with self.assertRaisesRegex(ValueError, "isolated atom"):
            raise_if_graph_has_isolated_atoms(graphs)

        cutoff_boundary = Atoms(
            "H3",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )
        with self.assertRaisesRegex(ValueError, "isolated atom"):
            atoms_to_graph_gpu(
                cutoff_boundary,
                atom_graph_cutoff=2.0,
                line_graph_cutoff=1.5,
            )
        boundary_graph = atoms_to_graph_gpu(
            cutoff_boundary,
            atom_graph_cutoff=2.0,
            line_graph_cutoff=1.5,
            check_isolated_atoms=False,
        )
        self.assertEqual(int(boundary_graph.isolated_atom_count.item()), 1)

    @unittest.skipUnless(
        torch.cuda.is_available() and _fixed_builder_available,
        "CUDA fixed-capacity graph builder is unavailable",
    )
    def test_fixed_capacity_builder_reports_isolated_atoms(self) -> None:
        atoms = _isolated_atoms()
        builder = FixedCapacityGraphBuilder(
            cell=torch.diag(torch.tensor(atoms.cell.lengths(), dtype=torch.float32)),
            atomic_numbers=torch.tensor(atoms.numbers, dtype=torch.int32),
            pbc=torch.ones(3, dtype=torch.bool),
            u_capacity=16,
            t_capacity=32,
            atom_graph_cutoff=2.0,
            line_graph_cutoff=1.5,
            candidate_cutoff=2.5,
            n_dummy=4,
            device="cuda",
        )
        positions = torch.tensor(
            atoms.positions,
            dtype=torch.float32,
            device="cuda",
        )
        _, status = builder.build(positions)
        self.assertEqual(int(status.isolated_atom_count.item()), 1)


if __name__ == "__main__":
    unittest.main()
