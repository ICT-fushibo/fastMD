from __future__ import annotations

import itertools
import unittest
from unittest.mock import patch

import torch
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from fastmd.execution.cuda_graph import pad_radius_graph
from fastmd.models.matris.converter import GraphConverter
from fastmd.graph.radiusgraph import RadiusGraph
from fastmd.models.matris.model import MatRIS


def make_graph(n_atoms: int, connected_pair: bool) -> RadiusGraph:
    frac_coords = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.50, 0.50, 0.50],
        ],
        dtype=torch.float32,
    )[:n_atoms].clone()

    if connected_pair:
        atom_graph = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        directed2undirected = torch.tensor([0, 0], dtype=torch.int32)
        undirected2directed = torch.tensor([0], dtype=torch.int32)
    else:
        atom_graph = torch.empty((0, 2), dtype=torch.int32)
        directed2undirected = torch.empty((0,), dtype=torch.int32)
        undirected2directed = torch.empty((0,), dtype=torch.int32)

    return RadiusGraph(
        graph_id=None,
        mp_id=None,
        composition=f"H{n_atoms}",
        atomic_number=torch.ones(n_atoms, dtype=torch.int32),
        atom_frac_coord=frac_coords,
        lattice=torch.eye(3, dtype=torch.float32) * 20.0,
        neighbor_image=torch.zeros((atom_graph.shape[0], 3), dtype=torch.float32),
        atom_graph=atom_graph,
        atom_graph_cutoff=2.0,
        line_graph=torch.empty((0, 5), dtype=torch.int32),
        line_graph_cutoff=1.5,
        directed2undirected=directed2undirected,
        undirected2directed=undirected2directed,
        atom_target_sorted=True,
        line_atom_sorted=True,
    )


def make_model(
    *,
    is_conservation: bool = True,
    reference_energy: str | None = "demo",
    num_layers: int = 1,
    norm_type: str = "rms",
    is_intensive: bool = True,
) -> MatRIS:
    model = MatRIS(
        num_layers=num_layers,
        norm_type=norm_type,
        node_feat_dim=8,
        edge_feat_dim=8,
        three_body_feat_dim=8,
        mlp_hidden_dims=(8,),
        num_radial=2,
        num_angular=2,
        max_n=1,
        max_l=1,
        pairwise_cutoff=2.0,
        three_body_cutoff=1.5,
        reference_energy=reference_energy,
        is_conservation=is_conservation,
        is_intensive=is_intensive,
        enable_compile=False,
        enable_checkpoint=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class IsolatedAtomHandlingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.model = make_model()

    def test_partial_isolated_atom_matches_active_subgraph(self) -> None:
        active = self.model(
            [make_graph(2, connected_pair=True)],
            task="efsm",
            is_training=False,
        )
        partial = self.model(
            [make_graph(3, connected_pair=True)],
            task="efsm",
            is_training=False,
        )

        h_ref = self.model.reference_energy.fc.weight[0, 0].detach()
        active_total = active["e"][0] * 2
        partial_total = partial["e"][0] * 3

        torch.testing.assert_close(
            partial_total,
            active_total + h_ref,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            partial["ref_energy"][0] * 3,
            active["ref_energy"][0] * 2 + h_ref,
        )
        torch.testing.assert_close(
            partial["f"][0][:2],
            active["f"][0],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(partial["f"][0][2], torch.zeros(3))
        torch.testing.assert_close(partial["s"][0], active["s"][0])
        torch.testing.assert_close(
            partial["m"][0][:2],
            active["m"][0],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(partial["m"][0][2], torch.tensor(0.0))
        self.assertEqual(int(partial["atoms_per_graph"][0]), 3)

    def test_all_isolated_returns_reference_only(self) -> None:
        output = self.model(
            [make_graph(1, connected_pair=False)],
            task="efsm",
            is_training=False,
        )
        h_ref = self.model.reference_energy.fc.weight[0, 0].detach()

        torch.testing.assert_close(output["e"][0], h_ref)
        torch.testing.assert_close(output["ref_energy"][0], h_ref)
        torch.testing.assert_close(output["f"][0], torch.zeros((1, 3)))
        torch.testing.assert_close(output["s"][0], torch.zeros((3, 3)))
        torch.testing.assert_close(output["m"][0], torch.zeros(1))

    def test_missing_reference_table_does_not_silently_drop_energy(self) -> None:
        torch.manual_seed(2)
        model = make_model(reference_energy=None)
        graph = make_graph(3, connected_pair=True)
        requested = model(
            [graph],
            task="em",
            is_training=False,
            handle_isolated_atoms=True,
        )
        unhandled = model(
            [graph],
            task="em",
            is_training=False,
            handle_isolated_atoms=False,
        )

        torch.testing.assert_close(requested["e"], unhandled["e"])
        torch.testing.assert_close(requested["m"][0], unhandled["m"][0])

    def test_direct_force_and_stress_exclude_isolated_atoms(self) -> None:
        torch.manual_seed(1)
        direct_model = make_model(is_conservation=False)
        active = direct_model(
            [make_graph(2, connected_pair=True)],
            task="efs",
            is_training=False,
        )
        partial = direct_model(
            [make_graph(3, connected_pair=True)],
            task="efs",
            is_training=False,
        )

        torch.testing.assert_close(partial["f"][0][:2], active["f"][0])
        torch.testing.assert_close(partial["f"][0][2], torch.zeros(3))
        torch.testing.assert_close(partial["s"][0], active["s"][0])
        # The direct force head reads the last block's edge features: its bond update still runs,
        # while a conservative model skips it (MATRIS_SKIP_DEAD_THREEBODY_TAIL).
        for model, expected in ((direct_model, 1), (self.model, 0)):
            calls = []
            hook = model.interaction_block[-1].refine_block_atom_graph.edge_FFN.register_forward_hook(
                lambda *args: calls.append(1))
            try:
                model([make_graph(3, connected_pair=True)], task="ef", is_training=False)
            finally:
                hook.remove()
            self.assertEqual(len(calls), expected)

    def test_handling_can_be_disabled_explicitly(self) -> None:
        graph = make_graph(3, connected_pair=True)
        handled = self.model([graph], task="em", is_training=False)
        unhandled = self.model(
            [graph], task="em", is_training=False, handle_isolated_atoms=False
        )
        self.assertEqual(float(handled["m"][0][2].detach()), 0.0)
        self.assertFalse(torch.equal(handled["e"], unhandled["e"]))

    def test_cpu_converter_accepts_empty_graph_only_when_requested(self) -> None:
        with patch("fastmd.models.matris.converter.op_available", False):
            converter = GraphConverter(atom_graph_cutoff=2.0, line_graph_cutoff=1.5)
            for positions, isolated_count in (
                ([[0, 0, 0], [1, 0, 0], [8, 8, 8]], 1),
                ([[0, 0, 0], [8, 0, 0], [0, 8, 0]], 3),
            ):
                with self.subTest(isolated_count=isolated_count):
                    atoms = Atoms("H3", positions=positions, cell=[20] * 3, pbc=True)
                    structure = AseAtomsAdaptor.get_structure(atoms)
                    with self.assertRaisesRegex(ValueError, "isolated atom"):
                        converter(structure, check_isolated_atoms=True)
                    graph = converter(structure, check_isolated_atoms=False)
                    self.assertEqual(int(graph.isolated_atom_count), isolated_count)
                    self.assertEqual(tuple(graph.line_graph.shape), (0, 5))
                    if isolated_count == 3:
                        self.assertEqual(tuple(graph.atom_graph.shape), (0, 2))
                        self.assertEqual(tuple(graph.neighbor_image.shape), (0, 3))
                    result = self.model([graph], task="efsm", is_training=False)
                    self.assertTrue(bool(torch.isfinite(result["e"]).all()))
                    self.assertEqual(tuple(result["f"][0].shape), (3, 3))

    def test_all_isolated_training_backward_remains_finite(self) -> None:
        for conservation in (True, False):
            with self.subTest(conservation=conservation):
                model = make_model(is_conservation=conservation)
                for name, parameter in model.named_parameters():
                    if not name.startswith("reference_energy."):
                        parameter.requires_grad_(True)
                output = model([make_graph(3, False)], task="efsm", is_training=True)
                loss = output["e"].sum()
                for key in ("f", "s", "m"):
                    loss = loss + output[key][0].square().sum()
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(
                    all(bool(torch.isfinite(grad).all()) for grad in gradients)
                )

    def test_batching_does_not_change_per_graph_predictions(self) -> None:
        # Every graph of a batch keeps its own isolation mask and three-body
        # interactions: partial, all-isolated and triplet-bearing graphs.
        layer_model = make_model(num_layers=2, norm_type="layer")
        with patch("fastmd.models.matris.converter.op_available", False):
            converter = GraphConverter(atom_graph_cutoff=2.0, line_graph_cutoff=1.5)
            triangle = Atoms(
                "H3",
                positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.7, 0]],
                cell=[20] * 3,
                pbc=True,
            )
            triangle_graph = converter(AseAtomsAdaptor.get_structure(triangle))
        self.assertGreater(triangle_graph.line_graph.shape[0], 0)
        for name, model, graphs, task, tolerance in (
            (
                "partial_and_connected",
                self.model,
                [make_graph(3, True), make_graph(2, True)],
                "efm",
                {},
            ),
            (
                "all_isolated_and_connected",
                self.model,
                [make_graph(1, False), make_graph(2, True), make_graph(3, False)],
                "efsm",
                {},
            ),
            (
                "three_body",
                layer_model,
                [
                    make_graph(2, True),
                    triangle_graph,
                    make_graph(3, True),
                    make_graph(1, False),
                ],
                "efsm",
                {"rtol": 2e-5, "atol": 1e-6},
            ),
        ):
            with self.subTest(batch=name):
                keys = ("e", "ref_energy", "f", "m")
                if "s" in task:
                    keys += ("s",)
                output = model(graphs, task=task, is_training=False)
                for index, graph in enumerate(graphs):
                    expected = model([graph], task=task, is_training=False)
                    for key in keys:
                        torch.testing.assert_close(
                            output[key][index], expected[key][0], **tolerance
                        )

    def test_padding_uses_each_graph_real_count(self) -> None:
        # Padded graphs match the unpadded ones on their real atoms, both for
        # a single graph with a scalar n_real and for a heterogeneous batch
        # with per-graph counts; padding must not activate absent three-body
        # interactions.
        # Every pairing of the model's and the reference table's intensive flags:
        # the per-atom readout when they agree, AtomRef added when they differ
        # (the MPtrj/OMat/MPA tables are intensive on any model); ref_energy
        # matches AtomRef either way.
        model = make_model(num_layers=2, norm_type="layer")
        for (graphs, per_graph_counts), intensive, table_intensive in itertools.product((
            ([make_graph(2, True)], False),
            ([make_graph(3, True)], False),
            ([make_graph(1, False), make_graph(2, True), make_graph(3, True)], True),
        ), (True, False), (True, False)):
            model.is_intensive, model.reference_energy.is_intensive = intensive, table_intensive
            with self.subTest(atoms=[len(graph.atomic_number) for graph in graphs], intensive=intensive,
                              table_intensive=table_intensive):
                padded = [pad_radius_graph(graph, 8, 32, n_dummy=4) for graph in graphs]
                real_counts = [int(count) for _, count in padded]
                n_real = (
                    torch.tensor(real_counts, dtype=torch.int32)
                    if per_graph_counts
                    else padded[0][1]
                )
                output = model(
                    [graph for graph, _ in padded],
                    task="efsm",
                    is_training=False,
                    n_real=n_real,
                )
                torch.testing.assert_close(
                    output["ref_energy"],
                    model.reference_energy([graph for graph, _ in padded], n_real=torch.tensor(real_counts)))
                for index, graph in enumerate(graphs):
                    expected = model([graph], task="efsm", is_training=False)
                    for key in ("e", "ref_energy", "s"):
                        torch.testing.assert_close(output[key][index], expected[key][0])
                    for key in ("f", "m"):
                        torch.testing.assert_close(
                            output[key][index][: real_counts[index]], expected[key][0]
                        )


if __name__ == "__main__":
    unittest.main()
