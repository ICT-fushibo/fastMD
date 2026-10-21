"""Prophet's actual upstream network: graph orientation, derivatives and padding.

These CPU tests build a small random network and never download checkpoints.
The optional official checkpoint test uses FASTMD_PROPHET_CHECKPOINT only.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import unittest

import numpy as np
import torch
from ase import Atoms

from fastmd.models.prophet import ProphetAdapter, is_prophet_model, load_prophet
from fastmd.adapters.protocol import GraphBatch
from fastmd.engine.energy_derivative import EnergyDerivative


PROPHET = importlib.util.find_spec("prophet") is not None


def graph_for(atoms, config):
    from prophet.graph import atomic_numbers_to_indices, dict_to_pytorch_geometric, preprocess_graph

    upstream = dict_to_pytorch_geometric(preprocess_graph(
        atoms, atomic_numbers_to_indices(config["atomic_numbers"]), max(config["cutoffs"])))
    upstream.n_graph = torch.zeros(len(atoms), dtype=torch.long)
    batch = GraphBatch(positions=upstream.positions.clone(),
                       frac_coords=torch.tensor(atoms.get_scaled_positions(), dtype=torch.float32),
                       cell=upstream.cell[0].clone(), species=torch.tensor(atoms.numbers, dtype=torch.int32),
                       n_real=len(atoms), edge_index=upstream.edge_index.flip(0).T.to(torch.int32),
                       edge_image=upstream.edge_attr.clone(), n_edges=None,
                       cutoff=max(config["cutoffs"]), padding=None)
    return batch, upstream


def reference(module, upstream):
    return module(upstream.x, upstream.positions.clone(), upstream.edge_attr,
                  upstream.edge_index, upstream.cell, upstream.n_node, upstream.n_edge,
                  upstream.n_graph)


class ProphetNamesTest(unittest.TestCase):
    def test_names_and_local_checkpoint_requirement(self):
        for name in ("prophet", "prophet-oame-mbd", "prophet_oame_mbd", "Prophet-OAME-MBD"):
            self.assertTrue(is_prophet_model(name))
        self.assertFalse(is_prophet_model("prophet-v1-mag"))
        self.assertFalse(is_prophet_model(torch.nn.Linear(1, 1)))
        with self.assertRaisesRegex(ValueError, "model_path"):
            load_prophet()
        with self.assertRaisesRegex(ValueError, "float32"):
            load_prophet(model_path="unused", model_dtype="float64")


@unittest.skipUnless(PROPHET, "requires optional prophet-mlip")
class ProphetAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from prophet.model import Prophet

        torch.set_num_threads(2)
        torch.manual_seed(18)
        cls.config = {"atomic_numbers": [1, 6, 8], "cutoffs": [2., 4.], "n_layers": 2}
        cls.module = Prophet(n_species=3, lmax=1, hidden_irreps="4x0e + 4x1o", n_layers=2,
                             radial_basis_size=4, radial_mlp_size=4, radial_mlp_layers=1,
                             cutoffs=[2., 4.], shift=.2, scale=1.1,
                             atom_energies=[-.5, -1.5, -2.5], kernel=False).eval()
        cls.adapter = ProphetAdapter(cls.module, config=cls.config)

    def atoms(self):
        return Atoms("HCO", positions=[[.25, 1., 1.], [7.75, 1.25, 1.], [2., 2., 2.]],
                     cell=[[8., 0., 0.], [1., 7., 0.], [.5, 1., 9.]], pbc=True)

    def evaluate(self, batch):
        return EnergyDerivative(self.adapter)(batch, self.adapter.prepare(batch), task="efs")

    def test_periodic_triclinic_energy_forces_stress_match_upstream(self):
        batch, upstream = graph_for(self.atoms(), self.config)
        self.assertTrue(bool(upstream.edge_attr.ne(0).any()))
        expected_e, expected_f, expected_s = reference(self.module, upstream)
        actual = self.evaluate(batch)
        torch.testing.assert_close(actual["e"][0], expected_e.double().sum(), rtol=0, atol=1e-8)
        torch.testing.assert_close(actual["f"][0], expected_f, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual["s"][0], expected_s[0], rtol=1e-5, atol=1e-8)

    def test_far_padding_preserves_energy_and_derivatives(self):
        batch, _ = graph_for(self.atoms(), self.config)
        expected = self.evaluate(batch)
        loops = torch.arange(12, dtype=torch.int32) % batch.n_real
        padded = dataclasses.replace(batch,
            edge_index=torch.cat([batch.edge_index, torch.stack([loops, loops], dim=1)]),
            edge_image=torch.cat([batch.edge_image, torch.tensor([[3., 0., 0.]]).repeat(12, 1)]),
            n_edges=torch.tensor([len(batch.edge_index)]), padding=self.adapter.padding_rule())
        actual = self.evaluate(padded)
        for key in ("e", "f", "s"):
            torch.testing.assert_close(actual[key][0], expected[key][0], rtol=0, atol=0)

    def test_isolated_atom_is_evaluated_and_has_zero_derivatives(self):
        atoms = Atoms("O", positions=[[2., 2., 2.]], cell=np.eye(3) * 12, pbc=True)
        batch, upstream = graph_for(atoms, self.config)
        self.assertEqual(len(batch.edge_index), 0)
        expected_e, expected_f, expected_s = reference(self.module, upstream)
        actual = self.evaluate(batch)
        torch.testing.assert_close(actual["e"][0], expected_e.double().sum())
        torch.testing.assert_close(actual["f"][0], expected_f, rtol=0, atol=0)
        torch.testing.assert_close(actual["s"][0], expected_s[0], rtol=0, atol=0)
        self.assertTrue(torch.equal(actual["f"][0], torch.zeros_like(actual["f"][0])))

    def test_prepared_batch_reads_updated_edges_and_rejects_unknown_species(self):
        batch, _ = graph_for(self.atoms(), self.config)
        prepared = self.adapter.prepare(batch)
        first = self.adapter.node_energy(prepared, batch.positions, batch.cell)
        batch.edge_image.fill_(5.)
        second = self.adapter.node_energy(prepared, batch.positions, batch.cell)
        self.assertFalse(torch.allclose(first, second))
        for species in (torch.tensor([2]), torch.tensor([100])):
            with self.assertRaisesRegex(ValueError, "species table"):
                self.adapter.prepare(dataclasses.replace(batch, species=species))
        self.assertFalse(self.adapter.capture_contract().sync_free)
        self.assertTrue(self.adapter.capture_contract().exact_cutoff)


@unittest.skipUnless(PROPHET and os.environ.get("FASTMD_PROPHET_CHECKPOINT"),
                     "set FASTMD_PROPHET_CHECKPOINT to test the official weights")
class ProphetCheckpointTest(unittest.TestCase):
    def test_official_checkpoint_matches_ase_calculator_on_cpu(self):
        from prophet import KairosCalculator

        torch.set_num_threads(2)
        path = os.environ["FASTMD_PROPHET_CHECKPOINT"]
        atoms = Atoms("Si2", scaled_positions=[[0, 0, 0], [.25, .25, .25]],
                     cell=[[0, 2.75, 2.75], [2.75, 0, 2.75], [2.75, 2.75, 0]], pbc=True)
        calculator = KairosCalculator(path, use_kernel=False, device="cpu")
        calculator.calculate(atoms)
        adapter = load_prophet(model_path=path, device="cpu")
        batch, _ = graph_for(atoms, adapter.config)
        actual = EnergyDerivative(adapter)(batch, adapter.prepare(batch), task="efs")
        self.assertAlmostEqual(float(actual["e"][0]), float(calculator.results["energy"]), places=5)
        np.testing.assert_allclose(actual["f"][0], calculator.results["forces"], atol=1e-5, rtol=1e-5)
        from ase.stress import full_3x3_to_voigt_6_stress
        np.testing.assert_allclose(full_3x3_to_voigt_6_stress(actual["s"][0]),
                                   calculator.results["stress"], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
