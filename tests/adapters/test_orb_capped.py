"""Nearest-K selection, padding, and real-checkpoint ORB regression tests."""
import os
from pathlib import Path
import unittest
import numpy as np
import torch
from fastmd.models.orb_graph import ORBGraphAdapter, nearest_k_mask
from fastmd.adapters.protocol import GraphBatch, PaddingRule, ModelOwnedDerivative, SystemEnergyTap
from fastmd.engine.energy_derivative import engine_derivative


class NearestKTests(unittest.TestCase):
    def select(self, senders, distances, k, n=4):
        senders = torch.tensor(senders, dtype=torch.long)
        vectors = torch.zeros((len(senders), 3))
        vectors[:, 0] = torch.tensor(distances)
        return nearest_k_mask(senders, vectors, torch.arange(len(senders)), n, k, 6.).tolist()

    def test_nearest_per_sender_with_ties_and_cutoff(self):
        self.assertEqual(self.select([2, 0, 2, 0, 2, 0, 3], [4., 2., 1., 2., 3., 6., 12.], 2),
                         [False, True, True, True, True, False, False])

    def test_empty_and_isolated_senders(self):
        self.assertEqual(self.select([], [], 20), [])
        self.assertEqual(self.select([2, 2], [1., 2.], 20), [True, True])

    def test_selection_changes_with_geometry(self):
        self.assertEqual(self.select([0, 0, 0], [1., 2., 3.], 1), [True, False, False])
        self.assertEqual(self.select([0, 0, 0], [3., 2., 1.], 1), [False, False, True])


def graph_batch(atoms, device='cpu', pad=0):
    from ase.neighborlist import neighbor_list
    i, j, images = neighbor_list('ijS', atoms, 6.)
    order = np.argsort(i, kind='stable')
    edges = np.stack([i[order], j[order]], axis=1)
    images = images[order]
    n_edges = len(edges)
    if pad:
        nodes = np.arange(pad) % len(atoms)
        edges = np.concatenate([edges, np.stack([nodes, nodes], axis=1)])
        images = np.concatenate([images, np.tile([4, 0, 0], (pad, 1))])
    return GraphBatch(
        positions=torch.tensor(atoms.positions, dtype=torch.float32, device=device),
        frac_coords=torch.tensor(atoms.get_scaled_positions(), dtype=torch.float32, device=device),
        cell=torch.tensor(atoms.cell.array, dtype=torch.float32, device=device),
        species=torch.tensor(atoms.numbers, device=device), n_real=len(atoms),
        edge_index=torch.tensor(edges, dtype=torch.int32, device=device),
        edge_image=torch.tensor(images, dtype=torch.float32, device=device),
        n_edges=torch.tensor([n_edges], device=device), cutoff=6.,
        padding=PaddingRule('far_self_loops') if pad else None)


def evaluate(adapter, batch):
    inputs = adapter.prepare(batch)
    derivative = engine_derivative(adapter)
    out = derivative(batch, inputs, task='efs') if derivative else adapter.evaluate([inputs], task='efs')
    return {'energy': out['e'].detach().cpu().numpy(),
            'forces': out['f'][0].detach().cpu().numpy(),
            'stress': out['s'][0].detach().cpu().numpy()}


@unittest.skipUnless(os.environ.get('ORB_CHECKPOINT_DIR'), 'set ORB_CHECKPOINT_DIR for local weights')
class CheckpointTests(unittest.TestCase):
    def test_capped_models_and_padding(self):
        from ase import Atoms
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        atoms = Atoms('CsSnI3', cell=[5.5]*3, pbc=True,
                      scaled_positions=[[0,0,0],[.5,.5,.5],[.5,.5,0],[.5,0,.5],[0,.5,.5]]).repeat(2)
        atoms.positions += np.random.default_rng(81).normal(0, .09, atoms.positions.shape)
        from ase.neighborlist import neighbor_list
        self.assertGreater(np.bincount(neighbor_list('i', atoms, 6.)).max(), 20)
        for name in ('orb-d3-xs-v2', 'orb-v3-direct-20-mpa', 'orb-v3-conservative-20-mpa'):
            with self.subTest(checkpoint=name):
                module = pretrained.ORB_PRETRAINED_MODELS[name](
                    weights_path=str(Path(os.environ['ORB_CHECKPOINT_DIR']) / (name + '.ckpt')),
                    device='cpu', precision='float32-highest', compile=False)
                atoms.calc = ORBCalculator(module, device='cpu')
                expected = {'energy': atoms.get_potential_energy(), 'forces': atoms.get_forces(),
                            'stress': atoms.get_stress(voigt=False)}
                adapter = ORBGraphAdapter(module)
                try:
                    conservative = 'conservative' in name
                    self.assertEqual(isinstance(adapter, SystemEnergyTap), conservative)
                    self.assertEqual(isinstance(adapter, ModelOwnedDerivative), not conservative)
                    for pad in (0, 37):
                        actual = evaluate(adapter, graph_batch(atoms, pad=pad))
                        for key, tolerance in [('energy', .001), ('forces', .0001), ('stress', .00001)]:
                            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=tolerance,
                                                       err_msg=f'{name}, pad={pad}, {key}')
                    bad = graph_batch(atoms)
                    bad.cell.zero_()
                    with self.assertRaisesRegex(ValueError, 'periodic cell'):
                        adapter.prepare(bad)
                finally:
                    adapter.close()


if __name__ == '__main__':
    unittest.main()
