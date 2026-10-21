"""Per-instance installation/reversion and empty/polar eSEN energy derivatives."""
import copy
import importlib.metadata
import os
import unittest

import torch

try:
    LEGACY_ESEN = importlib.metadata.version('fairchem-core') == '1.10.0'
except importlib.metadata.PackageNotFoundError:
    LEGACY_ESEN = False


@unittest.skipUnless(LEGACY_ESEN, 'requires optional fairchem-core==1.10.0')
class OperatorLifecycle(unittest.TestCase):
    def test_instance_scope_reversion_empty_and_polar_gradients(self):
        from fairchem.core.models.esen.esen import eSEN_Backbone, MLP_EFS_Head
        from fairchem.core.models.esen.nn.so2_layers import SO2_Convolution
        from fastmd.models.esen import ESENAdapter
        from fastmd.adapters.protocol import GraphBatch
        from fastmd.engine.energy_derivative import EnergyDerivative

        device = os.environ.get('FASTMD_TEST_DEVICE', 'cpu')
        torch.manual_seed(28)
        torch.set_num_threads(2)
        module = torch.nn.Module()
        module.backbone = eSEN_Backbone(
            sphere_channels=8, hidden_channels=8, edge_channels=8, num_layers=2,
            lmax=2, mmax=1, regress_forces=True, regress_stress=True, cutoff=6.,
            direct_forces=False, use_envelope=True, act_type='gate', mlp_type='spectral')
        module.output_heads = torch.nn.ModuleDict({'head': MLP_EFS_Head(module.backbone)})
        module.to(device)
        state = {name: value.clone() for name, value in module.state_dict().items()}
        original_forward = SO2_Convolution.forward

        def adapter(model):
            normalizers = {key: dict(mean=torch.tensor(0.), rmsd=torch.tensor(1.))
                           for key in ('energy', 'forces', 'stress')}
            return ESENAdapter(model, normalizers=normalizers, reference=torch.zeros(119))

        baseline = adapter(copy.deepcopy(module))
        self.addCleanup(baseline.close)

        def graph(empty, padded):
            # The y axis exercises eSEN's Euler-angle pole branch.
            positions = torch.tensor([[1., 1., 1.], [1., 12. if empty else 2.2, 1.]], device=device)
            edges = [] if empty else [[0, 1], [1, 0]]
            images = [[0., 0., 0.]] * len(edges)
            count = len(edges)
            if padded:
                edges += [[0, 0], [1, 1]] * 3
                images += [[1., 0., 0.]] * 6
            return GraphBatch(
                positions=positions, frac_coords=positions/30,
                cell=torch.eye(3, device=device)*30,
                species=torch.tensor([1, 6], dtype=torch.int32, device=device), n_real=2,
                edge_index=torch.tensor(edges, dtype=torch.int32, device=device).reshape(-1, 2),
                edge_image=torch.tensor(images, device=device).reshape(-1, 3),
                n_edges=torch.tensor([count], device=device), cutoff=6.,
                padding=baseline.padding_rule() if padded else None)

        for backend in ('torch', 'triton'):
            candidate = adapter(copy.deepcopy(module))
            self.addCleanup(candidate.close)
            with self.assertRaises(ValueError):
                candidate.enable_operator_rewrites('invalid')
            candidate.enable_operator_rewrites(backend)
            self.assertIs(SO2_Convolution.forward, original_forward)
            self.assertIs(next(m for m in baseline.module.modules()
                               if isinstance(m, SO2_Convolution)).forward.__func__, original_forward)
            for empty in (False, True):
                for padded in (False, True):
                    with self.subTest(backend=backend, empty=empty, padded=padded):
                        batch = graph(empty, padded)
                        ref = EnergyDerivative(baseline)(batch, baseline.prepare(batch), task='efs')
                        out = EnergyDerivative(candidate)(batch, candidate.prepare(batch), task='efs')
                        for key in ('e', 'f', 's'):
                            expected = ref[key][0] if key in ('f', 's') else ref[key]
                            actual = out[key][0] if key in ('f', 's') else out[key]
                            self.assertTrue(bool(actual.isfinite().all()))
                            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            with self.assertRaises(RuntimeError):
                candidate.enable_operator_rewrites(backend)
            for name, value in candidate.module.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
            candidate.close()
            self.assertIs(next(m for m in candidate.module.modules()
                               if isinstance(m, SO2_Convolution)).forward.__func__, original_forward)
            self.assertFalse(candidate.capture_contract().sync_free)


if __name__ == '__main__':
    unittest.main()
