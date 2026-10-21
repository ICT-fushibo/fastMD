"""Conservative eSEN rotations: parity at poles and away from poles."""
import importlib.metadata
from pathlib import Path
import unittest

import torch

try:
    LEGACY_ESEN = importlib.metadata.version('fairchem-core') == '1.10.0'
except importlib.metadata.PackageNotFoundError:
    LEGACY_ESEN = False


@unittest.skipUnless(LEGACY_ESEN, 'requires optional fairchem-core==1.10.0')
class ESENRotationTest(unittest.TestCase):
    def test_rotation_values_and_derivatives(self):
        import fairchem.core.models.esen as upstream
        from fairchem.core.models.esen.common.rotation import init_edge_rot_mat, rotation_to_wigner
        from fastmd.models.esen import _dense_wigner, _edge_rotation
        torch.set_num_threads(2)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                jd = [j.to(dtype) for j in torch.load(Path(upstream.__file__).parent/'Jd.pt', weights_only=True)]
                torch.manual_seed(10)
                vectors = torch.cat((torch.randn(60,3,dtype=dtype), torch.tensor([
                    [0.,1.,0.], [0.,-1.,0.], [1e-5,1.,0.], [1.,0.,0.], [0.,0.,1.]
                ],dtype=dtype))).requires_grad_(True)
                y = vectors.new_tensor([0.,1.,0.])
                safe = vectors.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
                torch.manual_seed(42)
                native = init_edge_rot_mat(vectors,rot_clip=True)
                torch.manual_seed(42)
                actual = _edge_rotation(vectors,y)
                torch.testing.assert_close(actual,native,rtol=0,atol=0)
                expected = rotation_to_wigner(native,0,3,jd,rot_clip=True)
                actual = _dense_wigner(actual,3,jd,safe)
                tolerance = 2e-6 if dtype == torch.float32 else 1e-12
                torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
                weight = torch.randn_like(expected)
                grad = torch.autograd.grad((expected*weight).sum(),vectors,retain_graph=True)[0]
                actual_grad = torch.autograd.grad((actual*weight).sum(),vectors)[0]
                self.assertTrue(bool(actual_grad.isfinite().all()))
                torch.testing.assert_close(actual_grad,grad,rtol=tolerance*10,atol=tolerance*10)


if __name__ == '__main__':
    unittest.main()
