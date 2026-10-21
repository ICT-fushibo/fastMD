"""Padding must preserve eSEN's skip-all-blocks branch on an empty graph."""
import importlib.metadata
import unittest
import torch

try:
    LEGACY_ESEN=importlib.metadata.version('fairchem-core')=='1.10.0'
except importlib.metadata.PackageNotFoundError:
    LEGACY_ESEN=False


@unittest.skipUnless(LEGACY_ESEN,'requires optional fairchem-core==1.10.0')
class ESENPaddingTest(unittest.TestCase):
    def test_empty_ragged_and_padded_energy_forces_stress(self):
        from fairchem.core.models.esen.esen import eSEN_Backbone, MLP_EFS_Head
        from fastmd.models.esen import ESENAdapter
        from fastmd.adapters.protocol import GraphBatch
        from fastmd.engine.energy_derivative import EnergyDerivative
        torch.set_num_threads(2)
        torch.manual_seed(10)
        module=torch.nn.Module()
        module.backbone=eSEN_Backbone(sphere_channels=8,hidden_channels=8,edge_channels=8,
                                     num_layers=2,lmax=2,mmax=1,regress_forces=True,regress_stress=True,
                                     cutoff=6.,direct_forces=False,use_envelope=True,act_type='gate',mlp_type='spectral')
        module.output_heads=torch.nn.ModuleDict({'head':MLP_EFS_Head(module.backbone)})
        normalizers={key:dict(mean=torch.tensor(0.),rmsd=torch.tensor(1.)) for key in ('energy','forces','stress')}
        adapter=ESENAdapter(module,normalizers=normalizers,reference=torch.zeros(119))
        self.addCleanup(adapter.close)
        positions=torch.tensor([[1.,1.,1.],[12.,12.,12.]])
        cell=torch.eye(3)*30
        def batch(padded):
            count=8 if padded else 0
            edge=torch.zeros(count,2,dtype=torch.int32)
            if padded:edge[1::2]=1
            return GraphBatch(positions=positions,frac_coords=positions/30,cell=cell,
                              species=torch.tensor([1,2],dtype=torch.int32),n_real=2,edge_index=edge,
                              edge_image=torch.tensor([[1.,0.,0.]]).repeat(count,1),n_edges=torch.tensor([0]),
                              cutoff=6.,padding=adapter.padding_rule() if padded else None)
        results=[]
        for padded in (False,True):
            graph=batch(padded)
            results.append(EnergyDerivative(adapter)(graph,adapter.prepare(graph),task='efs'))
        torch.testing.assert_close(results[1]['e'],results[0]['e'],rtol=0,atol=1e-7)
        for result in results:
            torch.testing.assert_close(result['f'][0],torch.zeros_like(positions),rtol=0,atol=0)
            torch.testing.assert_close(result['s'][0],torch.zeros_like(cell),rtol=0,atol=0)
        adapter.close()
        self.assertFalse(adapter.capture_contract().sync_free)


if __name__=='__main__':unittest.main()
