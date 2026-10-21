"""CPU regression for serialized MACE CG phases lost by cuEq conversion."""
import importlib.util
import os
import unittest
from unittest.mock import patch
import torch
from fastmd.models.mace import _align_cueq_contraction_basis

AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ('mace','cuequivariance_torch'))

@unittest.skipUnless(AVAILABLE,'optional MACE/cuEquivariance packages required')
class LegacyBasis(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        import mace  # noqa: F401 (upstream's compatibility setup before e3nn)
        from e3nn import o3
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        from mace.modules.symmetric_contraction import SymmetricContraction
        from mace.tools.cg import O3_e3nn
        self.old_dtype=torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype,self.old_dtype)
        torch.set_default_dtype(torch.float64)
        self.irreps=o3.Irreps('2x0e+2x1o')
        self.source=torch.nn.Module();self.target=torch.nn.Module()
        a=torch.nn.Module();b=torch.nn.Module()
        a.symmetric_contractions=SymmetricContraction(self.irreps,self.irreps,correlation=3,num_elements=2,use_reduced_cg=False)
        b.symmetric_contractions=cuet.SymmetricContraction(cue.Irreps(O3_e3nn,str(self.irreps)),cue.Irreps(O3_e3nn,str(self.irreps)),contraction_degree=3,num_elements=2,layout_in=cue.ir_mul,layout_out=cue.ir_mul,original_mace=True,method='naive',dtype=torch.float64,math_dtype=torch.float64)
        self.source.products=torch.nn.ModuleList([a]);self.target.products=torch.nn.ModuleList([b])
        weights=[]
        for c in a.symmetric_contractions.contractions:
            weights.extend([c.weights_max,*c.weights])
        with torch.no_grad():b.symmetric_contractions.weight.copy_(torch.cat(weights,dim=1))

    def outputs(self):
        generator=torch.Generator().manual_seed(7)
        x=torch.randn(3,2,4,generator=generator,requires_grad=True)
        y=torch.eye(2)[[0,1,0]]
        a=self.source.products[0].symmetric_contractions(x,y)
        a=torch.cat([v.reshape(3,mul,ir.dim).transpose(1,2).reshape(3,-1)
                     for v,(mul,ir) in zip(a.split([mul*ir.dim for mul,ir in self.irreps],dim=-1),self.irreps)],dim=-1)
        b=self.target.products[0].symmetric_contractions(x.transpose(1,2).reshape(3,-1),y.argmax(-1).int())
        ga,=torch.autograd.grad(a.square().sum(),x,retain_graph=True)
        gb,=torch.autograd.grad(b.square().sum(),x)
        return a,b,ga,gb

    def test_legacy_signs_preserve_outputs_gradients_and_reference(self):
        source=self.source.products[0].symmetric_contractions
        # A legacy serialized basis can have a different phase for one path.
        with torch.no_grad():source.contractions[1].U_matrix_3[...,0].neg_()
        before={k:v.clone() for k,v in self.source.state_dict().items()}
        a,b,_,_=self.outputs()
        self.assertGreater(float((a-b).detach().abs().max()),1e-4)
        changes=_align_cueq_contraction_basis(self.source,self.target)
        self.assertEqual(changes,[dict(layer=0,irrep='1o',degree=3,columns=[0])])
        for k,v in self.source.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        a,b,ga,gb=self.outputs()
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(ga,gb,rtol=1e-10,atol=1e-10)

    def test_current_basis_needs_no_changes(self):
        before=self.target.products[0].symmetric_contractions.weight.detach().clone()
        self.assertEqual(_align_cueq_contraction_basis(self.source,self.target),[])
        self.assertTrue(torch.equal(before,self.target.products[0].symmetric_contractions.weight))

    def test_unsupported_basis_rejects_conversion(self):
        with torch.no_grad():self.source.products[0].symmetric_contractions.contractions[1].U_matrix_3.mul_(1.1)
        with self.assertRaisesRegex(ValueError,'beyond column signs'):
            _align_cueq_contraction_basis(self.source,self.target)

if __name__=='__main__':unittest.main()
