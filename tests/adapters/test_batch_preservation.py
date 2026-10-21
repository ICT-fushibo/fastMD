"""Migration regressions: empty-angle unions and SevenNet fixed modality."""
import dataclasses
import importlib.util
import os
from pathlib import Path
import unittest
import torch
from ase import Atoms

from fastmd.engine.energy_derivative import EnergyDerivative
from fastmd.execution.replica_batch import replica_energy_derivative
from adapters.batch_helpers import union, ase_edges


@unittest.skipUnless(importlib.util.find_spec('chgnet'), 'optional CHGNet unavailable')
class CHGNetBatchPreservationTest(unittest.TestCase):
    def test_empty_and_nonempty_systems_keep_independent_bond_branches(self):
        from adapters.test_chgnet_empty_angles import EmptyAngleTest, batch_of, structures
        from pymatgen.io.ase import AseAtomsAdaptor
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        helper=EmptyAngleTest()
        for version in ('0.2.0','0.3.0','r2scan'):
            with self.subTest(version=version):
                adapter,_,expected=helper.case(version)
                try:
                    batches=[batch_of(adapter.module.graph_converter(AseAtomsAdaptor.get_structure(a)))
                             for a in structures()]
                    batch=union(batches)
                    energies,forces=replica_energy_derivative(adapter,batch,adapter.prepare(batch),
                        replicas=2,replica_atoms=batches[0].n_real)
                    for i,(energy,force,_) in enumerate(expected):
                        self.assertAlmostEqual(float(energies[i]),energy,delta=2e-4)
                        torch.testing.assert_close(forces.reshape(2,-1,3)[i],force,atol=2e-4,rtol=0)
                    # Far self-loop padding has a REAL centre: it must not turn on
                    # the biased bond layer when the real angle graph is empty.
                    empty=batches[0];e=len(empty.edge_index);u=len(empty.pair_u2d)
                    padded=dataclasses.replace(empty,
                        edge_index=torch.cat([empty.edge_index,torch.zeros(2,2,dtype=torch.int32)]),
                        edge_image=torch.cat([empty.edge_image,torch.tensor([[3.,0.,0.],[-3.,0.,0.]])]),
                        pair_d2u=torch.cat([empty.pair_d2u,torch.tensor([u,u],dtype=torch.int32)]),
                        pair_u2d=torch.cat([empty.pair_u2d,torch.tensor([e],dtype=torch.int32)]),
                        triplets=torch.tensor([[0,u,e,u,e]],dtype=torch.int32),
                        padding=adapter.padding_rule(),n_edges=torch.tensor([e]))
                    for count in (padded.n_edges,None):
                        padded.n_edges=count
                        out=EnergyDerivative(adapter)(padded,adapter.prepare(padded),task='efs')
                        helper.check_efs(out,expected[0],empty.n_real)
                finally:adapter.close()


@unittest.skipUnless(importlib.util.find_spec('sevenn'), 'optional SevenNet unavailable')
class SevenNetBatchPreservationTest(unittest.TestCase):
    def test_builtin_model_and_replica_derivative_on_cpu(self):
        self.check_model('sevennet_0',None,None,False)

    @unittest.skipUnless(os.environ.get('FASTMD_TEST_CHECKPOINT_DIR'), 'set FASTMD_TEST_CHECKPOINT_DIR for local weights')
    def test_fixed_modal_and_eager_only_checkpoints(self):
        root=Path(os.environ['FASTMD_TEST_CHECKPOINT_DIR'])
        for name,modal,eager in [('7net-mf-ompa','mpa',False),('7net-omni','mpa',False),('7net-l3i5',None,True)]:
            with self.subTest(name=name):self.check_model('sevennet',str(root/(name+'.ckpt')),modal,eager)

    def check_model(self,name,path,modal,eager):
        from sevenn.calculator import SevenNetCalculator
        from fastmd.models.sevennet import load_sevennet
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        adapter=load_sevennet(name,model_path=path,modal=modal,allow_eager=eager,
                             device='cpu',optimization_profile='generic')
        self.addCleanup(adapter.close)
        upstream=SevenNetCalculator(model=path or '7net-0',modal=modal,device='cpu')
        batches=[];values=[]
        for distance in (2.1,2.3):
            atoms=Atoms('Si2',positions=[[1,1,1],[1+distance,1,1]],cell=[20]*3,pbc=True)
            atoms.calc=upstream
            batch=ase_edges(atoms,adapter.topology_spec().cutoff)
            out=EnergyDerivative(adapter)(batch,adapter.prepare(batch),task='efs')
            self.assertAlmostEqual(float(out['e'][0]),atoms.get_potential_energy(),delta=5e-4)
            torch.testing.assert_close(out['f'][0],torch.tensor(atoms.get_forces(),dtype=torch.float32),rtol=0,atol=5e-5)
            torch.testing.assert_close(out['s'][0],torch.tensor(atoms.get_stress(voigt=False),dtype=torch.float32),rtol=0,atol=2e-6)
            batches.append(batch);values.append(out)
        batch=union(batches)
        energies,forces=replica_energy_derivative(adapter,batch,adapter.prepare(batch),replicas=2,replica_atoms=2)
        for i,out in enumerate(values):
            self.assertAlmostEqual(float(energies[i]),float(out['e'][0]),delta=5e-4)
            torch.testing.assert_close(forces.reshape(2,2,3)[i],out['f'][0],atol=5e-5,rtol=0)
        self.assertEqual(adapter.system_inputs()['modal'],modal)
        if eager:
            self.assertFalse(adapter.capture_contract().sync_free)
            with self.assertRaisesRegex(ValueError,'unpadded eager'):
                adapter.prepare(dataclasses.replace(batches[0],padding=adapter.padding_rule()))


@unittest.skipUnless(importlib.util.find_spec('orb_models') and os.environ.get('FASTMD_TEST_CHECKPOINT_DIR'),
                     'requires ORB and FASTMD_TEST_CHECKPOINT_DIR')
class ORBBatchPreservationTest(unittest.TestCase):
    def test_segment_pooling_and_forces_match_independent_calls(self):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        from fastmd.models.orb import ORBAdapter
        name='orb-v3-conservative-inf-mpa'
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        module=pretrained.ORB_PRETRAINED_MODELS[name](
            weights_path=str(Path(os.environ['FASTMD_TEST_CHECKPOINT_DIR'])/(name+'.ckpt')),
            device='cpu',precision='float32-highest',compile=False)
        reference=ORBCalculator(module,device='cpu')
        atoms=[Atoms('Si2',positions=[[1,1,1],[1+d,1,1]],cell=[20]*3,pbc=True) for d in (2.1,2.4)]
        expected=[]
        for a in atoms:
            a.calc=reference
            expected.append((a.get_potential_energy(),a.get_forces()))
        adapter=ORBAdapter(module)
        self.addCleanup(adapter.close)
        self.assertEqual(adapter.system_batching(),'segments')
        batches=[ase_edges(a,adapter.topology_spec().cutoff) for a in atoms]
        batch=union(batches)
        energies,forces=replica_energy_derivative(adapter,batch,adapter.prepare(batch),replicas=2,replica_atoms=2)
        for i,(energy,force) in enumerate(expected):
            self.assertAlmostEqual(float(energies[i]),energy,delta=5e-4)
            torch.testing.assert_close(forces.reshape(2,2,3)[i],torch.tensor(force,dtype=torch.float32),atol=5e-5,rtol=0)


class HeterogeneousModelBatchTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('sevenn'), 'optional SevenNet unavailable')
    def test_sevennet_different_cells_and_atom_counts(self):
        from adapters.batch_helpers import check_heterogeneous
        from fastmd.models.sevennet import load_sevennet
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        adapter=load_sevennet('sevennet_0',device='cpu',optimization_profile='generic')
        self.addCleanup(getattr(adapter,"close",lambda: None))
        check_heterogeneous(self,adapter)

    @unittest.skipUnless(importlib.util.find_spec('mace') and os.environ.get('FASTMD_TEST_CHECKPOINT_DIR'),
                         'requires MACE and FASTMD_TEST_CHECKPOINT_DIR')
    def test_mace_different_cells_and_atom_counts(self):
        from adapters.batch_helpers import check_heterogeneous
        from fastmd.models.mace import load_mace
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        adapter=load_mace('mace',model_path=str(Path(os.environ['FASTMD_TEST_CHECKPOINT_DIR'])/'mace-small.model'),
                         device='cpu',optimization_profile='generic',model_dtype='float32')
        self.addCleanup(getattr(adapter,"close",lambda: None))
        check_heterogeneous(self,adapter)
