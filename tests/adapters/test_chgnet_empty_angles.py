"""Empty real angle graphs must stay empty semantically after sink padding."""
from __future__ import annotations

import dataclasses
import importlib.util
import unittest

import torch
from ase import Atoms, units

from fastmd.adapters.protocol import GraphBatch
from fastmd.engine.energy_derivative import EnergyDerivative


def structures():
    empty = Atoms('CsSnI3', cell=[6.4]*3, pbc=True,
                  scaled_positions=[[0,0,0],[.5,.5,.5],[.5,.5,0],[.5,0,.5],[0,.5,.5]])
    angled = empty.copy()
    angled.positions[2,2] += .4
    angled.positions[3,1] += .4
    return empty, angled


def batch_of(graph):
    frac, cell = graph.atom_frac_coord.detach(), graph.lattice.detach()
    return GraphBatch(positions=frac@cell, frac_coords=frac, cell=cell,
        species=graph.atomic_number.int(), n_real=len(frac),
        edge_index=graph.atom_graph.int(), edge_image=graph.neighbor_image,
        n_edges=None, cutoff=6., padding=None, triplet_cutoff=3.,
        pair_d2u=graph.directed2undirected.int(), pair_u2d=graph.undirected2directed.int(),
        triplets=graph.bond_graph.reshape(-1,5).int())


def padded(batch, adapter, pair_capacity, triplet_capacity):
    n, e, u = batch.n_real, len(batch.edge_index), len(batch.pair_u2d)
    extra = pair_capacity-u
    assert extra > 0 and triplet_capacity > len(batch.triplets)
    return dataclasses.replace(batch,
        positions=torch.cat([batch.positions,torch.zeros(1,3)]),
        frac_coords=torch.cat([batch.frac_coords,torch.zeros(1,3)]),
        species=torch.cat([batch.species,torch.ones(1,dtype=torch.int32)]),
        edge_index=torch.cat([batch.edge_index,torch.full((2*extra,2),n,dtype=torch.int32)]),
        edge_image=torch.cat([batch.edge_image,torch.tensor([[3.,3.,3.],[-3.,-3.,-3.]]).repeat(extra,1)]),
        pair_d2u=torch.cat([batch.pair_d2u,torch.arange(u,pair_capacity,dtype=torch.int32).repeat_interleave(2)]),
        pair_u2d=torch.cat([batch.pair_u2d,torch.arange(e,e+2*extra,2,dtype=torch.int32)]),
        triplets=torch.cat([batch.triplets,torch.tensor([[n,u,e,u,e+1]],dtype=torch.int32).repeat(triplet_capacity-len(batch.triplets),1)]),
        padding=adapter.padding_rule())


def on_device(batch, device):
    return dataclasses.replace(batch, **{f.name:getattr(batch,f.name).to(device)
        for f in dataclasses.fields(batch) if isinstance(getattr(batch,f.name),torch.Tensor)})


@unittest.skipUnless(importlib.util.find_spec('chgnet'), 'optional CHGNet package unavailable')
class EmptyAngleTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')

    def case(self, version):
        from chgnet.model.model import CHGNet
        from pymatgen.io.ase import AseAtomsAdaptor
        from fastmd.models.chgnet import CHGNetAdapter
        module = CHGNet.load(model_name=version, use_device='cpu', verbose=False)
        adapter = CHGNetAdapter(module)
        self.addCleanup(adapter.close)
        graphs = [module.graph_converter(AseAtomsAdaptor.get_structure(a)) for a in structures()]
        batches = [batch_of(g) for g in graphs]
        self.assertEqual(len(batches[0].triplets),0)
        self.assertGreater(len(batches[1].triplets),0)
        pair_capacity = max(len(b.pair_u2d) for b in batches)+4
        triplet_capacity = max(len(b.triplets) for b in batches)+8
        padded_batches = [padded(b,adapter,pair_capacity,triplet_capacity) for b in batches]
        expected=[]
        for graph in graphs:
            value=module([graph],task='efs')
            expected.append((float(value['e'].detach()[0])*len(graph.atomic_number),
                             value['f'][0].detach().clone(),value['s'][0].detach().clone()*units.GPa))
        return adapter,padded_batches,expected

    def check_efs(self, out, expected, n_real):
        self.assertLess(abs(float(out['e'].detach()[0])-expected[0]),2e-4)
        self.assertLess(float((out['f'][0][:n_real].detach().cpu()-expected[1]).abs().max()),2e-4)
        self.assertLess(float((out['s'][0].detach().cpu()-expected[2]).abs().max()),2e-6)

    def test_all_checkpoints_with_and_without_angles(self):
        for version in ('0.2.0','0.3.0','r2scan'):
            with self.subTest(version=version):
                adapter,batches,expected=self.case(version)
                for batch,reference in zip(batches,expected):
                    out=EnergyDerivative(adapter)(batch,adapter.prepare(batch),task='efs',n_real=batch.n_real)
                    self.check_efs(out,reference,batch.n_real)
                adapter.close()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_capture_replay_empty_nonempty_empty(self):
        adapter,batches,expected=self.case('0.2.0')
        adapter.module.to('cuda')
        batch=on_device(batches[0],'cuda')
        inputs=adapter.prepare(batch)
        derivative=EnergyDerivative(adapter)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):derivative(batch,inputs,task='efs',n_real=batch.n_real)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out=derivative(batch,inputs,task='efs',n_real=batch.n_real)
        for index in (0,1,0):
            for field in dataclasses.fields(batch):
                value=getattr(batch,field.name)
                if isinstance(value,torch.Tensor):value.copy_(getattr(batches[index],field.name))
            graph.replay()
            torch.cuda.synchronize()
            self.check_efs(out,expected[index],batch.n_real)


if __name__=='__main__':
    unittest.main()
