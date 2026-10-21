"""CPU-built disjoint graphs for independent model/batch parity checks."""
import dataclasses
import torch
from fastmd.adapters.protocol import GraphBatch, SystemSegments


def union(batches):
    rows, edges, pairs = 0, 0, 0
    edge_index, triplets, d2u, u2d = [], [], [], []
    row_ptr = [0]
    for batch in batches:
        edge_index.append(batch.edge_index + rows)
        if batch.triplets is not None:
            triplets.append(batch.triplets + batch.triplets.new_tensor([rows, pairs, edges, pairs, edges]))
            d2u.append(batch.pair_d2u + pairs)
            u2d.append(batch.pair_u2d + edges)
            pairs += len(batch.pair_u2d)
        rows += batch.n_real
        edges += len(batch.edge_index)
        row_ptr.append(rows)
    return dataclasses.replace(batches[0],
        positions=torch.cat([b.positions for b in batches]),
        frac_coords=torch.cat([b.frac_coords for b in batches]),
        species=torch.cat([b.species for b in batches]), n_real=rows,
        edge_index=torch.cat(edge_index), edge_image=torch.cat([b.edge_image for b in batches]),
        n_edges=torch.tensor([edges]),
        pair_u2d=torch.cat(u2d) if u2d else None, pair_d2u=torch.cat(d2u) if d2u else None,
        triplets=torch.cat(triplets) if triplets else None,
        systems=SystemSegments(len(batches), torch.tensor(row_ptr),
            torch.repeat_interleave(torch.arange(len(batches), dtype=torch.int32),
                                    torch.tensor([b.n_real for b in batches]))))


def ase_edges(atoms, cutoff):
    import numpy as np
    from ase.neighborlist import neighbor_list
    i,j,shifts=neighbor_list('ijS',atoms,cutoff)
    positions=torch.tensor(atoms.positions,dtype=torch.float32)
    return GraphBatch(positions=positions, frac_coords=torch.tensor(atoms.get_scaled_positions(),dtype=torch.float32),
        cell=torch.tensor(atoms.cell.array,dtype=torch.float32), species=torch.tensor(atoms.numbers,dtype=torch.int32),
        n_real=len(atoms), edge_index=torch.tensor(np.stack((i,j),axis=1),dtype=torch.int32),
        edge_image=torch.tensor(shifts,dtype=torch.float32), n_edges=torch.tensor([len(i)]),
        cutoff=cutoff,padding=None)


def check_heterogeneous(test, adapter):
    from ase import Atoms
    from fastmd.engine.energy_derivative import EnergyDerivative
    from fastmd.execution.system_batch import system_energy_derivative
    atoms=[Atoms('Si2',positions=[[1,.5,.1],[4.5,.2,.3]],cell=[6.5]*3,pbc=True),
           Atoms('Si3',positions=[[.2,.1,.1],[2.4,.1,.2],[7.,1.,.3]],cell=[8.,7.,6.],pbc=True)]
    batches=[ase_edges(a,adapter.topology_spec().cutoff) for a in atoms]
    expected=[EnergyDerivative(adapter)(b,adapter.prepare(b),task='efs') for b in batches]
    batch=union(batches)
    batch.systems=dataclasses.replace(batch.systems,cells=torch.stack([b.cell for b in batches]),
                                     atoms=torch.tensor([b.n_real for b in batches]))
    # A deliberately unrelated cell: the new interface must read each edge's
    # system shift rather than silently using the first system's cell.
    energy,force,stress=system_energy_derivative(adapter,batch,adapter.prepare(batch),
                                               cell=torch.eye(3)*97,stress=True)
    start=0
    for i,(b,out) in enumerate(zip(batches,expected)):
        test.assertAlmostEqual(float(energy[i]),float(out['e'][0]),delta=5e-4)
        torch.testing.assert_close(force[start:start+b.n_real],out['f'][0],atol=7e-5,rtol=1e-5)
        torch.testing.assert_close(stress[i],out['s'][0],atol=3e-6,rtol=1e-5)
        start+=b.n_real
