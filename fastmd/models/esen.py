"""Legacy eSEN (fairchem-core 1.10): upstream network with a FastMD energy tap.

The original OCPCalculator loads the checkpoint, reference energies and scales.
Only this adapter's backbone instance receives the external graph and the
capture-safe form of upstream's rotations. The trained network is unchanged.
"""
# Rotation arithmetic adapted from fairchem, Copyright (c) Meta, Inc. and
# affiliates, MIT license: https://github.com/FAIR-Chem/fairchem/blob/v1.10.0/LICENSE.md
from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path
from types import MethodType

import torch
from torch import Tensor

from fastmd.adapters.patches import PatchRecord, apply_patches, revert_patches
from fastmd.adapters.protocol import CapacityStep, CaptureContract, GraphBatch, PaddingRule, TopologySpec

_PREFIX = 'fairchem.core.models.esen.'
_PINS = {
    'esen.eSEN_Backbone.forward': '4890242b60f2cfc66528036415102f115c74b79cacadb30da962a85d18d2058b',
    'esen.MLP_EFS_Head.forward': '7b6daeb1ea1217d4de8389afd3545d014ea417e6cd48ce99a6c06293840d1c06',
    'common.rotation.init_edge_rot_mat': '3255e794d88708467183f4fd2c32f1b8b48575d6650ede7fb48ce4409c98ce5e',
    'common.rotation.rotation_to_wigner': '119fd151c4f1b3879caaaa92c15c4de87da99de9ac26c2a14fcb717d102b4300',
    'esen_block.eSEN_Block.forward': '1b187b57563b51f3623a7b8c5d031789eae592793caf454042b43241044f5364',
    'nn.embedding.EdgeDegreeEmbedding.forward': '052f4268d8574f29f26979e3b87670a2325b2bd264f0045c1b7b09945cafdf19',
    'nn.radial.PolynomialEnvelope.forward': 'b5d59cde8279ca79577a87d0ec445ae2220d70ca0ad6c2bc0129acb75c522e86',
}


def _record(name, target, sha, reason, **kwargs):
    return PatchRecord(name='esen.'+name, target=_PREFIX+target,
                       distribution='fairchem-core', versions=('1.10.0',),
                       source_sha256=sha, reason=reason, **kwargs)


def _external_graph(self, data):
    return data['_fastmd_graph']


def _edge_rotation(vectors: Tensor, y_axis: Tensor) -> Tensor:
    """Upstream rot_clip=True, without host assertions or boolean-index writes."""
    from fairchem.core.models.esen.common.rotation import YTOL
    distance = torch.sqrt(torch.sum(vectors**2, dim=1))
    x = vectors / distance[:, None]
    yprod = x @ y_axis
    x = torch.where((yprod > YTOL)[:, None], y_axis,
                    torch.where((yprod < -YTOL)[:, None], -y_axis, x))
    other = torch.rand_like(vectors) - .5
    other = other / torch.sqrt(torch.sum(other**2, dim=1))[:, None]
    b = torch.stack((-other[:, 1], other[:, 0], other[:, 2]), dim=1)
    c = torch.stack((other[:, 0], -other[:, 2], other[:, 1]), dim=1)
    dot = lambda value: torch.abs(torch.sum(value*x, dim=1))[:, None]
    other = torch.where(dot(other) > dot(b), b, other)
    other = torch.where(dot(other) > dot(c), c, other)
    z = torch.cross(x, other, dim=1)
    z = z / torch.sqrt(torch.sum(z**2, dim=1, keepdim=True))
    z = z / torch.sqrt(torch.sum(z**2, dim=1))[:, None]
    y = torch.cross(x, z, dim=1)
    y = y / torch.sqrt(torch.sum(y**2, dim=1, keepdim=True))
    return torch.stack((z, x, -y), dim=2).transpose(1, 2)


def _dense_wigner(rotation: Tensor, lmax: int, jd: list[Tensor], safe_rotation: Tensor) -> Tensor:
    """Upstream clipped Wigner blocks with a fixed shape and finite unused gradients.

    Nonpolar edges retain their full derivatives. Polar edges use the original
    detached angles; a harmless rotation substitutes for those edges in the
    differentiable branch, avoiding 0*inf from acos at a pole.
    """
    from e3nn import o3
    from fairchem.core.models.esen.common.rotation import YTOL, wigner_D
    yprod = rotation[:, 1, 1].detach()
    live = (yprod > -YTOL) & (yprod < YTOL)

    def angles(matrix):
        alpha, beta = o3.xyz_to_angles(matrix[:, :, 1])
        remainder = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2) @ matrix
        return alpha, beta, torch.atan2(remainder[:, 0, 2], remainder[:, 0, 0])

    a, b, c = angles(torch.where(live[:, None, None], rotation, safe_rotation))
    ad, bd, cd = angles(rotation.detach())
    bd = torch.where(yprod > YTOL, 0., torch.where(yprod < -YTOL, math.pi, bd))
    a, b, c = (torch.where(live, value, frozen) for value, frozen in ((a,ad),(b,bd),(c,cd)))
    result = rotation.new_zeros((len(rotation), (lmax+1)**2, (lmax+1)**2))
    for degree in range(lmax+1):
        start, end = degree**2, (degree+1)**2
        result[:, start:end, start:end] = wigner_D(degree, a, b, c, jd)
    return result


def _capture_rotations(self, vectors):
    rotation = _edge_rotation(vectors, self._fastmd_y_axis)
    jd = [getattr(self, f'Jd_{degree}').to(dtype=rotation.dtype) for degree in range(self.lmax+1)]
    wigner = _dense_wigner(rotation, self.lmax, jd, self._fastmd_safe_rotation)
    return rotation, wigner, wigner.transpose(1,2).contiguous()


def is_esen_model(model, model_path=None):
    return isinstance(model, str) and model in ('esen', 'esen_30m_oam', 'esen-30m-oam')


def load_esen(model='esen', *, model_path=None, device=None, model_dtype=None, optimization_profile=None):
    if model_path is None:
        raise ValueError('eSEN requires an explicit local model_path')
    if model_dtype not in (None, 'float32'):
        raise ValueError('Legacy eSEN supports float32 on this runtime')
    if optimization_profile == 'legacy':
        raise ValueError("optimization_profile='legacy' is MatRIS-only")
    from importlib.metadata import version
    if version('fairchem-core') != '1.10.0':
        raise ValueError('Legacy eSEN requires fairchem-core==1.10.0')
    from fairchem.core import OCPCalculator
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    # Load on CPU exactly as upstream, then put the module on the requested device.
    # The shipped checkpoint is a trusted legacy torch pickle, as in OCPCalculator.
    calculator = OCPCalculator(checkpoint_path=str(model_path), cpu=True, seed=0)
    trainer = calculator.trainer
    if trainer.ema is not None:
        trainer.ema.copy_to()
    module = trainer._unwrapped_model.to(device=device, dtype=torch.float32).eval()
    normalizers = {key:normalizer.state_dict() for key,normalizer in trainer.normalizers.items()}
    reference = trainer.elementrefs['energy'].state_dict()['element_references']
    path = Path(model_path).resolve()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return ESENAdapter(module, normalizers=normalizers, reference=reference,
                       checkpoint=dict(name=model, path=str(path), sha256=sha))


@dataclasses.dataclass
class ESENInput:
    batch: GraphBatch
    owners: Tensor
    natoms: Tensor


class ESENAdapter:
    name = 'esen'
    compiled = False

    def __init__(self, module, *, normalizers, reference, checkpoint=None):
        self.module = module.eval()
        self._operator_patches = []
        self.operator_backend = 'upstream'
        self._in_use = False
        self.checkpoint = dict(checkpoint or {})
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        backbone = module.backbone
        if not backbone.use_envelope or backbone.direct_forces:
            raise ValueError('eSEN requires a conservative, envelope-enabled checkpoint')
        if len(module.output_heads) != 1:
            raise ValueError('eSEN adapter expects one energy/force/stress head')
        self.head = next(iter(module.output_heads.values()))
        self._original_options = (backbone.otf_graph, backbone.regress_forces, backbone.regress_stress)
        backbone.otf_graph, backbone.regress_forces, backbone.regress_stress = True, False, False
        device = next(module.parameters()).device
        self.scale = normalizers['energy']['rmsd'].detach().to(device)
        self.mean = normalizers['energy']['mean'].detach().to(device)
        for key in ('forces','stress'):
            if not torch.equal(normalizers[key]['rmsd'].cpu(), normalizers['energy']['rmsd'].cpu()) or bool(normalizers[key]['mean'] != 0):
                raise ValueError('Energy differentiation requires matching E/F/S scales and zero derivative means')
        self.reference = reference.detach().to(device)
        backbone.register_buffer('_fastmd_y_axis', torch.tensor([0.,1.,0.],device=device), persistent=False)
        backbone.register_buffer('_fastmd_safe_rotation', torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device=device), persistent=False)
        records = [_record('pin_'+target, target, sha, 'validated upstream eSEN arithmetic') for target,sha in _PINS.items()]
        records += [
            _record('external_graph', 'esen.eSEN_Backbone.generate_graph',
                    'f3951b0443e37b60df73d18003b8b07f6fa4a17f9f8c0a516b23f12872cf358a',
                    'read the differentiable FastMD graph, with upstream edge orientation',
                    replacement=MethodType(_external_graph,backbone), per_instance=True),
            _record('dense_rotations', 'esen.eSEN_Backbone.get_rotmat_and_wigner',
                    '144c4c61fd4e24f790c277e8277eb8241cdcaaaae9e9a5bde89a56f63130a886',
                    'fixed-shape clipped rotations without device-to-host checks or dynamic boolean indexing',
                    replacement=MethodType(_capture_rotations,backbone), per_instance=True),
        ]
        self._patches = apply_patches(records, instance=backbone)
        # Upstream keeps these aliases as ordinary CPU tensors; move each alias
        # once, so advanced indexing does not copy host indices during capture.
        index_sites = [('degree', backbone.edge_degree_embedding,
                        'nn.embedding.EdgeDegreeEmbedding', _PINS['nn.embedding.EdgeDegreeEmbedding.forward'])]
        index_sites += [(f'block_{i}', block.edge_wise, 'esen_block.Edgewise',
                         '6cf527a5d3f9634df18818afdbcbb347a862718f6189657ab826a1621b94f8b0')
                        for i,block in enumerate(backbone.blocks)]
        for name, site, target, sha in index_sites:
            self._patches += apply_patches([_record(
                'device_indices_'+name, target+'.out_mask', sha,
                'keep fixed coefficient indices on the model device before CUDA capture',
                source=_PREFIX+target+'.forward', replacement=site.out_mask.to(device), per_instance=True,
            )], instance=site)
        refused = [p for p in self._patches if not p.applied]
        if refused:
            self.close()
            raise RuntimeError(f'Unsupported upstream eSEN source: {[(p.record.name,p.reason) for p in refused]}')
        self._checked_species = None

    def topology_spec(self):
        b=self.module.backbone
        return TopologySpec(cutoff=b.cutoff, num_interactions=b.num_layers, model_dtype='float32',
                            max_neighbors=b.max_neighbors)

    def padding_rule(self):
        return PaddingRule('far_self_loops')

    def capacity_step(self):
        # Every padded edge traverses ten large message-passing blocks.
        return CapacityStep(u_step=32, min_pad_u=12)

    def prepare(self, batch):
        self._in_use = True
        if batch.species is not self._checked_species:
            if bool(((batch.species<1)|(batch.species>=self.module.backbone.max_num_elements)).any()):
                raise ValueError('Atomic number outside the eSEN embedding table')
            self._checked_species=batch.species
        return ESENInput(batch, torch.zeros(len(batch.species),dtype=torch.long,device=batch.species.device),
                         torch.tensor([len(batch.species)],dtype=torch.long,device=batch.species.device))

    def node_energy(self, inputs, positions, cell):
        b=inputs.batch
        source, target = b.edge_index[:,1].long(), b.edge_index[:,0].long()
        vectors = positions[source]-positions[target]+b.edge_image.to(positions.dtype)@cell
        distance = torch.linalg.vector_norm(vectors,dim=-1)
        species = b.species.long()
        graph = dict(edge_index=torch.stack((source,target)), edge_distance=distance, edge_distance_vec=vectors,
                     node_offset=0, batch_full=inputs.owners, atomic_numbers_full=species)
        data = dict(pos=positions, cell=cell[None], atomic_numbers=species, batch=inputs.owners,
                    natoms=inputs.natoms, num_graphs=1, _fastmd_graph=graph)
        hooks=[]
        try:
            if b.padding is not None:
                # Upstream skips every interaction block when the entire graph is empty.
                # Padded edges must not trigger the atomwise part of those blocks.
                has_edges=(distance<self.module.backbone.cutoff).any()
                def preserve_empty(_module,args,output):
                    return torch.where(has_edges,output,args[0])
                hooks=[block.register_forward_hook(preserve_empty) for block in self.module.backbone.blocks]
            embedding=self.module.backbone(data)['node_embedding']
        finally:
            for hook in hooks:hook.remove()
        energy=self.head.energy_block(embedding[:,0,:]).reshape(-1)
        # Keep a zero derivative for an entirely empty ragged graph as well.
        return (energy*self.scale+self.reference[species]+self.mean/b.n_real
                + positions.sum(dim=1)*0. + cell.sum()*0.)

    def reference_energies(self):
        return self.reference[1:].reshape(1,-1)

    def isolated_atom_policy(self):
        return 'model'

    def capture_contract(self):
        return CaptureContract(sync_free=all(p.applied and not p.reverted for p in self._patches),
                               exact_cutoff=True, accuracy_class='fp32_noise',
                               tol_multiplier=2., force_mode='engine_autograd',stress_unit='eV/A^3',compiled='none')

    def upstream_patches(self):
        return list(self._patches) + list(self._operator_patches)

    def enable_operator_rewrites(self, backend='torch'):
        if backend not in ('torch', 'triton'):
            raise ValueError("operator backend must be 'torch' or 'triton'")
        if self._in_use or self._operator_patches:
            raise RuntimeError('Select operator rewrites once, before preparing model inputs')
        from fastmd.models.operator_rewrites import install_esen
        self._operator_patches = install_esen(self.module, backend)
        self.operator_backend = backend

    def system_inputs(self):
        return {'operator_backend': self.operator_backend}

    @property
    def upstream(self):
        return f"fairchem-core 1.10.0, eSEN weights {self.checkpoint.get('sha256','')[:16]}"

    def close(self):
        revert_patches(self._operator_patches)
        revert_patches(self._patches)
        b=self.module.backbone
        b.otf_graph,b.regress_forces,b.regress_stress=self._original_options


def check_load_options(model_dtype, optimization_profile, model_path):
    """Validate registry options without loading the optional framework."""
    from fastmd.adapters.protocol import NegotiationError
    if optimization_profile == "legacy":
        raise NegotiationError("optimization_profile='legacy' is MatRIS-only")
    if model_dtype == "float64":
        raise ValueError("Legacy eSEN supports float32 on this runtime")
    if model_path is None:
        raise ValueError("eSEN requires an explicit local model_path")
