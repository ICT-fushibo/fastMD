"""Opt-in, source-pinned operator replacements; select before preparing inputs.

These candidates are separate from the published benchmark snapshot. Explicit
selection is required until GPU correctness and timing gates have completed.
"""
from __future__ import annotations

from types import MethodType
import torch
from fastmd.adapters.patches import PatchRecord, apply_patches, revert_patches


def _install(outcomes, instance, family, target, sha, implementation):
    prefix = 'orb_models.forcefield.' if family == 'orb' else 'fairchem.core.models.esen.'
    record = PatchRecord(name=f'{family}.operators.{len(outcomes)}',target=prefix+target,
        distribution='orb-models' if family=='orb' else 'fairchem-core',
        versions=('0.5.4',) if family=='orb' else ('1.10.0',),source_sha256=sha,
        reason='opt-in inference operator rewrite with input-gradient parity gates',
        replacement=MethodType(implementation,instance),per_instance=True)
    added = apply_patches([record],instance=instance)
    outcomes.extend(added)
    if not all(p.applied for p in added):
        raise RuntimeError(f'Operator rewrite refused: {[p.reason for p in added]}')


def install_orb(module, backend):
    from fastmd.models.orb_ops.orb_inference import interaction
    outcomes=[]
    try:
        for layer in module.model.gnn_stacks:
            if layer._attention_gate != 'sigmoid' or not layer._distance_cutoff:
                raise ValueError('ORB rewrites require sigmoid attention and distance cutoff')
            linear=layer._edge_mlp.mlp[0]
            if linear.weight.requires_grad:
                raise ValueError('Operator rewrites require frozen inference weights')
            width=linear.in_features//3
            if linear.in_features != 3*width:
                raise ValueError('Unexpected ORB first edge projection shape')
            layer._fastmd_projection_weights=tuple(w.contiguous() for w in linear.weight.split(width,dim=1))
            layer._fastmd_operator_backend=backend
            _install(outcomes,layer,'orb','gns.AttentionInteractionNetwork.forward',
                     '5c3b6f57cdca3abafc47869a6d363dae2582c26a863d9e5a34507b39074f90d0',interaction)
    except BaseException:
        revert_patches(outcomes)
        raise
    return outcomes


def install_esen(module, backend):
    from fairchem.core.models.esen.nn.so2_layers import SO2_Convolution
    from fairchem.core.models.esen.nn.activation import GateActivation
    from fairchem.core.models.esen.nn.layer_norm import EquivariantRMSNormArraySphericalHarmonicsV2
    from fairchem.core.models.esen.esen_block import Edgewise
    from fastmd.models.esen_ops.esen_inference import so2_convolution, gate, rms_norm, edgewise
    outcomes=[]
    try:
        for site in module.modules():
            if isinstance(site,SO2_Convolution):
                matrix=site.mappingReduced.to_m
                if not (bool(((matrix==0)|(matrix==1)).all()) and bool((matrix.sum(0)==1).all())
                        and bool((matrix.sum(1)==1).all())):
                    raise ValueError('eSEN coefficient mapping is not a permutation')
                site._fastmd_to_m=matrix.argmax(1)
                site._fastmd_to_l=matrix.argmax(0)
                _install(outcomes,site,'esen','nn.so2_layers.SO2_Convolution.forward',
                    'bb154ca36eabbbea7843a7d3212282a77ceb0f46fdb224c2391bcc3c052061be',so2_convolution)
            elif isinstance(site,GateActivation):
                if not isinstance(site.scalar_act,torch.nn.SiLU) or not isinstance(site.gate_act,torch.nn.Sigmoid):
                    raise ValueError('Unsupported eSEN gate activations')
                site._fastmd_operator_backend=backend
                _install(outcomes,site,'esen','nn.activation.GateActivation.forward',
                    '7b00e9fab8923ed4cdacd922b599d2b4410ab814af4ae98b216301622d794fc5',gate)
            elif isinstance(site,EquivariantRMSNormArraySphericalHarmonicsV2):
                if not (site.affine and site.centering and site.std_balance_degrees
                        and site.normalization=='component'):
                    raise ValueError('Unsupported eSEN RMSNorm configuration')
                site._fastmd_operator_backend=backend
                _install(outcomes,site,'esen','nn.layer_norm.EquivariantRMSNormArraySphericalHarmonicsV2.forward',
                    '184ca7fb61290d25db0d0b812a73ed7c5bb889ec23a3b8baa40a797b10b4683b',rms_norm)
            elif isinstance(site,Edgewise):
                if site.act_type!='gate' or not site.use_envelope:
                    raise ValueError('eSEN rotation rewrite requires gated, envelope-enabled blocks')
                expected=torch.tensor([j for l in range(site.lmax+1)
                    for j in range(l*l+max(l-site.mmax,0),(l+1)**2-max(l-site.mmax,0))],device=site.out_mask.device)
                if not torch.equal(site.out_mask,expected):
                    raise ValueError('Unexpected eSEN selected spherical coefficients')
                site._fastmd_operator_backend=backend
                _install(outcomes,site,'esen','esen_block.Edgewise.forward',
                    '6cf527a5d3f9634df18818afdbcbb347a862718f6189657ab826a1621b94f8b0',edgewise)
    except BaseException:
        revert_patches(outcomes)
        raise
    return outcomes
