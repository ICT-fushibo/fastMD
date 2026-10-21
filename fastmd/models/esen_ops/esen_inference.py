"""eSEN inference arithmetic: permutations, block rotations and fused nonlinearities.

SO(2) convolution ordering follows fairchem 1.10 (Meta, Inc. and affiliates,
MIT; see licenses/fairchem-MIT.txt). The learned modules and weights are reused.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def so2_convolution(self, x, x_edge):
    n=len(x_edge)
    # Both maps are genuine permutations, validated once at installation.
    x=x.index_select(1,self._fastmd_to_m)
    radial=self.rad_func(x_edge) if self.rad_func is not None else None
    size0=self.mappingReduced.m_size[0]
    x0=x[:, :size0].reshape(n,self.fc_m0.in_features)
    if radial is not None:x0=x0*radial[:, :self.fc_m0.in_features]
    x0=self.fc_m0(x0)
    extra=None
    if self.extra_m0_output_channels is not None:
        extra=x0[:, :self.extra_m0_output_channels]
        x0=x0[:, self.extra_m0_output_channels:]
    output=[x0.reshape(n,size0,self.m_output_channels)]
    offset=size0;offset_rad=self.fc_m0.in_features
    for m,conv in enumerate(self.so2_m_conv,1):
        size=self.mappingReduced.m_size[m]
        xm=x[:, offset:offset+2*size].reshape(n,2,conv.fc.in_features)
        if radial is not None:xm=xm*radial[:, None,offset_rad:offset_rad+conv.fc.in_features]
        output.append(conv(xm).reshape(n,2*size,self.m_output_channels))
        offset+=2*size;offset_rad+=conv.fc.in_features
    output=torch.cat(output,dim=1).index_select(1,self._fastmd_to_l)
    return (output,extra) if self.extra_m0_output_channels is not None else output


def gate_torch(gating,x,index):
    degree_count=math.prod(gating.shape[1:])//x.shape[-1]
    sig=torch.sigmoid(gating.reshape(len(x),degree_count,x.shape[-1]))
    return torch.cat((F.silu(x[:,:1]), x[:,1:]*sig.index_select(1,index)),dim=1)


def gate(self,gating_scalars,input_tensors):
    if self._fastmd_operator_backend=='triton':
        from fastmd.models.esen_ops.triton_esen_nonlinear import fused_gate
        return fused_gate(gating_scalars,input_tensors,self.expand_index,self.lmax)
    return gate_torch(gating_scalars,input_tensors,self.expand_index)


def rms_norm_torch(x,weight,bias,index,balance,eps):
    centered=torch.cat((x[:,:1]-x[:,:1].mean(-1,keepdim=True),x[:,1:]),dim=1)
    variance=(centered.square()*balance).sum(1,keepdim=True).mean(-1,keepdim=True)
    result=centered*torch.rsqrt(variance+eps)*weight.index_select(0,index)
    return torch.cat((result[:,:1]+bias,result[:,1:]),dim=1)


def rms_norm(self,node_input):
    if self._fastmd_operator_backend=='triton':
        from fastmd.models.esen_ops.triton_esen_nonlinear import fused_rms_norm
        return fused_rms_norm(node_input,self.affine_weight,self.affine_bias,self.expand_index,
                              self.balance_degree_weight,self.eps)
    return rms_norm_torch(node_input,self.affine_weight,self.affine_bias,self.expand_index,
                          self.balance_degree_weight,self.eps)


def rotate_gather_torch(nodes,edges,wigner,lmax,mmax):
    """Block-diagonal Wigner application, gather each angular degree separately."""
    result=[]
    for degree in range(lmax+1):
        start=degree**2;end=(degree+1)**2
        trim=max(degree-mmax,0)
        value=torch.cat((nodes[edges[0],start:end],nodes[edges[1],start:end]),dim=2)
        result.append(torch.bmm(wigner[:,start+trim:end-trim,start:end],value))
    return torch.cat(result,dim=1)


def rotate_scatter_torch(value,wigner_inv,targets,n_nodes,envelope,lmax,mmax):
    result=[];offset=0
    for degree in range(lmax+1):
        start=degree**2;end=(degree+1)**2;trim=max(degree-mmax,0)
        width=end-start-2*trim
        scaled=value[:,offset:offset+width]*envelope[:,None,None]
        rotated=torch.bmm(wigner_inv[:,start:end,start+trim:end-trim],scaled)
        result.append(rotated.new_zeros((n_nodes,end-start,value.shape[-1])).index_add(0,targets,rotated))
        offset+=width
    return torch.cat(result,dim=1)


def edgewise(self,x,x_edge,edge_distance,edge_index,wigner,wigner_inv,node_offset=0):
    if self._fastmd_operator_backend=='triton':
        from fastmd.models.esen_ops.triton_esen_rotation import rotate_gather,rotate_scatter
    else:
        rotate_gather,rotate_scatter=rotate_gather_torch,rotate_scatter_torch
    message=rotate_gather(x,edge_index,wigner,self.lmax,self.mmax)
    message,gating=self.so2_conv_1(message,x_edge)
    message=self.act(gating,message)
    message=self.so2_conv_2(message,x_edge)
    envelope=self.envelope(edge_distance/self.cutoff)
    return rotate_scatter(message,wigner_inv,edge_index[1]-node_offset,len(x),envelope,self.lmax,self.mmax)
