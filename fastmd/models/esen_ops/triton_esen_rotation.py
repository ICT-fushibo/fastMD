"""Block-diagonal eSEN rotations fused with gather/scatter and envelope.

Both feature and Wigner-matrix VJPs are implemented: dropping the latter would
give wrong conservative forces. Cross-degree matrix entries are structural
zeros and are never differentiated; the upstream Wigner builder only fills
the diagonal degree blocks. No TensorFloat-32 or reduced precision is used.
"""
from __future__ import annotations
import os
import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def _degree_layout(degree,MM:tl.constexpr):
    trim=tl.maximum(degree-MM,0)
    start=degree*degree+trim
    count=2*degree+1-2*trim
    offset=tl.where(degree<=MM+1,degree*degree,(MM+1)*(MM+1)+(degree-MM-1)*(2*MM+1))
    return start,count,offset


@triton.jit
def _full_degree(row,LM:tl.constexpr):
    degree=tl.full((),0,tl.int32)
    # Exact integer boundaries; no approximate sqrt at l^2.
    for l in tl.static_range(1,LM+1):
        degree+=tl.cast(row>=l*l,tl.int32)
    return degree


@triton.jit
def _selected_full(p,LM:tl.constexpr,MM:tl.constexpr):
    full=tl.full((),0,tl.int32)
    for degree in tl.static_range(LM+1):
        start,count,offset=_degree_layout(degree,MM)
        full=tl.where((p>=offset)&(p<offset+count),start+p-offset,full)
    return full


@triton.jit
def _gather_fwd(X,EDGES,W,Y,E:tl.constexpr,S:tl.constexpr,P:tl.constexpr,C:tl.constexpr,
                LM:tl.constexpr,MM:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0).to(tl.int64);p=tl.program_id(1)
    c=tl.arange(0,B);live=c<2*C
    node=tl.load(EDGES+tl.where(c<C,0,E)+e).to(tl.int64)
    r=_selected_full(p,LM,MM);degree=_full_degree(r,LM)
    acc=tl.full((B,),0.,tl.float32)
    for q in range(degree*degree,(degree+1)*(degree+1)):
        x=tl.load(X+(node*S+q)*C+c%C,live,other=0.)
        w=tl.load(W+(e*S+r)*S+q)
        acc+=x*w
    tl.store(Y+(e*P+p)*(2*C)+c,acc,live)


@triton.jit
def _gather_dx(EDGES,W,DY,DX,E:tl.constexpr,S:tl.constexpr,P:tl.constexpr,C:tl.constexpr,
              LM:tl.constexpr,MM:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0).to(tl.int64);q=tl.program_id(1)
    c=tl.arange(0,B);live=c<2*C
    node=tl.load(EDGES+tl.where(c<C,0,E)+e).to(tl.int64)
    degree=_full_degree(q,LM)
    start,count,offset=_degree_layout(degree,MM)
    acc=tl.full((B,),0.,tl.float32)
    for j in range(count):
        w=tl.load(W+(e*S+start+j)*S+q)
        dy=tl.load(DY+(e*P+offset+j)*(2*C)+c,live,other=0.)
        acc+=w*dy
    tl.atomic_add(DX+(node*S+q)*C+c%C,acc,live,sem='relaxed')


@triton.jit
def _gather_dw(X,EDGES,DY,DW,E:tl.constexpr,S:tl.constexpr,P:tl.constexpr,C:tl.constexpr,
              LM:tl.constexpr,MM:tl.constexpr,BQ:tl.constexpr,BC:tl.constexpr):
    e=tl.program_id(0).to(tl.int64);p=tl.program_id(1)
    r=_selected_full(p,LM,MM);degree=_full_degree(r,LM)
    q=degree*degree+tl.arange(0,BQ);c=tl.arange(0,BC)
    node=tl.load(EDGES+tl.where(c<C,0,E)+e).to(tl.int64)
    live=(q[:,None]<(degree+1)*(degree+1))&(c[None,:]<2*C)
    x=tl.load(X+(node[None,:]*S+q[:,None])*C+c[None,:]%C,live,other=0.)
    dy=tl.load(DY+(e*P+p)*(2*C)+c,c<2*C,other=0.)
    total=tl.sum(x*dy[None,:],1)
    tl.store(DW+(e*S+r)*S+q,total,q<(degree+1)*(degree+1))


@triton.jit
def _scatter_fwd(X,W,DST,ENV,Y,S:tl.constexpr,P:tl.constexpr,C:tl.constexpr,LM:tl.constexpr,MM:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0).to(tl.int64);r=tl.program_id(1)
    c=tl.arange(0,B);live=c<C
    node=tl.load(DST+e).to(tl.int64);env=tl.load(ENV+e)
    degree=_full_degree(r,LM)
    start,count,offset=_degree_layout(degree,MM)
    acc=tl.full((B,),0.,tl.float32)
    for j in range(count):
        x=tl.load(X+(e*P+offset+j)*C+c,live,other=0.)*env
        w=tl.load(W+(e*S+r)*S+start+j)
        acc+=x*w
    tl.atomic_add(Y+(node*S+r)*C+c,acc,live,sem='relaxed')


@triton.jit
def _scatter_dx(X,W,DST,ENV,DY,DX,DE,S:tl.constexpr,P:tl.constexpr,C:tl.constexpr,
                LM:tl.constexpr,MM:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0).to(tl.int64);p=tl.program_id(1)
    c=tl.arange(0,B);live=c<C
    node=tl.load(DST+e).to(tl.int64);env=tl.load(ENV+e)
    q=_selected_full(p,LM,MM);degree=_full_degree(q,LM)
    acc=tl.full((B,),0.,tl.float32)
    for r in range(degree*degree,(degree+1)*(degree+1)):
        dy=tl.load(DY+(node*S+r)*C+c,live,other=0.)
        w=tl.load(W+(e*S+r)*S+q)
        acc+=dy*w
    tl.store(DX+(e*P+p)*C+c,acc*env,live)
    x=tl.load(X+(e*P+p)*C+c,live,other=0.)
    tl.store(DE+e*P+p,tl.sum(x*acc,0))


@triton.jit
def _scatter_dw(X,DST,ENV,DY,DW,S:tl.constexpr,P:tl.constexpr,C:tl.constexpr,
                LM:tl.constexpr,MM:tl.constexpr,BQ:tl.constexpr,BC:tl.constexpr):
    e=tl.program_id(0).to(tl.int64);r=tl.program_id(1)
    node=tl.load(DST+e).to(tl.int64);env=tl.load(ENV+e)
    degree=_full_degree(r,LM)
    start,count,offset=_degree_layout(degree,MM)
    q=tl.arange(0,BQ);c=tl.arange(0,BC)
    live=(q[:,None]<count)&(c[None,:]<C)
    x=tl.load(X+(e*P+offset+q[:,None])*C+c[None,:],live,other=0.)*env
    dy=tl.load(DY+(node*S+r)*C+c,c<C,other=0.)
    total=tl.sum(x*dy[None,:],1)
    tl.store(DW+(e*S+r)*S+start+q,total,q<count)


def _eligible(value):
    return value.dtype==torch.float32 and (value.is_cuda or os.environ.get('TRITON_INTERPRET')=='1')


class _Gather(Function):
    @staticmethod
    def forward(ctx,x,edges,w,lmax,mmax):
        ctx.save_for_backward(x,edges,w);ctx.orders=(lmax,mmax)
        x,edges,w=(t.contiguous() for t in (x,edges,w))
        e=edges.shape[1];s=(lmax+1)**2;p=sum(min(2*l+1,2*mmax+1) for l in range(lmax+1));c=x.shape[-1]
        y=x.new_empty((e,p,2*c))
        if e:_gather_fwd[(e,p)](x,edges,w,y,e,s,p,c,lmax,mmax,triton.next_power_of_2(2*c),enable_fp_fusion=False)
        return y

    @staticmethod
    def backward(ctx,dy):
        x,edges,w=ctx.saved_tensors;lmax,mmax=ctx.orders
        if torch.is_grad_enabled():
            from fastmd.models.esen_ops.esen_inference import rotate_gather_torch
            needed=(ctx.needs_input_grad[0],ctx.needs_input_grad[2]);inputs=(x,w)
            grads=iter(torch.autograd.grad(rotate_gather_torch(x,edges,w,lmax,mmax),
                [t for t,n in zip(inputs,needed) if n],dy,create_graph=True))
            dx,dw=(next(grads) if n else None for n in needed)
            return dx,None,dw,None,None
        x,edges,w,dy=(t.contiguous() for t in (x,edges,w,dy))
        e,p,c2=dy.shape;s=x.shape[1];c=c2//2
        dx=torch.zeros_like(x);dw=torch.zeros_like(w)
        if e:
            _gather_dx[(e,s)](edges,w,dy,dx,e,s,p,c,lmax,mmax,triton.next_power_of_2(c2),enable_fp_fusion=False)
            _gather_dw[(e,p)](x,edges,dy,dw,e,s,p,c,lmax,mmax,triton.next_power_of_2(2*lmax+1),
                             triton.next_power_of_2(c2),enable_fp_fusion=False)
        return dx,None,dw,None,None


class _Scatter(Function):
    @staticmethod
    def forward(ctx,x,w,targets,n_nodes,env,lmax,mmax):
        ctx.save_for_backward(x,w,targets,env);ctx.orders=(lmax,mmax);ctx.n_nodes=n_nodes
        x,w,targets,env=(t.contiguous() for t in (x,w,targets,env))
        e,p,c=x.shape;s=(lmax+1)**2
        y=x.new_zeros((n_nodes,s,c))
        if e:_scatter_fwd[(e,s)](x,w,targets,env,y,s,p,c,lmax,mmax,triton.next_power_of_2(c),enable_fp_fusion=False)
        return y

    @staticmethod
    def backward(ctx,dy):
        x,w,targets,env=ctx.saved_tensors;lmax,mmax=ctx.orders
        if torch.is_grad_enabled():
            from fastmd.models.esen_ops.esen_inference import rotate_scatter_torch
            needed=(ctx.needs_input_grad[0],ctx.needs_input_grad[1],ctx.needs_input_grad[4]);inputs=(x,w,env)
            grads=iter(torch.autograd.grad(rotate_scatter_torch(x,w,targets,ctx.n_nodes,env,lmax,mmax),
                [t for t,n in zip(inputs,needed) if n],dy,create_graph=True))
            dx,dw,de=(next(grads) if n else None for n in needed)
            return dx,dw,None,None,de,None,None
        x,w,targets,env,dy=(t.contiguous() for t in (x,w,targets,env,dy))
        e,p,c=x.shape;s=(lmax+1)**2
        dx=torch.empty_like(x);dw=torch.zeros_like(w);de=x.new_empty((e,p))
        if e:
            _scatter_dx[(e,p)](x,w,targets,env,dy,dx,de,s,p,c,lmax,mmax,
                               triton.next_power_of_2(c),enable_fp_fusion=False)
            _scatter_dw[(e,s)](x,targets,env,dy,dw,s,p,c,lmax,mmax,triton.next_power_of_2(2*lmax+1),
                               triton.next_power_of_2(c),enable_fp_fusion=False)
        return dx,dw,None,None,de.sum(1),None,None


def rotate_gather(x,edges,w,lmax,mmax):
    if _eligible(x):return _Gather.apply(x,edges,w,lmax,mmax)
    from fastmd.models.esen_ops.esen_inference import rotate_gather_torch
    return rotate_gather_torch(x,edges,w,lmax,mmax)


def rotate_scatter(x,w,targets,n_nodes,envelope,lmax,mmax):
    if _eligible(x):return _Scatter.apply(x,w,targets,n_nodes,envelope,lmax,mmax)
    from fastmd.models.esen_ops.esen_inference import rotate_scatter_torch
    return rotate_scatter_torch(x,w,targets,n_nodes,envelope,lmax,mmax)
