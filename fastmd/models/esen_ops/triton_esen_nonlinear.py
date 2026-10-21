"""FP32 eSEN gate and degree-balanced, scalar-centered RMSNorm with input VJPs.

These are inference kernels: affine parameters must be frozen. Higher order
derivatives and training of affine parameters retain the Torch reference path.
"""
from __future__ import annotations
import os
import torch
import triton
import triton.language as tl
from torch.autograd import Function


def _eligible(x):
    return x.dtype==torch.float32 and (x.is_cuda or os.environ.get('TRITON_INTERPRET')=='1')


@triton.jit
def _gate_forward(G,X,INDEX,Y,S:tl.constexpr,C:tl.constexpr,L:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);offset=tl.arange(0,B)
    s=offset//C;c=offset%C;live=offset<S*C
    degree=tl.load(INDEX+s-1,live&(s>0),other=0).to(tl.int64)
    g=tl.load(G+row*L*C+degree*C+c,live,other=0.)
    x=tl.load(X+row*S*C+offset,live,other=0.)
    out=tl.where(s==0,x*tl.sigmoid(x),x*tl.sigmoid(g))
    tl.store(Y+row*S*C+offset,out,live)


@triton.jit
def _gate_backward(G,X,INDEX,DY,DX,DG,S:tl.constexpr,C:tl.constexpr,L:tl.constexpr,BS:tl.constexpr,BC:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    s=tl.arange(0,BS);c=tl.arange(0,BC)
    degree=tl.load(INDEX+s-1,(s>0)&(s<S),other=0).to(tl.int64)
    live=(s[:,None]<S)&(c[None,:]<C)
    offset=row*S*C+s[:,None]*C+c[None,:]
    x=tl.load(X+offset,live,other=0.);dy=tl.load(DY+offset,live,other=0.)
    g=tl.load(G+row*L*C+degree[:,None]*C+c[None,:],live,other=0.)
    sig=tl.sigmoid(g);sx=tl.sigmoid(x)
    dx=tl.where(s[:,None]==0,dy*sx*(1.+x*(1.-sx)),dy*sig)
    tl.store(DX+offset,dx,live)
    dg=dy*x*sig*(1.-sig)
    for l in tl.static_range(L):
        total=tl.sum(tl.where((s[:,None]>0)&(s[:,None]<S)&(degree[:,None]==l),dg,0.),0)
        tl.store(DG+row*L*C+l*C+c,total,c<C)


class _Gate(Function):
    @staticmethod
    def forward(ctx,g,x,index,lmax):
        ctx.original_shape=g.shape
        ctx.save_for_backward(g,x,index)
        g=g.contiguous();x=x.contiguous();index=index.contiguous()
        n,s,c=x.shape;y=torch.empty_like(x)
        if n:_gate_forward[(n,)](g,x,index,y,s,c,lmax,triton.next_power_of_2(s*c),enable_fp_fusion=False)
        ctx.lmax=lmax
        return y

    @staticmethod
    def backward(ctx,dy):
        g,x,index=ctx.saved_tensors;n,s,c=x.shape
        if torch.is_grad_enabled():
            from fastmd.models.esen_ops.esen_inference import gate_torch
            needed=ctx.needs_input_grad[:2]
            grads=iter(torch.autograd.grad(gate_torch(g,x,index),[t for t,k in zip((g,x),needed) if k],dy,create_graph=True))
            return (*(next(grads) if k else None for k in needed),None,None)
        g,x,index=(t.contiguous() for t in (g,x,index))
        dx=torch.empty_like(x);dg=torch.empty_like(g)
        if n:_gate_backward[(n,)](g,x,index,dy.contiguous(),dx,dg,s,c,ctx.lmax,
            triton.next_power_of_2(s),triton.next_power_of_2(c),enable_fp_fusion=False)
        return dg.reshape(ctx.original_shape),dx,None,None


def fused_gate(g,x,index,lmax):
    if _eligible(x):return _Gate.apply(g,x,index,lmax)
    from fastmd.models.esen_ops.esen_inference import gate_torch
    return gate_torch(g,x,index)


@triton.jit
def _rms_forward(X,W,BIAS,INDEX,BAL,Y,MEAN,RS,S:tl.constexpr,C:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);off=tl.arange(0,B)
    live=off<S*C;s=off//C;c=off%C
    x=tl.load(X+row*S*C+off,live,other=0.)
    mean=tl.sum(tl.where(off<C,x,0.),0)/C
    u=tl.where(s==0,x-mean,x)
    balance=tl.load(BAL+s,live,other=0.)
    variance=tl.sum(u*u*balance,0)/C
    rs=tl.rsqrt(variance+EPS)
    degree=tl.load(INDEX+s,live,other=0).to(tl.int64)
    weight=tl.load(W+degree*C+c,live,other=0.)
    bias=tl.load(BIAS+c,live&(s==0),other=0.)
    tl.store(Y+row*S*C+off,u*rs*weight+bias,live)
    tl.store(MEAN+row,mean);tl.store(RS+row,rs)


@triton.jit
def _rms_backward(X,W,INDEX,BAL,DY,MEAN,RS,DX,S:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);off=tl.arange(0,B)
    live=off<S*C;s=off//C;c=off%C
    x=tl.load(X+row*S*C+off,live,other=0.)
    mean=tl.load(MEAN+row);rs=tl.load(RS+row)
    u=tl.where(s==0,x-mean,x)
    degree=tl.load(INDEX+s,live,other=0).to(tl.int64)
    weight=tl.load(W+degree*C+c,live,other=0.)
    balance=tl.load(BAL+s,live,other=0.)
    grad=tl.load(DY+row*S*C+off,live,other=0.)*weight
    dot=tl.sum(grad*u,0)
    du=grad*rs-u*balance*(rs*rs*rs*dot/C)
    scalar_mean=tl.sum(tl.where(off<C,du,0.),0)/C
    tl.store(DX+row*S*C+off,tl.where(s==0,du-scalar_mean,du),live)


class _RMSNorm(Function):
    @staticmethod
    def forward(ctx,x,w,b,index,balance,eps):
        original=(x,w,b,index,balance)
        x=x.contiguous();w=w.contiguous();b=b.contiguous();index=index.contiguous();balance=balance.contiguous()
        n,s,c=x.shape;y=torch.empty_like(x)
        mean=x.new_empty(n);rs=x.new_empty(n)
        if n:_rms_forward[(n,)](x,w,b,index,balance,y,mean,rs,s,c,eps,
                                triton.next_power_of_2(s*c),enable_fp_fusion=False)
        ctx.save_for_backward(*original,mean,rs);ctx.eps=eps
        return y

    @staticmethod
    def backward(ctx,dy):
        x,w,b,index,balance,mean,rs=ctx.saved_tensors
        if torch.is_grad_enabled():
            from fastmd.models.esen_ops.esen_inference import rms_norm_torch
            dx=torch.autograd.grad(rms_norm_torch(x,w,b,index,balance,ctx.eps),x,dy,create_graph=True)[0]
            return dx,None,None,None,None,None
        x,w,index,balance=(t.contiguous() for t in (x,w,index,balance))
        n,s,c=x.shape;dx=torch.empty_like(x)
        if n:_rms_backward[(n,)](x,w,index,balance,dy.contiguous(),mean,rs,dx,s,c,
                                 triton.next_power_of_2(s*c),enable_fp_fusion=False)
        return dx,None,None,None,None,None


def fused_rms_norm(x,w,b,index,balance,eps):
    if _eligible(x) and not w.requires_grad and not b.requires_grad:
        return _RMSNorm.apply(x,w,b,index,balance,eps)
    from fastmd.models.esen_ops.esen_inference import rms_norm_torch
    return rms_norm_torch(x,w,b,index,balance,eps)
