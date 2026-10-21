"""ORB scalar gate/dual aggregation and stable per-atom compact top-K.

FP32 inference kernels. CPU/FP64 use the differentiable Torch reference.
TRITON_INTERPRET=1 permits testing the exact kernels on CPU, without a GPU.
"""
from __future__ import annotations
import os
import torch
import triton
import triton.language as tl
from torch.autograd import Function


def _eligible(value):
    return value.dtype==torch.float32 and (value.is_cuda or os.environ.get('TRITON_INTERPRET')=='1')


@triton.jit
def _aggregate_forward(V,SLOG,RLOG,CUT,SRC,DST,OS,OR,E:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    c=tl.arange(0,B)
    value=tl.load(V+row*D+c,c<D,other=0.)
    a=tl.sigmoid(tl.load(SLOG+row));b=tl.sigmoid(tl.load(RLOG+row));cut=tl.load(CUT+row)
    source=tl.load(SRC+row).to(tl.int64);target=tl.load(DST+row).to(tl.int64)
    tl.atomic_add(OS+source*D+c,value*(a*cut),c<D,sem='relaxed')
    tl.atomic_add(OR+target*D+c,value*(b*cut),c<D,sem='relaxed')


@triton.jit
def _aggregate_backward(V,SLOG,RLOG,CUT,SRC,DST,GS,GR,DV,DS,DR,DC,D:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);c=tl.arange(0,B)
    source=tl.load(SRC+row).to(tl.int64);target=tl.load(DST+row).to(tl.int64)
    value=tl.load(V+row*D+c,c<D,other=0.)
    gs=tl.load(GS+source*D+c,c<D,other=0.);gr=tl.load(GR+target*D+c,c<D,other=0.)
    a=tl.sigmoid(tl.load(SLOG+row));b=tl.sigmoid(tl.load(RLOG+row));cut=tl.load(CUT+row)
    da=tl.sum(gs*value,0);db=tl.sum(gr*value,0)
    tl.store(DV+row*D+c,gs*(a*cut)+gr*(b*cut),c<D)
    tl.store(DS+row,da*cut*a*(1.-a));tl.store(DR+row,db*cut*b*(1.-b))
    tl.store(DC+row,da*a+db*b)


class _GatedAggregate(Function):
    @staticmethod
    def forward(ctx,value,send,receive,cut,source,target,n_nodes):
        ctx.save_for_backward(value,send,receive,cut,source,target)
        value,send,receive,cut,source,target=(t.contiguous() for t in (value,send,receive,cut,source,target))
        e,d=value.shape
        out_s=value.new_zeros((n_nodes,d));out_r=torch.zeros_like(out_s)
        if e:
            _aggregate_forward[(e,)](value,send,receive,cut,source,target,out_s,out_r,
                                     e,d,triton.next_power_of_2(d),enable_fp_fusion=False)
        ctx.n_nodes=n_nodes
        return out_s,out_r

    @staticmethod
    def backward(ctx,gs,gr):
        value,send,receive,cut,source,target=ctx.saved_tensors
        if torch.is_grad_enabled():
            from fastmd.models.orb_ops.orb_inference import gated_aggregate_torch
            inputs=(value,send,receive,cut);needed=ctx.needs_input_grad[:4]
            outputs=gated_aggregate_torch(*inputs,source,target,ctx.n_nodes)
            grads=iter(torch.autograd.grad(outputs,[x for x,n in zip(inputs,needed) if n],(gs,gr),create_graph=True))
            return (*(next(grads) if n else None for n in needed),None,None,None)
        value,send,receive,cut,source,target=(t.contiguous() for t in (value,send,receive,cut,source,target))
        dv,ds,dr,dc=(torch.empty_like(t) for t in (value,send,receive,cut))
        e,d=value.shape
        if e:
            _aggregate_backward[(e,)](value,send,receive,cut,source,target,gs.contiguous(),gr.contiguous(),
                                      dv,ds,dr,dc,d,triton.next_power_of_2(d),enable_fp_fusion=False)
        return dv,ds,dr,dc,None,None,None


def gated_aggregate(value,send,receive,cut,source,target,n_nodes):
    if _eligible(value):
        return _GatedAggregate.apply(value,send,receive,cut,source,target,n_nodes)
    from fastmd.models.orb_ops.orb_inference import gated_aggregate_torch
    return gated_aggregate_torch(value,send,receive,cut,source,target,n_nodes)


@triton.jit
def _bounds(SRC,DIST,LIMIT,FIRST,LAST,E:tl.constexpr,RC2:tl.constexpr,HAS_LIMIT:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B)
    limit=E
    if HAS_LIMIT:limit=tl.load(LIMIT)
    live=row<tl.minimum(limit,E)
    source=tl.load(SRC+row,live,other=0).to(tl.int64)
    distance=tl.load(DIST+row,live,other=float('inf'))
    live=live&(distance<RC2)
    tl.atomic_min(FIRST+source,row,live,sem='relaxed')
    tl.atomic_max(LAST+source,row+1,live,sem='relaxed')


@triton.jit
def _compact_select(SRC,DIST,FIRST,LAST,OUT,K:tl.constexpr,RC2:tl.constexpr,B:tl.constexpr):
    node=tl.program_id(0).to(tl.int64)
    start=tl.load(FIRST+node);end=tl.load(LAST+node)
    lanes=tl.arange(0,B)
    sentinel=tl.full((B,),-1,tl.int64).to(tl.uint64)
    keys=sentinel
    for base in range(start,end,B-K):
        rows=base+lanes-K
        live=(lanes>=K)&(rows<end)
        source=tl.load(SRC+rows,live,other=-1)
        distance=tl.load(DIST+rows,live,other=float('inf'))
        live=live&(source==node)&(distance<RC2)
        # Nonnegative FP32 distance bits preserve numeric order; row breaks ties.
        new=(distance.to(tl.uint32,bitcast=True).to(tl.uint64)<<32)|rows.to(tl.uint32).to(tl.uint64)
        new=tl.where(live,new,sentinel)
        keys=tl.sort(tl.where(lanes<K,keys,new),descending=False)
    indices=tl.where(keys!=sentinel,(keys&0xffffffff).to(tl.int64),-1)
    tl.store(OUT+node*K+lanes,indices,lanes<K)


def compact_topk(senders,distances,n_nodes,k,cutoff,n_edges=None):
    if not _eligible(distances) or k>256:
        from fastmd.models.orb_ops.orb_inference import compact_topk_torch
        return compact_topk_torch(senders,distances,n_nodes,k,cutoff,n_edges)
    senders=senders.contiguous();distances=distances.contiguous()
    e=len(senders)
    if e>=2**31:raise ValueError('Compact top-K requires fewer than 2^31 candidate rows')
    first=torch.full((n_nodes,),e,dtype=torch.int32,device=senders.device)
    last=torch.zeros_like(first)
    output=torch.empty((n_nodes,k),dtype=torch.int64,device=senders.device)
    if e:
        _bounds[(triton.cdiv(e,256),)](senders,distances,n_edges if n_edges is not None else senders,
                                      first,last,e,cutoff*cutoff,n_edges is not None,256)
    if n_nodes:
        _compact_select[(n_nodes,)](senders,distances,first,last,output,k,cutoff*cutoff,
                                    triton.next_power_of_2(2*k),num_warps=4)
    return output
