"""Fusion C: Triton fused LayerNorm + activation (SiLU/Sigmoid).

Replaces nn.LayerNorm(x) -> SiLU/Sigmoid (two kernels + 2 HBM round-trips of the
[rows,128] tensor) with one fused kernel (1 read, 1 write). act='silu' for the
GatedMLP core, 'sigmoid' for the gate, 'none' for plain LN (edge_init_norm).

Validated standalone (fwd allclose vs nn.LayerNorm+act, FP64 gradcheck) before
model integration. D=128 fixed (one row = one program).
"""
import torch, triton
import torch.nn.functional as F
import triton.language as tl
from torch.autograd import Function

from fastmd.engine import recorder as _recorder

ACT_NONE, ACT_SILU, ACT_SIGMOID = 0, 1, 2


@triton.jit
def _ln_act_fwd(x, w, b, out, mean, rstd, D: tl.constexpr, BLK: tl.constexpr, EPS: tl.constexpr, ACT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLK); m = c < D
    xr = tl.load(x + row * D + c, mask=m, other=0.0).to(tl.float32)
    mu = tl.sum(xr, axis=0) / D
    xc = tl.where(m, xr - mu, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rs = 1.0 / tl.sqrt(var + EPS)
    xhat = xc * rs
    y = xhat * tl.load(w + c, mask=m, other=0.0) + tl.load(b + c, mask=m, other=0.0)
    if ACT == 1:
        o = y * tl.sigmoid(y)
    elif ACT == 2:
        o = tl.sigmoid(y)
    else:
        o = y
    tl.store(out + row * D + c, o, mask=m)
    tl.store(mean + row, mu)
    tl.store(rstd + row, rs)


@triton.jit
def _ln_act_bwd(x, w, b, dout, mean, rstd, dx, dw_part, db_part,
               D: tl.constexpr, BLK: tl.constexpr, ACT: tl.constexpr, NEED_WB: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLK); m = c < D
    xr = tl.load(x + row * D + c, mask=m, other=0.0).to(tl.float32)
    mu = tl.load(mean + row); rs = tl.load(rstd + row)
    xhat = (xr - mu) * rs
    wv = tl.load(w + c, mask=m, other=0.0)
    y = xhat * wv + tl.load(b + c, mask=m, other=0.0)
    go = tl.load(dout + row * D + c, mask=m, other=0.0).to(tl.float32)
    # d(out)/d(y)
    sig = tl.sigmoid(y)
    if ACT == 1:
        dy = go * (sig * (1.0 + y * (1.0 - sig)))
    elif ACT == 2:
        dy = go * (sig * (1.0 - sig))
    else:
        dy = go
    # weight/bias partial grads (per-row; reduced across rows by caller) — only when training
    if NEED_WB:
        tl.store(dw_part + row * D + c, dy * xhat, mask=m)
        tl.store(db_part + row * D + c, dy, mask=m)
    # LayerNorm input grad
    dxhat = dy * wv
    mean_dxhat = tl.sum(tl.where(m, dxhat, 0.0), axis=0) / D
    mean_dxhat_xhat = tl.sum(tl.where(m, dxhat * xhat, 0.0), axis=0) / D
    dxr = (dxhat - mean_dxhat - xhat * mean_dxhat_xhat) * rs
    tl.store(dx + row * D + c, dxr, mask=m)


def _ln_act_torch(x, weight, bias, eps, act):
    y = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return {ACT_NONE: lambda: y, ACT_SILU: lambda: F.silu(y), ACT_SIGMOID: lambda: torch.sigmoid(y)}[act]()


class _LNAct(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps, act):
        rec = _recorder._ACTIVE
        if rec is not None:
            copied = not x.is_contiguous()
        x_in = x
        x = x.contiguous()
        N, D = x.shape
        BLK = triton.next_power_of_2(D)
        out = torch.empty_like(x)
        mean = torch.empty(N, device=x.device, dtype=torch.float32)
        rstd = torch.empty(N, device=x.device, dtype=torch.float32)
        _ln_act_fwd[(N,)](x, weight.contiguous(), bias.contiguous(), out, mean, rstd,
                          D=D, BLK=BLK, EPS=eps, ACT=act)
        # Save the inputs as given: the create_graph backward differentiates through them
        # (a strided x therefore keeps its base alive and is copied again for the kernel).
        ctx.save_for_backward(x_in, weight, bias, mean, rstd)
        ctx.eps, ctx.act = eps, act
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("layernorm_act", "triton", {
                "rows": N, "width": D, "BLK": BLK, "EPS": eps, "ACT": act,
                "dtype": x.dtype, "contiguous_copy": copied,
            })
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        if torch.is_grad_enabled():
            # create_graph=True (force-loss training differentiates this backward):
            # the Triton kernel is not differentiable, so differentiate torch ops.
            need = ctx.needs_input_grad[:3]
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("layernorm_act", "torch_create_graph", {
                    "rows": x.shape[0], "width": x.shape[-1], "ACT": ctx.act,
                }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
            y = _ln_act_torch(x, weight, bias, ctx.eps, ctx.act)
            grads = iter(torch.autograd.grad(
                y, [t for t, n in zip((x, weight, bias), need) if n], dout, create_graph=True))
            return (*(next(grads) if n else None for n in need), None, None)
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        N, D = x.shape
        BLK = triton.next_power_of_2(D)
        dx = torch.empty_like(x)
        need_wb = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("layernorm_act", "triton", {
                "rows": N, "width": D, "BLK": BLK, "ACT": ctx.act, "NEED_WB": bool(need_wb),
                "contiguous_copy": not dout.is_contiguous(),
            }, phase="bwd", site=getattr(ctx, "_opsel_site", None))
        if need_wb:
            dw_part = torch.empty_like(x); db_part = torch.empty_like(x)
            _ln_act_bwd[(N,)](x, weight, bias, dout.contiguous(), mean, rstd, dx, dw_part, db_part,
                              D=D, BLK=BLK, ACT=ctx.act, NEED_WB=True)
            return dx, dw_part.sum(0), db_part.sum(0), None, None
        # inference (forces): only dx needed -> skip weight/bias grad work
        _ln_act_bwd[(N,)](x, weight, bias, dout.contiguous(), mean, rstd, dx, dx, dx,
                          D=D, BLK=BLK, ACT=ctx.act, NEED_WB=False)
        return dx, None, None, None, None


def fused_ln_act(x, weight, bias, eps=1e-5, act=ACT_SILU):
    if x.is_cuda:
        return _LNAct.apply(x, weight, bias, eps, act)
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("layernorm_act", "torch", {"rows": x.shape[0], "width": x.shape[-1], "ACT": act})
    return _ln_act_torch(x, weight, bias, eps, act)


if __name__ == "__main__":
    dev = "cuda"; torch.manual_seed(0)
    # The kernel computes in FP32 (x, mean and rstd are FP32 even for FP64 inputs), so
    # errors are measured against an FP64 reference, relative to its largest magnitude.
    # The last case has low-variance and constant rows, where EPS matters.
    rel = lambda a, r: ((a.double() - r).abs().max() / r.abs().max()).item()
    for act, fn in [(ACT_SILU, F.silu), (ACT_SIGMOID, torch.sigmoid), (ACT_NONE, lambda z: z)]:
        for (N, D, s) in [(26174, 128, 1.0), (152446, 128, 1.0), (1165, 128, 1.0), (4096, 128, 1e-2)]:
            x = torch.randn(N, D, device=dev) * s; w = torch.randn(D, device=dev); b = torch.randn(D, device=dev)
            if s < 1:
                x[:64] = 0.25
            x1 = x.double().requires_grad_(True); w1 = w.double().requires_grad_(True); b1 = b.double().requires_grad_(True)
            x2 = x.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True); b2 = b.clone().requires_grad_(True)
            ref = fn(F.layer_norm(x1, (D,), w1, b1, 1e-5))
            tri = fused_ln_act(x2, w2, b2, 1e-5, act)
            fe = rel(tri, ref)
            g = torch.randn_like(tri); ref.backward(g.double()); tri.backward(g)
            de = max(rel(x2.grad, x1.grad), rel(w2.grad, w1.grad), rel(b2.grad, b1.grad))
            print(f"act={act} N={N} scale={s:g}: fwd={fe:.2e} grad={de:.2e} (relative to FP64)")
            assert fe < 1e-5 and de < 1e-5
    # FP64 gradcheck: x is rounded to FP32 inside the kernel, so a finite-difference step
    # of 1e-6 only measures that rounding; use a step and tolerance above FP32 resolution.
    # Double backward (create_graph) differentiates FP64 torch ops: default tolerances.
    Ns, Ds = 40, 16
    xe = torch.randn(Ns, Ds, device=dev, dtype=torch.float64, requires_grad=True)
    we = torch.randn(Ds, device=dev, dtype=torch.float64, requires_grad=True)
    be = torch.randn(Ds, device=dev, dtype=torch.float64, requires_grad=True)
    for act in (ACT_SILU, ACT_SIGMOID, ACT_NONE):
        f = lambda a, c, d: _LNAct.apply(a, c, d, 1e-5, act)
        ok = torch.autograd.gradcheck(f, (xe, we, be), eps=1e-3, atol=1e-3, rtol=1e-3)
        ok2 = torch.autograd.gradgradcheck(f, (xe, we, be))
        print(f"act={act}: FP64 gradcheck (step 1e-3): {ok}; gradgradcheck: {ok2}")
