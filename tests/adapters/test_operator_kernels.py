"""Run with TRITON_INTERPRET=1 on CPU, or FASTMD_TEST_DEVICE=cuda on an idle GPU."""
import os
import unittest
import torch

DEVICE=os.environ.get('FASTMD_TEST_DEVICE','cpu')


def compare_gradients(test,reference,actual,inputs,atol=3e-5,rtol=3e-5):
    refs=(reference,) if isinstance(reference,torch.Tensor) else reference
    acts=(actual,) if isinstance(actual,torch.Tensor) else actual
    for r,a in zip(refs,acts):torch.testing.assert_close(a,r,atol=atol,rtol=rtol)
    weights=tuple(torch.randn_like(r) for r in refs)
    expected=torch.autograd.grad(refs,inputs,weights,retain_graph=True)
    observed=torch.autograd.grad(acts,inputs,weights)
    for r,a in zip(expected,observed):
        test.assertTrue(bool(a.isfinite().all()))
        torch.testing.assert_close(a,r,atol=atol,rtol=rtol)


class OperatorKernels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19);torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False

    def test_compact_topk_stable_ties_empty_padding_and_unsorted(self):
        from fastmd.models.orb_ops.orb_inference import compact_topk_torch
        from fastmd.models.orb_ops.triton_orb_inference import compact_topk
        for e,k in ((0,3),(1,20),(79,1),(137,3),(309,20),(519,120)):
            with self.subTest(edges=e,k=k):
                n=7
                src=torch.randint(n,(e,),device=DEVICE)
                # Deliberate ties, exact-cutoff edges, and shuffled senders.
                dist=torch.randint(0,40,(e,),device=DEVICE).float()
                limit=torch.tensor([max(0,e-9)],device=DEVICE)
                for lim in (None,limit):
                    expected=compact_topk_torch(src,dist,n,k,6.,lim)
                    actual=compact_topk(src,dist,n,k,6.,lim)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_dual_gate_aggregation_forward_and_all_input_gradients(self):
        from fastmd.models.orb_ops.orb_inference import gated_aggregate_torch
        from fastmd.models.orb_ops.triton_orb_inference import gated_aggregate
        for e,d in ((0,7),(17,7),(21,33),(37,256)):
            with self.subTest(edges=e,width=d):
                value=torch.randn(e,d*2,device=DEVICE)[:,::2].requires_grad_(True)
                send=torch.randn(e,1,device=DEVICE,requires_grad=True)
                receive=torch.randn(e,1,device=DEVICE,requires_grad=True)
                cut=torch.rand(e,1,device=DEVICE,requires_grad=True)
                src=torch.randint(5,(e,),device=DEVICE);dst=torch.randint(5,(e,),device=DEVICE)
                expected=gated_aggregate_torch(value,send,receive,cut,src,dst,5)
                actual=gated_aggregate(value,send,receive,cut,src,dst,5)
                compare_gradients(self,expected,actual,(value,send,receive,cut))

    def test_gate_input_gradients_and_noncontiguous_channels(self):
        from fastmd.models.esen_ops.esen_inference import gate_torch
        from fastmd.models.esen_ops.triton_esen_nonlinear import fused_gate
        for n,c,l,m in ((3,7,2,1),(4,128,3,2),(2,8,3,3)):
            with self.subTest(channels=c,l=l,m=m):
                index=torch.tensor([i-1 for i in range(1,l+1) for _ in range(min(2*i+1,2*m+1))],device=DEVICE)
                x=torch.randn(n,len(index)+1,c*2,device=DEVICE)[:,:,::2].requires_grad_(True)
                gate=torch.randn(n,1,l*c,device=DEVICE,requires_grad=True)
                compare_gradients(self,gate_torch(gate,x,index),fused_gate(gate,x,index,l),(gate,x))

    def test_degree_rmsnorm_centering_weights_and_small_variance(self):
        from fastmd.models.esen_ops.esen_inference import rms_norm_torch
        from fastmd.models.esen_ops.triton_esen_nonlinear import fused_rms_norm
        for c,l,scale in ((7,2,1.),(128,3,1.),(8,3,1e-4),(8,3,0.)):
            with self.subTest(channels=c,l=l,scale=scale):
                s=(l+1)**2
                index=torch.tensor([i for i in range(l+1) for _ in range(2*i+1)],device=DEVICE)
                balance=torch.tensor([1/((l+1)*(2*i+1)) for i in range(l+1) for _ in range(2*i+1)],device=DEVICE)[:,None]
                x=(torch.randn(3,s,c*2,device=DEVICE)[:,:,::2]*scale).requires_grad_(True)
                w=torch.randn(l+1,c,device=DEVICE);b=torch.randn(c,device=DEVICE)
                expected=rms_norm_torch(x,w,b,index,balance,1e-5)
                actual=fused_rms_norm(x,w,b,index,balance,1e-5)
                compare_gradients(self,expected,actual,(x,),atol=2e-4,rtol=3e-5)

    def test_block_rotations_include_matrix_and_envelope_derivatives(self):
        from fastmd.models.esen_ops.triton_esen_rotation import rotate_gather,rotate_scatter
        for e,c,l,m in ((0,7,2,1),(3,7,2,1),(5,16,3,2),(3,128,3,3)):
            with self.subTest(edges=e,channels=c,l=l,m=m):
                s=(l+1)**2;n=4
                index=torch.tensor([j for degree in range(l+1)
                    for j in range(degree**2+max(degree-m,0),(degree+1)**2-max(degree-m,0))],device=DEVICE)
                blocks=[torch.randn(e,2*d+1,2*d+1,device=DEVICE,requires_grad=True) for d in range(l+1)]
                matrix=torch.zeros(e,s,s,device=DEVICE)
                for d,block in enumerate(blocks):matrix[:,d*d:(d+1)**2,d*d:(d+1)**2]=block
                nodes=torch.randn(n,s,c,device=DEVICE,requires_grad=True)
                edges=torch.randint(n,(2,e),device=DEVICE)
                ref=torch.bmm(matrix[:,index,:],torch.cat((nodes[edges[0]],nodes[edges[1]]),dim=2))
                actual=rotate_gather(nodes,edges,matrix,l,m)
                compare_gradients(self,ref,actual,(nodes,*blocks),atol=5e-5,rtol=5e-5)
                # A fresh autograd graph for inverse rotation and weighted scatter.
                matrix=torch.zeros(e,s,s,device=DEVICE)
                for d,block in enumerate(blocks):matrix[:,d*d:(d+1)**2,d*d:(d+1)**2]=block
                inverse=matrix.transpose(1,2)
                values=torch.randn(e,len(index),c,device=DEVICE,requires_grad=True)
                envelope=torch.rand(e,device=DEVICE,requires_grad=True)
                rotated=torch.bmm(inverse[:,:,index],values*envelope[:,None,None])
                ref=values.new_zeros((n,s,c)).index_add(0,edges[1],rotated)
                actual=rotate_scatter(values,inverse,edges[1],n,envelope,l,m)
                compare_gradients(self,ref,actual,(values,envelope,*blocks),atol=6e-5,rtol=5e-5)

    def test_second_derivative_fallback_preserves_strided_inputs(self):
        from fastmd.models.orb_ops.orb_inference import gated_aggregate_torch
        from fastmd.models.orb_ops.triton_orb_inference import gated_aggregate
        from fastmd.models.esen_ops.esen_inference import gate_torch, rms_norm_torch
        from fastmd.models.esen_ops.esen_inference import rotate_gather_torch, rotate_scatter_torch
        from fastmd.models.esen_ops.triton_esen_nonlinear import fused_gate, fused_rms_norm
        from fastmd.models.esen_ops.triton_esen_rotation import rotate_gather, rotate_scatter

        def check(ref_fn, actual_fn, inputs):
            ref, actual = ref_fn(), actual_fn()
            refs = (ref,) if isinstance(ref, torch.Tensor) else ref
            acts = (actual,) if isinstance(actual, torch.Tensor) else actual
            weights = tuple(torch.randn_like(t) for t in refs)
            first_ref = torch.autograd.grad(refs, inputs, weights, create_graph=True)
            first_actual = torch.autograd.grad(acts, inputs, weights, create_graph=True)
            for expected, observed in zip(first_ref, first_actual):
                torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-5)
            tangent = tuple(torch.randn_like(t) for t in inputs)
            second_ref = torch.autograd.grad(first_ref, inputs, tangent, allow_unused=True)
            second_actual = torch.autograd.grad(first_actual, inputs, tangent, allow_unused=True)
            for expected, observed in zip(second_ref, second_actual):
                self.assertEqual(observed is None, expected is None)
                if expected is not None:
                    torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-5)

        value = torch.randn(3,14,device=DEVICE)[:,::2].requires_grad_()
        a,b,cut = [torch.randn(3,1,device=DEVICE,requires_grad=True) for _ in range(3)]
        src = torch.tensor([0,0,1],device=DEVICE)
        dst = torch.tensor([1,2,0],device=DEVICE)
        args = (value,a,b,cut,src,dst,3)
        check(lambda:gated_aggregate_torch(*args),lambda:gated_aggregate(*args),(value,a,b,cut))
        index = torch.tensor([0]*3+[1]*3,device=DEVICE)
        x = torch.randn(2,7,8,device=DEVICE)[:,:,::2].requires_grad_()
        g = torch.randn(2,8,device=DEVICE,requires_grad=True)
        check(lambda:gate_torch(g,x,index),lambda:fused_gate(g,x,index,2),(g,x))
        index = torch.tensor([0]+[1]*3+[2]*5,device=DEVICE)
        balance = torch.tensor([1/3]+[1/9]*3+[1/15]*5,device=DEVICE)[:,None]
        x = torch.randn(2,9,8,device=DEVICE)[:,:,::2].requires_grad_()
        w = torch.randn(3,4,device=DEVICE);b = torch.randn(4,device=DEVICE)
        args = (x,w,b,index,balance,1e-5)
        check(lambda:rms_norm_torch(*args),lambda:fused_rms_norm(*args),(x,))
        edges = torch.stack((src,dst))
        nodes = torch.randn(3,9,8,device=DEVICE)[:,:,::2].requires_grad_()
        matrix = torch.randn(3,9,9,device=DEVICE).transpose(1,2).requires_grad_()
        args = (nodes,edges,matrix,2,1)
        check(lambda:rotate_gather_torch(*args),lambda:rotate_gather(*args),(nodes,matrix))
        values = torch.randn(3,7,8,device=DEVICE)[:,:,::2].requires_grad_()
        env = torch.randn(3,device=DEVICE,requires_grad=True)
        args = (values,matrix,dst,3,env,2,1)
        check(lambda:rotate_scatter_torch(*args),lambda:rotate_scatter(*args),(values,matrix,env))


if __name__=='__main__':unittest.main()
