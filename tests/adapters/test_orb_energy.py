"""Safe whole-energy math, real MD negotiation, and explicit kernel lifecycle.

CPU AOTAutograd tests cannot establish CUDA Inductor correctness or timing.
"""
import copy
import importlib.metadata
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

try:
    ORB = importlib.metadata.version('orb-models') == '0.5.4'
except importlib.metadata.PackageNotFoundError:
    ORB = False


@unittest.skipUnless(ORB, 'requires optional orb-models==0.5.4')
class ORBEnergyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(211)
        self.old_threads = torch.get_num_threads()
        self.old_precision = (torch.get_float32_matmul_precision(),
                              torch.backends.cuda.matmul.allow_tf32,
                              torch.backends.cudnn.allow_tf32)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.addCleanup(self.restore_process_state)

    def restore_process_state(self):
        torch.set_num_threads(self.old_threads)
        precision, matmul, cudnn = self.old_precision
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn

    @staticmethod
    def model(direct=False):
        from orb_models.forcefield.atomic_system import SystemConfig
        from orb_models.forcefield.pretrained import (
            orb_v3_conservative_architecture, orb_v3_direct_architecture)
        factory = orb_v3_direct_architecture if direct else orb_v3_conservative_architecture
        return factory(latent_dim=8, base_mlp_hidden_dim=16, base_mlp_depth=1,
                       head_mlp_hidden_dim=8, head_mlp_depth=1,
                       num_message_passing_steps=2, device='cpu',
                       system_config=SystemConfig(radius=6., max_num_neighbors=20))

    def adapter(self, model=None, direct=False, rewrite=True, steady=False):
        from fastmd.models.orb_graph import ORBGraphAdapter
        adapter = ORBGraphAdapter(self.model(direct=direct) if model is None else model)
        if rewrite:
            adapter.enable_operator_rewrites('torch', steady_options=(
                dict(prune=True, merge_node=True) if steady else None))
        self.addCleanup(adapter.close)
        return adapter

    @staticmethod
    def batch(empty=False, padded=False):
        from fastmd.adapters.protocol import GraphBatch, PaddingRule
        positions = torch.tensor([[0., 0., 0.], [1.3, .2, .1]])
        edges = [] if empty else [[0, 1], [1, 0]]
        images = [] if empty else [[0., 0., 0.]] * 2
        n_edges = len(edges)
        if padded:
            edges += [[0, 0], [1, 1]]
            images += [[1., 0., 0.]] * 2
        cell = 30. * torch.eye(3)
        return GraphBatch(positions=positions, frac_coords=positions / 30., cell=cell,
            species=torch.tensor([55, 53]), n_real=2, cutoff=6.,
            edge_index=torch.tensor(edges, dtype=torch.int32).reshape(-1, 2),
            edge_image=torch.tensor(images).reshape(-1, 3),
            n_edges=torch.tensor([n_edges]),
            padding=PaddingRule('far_self_loops') if padded else None)

    @staticmethod
    def evaluate(adapter, batch):
        from fastmd.engine.energy_derivative import EnergyDerivative
        result = EnergyDerivative(adapter)(batch, adapter.prepare(batch), task='efs')
        return result['e'], result['f'][0], result['s'][0]

    def aot_compiler(self):
        original = torch.compile

        def compile_cpu(function, **options):
            return original(function, backend='aot_eager', dynamic=options['dynamic'],
                            fullgraph=options['fullgraph'])
        return mock.patch.object(torch, 'compile', side_effect=compile_cpu)

    def test_aot_energy_force_stress_empty_padding_and_nonzero_zbl(self):
        raw = self.model()
        reference = self.adapter(copy.deepcopy(raw), steady=True)
        candidate = self.adapter(copy.deepcopy(raw), steady=True)
        with self.aot_compiler():
            candidate.enable_energy_compile()
        for empty, padded in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(empty=empty, padded=padded):
                batch = self.batch(empty=empty, padded=padded)
                for observed, expected in zip(self.evaluate(candidate, batch), self.evaluate(reference, batch)):
                    self.assertTrue(bool(observed.isfinite().all()))
                    torch.testing.assert_close(observed, expected, atol=3e-6, rtol=3e-5)
        batch = self.batch()
        graph = candidate._graph(candidate.prepare(batch), batch.positions, batch.cell)
        self.assertGreater(float(candidate.module.pair_repulsion_fn(graph)['energy']), .1)

    def test_true_eager_switch_default_fingerprint_and_disable_before_prepare(self):
        adapter = self.adapter(steady=True)
        original_inputs = adapter.system_inputs()
        original_choices = adapter.kernel_backends()
        original_contract = adapter.capture_contract()
        original_forward = adapter.system_energy
        original_head = adapter.module.heads[adapter.module.energy_name].forward
        prior_compiled = object()
        adapter.module.model._compiled_call_impl = prior_compiled
        calls = []

        def compiled(function, **options):
            def call(*args, **kwargs):
                calls.append(1)
                return function(*args, **kwargs)
            return call

        with mock.patch.object(torch, 'compile', side_effect=compiled):
            adapter.enable_energy_compile()
        self.assertTrue(adapter.compiled)
        self.assertEqual(adapter.capture_contract().compiled, 'capture_safe')
        self.assertEqual(adapter.kernel_backends()[0].site, 'energy')
        self.assertEqual(adapter.system_inputs()['energy_compile']['backend'], 'inductor')
        with self.assertRaisesRegex(ValueError, 'disable'):
            adapter.select_kernel('backbone', 'eager')
        adapter.select_kernel('energy', 'eager')
        self.assertFalse(adapter.compiled)
        self.assertEqual(adapter.capture_contract().compiled, 'none')
        # Use another adapter only to prepare inputs; the candidate stays unbound
        # so its own before-prepare state machine can be tested with real calls.
        input_adapter = self.adapter(copy.deepcopy(self.model()))
        batch = self.batch(padded=True)
        inputs = input_adapter.prepare(batch)
        eager = adapter.system_energy(inputs, batch.positions, batch.cell)
        self.assertEqual(calls, [])
        adapter.select_kernel('energy', 'inductor')
        compiled_value = adapter.system_energy(inputs, batch.positions, batch.cell)
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(compiled_value, eager, rtol=0, atol=0)
        handle = adapter._energy_installation
        adapter.disable_energy_compile()
        self.assertTrue(handle.restored)
        self.assertTrue(all(outcome.reverted for outcome in handle.outcomes))
        self.assertIs(adapter.module.model._compiled_call_impl, prior_compiled)
        self.assertEqual(adapter.system_inputs(), original_inputs)
        self.assertEqual(adapter.kernel_backends(), original_choices)
        self.assertEqual(adapter.capture_contract(), original_contract)
        self.assertEqual(adapter.system_energy, original_forward)
        self.assertEqual(adapter.module.heads[adapter.module.energy_name].forward, original_head)
        adapter.module.model._compiled_call_impl = None
        adapter.prepare(batch)
        adapter.disable_energy_compile()  # Already disabled: idempotent even after prepare.
        with self.assertRaisesRegex(RuntimeError, 'before prepare'):
            adapter.enable_energy_compile()

    def test_real_md_topology_and_generic_negotiation_select_energy_site(self):
        # TensorGraphBuilder requires CUDA even with a CPU MD device. Construct
        # only the minimal CPU state consumed by the real MD negotiation method;
        # do not mock that method, negotiate, profiles, or adapter selection.
        # Full CUDA runtime construction/capture remains a GPU validation task.
        from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
        from fastmd.config import InferenceConfig, resolve_config
        for profile, expected in (('topology', 'inductor'), ('generic', 'eager')):
            with self.subTest(profile=profile):
                adapter = self.adapter(steady=True)
                with self.aot_compiler():
                    adapter.enable_energy_compile()
                md = object.__new__(GPUResidentMolecularDynamics)
                md._adapter = adapter
                md.model = adapter.module
                md.device = torch.device('cpu')
                md.pbc = torch.ones(3, dtype=torch.bool)
                md._execution_auto = False
                md.log_stress = False
                md._inference_config = resolve_config(InferenceConfig(optimization_profile=profile,
                    execution='eager', matmul_precision='highest'), default_execution='eager')
                plan = md._negotiate_plan()
                self.assertEqual(plan.decision('kernels.energy').chosen, expected)
                self.assertFalse(any(d.id == 'kernels.backbone' for d in plan.decisions))
                self.assertEqual(adapter._energy_installation.backend, expected)
                self.assertEqual(adapter.compiled, expected == 'inductor')
                self.assertIs(md._engine_plan[0], adapter)
                for value in self.evaluate(adapter, self.batch()):
                    self.assertTrue(bool(value.isfinite().all()))
                adapter.select_kernel('energy', expected)  # same selection remains legal
                with self.assertRaisesRegex(RuntimeError, 'before prepare'):
                    adapter.select_kernel('energy', 'eager' if expected == 'inductor' else 'inductor')
                with self.assertRaisesRegex(RuntimeError, 'before prepare'):
                    adapter.disable_energy_compile()

    def test_instance_and_parameter_storage_isolation_close_order(self):
        from orb_models.forcefield import segment_ops
        from orb_models.forcefield.forcefield_heads import EnergyHead
        from orb_models.forcefield.pair_repulsion import ZBLBasis
        from orb_models.forcefield.gns import MoleculeGNS, AttentionInteractionNetwork
        from fastmd.models.orb_graph import ORBConservativeGraphAdapter
        raw = self.model()
        reference = self.adapter(copy.deepcopy(raw))
        candidate = self.adapter(copy.deepcopy(raw), steady=True)
        aggregate = segment_ops.aggregate_nodes
        parameters = {name: (p.clone(), p.data_ptr(), p.stride())
                      for name, p in candidate.module.named_parameters()}
        with mock.patch.object(torch, 'compile', side_effect=lambda fn, **kw: fn):
            candidate.enable_energy_compile()
        handle = candidate._energy_installation
        steady = candidate._base._steady_installation
        self.assertTrue(all(outcome in candidate.upstream_patches() for outcome in handle.outcomes))
        self.assertIs(reference.system_energy.__func__, ORBConservativeGraphAdapter.system_energy)
        self.assertIs(reference.module.heads[reference.module.energy_name].forward.__func__, EnergyHead.forward)
        self.assertIs(reference.module.pair_repulsion_fn.forward.__func__, ZBLBasis.forward)
        self.assertIs(segment_ops.aggregate_nodes, aggregate)
        candidate.prepare(self.batch())
        candidate.close()
        candidate.close()
        self.assertTrue(handle.restored and steady.restored)
        self.assertTrue(all(outcome.reverted for outcome in handle.outcomes))
        self.assertIs(candidate.system_energy.__func__, ORBConservativeGraphAdapter.system_energy)
        self.assertIs(candidate.module.model.forward.__func__, MoleculeGNS.forward)
        self.assertIs(candidate.module.model.gnn_stacks[0].forward.__func__, AttentionInteractionNetwork.forward)
        self.assertIs(candidate.module.heads[candidate.module.energy_name].forward.__func__, EnergyHead.forward)
        self.assertIs(candidate.module.pair_repulsion_fn.forward.__func__, ZBLBasis.forward)
        self.assertIsNone(candidate.module.model._compiled_call_impl)
        for name, p in candidate.module.named_parameters():
            value, pointer, stride = parameters[name]
            torch.testing.assert_close(p, value, rtol=0, atol=0)
            self.assertEqual((p.data_ptr(), p.stride()), (pointer, stride))

    def test_guards_source_frozen_precision_and_multi_system(self):
        from fastmd.models.orb_ops import orb_energy
        with self.assertRaisesRegex(ValueError, 'conservative'):
            self.adapter(direct=True).enable_energy_compile()
        with self.assertRaisesRegex(ValueError, 'operator rewrites'):
            self.adapter(rewrite=False).enable_energy_compile()
        adapter = self.adapter()
        with self.assertRaisesRegex(TypeError, 'bool'):
            adapter.enable_energy_compile(autotune=1)
        with mock.patch.dict(orb_energy._PINS, {'EnergyHead.forward': '0' * 64}):
            with self.assertRaisesRegex(RuntimeError, 'source SHA256'):
                adapter.enable_energy_compile()
        parameter = next(adapter.module.parameters())
        parameter.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen FP32'):
            adapter.enable_energy_compile()
        parameter.requires_grad_(False)
        adapter.module.double()
        with self.assertRaisesRegex(ValueError, 'FP32'):
            adapter.enable_energy_compile()
        adapter.module.float()
        adapter.module.train()
        with self.assertRaisesRegex(ValueError, 'eval'):
            adapter.enable_energy_compile()
        adapter.module.eval()
        torch.backends.cudnn.allow_tf32 = True
        with self.assertRaisesRegex(ValueError, 'TF32 off'):
            adapter.enable_energy_compile()
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, 'autocast off'):
                adapter.enable_energy_compile()
        with self.assertRaisesRegex(ValueError, 'one nonempty'):
            orb_energy._single_system_aggregate(torch.ones(2, 3), torch.tensor([1, 1]), 'mean')
        invalid_batch = self.batch()
        invalid_batch.n_real = 1
        with self.assertRaisesRegex(ValueError, 'without sink rows'):
            adapter.prepare(invalid_batch)
        self.assertIsNone(adapter._energy_installation)
        self.assertNotIn('forward', vars(adapter.module.heads[adapter.module.energy_name]))

    def test_failed_compile_or_patch_rolls_back_and_retry_works(self):
        from fastmd.models.orb_ops import orb_energy
        adapter = self.adapter(steady=True)
        original = adapter.system_energy
        head = adapter.module.heads[adapter.module.energy_name]
        zbl = adapter.module.pair_repulsion_fn
        original_head, original_zbl = head.forward, zbl.forward
        prior_compiled = object()
        adapter.module.model._compiled_call_impl = prior_compiled
        for point in ('compile', 'third_patch'):
            count = []
            apply = orb_energy.apply_patches

            def fail_third(*args, **kwargs):
                count.append(1)
                if len(count) == 3:
                    raise RuntimeError('test third patch failure')
                return apply(*args, **kwargs)

            compiler = (mock.Mock(side_effect=RuntimeError('test compilation failure')) if point == 'compile'
                        else lambda fn, **kw: fn)
            with mock.patch.object(torch, 'compile', side_effect=compiler), \
                 mock.patch.object(orb_energy, 'apply_patches', side_effect=fail_third):
                with self.assertRaisesRegex(RuntimeError, 'test .* failure'):
                    adapter.enable_energy_compile()
            self.assertIsNone(adapter._energy_installation)
            self.assertIs(adapter.module.model._compiled_call_impl, prior_compiled)
            self.assertEqual(adapter.system_energy, original)
            self.assertEqual(head.forward, original_head)
            self.assertEqual(zbl.forward, original_zbl)
        with mock.patch.object(torch, 'compile', side_effect=lambda fn, **kw: fn):
            adapter.enable_energy_compile()
        with self.assertRaisesRegex(RuntimeError, 'already enabled'):
            adapter.enable_energy_compile()
        with self.assertRaisesRegex(RuntimeError, 'Disable energy'):
            adapter.enable_operator_rewrites('torch')
        adapter.disable_energy_compile()
        self.assertIs(adapter.module.model._compiled_call_impl, prior_compiled)
        adapter.module.model._compiled_call_impl = None

    def test_autotune_options_are_local_and_execution_checks_precision(self):
        from torch._inductor import config
        adapter = self.adapter()
        previous = (config.max_autotune, config.force_same_precision, config.triton.cudagraphs)
        with mock.patch.object(torch, 'compile', side_effect=lambda fn, **kw: fn) as compile_call:
            adapter.enable_energy_compile(autotune=True)
        options = compile_call.call_args.kwargs['options']
        self.assertEqual(options, dict(force_same_precision=True, **{'triton.cudagraphs': False},
            max_autotune=True, max_autotune_gemm_backends='ATEN,TRITON',
            max_autotune_gemm_search_space='DEFAULT', epilogue_fusion=True))
        self.assertEqual((config.max_autotune, config.force_same_precision, config.triton.cudagraphs), previous)
        self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32)
        batch = self.batch()
        inputs = adapter.prepare(batch)
        torch.backends.cuda.matmul.allow_tf32 = True
        with self.assertRaisesRegex(ValueError, 'TF32 off'):
            adapter.system_energy(inputs, batch.positions, batch.cell)
        torch.backends.cuda.matmul.allow_tf32 = False
        with self.assertRaisesRegex(ValueError, 'FP32 positions'):
            adapter.system_energy(inputs, batch.positions.double(), batch.cell)


if __name__ == '__main__':
    unittest.main()
