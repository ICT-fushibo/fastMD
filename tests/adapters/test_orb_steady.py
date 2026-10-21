"""FP32 conservative ORB output/gradient parity and instance patch lifecycle."""
import copy
import importlib.metadata
import itertools
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

try:
    ORB = importlib.metadata.version("orb-models") == "0.5.4"
except importlib.metadata.PackageNotFoundError:
    ORB = False


@unittest.skipUnless(ORB, "requires optional orb-models==0.5.4")
class ORBSteadyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(216)
        self.previous_threads = torch.get_num_threads()
        self.previous_precision = torch.get_float32_matmul_precision()
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.addCleanup(torch.set_float32_matmul_precision, self.previous_precision)

    def model(self, direct=False):
        from orb_models.forcefield.atomic_system import SystemConfig
        from orb_models.forcefield.pretrained import (
            orb_v3_conservative_architecture, orb_v3_direct_architecture)
        factory = orb_v3_direct_architecture if direct else orb_v3_conservative_architecture
        return factory(latent_dim=8, base_mlp_hidden_dim=16, base_mlp_depth=1,
                       head_mlp_hidden_dim=8, head_mlp_depth=1,
                       num_message_passing_steps=2, device="cpu",
                       system_config=SystemConfig(radius=6., max_num_neighbors=20))

    def adapter(self, model=None, direct=False, rewrite=True):
        from fastmd.models.orb_graph import ORBGraphAdapter
        adapter = ORBGraphAdapter(self.model(direct=direct) if model is None else model)
        if rewrite:
            adapter.enable_operator_rewrites("torch")
        self.addCleanup(adapter.close)
        return adapter

    @staticmethod
    def graph(empty=False, padded=False):
        senders = [] if empty else [0, 1, 1, 2]
        receivers = [] if empty else [1, 0, 2, 1]
        vectors = [] if empty else [[2., .2, .1], [-2., -.2, -.1], [.1, 2., .3], [-.1, -2., -.3]]
        if padded:
            senders += [0, 1, 2]
            receivers += [0, 1, 2]
            vectors += [[12., 0., 0.]] * 3
        vectors = torch.tensor(vectors, dtype=torch.float32).reshape(-1, 3).requires_grad_()
        numbers = torch.tensor([55, 50, 53])
        graph = SimpleNamespace(senders=torch.tensor(senders, dtype=torch.long),
            receivers=torch.tensor(receivers, dtype=torch.long),
            node_features={"atomic_numbers": numbers,
                           "atomic_numbers_embedding": torch.eye(118)[numbers]},
            edge_features={"vectors": vectors})
        return graph, vectors

    def test_outputs_and_vector_gradients_padding_empty_and_all_switches(self):
        from fastmd.models.orb_ops.orb_steady import install
        adapter = self.adapter()
        backbone = adapter.module.model
        state = {name: value.clone() for name, value in adapter.module.state_dict().items()}
        cotangent = torch.randn(3, 8)
        for empty, padded in itertools.product((False, True), repeat=2):
            graph, vectors = self.graph(empty=empty, padded=padded)
            expected = backbone(graph)["node_features"]
            expected_grad = torch.autograd.grad((expected * cotangent).sum(), vectors)[0]
            for prune, merge_node, merge_attention in itertools.product((False, True), repeat=3):
                options = dict(prune=prune, merge_node=merge_node, merge_attention=merge_attention)
                with self.subTest(empty=empty, padded=padded, **options):
                    graph, vectors = self.graph(empty=empty, padded=padded)
                    decode_calls = []
                    hook = backbone._decoder.register_forward_hook(lambda *args: decode_calls.append(1))
                    try:
                        with install(adapter, **options) as installed:
                            output = backbone(graph)
                            actual = output["node_features"]
                            actual_grad = torch.autograd.grad((actual * cotangent).sum(), vectors)[0]
                            self.assertEqual(set(output), {"node_features"} if prune else
                                             {"node_features", "edge_features", "pred"})
                            self.assertEqual(len(decode_calls), 0 if prune else 1)
                            self.assertTrue(all(p.applied for p in installed.outcomes))
                    finally:
                        hook.remove()
                    self.assertTrue(bool(actual.isfinite().all()))
                    self.assertTrue(bool(actual_grad.isfinite().all()))
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-5, atol=3e-6)
                    self.assertTrue(all(p.reverted for p in installed.outcomes))
                    self.assertNotIn("forward", vars(backbone))
        for name, value in state.items():
            torch.testing.assert_close(adapter.module.state_dict()[name], value, rtol=0, atol=0)

    def test_layer_all_input_gradients_and_instance_isolation(self):
        from fastmd.models.orb_ops.orb_inference import interaction
        from fastmd.models.orb_ops.orb_steady import install
        from orb_models.forcefield.gns import AttentionInteractionNetwork
        raw = self.model()
        baseline = self.adapter(copy.deepcopy(raw))
        candidate = self.adapter(copy.deepcopy(raw))
        original_class_forward = AttentionInteractionNetwork.forward
        inputs = (torch.randn(5, 8, requires_grad=True),
                  torch.randn(13, 8, requires_grad=True),
                  torch.rand(13, 1, requires_grad=True))
        senders, receivers = torch.randint(5, (13,)), torch.randint(5, (13,))
        reference = baseline.module.model.gnn_stacks[0](*inputs[:2], senders, receivers, inputs[2])
        weights = tuple(torch.randn_like(value) for value in reference)
        expected_grad = torch.autograd.grad(reference, inputs, weights, retain_graph=True)
        with install(candidate, prune=True, merge_node=True, merge_attention=True):
            actual = candidate.module.model.gnn_stacks[0](*inputs[:2], senders, receivers, inputs[2])
            observed_grad = torch.autograd.grad(actual, inputs, weights)
            for expected, observed in zip((*reference, *expected_grad), (*actual, *observed_grad)):
                torch.testing.assert_close(observed, expected, rtol=3e-5, atol=3e-6)
            self.assertIs(AttentionInteractionNetwork.forward, original_class_forward)
            self.assertIs(baseline.module.model.gnn_stacks[0].forward.__func__, interaction)
        self.assertIs(candidate.module.model.gnn_stacks[0].forward.__func__, interaction)
        candidate.close()
        self.assertIs(candidate.module.model.gnn_stacks[0].forward.__func__, original_class_forward)
        self.assertIs(baseline.module.model.gnn_stacks[0].forward.__func__, interaction)

    def test_refuses_direct_changed_source_precision_training_and_reentry(self):
        from fastmd.models.orb_ops import orb_steady
        direct = self.adapter(direct=True)
        with self.assertRaisesRegex(ValueError, "conservative"):
            orb_steady.install(direct, prune=True)
        adapter = self.adapter()
        with mock.patch.dict(orb_steady._PINS, {"MoleculeGNS.forward": "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "source SHA256"):
                orb_steady.install(adapter, prune=True)
        self.assertNotIn("forward", vars(adapter.module.model))
        adapter.module.double()
        with self.assertRaisesRegex(ValueError, "FP32"):
            orb_steady.install(adapter, merge_node=True)
        adapter.module.float()
        adapter.module.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            orb_steady.install(adapter, prune=True)
        adapter.module.eval()
        with orb_steady.install(adapter, prune=True):
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                orb_steady.install(adapter)
        adapter._base._in_use = True
        with self.assertRaisesRegex(RuntimeError, "before prepare"):
            orb_steady.install(adapter)

    def test_failed_install_and_context_exception_restore_original_callable(self):
        from fastmd.models.orb_ops import orb_steady
        from fastmd.models.orb_ops.orb_inference import interaction
        adapter = self.adapter()
        backbone = adapter.module.model
        prior_callable = object()
        backbone._compiled_call_impl = prior_callable
        calls = []
        original_apply = orb_steady.apply_patches

        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("simulated second per-instance patch failure")
            return original_apply(*args, **kwargs)

        with mock.patch.object(orb_steady, "apply_patches", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "second per-instance"):
                orb_steady.install(adapter, prune=True, merge_node=True, merge_attention=True)
        self.assertNotIn("forward", vars(backbone))
        self.assertIs(backbone._compiled_call_impl, prior_callable)
        for layer in backbone.gnn_stacks:
            self.assertIs(layer.forward.__func__, interaction)
            self.assertFalse(any(name.startswith("_orb_steady_") for name in vars(layer)))
        self.assertNotIn("_orb_steady_installed", vars(backbone))
        with self.assertRaisesRegex(RuntimeError, "user failure"):
            with orb_steady.install(adapter, prune=True, merge_attention=True) as handle:
                raise RuntimeError("user failure")
        self.assertIs(backbone._compiled_call_impl, prior_callable)
        self.assertTrue(handle.restored)
        handle.close()  # idempotent; adapter.close subsequently unwinds Torch.
        self.assertIs(backbone.gnn_stacks[0].forward.__func__, interaction)
        backbone._compiled_call_impl = None

    def test_adapter_api_default_fingerprint_forwarding_and_close(self):
        from orb_models.forcefield.gns import MoleculeGNS, AttentionInteractionNetwork
        baseline = self.adapter(rewrite=False)
        self.assertEqual(baseline.system_inputs(), {
            "max_neighbors": 20, "checkpoint": {},
            "neighbor_selection": "nearest_k_fixed_shape_v1", "operator_backend": "upstream"})
        baseline.enable_operator_rewrites("torch")
        self.assertEqual(baseline._base.system_inputs(), {"operator_backend": "torch"})
        self.assertEqual(baseline.system_inputs(), {
            "max_neighbors": 20, "checkpoint": {},
            "neighbor_selection": "nearest_k_compact_v2", "operator_backend": "torch"})
        self.assertFalse(any(outcome.record.name.startswith("orb.steady.")
                             for outcome in baseline.upstream_patches()))

        candidate = self.adapter(rewrite=False)
        options = dict(prune=True, merge_attention=True)
        candidate.enable_operator_rewrites("torch", steady_options=options)
        options["prune"] = False
        info = candidate.system_inputs()
        self.assertEqual(info["steady_options"],
                         dict(prune=True, merge_node=False, merge_attention=True))
        self.assertEqual(info["steady_source_pins"], candidate._base.system_inputs()["steady_source_pins"])
        handle = candidate._base._steady_installation
        self.assertTrue(all(outcome in candidate.upstream_patches() for outcome in handle.outcomes))
        self.assertEqual(len(handle.outcomes), 3)
        candidate.close()
        candidate.close()
        self.assertTrue(handle.restored)
        self.assertTrue(all(outcome.reverted for outcome in handle.outcomes))
        self.assertIs(candidate.module.model.forward.__func__, MoleculeGNS.forward)
        for layer in candidate.module.model.gnn_stacks:
            self.assertIs(layer.forward.__func__, AttentionInteractionNetwork.forward)
            self.assertFalse(any(name.startswith("_orb_steady_") for name in vars(layer)))
        self.assertNotIn("steady_options", candidate.system_inputs())
        self.assertIsNone(candidate.module.model._compiled_call_impl)

    def test_adapter_api_invalid_options_and_second_stage_failure_are_atomic(self):
        from fastmd.models.orb_ops import orb_steady
        from orb_models.forcefield.gns import MoleculeGNS, AttentionInteractionNetwork
        adapter = self.adapter(rewrite=False)
        backbone = adapter.module.model
        prior_callable = object()
        backbone._compiled_call_impl = prior_callable

        def assert_unmodified():
            self.assertEqual(adapter._base.operator_backend, "upstream")
            self.assertEqual(adapter._base._operator_patches, [])
            self.assertIsNone(adapter._base._steady_installation)
            self.assertIs(backbone._compiled_call_impl, prior_callable)
            self.assertIs(backbone.forward.__func__, MoleculeGNS.forward)
            for layer in backbone.gnn_stacks:
                self.assertIs(layer.forward.__func__, AttentionInteractionNetwork.forward)
                self.assertFalse(any(name.startswith(("_orb_steady_", "_fastmd_")) for name in vars(layer)))

        for backend, options, error in (
                ("torch", [], TypeError), ("torch", {"prune": 1}, TypeError),
                ("torch", {"unknown": True}, ValueError),
                ("triton", {"prune": True}, ValueError)):
            with self.subTest(backend=backend, options=options):
                with self.assertRaises(error):
                    adapter.enable_operator_rewrites(backend, steady_options=options)
                assert_unmodified()
        with mock.patch.dict(orb_steady._PINS, {"MoleculeGNS.forward": "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "source SHA256"):
                adapter.enable_operator_rewrites("torch", steady_options={"prune": True})
        assert_unmodified()

        original_apply = orb_steady.apply_patches
        patch_calls = []
        def fail_second(*args, **kwargs):
            patch_calls.append(1)
            if len(patch_calls) == 2:
                raise RuntimeError("simulated API second-stage failure")
            return original_apply(*args, **kwargs)
        with mock.patch.object(orb_steady, "apply_patches", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "API second-stage"):
                adapter.enable_operator_rewrites("torch", steady_options={"prune": True, "merge_node": True})
        assert_unmodified()
        adapter._base.kernel = "inductor"
        with mock.patch.object(backbone, "compile", side_effect=RuntimeError("simulated compile failure")):
            with self.assertRaisesRegex(RuntimeError, "compile failure"):
                adapter.enable_operator_rewrites("torch", steady_options={"prune": True})
        assert_unmodified()
        adapter._base.kernel = "eager"
        adapter.enable_operator_rewrites("torch", steady_options={"prune": True})
        self.assertEqual(adapter._base.operator_backend, "torch")
        self.assertTrue(adapter._base._steady_installation.options["prune"])

    def test_adapter_api_refuses_direct_and_prepare_then_allows_valid_retry(self):
        from orb_models.forcefield.gns import AttentionInteractionNetwork
        direct = self.adapter(direct=True, rewrite=False)
        with self.assertRaisesRegex(ValueError, "conservative"):
            direct.enable_operator_rewrites("torch", steady_options={"prune": True})
        self.assertEqual(direct._base.operator_backend, "upstream")
        self.assertEqual(direct._base._operator_patches, [])
        self.assertIs(direct.module.model.gnn_stacks[0].forward.__func__, AttentionInteractionNetwork.forward)
        direct.enable_operator_rewrites("torch")
        adapter = self.adapter(rewrite=False)
        adapter._base._in_use = True
        with self.assertRaisesRegex(RuntimeError, "before preparing"):
            adapter.enable_operator_rewrites("torch", steady_options={"prune": True})
        self.assertEqual(adapter._base.operator_backend, "upstream")


if __name__ == "__main__":
    unittest.main()
