"""Instance-scoped reduced linear arithmetic and its force-gradient lifecycle."""
from __future__ import annotations

import copy
import json
import unittest
from collections import OrderedDict
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from fastmd.accuracy import apply_layered_precision, discover_linear_groups


def frozen(module):
    return module.eval().requires_grad_(False)


class ToyEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.message = nn.Linear(3, 12)
        self.readout = nn.Linear(12, 1)

    def forward(self, positions):
        return self.readout(torch.sin(self.message(positions))).sum()


class LayeredPrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = frozen(ToyEnergy())
        self.groups = {"message": ["message"]}
        self.x = torch.randn(8, 3)

    def install(self, dtype="bf16", **kwargs):
        return apply_layered_precision(self.model, {"message": dtype}, groups=self.groups, **kwargs)

    def test_fp32_is_exact_identity_and_state_dict_is_unchanged(self):
        original = self.model.message.forward
        checkpoint = {key: value.clone() for key, value in self.model.state_dict().items()}
        x = self.x.clone().requires_grad_()
        expected = self.model(x)
        force = -torch.autograd.grad(expected, x)[0]
        with self.install("fp32") as installation:
            self.assertEqual(self.model.message.forward, original)
            y = self.model(x)
            torch.testing.assert_close(y, expected, rtol=0, atol=0)
            torch.testing.assert_close(-torch.autograd.grad(y, x)[0], force, rtol=0, atol=0)
            self.assertEqual(installation.manifest()["coverage"]["patched_linear_modules"], 0)
            self.assertEqual(installation.manifest()["cached_weight_bytes"], 0)
        self.assertEqual(set(checkpoint), set(self.model.state_dict()))
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, checkpoint[name], rtol=0, atol=0)

    def test_forward_and_input_force_match_explicit_cast_program(self):
        original_forward = self.model.readout.forward
        original_parameter = self.model.message.weight
        checkpoint = copy.deepcopy(self.model.state_dict())
        for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            with self.subTest(dtype=name), self.install(name) as installation:
                x = self.x.clone().requires_grad_()
                actual = self.model(x)
                force = -torch.autograd.grad(actual, x)[0]
                reference_x = self.x.clone().requires_grad_()
                low = F.linear(reference_x.to(dtype), original_parameter.to(dtype),
                               self.model.message.bias.to(dtype)).float()
                expected = self.model.readout(torch.sin(low)).sum()
                expected_force = -torch.autograd.grad(expected, reference_x)[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(force, expected_force, rtol=0, atol=0)
                self.assertEqual(force.dtype, torch.float32)
                self.assertTrue(torch.isfinite(force).all())
                self.assertIs(self.model.message.weight, original_parameter)
                self.assertEqual(self.model.readout.forward, original_forward)
                installation.validate()
                manifest = installation.manifest()
                json.dumps(manifest)
                self.assertEqual(manifest["coverage"]["linear_module_fraction"], 0.5)
                self.assertEqual(manifest["linear_sites"][0]["forward_compute_dtype"], str(dtype)[6:])
                self.assertEqual(manifest["linear_sites"][0]["backward_input_gradient_compute_dtype"], str(dtype)[6:])
                self.assertEqual(set(checkpoint), set(self.model.state_dict()))
                for key, value in self.model.state_dict().items():
                    torch.testing.assert_close(value, checkpoint[key], rtol=0, atol=0)

    def test_restore_is_idempotent_and_exception_safe(self):
        expected = self.model(self.x)
        other = copy.deepcopy(self.model)
        original = self.model.message.forward
        with self.assertRaisesRegex(RuntimeError, "experiment failed"):
            with self.install() as installation:
                self.assertNotEqual(self.model.message.forward, original)
                self.assertEqual(other.message.forward.__func__, nn.Linear.forward)
                raise RuntimeError("experiment failed")
        self.assertFalse(installation.active)
        installation.restore()
        self.assertNotIn("forward", self.model.message.__dict__)
        self.assertEqual(self.model.message.forward, original)
        torch.testing.assert_close(self.model(self.x), expected, rtol=0, atol=0)
        self.assertFalse(any(key.startswith("_fastmd_layered") for key in self.model.message.__dict__))

    def test_rejects_bad_plan_before_mutating_any_layer(self):
        for precisions, groups in (({"protected": "bf16"}, self.groups),
                                   ({"unknown": "bf16"}, self.groups),
                                   ({"message": "int8"}, self.groups),
                                   ({"message": "bf16"}, {"message": ["missing"]})):
            with self.subTest(precisions=precisions, groups=groups), self.assertRaises(ValueError):
                apply_layered_precision(self.model, precisions, groups=groups)
            self.assertNotIn("forward", self.model.message.__dict__)
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            self.install()
        self.model.eval().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "freeze"):
            self.install()

    def test_conflicting_shared_aliases_are_rejected(self):
        self.model.alias = self.model.message
        with self.assertRaisesRegex(ValueError, "shared linear aliases"):
            self.install()
        with apply_layered_precision(self.model, {"shared": "bf16"},
                                     groups={"shared": ["message", "alias"]}) as installation:
            self.assertIs(self.model.message, self.model.alias)
            self.assertEqual(installation.manifest()["coverage"]["patched_linear_modules"], 1)

    def test_model_change_and_compilation_guards(self):
        with self.install() as installation:
            with self.assertRaisesRegex(RuntimeError, "already"):
                self.install()
            self.model.message.weight.add_(0.01)
            with self.assertRaisesRegex(RuntimeError, "weights changed"):
                installation.validate()
        with patch.object(self.model, "_compiled_call_impl", lambda x: x):
            with self.assertRaisesRegex(ValueError, "before torch.compile"):
                self.install()
        self.model.message._fastmd_projection_weights = ()
        with self.assertRaisesRegex(ValueError, "bypass nn.Linear"):
            self.install()

    def test_partial_installation_error_restores_all_forwards(self):
        original_setattr = nn.Module.__setattr__
        calls = []

        def fail_once(module, name, value):
            if module is self.model.readout and name == "forward" and not calls:
                calls.append(True)
                raise RuntimeError("simulated install failure")
            return original_setattr(module, name, value)

        with patch.object(nn.Module, "__setattr__", fail_once):
            with self.assertRaisesRegex(RuntimeError, "simulated install failure"):
                apply_layered_precision(self.model, {"both": "bf16"},
                                        groups={"both": ["message", "readout"]})
        for module in self.model.modules():
            self.assertFalse(any(name.startswith("_fastmd_layered") for name in module.__dict__))
        self.assertNotIn("forward", self.model.message.__dict__)
        self.assertNotIn("forward", self.model.readout.__dict__)

    def test_compile_after_installation_keeps_input_gradient(self):
        # Dynamo's eager backend validates capture of the bound forward and
        # casts without an expensive platform-dependent Inductor compilation.
        with self.install():
            compiled = torch.compile(self.model, backend="eager", fullgraph=True)
            x = self.x.clone().requires_grad_()
            expected = self.model(x)
            expected_force = torch.autograd.grad(expected, x)[0]
            actual = compiled(x)
            actual_force = torch.autograd.grad(actual, x)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_force, expected_force, rtol=0, atol=0)
            del compiled


class GroupDiscoveryTests(unittest.TestCase):
    def test_chgnet_protects_basis_embedding_and_readout(self):
        model = nn.Module()
        for name in ("atom_conv_layers", "bond_conv_layers", "angle_layers"):
            setattr(model, name, nn.ModuleList([nn.Sequential(nn.Linear(3, 3))]))
        for name in ("bond_embedding", "angle_embedding", "bond_weights_ag", "mlp", "site_wise"):
            setattr(model, name, nn.Linear(3, 3))
        groups = discover_linear_groups(model, family="chgnet")
        self.assertEqual(groups["atom_message"], ["atom_conv_layers.0.0"])
        self.assertEqual(groups["bond_message"], ["bond_conv_layers.0.0"])
        self.assertEqual(groups["angle_message"], ["angle_layers.0.0"])
        self.assertEqual(set(groups["protected"]), {"bond_embedding", "angle_embedding", "bond_weights_ag", "mlp", "site_wise"})

    def test_orb_protects_encoder_decoder_and_energy_head(self):
        model = nn.Module()
        model.model = nn.Module()
        layer = nn.Module()
        for name in ("_edge_mlp", "_node_mlp"):
            setattr(layer, name, nn.Sequential(OrderedDict(mlp=nn.Sequential(nn.Linear(3, 3)),
                                                         layer_norm=nn.LayerNorm(3))))
        for name in ("_receive_attn", "_send_attn", "_cond_node_proj"):
            setattr(layer, name, nn.Linear(3, 1))
        model.model.gnn_stacks = nn.ModuleList([layer])
        model.model._encoder = nn.Linear(3, 3)
        model.model._decoder = nn.Linear(3, 3)
        model.heads = nn.ModuleDict({"energy": nn.Linear(3, 1)})
        groups = discover_linear_groups(model, family="orb")
        self.assertEqual(groups["edge_message"], ["model.gnn_stacks.0._edge_mlp.mlp.0"])
        self.assertEqual(len(groups["attention"]), 2)
        self.assertIn("heads.energy", groups["protected"])
        self.assertIn("model._encoder", groups["protected"])
        self.assertIn("model._decoder", groups["protected"])
        self.assertIn("model.gnn_stacks.0._cond_node_proj", groups["protected"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for whole tensor-program capture")
class CUDACaptureTests(unittest.TestCase):
    def test_fixed_plan_force_program_replays(self):
        model = frozen(ToyEnergy().cuda())
        x = torch.randn(16, 3, device="cuda", requires_grad=True)
        with apply_layered_precision(model, {"message": "fp16"}, groups={"message": ["message"]}):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    energy = model(x)
                    torch.autograd.grad(energy, x)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                energy = model(x)
                force = -torch.autograd.grad(energy, x)[0]
            graph.replay()
            torch.cuda.synchronize()
            expected = model(x)
            expected_force = -torch.autograd.grad(expected, x)[0]
            torch.testing.assert_close(energy, expected)
            torch.testing.assert_close(force, expected_force)
            del graph


if __name__ == "__main__":
    unittest.main()
