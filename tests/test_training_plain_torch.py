"""Training (is_training=True) runs every operator on its plain torch path.

Fused kernels are inference-only: their backward is not differentiable, so a
force loss (forces keep create_graph) silently lost its second-order terms.
CUDA force-loss parameter gradients must match the same model on CPU (pure
torch) for legacy without any scope (as example/full_finetuning.py), the
topology profile, activation checkpointing (the model's enable_checkpoint,
which stays the training lever and warns nothing) and a frozen
embedding+interaction preset; no matris autograd Function may run while training, while inference
on the same graph does apply them. That inference keeps its kernel selection,
energy and scope bindings across a training step is a tier-3 perf guard
(MATRIS_RUN_PERF_TIER=1).
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import copy
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
from ase import Atoms

from fastmd.config import OPTION_REGISTRY
from fastmd.config import InferenceConfig
from fastmd.config import resolve_config
from fastmd.graph.gpu_graph_builder import atoms_to_graph_gpu
from fastmd.models.matris.model import MatRIS
from matris_test_tiers import perf_tier


def _model():
    torch.manual_seed(3)
    # LayerNorm + SiLU + 3-frequency Fourier: every fused path is eligible.
    return MatRIS(num_layers=2, node_feat_dim=16, edge_feat_dim=16, three_body_feat_dim=16,
                  mlp_hidden_dims=(16,), num_radial=7, num_angular=7, norm_type="layer",
                  three_body_expansion="fourier", pairwise_cutoff=5.0, three_body_cutoff=3.6,
                  use_bias=True, enable_compile=False, enable_checkpoint=False).eval()


def _atoms():
    # 180 atoms, ~75k triplets and ~10k bonds: above the fused row thresholds.
    rng = np.random.default_rng(5)
    grid = np.stack(np.meshgrid(*(np.arange(n) for n in (6, 6, 5)), indexing="ij"), -1).reshape(-1, 3)
    positions = (grid + rng.uniform(-0.12, 0.12, grid.shape)) * 2.1
    numbers = rng.choice([1, 6, 7, 8], len(grid))
    return Atoms(numbers=numbers, positions=positions, cell=np.diag([6, 6, 5]) * 2.1, pbc=True)


def _recorder():
    seen = Counter()
    original = torch.autograd.Function.__dict__["apply"].__func__

    def apply(cls, *args, **kwargs):
        if cls.__module__.startswith("fastmd."):
            seen[cls.__qualname__] += 1
        return original(cls, *args, **kwargs)

    return seen, patch.object(torch.autograd.Function, "apply", classmethod(apply))


def _scope(profile):
    if profile is None:
        return nullcontext()
    return resolve_config(InferenceConfig(execution="eager", optimization_profile=profile,
                                          report=False)).scope()


def _bound_state():
    state = {}
    for spec in OPTION_REGISTRY.values():
        for module_name, attribute in spec.bindings:
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, attribute):
                state[module_name, attribute] = getattr(module, attribute)
    from fastmd.engine.policy import fusion_scope_active
    from fastmd.models.matris import functions
    from fastmd.models.matris import interaction_block
    state["checkpoint"] = interaction_block.checkpoint
    state["activations"] = getattr(functions, "_FUSED_ACTIVATIONS", None)
    state["fusion_scope"] = fusion_scope_active()
    return state


def _inference(model, graph, profile):
    """An is_training=False forward: (matris autograd Functions applied, energy)."""
    seen, recorder = _recorder()
    with recorder, _scope(profile):
        out = model([graph], task="ef", is_training=False)
    return dict(seen), out["e"].detach().clone()


def _force_loss_grads(model, graph, weights, profile=None, frozen=False, checkpoint=False):
    model.requires_grad_(True)
    if frozen:
        for module in (model.atom_embedding, model.edge_embedding, model.three_body_embedding,
                       model.interaction_block):
            module.requires_grad_(False)
    for layer in model.interaction_block:
        layer.enable_checkpoint = checkpoint
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    seen, recorder = _recorder()
    try:
        with recorder, _scope(profile):
            pred = model([graph], task="ef", is_training=True)
            force = torch.cat(list(pred["f"]), 0)
            loss = 0.1 * pred["e"].sum() + (force * weights.to(force)).sum()
            grads = torch.autograd.grad(loss, params, allow_unused=True)  # e.g. the magmom head
    finally:
        model.eval().requires_grad_(False)
        for layer in model.interaction_block:
            layer.enable_checkpoint = False
    used = tuple(g is not None for g in grads)
    return used, torch.cat([g.detach().double().cpu().reshape(-1) for g in grads if g is not None]), seen


@unittest.skipUnless(torch.cuda.is_available(), "requires coordinator-assigned idle CUDA device")
class TrainingPlainTorchGPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        cls.model = _model().cuda().requires_grad_(False)
        cls.cpu = copy.deepcopy(cls.model).cpu()
        atoms = _atoms()
        cls.graph = atoms_to_graph_gpu(atoms, device=torch.device("cuda"),
                                       atom_graph_cutoff=cls.model.config["pairwise_cutoff"],
                                       line_graph_cutoff=cls.model.config["three_body_cutoff"],
                                       check_isolated_atoms=False)
        cls.graph_cpu = cls.graph.to("cpu")
        cls.weights = torch.randn(len(atoms), 3, generator=torch.Generator().manual_seed(7))

    @classmethod
    def tearDownClass(cls):
        torch.set_float32_matmul_precision(cls.precision)

    def _reference(self, frozen):
        used, grads, _ = _force_loss_grads(self.cpu, self.graph_cpu, self.weights, frozen=frozen)
        return used, grads

    def test_force_loss_gradients_match_cpu_reference(self):
        # Preconditions: the graph is above the fused row thresholds and
        # inference on it applies fused/custom kernels, so their absence in
        # training below is not vacuous.
        self.assertGreater(len(self.graph.line_graph), 60000)
        self.assertGreater(len(self.graph.undirected2directed), 5000)
        for profile in ("legacy", "topology"):
            with self.subTest(inference=profile):
                self.assertTrue(_inference(self.model, self.graph, profile)[0],
                                "inference no longer selects fused kernels")
        cases = {"legacy (no scope)": dict(), "legacy": dict(profile="legacy"),
                 "legacy (no scope) + checkpoint": dict(checkpoint=True),
                 "topology": dict(profile="topology"),
                 "topology + checkpoint": dict(profile="topology", checkpoint=True),
                 "frozen, legacy (no scope)": dict(frozen=True),
                 "frozen, topology": dict(profile="topology", frozen=True)}
        references = {frozen: self._reference(frozen) for frozen in (False, True)}
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    used, grads, seen = _force_loss_grads(self.model, self.graph, self.weights, **kwargs)
                # enable_checkpoint stays the training lever: only InferenceConfig.checkpoint is deprecated
                self.assertEqual([str(w.message) for w in caught if "checkpoint" in str(w.message)], [])
                reference_used, reference = references[kwargs.get("frozen", False)]
                self.assertEqual(used, reference_used)
                self.assertLess(float((grads - reference).norm() / reference.norm()), 1e-4)
                self.assertEqual(dict(seen), {}, "a fused/custom kernel ran in training")

    @perf_tier
    def test_inference_selection_unchanged_by_training(self):
        for profile in ("legacy", "topology"):
            with self.subTest(profile=profile):
                state = _bound_state()
                before, energy_before = _inference(self.model, self.graph, profile)
                self.assertTrue(before, "inference no longer selects fused kernels")
                _force_loss_grads(self.model, self.graph, self.weights, profile=profile, checkpoint=True)
                after, energy_after = _inference(self.model, self.graph, profile)
                self.assertEqual(after, before)
                torch.testing.assert_close(energy_after, energy_before, rtol=1e-6, atol=1e-6)
                restored = _bound_state()  # lazily imported op modules only add keys
                self.assertEqual({k: restored[k] for k in state}, state)


class PlainTorchScopeTest(unittest.TestCase):
    """The scope mechanism itself, on CPU."""

    def _plain_state(self):
        from fastmd.engine.kernels import segment
        from fastmd.engine.policy import fusion_scope_active
        from fastmd.models.matris import functions
        from fastmd.models.matris import interaction_block
        return (segment._FUSED_SEGMENT_SOFTMAX, functions._LN_FUSE_THRESH, functions._FUSED_ACTIVATIONS,
                interaction_block._FUSED_DIRECTED_PAIR_AGGREGATE, fusion_scope_active())

    def test_binds_plain_values_and_restores_after_an_error(self):
        from fastmd.config import PLAIN_TORCH_OPTIONS
        from fastmd.engine.policy import fusion_scope
        from fastmd.models.matris import interaction_block
        from fastmd.models.matris.model import plain_torch_scope

        self.enterContext(fusion_scope())  # a runner's scope is cleared inside, restored after
        state, checkpoint = _bound_state(), interaction_block.checkpoint
        with self.assertRaises(RuntimeError), plain_torch_scope():
            self.assertEqual(self._plain_state(), (False, 1_000_000_000, False, False, False))
            for name, value in PLAIN_TORCH_OPTIONS.items():
                for module_name, attribute in OPTION_REGISTRY[name].bindings:
                    module = sys.modules.get(module_name)
                    if module is not None and hasattr(module, attribute):
                        self.assertEqual(getattr(module, attribute), value, name)
            raise RuntimeError
        self.assertEqual(_bound_state(), state)
        self.assertIs(interaction_block.checkpoint, checkpoint)

    def test_checkpoint_recomputes_with_plain_dispatch_after_the_scope(self):
        from fastmd.models.matris import interaction_block
        from fastmd.models.matris.model import plain_torch_scope

        outside, calls = self._plain_state(), []

        def layer(x):
            calls.append(self._plain_state())
            return (x * x).sin()

        x = torch.randn(8, requires_grad=True)
        with plain_torch_scope():
            y = interaction_block.checkpoint(layer, x, use_reentrant=False)
        y.sum().backward()  # recomputation runs here, outside the scope
        plain = (False, 1_000_000_000, False, False, False)
        self.assertEqual(calls, [plain, plain])
        self.assertEqual(self._plain_state(), outside)

    def test_dispatch_restores_on_the_last_exit_in_any_order(self):
        from fastmd.models.matris.model import _PlainTorchDispatch

        outside = self._plain_state()
        first, second = _PlainTorchDispatch(), _PlainTorchDispatch()
        first.__enter__()
        second.__enter__()
        first.__exit__(None, None, None)  # e.g. two autograd threads recomputing
        self.assertEqual(self._plain_state()[0], False)
        second.__exit__(None, None, None)
        self.assertEqual(self._plain_state(), outside)


if __name__ == "__main__":
    unittest.main()
