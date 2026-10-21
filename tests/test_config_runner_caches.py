"""CPU-only regressions for configured runners and configuration-sensitive caches.

The compiled-artifact cache keying check (recompile counts) is a tier-3 perf
guard (MATRIS_RUN_PERF_TIER=1).
"""

import os
import unittest
from unittest.mock import patch

import torch

from fastmd.execution.cuda_graph import BucketedGraphRunner
from fastmd.config import CapacityConfig, InferenceConfig, resolve_config
from fastmd.models.matris import functions
from fastmd.models.matris.op import compiled_lowerings
from matris_test_tiers import perf_tier


class RunnerConfigurationTests(unittest.TestCase):
    def test_low_level_runner_inherits_profile_and_capacity(self):
        capacity = CapacityConfig(u_step=1024, t_step=16384, warmup=5, n_dummy=32, min_pad_u=128)
        original_flag = functions._INDEXED_CAT_LINEAR_FUSE
        with patch.dict(os.environ, {}, clear=True), patch.object(torch.cuda, "graph_pool_handle", return_value=object()):
            for profile, expected_fusions in (("generic", False), ("topology", True)):
                with self.subTest(profile=profile):
                    resolved = resolve_config(InferenceConfig(optimization_profile=profile, capacity=capacity, report=False))
                    with resolved.scope():
                        runner = BucketedGraphRunner(object())
                        self.assertEqual(runner.enable_model_fusions, expected_fusions)
                        self.assertEqual((runner.u_step, runner.t_step, runner.warmup), (1024, 16384, 5))
                        self.assertEqual((runner.n_dummy, runner.min_pad_u), (32, 128))
                        self.assertEqual(functions._INDEXED_CAT_LINEAR_FUSE, expected_fusions)
                    self.assertEqual(functions._INDEXED_CAT_LINEAR_FUSE, original_flag)

    def test_runner_rejects_conflicting_explicit_capacity(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(torch.cuda, "graph_pool_handle") as pool:
            resolved = resolve_config(InferenceConfig(capacity=CapacityConfig(u_step=1024), report=False))
            with resolved.scope():
                with self.assertRaises(ValueError):
                    BucketedGraphRunner(object(), u_step=512)
            pool.assert_not_called()


class CompiledArtifactConfigurationTests(unittest.TestCase):
    @perf_tier
    def test_artifact_pair_reuses_only_matching_compile_mode(self):
        fwd, vjp = object(), object()
        artifacts = compiled_lowerings._ArtifactPair(fwd, vjp)
        with patch.dict(os.environ, {}, clear=True), patch.object(torch.cuda, "is_available", return_value=False), patch.object(compiled_lowerings, "_compile", side_effect=lambda formula: (compiled_lowerings._COMPILE_MODE, formula)) as compile_fn:
            default = resolve_config(InferenceConfig(expert_overrides={"MATRIS_COMPILED_LOWERINGS_MODE": "default"}))
            tuned = resolve_config(InferenceConfig(expert_overrides={"MATRIS_COMPILED_LOWERINGS_MODE": "max-autotune-no-cudagraphs"}))
            with default.scope():
                first = artifacts.get()
                self.assertIs(first, artifacts.get())
            with tuned.scope():
                second = artifacts.get()
                self.assertIs(second, artifacts.get())
            with default.scope():
                self.assertIs(first, artifacts.get())
            self.assertIsNot(first, second)
            self.assertEqual(first[0][0], "default")
            self.assertEqual(second[0][0], "max-autotune-no-cudagraphs")
            self.assertEqual(compile_fn.call_count, 4)


if __name__ == "__main__":
    unittest.main()
