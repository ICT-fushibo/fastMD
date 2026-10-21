"""Configuration contract tests; intentionally require neither torch nor CUDA."""

import ast
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
import warnings

from fastmd.config import (
    CapacityConfig, ConfigurationWarning, InferenceConfig, OPTION_REGISTRY,
    current_config, current_options_environment, env_value, parse_bool,
    profile_env, resolve_config,
)


@contextmanager
def clean_environment(**values):
    with patch.dict(os.environ, values, clear=True):
        yield


class ConfigurationParsingTests(unittest.TestCase):
    def test_bool_spellings(self):
        for value in (True, 1, "1", "TRUE", " true ", "yes", "ON"):
            self.assertTrue(parse_bool(value))
        for value in (False, 0, "0", "FALSE", " false ", "no", "OFF"):
            self.assertFalse(parse_bool(value))
        for value in ("", "maybe", 2, -1, None, [], 1.0):
            with self.assertRaises(ValueError):
                parse_bool(value)

    def test_env_value_uses_registry_default_and_normalizes(self):
        name = "MATRIS_FUSED_INDEXED_CAT_LINEAR"
        with clean_environment():
            self.assertEqual(env_value(name), "0")
            self.assertEqual(env_value(name, "false"), "0")
            self.assertEqual(env_value(name, True), "0")
            self.assertEqual(env_value("UNREGISTERED_OPTION", "fallback"), "fallback")
        with clean_environment(**{name: "off"}):
            self.assertEqual(env_value(name, "1"), "0")
        with clean_environment(**{name: "perhaps"}):
            with self.assertRaisesRegex(ValueError, name):
                env_value(name)
        with clean_environment(MATRIS_TIER_HYSTERESIS_WINDOW="17"):
            self.assertEqual(env_value("MATRIS_TIER_DOWN_COOLDOWN", "4000"), "34")

    def test_numeric_and_enum_errors_are_early(self):
        bad = {
            "MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS": -1,
            "MATRIS_FUSED_RESIDUAL_ADD_BM": 0,
            "MATRIS_TIER_CHECK_INTERVAL": "1.5",
            "MATRIS_WHOLE_STEP_MAX_TIERS": True,
            "MATRIS_TIER_MIN_FREE_GIB": "nan",
            "MATRIS_TIER_HYSTERESIS_SIGMA": "inf",
            "MATRIS_FUSED_INDEXED_CAT_LINEAR_BACKEND": "typo",
            "MATRIS_TOPOLOGY_VERIFY": "off",
            "MATRIS_FUSED_SEGMENT_ATTENTION_NUM_WARPS": 3,
        }
        for name, value in bad.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                InferenceConfig(expert_overrides={name: value})

    def test_unknown_and_obsolete_expert_options_rejected(self):
        with self.assertRaises(ValueError):
            InferenceConfig(expert_overrides={"MATRIS_TYPO": 1})
        with self.assertRaises(ValueError):
            InferenceConfig(expert_overrides={"MATRIS_CUDA_GRAPH_U_STEP": 1024})

    def test_capacity_validation(self):
        from fastmd.adapters.protocol import CapacityStep
        CapacityConfig(warmup=0, min_pad_u=0, candidate_skin=0)
        for values in ({"u_step": 0}, {"n_dummy": 0}, {"warmup": -1}, {"t_step": 2.0}, {"candidate_skin": float("nan")}, {"min_pad_t": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CapacityConfig(**values)
        # An adapter-declared step: triplet tiers are optional, but step and padding come together.
        CapacityStep(u_step=32, min_pad_u=0, t_step=1, min_pad_t=0, t_thermal_factor=1.25)
        CapacityStep(u_step=32, min_pad_u=0, u_thermal_factor=2.0)  # a pair factor needs no triplet tiers
        CapacityStep(u_step=32, min_pad_u=0, u_union_headroom=0.0, t_union_headroom=1.0)  # a union's growth
        for values in ({"u_step": 0}, {"min_pad_u": -1}, {"t_step": 64}, {"min_pad_t": 64}, {"t_step": 0, "min_pad_t": 0},
                       {"t_step": 64, "min_pad_t": 2.0}, {"t_step": True, "min_pad_t": 0}, {"t_thermal_factor": 1.1},
                       {"t_step": 64, "min_pad_t": 64, "t_thermal_factor": 0.9},
                       {"t_step": 64, "min_pad_t": 64, "t_thermal_factor": float("inf")},
                       {"t_step": 64, "min_pad_t": 64, "t_thermal_factor": 2.5}, {"u_thermal_factor": 0.99},
                       {"u_thermal_factor": 2.01}, {"u_thermal_factor": True}, {"u_union_headroom": -0.01},
                       {"t_union_headroom": 1.5}, {"u_union_headroom": True}, {"t_union_headroom": float("nan")}):
            with self.subTest(step=values), self.assertRaises(ValueError):
                CapacityStep(**{"u_step": 32, "min_pad_u": 12, **values})

    def test_public_options_validation_and_immutability(self):
        for values in ({"execution": "graph"}, {"optimization_profile": "fast"}, {"checkpoint": True}, {"isolated_atoms": "ignore"},
                       {"topology_reuse": "epoch"}):
            with self.assertRaises(ValueError):
                InferenceConfig(**values)
        original = {"MATRIS_FUSED_INDEXED_CAT_LINEAR": "yes"}
        config = InferenceConfig(expert_overrides=original)
        original["MATRIS_FUSED_INDEXED_CAT_LINEAR"] = "no"
        self.assertTrue(config.expert_overrides["MATRIS_FUSED_INDEXED_CAT_LINEAR"])
        with self.assertRaises(FrozenInstanceError):
            config.execution = "eager"
        with self.assertRaises(TypeError):
            config.expert_overrides["MATRIS_FUSED_INDEXED_CAT_LINEAR"] = False
        with clean_environment():
            resolved = resolve_config(InferenceConfig(report=False))
        with self.assertRaises(TypeError):
            resolved.kernel_options["MATRIS_FUSED_INDEXED_CAT_LINEAR"] = True


class ConfigurationResolutionTests(unittest.TestCase):
    def test_no_config_retains_entrypoint_semantics(self):
        with clean_environment():
            resolved = resolve_config(None, default_execution="model_graph", default_isolated_atoms="error")
        self.assertEqual(resolved.execution, "model_graph")
        self.assertFalse(resolved.checkpoint_enabled)
        self.assertFalse(resolved.handle_isolated_atoms)
        self.assertFalse(resolved.report)
        # The default profile is topology (the fused kernels).
        self.assertEqual(resolved.optimization_profile, "topology")
        self.assertTrue(resolved.kernel_options["MATRIS_FUSED_INDEXED_CAT_LINEAR"])
        self.assertTrue(resolved.kernel_options["MATRIS_MERGED_ATTENTION_PROJECTIONS"])
        self.assertIsNone(InferenceConfig().optimization_profile)  # unset: topology
        self.assertEqual(resolve_config(InferenceConfig()).optimization_profile, "topology")
        self.assertEqual(resolve_config(None, default_execution="auto").execution, "auto")

    def test_topology_reuse_resolves_and_is_fingerprinted(self):
        with clean_environment():
            resolved = {mode: resolve_config(InferenceConfig(topology_reuse=mode)) for mode in ("off", "skin", "auto")}
            self.assertEqual(resolve_config(None).topology_reuse, "auto")
        self.assertEqual({mode: r.topology_reuse for mode, r in resolved.items()},
                         {"off": "off", "skin": "skin", "auto": "auto"})
        self.assertEqual(len({r.fingerprint for r in resolved.values()}), 3)

    def test_checkpoint_on_and_auto_are_deprecated_off_is_silent(self):
        import fastmd.config as config_module

        for checkpoint, warns in (("off", False), ("on", True), ("auto", True)):
            with self.subTest(checkpoint=checkpoint), \
                    patch.object(config_module, "_checkpoint_deprecation_warned", False), \
                    warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                InferenceConfig(checkpoint=checkpoint)
                InferenceConfig(checkpoint=checkpoint)  # once per process
                messages = [str(w.message) for w in caught if issubclass(w.category, ConfigurationWarning)]
                self.assertEqual(len(messages), int(warns), messages)
                if warns:
                    self.assertIn(f"checkpoint={checkpoint!r}", messages[0])
                    self.assertIn("saved_tensor_recompute", messages[0])
                    self.assertIn("next version", messages[0])
                    self.assertEqual(caught[0].filename, __file__)  # points at the caller

    def test_explicit_checkpoint_and_execution(self):
        with clean_environment():
            for mode, expected in (("off", False), ("on", True), ("auto", None)):
                resolved = resolve_config(InferenceConfig(checkpoint=mode), default_execution="model_graph")
                self.assertIs(resolved.checkpoint_enabled, expected)
                self.assertEqual(resolved.execution, "model_graph")

    def test_obsolete_whole_step_environment_cannot_select_execution(self):
        with clean_environment(MATRIS_WHOLE_STEP_CUDA_GRAPH="1"):
            with self.assertWarnsRegex(ConfigurationWarning, "whole_step"):
                resolved = resolve_config(None)
            self.assertEqual(resolved.execution, "eager")
            self.assertNotIn("MATRIS_WHOLE_STEP_CUDA_GRAPH", resolved.kernel_options)
            self.assertNotIn("MATRIS_WHOLE_STEP_CUDA_GRAPH", profile_env("topology"))
        with self.assertRaises(ValueError):
            InferenceConfig(expert_overrides={"MATRIS_WHOLE_STEP_CUDA_GRAPH": True})

    def test_legacy_environment_and_dependent_default(self):
        with clean_environment(MATRIS_FUSED_INDEXED_CAT_LINEAR="yes", MATRIS_TIER_HYSTERESIS_WINDOW="17"):
            # MATRIS_* environment options apply to the legacy profile only.
            resolved = resolve_config(InferenceConfig(optimization_profile="legacy"))
            self.assertTrue(resolved.kernel_options["MATRIS_FUSED_INDEXED_CAT_LINEAR"])
            self.assertEqual(resolved.kernel_options["MATRIS_TIER_DOWN_COOLDOWN"], 34)
            resolved = resolve_config(InferenceConfig(optimization_profile="legacy",
                                                      expert_overrides={"MATRIS_FUSED_INDEXED_CAT_LINEAR": False}))
            self.assertFalse(resolved.kernel_options["MATRIS_FUSED_INDEXED_CAT_LINEAR"])
            with self.assertWarnsRegex(ConfigurationWarning, "ignores registered environment options"):
                resolved = resolve_config(None)
            self.assertEqual(resolved.kernel_options["MATRIS_TIER_DOWN_COOLDOWN"], 4000)
        # Runtime controls (config.RUNTIME_ENVIRONMENT) apply in every profile, without the warning.
        runtime = {"MATRIS_TOPOLOGY_REUSE_CHECK": ("3", 3), "MATRIS_TOPOLOGY_VERIFY": ("sampled", "sampled"),
                   "MATRIS_DECOMPOSED_INDEXED_DEBUG_CHECKS": ("1", True), "MATRIS_SAVED_TENSOR_RECOMPUTE": ("deep", "deep")}
        with clean_environment(**{name: raw for name, (raw, _) in runtime.items()}):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConfigurationWarning)
                resolved = resolve_config(None)
            for name, (_, value) in runtime.items():
                self.assertEqual(resolved.kernel_options[name], value, name)

    def test_profiles_are_deterministic_and_expert_wins(self):
        with clean_environment():
            expected = resolve_config(InferenceConfig(optimization_profile="topology"))
        with clean_environment(MATRIS_FUSED_INDEXED_CAT_LINEAR="false", MATRIS_TIER_HYSTERESIS_WINDOW="11"):
            with self.assertWarnsRegex(ConfigurationWarning, "ignores"):
                actual = resolve_config(InferenceConfig(optimization_profile="topology"))
            self.assertEqual(expected.fingerprint, actual.fingerprint)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConfigurationWarning)
                overridden = resolve_config(InferenceConfig(optimization_profile="topology", expert_overrides={"MATRIS_FUSED_INDEXED_CAT_LINEAR": "no"}))
            self.assertFalse(overridden.kernel_options["MATRIS_FUSED_INDEXED_CAT_LINEAR"])
            self.assertNotEqual(expected.fingerprint, overridden.fingerprint)

    def test_unknown_obsolete_environment_warnings_and_benchmark_metadata(self):
        with clean_environment(MATRIS_UNKNOWN="1", MATRIS_CUDA_GRAPH_U_STEP="256", MATRIS_MODEL_PATH="/tmp/model", MATRIS_REPOSITORY="/tmp/repo", MATRIS_BASELINE_REPOSITORY="/tmp/base", MATRIS_RUN_PERF_TIER="1"):
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                resolved = resolve_config(None)
            messages = [str(item.message) for item in seen]
            self.assertTrue(any("MATRIS_UNKNOWN" in message for message in messages), messages)
            self.assertTrue(any("obsolete" in message for message in messages), messages)
            # Benchmark metadata and test-tier variables are external: no warning.
            for name in ("MATRIS_MODEL_PATH", "MATRIS_REPOSITORY", "MATRIS_BASELINE_REPOSITORY", "MATRIS_RUN_PERF_TIER"):
                self.assertFalse([message for message in messages if name in message])
            self.assertEqual(resolved.capacity.u_step, CapacityConfig().u_step)
        # The CuTeDSL backends were removed: selecting one fails in every
        # profile and as an expert override; a former tuning knob warns and is
        # dropped, never resolved.
        for name, value in (("MATRIS_FUSED_INDEXED_CAT_LINEAR_BACKEND", "cutedsl_dense_silu"),
                            ("MATRIS_DECOMPOSED_INDEXED_GEMM_BACKEND", "CuTeDSL")):
            for profile in ("legacy", "topology"):
                with self.subTest(name=name, profile=profile), clean_environment(**{name: value}):
                    with self.assertRaisesRegex(ValueError, f"{name}.*removed"):
                        resolve_config(InferenceConfig(optimization_profile=profile))
                    with self.assertRaisesRegex(ValueError, f"{name}.*removed"):
                        profile_env(profile)  # subprocess environments
            with self.subTest(name=name, source="expert_overrides"):
                with self.assertRaisesRegex(ValueError, f"{name}.*removed"):
                    InferenceConfig(expert_overrides={name: value})
        knob = "MATRIS_CUTEDSL_AUTOTUNE"
        with clean_environment(**{knob: "0"}):
            with self.assertWarnsRegex(ConfigurationWarning, f"{knob} was removed"):
                resolved = resolve_config(None)
            self.assertNotIn(knob, resolved.kernel_options)
        with self.assertWarnsRegex(ConfigurationWarning, "MATRIS_CUTEDSL_MMA_TILER_MN was removed"):
            config = InferenceConfig(expert_overrides={"MATRIS_CUTEDSL_MMA_TILER_MN": "128,256"})
        self.assertEqual(dict(config.expert_overrides), {})
        with clean_environment():
            self.assertNotIn("MATRIS_CUTEDSL_MMA_TILER_MN", resolve_config(config).kernel_options)
        # The legacy operator engine (step 2.6) and the legacy forward runtimes
        # (step 4.2) were removed: each rollback switch fails when its module is
        # imported (concurrent subprocesses: no torch here).
        processes = {name: subprocess.Popen(
            [sys.executable, "-c", f"import {module}"],
            cwd=str(Path(__file__).resolve().parents[1]), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env={**os.environ, name: "legacy"})
            for name, module in (("MATRIS_DISPATCH_ENGINE", "fastmd.engine.dispatch"),
                                 ("MATRIS_RUNTIME_ENGINE", "fastmd.engine.runtime"))}
        errors = {name: process.communicate()[1] for name, process in processes.items()}
        for name, process in processes.items():
            with self.subTest(name=name):
                self.assertNotEqual(process.returncode, 0)
                self.assertRegex(errors[name], f"ValueError: {name}")

    def test_matmul_precision_is_optional_validated_and_fingerprinted(self):
        self.assertIsNone(InferenceConfig().matmul_precision)
        for value in ("highest", "high", "medium"):
            self.assertEqual(InferenceConfig(matmul_precision=value).matmul_precision, value)
        for value in ("HIGHEST", "tf32", "ieee", "", 1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                InferenceConfig(matmul_precision=value)
        with clean_environment():
            default = resolve_config(InferenceConfig())
            pinned = resolve_config(InferenceConfig(matmul_precision="highest"))
            unconfigured = resolve_config(None)
        self.assertIsNone(default.matmul_precision)
        self.assertIsNone(unconfigured.matmul_precision)
        self.assertEqual(pinned.matmul_precision, "highest")

    def test_generic_matches_historical_ablation_switches(self):
        values = profile_env("generic")
        for suffix in ("FUSED_INDEXED_CAT_LINEAR", "FUSED_SEGMENT_ATTENTION", "FUSED_GATHER_CAT_EAGER", "FUSED_THREEBODY_LINEAR", "UNDIRECTED_EDGE_INIT"):
            self.assertEqual(values["MATRIS_" + suffix], "0")
        self.assertEqual(values["MATRIS_GATED_LN_FUSE_MIN_ROWS"], "1000000000")
        self.assertEqual(profile_env("topology")["MATRIS_FUSED_INDEXED_CAT_LINEAR"], "1")


class ConfigurationScopeTests(unittest.TestCase):
    def test_loaded_module_restored_and_environment_untouched(self):
        # An invalid environment value that the profile ignores must not break
        # the binding or its restoration either.
        for environment in ({}, {"MATRIS_FUSED_INDEXED_CAT_LINEAR": "invalid"}):
            with self.subTest(environment=environment):
                fake = types.ModuleType("fastmd.models.matris.functions")
                fake._INDEXED_CAT_LINEAR_FUSE = False
                with clean_environment(**environment), patch.dict(sys.modules, {fake.__name__: fake}):
                    before = dict(os.environ)
                    with warnings.catch_warnings(record=True) as seen:
                        warnings.simplefilter("always")
                        topology = resolve_config(InferenceConfig(optimization_profile="topology"))
                    if environment:
                        self.assertTrue(any(issubclass(item.category, ConfigurationWarning)
                                            and "ignores" in str(item.message) for item in seen))
                    with topology.scope():
                        self.assertIs(current_config(), topology)
                        self.assertTrue(fake._INDEXED_CAT_LINEAR_FUSE)
                        self.assertEqual(env_value("MATRIS_FUSED_INDEXED_CAT_LINEAR"), "1")
                        with topology.scope():
                            self.assertTrue(fake._INDEXED_CAT_LINEAR_FUSE)
                        self.assertEqual(before, dict(os.environ))
                    self.assertFalse(fake._INDEXED_CAT_LINEAR_FUSE)
                    self.assertIsNone(current_config())

    def test_nested_different_profiles_and_exception_restoration(self):
        fake = types.ModuleType("fastmd.models.matris.functions")
        fake._INDEXED_CAT_LINEAR_FUSE = False
        with clean_environment(), patch.dict(sys.modules, {fake.__name__: fake}):
            topology = resolve_config(InferenceConfig(optimization_profile="topology"))
            generic = resolve_config(InferenceConfig(optimization_profile="generic"))
            with topology.scope():
                with self.assertRaisesRegex(RuntimeError, "test"):
                    with generic.scope():
                        self.assertFalse(fake._INDEXED_CAT_LINEAR_FUSE)
                        raise RuntimeError("test")
                self.assertTrue(fake._INDEXED_CAT_LINEAR_FUSE)
                self.assertIs(current_config(), topology)
            self.assertFalse(fake._INDEXED_CAT_LINEAR_FUSE)

    def test_lazy_import_binding_restored(self):
        name = "fastmd.models.matris.functions"
        with clean_environment(), patch.dict(sys.modules):
            sys.modules.pop(name, None)
            topology = resolve_config(InferenceConfig(optimization_profile="topology"))
            with topology.scope():
                fake = types.ModuleType(name)
                fake._INDEXED_CAT_LINEAR_FUSE = env_value("MATRIS_FUSED_INDEXED_CAT_LINEAR", "0") == "1"
                sys.modules[name] = fake
                self.assertTrue(fake._INDEXED_CAT_LINEAR_FUSE)
            self.assertFalse(fake._INDEXED_CAT_LINEAR_FUSE)

    def test_unconfigured_entrypoint_inherits_outer_kernel_profile(self):
        with clean_environment():
            topology = resolve_config(InferenceConfig(optimization_profile="topology", execution="whole_step"))
            with topology.scope():
                inherited = resolve_config(None, default_execution="eager", default_isolated_atoms="error")
                self.assertEqual(inherited.optimization_profile, "topology")
                self.assertEqual(inherited.kernel_options, topology.kernel_options)
                self.assertEqual(inherited.execution, "eager")
                self.assertFalse(inherited.checkpoint_enabled)
                self.assertFalse(inherited.handle_isolated_atoms)
                explicit_legacy = resolve_config(InferenceConfig(optimization_profile="legacy"))
                self.assertFalse(explicit_legacy.kernel_options["MATRIS_FUSED_INDEXED_CAT_LINEAR"])

    def test_worker_export_respects_active_profile(self):
        with clean_environment():
            topology = resolve_config(InferenceConfig(optimization_profile="topology"))
            with topology.scope():
                self.assertEqual(current_options_environment()["MATRIS_FUSED_INDEXED_CAT_LINEAR"], "1")


class RegistryTests(unittest.TestCase):
    def test_registry_is_complete_unique_and_defaults_validate(self):
        bindings = []
        for spec in OPTION_REGISTRY.values():
            self.assertEqual(spec.parse(spec.default), spec.default)
            self.assertTrue(spec.description)
            bindings.extend(spec.bindings)
        self.assertEqual(len(bindings), len(set(bindings)))

    def test_source_option_consumers_are_registered(self):
        root = Path(__file__).resolve().parents[1] / "fastmd"
        used = set()
        for path in root.rglob("*.py"):
            if path.name == "config.py":
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                    if name in {"getenv", "env_value", "_env_flag", "_env_int", "_env_flag_fixed"} and node.args:
                        first = node.args[0]
                        if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("MATRIS_"):
                            used.add(first.value)
        self.assertFalse(used - OPTION_REGISTRY.keys())

    def test_import_time_environment_constants_have_scope_bindings(self):
        root = Path(__file__).resolve().parents[1] / "fastmd"
        expected = set()
        for path in root.rglob("*.py"):
            if path.name == "config.py":
                continue
            module = "fastmd." + ".".join(path.relative_to(root).with_suffix("").parts)
            for assignment in ast.parse(path.read_text()).body:
                if not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
                    continue
                for node in ast.walk(assignment.value):
                    if isinstance(node, ast.Call) and node.args:
                        func = node.func
                        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                        first = node.args[0]
                        if name in {"getenv", "env_value", "_env_flag", "_env_int"} and isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value in OPTION_REGISTRY:
                            expected.add((first.value, module, assignment.targets[0].id))
        actual = {(name, module, attr) for name, spec in OPTION_REGISTRY.items() for module, attr in spec.bindings}
        self.assertFalse(expected - actual, f"Missing scoped bindings: {expected - actual}")

    def test_cli_works_without_importing_torch(self):
        root = str(Path(__file__).resolve().parents[1])
        with self.subTest(case="CLI"):
            result = subprocess.run(
                [sys.executable, "-c", "import runpy,sys; sys.argv=['fastmd.config','--profile','topology']; runpy.run_module('fastmd.config',run_name='__main__'); assert 'torch' not in sys.modules"],
                cwd=root, capture_output=True, text=True, check=True,
            )
            self.assertEqual(json.loads(result.stdout)["MATRIS_FUSED_INDEXED_CAT_LINEAR"], "1")
        with self.subTest(case="unpinned scope"):
            # Only matmul_precision needs torch; the backward-compatible default
            # scope must remain importable and usable without it.
            code = ("import sys\n"
                    "from fastmd.config import InferenceConfig, resolve_config\n"
                    "with resolve_config(InferenceConfig(optimization_profile='generic')).scope():\n"
                    "    pass\n"
                    "assert 'torch' not in sys.modules\n")
            subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True)
        with self.subTest(case="package"):
            # The package's public names are imported from their modules on first access; no other name is an
            # attribute of the package (a submodule is one once imported).
            public = {"MLIPCalculator": "fastmd.interfaces.calculator",
                      "GPUResidentMolecularDynamics": "fastmd.interfaces.gpu_md",
                      "MolecularDynamics": "fastmd.interfaces.md", "StructOptimizer": "fastmd.interfaces.relax",
                      "InferenceConfig": "fastmd.config", "load_adapter": "fastmd.adapters",
                      "MatRIS": "fastmd.models.matris.model"}
            code = ("import sys\n"
                    "import fastmd\n"
                    "assert 'torch' not in sys.modules\n"
                    "assert not hasattr(fastmd, 'config') and not hasattr(fastmd, 'no_such_name')\n"
                    f"assert {set(public)!r} <= set(dir(fastmd))\n")
            result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            import importlib
            import fastmd
            for name, module in public.items():
                self.assertIs(getattr(fastmd, name), getattr(importlib.import_module(module), name), name)


if __name__ == "__main__":
    unittest.main()
