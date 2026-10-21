"""Effective FP32 GEMM precision, scoped application, TF32 warnings and oracle gates.

CPU-only except the explicitly skipped CUDA cases: torch's precision flags are
host state, so the effective-mode logic is tested without cuBLAS. The flag
combinations follow the GPU truth table in fastmd/precision.py.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

from ase import Atoms
from ase.io import write
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
import torch

from fastmd import precision
from fastmd.config import CapacityConfig
from fastmd.config import ConfigurationWarning
from fastmd.config import InferenceConfig
from fastmd.config import resolve_config
from fastmd.precision import IMPLEMENTATION_TOLERANCES
from fastmd.precision import compute_references
from fastmd.precision import fp32_gemm_mode
from fastmd.precision import matmul_precision_scope
from fastmd.precision import oracle_report
from fastmd.precision import precision_state
from fastmd.precision import warn_implicit_tf32
from helpers import toy_model

NEW_API = hasattr(torch._C, "_get_fp32_precision_getter")
REPOSITORY = Path(__file__).resolve().parents[1]


def set_entry(backend, op, value):
    torch._C._set_fp32_precision_setter(backend, op, value)


def load_torchsim_interface():
    """The wrapper module with torch-sim stubbed, as in test_runtime_config."""
    names = ("torch_sim", "torch_sim.units", "torch_sim.models", "torch_sim.models.interface", "torch_sim.state")
    modules = {name: ModuleType(name) for name in names}
    modules["torch_sim"].units = modules["torch_sim.units"]
    modules["torch_sim.units"].MetalUnits = SimpleNamespace(pressure=1.0)
    modules["torch_sim.models.interface"].ModelInterface = torch.nn.Module
    modules["torch_sim.state"].SimState = object
    path = REPOSITORY / "fastmd/interfaces/torchsim_interface.py"
    spec = importlib.util.spec_from_file_location("fastmd.interfaces._precision_torchsim_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def record_forward_modes(model):
    """Effective GEMM mode at every Python-level module call (eager or capture)."""
    modes = []
    for module in model.modules():
        module.register_forward_pre_hook(lambda *_: modes.append(fp32_gemm_mode()))
    return modes


class PrecisionStateTestCase(unittest.TestCase):
    """Every test restores and verifies the process-wide torch flags."""

    def setUp(self):
        self.saved = precision._snapshot()
        self.saved_state = precision_state()
        precision._WARNED.clear()

    def tearDown(self):
        precision._restore(self.saved)
        precision._WARNED.clear()
        self.assertEqual(precision_state(), self.saved_state)


class EffectiveModeTest(PrecisionStateTestCase):
    def test_fp32_gemm_mode_truth_table(self):
        def new_api(value):
            def apply():
                torch.backends.cuda.matmul.fp32_precision = value
            return apply

        def entry(backend, op, value):
            return lambda: set_entry(backend, op, value)

        def legacy_torch(allow_tf32):
            # torch < 2.9 ORs TORCH_ALLOW_TF32_CUBLAS_OVERRIDE into allow_tf32
            # even under 'highest'. Gates must follow that flag, not the string.
            return (patch("fastmd.precision._fp32_getter", return_value=None),
                    patch.object(torch.backends.cuda, "matmul", SimpleNamespace(allow_tf32=allow_tf32)))

        # (case, legacy string, new-API steps, active contexts, effective mode)
        cases = [
            ("highest", "highest", (), (), "ieee"),
            ("high", "high", (), (), "tf32"),
            ("medium", "medium", (), (), "tf32"),
            ("NVIDIA_TF32_OVERRIDE=0", "high", (), (patch.dict(os.environ, {"NVIDIA_TF32_OVERRIDE": "0"}),),
             "ieee"),
            ("NVIDIA_TF32_OVERRIDE=1", "high", (), (patch.dict(os.environ, {"NVIDIA_TF32_OVERRIDE": "1"}),),
             "tf32"),
            ("legacy torch, env-forced allow_tf32", "highest", (), legacy_torch(True), "tf32"),
            ("legacy torch, allow_tf32 off", "highest", (), legacy_torch(False), "ieee"),
        ]
        if NEW_API:
            cases += [
                # Measured on GB300: 'high' then fp32_precision='ieee' runs IEEE
                # GEMMs while get_float32_matmul_precision() still reports 'high'.
                ("high, then cuda.matmul=ieee", "high", (new_api("ieee"),), (), "ieee"),
                # A new-API-only TF32 request (the legacy getter raises here).
                ("highest, then cuda.matmul=tf32", "highest", (new_api("tf32"),), (), "tf32"),
                # An unset matmul entry inherits its parent backends.
                ("cuda.matmul unset", "highest", (entry("cuda", "matmul", "none"),), (), "ieee"),
                ("cuda.matmul unset, generic.all=tf32", "highest",
                 (entry("cuda", "matmul", "none"), entry("generic", "all", "tf32")), (), "tf32"),
                ("cuda.matmul unset, cuda.all=tf32", "highest",
                 (entry("cuda", "matmul", "none"), entry("generic", "all", "none"), entry("cuda", "all", "tf32")),
                 (), "tf32"),
            ]
        for case, legacy, steps, contexts, expected in cases:
            with self.subTest(case=case):
                precision._restore(self.saved)
                torch.set_float32_matmul_precision(legacy)
                for step in steps:
                    step()
                with contextlib.ExitStack() as stack:
                    for context in contexts:
                        stack.enter_context(context)
                    self.assertEqual(fp32_gemm_mode(), expected)


class GateTest(PrecisionStateTestCase):

    @unittest.skipUnless(NEW_API, "requires the fp32_precision API")
    def test_threebody_cache_key_is_effective_and_never_raises(self):
        from fastmd.models.matris.feature_embed import ThreebodyEmbedding
        module = ThreebodyEmbedding(num_angular=7, three_body_feat_dim=32, learnable=False).eval()
        module.requires_grad_(False)
        basis = torch.randn(3, 7)
        build = lambda: module._combined_swish_linear_weight(basis, apply_profile=False)
        torch.set_float32_matmul_precision("highest")
        highest = build()
        self.assertIs(build(), highest)
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        tf32 = build()
        self.assertIsNot(tf32, highest)
        # Same reported string ('high'), different effective math.
        torch.set_float32_matmul_precision("high")
        high = build()
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        self.assertIsNot(build(), high)


class ScopeTest(PrecisionStateTestCase):
    def states(self):
        yield "highest", lambda: torch.set_float32_matmul_precision("highest")
        yield "high", lambda: torch.set_float32_matmul_precision("high")
        yield "medium", lambda: torch.set_float32_matmul_precision("medium")
        if NEW_API:
            def high_then_ieee():
                torch.set_float32_matmul_precision("high")
                torch.backends.cuda.matmul.fp32_precision = "ieee"

            def new_api_tf32():
                torch.set_float32_matmul_precision("highest")
                torch.backends.cuda.matmul.fp32_precision = "tf32"

            def parent_tf32():
                torch.set_float32_matmul_precision("highest")
                set_entry("cuda", "matmul", "none")
                set_entry("cuda", "all", "tf32")
            yield "high_then_ieee", high_then_ieee
            yield "new_api_tf32", new_api_tf32
            yield "parent_tf32", parent_tf32

    def test_scope_applies_and_restores_every_state(self):
        for name, prepare in self.states():
            for requested, expected in (("highest", "ieee"), ("high", "tf32"), ("medium", "tf32")):
                with self.subTest(state=name, requested=requested):
                    prepare()
                    before = precision_state()
                    with matmul_precision_scope(requested):
                        self.assertEqual(fp32_gemm_mode(), expected)
                        self.assertEqual(torch.get_float32_matmul_precision(), requested)
                    after = precision_state()
                    self.assertEqual(after["fp32_gemm"], before["fp32_gemm"])
                    self.assertEqual(after["fp32_precision"], before["fp32_precision"])
                    if not before["float32_matmul_precision"].startswith("error"):
                        self.assertEqual(after, before)

    def test_none_is_a_no_op_and_errors_restore(self):
        torch.set_float32_matmul_precision("high")
        with matmul_precision_scope(None):
            self.assertEqual(fp32_gemm_mode(), "tf32")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with matmul_precision_scope("highest"):
                raise RuntimeError("boom")
        self.assertEqual(torch.get_float32_matmul_precision(), "high")
        self.assertEqual(fp32_gemm_mode(), "tf32")
        with self.assertRaises(ValueError):
            with matmul_precision_scope("tf32"):
                pass

    def test_order_independent_suspends_and_restores_the_deterministic_flag(self):
        previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            for warn_only in (None, False, True):  # None: deterministic mode off
                torch.use_deterministic_algorithms(warn_only is not None, warn_only=bool(warn_only))
                with self.subTest(warn_only=warn_only):
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        with precision.order_independent():
                            self.assertFalse(precision.deterministic_mode())
                            raise RuntimeError("boom")
                    self.assertEqual(precision.deterministic_mode(), warn_only is not None)
                    self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), bool(warn_only))
        finally:
            torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])

    def test_resolved_config_scope_pins_only_when_requested(self):
        torch.set_float32_matmul_precision("high")
        pinned = resolve_config(InferenceConfig(matmul_precision="highest", report=False))
        unpinned = resolve_config(InferenceConfig(report=False))
        with pinned.scope():
            self.assertEqual(fp32_gemm_mode(), "ieee")
            # Nested unconfigured entries inherit the outer precision.
            with resolve_config(None).scope():
                self.assertEqual(fp32_gemm_mode(), "ieee")
            with pinned.scope():
                self.assertEqual(fp32_gemm_mode(), "ieee")
            # Like its kernel options, an entry built here keeps the precision.
            inner = resolve_config(None)
        self.assertEqual(inner.matmul_precision, "highest")
        self.assertEqual(fp32_gemm_mode(), "tf32")
        with inner.scope():
            self.assertEqual(fp32_gemm_mode(), "ieee")
        self.assertEqual(fp32_gemm_mode(), "tf32")
        with unpinned.scope():
            self.assertEqual(fp32_gemm_mode(), "tf32")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with pinned.scope():
                raise RuntimeError("boom")
        self.assertEqual(torch.get_float32_matmul_precision(), "high")


class TritonDotPrecisionTest(PrecisionStateTestCase):
    """InferenceConfig.triton_dot_precision: the per-site dot precision (None: the site's cuBLAS path) in
    both cuBLAS modes, and where the requested value comes from."""

    def test_site_table(self):
        sites = precision.TRITON_DOT_SITES
        expected = {  # (requested, torch precision): row_gemm, edge_ffn, gated_tail, act_linear, wide_gemm
            ("auto", "highest"): ("tf32x3", "tf32x3", "tf32x3", "tf32x3", "tf32x3"),
            ("auto", "high"): (None, "tf32", "tf32", None, None),
            ("ieee", "highest"): (None, None, "ieee", "ieee", None),
            ("ieee", "high"): ("ieee", "ieee", "ieee", None, None),
            ("tf32x3", "highest"): ("tf32x3",) * 5,
            ("tf32x3", "high"): ("tf32x3", "tf32x3", "tf32x3", None, "tf32x3"),
            ("tf32", "highest"): ("tf32",) * 5,
            ("tf32", "high"): (None, "tf32", "tf32", None, None),
        }
        for (requested, torch_precision), choices in expected.items():
            with self.subTest(requested=requested, torch=torch_precision), \
                    matmul_precision_scope(torch_precision), precision.triton_dot_precision_scope(requested):
                self.assertEqual(tuple(precision.triton_dot_input_precision(site) for site in sites), choices)
                recorded = precision_state()["triton_dot"]
                self.assertEqual(recorded, {"requested": requested, **{
                    site: choice or "cublas" for site, choice in zip(sites, choices)}})
                self.assertEqual(precision.gemm_precision_key(), (fp32_gemm_mode(), requested))

    def test_oracle_references_follow_the_tested_dot_precision(self):
        seen = []

        def predict():
            seen.append((fp32_gemm_mode(), precision.triton_dot_precision()))
            return 0.0, torch.zeros(1, 3)
        with patch.dict(os.environ, {"MATRIS_TRITON_DOT_PRECISION": "tf32"}):
            # A TF32 Triton site makes the run TF32: the gate and a same-precision rerun at the tested value;
            # the highest reference pins auto (3xTF32), not the environment's tf32.
            references = compute_references(predict, "highest", "tf32")
            self.assertEqual((references["run_fp32_gemm"], references["run_triton_dot"]), ("tf32", "tf32"))
            self.assertEqual(seen, [("ieee", "auto"), ("ieee", "tf32")])
            seen.clear()
            references = compute_references(predict, "highest", "ieee")
            self.assertEqual(references["run_fp32_gemm"], "ieee")
            self.assertIsNone(references["same_precision"])
            self.assertEqual(seen, [("ieee", "auto")])

    def test_config_override_environment_and_scopes(self):
        get = precision.triton_dot_precision
        with patch.dict(os.environ, {"MATRIS_TRITON_DOT_PRECISION": "tf32"}):
            self.assertEqual(get(), "tf32")
            with warnings.catch_warnings():
                # Read in every profile: no "ignored environment option" warning.
                warnings.simplefilter("error", ConfigurationWarning)
                topology = resolve_config(InferenceConfig(optimization_profile="topology", report=False))
            self.assertIsNone(topology.triton_dot_precision)
            with topology.scope():
                self.assertEqual(get(), "tf32")
            override = resolve_config(InferenceConfig(
                report=False, expert_overrides={"MATRIS_TRITON_DOT_PRECISION": "ieee"}))
            explicit = resolve_config(InferenceConfig(
                report=False, triton_dot_precision="tf32x3", expert_overrides={"MATRIS_TRITON_DOT_PRECISION": "ieee"}))
            self.assertNotEqual(override.fingerprint, explicit.fingerprint)
            with override.scope():
                self.assertEqual(get(), "ieee")
                # Nested unconfigured entries inherit it; explicit ones replace it.
                inner = resolve_config(None)
                with inner.scope():
                    self.assertEqual(get(), "ieee")
                with explicit.scope():
                    self.assertEqual(get(), "tf32x3")
                with topology.scope():
                    self.assertEqual(get(), "ieee")
                self.assertEqual(get(), "ieee")
            self.assertEqual(inner.triton_dot_precision, "ieee")
            self.assertEqual(get(), "tf32")
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with explicit.scope():
                    raise RuntimeError("boom")
            self.assertEqual(get(), "tf32")
        self.assertEqual(get(), "auto")
        with patch.dict(os.environ, {"MATRIS_TRITON_DOT_PRECISION": "fp16"}), self.assertRaises(ValueError):
            get()
        with self.assertRaises(ValueError):
            InferenceConfig(triton_dot_precision="highest")
        with self.assertRaises(ValueError):
            InferenceConfig(expert_overrides={"MATRIS_TRITON_DOT_PRECISION": "fp16"})



class ScopeLockTest(PrecisionStateTestCase):
    """Precision scopes snapshot/restore process-wide flags; threads serialize."""

    def assert_blocks(self, hold, contend):
        seen = []
        thread = threading.Thread(target=lambda: seen.append(contend()))
        with hold():
            before = fp32_gemm_mode()
            thread.start()
            thread.join(0.5)
            # Without the lock the contender would finish now and could restore
            # its own snapshot while this scope still runs.
            self.assertTrue(thread.is_alive())
            self.assertEqual(seen, [])
            self.assertEqual(fp32_gemm_mode(), before)
        thread.join(30)
        self.assertFalse(thread.is_alive())
        return seen[0]

    def test_bare_precision_scopes_and_managed_scopes_exclude_each_other(self):
        from fastmd.interfaces.calculator import MLIPCalculator
        torch.set_float32_matmul_precision("high")
        pinned = resolve_config(InferenceConfig(matmul_precision="highest", report=False))
        calculator = MLIPCalculator(model=toy_model(), device="cpu",
                                      config=InferenceConfig(matmul_precision="highest", report=False))

        def managed():
            with pinned.scope():
                return fp32_gemm_mode()

        def bare():
            with matmul_precision_scope("high"):
                return fp32_gemm_mode()

        # (holder, contender, the contender's effective mode if it reports one)
        cases = (
            ("bare holds, managed waits", lambda: matmul_precision_scope("highest"), managed, "ieee"),
            ("managed holds, bare waits", pinned.scope, bare, "tf32"),
            ("managed holds, config_summary waits", pinned.scope, calculator.config_summary, None),
        )
        for case, hold, contend, expected in cases:
            with self.subTest(case=case):
                result = self.assert_blocks(hold, contend)
                if expected is not None:
                    self.assertEqual(result, expected)
                self.assertEqual(fp32_gemm_mode(), "tf32")


class TF32WarningTest(PrecisionStateTestCase):
    def test_tf32_warnings(self):
        """The TF32 warning fires once, and only for TF32 nobody requested."""
        def cublas(requested):
            return warn_implicit_tf32(requested, "test")

        # (case, process precision, patched effective mode, [(warning check, requested, warns)] in order)
        cases = (
            ("cuBLAS TF32, nothing requested", "high", None,
             [(cublas, "high", False), (cublas, None, True), (cublas, None, False)]),
            ("'highest' requested, effective TF32", "highest", "tf32", [(cublas, "highest", True)]),
            ("'highest' requested, effective IEEE", "highest", None, [(cublas, "highest", False)]),
        )
        for case, process, effective, calls in cases:
            with self.subTest(case=case):
                precision._WARNED.clear()
                torch.set_float32_matmul_precision(process)
                with contextlib.ExitStack() as stack:
                    if effective is not None:
                        stack.enter_context(patch("fastmd.precision.fp32_gemm_mode", return_value=effective))
                    seen = stack.enter_context(warnings.catch_warnings(record=True))
                    warnings.simplefilter("always")
                    results = [check(requested) for check, requested, _ in calls]
                self.assertEqual(results, [warns for *_, warns in calls])
                tf32 = [w for w in seen if issubclass(w.category, ConfigurationWarning) and "TF32" in str(w.message)]
                self.assertEqual(len(tf32), sum(warns for *_, warns in calls))


class TorchSimPrecisionTest(PrecisionStateTestCase):
    """Each wrapper call runs under the configured precision, whatever the caller set."""

    def state(self):
        atoms = Atoms(numbers=[1, 6, 8, 1], positions=[[1, 1, 1], [2.1, 1, 1], [1.2, 2.2, 1], [8, 8, 8]],
                      cell=[12, 12, 12], pbc=True)
        return SimpleNamespace(positions=torch.as_tensor(atoms.positions), cell=torch.as_tensor(atoms.cell.array),
                               to_structures=lambda: [AseAtomsAdaptor.get_structure(atoms)])

    def test_forward_follows_config_and_restores_caller_flags(self):
        interface = load_torchsim_interface()
        for process, requested, expected in (("high", "highest", "ieee"), ("highest", "high", "tf32"),
                                             ("high", None, "tf32")):
            with self.subTest(process=process, requested=requested):
                torch.set_float32_matmul_precision(process)
                model = toy_model()
                wrapper = interface.MLIPModel(
                    model=model, compute_stress=False, device="cpu", enable_compile=False,
                    config=InferenceConfig(report=False, isolated_atoms="reference", matmul_precision=requested))
                modes = record_forward_modes(model)
                output = wrapper(self.state())
                self.assertEqual(tuple(output["forces"].shape), (4, 3))
                self.assertTrue(modes)
                self.assertEqual(set(modes), {expected})
                self.assertEqual(torch.get_float32_matmul_precision(), process)


class OracleTest(PrecisionStateTestCase):
    forces = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]])

    def predictor(self, offsets):
        calls = []

        def predict():
            mode = fp32_gemm_mode()
            calls.append(mode)
            return -5.0 + offsets[mode][0], self.forces + offsets[mode][1]
        return predict, calls

    def test_ieee_run_uses_one_ieee_reference_and_tight_gate(self):
        # 'highest', or 'high' with TF32 disabled in the library.
        for requested, environment in (("highest", {}), ("high", {"NVIDIA_TF32_OVERRIDE": "0"})):
            with self.subTest(requested=requested, environment=environment):
                torch.set_float32_matmul_precision("high")  # caller state must not leak in
                predict, calls = self.predictor({"ieee": (0.0, 0.0)})
                with patch.dict(os.environ, environment):
                    references = compute_references(predict, requested)
                self.assertEqual(calls, ["ieee"])
                self.assertEqual(references["run_fp32_gemm"], "ieee")
                self.assertIsNone(references["same_precision"])
                good = oracle_report(-5.0 + 3e-5, self.forces + 2e-6, references)
                self.assertTrue(good["passed"])
                self.assertEqual(good["implementation"]["reference"], "highest")
                self.assertEqual(good["implementation"]["tolerance"], IMPLEMENTATION_TOLERANCES["ieee"])
                self.assertIsNone(good["precision"])
                # The historical 0.005 eV/A gate would have accepted this.
                self.assertFalse(oracle_report(-5.0, self.forces + 1e-4, references)["passed"])
                self.assertFalse(oracle_report(-5.0 + 2e-4, self.forces, references)["passed"])
                self.assertFalse(oracle_report(-5.0, self.forces * float("nan"), references)["passed"])
                # An IEEE run has no precision error to gate.
                noted = oracle_report(-5.0 + 3e-5, self.forces + 2e-6, references,
                                      precision_force_rmse_tolerance=1e-3)
                self.assertTrue(noted["passed"])
                self.assertIsNone(noted["precision"])
                self.assertEqual(torch.get_float32_matmul_precision(), "high")

    def test_tf32_run_separates_implementation_and_precision_error(self):
        torch.set_float32_matmul_precision("high")
        predict, calls = self.predictor({"ieee": (0.0, 0.0), "tf32": (1e-5, 2e-3)})
        references = compute_references(predict, "high")
        self.assertEqual(calls, ["ieee", "tf32"])
        self.assertEqual(references["run_fp32_gemm"], "tf32")
        self.assertEqual(references["highest_fp32_gemm"], "ieee")
        tested = self.forces + 2e-3 + 1e-4
        report = oracle_report(-5.0 + 1e-5, tested, references)
        self.assertTrue(report["passed"])
        self.assertEqual(report["implementation"]["reference"], "same_precision")
        self.assertAlmostEqual(report["implementation"]["force_RMSE_eV_A"], 1e-4, delta=1e-7)
        self.assertEqual(report["implementation"]["tolerance"], IMPLEMENTATION_TOLERANCES["tf32"])
        self.assertAlmostEqual(report["precision"]["tested_vs_highest"]["force_RMSE_eV_A"], 2.1e-3, delta=1e-7)
        self.assertAlmostEqual(report["precision"]["reference_vs_highest"]["force_RMSE_eV_A"], 2e-3, delta=1e-7)
        self.assertIsNone(report["precision"]["passed"])
        gated = oracle_report(-5.0 + 1e-5, tested, references, precision_force_rmse_tolerance=1e-3)
        self.assertFalse(gated["precision"]["passed"])
        self.assertFalse(gated["passed"])
        loose = oracle_report(-5.0 + 1e-5, tested, references, precision_force_rmse_tolerance=1e-2)
        self.assertTrue(loose["passed"])

    def test_implementation_tolerance_replaces_only_its_mode(self):
        tight = {"ieee": {"energy_error_eV_per_atom": 1e-5, "force_RMSE_eV_A": 1e-5,
                          "force_max_error_eV_A": 1e-5}}
        predict, _ = self.predictor({"ieee": (0.0, 0.0)})
        references = compute_references(predict, "highest")
        self.assertTrue(oracle_report(-5.0 + 3e-5, self.forces + 2e-6, references)["passed"])
        report = oracle_report(-5.0 + 3e-5, self.forces + 2e-6, references, implementation_tolerance=tight)
        self.assertFalse(report["passed"])
        self.assertEqual(report["implementation"]["tolerance"], tight["ieee"])
        self.assertTrue(oracle_report(-5.0 + 5e-6, self.forces + 2e-6, references,
                                      implementation_tolerance=tight)["passed"])
        self.assertFalse(oracle_report(-5.0, self.forces + 2e-5, references, implementation_tolerance=tight)["passed"])
        predict, _ = self.predictor({"ieee": (0.0, 0.0), "tf32": (1e-5, 2e-3)})
        references = compute_references(predict, "high")
        report = oracle_report(-5.0 + 1e-5, self.forces + 2e-3 + 1e-4, references, implementation_tolerance=tight)
        self.assertEqual(report["implementation"]["tolerance"], IMPLEMENTATION_TOLERANCES["tf32"])
        self.assertTrue(report["passed"])










@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CudaPrecisionTest(PrecisionStateTestCase):
    def test_gemm_math_follows_effective_mode(self):
        a = torch.randn(512, 512, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
        reference = a.double() @ a.double()
        for requested in ("highest", "high"):
            with self.subTest(requested=requested), matmul_precision_scope(requested):
                error = float(((a @ a).double() - reference).norm() / reference.norm())
                expected_tf32 = fp32_gemm_mode() == "tf32"
                self.assertEqual(error > 1e-5, expected_tf32, error)

    def test_calculator_warns_only_without_requested_precision(self):
        from fastmd.interfaces.calculator import MLIPCalculator
        torch.set_float32_matmul_precision("high")
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            MLIPCalculator(model=toy_model(), device="cuda",
                             config=InferenceConfig(matmul_precision="highest", report=False))
        self.assertFalse([w for w in seen
                          if issubclass(w.category, ConfigurationWarning) and "TF32" in str(w.message)])
        with self.assertWarnsRegex(ConfigurationWarning, "TF32"):
            MLIPCalculator(model=toy_model(), device="cuda", config=InferenceConfig(report=False))


def _graph_builders_available():
    if not torch.cuda.is_available():
        return False
    from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
    from fastmd.graph.gpu_graph_builder import op_available
    return op_available and _fixed_builder_available




@unittest.skipUnless(_graph_builders_available(), "CUDA graph builders are unavailable")
class ApplicationPrecisionCudaTest(PrecisionStateTestCase):
    """GEMM-level evidence that entry points run their configured precision.

    The caller's process precision is the opposite of the configured one.
    Forces are compared with eager generic references at 'highest' and 'high':
    IEEE paths agree with the former to ~3e-7 eV/A, TF32 paths differ by
    ~6e-5 (200x). CUDA graphs replay the GEMM kernels chosen at capture, so a
    replayed IEEE result proves the capture ran under the configured scope.
    """

    def model(self):
        from fastmd.models.matris.model import MatRIS
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(1234)
            model = MatRIS(num_layers=2, norm_type="layer", node_feat_dim=64, edge_feat_dim=32,
                           three_body_feat_dim=32, mlp_hidden_dims=(64, 64), num_radial=8, num_angular=8,
                           max_n=3, max_l=3, pairwise_cutoff=2.0, three_body_cutoff=1.5,
                           reference_energy="demo", enable_compile=False, enable_checkpoint=False)
        return model.eval().requires_grad_(False).cuda()

    @classmethod
    def setUpClass(cls):
        # 64 atoms on a jittered 1.25 A lattice: 576 bonds, 1920 angles.
        rng = np.random.default_rng(5)
        grid = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3) * 1.25
        cls.atoms = Atoms(numbers=rng.choice([1, 6, 8], len(grid)), cell=[5.0] * 3, pbc=True,
                          positions=grid + rng.uniform(-0.05, 0.05, grid.shape))

    def setUp(self):
        super().setUp()
        from fastmd.graph import atoms_to_graph_gpu
        references = {}
        for requested in ("highest", "high"):
            model = self.model()
            generic = resolve_config(InferenceConfig(execution="eager", optimization_profile="generic",
                                                     report=False))
            with generic.scope(), matmul_precision_scope(requested):
                graph = atoms_to_graph_gpu(self.atoms, atom_graph_cutoff=2.0, line_graph_cutoff=1.5,
                                           check_isolated_atoms=False)
                references[requested] = model([graph], task="ef", is_training=False)["f"][0].detach().cpu()
        self.ieee = references["highest"]
        self.tf32_gap = float((references["high"] - self.ieee).abs().max())
        self.assertGreater(self.tf32_gap, 1e-5, "TF32 is indistinguishable here; the test cannot discriminate")

    def assert_math(self, forces, expected):
        error = max(float((f.detach().cpu() - self.ieee).abs().max()) for f in forces)
        if expected == "ieee":
            self.assertLess(error, self.tf32_gap / 20, error)
        else:
            self.assertGreater(error, self.tf32_gap / 5, error)

    def test_gpu_md_capture_and_replay_follow_config(self):
        from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
        capacity = CapacityConfig(u_step=8, t_step=32, min_pad_u=4, min_pad_t=8,
                                  n_dummy=4, warmup=1, candidate_skin=1.0)
        for execution in ("model_graph", "whole_step"):
            for process, requested, expected in (("high", "highest", "ieee"), ("highest", "high", "tf32")):
                with self.subTest(execution=execution, process=process, requested=requested):
                    torch.set_float32_matmul_precision(process)
                    model = self.model()
                    modes = record_forward_modes(model)
                    config = InferenceConfig(execution=execution, optimization_profile="topology",
                                             checkpoint="off", capacity=capacity, report=False,
                                             matmul_precision=requested)
                    with GPUResidentMolecularDynamics(self.atoms, model=model, task="ef", device="cuda",
                                                      ensemble="nve", timestep=0.0, config=config) as md:
                        # Construction captures (buckets / whole step) under the scope.
                        self.assertTrue(modes)
                        self.assertEqual(set(modes), {expected})
                        forces = []
                        for _ in range(3):
                            md.run(1)
                            forces.append(md.state.forces.clone())
                        forces.append(md.evaluate(md.state.positions)[0].clone())
                        self.assertEqual(set(modes), {expected})
                        if execution == "model_graph":  # the captured buckets ran
                            self.assertIsNotNone(md.runner)
                            self.assertGreater(md.runner.captures, 0)
                    self.assert_math(forces, expected)
                    self.assertEqual(torch.get_float32_matmul_precision(), process)

    def test_engines_sharing_a_model_keep_their_captured_caches(self):
        """Model-graph engines sharing ONE model with highest / high / highest (GPU MD and calculator):
        each later engine rebuilds the precision-keyed frozen projection caches that an earlier
        engine's captured graph still reads (it keeps them alive; the released blocks are refilled
        with NaN here). Each engine matches an engine of its precision on its own copy of the model;
        a weight edited after the capture stops the replay (StaleCaptureError), eager still works."""
        import copy

        from fastmd.interfaces.calculator import MLIPCalculator
        from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
        from fastmd.engine.cache_provenance import StaleCaptureError
        # Three-body Fourier basis and enough rows for all three cache groups (combined three-body,
        # merged attention and indexed split projections).
        from test_training_plain_torch import _atoms, _model

        atoms = _atoms()

        def md(model, precision, execution="model_graph"):
            engine = GPUResidentMolecularDynamics(
                atoms, model=model, task="ef", device="cuda", ensemble="nve", timestep=0.0,
                config=InferenceConfig(execution=execution, optimization_profile="topology",
                                       report=False, matmul_precision=precision))
            positions = engine.state.positions.clone()
            return lambda: engine.evaluate(positions)[0].detach().cpu()

        def calculator(model, precision):
            calc = MLIPCalculator(model=model, task="ef", device="cuda", config=InferenceConfig(
                execution="model_graph", optimization_profile="topology", report=False,
                matmul_precision=precision))
            return lambda: torch.as_tensor(calc.get_forces(atoms.copy()))

        model = _model().cuda().requires_grad_(False)
        precisions = ("highest", "high", "highest")
        engines = [(factory, p, factory(model, p)) for factory in (md, calculator) for p in precisions]
        sizes = {t.numel() for m in model.modules() for name in ("_combined_swish_weight", "_merged_attn_proj_weight")
                 if isinstance(t := vars(m).get(name), torch.Tensor)}
        poison = [torch.full((n,), float("nan"), device="cuda") for n in sizes for _ in range(32)]
        references = {(factory, p): factory(copy.deepcopy(model), p)() for factory in (md, calculator)
                      for p in set(precisions)}
        gap = float((references[md, "highest"] - references[md, "high"]).abs().max())
        for _ in range(2):
            for factory, p, engine in engines:
                with self.subTest(engine=factory.__name__, precision=p):
                    error = float((engine() - references[factory, p]).abs().max())
                    # TF32 run-to-run noise (atomics) reaches ~gap/8; a cache of the other precision
                    # is off by ~gap, a released one by NaN or garbage.
                    self.assertLess(error, gap / 3, (error, gap))
        del poison
        with torch.no_grad():
            model.three_body_embedding.angle_embedding.weight.add_(0.0)  # an in-place weight load
        with self.assertRaises(StaleCaptureError):
            engines[0][2]()
        self.assertTrue(torch.isfinite(md(model, "highest", "eager")()).all())


if __name__ == "__main__":
    unittest.main()
