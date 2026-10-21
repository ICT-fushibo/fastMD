"""CPU-only tests: entrypoint migration, per-instance scope, and reporting.

``assert_adapter_matches_model`` also runs on CUDA (test_config_cuda).
"""
import ast
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace
from types import ModuleType
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import torch
from ase import Atoms

from fastmd.adapters import MatRISAdapter, ModelAdapter, ModelOwnedDerivative, as_adapter, load_adapter
from fastmd.adapters.memory import MeasuredMemoryModel, MemoryPoint
from fastmd.adapters.negotiate import (NegotiationError, dry_run, loosest_class, negotiate, negotiated,
                                       trial_candidates)
from fastmd.adapters.protocol import CaptureContract, KernelChoice, PaddingRule, TopologySpec
from fastmd.engine import memory_admission
import fastmd.models
from fastmd.precision import fp32_gemm_mode
from fastmd.config import CapacityConfig, InferenceConfig
from fastmd.interfaces.calculator import MLIPCalculator
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.interfaces.md import MolecularDynamics
from fastmd.execution.md_protocol import HeatingRamp
from fastmd.interfaces.relax import StructOptimizer
from fastmd.engine.entry_config import REMOVED_INFERENCE_ARGUMENTS
from fastmd.graph import RadiusGraph
from fastmd.models.matris.model import MatRIS
from fastmd.models.matris import functions

from test_isolated_atom_handling import make_graph, make_model


def fake_model(*, gpu_graph=False, reference=True):
    model = Mock(spec=MatRIS)
    model.to.return_value = model
    model.parameters.return_value = []
    model.modules.return_value = []
    model.reference_energy = object() if reference else None
    model.enable_checkpoint = True
    model.config = {"enable_checkpoint": True}
    model.interaction_block = [SimpleNamespace(enable_checkpoint=True)]
    model.graph_converter = Mock()
    model.graph_converter.algorithm = "gpu" if gpu_graph else "legacy"
    model.graph_converter.atom_graph_cutoff = 2.0
    model.graph_converter.line_graph_cutoff = 1.5
    graph = Mock(spec=RadiusGraph)
    graph.to.return_value = graph
    model.graph_converter.return_value = graph
    model.return_value = {"e": torch.tensor([1.0]), "f": [torch.zeros(2, 3)], "ref_energy": torch.zeros(1)}
    return model


def atoms():
    return Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[20, 20, 20], pbc=True)


@contextmanager
def deterministic_algorithms():
    """Run-to-run bitwise CUDA forces (ordered sums instead of atomics); restores the flags."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ...but an op without a deterministic implementation fails the test.
            warnings.filterwarnings("error", message=".*does not have a deterministic implementation.*")
            yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def assert_adapter_matches_model(test, device):
    """E/F/S/m through the adapter are the direct model call's, bitwise; its extensive energy is the
    runners' FP32 multiply by the real atom count (intensive) or the energy itself (extensive)."""
    def parts(value):
        return list(value) if isinstance(value, (list, tuple)) else [value]

    for intensive in (True, False):
        with test.subTest(device=device, intensive=intensive):
            torch.manual_seed(0)
            model = make_model(num_layers=2, norm_type="layer", is_intensive=intensive).to(device)
            graphs = [make_graph(3, True).to(device), make_graph(2, True).to(device)]
            adapter = load_adapter(model)
            with deterministic_algorithms():
                model(graphs, task="efsm", is_training=False)  # first call: lazy caches
                expected = model(graphs, task="efsm", is_training=False)
                output = adapter.evaluate(graphs, task="efsm", is_training=False)
            for key in ("e", "f", "s", "m"):
                for got, want in zip(parts(output[key]), parts(expected[key]), strict=True):
                    test.assertTrue(torch.equal(got, want), key)
            counts = expected["atoms_per_graph"]
            test.assertTrue(torch.equal(adapter.extensive_energy(output["e"], counts),
                                        expected["e"] * counts if intensive else expected["e"]))
            first = adapter.extensive_energy(output["e"].reshape(-1)[:1], 3)  # a single-system runner's count
            test.assertEqual(first.dtype, torch.float32)
            test.assertTrue(torch.equal(first, expected["e"][:1] * 3 if intensive else expected["e"][:1]))
            test.assertIs(adapter.reference_energies(), model.reference_energy.fc.weight)
            test.assertEqual(adapter.isolated_atom_policy(), "reference")
            spec = adapter.topology_spec()
            test.assertEqual((spec.cutoff, spec.triplet_cutoff, spec.num_interactions, spec.undirected_pairs,
                              spec.model_dtype), (2.0, 1.5, 2, True, "float32"))


class RuntimeConfigTest(unittest.TestCase):
    def test_calculator_omission_preserves_old_load_defaults(self):
        with patch("fastmd.models.matris.model.MatRIS.load", return_value=fake_model()) as load:
            calc = MLIPCalculator(device="cpu")
        self.assertFalse(load.call_args.kwargs["enable_checkpoint"])
        self.assertFalse(load.call_args.kwargs["enable_compile"])
        self.assertTrue(calc.handle_isolated_atoms)
        load.return_value.eval.assert_called_once_with()  # inference: eval mode

    def test_removed_flags_fail_before_ase_or_model_loading(self):
        with patch("fastmd.models.matris.model.MatRIS.load") as load:
            for name in REMOVED_INFERENCE_ARGUMENTS:
                with self.subTest(name=name), self.assertRaises(TypeError):
                    MLIPCalculator(device="cpu", **{name: False})
            # Mixed interfaces: a config plus a removed flag.
            with self.subTest(name="config + use_cuda_graph"), self.assertRaises(TypeError):
                MLIPCalculator(config=InferenceConfig(report=False), use_cuda_graph=False)
        load.assert_not_called()

    def test_old_import_paths_are_the_moved_modules(self):
        """Every module the layout move moved keeps a stub at its old path (its ``_NEW`` names the module): the old
        path is the moved module itself, also as ``from <old package> import <module>``, and a name written through
        an old package is written to its new package."""
        root = Path(__file__).resolve().parents[1]
        stubs = {}
        for path in sorted((root / "fastmd").rglob("*.py")):
            for node in ast.parse(path.read_text()).body:
                targets = [getattr(target, "id", None) for target in getattr(node, "targets", ())]
                if isinstance(node, ast.Assign) and targets == ["_NEW"]:
                    stubs[path] = node.value.value
        self.assertTrue(stubs)
        for path, new in stubs.items():
            parts = path.relative_to(root).with_suffix("").parts
            old = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            with self.subTest(old=old):
                try:
                    moved = importlib.import_module(new)
                except ImportError:  # an optional dependency of the moved module (torch-sim) is missing
                    continue
                if path.name == "__init__.py":  # an old package: its names are the new package's
                    package, probe = importlib.import_module(old), object()
                    package._old_path_probe = probe
                    try:
                        self.assertIs(moved._old_path_probe, probe)
                    finally:
                        del package._old_path_probe
                    self.assertFalse(hasattr(moved, "_old_path_probe"))
                else:
                    self.assertIs(importlib.import_module(old), moved)
                    parent, _, leaf = old.rpartition(".")
                    self.assertIs(getattr(importlib.import_module(parent), leaf), moved)

    def test_unsupported_and_unavailable_execution_fail(self):
        with patch("fastmd.models.matris.model.MatRIS.load") as load:
            for execution in ("model_graph", "whole_step"):
                with self.subTest(execution=execution), self.assertRaises(ValueError):
                    MLIPCalculator(device="cpu", config=InferenceConfig(execution=execution, report=False))
        load.assert_not_called()

    def test_profile_applies_during_load_and_calls_then_restores(self):
        previous = functions._INDEXED_CAT_LINEAR_FUSE
        observed = []
        model = fake_model()
        output = model.return_value
        model.side_effect = lambda *a, **k: (observed.append(functions._INDEXED_CAT_LINEAR_FUSE), output)[1]
        def loader(**kwargs):
            observed.append(functions._INDEXED_CAT_LINEAR_FUSE)
            return model
        with patch("fastmd.models.matris.model.MatRIS.load", side_effect=loader):
            calc = MLIPCalculator(device="cpu", task="ef", config=InferenceConfig(optimization_profile="topology", report=False))
        self.assertEqual(functions._INDEXED_CAT_LINEAR_FUSE, previous)
        calc._predict_structure(None, atoms=atoms())
        self.assertEqual(observed, [True, True])
        self.assertEqual(functions._INDEXED_CAT_LINEAR_FUSE, previous)

    def test_two_profiles_do_not_leak(self):
        topology_model, generic_model = fake_model(), fake_model()
        observations = []
        for model in (topology_model, generic_model):
            output = model.return_value
            model.side_effect = lambda *a, _out=output, **k: (observations.append(functions._INDEXED_CAT_LINEAR_FUSE), _out)[1]
        with patch("fastmd.models.matris.model.MatRIS.load", side_effect=[topology_model, generic_model]):
            topology = MLIPCalculator(task="ef", config=InferenceConfig(optimization_profile="topology", report=False))
            generic = MLIPCalculator(task="ef", config=InferenceConfig(optimization_profile="generic", report=False))
        previous = functions._INDEXED_CAT_LINEAR_FUSE
        topology._predict_structure(None)
        generic._predict_structure(None)
        topology._predict_structure(None)
        self.assertEqual(observations, [True, False, True])
        self.assertEqual(functions._INDEXED_CAT_LINEAR_FUSE, previous)

    def test_scope_restores_on_model_error(self):
        model = fake_model()
        with patch("fastmd.models.matris.model.MatRIS.load", return_value=model):
            calc = MLIPCalculator(task="ef", config=InferenceConfig(optimization_profile="topology", report=False))
        previous = functions._INDEXED_CAT_LINEAR_FUSE
        model.side_effect = RuntimeError("test failure")
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            calc._predict_structure(None)
        self.assertEqual(functions._INDEXED_CAT_LINEAR_FUSE, previous)

    def test_no_reference_table_reports_strict_policy(self):
        model = fake_model(reference=False)
        with patch("fastmd.models.matris.model.MatRIS.load", return_value=model):
            calc = MLIPCalculator(config=InferenceConfig(report=False))
        calc._predict_structure(None)
        self.assertTrue(model.graph_converter.call_args.kwargs["check_isolated_atoms"])

    def test_calculate_no_reference_table_keeps_gpu_fast_path(self):
        model = fake_model(gpu_graph=True, reference=False)
        model.is_intensive = True
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("fastmd.execution.cuda_graph.BucketedGraphRunner") as runner,
        ):
            # The per-call rebuild path (a mocked device cannot run the skin path).
            calc = MLIPCalculator(model=model, device="cuda", task="ef", config=InferenceConfig(
                execution="model_graph", report=False, topology_reuse="off"))
        runner.return_value.run.return_value = (model.return_value, 2)
        for _ in range(2):
            calc.calculate(atoms(), properties=["energy", "forces"], system_changes=[])
        self.assertTrue(model.graph_converter.call_args.kwargs["check_isolated_atoms"])
        self.assertIsNotNone(model.graph_converter.call_args.kwargs["atoms"])
        model.graph_converter.side_effect = ValueError("Detected isolated atom")
        with self.assertRaisesRegex(ValueError, "isolated atom"):
            calc.calculate(atoms(), properties=["energy", "forces"], system_changes=[])

    def test_no_cuda_uses_cpu_converter_even_when_gpu_extension_is_importable(self):
        from test_isolated_atom_handling import make_model
        # A model left in training mode with live dropout: the calculator runs it in eval mode,
        # so two calls give the same energy and forces.
        model = make_model(reference_energy=None).train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.5
        # Simulate an installed GPU graph extension without an available GPU.
        model.graph_converter.algorithm = "gpu"
        with patch("torch.cuda.is_available", return_value=False):
            calc = MLIPCalculator(model=model, task="ef", device="cpu", config=InferenceConfig(report=False))
            results = []
            for _ in range(2):
                calc.calculate(atoms(), properties=["energy", "forces"], system_changes=[])
                results.append((calc.results["energy"], calc.results["forces"].copy()))
        self.assertTrue(bool(torch.isfinite(torch.as_tensor(calc.results["energy"]))))
        self.assertTrue(bool(torch.isfinite(torch.as_tensor(calc.results["forces"])).all()))
        self.assertFalse(any(module.training for module in calc.model.modules()))
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-6, atol=1e-7)

    def test_execution_auto_resolves_per_entrypoint(self):
        from fastmd.engine.entry_config import _auto_execution
        from fastmd.config import resolve_config
        resolved = resolve_config(None, default_execution="auto")
        self.assertEqual((resolved.execution, resolved.optimization_profile), ("auto", "topology"))
        checkpoint = resolve_config(InferenceConfig(execution="auto", checkpoint="on", report=False))
        cuda, cpu = torch.device("cuda"), torch.device("cpu")
        with patch("torch.cuda.is_available", return_value=True):
            for kind, mode in (("calculator", "model_graph"), ("gpu_md", "whole_step"), ("torchsim", "eager")):
                with self.subTest(kind=kind):
                    self.assertEqual(_auto_execution(kind, cuda, resolved), mode)
                    # Checkpoint recomputation is not validated inside a capture.
                    self.assertEqual(_auto_execution(kind, cuda, checkpoint), "eager")
                    self.assertEqual(_auto_execution(kind, cpu, resolved), "eager")
        # CPU: every default resolves to the plain eager path without an error.
        with patch("fastmd.models.matris.model.MatRIS.load", return_value=fake_model(gpu_graph=True)):
            calc = MLIPCalculator(device="cpu", task="ef")
        calc._predict_structure(None, atoms=atoms())
        summary = calc.config_summary()
        self.assertEqual(summary["requested"]["execution"], "auto")
        self.assertEqual(summary["effective"]["execution"], "eager")
        self.assertEqual(summary["effective"]["auto_execution"]["mode"], "eager")
        self.assertIsNone(calc._auto)

    def test_capacity_passed_to_model_runner(self):
        capacity = CapacityConfig(u_step=1024, t_step=16384, n_dummy=128, min_pad_u=512, warmup=2)
        with (
            patch("fastmd.models.matris.model.MatRIS.load", return_value=fake_model(gpu_graph=True)),
            patch("torch.cuda.is_available", return_value=True),
            patch("fastmd.execution.cuda_graph.BucketedGraphRunner") as runner,
        ):
            MLIPCalculator(device="cuda", config=InferenceConfig(execution="model_graph", capacity=capacity, report=False))
        for name in ("u_step", "t_step", "n_dummy", "min_pad_u", "warmup"):
            self.assertEqual(runner.call_args.kwargs[name], getattr(capacity, name))

    def test_checkpoint_config_applies_to_passed_model(self):
        # The fake model arrives with checkpointing on: an explicit
        # checkpoint="off" and the entrypoints' default config both turn it off.
        explicit = InferenceConfig(execution="eager", checkpoint="off", isolated_atoms="error", report=False)
        for entrypoint, config in (("gpu_md", explicit), ("gpu_md", None), ("calculator", None)):
            with self.subTest(entrypoint=entrypoint, explicit=config is not None):
                model = fake_model()
                if entrypoint == "calculator":
                    MLIPCalculator(model=model, device="cpu")
                else:
                    with (
                        patch("fastmd.interfaces.gpu_md.TensorGraphBuilder"),
                        patch("fastmd.interfaces.gpu_md.AsyncGPUMDLogger"),
                    ):
                        kwargs = {} if config is None else {"config": config}
                        md = GPUResidentMolecularDynamics(atoms(), model=model, device="cpu", **kwargs)
                    self.assertFalse(model.config["enable_checkpoint"])
                    if config is not None:
                        self.assertTrue(md.validate_isolated_atoms)
                self.assertFalse(model.enable_checkpoint)
                self.assertFalse(model.interaction_block[0].enable_checkpoint)
                model.eval.assert_called_once_with()  # every entry point runs the model in eval mode

    def test_load_adapter_wraps_instances_and_loads_names_and_paths(self):
        # An adapter is kept and an instance wrapped as it is (evaluate is the module's own call); a
        # name or a path is loaded with MatRIS.load, and a checkpoint reaches every entry point with
        # its parameters under the same state_dict keys (the adapter registers nothing).
        model = make_model()
        adapter = load_adapter(model)
        for protocol in (MatRISAdapter, ModelAdapter, ModelOwnedDerivative):
            self.assertIsInstance(adapter, protocol)
        self.assertIs(adapter.module, model)
        self.assertIs(adapter.evaluate, model)
        self.assertIs(load_adapter(adapter), adapter)
        self.assertIs(as_adapter(adapter), adapter)
        self.assertEqual(load_adapter(make_model(reference_energy=None)).isolated_atom_policy(), "error")
        # Anything but an adapter or a MatRIS goes to MatRIS.load, as the entry points did (a
        # torch.compile wrapper is not taken for the model); the runners' as_adapter wraps and never loads.
        compiled = torch.compile(model)
        for given in ("matris_10m_mp", compiled):
            with self.subTest(given=type(given).__name__), patch.object(MatRIS, "load", return_value=model) as load:
                self.assertIs(load_adapter(given).module, model)
                self.assertIs(load.call_args.kwargs["model_name"], given)
                self.assertIs(as_adapter(compiled).module, compiled)
                load.assert_called_once()
        calc = MLIPCalculator(model=adapter, device="cpu")
        self.assertIs(calc.adapter, adapter)
        calc.model = rebound = make_model()  # eager calls follow a rebound model
        self.assertIs(calc.adapter.evaluate, rebound)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pth.tar")
            torch.save({"config": model.config, "state_dict": model.state_dict()}, path)
            with (
                patch("fastmd.interfaces.gpu_md.TensorGraphBuilder"),
                patch("fastmd.interfaces.gpu_md.AsyncGPUMDLogger"),
            ):
                loaded = {"load_adapter": load_adapter(model_path=path, device="cpu").module,
                          "calculator": MLIPCalculator(model_path=path, device="cpu").model,
                          "gpu_md": GPUResidentMolecularDynamics(atoms(), model_path=path, device="cpu").model}
        expected = model.state_dict()
        for entrypoint, module in loaded.items():
            with self.subTest(entrypoint=entrypoint):
                state = module.state_dict()
                self.assertEqual(list(state), list(expected))
                for key, value in expected.items():
                    self.assertTrue(torch.equal(state[key], value), key)
        # The registry is a table: a model added as a row is checked and loaded by its name, ahead of the
        # MatRIS default, which still loads every name no row claims.
        family = ModuleType("registry_test_family")
        family.claims = lambda given, model_path=None: given == "added"
        family.load = Mock(return_value=adapter)
        family.check_load_options = Mock()
        row = fastmd.models.ModelEntry("added", family.__name__, claims="claims", load="load")
        with patch.dict(sys.modules, {family.__name__: family}), \
                patch.object(fastmd.models, "MODELS", (row, *fastmd.models.MODELS)), \
                patch.object(MatRIS, "load", return_value=model) as load:
            self.assertIs(load_adapter("added", device="cpu", model_dtype="float32"), adapter)
            family.check_load_options.assert_called_once_with("float32", None, None)
            family.load.assert_called_once_with("added", model_path=None, device="cpu", model_dtype="float32",
                                                optimization_profile=None)
            self.assertIs(load_adapter("matris_10m_mp").module, model)
            load.assert_called_once()

    def test_entry_points_close_the_adapters_they_load(self):
        """An adapter an entry point loads by name is closed once, when the last entry point using it closes,
        exits its context or is finalised (an adapter's close reverts its upstream patches): the loading one, or
        another one it was passed to; an adapter the caller passed in stays open; a closed calculator refuses to
        calculate. MolecularDynamics and StructOptimizer close the calculator they made."""
        import gc
        model = make_model()
        made = []

        def loaded(given, *args, **kwargs):
            if isinstance(given, MatRISAdapter):
                return given
            adapter = MatRISAdapter(model)
            adapter.close = Mock()
            made.append(adapter)
            return adapter
        with (patch("fastmd.interfaces.calculator.load_adapter", side_effect=loaded),
              patch("fastmd.interfaces.gpu_md.load_adapter", side_effect=loaded),
              patch("fastmd.interfaces.gpu_md.TensorGraphBuilder"),
              patch("fastmd.interfaces.gpu_md.AsyncGPUMDLogger"),
              patch("fastmd.interfaces.md.VelocityVerlet"), patch("builtins.print")):
            calc = MLIPCalculator(model="by_name", device="cpu")
            calc.close()
            calc.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                calc.calculate(atoms(), properties=["energy"], system_changes=["positions"])
            with MLIPCalculator(model="by_name", device="cpu"):
                pass
            with GPUResidentMolecularDynamics(atoms(), model="by_name", device="cpu"):
                pass
            GPUResidentMolecularDynamics(atoms(), model="by_name", device="cpu").close()
            StructOptimizer(model="by_name", device="cpu").close()
            with MolecularDynamics(atoms(), model="by_name", ensemble="nve", device="cpu"):
                pass
            MLIPCalculator(model="by_name", device="cpu")  # dropped: closed when it is finalised
            gc.collect()
            self.assertEqual([adapter.close.call_count for adapter in made], [1] * 7)
            # Passed on to another entry point: open until the last user is closed.
            calc = MLIPCalculator(model="by_name", device="cpu")
            md = GPUResidentMolecularDynamics(atoms(), model=calc.adapter, device="cpu")
            calc.close()
            self.assertEqual(made[-1].close.call_count, 0)
            md.close()
            self.assertEqual(made[-1].close.call_count, 1)
            given = MatRISAdapter(model)
            given.close = Mock()
            MLIPCalculator(model=given, device="cpu").close()
            GPUResidentMolecularDynamics(atoms(), model=given, device="cpu").close()
            gc.collect()
            given.close.assert_not_called()

    def test_adapter_outputs_equal_the_direct_model_call(self):
        assert_adapter_matches_model(self, "cpu")

    def test_high_level_signatures_have_only_unified_inference_config(self):
        for cls in (MLIPCalculator, GPUResidentMolecularDynamics, MolecularDynamics, StructOptimizer):
            with self.subTest(cls=cls.__name__):
                params = inspect.signature(cls).parameters
                self.assertIn("config", params)
                self.assertFalse(REMOVED_INFERENCE_ARGUMENTS.intersection(params))

    def test_md_and_relax_forward_config(self):
        config = InferenceConfig(optimization_profile="topology", report=False)
        calc = Mock()
        calc._graph_runner = None
        with (
            patch("fastmd.interfaces.md.MLIPCalculator", return_value=calc) as load,
            patch("fastmd.interfaces.md.VelocityVerlet"),
            patch("builtins.print"),
        ):
            MolecularDynamics(atoms(), ensemble="nve", config=config)
        self.assertIs(load.call_args.kwargs["config"], config)
        self.assertFalse(REMOVED_INFERENCE_ARGUMENTS.intersection(load.call_args.kwargs))
        with patch("fastmd.interfaces.relax.MLIPCalculator", return_value=calc) as load:
            StructOptimizer(config=config)
        self.assertIs(load.call_args.kwargs["config"], config)
        self.assertFalse(REMOVED_INFERENCE_ARGUMENTS.intersection(load.call_args.kwargs))

    def test_md_relax_and_gpu_md_reject_removed_constructor_flags(self):
        for cls, positional in ((MolecularDynamics, (atoms(),)), (StructOptimizer, ()), (GPUResidentMolecularDynamics, (atoms(),))):
            with self.subTest(cls=cls.__name__), self.assertRaises(TypeError):
                cls(*positional, enable_checkpoint=False)

    def test_relax_does_not_forward_removed_flags_to_optimizer(self):
        with patch("fastmd.interfaces.relax.MLIPCalculator"):
            optimizer = StructOptimizer()
        with self.assertRaisesRegex(TypeError, "Removed inference arguments"):
            optimizer.relax(atoms(), use_cuda_graph=False)

    def test_checkpoint_setter_refuses_any_captured_graph(self):
        for config in (None, InferenceConfig(execution="eager", report=False)):
            with (
                self.subTest(explicit=config is not None),
                patch("fastmd.interfaces.gpu_md.TensorGraphBuilder"),
                patch("fastmd.interfaces.gpu_md.AsyncGPUMDLogger"),
            ):
                md = GPUResidentMolecularDynamics(atoms(), model=fake_model(), device="cpu", config=config)
            md.runner = SimpleNamespace(captures=1)
            with self.assertRaises(ValueError):
                md.set_activation_checkpoint(True)

    def test_gpu_md_capacity_reaches_builder_runner_and_policy(self):
        def make_md(capacity, **kwargs):
            with (
                patch("fastmd.interfaces.gpu_md.TensorGraphBuilder"),
                patch("fastmd.interfaces.gpu_md.AsyncGPUMDLogger"),
            ):
                return GPUResidentMolecularDynamics(atoms(), model=fake_model(), device="cpu", config=InferenceConfig(execution="eager", capacity=capacity, report=False), **kwargs)

        with self.subTest(target="fixed builder and runner"):
            md = make_md(CapacityConfig(n_dummy=128, warmup=1, candidate_skin=3.0))
            md._whole_u_capacity, md._whole_t_capacity = 64, 128
            with (
                patch("fastmd.interfaces.gpu_md.FixedCapacityGraphBuilder") as builder,
                patch("fastmd.interfaces.gpu_md.WholeStepGraphRunner") as runner,
            ):
                md._make_whole_step_runner(positions=md.state.positions, momenta=md.state.momenta)
            self.assertEqual(builder.call_args.kwargs["n_dummy"], 128)
            self.assertEqual(builder.call_args.kwargs["candidate_cutoff"], 5.0)
            self.assertEqual(runner.call_args.kwargs["warmup"], 1)
        with self.subTest(target="whole-step capacity policy"):
            capacity = CapacityConfig(u_step=1024, t_step=16384, min_pad_u=512, min_pad_t=8192)
            md = make_md(capacity)
            # The constructor stays CPU-only; replace capture, not capacity policy.
            md.device = torch.device("cuda")
            md.graph_builder.build.return_value = SimpleNamespace(
                undirected2directed=torch.zeros(2000), line_graph=torch.zeros(18000),
            )
            with patch.object(md, "_make_whole_step_runner"), patch.object(md, "_sync_whole_state"):
                md.enable_whole_step_cuda_graph()
            for name in ("u_step", "t_step", "min_pad_u", "min_pad_t"):
                self.assertEqual(getattr(md.capacity_policy, name), getattr(capacity, name))
        with self.subTest(target="fix_com"):
            # Default on for nve/nvt (default_fix_com, also from_evaluator's); NHC has none.
            self.assertTrue(make_md(CapacityConfig()).integrator.fix_com)
            self.assertFalse(make_md(CapacityConfig(), fix_com=False).integrator.fix_com)
            self.assertFalse(make_md(CapacityConfig(), ensemble="nvt_nhc").fix_com)
            with self.assertRaisesRegex(ValueError, "nvt_nhc.*fix_com"):
                make_md(CapacityConfig(), ensemble="nvt_nhc", fix_com=True)
        with self.subTest(target="ramp"):
            ramp = HeatingRamp(300.0, 600.0, 10)
            with self.assertRaisesRegex(ValueError, "heating ramp needs a thermostat"):
                make_md(CapacityConfig(), ensemble="nve", ramp=ramp)
            md = make_md(CapacityConfig(), ramp=ramp)
            self.assertIs(md.ramp, ramp)
            self.assertEqual(md.temperature_target, 300.0)
            md.set_temperature(450.0)
            self.assertEqual(md.temperature_target, 450.0)
            with self.assertRaisesRegex(ValueError, "nve has no thermostat"):
                make_md(CapacityConfig(), ensemble="nve").set_temperature(450.0)





def fake_adapter(*, contract=None, kernels=None, exchange=None, levels=None, model_dtype="float32",
                 compiled=False, system=None, policy="model", on_select=None, padding=PaddingRule("far_self_loops"),
                 patches=None, triplet_cutoff=None, tap=False):
    """A model other than MatRIS for the negotiation: the protocol's members, and of the optional
    capabilities only the declared ones (the negotiation discovers them with isinstance). ``kernels``:
    KernelChoice sites; ``on_select(site, option)`` may raise (the option is unavailable). ``selected``
    lists the installed options. ``patches``: the outcomes of its upstream patch records.
    ``triplet_cutoff``: a line graph; ``tap``: an energy tap (``node_energy``)."""
    members = {
        "name": "fake", "compiled": compiled,
        "topology_spec": lambda self: TopologySpec(cutoff=5.0, num_interactions=2, model_dtype=model_dtype,
                                                   triplet_cutoff=triplet_cutoff),
        "padding_rule": lambda self: padding, "prepare": lambda self, batch: batch,
        "isolated_atom_policy": lambda self: policy, "reference_energies": lambda self: None,
    }
    if contract is not None:
        members["capture_contract"] = lambda self: contract
    if kernels is not None:
        members["kernel_backends"] = lambda self: kernels

        def select_kernel(self, site, option):
            if on_select is not None:
                on_select(site, option)
            self.selected.append((site, option))
        members["select_kernel"] = select_kernel
    if exchange is not None:
        members["exchange_strategy"] = lambda self: exchange
    if levels is not None:
        members["recompute_levels"] = levels
    if system is not None:
        members["system_inputs"] = lambda self: system
    if patches is not None:
        members["upstream_patches"] = lambda self: patches
        members["close"] = lambda self: None
    if tap:
        members["node_energy"] = lambda self, inputs, positions, cell: positions[:, 0]
    adapter = type("FakeAdapter", (), members)()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        adapter.module, adapter.selected = torch.nn.Linear(3, 2), []
    return adapter


def memory_points(kind, edges, per_edge=2048.0, base=64 * 2**20, noise=(), triplets=None):
    """Synthetic peaks, linear in the edges (plus ``noise`` bytes per point)."""
    noise = list(noise) + [0] * len(edges)
    return [MemoryPoint(kind, e, 0 if triplets is None else triplets[i], int(base + per_edge * e + noise[i]))
            for i, e in enumerate(edges)]


def measured(valid=True):
    """A measured memory model: eager and capture fitted on three points (valid), or on two."""
    edges = (10_000, 20_000, 40_000) if valid else (10_000, 20_000)
    return MeasuredMemoryModel(memory_points("eager", edges) + memory_points("capture", edges, per_edge=4096.0),
                               triplets=False)


class NegotiationTest(unittest.TestCase):
    """The capability negotiation (fastmd.adapters.negotiate): fallback chains and their reasons, the
    accuracy-class rule, its errors, the precision scope, the measured memory model and MatRIS's
    record (decisions read from the runtime, none made anew)."""

    CONTRACT = CaptureContract(sync_free=True, exact_cutoff=True)

    def negotiate(self, adapter, config=None, *, cuda=False, builder=True, **kwargs):
        """negotiate() as on a CUDA device (mocked: availability and the builder only) or on the CPU."""
        with patch.object(torch.cuda, "is_available", return_value=cuda), \
                patch("fastmd.graph.fixed_capacity_builder._fixed_builder_available", builder):
            return negotiate(adapter, config, device="cuda" if cuda else "cpu", **kwargs)

    def test_missing_capabilities_resolve_to_off_with_reasons(self):
        adapter = fake_adapter()  # declares nothing: no contract, kernels, recompute or exchange
        for recompute in (None, "on"):
            with self.subTest(recompute=recompute):
                record = self.negotiate(adapter, InferenceConfig(saved_tensor_recompute=recompute, report=False))
                chosen = {d.id: d.chosen for d in record.decisions}
                self.assertEqual(chosen, {"neighbor_list": "host", "topology_reuse": "rebuild", "padding": "none",
                                          "kernels": "upstream", "derivative": "engine_autograd", "execution": "eager",
                                          "precision": "ieee", "recompute": "off", "multi_gpu": "single_gpu"})
                self.assertIn("no CUDA device", record.decision("neighbor_list").reason)
                self.assertIn("eager: the graph is rebuilt every step", record.decision("topology_reuse").reason)
                self.assertIn("not declared sync-free", record.decision("execution").reason)
                self.assertIn("implements no saved-tensor recompute", record.decision("recompute").reason)
                self.assertIn("not declared", record.decision("derivative").reason)
                self.assertFalse(record.contract_declared)
                self.assertEqual(record.decision("execution").chain, ("whole_step", "model_graph", "eager"))
                self.assertEqual(record.accuracy_class, "fp32_noise")
                self.assertEqual(record.isolated_atoms["policy"], "model")
                self.assertEqual(record.memory["calibration"], "uncalibrated")
        record = self.negotiate(fake_adapter(levels=("off", "on")), InferenceConfig(saved_tensor_recompute="on"))
        self.assertEqual(record.decision("recompute").chosen, "on")
        record = self.negotiate(fake_adapter(levels=("off", "on")))  # auto: no calibrated rule for this model
        self.assertEqual(record.decision("recompute").chosen, "off")
        self.assertIn("no calibrated recompute rule", record.decision("recompute").reason)

    def test_execution_chain_needs_a_valid_measured_memory_bound(self):
        """CUDA (mocked): whole_step -> model_graph -> eager; a graph only on a valid measured fit whose
        bound fits the budget (K6 has no calibration for the model); the whole-step trial compares with
        model_graph at the 1.01x threshold; explicit requests are kept."""
        capacity, budget = {"edges": 30_000}, 64 * 2**30
        cases = [
            ("no memory model", {}, "eager", "no memory bound"),
            ("invalid fit", {"memory": measured(valid=False)}, "eager", "measured memory fit invalid"),
            ("too little memory", {"memory": measured(), "budget": 2**20}, "eager", "measured capture bound"),
            ("beyond the measured range", {"memory": measured(), "capacity": {"edges": 10**6}}, "eager",
             "outside the measured range"),
            ("fits, trial pending", {"memory": measured()}, "whole_step", "is pending"),
            ("not periodic", {"memory": measured(), "pbc": (True, True, False)}, "model_graph", "full PBC"),
            ("trial below 1.01x", {"memory": measured(), "whole_step_trial": lambda: 1.005}, "model_graph",
             "1.005x of model_graph (< 1.01)"),
            ("trial above 1.01x", {"memory": measured(), "whole_step_trial": lambda: 1.2}, "whole_step",
             "1.200x of model_graph (>= 1.01)"),
            ("trial failed", {"memory": measured(), "whole_step_trial": lambda: float("nan")}, "model_graph",
             "the whole-step trial failed"),
        ]
        for name, kwargs, mode, reason in cases:
            with self.subTest(name):
                kwargs = {"capacity": capacity, "budget": budget, **kwargs}
                record = self.negotiate(fake_adapter(contract=self.CONTRACT), cuda=True, **kwargs)
                self.assertEqual(record.decision("execution").chosen, mode)
                self.assertIn(reason, record.decision("execution").reason)
        bound = measured().bound("capture", 30_000)
        record = self.negotiate(fake_adapter(contract=self.CONTRACT), cuda=True, memory=measured(), capacity=capacity,
                                budget=budget)
        self.assertEqual(record.decision("execution").evidence["memory"]["capture_bound_gib"], round(bound / 2**30, 3))
        # model_graph holding two graph pools (GPU-resident MD with log_stress): its bound is two captures'. Where
        # one capture fits but two do not, whole_step without the trial (which would run model_graph), or eager
        # where whole_step is not available; where both fit, the trial as before.
        from fastmd.engine.saved_recompute import AUTO_THRESHOLD
        between = int(1.5 * bound / AUTO_THRESHOLD)

        def no_trial():
            raise AssertionError("the trial runs model_graph, whose two pools do not fit")
        for name, kwargs, mode, reason in (
                ("two pools fit", {"budget": budget, "whole_step_trial": lambda: 1.0}, "model_graph", "trial"),
                ("one pool fits", {"budget": between, "whole_step_trial": no_trial}, "whole_step",
                 "model_graph holds 2 graph pools (measured bound of 2 captures"),
                ("one pool fits, no whole_step", {"budget": between, "pbc": (True, True, False)}, "eager",
                 "model_graph holds 2 graph pools: measured bound of 2 captures")):
            with self.subTest(name):
                pooled = self.negotiate(fake_adapter(contract=self.CONTRACT), cuda=True, memory=measured(),
                                        capacity=capacity, model_graph_pools=2, **kwargs)
                execution = pooled.decision("execution")
                self.assertEqual(execution.chosen, mode)
                self.assertIn(reason, execution.reason)
                self.assertEqual(execution.evidence["memory"]["model_graph"]["pools"], 2)
        self.assertTrue(record.memory["valid"])
        # Topology reuse and padding follow the execution decision (the paths that implement them).
        self.assertEqual((record.decision("topology_reuse").chosen, record.decision("padding").chosen),
                         ("exact_skin_refilter", "far_self_loops"))
        eager = self.negotiate(fake_adapter(contract=self.CONTRACT), cuda=True)
        self.assertEqual((eager.decision("execution").chosen, eager.decision("topology_reuse").chosen,
                          eager.decision("padding").chosen), ("eager", "rebuild", "none"))
        # The default budget is what a capture may use above the current allocation.
        gib = 2**30
        with patch.object(torch.cuda, "mem_get_info", return_value=(60 * gib, 185 * gib)), \
                patch.object(torch.cuda, "memory_reserved", return_value=40 * gib), \
                patch.object(torch.cuda, "memory_allocated", return_value=99 * gib):
            tight = self.negotiate(fake_adapter(contract=self.CONTRACT), cuda=True, memory=measured(),
                                   capacity={"edges": 80_000})
        self.assertEqual(tight.decision("execution").chosen, "eager")
        self.assertLess(tight.decision("execution").evidence["memory"]["budget_gib"], 1.0)
        # One accuracy class: a "reduced" model's trial still chooses between its two graph paths.
        reduced = CaptureContract(sync_free=True, exact_cutoff=True, accuracy_class="reduced")
        self.assertEqual(self.negotiate(fake_adapter(contract=reduced), cuda=True, memory=measured(), capacity=capacity,
                                        budget=budget, whole_step_trial=lambda: 2.0).decision("execution").chosen,
                         "whole_step")
        # An explicit capture of a model that is not declared sync-free resolves to eager, with the reason.
        unsafe = self.negotiate(fake_adapter(), InferenceConfig(execution="whole_step"), cuda=True)
        self.assertEqual(unsafe.decision("execution").chosen, "eager")
        self.assertIn("requested whole_step, but not declared sync-free", unsafe.decision("execution").reason)
        calculator = self.negotiate(fake_adapter(contract=self.CONTRACT), entry="calculator", cuda=True,
                                    memory=measured(), capacity=capacity, budget=budget)
        self.assertEqual((calculator.decision("execution").chosen, calculator.decision("topology_reuse").chosen),
                         ("model_graph", "exact_skin_refilter"))
        for name, adapter, config, reason in (
                ("dynamic compile", fake_adapter(contract=self.CONTRACT, compiled=True), None, "dynamic torch.compile"),
                ("checkpointing", fake_adapter(contract=self.CONTRACT), InferenceConfig(checkpoint="on"),
                 "checkpointing"),
                ("no warmup", fake_adapter(contract=self.CONTRACT), InferenceConfig(capacity=CapacityConfig(warmup=0)),
                 "capacity.warmup=0")):
            with self.subTest(name):
                record = self.negotiate(adapter, config, cuda=True, memory=measured(), capacity=capacity, budget=budget)
                self.assertEqual(record.decision("execution").chosen, "eager")
                self.assertIn(reason, record.decision("execution").reason)
        safe = CaptureContract(sync_free=True, exact_cutoff=True, compiled="capture_safe")
        record = self.negotiate(fake_adapter(contract=safe, compiled=True), cuda=True, memory=measured(),
                                capacity=capacity, budget=budget)
        self.assertEqual(record.decision("execution").chosen, "whole_step")  # static compiled modules capture
        record = self.negotiate(fake_adapter(contract=self.CONTRACT), InferenceConfig(execution="model_graph"),
                                cuda=True)
        self.assertEqual((record.decision("execution").chosen, record.decision("execution").reason),
                         ("model_graph", "explicit request"))
        for config, contract, reason in (
                (InferenceConfig(topology_reuse="off", report=False), self.CONTRACT, "topology_reuse='off'"),
                (None, CaptureContract(sync_free=True, exact_cutoff=False), "no exact cutoff"),
                (InferenceConfig(capacity=CapacityConfig(candidate_skin=0.0), report=False), self.CONTRACT,
                 "candidate_skin is 0")):
            reuse = self.negotiate(fake_adapter(contract=contract), config, entry="calculator", cuda=True,
                                   memory=measured(), capacity=capacity, budget=budget).decision("topology_reuse")
            self.assertEqual(reuse.chosen, "rebuild")
            self.assertIn(reason, reuse.reason)
        no_cutoff = self.negotiate(fake_adapter(contract=CaptureContract(sync_free=True, exact_cutoff=False)),
                                   cuda=True, memory=measured(), capacity=capacity, budget=budget).decision("execution")
        self.assertEqual(no_cutoff.chosen, "model_graph")  # whole_step reuses candidates: an exact cutoff
        self.assertIn("an exact cutoff", no_cutoff.reason)
        # A cutoff exact at rc only (the builders' list is re-filtered exactly): whole_step and topology reuse,
        # with padded rows at the cutoff; far self-loops (beyond the cutoff) are rejected for such a model.
        at_cutoff = CaptureContract(sync_free=True, exact_cutoff=False, exact_at_cutoff=True)
        at_rc = PaddingRule("exact_rc_loops")
        for entry, decision, chosen in (("gpu_md", "execution", "whole_step"),
                                        ("calculator", "topology_reuse", "exact_skin_refilter")):
            record = self.negotiate(fake_adapter(contract=at_cutoff, padding=at_rc), entry=entry, cuda=True,
                                    memory=measured(), capacity=capacity, budget=budget)
            self.assertEqual(record.decision(decision).chosen, chosen)
            rejected = fake_adapter(contract=at_cutoff, kernels=[KernelChoice("conv", ("a", "b"), reference="b")])
            with self.assertRaisesRegex(NegotiationError, "exact_rc_loops"):
                self.negotiate(rejected, entry=entry, cuda=True)
            self.assertEqual(rejected.selected, [])  # rejected before any kernel is installed
        not_periodic = self.negotiate(fake_adapter(contract=self.CONTRACT), entry="calculator", cuda=True,
                                      pbc=(True, True, False), memory=measured(), capacity=capacity,
                                      budget=budget).decision("topology_reuse")
        self.assertEqual((not_periodic.chosen, not_periodic.reason), ("rebuild", "not periodic in all directions"))

    def test_graph_source_padding_and_derivative(self):
        """The neighbor list (the builder of the path: the fixed-capacity builder where candidates are
        reused, else the entry point's rebuilding builder; the host without CUDA), the padding (exact
        shapes for eager; on a padded path the adapter's declared rule) and the recorded derivative."""
        capacity, budget = {"edges": 30_000}, 64 * 2**30
        record = self.negotiate(fake_adapter(contract=self.CONTRACT), cuda=True, memory=measured(),
                                capacity=capacity, budget=budget)
        self.assertEqual(record.decision("neighbor_list").chosen, "fixed_capacity_builder")
        self.assertIn("edges only: candidates within cutoff + skin", record.decision("neighbor_list").reason)
        self.assertEqual(record.decision("padding").chosen, "far_self_loops")
        unavailable = self.negotiate(fake_adapter(contract=self.CONTRACT), cuda=True, builder=False)
        self.assertEqual(unavailable.decision("neighbor_list").chosen, "tensor_graph_builder")
        self.assertIn("rebuilt every evaluation", unavailable.decision("neighbor_list").reason)
        self.assertEqual(unavailable.decision("neighbor_list").chain,
                         ("fixed_capacity_builder", "tensor_graph_builder", "host"))
        for reuse, chosen in (("skin", ("fixed_capacity_builder", "far_self_loops")), ("auto", ("gpu_graph_builder", "none"))):
            with self.subTest(calculator_eager=reuse):
                plan = self.negotiate(fake_adapter(contract=self.CONTRACT), InferenceConfig(
                    execution="eager", topology_reuse=reuse, report=False), entry="calculator", cuda=True)
                self.assertEqual((plan.decision("neighbor_list").chosen, plan.decision("padding").chosen), chosen)
        declared = fake_adapter(contract=self.CONTRACT, padding=PaddingRule("exact_rc_loops", pad_vector=(5.0, 0, 0)))
        for execution, padding in (("auto", "exact_rc_loops"), ("eager", "none")):
            with self.subTest(execution=execution):
                plan = self.negotiate(declared, InferenceConfig(execution=execution, report=False), cuda=True,
                                      memory=measured(), capacity=capacity, budget=budget)
                self.assertEqual(plan.decision("padding").chosen, padding)
        for force_mode, unit in (("model_owned", "GPa"), ("engine_autograd", "eV/A^3"), ("direct_head", "eV/A^3")):
            with self.subTest(force_mode=force_mode):
                contract = CaptureContract(sync_free=True, exact_cutoff=True, force_mode=force_mode, stress_unit=unit)
                derivative = self.negotiate(fake_adapter(contract=contract)).decision("derivative")
                self.assertEqual((derivative.chosen, derivative.evidence["stress_unit"]), (force_mode, unit))
                self.assertNotIn("not declared", derivative.reason)

    def test_kernel_choices_fall_back_and_never_default_to_reduced(self):
        site = KernelChoice("conv", ("fast", "tf32", "x3", "ref"), reference="ref",
                            accuracy={"tf32": "reduced", "x3": "fp32_equivalent"})

        def unavailable(*names):
            def select(site_name, option):
                if option in names:
                    raise ImportError(f"{option} is not installed")
            return select
        for profile, fails, chosen, cls in ((None, (), "fast", "fp32_noise"),
                                            (None, ("fast",), "x3", "fp32_equivalent"),
                                            (None, ("fast", "x3"), "ref", "fp32_noise"),
                                            ("generic", (), "ref", "fp32_noise")):
            with self.subTest(profile=profile, fails=fails):
                adapter = fake_adapter(contract=self.CONTRACT, kernels=[site], on_select=unavailable(*fails))
                record = self.negotiate(adapter, InferenceConfig(optimization_profile=profile))
                decision = record.decision("kernels.conv")
                self.assertEqual((decision.chosen, record.accuracy_class), (chosen, cls))
                self.assertNotIn(("conv", "tf32"), adapter.selected)  # reduced is never a default
                self.assertEqual(sorted(decision.evidence.get("failed", {})), sorted(fails))
                if profile is None:
                    self.assertIn("tf32 skipped", decision.reason)
        with self.assertRaisesRegex(NegotiationError, "reference 'ref' failed"):
            self.negotiate(fake_adapter(kernels=[site], on_select=unavailable("fast", "x3", "ref")))
        # A model whose own class is "reduced": its options of that class are its defaults.
        reduced = CaptureContract(sync_free=True, exact_cutoff=True, accuracy_class="reduced")
        plan = self.negotiate(fake_adapter(contract=reduced, kernels=[KernelChoice("conv", ("fast", "ref"), "ref")]))
        self.assertEqual(plan.decision("kernels.conv").chosen, "fast")
        with self.assertRaisesRegex(ValueError, "reference"):
            KernelChoice("conv", ("a",), reference="b")
        with self.assertRaisesRegex(ValueError, "cannot be 'reduced'"):
            KernelChoice("conv", ("a", "b"), reference="b", accuracy={"b": "reduced"})
        with self.assertRaisesRegex(ValueError, "by_execution"):
            KernelChoice("conv", ("a", "b"), reference="b", by_execution={"eager": ("a",)})
        # Options ranked per execution mode: installed for the requested mode (auto: the mode it aims for,
        # until the entry point decides with a measured memory model) and following the decided mode.
        ranked = KernelChoice("conv", ("graph", "eager", "ref"), reference="ref",
                              by_execution={"eager": ("eager", "graph", "ref")}, accuracy={"eager": "fp32_equivalent"})
        capacity, budget = {"edges": 30_000}, 64 * 2**30
        for name, config, kwargs, execution, chosen in (
                ("eager request", InferenceConfig(execution="eager"), {}, "eager", "eager"),
                ("whole_step request", InferenceConfig(execution="whole_step"), {}, "whole_step", "graph"),
                ("auto, memory pending", None, {}, "eager", "graph"),
                ("auto, decided", None, {"memory": measured(), "capacity": capacity, "budget": budget,
                                         "whole_step_trial": lambda: 2.0}, "whole_step", "graph"),
                ("auto, too little memory", None, {"memory": measured(), "capacity": capacity, "budget": 2**20},
                 "eager", "eager"),
                ("auto, no warmup", InferenceConfig(capacity=CapacityConfig(warmup=0)), {}, "eager", "eager")):
            with self.subTest(name):
                adapter = fake_adapter(contract=self.CONTRACT, kernels=[ranked])
                record = self.negotiate(adapter, config, cuda=True, **kwargs)
                decision = record.decision("kernels.conv")
                self.assertEqual((record.decision("execution").chosen, decision.chosen), (execution, chosen))
                self.assertEqual(adapter.selected[-1], ("conv", chosen))
                self.assertEqual(decision.chain, ranked.order(decision.evidence["execution"]))
        adapter = fake_adapter(contract=self.CONTRACT, kernels=[ranked])
        pending = self.negotiate(adapter, None, cuda=True)
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch("fastmd.graph.fixed_capacity_builder._fixed_builder_available", True):
            from fastmd.adapters.negotiate import renegotiate_execution
            from fastmd.config import resolve_config
            config = resolve_config(None, default_execution="auto", default_isolated_atoms="reference")
            eager = renegotiate_execution(pending, adapter, config, entry="gpu_md", device="cuda", capacity=capacity,
                                          memory=measured(), budget=2**20)
        self.assertEqual((eager.decision("execution").chosen, eager.decision("kernels.conv").chosen), ("eager", "eager"))
        self.assertIn("under eager execution", eager.decision("kernels.conv").reason)
        self.assertEqual(pending.decision("kernels.conv").chosen, "graph")  # the record it was copied from stays
        # The contract is read again after a per-mode kernel installation (an installed kernel may change what the
        # model declares, e.g. a compiled backbone): the record's contract and derivative are the installed
        # kernel's, and a contract that no longer allows the decided mode leaves it.
        def kernel_contract(adapter, **by_option):
            type(adapter).capture_contract = lambda self: by_option.get(
                self.selected[-1][1] if self.selected else None, self.CONTRACT_DEFAULT)
            type(adapter).CONTRACT_DEFAULT = self.CONTRACT
            return adapter
        compiled = CaptureContract(sync_free=True, exact_cutoff=True, compiled="capture_safe", force_mode="model_owned",
                                   stress_unit="GPa")
        unsafe = CaptureContract(sync_free=False, exact_cutoff=True)
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch("fastmd.graph.fixed_capacity_builder._fixed_builder_available", True):
            from fastmd.adapters.negotiate import renegotiate_execution
            from fastmd.config import resolve_config
            config = resolve_config(None, default_execution="auto", default_isolated_atoms="reference")
            adapter = kernel_contract(fake_adapter(contract=self.CONTRACT, kernels=[ranked]), eager=compiled)
            plan = renegotiate_execution(negotiate(adapter, None, device="cuda"), adapter, config, entry="gpu_md",
                                         device="cuda", capacity=capacity, memory=measured(), budget=2**20)
            self.assertEqual((plan.decision("kernels.conv").chosen, plan.contract), ("eager", compiled))
            self.assertEqual(plan.decision("derivative").chosen, "model_owned")
            adapter = kernel_contract(fake_adapter(contract=self.CONTRACT, kernels=[ranked]), graph=unsafe)
            for record in (negotiate(adapter, InferenceConfig(execution="eager"), device="cuda"),
                           negotiate(adapter, None, device="cuda", capacity=capacity, memory=measured(),
                                     budget=budget, whole_step_trial=lambda: 2.0)):
                # auto aims at whole_step; the graph kernel's contract is not sync-free: eager, its kernel kept
                execution = record.decision("execution")
                self.assertEqual((execution.chosen, record.contract.sync_free), ("eager", True), execution.reason)
            # A kernel that compiles dynamically (the adapter's compiled flag, its contract unchanged): installed
            # for model_graph, it makes the decided mode fall back to eager.
            dynamic = KernelChoice("conv", ("eager", "dyn", "ref"), reference="ref",
                                   by_execution={"model_graph": ("dyn", "eager", "ref")})
            adapter = fake_adapter(contract=self.CONTRACT, kernels=[dynamic],
                                   on_select=lambda site, option: setattr(adapter, "compiled", option == "dyn"))
            plan = negotiate(adapter, None, device="cuda", pbc=(True, True, False), capacity=capacity,
                             memory=measured(), budget=budget)
            self.assertEqual((plan.decision("kernels.conv").chosen, plan.decision("execution").chosen), ("dyn", "eager"))
            self.assertIn("dynamic torch.compile", plan.decision("execution").reason)
        # Another kernel: the plan's identity follows (fingerprint, accuracy class), as a direct eager plan's.
        direct = self.negotiate(fake_adapter(contract=self.CONTRACT, kernels=[ranked]), InferenceConfig(execution="eager"),
                                cuda=True)
        self.assertEqual((eager.accuracy_class, pending.accuracy_class), ("fp32_equivalent", "fp32_noise"))
        self.assertEqual(eager.fingerprint, direct.fingerprint)
        self.assertNotEqual(eager.fingerprint, pending.fingerprint)
        # The fingerprint covers the kernel choices, static system inputs and weights.
        prints = {self.negotiate(fake_adapter(kernels=[site], on_select=unavailable(*fails), system=system)).fingerprint
                  for fails, system in (((), None), (("fast",), None), ((), {"head": "default"}),
                                        ((), {"charge": float("nan")}), ((), {"charge": float("inf")}),
                                        ((), {"spin": np.arange(2000.0)}), ((), {"spin": np.arange(2000.0) + 1}))}
        self.assertEqual(len(prints), 7)
        with self.assertRaisesRegex(TypeError, "static system input"):
            self.negotiate(fake_adapter(system={"head": object()}))

    def test_accuracy_class_rule(self):
        self.assertEqual(trial_candidates([("whole_step", "fp32_noise"), ("x3", "fp32_equivalent"),
                                           ("model_graph", "fp32_noise"), ("tf32", "reduced")]),
                         ["whole_step", "model_graph"])  # the default's class only: never into "reduced"
        self.assertEqual(loosest_class("bitwise", None, "fp32_equivalent", "fp32_noise"), "fp32_equivalent")
        self.assertEqual(loosest_class(), "bitwise")
        for bad in ({"accuracy_class": "tf32"}, {"noise_floor_f": float("inf")}, {"tol_multiplier": float("nan")}):
            with self.assertRaises(ValueError):
                CaptureContract(sync_free=True, exact_cutoff=True, **bad)
        # A model other than MatRIS runs IEEE FP32 under a TF32 process default; TF32 only on request.
        tf32 = {"fp32_gemm": "tf32", "float32_matmul_precision": "high",
                "environment": {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}, "triton_dot": {"requested": "auto"}}
        with patch("fastmd.adapters.negotiate.measured_precision", return_value=tf32):
            default = self.negotiate(fake_adapter(contract=self.CONTRACT), cuda=True)
            asked = self.negotiate(fake_adapter(contract=self.CONTRACT), InferenceConfig(matmul_precision="high"),
                                   cuda=True)
            fp64 = self.negotiate(fake_adapter(contract=self.CONTRACT, model_dtype="float64"), cuda=True)
        self.assertEqual((default.decision("precision").chosen, default.matmul_precision, default.accuracy_class),
                         ("ieee", "highest", "fp32_noise"))
        self.assertEqual((asked.decision("precision").chosen, asked.accuracy_class), ("tf32", "reduced"))
        # MatRIS's Triton dot sites do not count for another model.
        dots = {**tf32, "fp32_gemm": "ieee", "triton_dot": {"requested": "tf32", "wide_gemm": "tf32"}}
        with patch("fastmd.adapters.negotiate.measured_precision", return_value=dots):
            self.assertEqual(self.negotiate(fake_adapter(contract=self.CONTRACT), cuda=True)
                             .decision("precision").chosen, "ieee")
        self.assertEqual((fp64.decision("precision").chosen, fp64.precision["chosen"]), ("fp64", "fp64"))
        self.assertIn("3xTF32 and TF32 do not apply", fp64.decision("precision").reason)

    def test_dry_run_notes_where_the_entry_point_measures(self):
        """dry_run() of a model that lacks only a memory bound reports eager, and notes that the entry point
        measures the model where it would: an edges-only model, or one with a line graph and an energy tap,
        with the fixed-capacity builder, and in the calculator only for a model that is not compiled."""
        capture_safe = CaptureContract(sync_free=True, exact_cutoff=True, compiled="capture_safe")
        for entry, contract, compiled, builder, noted, line_graph in (
                ("gpu_md", self.CONTRACT, False, True, True, None),
                ("calculator", self.CONTRACT, False, True, True, None),
                ("gpu_md", self.CONTRACT, False, False, False, None), ("gpu_md", capture_safe, True, True, True, None),
                ("calculator", capture_safe, True, True, False, None),
                ("gpu_md", self.CONTRACT, False, True, True, "tap"),
                ("calculator", self.CONTRACT, False, True, True, "tap"),
                ("gpu_md", self.CONTRACT, False, True, False, "no tap")):
            with self.subTest(entry=entry, compiled=compiled, builder=builder, line_graph=line_graph), \
                    patch.object(torch.cuda, "is_available", return_value=True), \
                    patch("fastmd.graph.fixed_capacity_builder._fixed_builder_available", builder):
                adapter = fake_adapter(contract=contract, compiled=compiled, tap=line_graph == "tap",
                                       triplet_cutoff=None if line_graph is None else 3.0)
                execution = dry_run(adapter, None, entry=entry, device="cuda", atoms=atoms()).decision("execution")
                self.assertEqual(execution.chosen, "eager")
                self.assertIn("no memory bound", execution.reason)
                self.assertEqual("the entry point measures the model" in execution.reason, noted)

    def test_multi_gpu_and_legacy_profile_errors(self):
        with self.assertRaisesRegex(NegotiationError, "one device per simulation"):
            self.negotiate(fake_adapter(), world_size=2)
        with self.assertRaisesRegex(NegotiationError, "one device per simulation"):
            self.negotiate(fake_adapter(exchange="node_halo"), world_size=2)
        with self.assertRaisesRegex(NegotiationError, "'legacy' is MatRIS-only"):
            self.negotiate(fake_adapter(), InferenceConfig(optimization_profile="legacy"))
        with self.assertRaisesRegex(NegotiationError, "'legacy' is MatRIS-only"):
            load_adapter(fake_adapter(), optimization_profile="legacy")
        model = make_model()
        self.assertIs(load_adapter(model, optimization_profile="legacy", model_dtype="float32").module, model)
        with self.assertRaisesRegex(ValueError, "MatRIS runs float32"):
            load_adapter(model, model_dtype="float64")
        with self.assertRaisesRegex(TypeError, "MatRIS is not negotiated"):
            negotiate(load_adapter(model), None)
        # A model the runtime plans by rules it does not have is an error, not MatRIS's path. The rules are read
        # from the adapter's class: None declares none, and a mock spec'd as MatRIS's adapter has MatRIS's.
        unknown = fake_adapter()
        type(unknown).runtime_rules = "other"
        with self.assertRaisesRegex(NegotiationError, "runtime rules 'other'"):
            dry_run(unknown, None, entry="gpu_md", device="cpu", atoms=atoms())
        type(unknown).runtime_rules = None
        self.assertTrue(negotiated(unknown))
        self.assertFalse(negotiated(Mock(spec=MatRISAdapter)))
        # Each set of rules the runtime has is recorded by the module the registry names for it; MatRIS's public rule
        # functions are still read from the negotiation.
        from fastmd.adapters import negotiate as negotiation, protocol
        self.assertEqual({rules for rules, _ in fastmd.models.PLAN_RECORDERS}, set(protocol.RUNTIME_RULES))
        for name in ("matris_execution_rule", "matris_recompute_decision"):
            self.assertIs(getattr(negotiation, name), getattr(fastmd.models.plan_recorder("matris"), name))
        with self.assertRaisesRegex(ValueError, "one device per simulation"):
            dry_run(load_adapter(model), None, entry="gpu_md", device="cpu", atoms=atoms(), world_size=2)

    def test_forward_scope_restores_precision_changed_by_upstream_code(self):
        def upstream_select(site, option):
            torch.set_float32_matmul_precision("medium")  # e.g. a model that sets TF32 when it loads
        adapter = fake_adapter(contract=self.CONTRACT, kernels=[KernelChoice("conv", ("a",), reference="a")],
                               on_select=upstream_select)
        before = (torch.get_float32_matmul_precision(), fp32_gemm_mode())
        record = self.negotiate(adapter)
        self.assertEqual((torch.get_float32_matmul_precision(), fp32_gemm_mode()), before)
        self.assertEqual(len(record.precision_restores), 1)
        self.assertEqual(record.precision_restores[0]["inside"]["fp32_gemm"], "tf32")
        for _ in range(3):  # a forward that sets the precision every call: one record, counted
            with record.forward_scope():
                torch.set_float32_matmul_precision("medium")
        self.assertEqual((torch.get_float32_matmul_precision(), fp32_gemm_mode()), before)
        self.assertEqual([change["count"] for change in record.precision_restores], [4])
        with record.forward_scope():
            pass  # no change, nothing recorded
        self.assertEqual([change["count"] for change in record.precision_restores], [4])
        self.assertTrue(record.check_precision("before_capture")["as_negotiated"])

    def test_measured_memory_model_fit(self):
        """Linear peaks within the residual bound (max(4 MiB, 2%)): valid, and a held-out capacity is
        predicted within it; too few points, a negative slope or nonlinear peaks: invalid (no
        prediction, so execution auto keeps the safe side)."""
        noise = (1.5 * 2**20, -1.0 * 2**20, 0.5 * 2**20)
        model = MeasuredMemoryModel(memory_points("capture", (10_000, 20_000, 40_000), noise=noise), triplets=False)
        self.assertTrue(model.valid, model.reason)
        held_out = memory_points("capture", (30_000,))[0].peak_bytes
        fit = model.fits["capture"]
        self.assertLessEqual(abs(model.predict("capture", 30_000) - held_out), fit.residual_bound)
        self.assertGreaterEqual(model.bound("capture", 30_000), model.predict("capture", 30_000))
        self.assertAlmostEqual(fit.b, 2048.0, delta=100.0)
        self.assertIsNone(model.predict("capture", 81_000))  # beyond 2x the largest measured capacity
        self.assertLessEqual(model.bound("capture", model.max_edges("capture", 2**30)), 2**30)
        with_triplets = MeasuredMemoryModel(
            memory_points("capture", (1_000, 2_000, 4_000, 3_000), triplets=(10_000, 15_000, 50_000, 80_000)),
            triplets=True)
        self.assertTrue(with_triplets.valid, with_triplets.reason)
        self.assertIsNone(with_triplets.predict("capture", 4_000, 200_000))  # beyond 2x the measured triplets
        proportional = MeasuredMemoryModel(memory_points("capture", (1_000, 2_000, 4_000, 3_000),
                                                         triplets=(4_000, 8_000, 16_000, 12_000)), triplets=True)
        self.assertFalse(proportional.valid)
        self.assertIn("cannot be told apart", proportional.reason)
        self.assertIsNone(with_triplets.predict("capture", 4_000, 4_000))  # a ratio outside the measured ones
        no_triplets = MeasuredMemoryModel(memory_points("capture", (1_000, 2_000, 4_000, 3_000),
                                                        triplets=(0, 8_000, 4_000, 12_000)), triplets=True)
        self.assertTrue(no_triplets.valid, no_triplets.reason)  # a point without triplets separates the costs
        self.assertEqual(model.max_edges("capture", 2**20), 0)  # below the headroom nothing fits
        with self.assertRaisesRegex(ValueError, "no 'eager' points"):
            model.headroom("eager")
        for name, points, reason in (
                ("two points", memory_points("capture", (10_000, 20_000)), "at least 3 needed"),
                ("negative slope", memory_points("capture", (10_000, 20_000, 40_000), per_edge=-512.0),
                 "negative bytes per edge"),
                ("nonlinear", [MemoryPoint("capture", e, 0, int(1e-1 * e * e))
                               for e in (10_000, 20_000, 40_000, 80_000)], "not linear"),
                ("eager only", memory_points("eager", (10_000, 20_000, 40_000)), "no capture points")):
            with self.subTest(name):
                bad = MeasuredMemoryModel(points, triplets=False)
                self.assertFalse(bad.valid)
                self.assertIn(reason, bad.reason)
                self.assertIsNone(bad.bound("capture", 20_000))
        # An invalid fit stays valid JSON (no NaN or Infinity) in a plan record.
        record = self.negotiate(fake_adapter(contract=self.CONTRACT), cuda=True, memory=measured(valid=False),
                                capacity={"edges": 20_000}, budget=2**36)
        json.loads(record.to_json(), parse_constant=lambda name: self.fail(f"non-finite {name} in the record"))

    def test_size_unions_bounds_a_union_at_its_own_capacity(self):
        """size_unions (the replica entry point's union size) on a cell of 22 pairs per replica, unions measured
        at 2, 4 and 6 replicas: a union is predicted at its own edge capacity (8 replicas: 448 rows, within the
        measured range, where 8 times one replica's rounded capacity is not), kept when its capture bound plus
        the other replicas' buffers outside the shared pool fits 85 % of the budget, else halved until it does
        (a union beyond the measured range too); those buffers count when the halving, not the request, leaves
        other unions; with a line graph at its own triplet capacity as well (a fit with a per-triplet cost, over
        the entry point's probe points: valid, and predicting every halved union); no valid capture fit or no
        model: kept, not bounded; one replica that does not fit: NegotiationError."""
        from fastmd.adapters.negotiate import size_unions

        record = self.negotiate(fake_adapter(contract=self.CONTRACT))
        edges = lambda k: 64 * (-(-(22 * k + 32) // 32))  # tiers of 32 pairs, 32 of padding: 2 rows per pair
        memory = MeasuredMemoryModel(memory_points("capture", [edges(k) for k in (2, 4, 6)]), triplets=False)
        self.assertGreater(8 * edges(1), 2 * edges(6))  # one replica's capacity times 8: beyond the range
        mib = 2**20
        cases = [  # (requested, replicas, persistent bytes per replica, budget, chosen or an error, reason)
            (8, 8, 0, 2**40, 8, "measured need"),
            (8, 8, 0, int(66.7 * mib / 0.85), 4, "8 replicas per union need"),  # 8: 66.875 MiB, 4: 66.5 MiB
            (8, 8, 2 * mib, int(66.7 * mib / 0.85), NegotiationError, "do not fit"),  # 4: 66.5 + 4 x 2 MiB
            (20, 20, 0, 2**40, 10, "beyond the measured range"),
        ]
        for requested, replicas, persistent, budget, expected, reason in cases:
            with self.subTest(requested=requested, persistent=persistent, budget=budget):
                arguments = dict(requested=requested, memory=memory, union_edges=edges, device="cpu",
                                 replicas=replicas, persistent_per_replica=persistent, budget=budget)
                if expected is NegotiationError:
                    with self.assertRaisesRegex(NegotiationError, reason):
                        size_unions(record, **arguments)
                    continue
                chosen, plan = size_unions(record, **arguments)
                decision = plan.decision("union_size")
                self.assertEqual((chosen, decision.chosen), (expected, str(expected)))
                self.assertIn(reason, decision.reason)
                self.assertEqual(decision.evidence["union_edges"], edges(requested))
                if chosen != requested:
                    self.assertEqual(decision.evidence["fitted_union_edges"], edges(chosen))
        # A line graph: the fit has a per-triplet cost, and a union is predicted at its own triplet capacity
        # too (4 KiB per triplet: 8 replicas fit without their triplets, not with them).
        triplets = lambda k: 128 * (-(-(40 * k + 64) // 128))
        probes = [(2, triplets(2)), (4, 2), (4, edges(4)), (6, triplets(6))]
        line_graph = MeasuredMemoryModel([MemoryPoint("capture", edges(k), t, 64 * mib + 2048 * edges(k) + 4096 * t)
                                          for k, t in probes], triplets=True)
        need = line_graph.bound("capture", edges(8), triplets(8))
        self.assertGreater(need, line_graph.bound("capture", edges(8), 0) or 0)
        budget = int((need - 4096) / 0.85)  # just below the need of 8 with their triplets
        chosen, plan = size_unions(record, requested=8, memory=line_graph, union_edges=edges, device="cpu",
                                   replicas=8, budget=budget, union_triplets=triplets)
        decision = plan.decision("union_size")
        self.assertEqual(chosen, 4)
        self.assertEqual((decision.evidence["union_triplets"], decision.evidence["fitted_union_triplets"]),
                         (triplets(8), triplets(4)))
        # The replica entry point's probes of a line graph on a cell without triplets at the start (1,026 pairs
        # per replica; each replica holds the minimum triplet padding): a fit over them (peaks linear in the
        # edges and triplets) is valid and predicts every union the halving may choose.
        from fastmd.adapters.protocol import CapacityStep
        from fastmd.execution.replica_batch import union_capacity, union_probe_capacities, union_triplet_capacity

        step = CapacityStep(u_step=128, min_pad_u=128, t_step=128, min_pad_t=128)
        line_edges = lambda k: 2 * union_capacity(1026 * k, step, headroom=0.1, line_graph=True)
        line_triplets = lambda k: union_triplet_capacity(0, step, headroom=0.25, replicas=k)
        for replicas in (4, 16, 256):
            probes = union_probe_capacities(replicas, line_edges, line_triplets)
            fit = MeasuredMemoryModel([MemoryPoint("capture", line_edges(k), t, 64 * mib + 16384 * line_edges(k)
                                                   + 15360 * t) for k, t in probes], triplets=True)
            self.assertTrue(fit.valid, fit.reason)
            size = replicas
            while True:
                self.assertIsNotNone(fit.bound("capture", line_edges(size), line_triplets(size)), (replicas, size))
                if size == 1:
                    break
                size //= 2
        for name, model, reason in (("no model", None, "too small to measure"),
                                    ("failed measurement", MeasuredMemoryModel.failed("out of memory", triplets=False),
                                     "not bounded")):
            with self.subTest(name):
                chosen, plan = size_unions(record, requested=8, memory=model, union_edges=edges, device="cpu",
                                           replicas=8, budget=1)
                self.assertEqual(chosen, 8)
                self.assertIn(reason, plan.decision("union_size").reason)

    def test_plan_record_json_and_op_plan_link(self):
        record = self.negotiate(fake_adapter(contract=self.CONTRACT))
        data = json.loads(record.to_json())
        self.assertEqual(data["schema"], "fastmd.engine_plan/1")
        self.assertEqual([d["id"] for d in data["decisions"]],
                         ["neighbor_list", "topology_reuse", "padding", "kernels", "derivative", "execution",
                          "precision", "recompute", "multi_gpu"])
        for key in ("model", "upstream", "fingerprint", "profile", "patches", "accuracy_class", "contract"):
            self.assertIn(key, data)
        self.assertTrue(data["fingerprint"].startswith("sha256:"))
        from fastmd.engine.recorder import OpPlanRecorder, canonical_json
        module = torch.nn.Linear(3, 2)
        with OpPlanRecorder(root=module) as recorder:
            module(torch.ones(1, 3))
        snapshot = recorder.snapshot()
        linked = record.link_op_plan(recorder)
        self.assertIsNone(record.op_plan)  # linked on a deep copy
        linked.check_precision("before_capture")
        self.assertNotIn("before_capture", record.precision)
        self.assertEqual(linked.op_plan["sha256"], hashlib.sha256(canonical_json(snapshot).encode()).hexdigest())
        self.assertIn("aten::linear", json.dumps(snapshot))
        self.assertIn("kernels", str(record))

    def test_patch_records_apply_decline_revert_and_plan(self):
        """Upstream patch records (fastmd.adapters.patches): applied only on a validated version and the
        pinned source, else declined with the reason and upstream left as it is; aliases replaced with the
        target; a process-wide installation shared and reverted by its last owner; a per-instance patch
        leaves the class; pins install nothing; the plan lists applied and declined records and its
        fingerprint covers the applied ones."""
        from dataclasses import replace as replace_record
        import importlib.metadata
        from fastmd.adapters.patches import PatchRecord, apply_patches, revert_patches, source_sha256
        version = importlib.metadata.version("numpy")
        with tempfile.TemporaryDirectory() as root:
            Path(root, "upstream_fake_mod.py").write_text(
                "def scale(x):\n    return 2 * x\n\n\nclass Layer:\n    def __init__(self):\n        self.table = 1\n\n"
                "    def forward(self, x):\n        return x + self.table\n\n"
                "    @staticmethod\n    def helper(x):\n        return x\n")
            Path(root, "upstream_fake_user.py").write_text("from upstream_fake_mod import scale\n")
            sys.path.insert(0, root)
            made = []  # every outcome, reverted at the end whatever happens
            try:
                import upstream_fake_mod as mod
                import upstream_fake_user as user
                original = mod.scale
                scale = PatchRecord("fake.scale", "upstream_fake_mod.scale", "numpy", (version,),
                                    source_sha256("upstream_fake_mod.scale"), "test: a replaced function",
                                    replacement=lambda x: 3 * x, bindings=("upstream_fake_user.scale",))
                for name, record, reason in (
                        ("version", replace_record(scale, versions=("0.0.0",)), "is not a validated version"),
                        ("source", replace_record(scale, source_sha256="0" * 64), "changed"),
                        ("distribution", replace_record(scale, distribution="no-such-distribution"),
                         "is not installed"),
                        ("module", replace_record(scale, target="upstream_fake_missing.scale"), "cannot read")):
                    with self.subTest(declined=name):
                        (outcome,) = apply_patches([record])
                        self.assertFalse(outcome.applied)
                        self.assertIn(reason, outcome.reason)
                        self.assertIs(mod.scale, original)
                        self.assertIs(user.scale, original)
                        revert_patches([outcome])  # a declined outcome reverts nothing
                user.scale = lambda x: x  # an alias re-bound elsewhere: not the target's object
                (outcome,) = apply_patches([scale])
                self.assertFalse(outcome.applied)
                self.assertIn("not the target's object", outcome.reason)
                self.assertIs(mod.scale, original)
                user.scale = original
                first, second = apply_patches([scale]), apply_patches([scale])  # two adapters, one installation
                made += first + second
                self.assertTrue(first[0].applied and second[0].applied)
                (rival,) = apply_patches([replace_record(scale, bindings=())])  # another record under that name
                self.assertFalse(rival.applied)
                self.assertIn("another record is installed", rival.reason)
                self.assertEqual((mod.scale(2), user.scale(2)), (6, 6))
                # An installed name is still checked: the version, and that the target holds the installation.
                with patch("importlib.metadata.version", return_value="0.0.0"):
                    (stale,) = apply_patches([scale])
                self.assertIn("is not a validated version", stale.reason)
                installed, mod.scale = mod.scale, (lambda x: x)
                (overwritten,) = apply_patches([scale])
                mod.scale = installed
                self.assertIn("not the installed replacement", overwritten.reason)
                (elsewhere,) = apply_patches([replace_record(scale, target="upstream_fake_mod.Layer.forward",
                                                             bindings=())])
                self.assertIn("another record is installed", elsewhere.reason)
                self.assertFalse(stale.applied or overwritten.applied or elsewhere.applied)
                # A static method replacement, applied by two adapters: one shared installation.
                helper = PatchRecord("fake.helper", "upstream_fake_mod.Layer.helper", "numpy", (version,),
                                     source_sha256("upstream_fake_mod.Layer.helper"), "test: a static method",
                                     replacement=staticmethod(lambda x: x + 9))
                shared = apply_patches([helper]) + apply_patches([helper])
                made += shared
                self.assertEqual([outcome.applied for outcome in shared], [True, True])
                self.assertEqual(mod.Layer.helper(1), 10)
                revert_patches(shared)
                self.assertEqual(mod.Layer.helper(1), 1)
                revert_patches(first)
                revert_patches(first)  # idempotent
                self.assertEqual(mod.scale(2), 6)  # still installed for the second
                revert_patches(second)
                self.assertIs(mod.scale, original)
                self.assertIs(user.scale, original)
                forward = source_sha256("upstream_fake_mod.Layer.forward")
                layer, other = mod.Layer(), mod.Layer()
                per_instance = PatchRecord("fake.forward", "upstream_fake_mod.Layer.forward", "numpy", (version,),
                                           forward, "test: one instance", replacement=lambda x: x + 2,
                                           per_instance=True)
                pin = PatchRecord("fake.forward_pin", "upstream_fake_mod.Layer.forward", "numpy", (version,),
                                  forward, "test: relied on unpatched")
                with self.assertRaises(ValueError):
                    apply_patches([per_instance])  # per instance: instance= is required
                outcomes = apply_patches([per_instance, pin, replace_record(scale, versions=("0.0.0",))],
                                         instance=layer)
                made += outcomes
                self.assertEqual([outcome.applied for outcome in outcomes], [True, True, False])
                self.assertEqual((layer.forward(1), other.forward(1)), (3, 2))
                self.assertNotIn("forward_pin", vars(layer))
                # An attribute only instances have, pinned by the source of the code that reads it.
                table = PatchRecord("fake.table", "upstream_fake_mod.Layer.table", "numpy", (version,), forward,
                                    "test: an instance attribute", replacement=5, per_instance=True,
                                    source="upstream_fake_mod.Layer.forward")
                (shadow,) = apply_patches([table], instance=other)
                made.append(shadow)
                self.assertTrue(shadow.applied)
                self.assertEqual(other.forward(1), 6)
                # Two adapters of one instance share its installation, reverted by the last; another record
                # under that name on the instance is declined.
                (twin,) = apply_patches([table], instance=other)
                made.append(twin)
                (rival,) = apply_patches([replace_record(table, replacement=7)], instance=other)
                self.assertEqual((twin.applied, rival.applied), (True, False))
                self.assertIn("another record is installed", rival.reason)
                other.table = 8  # set outside the records: a re-application is declined
                (outside,) = apply_patches([table], instance=other)
                self.assertIn("not the installed replacement", outside.reason)
                other.table = 5
                revert_patches([shadow])
                self.assertEqual(other.forward(1), 6)  # still installed for the twin
                revert_patches([twin])
                self.assertEqual(other.table, 1)
                record = self.negotiate(fake_adapter(contract=self.CONTRACT, patches=outcomes))
                self.assertEqual([entry["name"] for entry in record.patches["applied"]],
                                 ["fake.forward", "fake.forward_pin"])
                self.assertEqual([entry["kind"] for entry in record.patches["applied"]], ["per_instance", "pin"])
                self.assertIn("is not a validated version", record.patches["declined"]["fake.scale"])
                json.loads(record.to_json())
                plain = self.negotiate(fake_adapter(contract=self.CONTRACT))
                self.assertEqual(plain.patches, {"applied": [], "declined": {}})
                self.assertEqual(self.negotiate(fake_adapter(contract=self.CONTRACT, patches=[])).fingerprint,
                                 plain.fingerprint)  # no applied records: the fingerprint of a model without
                self.assertNotEqual(record.fingerprint, plain.fingerprint)
                # After a revert (the adapter closed), a renegotiated plan reports it and changes its identity.
                from fastmd.adapters.negotiate import renegotiate_execution
                from fastmd.config import resolve_config
                closing = fake_adapter(contract=self.CONTRACT, patches=outcomes)
                negotiated = self.negotiate(closing)
                revert_patches(outcomes)
                again = renegotiate_execution(negotiated, closing, resolve_config(None, default_execution="auto",
                                                                                  default_isolated_atoms="reference"),
                                              entry="gpu_md", device="cpu")
                self.assertEqual(again.patches["declined"]["fake.forward"], "reverted: the adapter was closed")
                self.assertEqual(again.fingerprint, self.negotiate(closing).fingerprint)
                self.assertNotEqual(again.fingerprint, negotiated.fingerprint)
                self.assertEqual(layer.forward(1), 2)
                self.assertNotIn("forward", vars(layer))
            finally:
                revert_patches(made)
                sys.path.remove(root)
                for name in ("upstream_fake_mod", "upstream_fake_user"):
                    sys.modules.pop(name, None)

    def test_matris_plan_records_the_runtime_decisions(self):
        """MatRIS is recorded: plan_record() of the calculator and GPU-resident MD reads the execution,
        recompute and topology-reuse decisions they made (CPU here; CUDA in test_config_cuda), and the
        dry run (python -m fastmd.explain) gives the same plan."""
        model = make_model()
        calc = MLIPCalculator(model=model, device="cpu")
        record = calc.plan_record()
        summary = calc.config_summary()
        self.assertEqual(record.decision("execution").chosen, summary["effective"]["execution"])
        self.assertEqual(record.decision("execution").reason, "auto: CPU or checkpoint (no CUDA graph)")
        self.assertEqual(record.decision("topology_reuse").reason, calc.topology_reuse_summary()["reason"])
        self.assertEqual((record.model, record.profile, record.accuracy_class, record.decision("derivative").chosen,
                          record.decision("derivative").evidence["stress_unit"]),
                         ("matris", "topology", "bitwise", "model_owned", "GPa"))
        self.assertEqual(record.memory["calibration"], memory_admission.memory_model(model).calibration)
        explicit = MLIPCalculator(model=model, device="cpu", config=InferenceConfig(execution="eager"))
        self.assertEqual(explicit.plan_record().decision("execution").reason, "explicit request")
        self.assertEqual(MLIPCalculator(model=model, device=None).plan_record().decision("precision").chosen,
                         "ieee")  # the device the model is on
        with (patch("fastmd.interfaces.gpu_md.TensorGraphBuilder"),
              patch("fastmd.interfaces.gpu_md.AsyncGPUMDLogger")):
            md = GPUResidentMolecularDynamics(atoms(), model=model, device="cpu")
        record = md.plan_record()
        effective = md.config_summary()["effective"]
        self.assertEqual((record.decision("execution").chosen, record.decision("execution").reason),
                         (effective["execution"], effective["auto_execution"]["reason"]))
        dry = dry_run(load_adapter(model), None, entry="gpu_md", device="cpu", atoms=atoms())
        self.assertEqual([(d.id, d.chosen) for d in dry.decisions], [(d.id, d.chosen) for d in record.decisions])
        self.assertEqual(dry.fingerprint, record.fingerprint)
        # A request the entry point rejects fails the dry run (and a negotiation) with the same error.
        for entry, execution, message in (("calculator", "whole_step", "calculator does not support"),
                                          ("gpu_md", "model_graph", "requires an available CUDA device")):
            with self.subTest(entry=entry, execution=execution):
                config = InferenceConfig(execution=execution, report=False)
                with self.assertRaisesRegex(ValueError, message):
                    dry_run(load_adapter(model), config, entry=entry, device="cpu", atoms=atoms())
                with self.assertRaisesRegex(ValueError, message):
                    self.negotiate(fake_adapter(), config, entry=entry)

    def test_explain_cli_prints_a_dry_run_plan(self):
        from fastmd import explain
        import ase.io
        import io
        from contextlib import redirect_stdout
        model = make_model()
        with tempfile.TemporaryDirectory() as directory:
            path, structure = str(Path(directory) / "model.pth.tar"), str(Path(directory) / "h3.extxyz")
            torch.save({"config": model.config, "state_dict": model.state_dict()}, path)
            ase.io.write(structure, Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.7, 0]], cell=[20] * 3,
                                          pbc=True))
            for entry in ("gpu_md", "calculator"):
                with self.subTest(entry=entry):
                    out = io.StringIO()
                    with redirect_stdout(out):
                        code = explain.main(["--structure", structure, "--model-path", path, "--device", "cpu",
                                             "--entry", entry, "--json",
                                             "--config", '{"saved_tensor_recompute": "on", "report": false}'])
                    self.assertEqual(code, 0)
                    data = json.loads(out.getvalue())
                    decisions = {d["id"]: d for d in data["decisions"]}
                    self.assertEqual((data["model"], data["entry"]), ("matris", entry))
                    self.assertEqual(decisions["execution"]["chosen"], "eager")
                    self.assertEqual(decisions["recompute"]["chosen"], "on")

class TorchSimConfigTest(unittest.TestCase):
    """The optional torch-sim package is stubbed; no install/download required."""
    @classmethod
    def setUpClass(cls):
        names = ("torch_sim", "torch_sim.units", "torch_sim.models", "torch_sim.models.interface", "torch_sim.state")
        modules = {name: ModuleType(name) for name in names}
        modules["torch_sim"].units = modules["torch_sim.units"]
        modules["torch_sim.units"].MetalUnits = SimpleNamespace(pressure=1.0)
        modules["torch_sim.models.interface"].ModelInterface = torch.nn.Module
        modules["torch_sim.state"].SimState = object
        root = Path(__file__).resolve().parents[1]
        path = root / "fastmd/interfaces/torchsim_interface.py"
        # Named in the file's package, so that its relative imports resolve as for the real module.
        package = ".".join(path.relative_to(root).parent.parts)
        spec = importlib.util.spec_from_file_location(f"{package}._runtime_torchsim_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(cls.module)

    def test_models_without_evaluate_rejected_up_front(self):
        """The torch-sim interface calls the model's own evaluate on MatRIS's graph: a model that reads edges
        only or whose energy the engine differentiates is rejected up front (one it loaded by name is closed
        first); so does the batched replica runner, for every model without evaluate."""
        from fastmd.execution.whole_step_cuda_graph import BatchedWholeStepGraphRunner
        tap = fake_adapter(tap=True, triplet_cutoff=3.0)
        with self.assertRaisesRegex(ValueError, "energy the engine differentiates"):
            self.module.MLIPModel(model=tap, device="cpu")
        with self.assertRaisesRegex(ValueError, "reads edges only"):
            self.module.MLIPModel(model=fake_adapter(tap=True), device="cpu")
        loaded = fake_adapter(tap=True)
        loaded.close = Mock()
        with patch.object(self.module, "load_adapter", return_value=loaded), \
                self.assertRaisesRegex(ValueError, "'by_name' reads edges only"):
            self.module.MLIPModel("by_name", device="cpu")
        loaded.close.assert_called_once()
        with self.assertRaisesRegex(ValueError, "model-owned derivative"):
            BatchedWholeStepGraphRunner(model=tap, builders=[object()], integrator=object(), positions=None,
                                        momenta=None)

    def test_unsupported_graph_mode_rejected_before_load(self):
        with patch.object(MatRIS, "load") as load:
            with self.assertRaises(ValueError):
                self.module.MLIPModel(config=InferenceConfig(execution="model_graph", report=False))
        load.assert_not_called()

    def test_removed_checkpoint_argument_is_rejected(self):
        with self.assertRaises(TypeError):
            self.module.MLIPModel(enable_checkpoint=False)
        self.assertFalse(REMOVED_INFERENCE_ARGUMENTS.intersection(inspect.signature(self.module.MLIPModel).parameters))

    def test_compile_default_and_legacy_isolated_default_preserved(self):
        with patch.object(MatRIS, "load", return_value=fake_model()) as load:
            wrapper = self.module.MLIPModel(device="cpu")
        self.assertTrue(load.call_args.kwargs["enable_compile"])
        self.assertFalse(load.call_args.kwargs["enable_checkpoint"])
        self.assertFalse(wrapper.handle_isolated_atoms)
        # Batched SimState inputs have no graph replay: eager, the topology kernels.
        self.assertEqual((wrapper._inference_config.execution, wrapper._inference_config.optimization_profile),
                         ("eager", "topology"))

    def test_checkpoint_auto_updates_passed_model(self):
        model = fake_model()
        wrapper = self.module.MLIPModel(model=model, device="cpu", config=InferenceConfig(checkpoint="auto", report=False))
        self.assertIsNone(wrapper.model.enable_checkpoint)
        self.assertIsNone(wrapper.model.interaction_block[0].enable_checkpoint)

    def test_isolated_policy_reaches_converter(self):
        for policy, reference, strict in (("reference", True, False), ("error", True, True), ("reference", False, True)):
            with self.subTest(policy=policy, reference=reference):
                model = fake_model(reference=reference)
                wrapper = self.module.MLIPModel(model=model, device="cpu", config=InferenceConfig(isolated_atoms=policy, report=False))
                wrapper._state_to_graphs(SimpleNamespace(positions=torch.zeros(1, 3), cell=torch.eye(3),
                                                         to_structures=lambda: [object()]))
                self.assertEqual(model.graph_converter.call_args.kwargs["check_isolated_atoms"], strict)
        # Non-finite coordinates would make every atom isolated (reference energy, zero forces).
        for positions, cell in ((torch.full((1, 3), float("nan")), torch.eye(3)),
                                (torch.zeros(1, 3), torch.eye(3) * float("inf"))):
            with self.subTest(positions=positions.tolist(), cell=cell.tolist()):
                model = fake_model()
                wrapper = self.module.MLIPModel(model=model, device="cpu", config=InferenceConfig(report=False))
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    wrapper._state_to_graphs(SimpleNamespace(positions=positions, cell=cell,
                                                             to_structures=lambda: [object()]))
                model.graph_converter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
