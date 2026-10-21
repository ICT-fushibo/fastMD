"""CPU checks of strict isolated-atom rejection on paths the GPU tests cannot reach,
and of the calculator's rejection of non-finite positions or cell.

The whole-step eager fallback rebuild (candidates still stale after the
recovery replay) is driven with the real runner constructor and mocked model
evaluation; the calculator check uses the CPU graph converter. Replay, recovery
and GPU MD rejection are covered in test_isolated_atom_cuda.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ase import Atoms
from ase.calculators.calculator import all_changes
import numpy as np
import torch

from fastmd.interfaces.calculator import MLIPCalculator
from fastmd.config import InferenceConfig
from fastmd.execution.whole_step_cuda_graph import BatchedWholeStepGraphRunner
from fastmd.execution.whole_step_cuda_graph import WholeStepGraphRunner

from test_isolated_atom_handling import make_model


MODULE = "fastmd.execution.whole_step_cuda_graph"


def _status(*, isolated=0):
    return SimpleNamespace(
        isolated_atom_count=torch.tensor([isolated], dtype=torch.int32),
        overflow=torch.tensor([False]),
        edge_count=torch.tensor([2], dtype=torch.int64),
        triplet_count=torch.tensor([0], dtype=torch.int64),
        invalid_pairs=torch.tensor([False]),
    )


def _make_runner(cls, *, strict):
    """Exercise the real constructor, substituting only CUDA/model work."""
    batched = cls is BatchedWholeStepGraphRunner
    builder = SimpleNamespace(
        n_real=3,
        device=torch.device("cpu"),
        e_capacity=32,
        t_capacity=64,
        refresh_candidates=Mock(),
        assert_candidate_capacity=Mock(),
    )
    shape = (2, 3, 3) if batched else (3, 3)
    positions = torch.zeros(shape)
    status = _status()
    energy = torch.zeros(2 if batched else 1)
    evaluation = (torch.zeros_like(positions), energy, [status, status] if batched else status)
    kwargs = dict(
        model=SimpleNamespace(reference_energy=object()),
        integrator=object(),
        positions=positions,
        momenta=torch.zeros_like(positions),
        validate_isolated_atoms=strict,
    )
    if batched:
        kwargs.update(builders=[builder, builder], parallel_graph_build=False)
    else:
        kwargs["builder"] = builder
    with (
        patch.object(cls, "_evaluate", return_value=evaluation),
        patch.object(cls, "_capture"),
        patch(f"{MODULE}.fusion_scope", side_effect=lambda *args: nullcontext()),
        patch(f"{MODULE}.BatchedCandidateRefresher"),
    ):
        runner = cls(**kwargs)
    runner.graph = Mock()
    return runner


class WholeStepIsolationPolicyTest(unittest.TestCase):
    def test_eager_fallback_obeys_isolation_policy(self):
        for cls in (WholeStepGraphRunner, BatchedWholeStepGraphRunner):
            with self.subTest(cls=cls.__name__):
                runner = _make_runner(cls, strict=True)
                # Candidate staleness persists after recovery replay, forcing eager
                # fallback; that rebuild discovers isolation for the first time.
                runner.window_overflow.fill_(True)
                status = _status(isolated=1)
                runner._step_body = Mock(
                    return_value=[status, status] if hasattr(runner, "builders") else status
                )
                with patch(
                    f"{MODULE}.fusion_scope",
                    side_effect=lambda *args: nullcontext(),
                ):
                    with self.assertRaisesRegex(ValueError, "isolated atom"):
                        runner._fallback()
                runner._step_body.assert_called_once_with(rebuild_candidates=True)


class CalculatorIsolationPolicyTest(unittest.TestCase):
    def test_eager_calculator_rejects_isolation_only_in_strict_mode(self):
        atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [8, 8, 8]], cell=[20] * 3, pbc=True)
        # The CPU converter path: no nvalchemiops graph builder.
        with patch("fastmd.models.matris.converter.op_available", False):
            for policy in ("error", "reference"):
                with self.subTest(isolated_atoms=policy):
                    calculator = MLIPCalculator(
                        model=make_model(),
                        device="cpu",
                        task="ef",
                        config=InferenceConfig(
                            execution="eager", isolated_atoms=policy, report=False
                        ),
                    )
                    if policy == "error":
                        with self.assertRaisesRegex(ValueError, "isolated atom"):
                            calculator.calculate(
                                atoms, properties=["energy", "forces"], system_changes=all_changes
                            )
                    else:
                        calculator.calculate(
                            atoms, properties=["energy", "forces"], system_changes=all_changes
                        )
                        self.assertTrue(np.isfinite(calculator.results["energy"]))
                        np.testing.assert_array_equal(calculator.results["forces"][2], 0.0)

    def test_calculator_rejects_non_finite_geometry_under_both_policies(self):
        # A NaN distance fails every cutoff test: without the check the atoms
        # count as isolated (reference energy, zero forces) under 'reference'.
        with patch("fastmd.models.matris.converter.op_available", False):
            for policy in ("error", "reference"):
                calculator = MLIPCalculator(
                    model=make_model(), device="cpu", task="ef",
                    config=InferenceConfig(execution="eager", isolated_atoms=policy, report=False),
                )
                for field, message in (("positions", "1 of 3 atoms have non-finite positions"),
                                       ("cell", "cell is not finite")):
                    with self.subTest(isolated_atoms=policy, field=field):
                        atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                      cell=[20] * 3, pbc=True)
                        if field == "positions":
                            atoms.positions[1, 2] = np.nan
                        else:
                            atoms.cell[2, 2] = np.inf
                        with self.assertRaisesRegex(ValueError, message):
                            calculator.calculate(
                                atoms, properties=["energy", "forces"], system_changes=all_changes
                            )


if __name__ == "__main__":
    unittest.main()
