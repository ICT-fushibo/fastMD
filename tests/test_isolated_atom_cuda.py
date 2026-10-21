"""GPU regressions for isolation masks on eager and captured research paths."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from ase import Atoms

from fastmd.execution.cuda_graph import BucketedGraphRunner
from fastmd.execution.gpu_integrator import GPUIntegrator
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.config import InferenceConfig
from fastmd.execution.whole_step_cuda_graph import (
    BatchedWholeStepGraphRunner,
    WholeStepGraphRunner,
)
from fastmd.engine.policy import fusion_scope
from fastmd.graph import FixedCapacityGraphBuilder, atoms_to_graph_gpu
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.gpu_graph_builder import op_available
from fastmd.models.matris import functions as model_functions
from fastmd.models.matris import interaction_block
from fastmd.engine.kernels import decomposed_indexed_cat_silu_linear as decomposed

from test_isolated_atom_handling import make_model


def make_atoms(state: str) -> Atoms:
    positions = {
        "connected": [[0, 0, 0], [1, 0, 0], [0.5, 0.7, 0]],
        "partial": [[0, 0, 0], [1, 0, 0], [8, 8, 8]],
        "isolated": [[0, 0, 0], [8, 0, 0], [0, 8, 0]],
    }[state]
    return Atoms("H3", positions=positions, cell=[20] * 3, pbc=True)


def make_builder() -> FixedCapacityGraphBuilder:
    return FixedCapacityGraphBuilder(
        cell=torch.eye(3) * 20,
        atomic_numbers=torch.ones(3, dtype=torch.int32),
        pbc=torch.ones(3, dtype=torch.bool),
        u_capacity=8,
        t_capacity=32,
        atom_graph_cutoff=2.0,
        line_graph_cutoff=1.5,
        candidate_cutoff=3.0,
        n_dummy=4,
        device="cuda",
    )


def positions_of(atoms: Atoms) -> torch.Tensor:
    return torch.tensor(atoms.positions, dtype=torch.float32, device="cuda")


def make_integrator() -> GPUIntegrator:
    # A zero timestep lets topology change independently of the MD state update.
    return GPUIntegrator(
        torch.ones(3, device="cuda"), dt_fs=0.0, ensemble="nve", fix_com=False
    )


@unittest.skipUnless(
    torch.cuda.is_available() and op_available and _fixed_builder_available,
    "CUDA graph builders are unavailable",
)
class IsolatedAtomCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matmul_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(0)
        cls.model = make_model(num_layers=2, norm_type="layer").cuda()

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_float32_matmul_precision(cls.matmul_precision)

    def eager(self, atoms: Atoms, *, task: str = "efsm") -> dict:
        graph = atoms_to_graph_gpu(
            atoms,
            atom_graph_cutoff=2.0,
            line_graph_cutoff=1.5,
            check_isolated_atoms=False,
        )
        return self.model([graph], task=task, is_training=False)

    def assert_matches(self, captured: dict, eager: dict, n_real: int) -> None:
        for key in ("e", "ref_energy", "s"):
            if key in eager:
                torch.testing.assert_close(
                    captured[key][0], eager[key][0], rtol=2e-4, atol=2e-5
                )
        for key in ("f", "m"):
            if key in eager:
                torch.testing.assert_close(
                    captured[key][0][:n_real], eager[key][0], rtol=2e-4, atol=2e-5
                )

    def test_gpu_eager_all_isolated_and_mixed_batch(self) -> None:
        for states in (("isolated", "isolated"), ("connected", "partial", "isolated")):
            with self.subTest(states=states):
                atoms = [make_atoms(state) for state in states]
                graphs = atoms_to_graph_gpu(
                    atoms,
                    atom_graph_cutoff=2.0,
                    line_graph_cutoff=1.5,
                    check_isolated_atoms=False,
                )
                with fusion_scope():
                    output = self.model(graphs, task="efsm", is_training=False)
                for index, state in enumerate(states):
                    single = self.eager(atoms[index])
                    for key in ("e", "ref_energy", "f", "s", "m"):
                        torch.testing.assert_close(
                            output[key][index], single[key][0], rtol=2e-4, atol=2e-5
                        )
                    if state == "isolated":
                        torch.testing.assert_close(
                            output["e"][index], output["ref_energy"][index]
                        )
                        torch.testing.assert_close(
                            output["f"][index], torch.zeros((3, 3), device="cuda")
                        )
                        torch.testing.assert_close(
                            output["s"][index], torch.zeros((3, 3), device="cuda")
                        )
                        torch.testing.assert_close(
                            output["m"][index], torch.zeros(3, device="cuda")
                        )

    def test_g1_replay_updates_mask_without_recapture(self) -> None:
        runner = BucketedGraphRunner(
            self.model,
            task="efsm",
            u_step=8,
            t_step=32,
            min_pad_u=4,
            n_dummy=4,
            warmup=1,
        )
        # Explicitly exercise the research decomposed/sorted projection path at
        # tiny sizes; production defaults otherwise gate it by row count.
        calls = {"cat3": 0, "cat4": 0, "sorted": 0, "merged": 0}

        def record(name, original, sorted_argument):
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                if result is not None:
                    calls[name] += 1
                    calls["sorted"] += int(args[sorted_argument])
                    calls["merged"] += int(kwargs.get("precomputed_p0") is not None)
                return result

            return wrapped

        with (
            patch.object(model_functions, "_INDEXED_CAT_LINEAR_FUSE", True),
            patch.object(model_functions, "_INDEXED_CAT_LINEAR_MIN_ROWS", 0),
            patch.object(interaction_block, "THRESHOLD_VALUE", 0),
            patch.object(decomposed, "_LAST_ERROR", None),
            patch.object(
                decomposed,
                "decomposed_indexed_cat3_silu_linear",
                side_effect=record(
                    "cat3", decomposed.decomposed_indexed_cat3_silu_linear, 6
                ),
            ),
            patch.object(
                decomposed,
                "decomposed_indexed_cat4_silu_linear",
                side_effect=record(
                    "cat4", decomposed.decomposed_indexed_cat4_silu_linear, 8
                ),
            ),
        ):
            for state in ("connected", "partial", "isolated", "connected"):
                with self.subTest(state=state):
                    atoms = make_atoms(state)
                    graph = atoms_to_graph_gpu(
                        atoms,
                        atom_graph_cutoff=2.0,
                        line_graph_cutoff=1.5,
                        check_isolated_atoms=False,
                    )
                    expected = self.eager(atoms)
                    output, n_real = runner.run(graph)
                    self.assert_matches(output, expected, int(n_real))
                    self.assertEqual(runner.captures, 1)
            self.assertGreater(calls["cat3"], 0)
            self.assertGreater(calls["cat4"], 0)
            self.assertGreater(calls["sorted"], 0)
            self.assertGreater(calls["merged"], 0)
            self.assertIsNone(
                decomposed.last_decomposed_indexed_cat_silu_linear_error()
            )

    def test_whole_step_replay_updates_mask_without_recapture(self) -> None:
        atoms = make_atoms("connected")
        positions = positions_of(atoms)
        runner = WholeStepGraphRunner(
            model=self.model,
            builder=make_builder(),
            integrator=make_integrator(),
            positions=positions,
            momenta=torch.zeros_like(positions),
            warmup=1,
        )
        captured_graph = runner.graph
        for state in ("connected", "partial", "isolated", "connected"):
            with self.subTest(state=state):
                atoms = make_atoms(state)
                runner.positions.copy_(positions_of(atoms))
                runner.builder.refresh_candidates(runner.positions)
                output = runner.replay(strict=True)
                expected = self.eager(atoms, task="ef")
                torch.testing.assert_close(
                    output.potential_energy[0],
                    expected["e"][0] * 3,
                    rtol=2e-4,
                    atol=2e-5,
                )
                torch.testing.assert_close(
                    output.forces, expected["f"][0], rtol=2e-4, atol=2e-5
                )
                self.assertIs(runner.graph, captured_graph)
        self.assertEqual(runner.fallback_windows, 0)
        # NaN positions would make every atom isolated (finite reference energy,
        # zero forces); the replay's displacement readback stops the run.
        runner.positions[1, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite MD state"):
            runner.replay(strict=True)

    def test_whole_step_stale_candidate_recovery_allows_isolation(self) -> None:
        positions = positions_of(make_atoms("connected"))
        runner = WholeStepGraphRunner(
            model=self.model,
            builder=make_builder(),
            integrator=make_integrator(),
            positions=positions,
            momenta=torch.zeros_like(positions),
            warmup=1,
        )
        # Move outside the skin without refreshing: replay must roll back and
        # refresh candidates, but isolated atoms are no longer an invalid step.
        atoms = make_atoms("isolated")
        runner.positions.copy_(positions_of(atoms))
        output = runner.replay(strict=True)
        expected = self.eager(atoms, task="ef")
        torch.testing.assert_close(output.potential_energy[0], expected["e"][0] * 3)
        torch.testing.assert_close(output.forces, expected["f"][0])
        self.assertGreater(runner.fallback_windows, 0)
        self.assertGreater(runner.recovery_replays, 0)

    def test_batched_whole_step_masks_and_stale_recovery(self) -> None:
        positions = torch.stack(
            [positions_of(make_atoms("partial")), positions_of(make_atoms("isolated"))]
        )
        runner = BatchedWholeStepGraphRunner(
            model=self.model,
            builders=[make_builder(), make_builder()],
            integrator=make_integrator(),
            positions=positions,
            momenta=torch.zeros_like(positions),
            warmup=1,
        )
        captured_graph = runner.graph
        for states in (
            ("partial", "isolated"),
            ("isolated", "connected"),
            ("connected", "partial"),
        ):
            with self.subTest(states=states):
                atoms = [make_atoms(state) for state in states]
                runner.positions.copy_(
                    torch.stack([positions_of(item) for item in atoms])
                )
                # Leave candidates stale after the first window to exercise
                # the batched rollback/refilter path as isolation changes.
                output = runner.replay(strict=True)
                for replica, item in enumerate(atoms):
                    expected = self.eager(item, task="ef")
                    torch.testing.assert_close(
                        output.potential_energy[replica],
                        expected["e"][0] * 3,
                        rtol=2e-4,
                        atol=2e-5,
                    )
                    torch.testing.assert_close(
                        output.forces[replica], expected["f"][0], rtol=2e-4, atol=2e-5
                    )
                self.assertIs(runner.graph, captured_graph)
        self.assertGreater(runner.fallback_windows, 0)
        runner.positions[1, 2, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, r"non-finite MD state .*\(replica 1\)"):
            runner.replay(strict=True)

    def test_reference_free_gpu_md_remains_strict(self) -> None:
        for graph_mode in ("eager", "g1", "g2"):
            with self.subTest(graph_mode=graph_mode):
                model = make_model(reference_energy=None).cuda()
                md = None
                try:
                    with self.assertRaisesRegex(ValueError, "isolated atom"):
                        md = GPUResidentMolecularDynamics(
                            make_atoms("partial"),
                            model=model,
                            task="ef",
                            config=InferenceConfig(
                                execution={"eager": "eager", "g1": "model_graph", "g2": "whole_step"}[graph_mode],
                                report=False,
                            ),
                        )
                        md.run(1)
                finally:
                    if md is not None:
                        md.close()

    def test_gpu_md_reference_model_accepts_isolation(self) -> None:
        for graph_mode in ("eager", "g1", "g2"):
            for state in ("partial", "isolated"):
                with self.subTest(graph_mode=graph_mode, state=state):
                    atoms = make_atoms(state)
                    md = GPUResidentMolecularDynamics(
                        atoms,
                        model=self.model,
                        task="ef",
                        ensemble="nve",
                        timestep=0,
                        config=InferenceConfig(
                            execution={"eager": "eager", "g1": "model_graph", "g2": "whole_step"}[graph_mode],
                            report=False,
                        ),
                    )
                    try:
                        output = md.run(2)
                        expected = self.eager(atoms, task="ef")
                        torch.testing.assert_close(
                            output.potential_energy.reshape(-1)[0],
                            expected["e"][0] * 3,
                            rtol=2e-4,
                            atol=2e-5,
                        )
                        torch.testing.assert_close(
                            output.forces, expected["f"][0], rtol=2e-4, atol=2e-5
                        )
                    finally:
                        md.close()

    def test_strict_whole_step_rejects_isolation_at_start_and_mid_trajectory(
        self,
    ) -> None:
        # Strict means validate_isolated_atoms=True or a reference-free model.
        # Isolation present at construction, or appearing after it with
        # refreshed or stale candidates (the stale-candidate recovery replay),
        # must raise in single and batched runners.
        reference_free = make_model(reference_energy=None).cuda()
        isolated = positions_of(make_atoms("isolated"))
        for has_reference in (True, False):
            model = self.model if has_reference else reference_free
            kwargs = {"validate_isolated_atoms": True} if has_reference else {}
            for batched in (False, True):
                for moment in ("construction", "refreshed", "stale"):
                    with self.subTest(
                        has_reference=has_reference, batched=batched, moment=moment
                    ):
                        start = "isolated" if moment == "construction" else "connected"
                        positions = positions_of(make_atoms(start))
                        if batched:
                            positions = positions.unsqueeze(0).repeat(2, 1, 1)
                            make_runner = lambda: BatchedWholeStepGraphRunner(
                                model=model,
                                builders=[make_builder(), make_builder()],
                                integrator=make_integrator(),
                                positions=positions,
                                momenta=torch.zeros_like(positions),
                                warmup=1,
                                **kwargs,
                            )
                        else:
                            make_runner = lambda: WholeStepGraphRunner(
                                model=model,
                                builder=make_builder(),
                                integrator=make_integrator(),
                                positions=positions,
                                momenta=torch.zeros_like(positions),
                                warmup=1,
                                **kwargs,
                            )
                        if moment == "construction":
                            with self.assertRaisesRegex(ValueError, "isolated atom"):
                                make_runner()
                            continue
                        runner = make_runner()
                        runner.replay(strict=True)
                        # Only the last replica of a batch becomes isolated.
                        replicas = runner.positions.reshape(-1, 3, 3)
                        replicas[-1].copy_(isolated)
                        if moment == "refreshed":
                            builders = runner.builders if batched else [runner.builder]
                            for builder, replica in zip(builders, replicas):
                                builder.refresh_candidates(replica)
                        with self.assertRaisesRegex(ValueError, "isolated atom"):
                            runner.replay(strict=True)
        for has_reference in (True, False):
            with self.subTest(entrypoint="gpu_md", has_reference=has_reference):
                md = GPUResidentMolecularDynamics(
                    make_atoms("connected"),
                    model=self.model if has_reference else reference_free,
                    task="ef",
                    ensemble="nve",
                    timestep=0,
                    config=InferenceConfig(
                        execution="whole_step",
                        isolated_atoms="error" if has_reference else "reference",
                        report=False,
                    ),
                )
                try:
                    md.run(1)
                    md.state.positions.copy_(isolated)
                    with self.assertRaisesRegex(ValueError, "isolated atom"):
                        md.run(1)
                finally:
                    md.close()


if __name__ == "__main__":
    unittest.main()
