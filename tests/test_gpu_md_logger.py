"""AsyncGPUMDLogger must snapshot the state it was given, however late its
side-stream D2H copy runs, and must never leave its caller waiting.

Ordering: the copy stream waits for the producer (current) stream before
copying, but before the fix the producer did not wait for the copy: the next
step (an in-place CUDA-graph replay into the same state buffers, or a
caching-allocator reuse of freed eager state/scalar tensors) could overwrite
the source before the copy read it (write-after-read). The tests delay the
logger stream with ``torch.cuda._sleep`` to make that ordering gap
deterministic.

Liveness: the writer thread synchronizes each frame's copy event while the MD
thread may (re)capture a CUDA graph (capacity grow, tier switch, pool reset,
new model-graph bucket). Captures in torch's default "global" mode reject
that call from any thread, which killed the writer and hung close().
MatRIS captures in "thread_local" mode (fastmd.engine._graph_capture),
and a failing writer now releases its slots and re-raises in the caller.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from ase import Atoms, units
from ase.io import read

from fastmd.interfaces.calculator import MLIPCalculator
from fastmd.execution.gpu_logging import AsyncGPUMDLogger
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.execution.whole_step_cuda_graph import BatchedWholeStepGraphRunner
from fastmd.config import InferenceConfig
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.gpu_graph_builder import op_available

from helpers import deterministic_forces, system
from test_isolated_atom_handling import make_model

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
CUDA_GRAPHS = torch.cuda.is_available() and op_available and _fixed_builder_available


def _sleep_cycles(milliseconds: float) -> int:
    # clock_rate is in kHz
    return int(torch.cuda.get_device_properties(0).clock_rate * milliseconds)


def _delay_logger_stream(logger: AsyncGPUMDLogger, milliseconds: float) -> None:
    """Queue a busy kernel on the logger's copy stream before every enqueue,
    as a busy copy engine / concurrent D2H traffic would."""
    enqueue = logger.enqueue
    cycles = _sleep_cycles(milliseconds)

    def delayed(**kwargs):
        with torch.cuda.stream(logger._stream):
            torch.cuda._sleep(cycles)
        enqueue(**kwargs)

    logger.enqueue = delayed


def _atoms(n: int = 5) -> Atoms:
    positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.9, 0.9, 0.2], [0.3, 0.4, 1.0]]
    return Atoms(f"H{n}", positions=positions[:n], cell=[20.0] * 3, pbc=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class AsyncLoggerOrderingTest(unittest.TestCase):
    def test_logged_frames_survive_in_place_overwrite_and_allocator_reuse(self) -> None:
        atoms = _atoms()
        n = len(atoms)
        device = torch.device("cuda", torch.cuda.current_device())
        steps = 8
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.traj"
            logger = AsyncGPUMDLogger(
                atoms_template=atoms, device=device, trajectory=str(path), ring=2
            )
            _delay_logger_stream(logger, 20.0)
            positions = torch.zeros((n, 3), device=device)  # persistent, in place
            cell = torch.eye(3, device=device) * 20.0
            try:
                for step in range(steps):
                    positions.fill_(float(step))
                    momenta = torch.full((n, 3), step + 0.5, device=device)
                    forces = torch.full((n, 3), -float(step), device=device)
                    energy = torch.full((1,), 10.0 * step, device=device)
                    logger.enqueue(
                        step=step,
                        time_ps=0.001 * step,
                        positions=positions,
                        momenta=momenta,
                        forces=forces,
                        cell=cell,
                        potential_energy=energy,
                        kinetic_energy=energy + 1.0,
                        temperature=energy + 2.0,
                    )
                    # temporaries die here; their blocks are reused next step
                    del momenta, forces, energy
            finally:
                logger.close()
            frames = read(str(path), index=":")
        self.assertEqual(len(frames), steps)
        for step, frame in enumerate(frames):
            with self.subTest(step=step):
                np.testing.assert_array_equal(frame.positions, np.full((n, 3), step))
                np.testing.assert_array_equal(frame.get_momenta(), np.full((n, 3), step + 0.5))
                np.testing.assert_array_equal(frame.get_forces(), np.full((n, 3), -step))
                self.assertEqual(frame.get_potential_energy(), 10.0 * step)


@unittest.skipUnless(
    torch.cuda.is_available() and op_available and _fixed_builder_available,
    "CUDA graph builders are unavailable",
)
class GPUMDTrajectoryLoggingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.model = make_model(num_layers=1).cuda()

    def test_logged_trajectory_matches_synchronous_reference(self) -> None:
        # log_stress: every frame also carries its state's stress, MLIPCalculator(task="efs")'s at the
        # frame's positions on each execution path, with logged and unlogged launches interleaved (and a
        # window of two steps per launch that logs every window, after precapture steps); off: no stress.
        calculator = MLIPCalculator(model=self.model, task="efs", device="cuda",
                                      config=InferenceConfig(execution="eager", report=False))
        for execution, window, loginterval, precapture, ensemble, log_stress in (
                ("eager", 1, 1, 0, "nve", False), ("whole_step", 1, 1, 0, "nve", False),
                ("eager", 1, 2, 0, "nve", True), ("model_graph", 1, 2, 0, "nvt_nhc", True),
                ("whole_step", 1, 2, 0, "nvt", True), ("whole_step", 2, 2, 2, "nvt_nhc", True)):
            with self.subTest(execution=execution, window=window, loginterval=loginterval, precapture=precapture,
                              ensemble=ensemble, log_stress=log_stress), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "md.traj"
                atoms = system() if log_stress else _atoms()  # stress: a dense periodic system
                atoms.set_momenta(
                    np.random.default_rng(0).normal(0.0, 0.05, (len(atoms), 3))
                )
                md = GPUResidentMolecularDynamics(
                    atoms,
                    model=self.model,
                    task="ef",
                    ensemble=ensemble,
                    timestep=1.0,
                    trajectory=str(path),
                    loginterval=loginterval,
                    logger_ring=3,
                    config=InferenceConfig(execution=execution, report=False),
                    steps_per_launch=window,
                    log_stress=log_stress,
                    precapture_steps=precapture,
                )
                _delay_logger_stream(md.logger, 20.0)
                reference, temperatures = [], []
                masses = md.masses.double().cpu().numpy()
                try:
                    for _ in range(6):
                        md.step()
                        # synchronous snapshot of the committed step (producer
                        # stream only; the logger copy is still pending)
                        if md.nsteps % loginterval == 0:
                            reference.append(
                                (
                                    md.nsteps,
                                    md.state.positions.detach().cpu().numpy().copy(),
                                    md.state.forces.detach().cpu().numpy().copy(),
                                )
                            )
                        temperatures.append(
                            (md.temperature(), md.state.momenta.double().cpu().numpy())
                        )
                finally:
                    md.close()
                # reported T: float64 ASE 0.5*sum(p*p/m) of the float32 state (nvt_nhc: of its float64 state)
                for temperature, p in temperatures if md.stateful_integrator is None else ():
                    self.assertEqual(temperature.dtype, torch.float64)
                    kinetic = 0.5 * np.sum(p * (p / masses[:, None]))
                    np.testing.assert_allclose(
                        float(temperature), 2 * kinetic / (3 * len(atoms) * units.kB), rtol=1e-12
                    )
                frames = read(str(path), index=":")
                # frame 0 is the initial state; then one frame per logged step
                self.assertEqual(len(frames), len(reference) + 1)
                np.testing.assert_allclose(
                    frames[0].positions, atoms.positions, rtol=0, atol=1e-6
                )
                for (step, positions, forces), frame in zip(reference, frames[1:]):
                    np.testing.assert_array_equal(
                        frame.positions.astype(np.float32), positions, err_msg=f"step {step}"
                    )
                    np.testing.assert_array_equal(
                        frame.get_forces().astype(np.float32), forces, err_msg=f"step {step}"
                    )
                for index, frame in enumerate(frames):
                    if not log_stress:
                        self.assertNotIn("stress", frame.calc.results)
                        continue
                    state = frame.copy()
                    state.calc = calculator
                    reference = state.get_stress()
                    # The padded graphs' float32 roundoff, far below one step's change of the stress.
                    np.testing.assert_allclose(frame.get_stress(), reference, rtol=0,
                                               atol=1e-4 * np.abs(reference).max(), err_msg=f"frame {index}")


class _FailingSink:
    """Trajectory sink whose write() raises for frame ``fail_at``."""

    def __init__(self, fail_at: int) -> None:
        self.fail_at = fail_at
        self.frames = 0
        self.closed = False
        self.error = OSError(28, "No space left on device")

    def write(self, atoms) -> None:
        if self.frames == self.fail_at:
            raise self.error
        self.frames += 1

    def close(self) -> None:
        self.closed = True


def _within(seconds: float, fn) -> None:
    """Run fn in a daemon thread and fail instead of hanging the suite."""
    outcome: dict[str, BaseException] = {}

    def target() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise AssertionError(f"logger call still blocked after {seconds} s")
    if "error" in outcome:
        raise outcome["error"]


def _enqueue(logger: AsyncGPUMDLogger, step: int) -> None:
    device = logger.device
    values = torch.full((5, 3), float(step), device=device)
    energy = torch.full((1,), float(step), device=device)
    logger.enqueue(
        step=step, time_ps=0.001 * step, positions=values, momenta=values,
        forces=values, cell=torch.eye(3, device=device) * 20.0,
        potential_energy=energy, kinetic_energy=energy, temperature=energy,
    )


class LoggerWriterFailureTest(unittest.TestCase):
    """A failing writer releases every slot and item, and the next
    enqueue/flush/close raises its error in the caller's thread. Before the
    fix the writer thread died: enqueue blocked once the ring was exhausted,
    flush()/close() waited forever."""

    def devices(self) -> list[torch.device]:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda", torch.cuda.current_device()))
        return devices

    def assert_writer_error(self, error: BaseException, sink: _FailingSink) -> None:
        self.assertIsInstance(error, RuntimeError)
        self.assertIs(error.__cause__, sink.error)

    def test_enqueue_raises_once_the_writer_failed(self) -> None:
        for device in self.devices():
            with self.subTest(device=device.type):
                sink = _FailingSink(fail_at=2)
                ring = 2
                logger = AsyncGPUMDLogger(
                    atoms_template=_atoms(), device=device, trajectory=sink, ring=ring
                )
                attempted = []

                def run() -> None:
                    with self.assertRaises(RuntimeError) as caught:
                        for step in range(50):
                            attempted.append(step)
                            _enqueue(logger, step)
                    self.assert_writer_error(caught.exception, sink)
                    # frames after the failed one are dropped, never written
                    self.assertEqual(sink.frames, 2)
                    with self.assertRaises(RuntimeError) as caught:
                        logger.flush()
                    self.assert_writer_error(caught.exception, sink)
                    logger.close()  # already reported: releases, no raise

                _within(30, run)
                # prompt: step fail_at + ring only got a slot after the
                # failed frame released its own, so the next enqueue raised
                self.assertLessEqual(attempted[-1], sink.fail_at + ring + 1)
                self.assertTrue(sink.closed)
                self.assertFalse(logger.enabled)

    def test_flush_raises_writer_error(self) -> None:
        for device in self.devices():
            with self.subTest(device=device.type):
                sink = _FailingSink(fail_at=1)
                logger = AsyncGPUMDLogger(
                    atoms_template=_atoms(), device=device, trajectory=sink, ring=8
                )

                def run() -> None:
                    _enqueue(logger, 0)
                    _enqueue(logger, 1)  # fails in the writer after this returns
                    with self.assertRaises(RuntimeError) as caught:
                        logger.flush()
                    self.assert_writer_error(caught.exception, sink)
                    with self.assertRaises(RuntimeError) as caught:
                        _enqueue(logger, 2)
                    self.assert_writer_error(caught.exception, sink)
                    logger.close()

                _within(30, run)
                self.assertEqual(sink.frames, 1)
                self.assertTrue(sink.closed)

    def test_non_finite_frame_raises_in_the_caller(self) -> None:
        # NaN positions: every atom isolated, finite Epot, zero forces, NaN T.
        for device in self.devices():
            with self.subTest(device=device.type):
                sink = _FailingSink(fail_at=-1)
                logger = AsyncGPUMDLogger(atoms_template=_atoms(), device=device, trajectory=sink)
                values = torch.zeros((5, 3), device=device)
                bad = values.clone()
                bad[3, 1] = float("nan")
                energy = torch.ones((1,), device=device)

                def run() -> None:
                    for step, positions, temperature in ((0, values, energy), (1, bad, energy * float("nan"))):
                        logger.enqueue(step=step, time_ps=0.0, positions=positions, momenta=values,
                                       forces=values, cell=torch.eye(3, device=device) * 20.0,
                                       potential_energy=energy, kinetic_energy=energy,
                                       temperature=temperature)
                    with self.assertRaises(RuntimeError) as caught:
                        logger.flush()
                    self.assertIsInstance(caught.exception.__cause__, ValueError)
                    self.assertIn("1 atoms with non-finite positions", str(caught.exception.__cause__))
                    logger.close()

                _within(30, run)
                self.assertEqual(sink.frames, 2)

    def test_close_raises_unreported_writer_error_after_cleanup(self) -> None:
        for device in self.devices():
            with self.subTest(device=device.type):
                sink = _FailingSink(fail_at=0)
                logger = AsyncGPUMDLogger(
                    atoms_template=_atoms(), device=device, trajectory=sink, ring=1
                )

                def run() -> None:
                    _enqueue(logger, 0)
                    with self.assertRaises(RuntimeError) as caught:
                        logger.close()
                    self.assert_writer_error(caught.exception, sink)
                    self.assertTrue(sink.closed)
                    self.assertFalse(logger.enabled)
                    logger.close()  # idempotent

                _within(30, run)
                self.assertEqual(sink.frames, 0)


@unittest.skipUnless(CUDA_GRAPHS, "CUDA graph builders are unavailable")
class GraphCaptureModeTest(unittest.TestCase):
    """Every MatRIS capture uses the "thread_local" capture error mode, which
    lets another thread synchronize a non-captured event mid-capture, and the
    mode does not change what is captured. No garbage collection runs inside
    a capture."""

    def test_event_synchronize_from_another_thread_during_capture(self) -> None:
        from fastmd.engine._graph_capture import capture_graph

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            source = torch.arange(1024, dtype=torch.float32, device="cuda")
            event = torch.cuda.Event()
            event.record(stream)
        torch.cuda.synchronize()
        target = torch.zeros_like(source)
        capturing, done = threading.Event(), threading.Event()
        errors = []

        def writer() -> None:
            capturing.wait(30)
            try:
                event.synchronize()
            except BaseException as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        graph = torch.cuda.CUDAGraph()
        with capture_graph(graph):
            target.copy_(source * 2.0)
            capturing.set()
            # the writer's cudaEventSynchronize happens while capture is live
            self.assertTrue(done.wait(30))
            target.add_(1.0)
        thread.join(30)
        self.assertEqual(errors, [])
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(target, source * 2.0 + 1.0, rtol=0, atol=0)

    def test_capture_survives_an_unreachable_captured_graph(self) -> None:
        """A collection inside a capture would finalize an earlier runner's
        dead graph, whose reset() invalidates the capture; capture_graph pauses
        automatic collection."""
        import gc

        from fastmd.engine._graph_capture import capture_graph

        source = torch.arange(1024, dtype=torch.float32, device="cuda")
        target = torch.zeros_like(source)
        thresholds = gc.get_threshold()
        gc.collect()
        gc.set_threshold(1 << 30)  # no collection before the capture
        try:
            stale = torch.cuda.CUDAGraph()
            with capture_graph(stale):
                target.copy_(source)
            cycle = [stale]
            cycle.append(cycle)  # only a collection frees it, like a dropped runner
            del stale, cycle
            graph = torch.cuda.CUDAGraph()
            with capture_graph(graph):
                self.assertFalse(gc.isenabled())
                target.copy_(source * 3.0)
                gc.set_threshold(1)  # any allocation would now collect
                _ = [[] for _ in range(64)]
        finally:
            gc.set_threshold(*thresholds)
        self.assertTrue(gc.isenabled())
        gc.collect()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(target, source * 3.0, rtol=0, atol=0)

    def test_all_capture_sites_use_thread_local_mode(self) -> None:
        from test_isolated_atom_cuda import (
            make_atoms, make_builder, make_integrator, positions_of,
        )

        modes = []
        original = torch.cuda.CUDAGraph.capture_begin

        def capture_begin(graph, *args, **kwargs):
            modes.append(kwargs.get("capture_error_mode", "global"))
            return original(graph, *args, **kwargs)

        torch.manual_seed(0)
        model = make_model(num_layers=1).cuda()
        counts = {}
        with mock.patch.object(torch.cuda.CUDAGraph, "capture_begin", capture_begin):
            for execution in ("whole_step", "model_graph"):
                before = len(modes)
                md = GPUResidentMolecularDynamics(
                    system(), model=model, task="ef", ensemble="nve", timestep=0.5,
                    config=InferenceConfig(execution=execution, report=False),
                )
                try:
                    md.run(2)
                    if execution == "whole_step":
                        md.reset_whole_step_graph_pool()
                        md.run(1)
                finally:
                    md.close()
                counts[execution] = len(modes) - before
            before = len(modes)
            positions = torch.stack([positions_of(make_atoms("connected"))] * 2)
            BatchedWholeStepGraphRunner(
                model=model, builders=[make_builder(), make_builder()],
                integrator=make_integrator(), positions=positions,
                momenta=torch.zeros_like(positions), warmup=1,
            )
            counts["batched_whole_step"] = len(modes) - before
        self.assertGreaterEqual(counts["whole_step"], 2, counts)
        self.assertGreaterEqual(counts["model_graph"], 1, counts)
        self.assertGreaterEqual(counts["batched_whole_step"], 1, counts)
        self.assertEqual(set(modes), {"thread_local"})

    def test_replay_is_bitwise_identical_to_global_mode_capture(self) -> None:
        from fastmd.engine import _graph_capture as _graph_capture

        torch.manual_seed(0)
        model = make_model(num_layers=1).cuda()
        deterministic_forces(model)  # MatRIS atomics are not bitwise reproducible

        def trajectory(execution: str) -> np.ndarray:
            md = GPUResidentMolecularDynamics(
                system(), model=model, task="ef", ensemble="nvt",
                temperature=500.0, timestep=0.5, taut=20.0,
                config=InferenceConfig(execution=execution, report=False),
            )
            states = []
            try:
                for _ in range(12):
                    state = md.step()
                    states.append(torch.cat([
                        state.positions.reshape(-1), state.momenta.reshape(-1),
                        state.forces.reshape(-1),
                        state.potential_energy.reshape(-1).float(),
                    ]).cpu().numpy())
            finally:
                md.close()
            return np.array(states)

        for execution in ("whole_step", "model_graph"):
            with self.subTest(execution=execution):
                with mock.patch.object(_graph_capture, "CAPTURE_ERROR_MODE", "global"):
                    reference = trajectory(execution)
                np.testing.assert_array_equal(trajectory(execution), reference)


@unittest.skipUnless(CUDA_GRAPHS, "CUDA graph builders are unavailable")
class LoggerDuringCaptureTest(unittest.TestCase):
    """GPU MD with a trajectory whose writer thread synchronizes a pending
    frame's copy event in the middle of a whole-step recapture
    (reset_whole_step_graph_pool) or a new model-graph bucket capture
    (tests/logger_capture_worker.py hooks force that interleaving). Before
    the fix the capture failed (cudaErrorStreamCaptureUnsupported), the
    writer thread died and close() hung (subprocess, watchdog + timeout)."""

    def start_worker(self, execution: str) -> subprocess.Popen:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(TESTS)])
        command = [sys.executable, str(TESTS / "logger_capture_worker.py"), str(REPO), execution]
        return subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    def check_worker(self, execution: str, proc: subprocess.Popen) -> None:
        try:
            stdout, stderr = proc.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise AssertionError(
                f"logger_capture_worker {execution} hung:\n{stdout}\n{stderr}"
            ) from None
        self.assertEqual(
            proc.returncode, 0,
            f"logger_capture_worker {execution} failed:\n{stdout[-3000:]}\n"
            f"{stderr[-3000:]}",
        )
        self.assertIn("OK", stdout.splitlines())

    def test_capture_with_a_pending_frame_completes(self) -> None:
        # The two workers are independent processes and run concurrently:
        # each one's wall time is mostly interpreter start-up (imports, CUDA
        # and Triton cold start).
        workers = {
            execution: self.start_worker(execution)
            for execution in ("whole_step", "model_graph")
        }
        try:
            for execution, proc in workers.items():
                with self.subTest(execution=execution):
                    self.check_worker(execution, proc)
        finally:
            for proc in workers.values():
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()


def _analytic_forces(model, device) -> None:
    """md_regression_worker's periodic force on any device: a pure function of the positions."""
    phase = torch.tensor(np.random.default_rng(7).uniform(0.0, 1.0, (4096, 3)), dtype=torch.float32,
                         device=device)

    def forward(graphs, task="ef", is_training=False, n_real=None, **_):
        arg = 2 * np.pi * (graphs[0].atom_frac_coord - phase[: graphs[0].atom_frac_coord.shape[0]])
        return {"e": (0.8 / (2 * np.pi) * torch.cos(arg)).sum().reshape(1), "f": [0.8 * torch.sin(arg)]}

    model.forward = forward


def _md_state(md) -> dict:
    """The native engine's full dynamic state (host numpy), float64 integrator rows for nvt_nhc."""
    state = {"positions": md.state.positions, "momenta": md.state.momenta, "forces": md.state.forces,
             "energy": md.state.potential_energy.reshape(1)}
    if md.thermostat is not None:
        integrator = md.integrator
        state.update(q64=integrator.positions, p64=integrator.momenta, eta=integrator.eta,
                     p_eta=integrator.p_eta)
    return {key: value.detach().cpu().numpy().copy() for key, value in state.items()}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class NativeRestartTest(unittest.TestCase):
    """P=1 native restart (eager, a history-free analytic force): a run saved at step k, resumed with
    from_restart and continued is bitwise the uninterrupted run (ramp target and NHC chain included);
    the file is engine "native" and also resumes through the from_evaluator driver, whose files resume
    here (the decomposed chain's options are not compared across engines)."""

    def test_restart_continues_bitwise(self) -> None:
        from fastmd.execution.md_protocol import HeatingRamp
        from fastmd.execution.md_protocol import read_restart

        for device in (torch.device("cuda", torch.cuda.current_device()),):
            torch.manual_seed(0)
            model = make_model(num_layers=1).to(device)
            _analytic_forces(model, device)
            options = {"model": model, "task": "ef", "device": device,
                       "config": InferenceConfig(execution="eager", report=False)}
            for ensemble in ("nve", "nvt", "nvt_nhc"):
                with self.subTest(device=device.type, ensemble=ensemble), tempfile.TemporaryDirectory() as tmp:
                    ramp = None if ensemble == "nve" else HeatingRamp(300.0, 900.0, 12, 4)
                    path = Path(tmp) / "restart5.npz"
                    md = GPUResidentMolecularDynamics(
                        system(), ensemble=ensemble, timestep=0.5, taut=20.0, ramp=ramp,
                        output_dir=Path(tmp) / "out", thermo_interval=2, restart_interval=3, **options)
                    try:
                        for _ in range(5):  # mid ramp chunk
                            md.step()
                            md.write_outputs()
                        md.save_restart(path)
                        for _ in range(4):
                            md.step()
                        expected = _md_state(md)
                    finally:
                        md.close()
                    data = read_restart(path)
                    self.assertEqual((data.engine, data.step), ("native", 5))
                    np.testing.assert_array_equal(data.ids, np.arange(64))
                    thermo = (Path(tmp) / "out" / "md_thermo.txt").read_text().splitlines()
                    self.assertEqual([line.split()[0] for line in thermo[1:]], ["2", "4"])
                    resumed = GPUResidentMolecularDynamics.from_restart(path, **options)
                    try:
                        self.assertEqual(resumed.nsteps, 5)
                        for _ in range(4):
                            resumed.step()
                        actual = _md_state(resumed)
                    finally:
                        resumed.close()
                    self.assertEqual(sorted(actual), sorted(expected))
                    for key in expected:
                        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
                    with self.assertRaisesRegex(ValueError, "differ from the restart file"):
                        GPUResidentMolecularDynamics.from_restart(path, timestep=0.25, **options)
                    if ramp is not None:  # the ramp continues from the saved target; temperature= is not used
                        with GPUResidentMolecularDynamics.from_restart(path, temperature=123.0, **options) as other:
                            self.assertEqual(other.thermostat_state()["temperature_K"], data.temperature_K)


    def test_multi_step_windows_follow_one_step_per_launch(self) -> None:
        # steps_per_launch=4: one whole-step replay integrates 4 steps; a ramp with a new target every
        # step follows the per-step target table, bitwise the trajectory of one step per launch (and of
        # eager steps); a restart at a window end continues the windows.
        from fastmd.execution.md_protocol import HeatingRamp

        torch.manual_seed(0)
        model = make_model(num_layers=1).cuda()
        _analytic_forces(model, "cuda")
        options = {"model": model, "task": "ef", "device": "cuda", "timestep": 0.5, "taut": 20.0}
        for ensemble in ("nvt", "nvt_nhc", "nve"):
            ramp = None if ensemble == "nve" else HeatingRamp(300.0, 900.0, 12, 1)
            states = {}
            with self.subTest(ensemble=ensemble), tempfile.TemporaryDirectory() as tmp:
                for execution, window in (("whole_step", 1), ("whole_step", 4), ("eager", 4)):
                    config = InferenceConfig(execution=execution, report=False)
                    with GPUResidentMolecularDynamics(system(), ensemble=ensemble, ramp=ramp, config=config,
                                                      steps_per_launch=window, output_dir=Path(tmp) / execution,
                                                      thermo_interval=4, restart_interval=4,
                                                      **options) as md:
                        self.assertEqual(md.config_summary()["effective"]["execution"], execution)
                        md.run(8)
                        self.assertEqual(md.nsteps, 8)
                        if execution == "whole_step":
                            self.assertEqual(md.whole_step_stats()["launches"], 8 // window)
                        md.save_restart(Path(tmp) / f"{execution}{window}.npz")
                        md.run(4)
                        states[execution, window] = _md_state(md)
                for key in states[("whole_step", 1)]:
                    np.testing.assert_array_equal(states[("whole_step", 4)][key], states[("whole_step", 1)][key],
                                                  err_msg=key)
                    if key != "energy":  # this analytic energy also sums the whole-step graph's padding atoms
                        np.testing.assert_array_equal(states[("eager", 4)][key], states[("whole_step", 1)][key],
                                                      err_msg=key)
                with GPUResidentMolecularDynamics.from_restart(
                        Path(tmp) / "whole_step4.npz", steps_per_launch=4,
                        config=InferenceConfig(execution="whole_step", report=False),
                        **{k: v for k, v in options.items() if k in ("model", "task", "device")}) as resumed:
                    resumed.run(4)
                    for key, value in _md_state(resumed).items():
                        np.testing.assert_array_equal(value, states[("whole_step", 1)][key], err_msg=key)
                with self.assertRaisesRegex(ValueError, "steps_per_launch=4 must divide .*thermo_interval"):
                    GPUResidentMolecularDynamics(system(), ensemble=ensemble, steps_per_launch=4,
                                                 output_dir=Path(tmp) / "bad", thermo_interval=6, **options)
                with GPUResidentMolecularDynamics(system(), ensemble=ensemble, steps_per_launch=4,
                                                  config=InferenceConfig(execution="eager", report=False),
                                                  **options) as md, \
                        self.assertRaisesRegex(ValueError, "not whole windows"):
                    md.run(6)

    def test_dropped_driver_stops_its_logger_thread(self) -> None:
        import gc

        torch.manual_seed(0)
        model = make_model(num_layers=1).cuda()
        _analytic_forces(model, "cuda")
        with tempfile.TemporaryDirectory() as tmp:
            md = GPUResidentMolecularDynamics(
                system(), model=model, task="ef", device="cuda", ensemble="nve", timestep=0.5,
                logfile=str(Path(tmp) / "md.log"), loginterval=1,
                config=InferenceConfig(execution="eager", report=False))
            md.run(2)
            thread = md.logger._thread
            self.assertTrue(thread.is_alive())
            del md
            gc.collect()
            thread.join(30)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len((Path(tmp) / "md.log").read_text().splitlines()), 4)  # header + 3 frames


class LogStressOptionTest(unittest.TestCase):
    """log_stress fails fast where frames could not carry the stress (CPU)."""

    def test_log_stress_needs_frames_a_periodic_cell_and_the_native_engine(self) -> None:

        options = {"model": make_model(), "task": "ef", "device": "cpu", "ensemble": "nve", "log_stress": True,
                   "config": InferenceConfig(execution="eager", report=False)}
        with self.assertRaisesRegex(ValueError, "pass a trajectory"):
            GPUResidentMolecularDynamics(_atoms(), logfile="-", **options)
        molecule = _atoms()
        molecule.pbc = False
        with self.assertRaisesRegex(ValueError, "periodic in all three directions"):
            GPUResidentMolecularDynamics(molecule, trajectory=_FailingSink(fail_at=-1), **options)






if __name__ == "__main__":
    unittest.main()
