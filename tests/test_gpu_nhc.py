"""GPU Nose-Hoover chain (ensemble="nvt_nhc") against ASE 3.29 NoseHooverChainNVT.

1. Integrator alone vs ASE with analytic forces (harmonic springs, ASE
   LennardJones), float64 and float32 forces, chain variants, both backends;
   also the rank-decomposed protocol at P=1,
   ``GPUNoseHooverChainNVT(protocol="decomposed")`` with capacity slots
   (both ke2 updates, padding slots). ``set_temperature(T)`` == a fresh
   integrator at T, bitwise, also after the decomposed ``warmup_compile``.
2. One CUDA graph (backup + pre-force + force + post-force) replayed ==
   eager steps, bitwise, including rollback and a target temperature change
   between replays.
3. GPU-resident MD plumbing with a small random MatRIS: eager / model graph
   / whole-step graph agree, the cached forces are F(current positions)
   (also across capacity-grow rollbacks, candidate fallbacks and whole-step
   tier-cache switches), logged trajectory frames hold the float64 state
   and the thermostat variables, the admission guard, precapture,
   graph-pool reset and eager -> graph switches keep the float64 state, a
   failed runner warmup/capture leaves the shared state untouched, a runner
   seeds a fresh integrator with the supplied forces, and (tier 2, opt-in
   via MATRIS_TEST_REFERENCE_TREE) nve / Berendsen trajectories are bitwise
   identical to another source tree, e.g. the commit before ``nvt_nhc``.
4. Tier 2, opt-in via the real-model assets (MATRIS_TEST_S01_FRAMES and the
   MatRIS-10M-OAM checkpoint, tests/matris_test_assets.py): whole-step NHC
   MD on S01 (default capacity, forced capacity grow + recapture + rollback,
   tier-cache switches, forced candidate fallback) and the eager and model
   graph paths; the GPU forces are replayed through ASE NoseHooverChainNVT
   and the trajectories must agree to fp64 round-off.
5. The replica integrators (N replicas of one system in flat buffers,
   ``fastmd.execution.replica_batch``) against N single-system integrators
   fed the same forces: the Nose-Hoover chain bit for bit, eager and replayed
   from a captured graph; NVE / Berendsen with each replica's own
   centre-of-mass correction and degrees of freedom, at float32 rounding.
6. The batched FIRE (``fastmd.execution.system_relax.SystemFIRE``: systems in the slots of one union) against
   ASE's FIRE, with a fixed cell and with ASE's FrechetCellFilter, ASE LennardJones forces (and stress) fed at
   the positions (and cells) each pre-force half returns, with ASE's parameters and others, eager and replayed
   from captured CUDA graphs: each system's step count, positions and cells at every step to float64 round-off.
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest import mock
from pathlib import Path

import numpy as np
import torch
import ase.io
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
try:
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
except ImportError:
    NoseHooverChainNVT = None
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fastmd.execution.gpu_integrator import GPUIntegrator
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.execution.gpu_nhc import GPUNoseHooverChainNVT, maxwell_boltzmann_momenta
from fastmd.execution.replica_batch import ReplicaNoseHooverChainNVT, ReplicaVelocityVerlet
from fastmd.execution.system_batch import SlotLayout, SystemNoseHooverChainNVT, SystemVelocityVerlet
from fastmd.execution.system_relax import ACTIVE, CONVERGED, PENDING, FIREParameters, SystemFIRE
from fastmd.config import CapacityConfig, InferenceConfig
from fastmd.engine.comm import LocalCommunicator
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.gpu_graph_builder import op_available

from matris_test_assets import CHECKPOINT, load_s01_frames, real_model_assets_available
from test_isolated_atom_handling import make_model

CUDA = torch.cuda.is_available()
KEYS = ("q", "p", "eta", "p_eta")
TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
REFERENCE_TREE = os.environ.get("MATRIS_TEST_REFERENCE_TREE", "")


def _backends() -> list[tuple[str, str]]:
    out = [("cpu", "torch")]
    if CUDA:
        out += [("cuda", "triton"), ("cuda", "torch")]
    return out


def _argon() -> Atoms:
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(3)  # 108 atoms
    atoms.rattle(0.05, seed=3)
    return atoms


class Harmonic(Calculator):
    """Springs to q0: forces = (-k) * (q - q0), 2 IEEE ops, bitwise identical
    in NumPy and torch. ``dtype=float32`` mimics a float32 model."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, q0, k, dtype=np.float64):
        super().__init__()
        self.q0, self.k, self.dtype = np.array(q0), float(k), dtype

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.dtype == np.float32:
            d = self.atoms.positions.astype(np.float32) - self.q0.astype(np.float32)
            forces = np.float32(-self.k) * d
        else:
            d = self.atoms.positions - self.q0
            forces = -self.k * d
        energy = 0.5 * self.k * float(np.sum(d.astype(np.float64) ** 2))
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


class Replay(Calculator):
    """Returns recorded (energy, forces) in call order and records how far the
    positions ASE asks for are from the recorded evaluation positions."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, positions, forces, energies):
        super().__init__()
        self.q, self.f, self.e = positions, forces, energies
        self.calls = 0
        self.max_dq = 0.0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        i = self.calls
        self.calls += 1
        self.max_dq = max(self.max_dq, float(np.abs(self.atoms.positions - self.q[i]).max()))
        energy = float(self.e[i])
        self.results = {"energy": energy, "free_energy": energy, "forces": self.f[i].copy()}


def run_ase(atoms: Atoms, calc, n_steps: int, *, timestep, temperature, tdamp, **kw):
    """ASE NoseHooverChainNVT, state recorded at step 0 and after every step."""
    atoms = atoms.copy()
    atoms.calc = calc
    dyn = NoseHooverChainNVT(atoms, timestep=timestep, temperature_K=temperature,
                             tdamp=tdamp, **kw)
    rec = {key: [] for key in KEYS}

    def observe():
        rec["q"].append(atoms.get_positions())
        rec["p"].append(atoms.get_momenta())
        rec["eta"].append(dyn._thermostat._eta.copy())
        rec["p_eta"].append(dyn._thermostat._p_eta.copy())

    dyn.attach(observe, interval=1)
    dyn.run(n_steps)
    return {key: np.array(value) for key, value in rec.items()}


def max_relative(ref: dict, test: dict) -> dict[str, float]:
    out = {}
    for key in KEYS:
        scale = np.abs(ref[key]).max()
        out[key] = float(np.abs(ref[key] - test[key]).max() / scale) if scale > 0 else 0.0
    return out


def _flat_to_record(flat: np.ndarray, n: int, tchain: int) -> dict:
    n3 = 3 * n
    return {
        "q": flat[:, :n3].reshape(-1, n, 3),
        "p": flat[:, n3:2 * n3].reshape(-1, n, 3),
        "f": flat[:, 2 * n3:3 * n3].reshape(-1, n, 3),
        "eta": flat[:, 3 * n3:3 * n3 + tchain],
        "p_eta": flat[:, 3 * n3 + tchain:3 * n3 + 2 * tchain],
        "e": flat[:, 3 * n3 + 2 * tchain],
    }


def run_decomposed(atoms: Atoms, force_fn, n_steps: int, *, device, **kw) -> dict:
    """GPUNoseHooverChainNVT(protocol="decomposed") with one rank
    (LocalCommunicator), capacity n + 7 (whose padding slots must stay zero)
    and ``force_fn(q) -> forces``. Records like run_ase."""
    n = len(atoms)
    nhc = GPUNoseHooverChainNVT.from_atoms(
        atoms, protocol="decomposed", comm=LocalCommunicator(), capacity=n + 7, device=device, **kw)
    rec = {key: [] for key in KEYS}

    def observe():
        for key, value in zip(KEYS, (nhc.positions, nhc.momenta, nhc.eta, nhc.p_eta)):
            rec[key].append(value.cpu().numpy().copy())

    nhc.set_forces(force_fn(nhc.positions))
    observe()
    for _ in range(n_steps):
        nhc.step(force_fn)
        observe()
    st = nhc.state
    assert nhc.capacity >= n + 7 and not (st.positions[n:].any() or st.momenta[n:].any()), \
        "padding slots were updated"
    return {key: np.array(value) for key, value in rec.items()}


class MaxwellBoltzmannTest(unittest.TestCase):
    def test_momenta_bitwise_equal_to_ase(self) -> None:
        atoms = _argon()
        reference = atoms.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            MaxwellBoltzmannDistribution(reference, temperature_K=500.0,
                                         rng=np.random.default_rng(0))
        np.testing.assert_array_equal(
            maxwell_boltzmann_momenta(atoms.get_masses(), 500.0, seed=0),
            reference.get_momenta(),
        )


@unittest.skipIf(NoseHooverChainNVT is None, "ASE NoseHooverChainNVT unavailable")
class NHCIntegratorVersusASETest(unittest.TestCase):
    """The GPU integrator alone, driven by host calculators like ASE is."""

    T = 300.0
    DT = 1.0 * units.fs
    TDAMP = 100.0 * units.fs

    def _compare(self, make_calc, n_steps, *, force_dtype=torch.float64, tol=1e-12,
                 temperature=None, decomposed=(), **kw) -> dict:
        """``decomposed``: also run GPUNoseHooverChainNVT(protocol="decomposed")
        at P=1 with these ``ke2_update`` modes for the first 100 steps."""
        temperature = self.T if temperature is None else temperature
        atoms = _argon()
        atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), temperature, seed=0))
        ref = run_ase(atoms, make_calc(), n_steps, timestep=self.DT,
                      temperature=temperature, tdamp=self.TDAMP, **kw)
        results = {}
        for device, backend in _backends():
            host = atoms.copy()
            host.calc = make_calc()

            def force_fn(q, host=host, device=device):
                host.set_positions(q.detach().cpu().numpy())
                forces = torch.from_numpy(np.ascontiguousarray(host.get_forces())).to(device)
                energy = torch.tensor([host.get_potential_energy()], dtype=torch.float64,
                                      device=device)
                return forces, energy

            nhc = GPUNoseHooverChainNVT.from_atoms(
                atoms, temperature_K=temperature, timestep=self.DT, tdamp=self.TDAMP,
                device=device, backend=backend, force_dtype=force_dtype, **kw,
            )
            forces, energy = force_fn(nhc.positions)
            nhc.set_forces(forces, energy=energy)
            flat = [nhc.state.flat.cpu().numpy().copy()]
            for _ in range(n_steps):
                nhc.step(force_fn)
                flat.append(nhc.state.flat.cpu().numpy().copy())
            runs = {"nhc": _flat_to_record(np.array(flat), len(atoms), nhc.tchain)}
            for ke2_update in decomposed:
                runs["decomposed-" + ke2_update] = run_decomposed(
                    atoms, lambda q, force_fn=force_fn: force_fn(q)[0], min(n_steps, 100),
                    device=device, temperature_K=temperature, timestep=self.DT,
                    tdamp=self.TDAMP, force_dtype=force_dtype, ke2_update=ke2_update,
                    backend=backend, **kw,
                )
            for name, test in runs.items():
                steps = len(test["q"])
                deviation = max_relative({key: ref[key][:steps] for key in KEYS}, test)
                results[f"{name}/{device}/{backend}"] = deviation
                with self.subTest(integrator=name, device=device, backend=backend):
                    for key in KEYS:
                        self.assertLess(deviation[key], tol, (key, deviation))
        return results

    def test_trajectories_match_ase(self) -> None:
        # 300 steps per case: a 1e-9 relative change of kT or of one
        # Suzuki-Yoshida delta already deviates by >= 1e-10, while round-off
        # grows slowly (measured max relative deviation <= 1.1e-14 after 300
        # steps, <= 3e-14 after 1000; GB300, both backends, and CPU), so
        # longer runs do not tighten the check.
        q0 = _argon().get_positions()
        # decomposed: also protocol="decomposed" at P=1 with these ke2_update modes
        cases = {
            "harmonic, float64 forces": (lambda: Harmonic(q0, 1.7), 300, {}),
            # a float32 model: ASE kicks with the NumPy float32 forces
            "harmonic, float32 forces": (lambda: Harmonic(q0, 1.7, dtype=np.float32), 300,
                                         {"force_dtype": torch.float32, "decomposed": ("scaled",)}),
            "tchain=1, tloop=1": (lambda: Harmonic(q0, 1.7), 300, {"tchain": 1, "tloop": 1}),
            "tchain=5, tloop=2": (lambda: Harmonic(q0, 1.7), 300,
                                  {"tchain": 5, "tloop": 2, "decomposed": ("scaled", "recompute")}),
            # chaotic dynamics: forces nonlinear in the positions (ASE's
            # LennardJones, ~4 ms per call, is most of this test's time)
            "Lennard-Jones": (lambda: LennardJones(sigma=3.4, epsilon=0.0104, rc=8.0, smooth=True),
                              300, {"temperature": 80.0}),
        }
        for name, (make_calc, n_steps, kw) in cases.items():
            with self.subTest(case=name):
                self._compare(make_calc, n_steps, **kw)

    def test_set_temperature_equals_fresh_integrator(self) -> None:
        # heating ramp: bitwise a fresh integrator at the new target; the
        # current target changes no bit
        atoms = _argon()
        atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), 300.0, seed=0))
        n, kw = len(atoms), {"timestep": self.DT, "tdamp": self.TDAMP, "tchain": 4}
        for device, backend in _backends():
            q0 = torch.tensor(atoms.get_positions(), device=device)
            force_fn = _torch_harmonic(q0, 1.7)

            def run(nhc):
                if isinstance(nhc, GPUIntegrator):  # Berendsen, float32 state as GPU-MD
                    q = q0.to(torch.float32)
                    p = torch.tensor(atoms.get_momenta(), dtype=torch.float32, device=device)
                    force32 = _torch_harmonic(q, 1.7)
                    forces = force32(q)[0]
                    for _ in range(20):
                        q, p, forces = nhc.step(q, p, forces, lambda x: force32(x)[0])
                    return torch.cat([q.reshape(-1), p.reshape(-1)])
                nhc.set_forces(*force_fn(nhc.positions))
                for _ in range(20):
                    nhc.step(force_fn)
                return nhc.state.flat

            makers = {
                "nhc": lambda T: GPUNoseHooverChainNVT.from_atoms(
                    atoms, temperature_K=T, device=device, backend=backend, **kw),
                "decomposed": lambda T: GPUNoseHooverChainNVT.from_atoms(
                    atoms, temperature_K=T, protocol="decomposed", comm=LocalCommunicator(),
                    capacity=n + 7, device=device, backend=backend, **kw),
                # Berendsen: the device target (float64 and its float32 rounding)
                "berendsen": lambda T: GPUIntegrator(
                    torch.tensor(atoms.get_masses(), dtype=torch.float32, device=device),
                    dt_fs=1.0, ensemble="nvt", temperature=T, taut_fs=20.0, fix_com=False),
            }

            def constants(integrator):
                if isinstance(integrator, GPUIntegrator):
                    return torch.stack((integrator._target64, integrator._target32.double()))
                return integrator.consts

            for name, make in makers.items():
                if name == "berendsen" and backend == "triton":
                    continue  # no NHC backend involved: once per device
                with self.subTest(integrator=name, device=device, backend=backend):
                    fresh, ramped = make(450.0), make(300.0)
                    if name == "decomposed":  # changes no bit either
                        ramped.warmup_compile()
                    consts = constants(ramped).clone()
                    ramped.set_temperature(300.0)
                    self.assertTrue(torch.equal(constants(ramped), consts))
                    ramped.set_temperature(450.0)
                    self.assertTrue(torch.equal(constants(ramped), constants(fresh)))
                    if name != "berendsen":
                        self.assertEqual((ramped.kT, ramped.ndof_kT), (fresh.kT, fresh.ndof_kT))
                    self.assertTrue(torch.equal(run(ramped), run(fresh)))


def _torch_harmonic(q0: torch.Tensor, k: float):
    minus_k = torch.tensor(-k, dtype=torch.float64).item()

    def force_fn(q):
        d = q - q0
        return d * minus_k, 0.5 * k * (d * d).sum().reshape(1)

    return force_fn


@unittest.skipUnless(CUDA, "CUDA is unavailable")
class NHCCudaGraphTest(unittest.TestCase):
    def test_graph_replay_is_bitwise_equal_to_eager_and_rolls_back(self) -> None:
        atoms = _argon()
        atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), 300.0, seed=0))
        q0 = torch.tensor(atoms.get_positions(), device="cuda")
        force_fn = _torch_harmonic(q0, 1.7)
        n_steps = 100  # bitwise: a stale or mis-captured value differs from step 1
        for backend in ("triton", "torch"):
            with self.subTest(backend=backend):
                def make():
                    nhc = GPUNoseHooverChainNVT.from_atoms(
                        atoms, temperature_K=300.0, timestep=units.fs,
                        tdamp=100 * units.fs, backend=backend,
                    )
                    forces, energy = force_fn(nhc.positions)
                    nhc.set_forces(forces, energy=energy)
                    nhc.warmup_compile(energy_dtype=torch.float64)
                    return nhc

                eager = make()
                for i in range(n_steps):
                    if i == n_steps // 2:  # heating ramp: new consts between replays
                        eager.set_temperature(450.0)
                    eager.step(force_fn)

                graphed = make()
                saved = graphed.snapshot()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        graphed.step(force_fn)
                torch.cuda.current_stream().wait_stream(side)
                graphed.load_snapshot(saved)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graphed.backup()
                    q = graphed.step_pre_force()
                    forces, energy = force_fn(q)
                    graphed.step_post_force(forces, energy=energy)
                graphed.load_snapshot(saved)
                for i in range(n_steps):
                    if i == n_steps // 2:
                        graphed.set_temperature(450.0)
                    graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(graphed.state.flat, eager.state.flat))
                graph.replay()
                self.assertFalse(torch.equal(graphed.state.flat, eager.state.flat))
                graphed.restore()
                self.assertTrue(torch.equal(graphed.state.flat, eager.state.flat))
        with self.subTest(integrator="berendsen"):
            # the Berendsen target is a graph input as well (whole-step runners)
            masses = torch.tensor(atoms.get_masses(), dtype=torch.float32, device="cuda")
            q32 = q0.to(torch.float32)
            force32 = _torch_harmonic(q32, 1.7)
            p0 = torch.tensor(atoms.get_momenta(), dtype=torch.float32, device="cuda")

            def berendsen():
                return GPUIntegrator(masses, dt_fs=1.0, ensemble="nvt", temperature=300.0,
                                     taut_fs=20.0, fix_com=True)

            eager, q, p = berendsen(), q32.clone(), p0.clone()
            forces = force32(q)[0]
            for i in range(n_steps):
                if i == n_steps // 2:
                    eager.set_temperature(450.0)
                q, p, forces = eager.step(q, p, forces, lambda x: force32(x)[0])
            graphed = berendsen()
            state = [q32.clone(), p0.clone(), force32(q32)[0]]

            def body():
                positions, momenta = graphed.step_pre_force(*state)
                new_forces = force32(positions)[0]
                momenta = graphed.step_post_force(momenta, new_forces)
                for buffer, value in zip(state, (positions, momenta, new_forces)):
                    buffer.copy_(value)

            initial = [buffer.clone() for buffer in state]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                body()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()
            for buffer, value in zip(state, initial):
                buffer.copy_(value)
            for i in range(n_steps):
                if i == n_steps // 2:
                    graphed.set_temperature(450.0)
                graph.replay()
            torch.cuda.synchronize()
            for buffer, value in zip(state, (q, p, forces)):
                self.assertTrue(torch.equal(buffer, value))


def _replicas(n: int, count: int, seed: int = 0):
    """Masses of mixed elements (n,), spring centres q0 (n, 3) and ``count`` replicas of one system: q0
    perturbed by 0.05 A and Maxwell-Boltzmann momenta at 300, 400, ... K, (count, n, 3) each."""
    rng = np.random.default_rng(seed)
    masses = rng.choice([1.008, 12.011, 63.546, 132.905], size=n)
    q0 = rng.uniform(0.0, 8.0, (n, 3))
    positions = q0 + rng.normal(0.0, 0.05, (count, n, 3))
    momenta = np.stack([maxwell_boltzmann_momenta(masses, 300.0 + 100.0 * r, seed=seed + r)
                        for r in range(count)])
    return masses, q0, positions, momenta


@unittest.skipUnless(CUDA, "CUDA is unavailable")
class ReplicaIntegratorTest(unittest.TestCase):
    """The replica integrators (``count`` replicas of one system in flat
    ``(count * n, 3)`` buffers) against ``count`` single-system integrators,
    each fed its replica's forces: random kicks plus springs to q0, computed
    elementwise from the positions each pre-force half returns."""

    STEPS = 20

    def test_replica_nhc_is_bitwise_single_integrators(self) -> None:
        """ReplicaNoseHooverChainNVT equals single-system GPUNoseHooverChainNVT
        integrators bit for bit (positions, momenta, cached forces, eta,
        p_eta, potential energy) over 20 steps with a target change halfway,
        eagerly and replayed from one captured CUDA graph (whose backup
        restore() rolls the last replay back), for odd and even 3n (n = 7, 64,
        135; each replica's rows padded to a multiple of 16 elements), 1 and 3
        replicas, tloop 1 (tchain 3, float32 forces: the NumPy kick) and tloop
        2 (tchain 5, float64 forces). The kinetic energy, temperature and
        conserved energy per replica are torch sums over the replicas' rows,
        in another order than one system's: relative 1e-14 (measured: equal,
        or 1 ulp at n = 135)."""
        steps = self.STEPS
        cases = itertools.product((7, 64, 135), (1, 3), ((3, 1, torch.float32), (5, 2, torch.float64)),
                                  (False, True))
        for n, count, (tchain, tloop, force_dtype), graphed in cases:
            with self.subTest(n=n, replicas=count, tchain=tchain, tloop=tloop, graphed=graphed):
                masses, q0, positions, momenta = _replicas(n, count)
                generator = torch.Generator().manual_seed(n + count)
                kicks = torch.randn((steps + 1, count * n, 3), generator=generator, dtype=torch.float64).cuda()
                energies = torch.randn((steps + 1, count), generator=generator, dtype=torch.float64).cuda()
                centres = torch.tensor(np.tile(q0, (count, 1)), device="cuda")

                def forces(q, kick, centre):
                    return (kick + (q - centre) * -1.7).to(force_dtype)

                kw = dict(temperature_K=300.0, timestep=units.fs, tdamp=20 * units.fs, tchain=tchain, tloop=tloop,
                          force_dtype=force_dtype, device="cuda")
                singles = [GPUNoseHooverChainNVT(masses, positions[r], momenta[r], backend="triton", **kw)
                           for r in range(count)]
                rows = [slice(r * n, (r + 1) * n) for r in range(count)]
                for r, (single, row) in enumerate(zip(singles, rows)):
                    single.set_forces(forces(single.positions, kicks[0, row], centres[row]),
                                      energy=energies[0, r:r + 1])
                replica = ReplicaNoseHooverChainNVT(masses, positions, momenta, replicas=count, **kw)
                replica.set_forces(forces(replica.positions, kicks[0], centres), energy=energies[0])
                kick, energy = torch.zeros_like(kicks[0]), torch.zeros_like(energies[0])  # the graph's inputs

                def replica_step():
                    replica.backup()
                    q = replica.step_pre_force()
                    replica.step_post_force(forces(q, kick, centres), energy=energy)

                if graphed:  # warm up (Triton compiles), then capture one step
                    saved = replica.snapshot()
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(2):
                            replica_step()
                    torch.cuda.current_stream().wait_stream(side)
                    replica.load_snapshot(saved)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        replica_step()
                    replica.load_snapshot(saved)
                for k in range(1, steps + 1):
                    if k == steps // 2:  # a new target between replays
                        for integrator in singles + [replica]:
                            integrator.set_temperature(450.0)
                    for r, (single, row) in enumerate(zip(singles, rows)):
                        q = single.step_pre_force()
                        single.step_post_force(forces(q, kicks[k, row], centres[row]), energy=energies[k, r:r + 1])
                    kick.copy_(kicks[k])
                    energy.copy_(energies[k])
                    if graphed:
                        graph.replay()
                    else:
                        replica_step()
                torch.cuda.synchronize()
                for name, actual, expected in (
                        ("positions", replica.positions.reshape(count, n, 3), [s.positions for s in singles]),
                        ("momenta", replica.momenta.reshape(count, n, 3), [s.momenta for s in singles]),
                        ("forces", replica.forces.reshape(count, n, 3), [s.forces for s in singles]),
                        ("eta", replica.eta, [s.eta for s in singles]),
                        ("p_eta", replica.p_eta, [s.p_eta for s in singles]),
                        ("potential_energy", replica.potential_energy, [s.potential_energy[0] for s in singles])):
                    self.assertTrue(torch.equal(actual, torch.stack(expected)), name)
                for name in ("kinetic_energy", "temperature", "conserved_energy"):
                    torch.testing.assert_close(getattr(replica, name)(),
                                               torch.stack([getattr(s, name)() for s in singles]),
                                               rtol=1e-14, atol=0, msg=name)
                if graphed:  # the captured backup: restore() rolls the last replay back
                    state = replica.state.flat.clone()
                    graph.replay()
                    self.assertFalse(torch.equal(replica.state.flat, state))
                    replica.restore()
                    self.assertTrue(torch.equal(replica.state.flat, state))

    def test_replica_velocity_verlet_matches_single_integrators(self) -> None:
        """ReplicaVelocityVerlet (NVE and Berendsen NVT, fix_com) against
        single-system GPUIntegrators (3n degrees of freedom, fix_com), each
        on its replica's rows, over 20 steps: positions and momenta within
        1e-5 of the largest value (float32 state). Not bit for bit in general:
        the per-replica sums (the Berendsen kinetic energy, the centre-of-mass
        momentum) reduce the replicas' (count, n, 3) view in another order than
        one (n, 3) system (measured: equal for these sizes; from the first step
        on for n = 500, under 2e-5 A after 100 steps). The temperatures are
        each replica's (relative 1e-6), and fix_com removes each replica's own
        centre-of-mass momentum (different and nonzero for each): after the
        pre-force half its total is below 1e-4 of the input's."""
        steps = self.STEPS
        for n, count, ensemble in itertools.product((7, 64, 135), (1, 3), ("nvt", "nve")):
            with self.subTest(n=n, replicas=count, ensemble=ensemble):
                masses, q0, positions, momenta = _replicas(n, count, seed=3)
                drift = np.random.default_rng(9).normal(0.0, 0.5, (count, 1, 3))  # a centre-of-mass velocity each
                momenta = momenta + drift * masses[:, None]
                masses32 = torch.tensor(masses, dtype=torch.float32, device="cuda")
                generator = torch.Generator().manual_seed(n + count)
                kicks = torch.randn((steps + 1, count * n, 3), generator=generator).cuda()
                centres = torch.tensor(np.tile(q0, (count, 1)), dtype=torch.float32, device="cuda")
                kw = dict(ensemble=ensemble, temperature=450.0, taut_fs=20.0, fix_com=True)
                replica = ReplicaVelocityVerlet(masses32, 1.0, replicas=count, **kw)
                singles = [GPUIntegrator(masses32, 1.0, degrees_of_freedom=3 * n, **kw) for _ in range(count)]
                rows = [slice(r * n, (r + 1) * n) for r in range(count)]

                def forces(q, k, row=slice(None)):
                    return kicks[k, row] + (q - centres[row]) * -1.7

                q = torch.tensor(positions.reshape(-1, 3), dtype=torch.float32, device="cuda")
                p = torch.tensor(momenta.reshape(-1, 3), dtype=torch.float32, device="cuda")
                f = forces(q, 0)
                state = [[q[row].clone(), p[row].clone(), f[row].clone()] for row in rows]
                torch.testing.assert_close(replica.temperature(p),
                                           torch.stack([s.temperature(p[row]) for s, row in zip(singles, rows)]),
                                           rtol=1e-6, atol=0)
                for k in range(1, steps + 1):
                    q, half = replica.step_pre_force(q, p, f)
                    if k == 1:  # each replica's own centre-of-mass momentum is removed
                        before = p.reshape(count, n, 3).sum(dim=1).norm(dim=1)
                        after = half.reshape(count, n, 3).sum(dim=1).norm(dim=1)
                        self.assertTrue(bool((after < 1e-4 * before).all()), (before, after))
                    f = forces(q, k)
                    p = replica.step_post_force(half, f)
                    for single, row, rows_state in zip(singles, rows, state):
                        single_q, single_half = single.step_pre_force(*rows_state)
                        single_f = forces(single_q, k, row)
                        rows_state[:] = single_q, single.step_post_force(single_half, single_f), single_f
                for index, (name, actual) in enumerate((("positions", q), ("momenta", p))):
                    expected = torch.cat([rows_state[index] for rows_state in state])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5 * float(expected.abs().max()),
                                               msg=name)

    def test_system_nhc_is_bitwise_single_integrators(self) -> None:
        """SystemNoseHooverChainNVT (different systems in the slots of one union, three slots with vacant rows)
        equals single-system GPUNoseHooverChainNVT integrators of each system's atom count bit for bit
        (positions, momenta, cached forces, eta, p_eta, potential energy) over 20 steps, eagerly and replayed
        from one captured CUDA graph, for systems of 7, 16, 43, 64, 100, 135 and 341 atoms (five size classes,
        blocks of 32 to 1024 elements; 3n a multiple of 16 or not), tloop 1 (tchain 3, float32 forces: the
        NumPy kick) and tloop 2 (tchain 5, float64 forces); the vacant rows never move. The kinetic energy,
        temperature and conserved energy per system: relative 1e-14."""
        steps, sizes, vacant = self.STEPS, (7, 16, 43, 64, 100, 135, 341), (0, 3, 0, 1, 0, 0, 5)
        for (tchain, tloop, force_dtype), graphed in itertools.product(
                ((3, 1, torch.float32), (5, 2, torch.float64)), (False, True)):
            with self.subTest(tchain=tchain, tloop=tloop, graphed=graphed):
                systems = [_replicas(n, 1, seed=n) for n in sizes]
                layout = SlotLayout(tuple(n + extra for n, extra in zip(sizes, vacant)))
                rows = [slice(start, start + n) for start, n in zip(layout.starts, sizes)]
                occupied = torch.zeros(layout.rows, dtype=torch.bool, device="cuda")
                centres = torch.zeros((layout.rows, 3), dtype=torch.float64, device="cuda")
                for row, (_, q0, _, _) in zip(rows, systems):
                    occupied[row] = True
                    centres[row] = torch.tensor(q0, device="cuda")
                generator = torch.Generator().manual_seed(tchain)
                kicks = torch.randn((steps + 1, layout.rows, 3), generator=generator, dtype=torch.float64).cuda()
                energies = torch.randn((steps + 1, len(sizes)), generator=generator, dtype=torch.float64).cuda()

                def forces(q, kick, row=slice(None)):  # zero on vacant rows; a fresh tensor (aligned)
                    return ((kick[row] + (q - centres[row]) * -1.7) * occupied[row, None]).to(force_dtype)

                kw = dict(temperature_K=300.0, timestep=units.fs, tdamp=20 * units.fs, tchain=tchain, tloop=tloop,
                          force_dtype=force_dtype, device="cuda")
                singles = [GPUNoseHooverChainNVT(masses, positions[0], momenta[0], backend="triton", **kw)
                           for masses, _, positions, momenta in systems]
                multi = SystemNoseHooverChainNVT(layout, [masses for masses, _, _, _ in systems],
                                                 [positions[0] for _, _, positions, _ in systems],
                                                 [momenta[0] for _, _, _, momenta in systems], **kw)
                for index, (single, row) in enumerate(zip(singles, rows)):
                    single.set_forces(forces(single.positions, kicks[0], row), energy=energies[0, index:index + 1])
                multi.set_forces(forces(multi.positions, kicks[0]), energy=energies[0])
                parked = multi.positions[~occupied].clone()
                kick, energy = torch.zeros_like(kicks[0]), torch.zeros_like(energies[0])  # the graph's inputs

                def multi_step():
                    multi.backup()
                    q = multi.step_pre_force()
                    multi.step_post_force(forces(q, kick), energy=energy)

                if graphed:  # warm up (Triton compiles), then capture one step
                    saved = multi.snapshot()
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(2):
                            multi_step()
                    torch.cuda.current_stream().wait_stream(side)
                    multi.load_snapshot(saved)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        multi_step()
                    multi.load_snapshot(saved)
                for k in range(1, steps + 1):
                    for index, (single, row) in enumerate(zip(singles, rows)):
                        q = single.step_pre_force()
                        single.step_post_force(forces(q, kicks[k], row), energy=energies[k, index:index + 1])
                    kick.copy_(kicks[k])
                    energy.copy_(energies[k])
                    if graphed:
                        graph.replay()
                    else:
                        multi_step()
                torch.cuda.synchronize()
                st = multi.state
                for index, single in enumerate(singles):
                    for name, actual, expected in (
                            ("positions", multi.system_view(st.q, index), single.positions),
                            ("momenta", multi.system_view(st.p, index), single.momenta),
                            ("forces", multi.system_view(st.f, index), single.forces),
                            ("eta", multi.eta[index], single.eta), ("p_eta", multi.p_eta[index], single.p_eta),
                            ("potential_energy", multi.potential_energy[index:index + 1], single.potential_energy)):
                        self.assertTrue(torch.equal(actual, expected), f"{name} of the {sizes[index]}-atom system")
                self.assertTrue(torch.equal(multi.positions[~occupied], parked), "a vacant row moved")
                for name in ("kinetic_energy", "temperature", "conserved_energy"):
                    torch.testing.assert_close(getattr(multi, name)(),
                                               torch.stack([getattr(s, name)()[0] if getattr(s, name)().ndim
                                                            else getattr(s, name)() for s in singles]),
                                               rtol=1e-14, atol=0, msg=name)

    def test_system_velocity_verlet_matches_single_integrators(self) -> None:
        """SystemVelocityVerlet (NVE and Berendsen NVT, fix_com) for systems of 7, 64 and 135 atoms in the slots
        of one union (two with vacant rows) against single-system GPUIntegrators (3n degrees of freedom,
        fix_com), each on its system's rows, over 20 steps: positions and momenta within 1e-5 of the largest
        value (float32 state; the per-system sums reduce in another order than one system's); the
        temperatures are each system's (relative 1e-6), fix_com removes each system's own centre-of-mass
        momentum, and the vacant rows never move."""
        steps, sizes, vacant = self.STEPS, (7, 64, 135), (2, 0, 5)
        for ensemble in ("nvt", "nve"):
            with self.subTest(ensemble=ensemble):
                layout = SlotLayout(tuple(n + extra for n, extra in zip(sizes, vacant)))
                rows = [slice(start, start + n) for start, n in zip(layout.starts, sizes)]
                masses = np.ones(layout.rows)
                q0 = np.zeros((layout.rows, 3))
                positions, momenta = np.zeros((layout.rows, 3)), np.zeros((layout.rows, 3))
                for index, (n, row) in enumerate(zip(sizes, rows)):
                    m, centre, q, p = _replicas(n, 1, seed=3 + index)
                    masses[row], q0[row], positions[row] = m, centre, q[0]
                    momenta[row] = p[0] + np.random.default_rng(index).normal(0.0, 0.5, (1, 3)) * m[:, None]
                occupied = torch.zeros(layout.rows, dtype=torch.bool, device="cuda")
                for row in rows:
                    occupied[row] = True
                masses32 = torch.tensor(masses, dtype=torch.float32, device="cuda")
                kw = dict(ensemble=ensemble, temperature=450.0, taut_fs=20.0, fix_com=True)
                multi = SystemVelocityVerlet(masses32, 1.0, ceilings=torch.tensor(layout.ceilings, device="cuda"),
                                             atoms=torch.tensor(sizes, device="cuda"),
                                             system_of_row=layout.system_of_row("cuda"), occupied=occupied, **kw)
                singles = [GPUIntegrator(masses32[row], 1.0, degrees_of_freedom=3 * n, **kw)
                           for n, row in zip(sizes, rows)]
                generator = torch.Generator().manual_seed(len(ensemble))
                kicks = torch.randn((steps + 1, layout.rows, 3), generator=generator).cuda() * occupied[:, None]
                centres = torch.tensor(q0, dtype=torch.float32, device="cuda")

                def forces(q, k, row=slice(None)):
                    return (kicks[k, row] + (q - centres[row]) * -1.7) * occupied[row, None]

                q = torch.tensor(positions, dtype=torch.float32, device="cuda")
                p = torch.tensor(momenta, dtype=torch.float32, device="cuda")
                parked = q[~occupied].clone()
                f = forces(q, 0)
                state = [[q[row].clone(), p[row].clone(), f[row].clone()] for row in rows]
                torch.testing.assert_close(multi.temperature(p),
                                           torch.stack([s.temperature(p[row]) for s, row in zip(singles, rows)]),
                                           rtol=1e-6, atol=0)
                for k in range(1, steps + 1):
                    q, half = multi.step_pre_force(q, p, f)
                    if k == 1:  # each system's own centre-of-mass momentum is removed
                        for row in rows:
                            before, after = p[row].sum(dim=0).norm(), half[row].sum(dim=0).norm()
                            self.assertLess(float(after), 1e-4 * float(before))
                    f = forces(q, k)
                    p = multi.step_post_force(half, f)
                    for single, row, rows_state in zip(singles, rows, state):
                        single_q, single_half = single.step_pre_force(*rows_state)
                        single_f = forces(single_q, k, row)
                        rows_state[:] = single_q, single.step_post_force(single_half, single_f), single_f
                self.assertTrue(torch.equal(q[~occupied], parked), "a vacant row moved")
                for index, (name, actual) in enumerate((("positions", q), ("momenta", p))):
                    expected = torch.cat([rows_state[index] for rows_state in state])
                    occupied_rows = torch.cat([actual[row] for row in rows])
                    torch.testing.assert_close(occupied_rows, expected, rtol=0,
                                               atol=1e-5 * float(expected.abs().max()), msg=name)


@unittest.skipUnless(CUDA, "CUDA is unavailable")
class SystemFIREVersusASETest(unittest.TestCase):
    """SystemFIRE (five rattled Ar systems of 3 to 12 atoms in slots with three vacant rows each; with the cell
    filter each cell also sheared by up to 6 %) against ASE's FIRE on each system alone, both fed ASE
    LennardJones forces (and stress) at their own positions (and cells) each step, to fmax 0.01 eV/A: with ASE's
    default parameters, with others (a small maxstep that clips most steps, Nmin 2, a lower dtmax, a mixing
    parameter other than astart), and with both step halves replayed from captured CUDA graphs."""

    DEVICE = "cuda"
    VARIANTS = ("defaults", "parameters", "captured")

    FMAX, MAX_STEPS = 0.01, 500
    PARAMETERS = dict(dt=0.07, maxstep=0.05, dtmax=0.6, Nmin=2, finc=1.2, fdec=0.4, astart=0.25, fa=0.93, a=0.05)

    @staticmethod
    def systems(cell_filter: bool) -> list[Atoms]:
        rng = np.random.default_rng(7)
        out = []
        for k, repeat in enumerate([(2, 2, 2), (3, 2, 2), (2, 3, 1), (1, 1, 3), (2, 2, 3)]):
            atoms = bulk("Ar", "fcc", a=5.4 + 0.1 * k).repeat(repeat)
            atoms.rattle(0.15, seed=k)
            if cell_filter:
                atoms.set_cell(atoms.cell.array @ (np.eye(3) + 0.03 * rng.standard_normal((3, 3))), scale_atoms=True)
            out.append(atoms)
        return out

    @staticmethod
    def calculator() -> LennardJones:
        return LennardJones(sigma=3.4, epsilon=0.0104, rc=8.0, smooth=True)

    def alone(self, atoms: Atoms, cell_filter: bool, parameters: dict) -> tuple:
        """ASE's FIRE on ``atoms`` alone: the positions and cell after every step, the energy and largest force row."""
        copy = atoms.copy()
        copy.calc = self.calculator()
        trajectory = [(copy.get_positions(), copy.cell.array.copy())]

        class Recorded(FIRE):  # the state after each step (observers see the start twice)
            def step(self, f=None):
                super().step(f)
                trajectory.append((copy.get_positions(), copy.cell.array.copy()))

        target = FrechetCellFilter(copy) if cell_filter else copy
        self.assertTrue(Recorded(target, logfile=None, **parameters).run(fmax=self.FMAX, steps=self.MAX_STEPS))
        return trajectory, copy.get_potential_energy(), float(np.linalg.norm(target.get_forces(), axis=1).max())

    def batched(self, systems: list[Atoms], cell_filter: bool, parameters: dict, captured: bool):
        """SystemFIRE over the systems, the LennardJones forces fed per step (through fixed buffers when the halves
        are replayed from CUDA graphs): the integrator and each system's positions (and cell) after each step."""
        counts = [len(atoms) for atoms in systems]
        layout = SlotLayout(tuple(n + 3 for n in counts))
        fire = SystemFIRE(layout, fmax=self.FMAX, max_steps=self.MAX_STEPS, cell_filter=cell_filter, device=self.DEVICE,
                          parameters=FIREParameters(**parameters) if parameters else None)
        for slot, atoms in enumerate(systems):
            fire.write_system(slot, positions=atoms.get_positions(), cell=atoms.cell.array)
        fire.sync()
        forces = torch.zeros((layout.rows, 3), dtype=torch.float64, device=self.DEVICE)
        energies = torch.zeros(layout.count, dtype=torch.float64, device=self.DEVICE)
        stresses = torch.zeros((layout.count, 3, 3), dtype=torch.float64, device=self.DEVICE)
        stress = stresses if cell_filter else None
        if captured:
            fire.backup()  # one eager pair first (cuBLAS and the kernels set up outside the capture), undone
            fire.step_pre_force()
            fire.step_post_force(forces, energy=energies, stress=stress)
            fire.restore()
            fire.sync()
            torch.cuda.synchronize()
            pre, post = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(pre):
                fire.step_pre_force()
            with torch.cuda.graph(post):
                fire.step_post_force(forces, energy=energies, stress=stress)
        seen = [dict() for _ in systems]
        for _ in range(self.MAX_STEPS + 2):
            pre.replay() if captured else fire.step_pre_force()
            positions = fire.positions.cpu().numpy().copy()
            cells = fire.cells.cpu().numpy().copy() if cell_filter else None
            for slot, atoms in enumerate(systems):
                copy = atoms.copy()
                if cell_filter:
                    copy.set_cell(cells[slot], scale_atoms=False)
                start = layout.starts[slot]
                copy.set_positions(positions[start:start + counts[slot]])
                copy.calc = self.calculator()
                forces[start:start + counts[slot]] = torch.as_tensor(copy.get_forces())
                energies[slot] = copy.get_potential_energy()
                if cell_filter:
                    stresses[slot] = torch.as_tensor(copy.get_stress(voigt=False))
            post.replay() if captured else fire.step_post_force(forces, energy=energies, stress=stress)
            for slot in range(layout.count):
                start = layout.starts[slot]
                seen[slot].setdefault(int(fire.state.steps[slot]), (
                    positions[start:start + counts[slot]], None if cells is None else cells[slot]))
            vacant = ~fire.occupied[:, 0]
            self.assertTrue(bool((fire.positions[vacant] == 0).all()), "a vacant row moved")
            if not any(int(phase) in (PENDING, ACTIVE) for phase in fire.phase.tolist()):
                break
        return fire, seen

    def test_steps_positions_and_cells_match_ase(self) -> None:
        """Fixed cell and FrechetCellFilter, each with ASE's defaults, the other parameters and captured halves:
        every system converges in ASE's number of steps, its positions (and cell) after every step within 1e-11 A of
        ASE's (float64 round-off; the cell filter's matrix exponential and logarithm are other algorithms than
        SciPy's), its final energy within 1e-12 eV and its largest filter force row ASE's; the vacant rows never move,
        and a system stops in the step it converges."""
        for cell_filter, variant in itertools.product((False, True), self.VARIANTS):
            with self.subTest(cell_filter=cell_filter, variant=variant):
                parameters = self.PARAMETERS if variant == "parameters" else {}
                systems = self.systems(cell_filter)
                expected = [self.alone(atoms, cell_filter, parameters) for atoms in systems]
                fire, seen = self.batched(systems, cell_filter, parameters, captured=variant == "captured")
                for slot, (trajectory, energy, fmax) in enumerate(expected):
                    self.assertEqual(int(fire.phase[slot]), CONVERGED)
                    self.assertEqual(int(fire.state.steps[slot]), len(trajectory) - 1, f"system {slot}: steps")
                    self.assertEqual(sorted(seen[slot]), list(range(len(trajectory))))
                    for step, (positions, cell) in enumerate(trajectory):
                        ours, our_cell = seen[slot][step]
                        np.testing.assert_allclose(ours, positions, rtol=0, atol=1e-11, err_msg=f"{slot}: {step}")
                        if cell_filter:
                            np.testing.assert_allclose(our_cell, cell, rtol=0, atol=1e-11, err_msg=f"{slot}: {step}")
                    self.assertAlmostEqual(float(fire.state.energy[slot]), energy, delta=1e-12)
                    self.assertAlmostEqual(float(fire.state.fmax[slot]), fmax, delta=1e-12)


class SystemFIREVersusASECPUtest(SystemFIREVersusASETest):
    """The same position/cell/step-count oracle on CPU (without CUDA capture)."""
    __unittest_skip__ = False
    DEVICE = "cpu"
    VARIANTS = ("defaults", "parameters")


def _small_system() -> Atoms:
    rng = np.random.default_rng(1)
    grid = np.stack(np.meshgrid(*(np.arange(3),) * 3, indexing="ij"), -1).reshape(-1, 3)
    atoms = Atoms("H27", positions=grid * 1.4 + rng.uniform(-0.1, 0.1, (27, 3)),
                  cell=[4.2] * 3, pbc=True)
    atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), 300.0, seed=0))
    return atoms


class SimulatedOutOfMemory(RuntimeError):
    pass


class FailingModel:
    """Delegates to ``model``; its ``fail_at``-th call raises."""

    def __init__(self, model, fail_at: int) -> None:
        self.model, self.fail_at, self.calls = model, fail_at, 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise SimulatedOutOfMemory(f"model call {self.calls}")
        return self.model(*args, **kwargs)


@unittest.skipUnless(
    CUDA and op_available and _fixed_builder_available, "CUDA graph builders are unavailable"
)
class GPUMDNoseHooverTest(unittest.TestCase):
    """GPU-resident MD plumbing with a small random MatRIS model."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.model = make_model(num_layers=1).cuda()

    def make_md(self, execution: str, atoms: Atoms | None = None,
                capacity: CapacityConfig | None = None, expert_overrides=None, **kw):
        return GPUResidentMolecularDynamics(
            (atoms or _small_system()).copy(), model=self.model, task="ef",
            ensemble=kw.pop("ensemble", "nvt_nhc"), temperature=300.0, timestep=0.5,
            config=InferenceConfig(
                execution=execution, capacity=capacity or CapacityConfig(),
                expert_overrides=expert_overrides or {}, report=False,
            ),
            **kw,
        )

    def assert_forces_of_positions(self, md, previous: torch.Tensor | None = None) -> None:
        """The cached forces (float64 state and float32 mirror) are
        F(current positions), not the forces of another step."""
        # The mirror before the evaluation: on the model_graph path
        # md.state.forces is a view of the bucket graph's output, which
        # evaluate() replays into, and a re-evaluation is bitwise only where
        # the model is (the index_add_ segment sums are atomics: with IEEE
        # FP32 GEMMs every replay differs by 4e-8..1.1e-7; with TF32 GEMMs,
        # TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, this model's sums are exact).
        torch.testing.assert_close(md.state.forces, md.integrator.forces.float(), rtol=0, atol=0)
        expected, _ = md.evaluate(md.integrator.positions.float())
        tolerance = 1e-4 * float(expected.abs().max()) + 1e-6  # fp32 model noise
        torch.testing.assert_close(
            md.integrator.forces, expected.double(), rtol=0, atol=tolerance
        )
        if previous is not None:  # the check can tell neighbouring steps apart
            self.assertGreater(float((previous - expected).abs().max()), 10 * tolerance)

    def test_execution_paths_agree_and_use_float64_state(self) -> None:
        with self.subTest(case="default tdamp = 100 timesteps, tchain=3, tloop=1"):
            md = self.make_md("eager")
            try:
                self.assertAlmostEqual(md.integrator.tdamp, 50.0 * units.fs)
                self.assertEqual((md.integrator.tchain, md.integrator.tloop), (3, 1))
            finally:
                md.close()
        states = {}
        for execution in ("eager", "model_graph", "whole_step"):
            md = self.make_md(execution, tdamp=40.0, tchain=4, tloop=2)
            try:
                self.assertIsInstance(md.integrator, GPUNoseHooverChainNVT)
                self.assertEqual((md.integrator.tchain, md.integrator.tloop), (4, 2))
                self.assertAlmostEqual(md.integrator.tdamp, 40.0 * units.fs)
                md.run(40)
                states[execution] = md.integrator.state.flat.clone()
                self.assertEqual(md.integrator.state.flat.dtype, torch.float64)
                torch.testing.assert_close(
                    md.state.positions, md.integrator.positions.float(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    md.state.momenta, md.integrator.momenta.float(), rtol=0, atol=0
                )
                self.assertTrue(torch.isfinite(md.conserved_energy()))
                if execution == "whole_step":  # the whole-step graph ran
                    self.assertIsNotNone(md.whole_step_stats())
                    self.assertGreater(md.whole_step_stats()["launches"], 0)
            finally:
                md.close()
        # different graph builders / padding: fp32 model noise only
        for execution in ("model_graph", "whole_step"):
            torch.testing.assert_close(states[execution], states["eager"], rtol=0, atol=1e-3)

    def test_forces_are_forces_of_current_positions(self) -> None:
        # name: (execution, capacity, expert_overrides, steps)
        cases = {
            "eager": ("eager", None, None, 30),
            "model_graph": ("model_graph", None, None, 30),
            "whole_step": ("whole_step", None, None, 30),
            # tight capacity (overflow rollback + grow + recapture) and a tiny
            # candidate skin (rollback + refresh + recovery / eager re-run)
            "whole_step_rollbacks": ("whole_step", CapacityConfig(
                u_step=1, t_step=8, min_pad_u=1, min_pad_t=1, candidate_skin=0.01,
            ), None, 30),
            # a fine tier ladder: the tier cache switches up (step 1) and
            # down (step 15), capturing a new tier each time; with two
            # cached tiers the down switch first evicts the oldest one. All
            # tiers share the float64 integrator state.
            "whole_step_tier_cache": ("whole_step", CapacityConfig(
                u_step=2, t_step=8, min_pad_u=1, min_pad_t=1,
            ), {
                "MATRIS_WHOLE_STEP_TIER_CACHE": True,
                "MATRIS_WHOLE_STEP_MAX_TIERS": 2,
                "MATRIS_TIER_HYSTERESIS_WINDOW": 10,
                "MATRIS_TIER_HYSTERESIS_SIGMA": 0.0,
                "MATRIS_TIER_CHECK_INTERVAL": 5,
                "MATRIS_TIER_DOWN_COOLDOWN": 0,
            }, 30),
        }
        for name, (execution, capacity, overrides, steps) in cases.items():
            with self.subTest(case=name):
                md = self.make_md(execution, capacity=capacity, expert_overrides=overrides)
                try:
                    md.run(steps - 1)
                    previous = md.integrator.forces.float().clone()
                    md.run(1)
                    self.assert_forces_of_positions(md, previous)
                    stats = md.whole_step_stats()
                    if name == "whole_step_rollbacks":
                        self.assertGreaterEqual(stats["capacity_recaptures"], 1, stats)
                        self.assertGreaterEqual(stats["fallback_windows"], 1, stats)
                    if name == "whole_step_tier_cache":
                        self.assertGreaterEqual(stats["tier_cache"]["switch_up"], 1, stats)
                        self.assertGreaterEqual(stats["tier_cache"]["switch_down"], 1, stats)
                        self.assertGreaterEqual(stats["tier_cache"]["evictions"], 1, stats)
                finally:
                    md.close()

    def test_logging_records_float64_state_and_thermostat(self) -> None:
        for execution in ("eager", "whole_step"):
            with self.subTest(execution=execution), tempfile.TemporaryDirectory() as tmp:
                trajectory, logfile = Path(tmp) / "md.traj", Path(tmp) / "md.log"
                md = self.make_md(execution, trajectory=str(trajectory),
                                  logfile=str(logfile), loginterval=2)
                records = {}
                try:
                    md._ensure_forces()
                    for step in range(9):
                        if step % 2 == 0:
                            records[step] = (
                                md.integrator.state.flat.cpu().numpy().copy(),
                                md.state.forces.cpu().numpy().copy(),
                                float(md.integrator.conserved_energy()),
                            )
                        md.step()
                    n, tchain = md.n_atoms, md.integrator.tchain
                finally:
                    md.close()
                frames = ase.io.read(trajectory, ":")
                self.assertEqual(len(frames), len(records))  # steps 0, 2, ..., 8
                for frame, (flat, forces, conserved) in zip(frames, records.values()):
                    state = _flat_to_record(flat[None], n, tchain)
                    # float64 state, not the float32 mirrors
                    np.testing.assert_array_equal(frame.get_positions(), state["q"][0])
                    np.testing.assert_array_equal(frame.get_momenta(), state["p"][0])
                    np.testing.assert_array_equal(frame.get_forces(), forces)
                    np.testing.assert_array_equal(frame.info["nhc_eta"], state["eta"][0])
                    np.testing.assert_array_equal(frame.info["nhc_p_eta"], state["p_eta"][0])
                    self.assertEqual(float(frame.info["nhc_conserved_energy"][0]), conserved)
                self.assertFalse(np.array_equal(
                    frames[-1].get_positions(),
                    frames[-1].get_positions().astype(np.float32).astype(np.float64),
                ))
                self.assertEqual(len(logfile.read_text().splitlines()), 1 + len(records))

    def test_pool_reset_and_path_switches_keep_float64_state(self) -> None:
        # whole-step graph pool reset: recapture from the committed state (the
        # fresh candidate list changes the edge order, hence fp32 model noise)
        reference, md = self.make_md("whole_step"), self.make_md("whole_step")
        try:
            reference.run(10)
            md.run(5)
            before = md.integrator.state.flat.clone()
            md.reset_whole_step_graph_pool()
            self.assertTrue(torch.equal(md.integrator.state.flat, before))
            self.assertGreaterEqual(md.whole_step_stats()["pool_resets"], 1)
            md.run(5)
            torch.testing.assert_close(md.integrator.state.flat,
                                       reference.integrator.state.flat, rtol=0, atol=1e-4)
            self.assert_forces_of_positions(md)
        finally:
            md.close()
            reference.close()
        # eager -> model graph / whole-step graph in the middle of a run
        for switch in ("enable_model_cuda_graph", "enable_whole_step_cuda_graph"):
            with self.subTest(switch=switch):
                reference, md = self.make_md("eager"), self.make_md("eager")
                try:
                    reference.run(10)
                    md.run(5)
                    before = md.integrator.state.flat.clone()
                    getattr(md, switch)()
                    self.assertTrue(torch.equal(md.integrator.state.flat, before))
                    md.run(5)
                    if switch == "enable_whole_step_cuda_graph":  # the whole-step graph ran
                        self.assertIsNotNone(md.whole_step_stats())
                        self.assertGreater(md.whole_step_stats()["launches"], 0)
                    # different graph builders: fp32 model noise only
                    torch.testing.assert_close(md.integrator.state.flat,
                                               reference.integrator.state.flat,
                                               rtol=0, atol=1e-4)
                    self.assert_forces_of_positions(md)
                finally:
                    md.close()
                    reference.close()

    def test_failed_runner_capture_keeps_the_shared_state(self) -> None:
        # Tier switches, capacity grows and pool resets capture a new runner
        # on the integrator of the active runner, and the capture warmup
        # advances it. A failure in a warmup window or in the capture (e.g.
        # out of memory in the model) must leave the committed state intact.
        reference, md = self.make_md("whole_step"), self.make_md("whole_step")
        try:
            reference.run(10)
            md.run(5)
            before = md.integrator.state.flat.clone()
            warmup = md._inference_config.capacity.warmup
            # A later capture of the same MD (growth, tier switch, pool reset:
            # recapture=True) runs one warmup window (CapturePolicy.warmups).
            for recapture in (False, True):
                windows = 1 if recapture else warmup
                # model calls: initial evaluation, warmup windows, capture
                phases = {"first warmup window": 2, "last warmup window": 1 + windows,
                          "capture": 2 + windows}
                for phase, fail_at in phases.items():
                    with self.subTest(recapture=recapture, phase=phase):
                        model = FailingModel(md.model, fail_at)
                        with mock.patch.object(md, "model", model), \
                                self.assertRaises(SimulatedOutOfMemory):
                            md._make_whole_step_runner(
                                positions=md.state.positions,
                                momenta=md.state.momenta,
                                forces=md.state.forces,
                                recapture=recapture,
                            )
                        self.assertEqual(model.calls, fail_at)
                        self.assertTrue(torch.equal(md.integrator.state.flat, before))
                with self.subTest(recapture=recapture, phase="success"):
                    model = FailingModel(md.model, 10**9)
                    with mock.patch.object(md, "model", model):
                        runner = md._make_whole_step_runner(positions=md.state.positions, momenta=md.state.momenta,
                                                            forces=md.state.forces, recapture=recapture)
                    self.assertEqual((runner.warmup_runs, model.calls), (windows, 2 + windows))
                    self.assertTrue(torch.equal(md.integrator.state.flat, before))
                    del runner
            # the active runner continues as if nothing had happened
            launches = md.whole_step_stats()["launches"]
            md.run(5)
            self.assertGreater(md.whole_step_stats()["launches"], launches)
            torch.testing.assert_close(md.integrator.state.flat,
                                       reference.integrator.state.flat, rtol=0, atol=1e-4)
            self.assert_forces_of_positions(md)
        finally:
            md.close()
            reference.close()

    def test_runner_seeds_a_fresh_integrator_with_supplied_forces(self) -> None:
        # Direct runner use: an integrator constructed without forces and a
        # runner given forces=F(q) must kick the first step with F(q).
        atoms = _small_system()
        md = self.make_md("whole_step", atoms)
        try:
            forces = md.state.forces.clone()
            nhc = md.integrator
            positions = []
            for seeded in (True, False):
                integrator = GPUNoseHooverChainNVT(
                    atoms.get_masses(), atoms.get_positions(), atoms.get_momenta(),
                    temperature_K=nhc.temperature_K, timestep=nhc.dt, tdamp=nhc.tdamp,
                    device=nhc.device, force_dtype=torch.float32,
                )
                if seeded:  # the MD driver's contract
                    integrator.set_forces(forces)
                with mock.patch.object(md, "integrator", integrator):
                    runner = md._make_whole_step_runner(
                        positions=md.state.positions,
                        momenta=md.state.momenta,
                        forces=forces,
                    )
                torch.testing.assert_close(integrator.forces, forces.double(), rtol=0, atol=0)
                runner.replay()
                positions.append(integrator.positions.clone())
                del runner
            # q1 depends on F(q0) only (the forces at q1 carry fp32 model noise)
            self.assertTrue(torch.equal(positions[1], positions[0]))
        finally:
            md.close()

    def test_precapture_and_admission_guard_restore_float64_state(self) -> None:
        # precapture replays are rolled back, including the thermostat state
        md = self.make_md("eager")
        try:
            md.run(3)
            before = md.integrator.state.flat.clone()
            md.enable_whole_step_cuda_graph(precapture_steps=4)
            self.assertTrue(torch.equal(md.integrator.state.flat, before))
            self.assertEqual(md.state.positions.data_ptr(),
                             md.whole_step_runner.positions.data_ptr())
        finally:
            md.close()

        for selected in ("dynamic_gpu", "whole_step_cuda_graph"):
            with self.subTest(selected=selected):
                # reference: the same path without the trials (F(q_0) from the
                # dynamic builder, then optionally the whole-step graph)
                reference = self.make_md("eager")
                reference._ensure_forces()
                if selected == "whole_step_cuda_graph":
                    reference.enable_whole_step_cuda_graph()
                md = self.make_md("eager")
                try:
                    trial = md._timed_trial
                    calls = []

                    def fake_trial(steps, trial=trial, calls=calls):
                        trial(steps)
                        calls.append(steps)
                        fast = (selected == "dynamic_gpu") == (len(calls) == 1)
                        return [1.0 if fast else 2.0] * steps

                    md._timed_trial = fake_trial
                    result = md.select_whole_step_cuda_graph(warmup_steps=3)
                    self.assertEqual(result["selected_path"], selected)
                    reference.run(3)
                    self.assertEqual(md.nsteps, 3)
                    # without the restore the trajectory would be 3 steps
                    # ahead (~2e-2 A); fp32 model noise is ~1e-6
                    torch.testing.assert_close(
                        md.integrator.state.flat, reference.integrator.state.flat,
                        rtol=0, atol=1e-4,
                    )
                    md.run(5)
                    reference.run(5)
                    torch.testing.assert_close(
                        md.integrator.state.flat, reference.integrator.state.flat,
                        rtol=0, atol=1e-4,
                    )
                finally:
                    md.close()
                    reference.close()


def _run_regression_worker(root: Path, out: Path) -> dict[str, np.ndarray]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(TESTS)])
    proc = subprocess.run(
        [sys.executable, str(TESTS / "md_regression_worker.py"), str(root), str(out)],
        env=env, capture_output=True, text=True, timeout=1200,
    )
    if proc.returncode != 0:
        raise AssertionError(f"worker failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return dict(np.load(out))


@unittest.skipUnless(
    CUDA and op_available and _fixed_builder_available and REFERENCE_TREE,
    "set MATRIS_TEST_REFERENCE_TREE to a reference source tree",
)
class NVEBerendsenRegressionTest(unittest.TestCase):
    """Tier 2 (opt-in via MATRIS_TEST_REFERENCE_TREE).

    nve / Berendsen nvt are untouched by the stateful-integrator plumbing:
    the same short MD (64 atoms, 40 steps, trajectory every 5 steps; a
    deterministic analytic force field in place of the MatRIS forward, whose
    atomics are not reproducible run to run) run with this tree and with
    MATRIS_TEST_REFERENCE_TREE is bitwise identical in the eager,
    model_graph and whole_step paths (states and logged frames).

    The reference must share the graph builders' coordinate transforms,
    e.g. a ``git archive`` of the commit before ``nvt_nhc``: trees before
    fastmd.graph.geometry.small_matmul compute fractional coordinates with
    cuBLAS (TF32 by default in NGC containers) and differ numerically.
    """

    def test_trajectories_match_reference_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ours = _run_regression_worker(REPO, Path(tmp) / "ours.npz")
            reference = _run_regression_worker(Path(REFERENCE_TREE), Path(tmp) / "ref.npz")
        self.assertEqual(sorted(ours), sorted(reference))
        for key in sorted(ours):
            with self.subTest(key=key):
                self.assertEqual(ours[key].shape, reference[key].shape)
                np.testing.assert_array_equal(ours[key], reference[key])


@unittest.skipUnless(
    CUDA and op_available and _fixed_builder_available and real_model_assets_available(),
    "MatRIS-10M-OAM checkpoint / S01 frames unavailable",
)
@unittest.skipIf(NoseHooverChainNVT is None, "ASE NoseHooverChainNVT unavailable")
class MatRISS01NoseHooverVersusASETest(unittest.TestCase):
    """Tier 2 (opt-in via the real-model assets). The Q3 validation: GPU forces
    replayed through ASE NoseHooverChainNVT."""

    DT_FS, TDAMP_FS = 0.25, 25.0

    @classmethod
    def setUpClass(cls) -> None:
        from fastmd.models.matris.model import MatRIS

        frames, cls.temperature = load_s01_frames()
        for atoms in frames:
            atoms.set_momenta(
                maxwell_boltzmann_momenta(atoms.get_masses(), cls.temperature, seed=0)
            )
        cls.frames = frames
        cls.atoms = frames[0]
        cls.model = MatRIS.load(model_path=str(CHECKPOINT), device="cuda")

    def run_gpu(self, n_steps: int, execution: str, capacity: CapacityConfig | None = None,
                *, atoms: Atoms | None = None, expert_overrides=None):
        md = GPUResidentMolecularDynamics(
            (atoms or self.atoms).copy(), model=self.model, task="ef", ensemble="nvt_nhc",
            temperature=self.temperature, timestep=self.DT_FS, tdamp=self.TDAMP_FS,
            config=InferenceConfig(
                execution=execution, optimization_profile="topology", checkpoint="off",
                capacity=capacity or CapacityConfig(),
                expert_overrides=expert_overrides or {}, report=False,
            ),
        )
        try:
            md._ensure_forces()
            flat = [md.integrator.state.flat.cpu().numpy().copy()]
            for _ in range(n_steps):
                md.step()
                flat.append(md.integrator.state.flat.cpu().numpy().copy())
            stats = md.whole_step_stats()
        finally:
            md.close()
        return _flat_to_record(np.array(flat), len(self.atoms), 3), stats

    def assert_matches_ase(self, gpu: dict, n_steps: int, atoms: Atoms | None = None) -> dict:
        replay = Replay(gpu["q"], gpu["f"].astype(np.float32), gpu["e"])
        ref = run_ase(atoms or self.atoms, replay, n_steps, timestep=self.DT_FS * units.fs,
                      temperature=self.temperature, tdamp=self.TDAMP_FS * units.fs)
        self.assertEqual(replay.calls, n_steps + 1)
        self.assertLess(replay.max_dq, 1e-12 * np.abs(gpu["q"]).max())
        deviation = max_relative(ref, gpu)
        for key in KEYS:
            self.assertLess(deviation[key], 1e-12, (key, deviation))
        return deviation

    def test_gpu_trajectories_match_ase(self) -> None:
        tier_cache = {
            "MATRIS_WHOLE_STEP_TIER_CACHE": True,
            "MATRIS_TIER_HYSTERESIS_WINDOW": 30,
            "MATRIS_TIER_HYSTERESIS_SIGMA": 0.0,
            "MATRIS_TIER_CHECK_INTERVAL": 10,
            "MATRIS_TIER_DOWN_COOLDOWN": 0,
        }
        # name: (execution, steps, capacity, frame, expert_overrides)
        cases = {
            "whole_step, default capacity": ("whole_step", 300, None, 0, None),
            # the edge demand of S01 rises past the tight initial capacity: the
            # overflowing window is rolled back (float64 state restored), the
            # capacity grows and the graph is recaptured
            "whole_step, capacity grow + recapture + rollback": (
                "whole_step", 300, CapacityConfig(u_step=16, min_pad_u=1), 0, None),
            # S01 frame 1000 with a tight tier ladder: the tier cache grows
            # (new tiers captured) and shrinks back to a cached tier
            # (runner.reset), all while sharing the float64 integrator state
            "whole_step, tier cache switches": (
                "whole_step", 400, CapacityConfig(u_step=8, min_pad_u=1), 1, tier_cache),
            # A 0.002 A skin is below twice the per-step displacement at 500 K,
            # so the candidate list is invalid in most windows even right after
            # a refresh: rollback (float64 state restored) -> candidate refresh
            # -> recovery replay (still invalid) -> rollback -> eager
            # re-execution with exact candidate rebuild.
            "whole_step, candidate fallback": (
                "whole_step", 100, CapacityConfig(candidate_skin=0.002), 0, None),
            # dynamic GPU graph builder + eager model / bucketed model CUDA graph
            "eager": ("eager", 60, None, 0, None),
            "model_graph": ("model_graph", 100, None, 0, None),
        }
        for name, (execution, n_steps, capacity, frame, overrides) in cases.items():
            with self.subTest(case=name):
                atoms = self.frames[frame]
                gpu, stats = self.run_gpu(n_steps, execution, capacity, atoms=atoms,
                                          expert_overrides=overrides)
                if execution == "whole_step":  # the whole-step graph ran
                    self.assertGreater(stats["launches"], 0, stats)
                if "capacity grow" in name:
                    self.assertGreaterEqual(stats["capacity_recaptures"], 1, stats)
                    self.assertEqual(stats["capacity_policy"]["overflow_events"],
                                     stats["capacity_recaptures"])
                if "tier cache" in name:
                    self.assertGreaterEqual(stats["tier_cache"]["switch_up"], 1, stats)
                    self.assertGreaterEqual(stats["tier_cache"]["switch_down"], 1, stats)
                if "candidate fallback" in name:
                    self.assertGreater(stats["fallback_windows"], 50, stats)
                    self.assertEqual(stats["recovery_replays"], stats["fallback_windows"])
                self.assert_matches_ase(gpu, n_steps, atoms=atoms)


if __name__ == "__main__":
    unittest.main()
