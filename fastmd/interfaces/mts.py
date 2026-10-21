"""Eager, fixed-cell conservative rRESPA reference integrator.

Evaluators accept positions in Angstrom and return (energy in eV, forces in
eV/Angstrom). Velocities use Angstrom/fs; masses use amu. No thermostat,
adaptive timestep, graph capture, or silent model switching is performed.
"""
from dataclasses import dataclass
import numpy as np
from ase import units

ACCELERATION_CONVERSION = units.fs ** 2  # eV/(Angstrom amu) -> Angstrom/fs^2


def kinetic_energy(masses, velocities):
    return float(0.5 * np.sum(np.asarray(masses)[:, None] * velocities**2)
                 / ACCELERATION_CONVERSION)


@dataclass
class MTSState:
    positions: np.ndarray
    velocities: np.ndarray
    fast_force: np.ndarray
    residual_force: np.ndarray
    target_energy: float
    time_fs: float = 0.0


class RESPA:
    """Symmetric residual-kick / inner velocity-Verlet / residual-kick."""
    def __init__(self, fast, target, masses, timestep_fs, inner_steps):
        if timestep_fs <= 0 or int(inner_steps) != inner_steps or inner_steps < 1:
            raise ValueError('Positive timestep and positive integer inner_steps required')
        self.fast, self.target = fast, target
        self.masses = np.asarray(masses, dtype=float)
        if np.any(self.masses <= 0):
            raise ValueError('Masses must be positive')
        self.h, self.m = float(timestep_fs), int(inner_steps)
        self.accel = ACCELERATION_CONVERSION / self.masses[:, None]

    def initialize(self, positions, velocities):
        q, v = np.array(positions, dtype=float, copy=True), np.array(velocities, dtype=float, copy=True)
        _, fs = self.fast(q)
        et, ft = self.target(q)
        return MTSState(q, v, fs.copy(), ft-fs, float(et))

    def step(self, state):
        # Do not mutate the caller's state if an evaluation fails mid-step.
        q, v, fs = state.positions.copy(), state.velocities.copy(), state.fast_force.copy()
        H = self.h*self.m
        v += 0.5*H*self.accel*state.residual_force
        for _ in range(self.m):
            v += 0.5*self.h*self.accel*fs
            q += self.h*v
            if not np.isfinite(q).all():
                raise FloatingPointError('Nonfinite position in rRESPA')
            _, fs = self.fast(q)
            v += 0.5*self.h*self.accel*fs
        et, ft = self.target(q)
        fd = ft-fs
        v += 0.5*H*self.accel*fd
        if not np.isfinite(v).all():
            raise FloatingPointError('Nonfinite velocity in rRESPA')
        return MTSState(q, v, fs, fd, float(et), state.time_fs+H)


def verlet_step(evaluator, positions, velocities, forces, masses, timestep_fs):
    accel = ACCELERATION_CONVERSION / np.asarray(masses)[:, None]
    v = velocities + 0.5*timestep_fs*accel*forces
    q = positions + timestep_fs*v
    e, f = evaluator(q)
    v += 0.5*timestep_fs*accel*f
    return q, v, float(e), f
