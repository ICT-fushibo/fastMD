"""GPU port of ASE 3.29 ``NoseHooverChainNVT`` and ``MaxwellBoltzmannDistribution``.

Used by :class:`~fastmd.interfaces.gpu_md.GPUResidentMolecularDynamics`
with ``ensemble="nvt_nhc"`` in the dynamic, model-graph and whole-step CUDA
graph paths. The matbench-discovery MD protocol, for example, is::

    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(0))
    NoseHooverChainNVT(atoms, timestep=0.25 * units.fs, temperature_K=T,
                       tdamp=25 * units.fs)             # tchain=3, tloop=1

Design:

* All integrator state (positions, momenta, cached forces, thermostat
  ``eta``/``p_eta``, potential energy) lives in ONE preallocated float64
  device buffer (views into it), so a transactional backup/restore is a
  single ``copy_``.
* A step is split into :meth:`GPUNoseHooverChainNVT.step_pre_force` (NHC half
  step, kick with the cached F(q_n), drift) and
  :meth:`GPUNoseHooverChainNVT.step_post_force` (kick with F(q_{n+1}), NHC half
  step, cache the forces). ASE evaluates forces exactly once per step (its
  first ``_integrate_p`` hits the calculator cache), so
  pre -> model -> post reproduces ``NoseHooverChainNVT.step()``.
* No host syncs, no data-dependent Python control flow, no allocation of
  state: both halves are CUDA-graph capturable.

Two numerically equivalent backends:

* ``"triton"`` (default on CUDA, 3N <= 16384): one single-CTA fused kernel per
  half (2 launches per step). FP contraction is disabled
  (``enable_fp_fusion=False``) so every elementwise op is rounded exactly like
  NumPy. Thermostat variables live in registers as fp64 scalars; the only
  reduction is sum(p**2/m). Optional ``fast_math=True`` uses reciprocal
  multiplies instead of the fp64 divisions and allows FMA contraction
  (<= 1 ulp per affected op vs ASE).
* ``"torch"``: op-by-op reference (several hundred tiny kernels per step),
  used for validation and as fallback for large systems or CPU tensors.

The order of floating-point operations follows ASE line by line. The only
intended differences to ASE are (a) the reduction order of sum(p**2/m)
(NumPy pairwise vs GPU tree), (b) CUDA ``exp`` vs glibc ``exp`` (last bit
for a few percent of arguments) and (c) NumPy scalar ``x**2`` vs ``x*x``,
i.e. ~1e-16 relative per operation. With ``force_dtype=torch.float32`` the
kick ``dt/2 * F`` is formed in float32 exactly like NumPy 2 (NEP 50) does
for the float32 forces a float32 calculator returns; the state stays
float64.

Heating ramps: ``set_temperature`` recomputes kT, Q and 3N*kT with the
constructor's expressions (:func:`nhc_constants`) and rewrites the device
constants in place, stream ordered and without a host sync; both backends
read them from device memory, so a captured graph sees the new target at
its next replay.

One class, two settings: ``protocol`` is how the thermostat scalars are
formed, ``backend`` which kernels run. ``protocol="single"`` (P=1, the
default without a ``comm``) is the single-CTA ASE port above.
``protocol="decomposed"`` is the same thermostat for atoms partitioned over
ranks (unified engine steps 3.3 and 3.5): the integrator keeps this rank's
rows in capacity slots, the chain is replicated, and one half step is a
local sum(p**2/m), one ``allreduce_sum_``, a single-program scalar kernel,
one broadcast ``allreduce_sum_`` and a multi-CTA elementwise pass
(:class:`_DecomposedChain`). The evaluator-driven MD driver
(``GPUResidentMolecularDynamics.from_evaluator``) uses it at every P.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ase import units
from torch import Tensor

from fastmd.execution.gpu_integrator import StatefulIntegrator
from fastmd.execution.gpu_integrator import _PinnedUpload
from fastmd.execution.gpu_integrator import slot_capacity

try:  # Triton is optional; the torch backend works everywhere
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None
    _HAS_TRITON = False

# Identical Python expressions to ase/md/nose_hoover_chain.py -> identical floats
FOURTH_ORDER_COEFFS = [
    1 / (2 - 2 ** (1 / 3)),
    -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
    1 / (2 - 2 ** (1 / 3)),
]

_TRITON_MAX_ELEMS = 16384  # single-CTA register-resident limit (3N)


@dataclass(frozen=True)
class NHCConstants:
    """ASE's thermostat scalars; ``values`` is the device constants layout
    (see ``_C_Q``). ``sub``: per Suzuki-Yoshida coefficient ``(delta,
    delta/2, -(delta/4), -delta)``."""

    kT: float
    ndof_kT: float
    Q: np.ndarray
    dt2: float
    sub: list
    values: list


def nhc_constants(n_atoms: int, *, temperature_K: float, timestep: float, tdamp: float,
                  tchain: int, tloop: int) -> NHCConstants:
    """The exact Python/NumPy expressions of ASE (equal arguments -> equal
    bits). ``n_atoms`` is the global atom count (DOF and Q[0])."""
    n = int(n_atoms)
    kT = units.kB * temperature_K
    Q = np.zeros(tchain)
    Q[0] = 3 * n * kT * tdamp**2
    Q[1:] = kT * tdamp**2
    ndof_kT = 3 * n * kT
    dt2 = timestep / 2
    sub = []
    for coeff in FOURTH_ORDER_COEFFS:
        delta = coeff * dt2 / tloop
        sub.append((delta, delta / 2, -(delta / 4), -delta))
    values = [timestep, dt2, kT, ndof_kT]
    for entry in sub:
        values.extend(entry)
    values.extend(Q.tolist())
    values.extend((1.0 / Q).tolist())  # fast_math only
    return NHCConstants(kT, ndof_kT, Q, dt2, sub, values)


class _TargetTemperature:
    """kT, Q, 3N*kT, the device constants and the restart state (eta, p_eta,
    target) of both NHC classes."""

    def _init_constants(self, n_atoms: int) -> None:
        self._dof_atoms = int(n_atoms)
        c = self._constants(self.temperature_K)
        self._keep(c, self.temperature_K)
        self.consts = torch.tensor(c.values, dtype=torch.float64, device=self.device)
        self.Q = self.consts[_C_Q: _C_Q + self.tchain]
        self._upload = _PinnedUpload(self.consts)

    def _constants(self, temperature_K: float) -> NHCConstants:
        return nhc_constants(self._dof_atoms, temperature_K=temperature_K, timestep=self.dt,
                             tdamp=self.tdamp, tchain=self.tchain, tloop=self.tloop)

    def _keep(self, c: NHCConstants, temperature_K: float) -> None:
        self.temperature_K = temperature_K
        self.kT, self.ndof_kT, self.Q_np, self.dt2, self.sub = c.kT, c.ndof_kT, c.Q, c.dt2, c.sub

    def set_temperature(self, temperature_K: float) -> None:
        """New target temperature (heating ramp): kT, Q and 3N*kT in place.

        Same expressions as the constructor, so ``set_temperature(T)`` equals
        a fresh integrator at ``T`` bitwise (and the current ``T`` changes no
        bit). Sync-free and stream ordered; call between steps or graph
        replays, not inside a capture.
        """
        temperature_K = float(temperature_K)
        c = self._constants(temperature_K)
        self._upload(c.values)  # raises inside a capture: host state untouched
        self._keep(c, temperature_K)

    def target_row(self, temperature_K: float) -> list[float]:
        """The device constants of ``temperature_K`` as one row of a multi-step window's per-step targets."""
        return list(self._constants(float(temperature_K)).values)

    def load_target_row(self, row) -> None:
        """Load a :meth:`target_row` from the device (inside a captured multi-step window, before each
        step): the constants ``set_temperature`` would upload (bitwise). The host mirror (kT, Q, ...)
        is the caller's (``set_temperature`` of the window's last target)."""
        self.consts.copy_(row)

    # ---------------------------------------------------------------- restart
    def state_dict(self) -> dict:
        """Replicated thermostat state for restart files (host readback)."""
        return {
            "eta": self.eta.detach().cpu().numpy().copy(),
            "p_eta": self.p_eta.detach().cpu().numpy().copy(),
            "temperature_K": self.temperature_K,
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        eta = torch.as_tensor(np.asarray(state["eta"], dtype=np.float64))
        p_eta = torch.as_tensor(np.asarray(state["p_eta"], dtype=np.float64))
        if eta.shape != (self.tchain,) or p_eta.shape != (self.tchain,):
            raise ValueError(f"eta/p_eta must have shape ({self.tchain},)")
        self.set_temperature(float(state["temperature_K"]))
        self.eta.copy_(eta)
        self.p_eta.copy_(p_eta)


def maxwell_boltzmann_momenta(masses, temperature_K: float, seed=0) -> np.ndarray:
    """Bit-identical ``MaxwellBoltzmannDistribution(atoms, temperature_K=T,
    rng=np.random.default_rng(seed))`` momenta (ASE 3.29).

    ASE path: ``thermalize_momenta`` (``exact_temperature=False``, no
    Stationary/ZeroRotation): ``xi = rng.standard_normal((N, 3))``;
    ``p = xi * sqrt(masses * kB * T)[:, None]``. ``masses`` =
    ``atoms.get_masses()`` in amu; ``seed`` is an int or a
    ``np.random.Generator`` (consumed exactly like ASE consumes it).
    """
    masses = np.asarray(masses, dtype=np.float64).reshape(-1)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    temp = units.kB * temperature_K
    xi = rng.standard_normal((len(masses), 3))
    return xi * np.sqrt(masses * temp)[:, np.newaxis]


@dataclass
class NHCState:
    """Views into one flat float64 buffer ``flat``.

    positions/momenta/forces: (N, 3); eta/p_eta: (tchain,); potential_energy:
    (1,). ``forces`` always holds F(positions) (the ASE calculator cache).
    Layout ``[q | p | F | eta | p_eta | s (scratch) | E]``. ``scratch``
    (decomposed protocol: the 3*tloop chain scale factors) is 0 at P=1, so
    the layout is the P=1 one; with scratch, ``chain`` is the contiguous
    view ``[eta | p_eta | s]`` (else None).
    """

    flat: Tensor
    positions: Tensor
    momenta: Tensor
    forces: Tensor
    eta: Tensor
    p_eta: Tensor
    potential_energy: Tensor
    chain: Tensor | None = None

    @staticmethod
    def allocate(n_atoms: int, tchain: int, device, scratch: int = 0) -> "NHCState":
        n3 = 3 * n_atoms
        flat = torch.zeros(3 * n3 + 2 * tchain + scratch + 1, dtype=torch.float64, device=device)
        offset = 0
        positions = flat[offset: offset + n3].view(n_atoms, 3)
        offset += n3
        momenta = flat[offset: offset + n3].view(n_atoms, 3)
        offset += n3
        forces = flat[offset: offset + n3].view(n_atoms, 3)
        offset += n3
        eta = flat[offset: offset + tchain]
        offset += tchain
        p_eta = flat[offset: offset + tchain]
        offset += tchain
        chain = None
        if scratch:
            chain = flat[offset - 2 * tchain: offset + scratch]
            offset += scratch
        energy = flat[offset: offset + 1]
        return NHCState(flat, positions, momenta, forces, eta, p_eta, energy, chain)

    def numpy(self) -> dict:
        return {
            "positions": self.positions.detach().cpu().numpy().copy(),
            "momenta": self.momenta.detach().cpu().numpy().copy(),
            "forces": self.forces.detach().cpu().numpy().copy(),
            "eta": self.eta.detach().cpu().numpy().copy(),
            "p_eta": self.p_eta.detach().cpu().numpy().copy(),
            "potential_energy": float(self.potential_energy.item()),
        }


# --------------------------------------------------------------------------- #
# Triton fused kernels
# --------------------------------------------------------------------------- #
# constants buffer (float64): dt, dt/2, kT, 3N*kT, 3 x (delta, delta/2,
# -(delta/4), -delta), Q[tchain], 1/Q[tchain]
_C_Q = 16

if _HAS_TRITON:
    # Thermostat variables are carried as Triton tuples of fp64 scalars
    # (uniform, thread-redundant); only sum(p**2/m) is a block reduction.

    @triton.jit
    def _tset(t, J: tl.constexpr, v):
        return t[:J] + (v,) + t[J + 1:]

    @triton.jit
    def _tload(ptr, M: tl.constexpr):
        t = ()
        for j in tl.static_range(M):
            t = t + (tl.load(ptr + j),)
        return t

    @triton.jit
    def _tstore(ptr, t, M: tl.constexpr):
        for j in tl.static_range(M):
            tl.store(ptr + j, t[j])

    @triton.jit
    def _qdiv(x, Q, J: tl.constexpr, FAST: tl.constexpr):
        # exact: x / Q[J] (as ASE); fast: x * (1/Q[J]) (Q tuple holds inverses)
        if FAST:
            r = x * Q[J]
        else:
            r = x / Q[J]
        return r

    @triton.jit
    def _mdiv(x, m, FAST: tl.constexpr):
        # exact: x / m (as ASE); fast: x * (1/m) (m holds inverse masses)
        if FAST:
            r = x * m
        else:
            r = x / m
        return r

    @triton.jit
    def _p_eta_j(ke2, pe, Q, kT, ndof_kT, delta2, neg_delta4,
                 J: tl.constexpr, M: tl.constexpr, FAST: tl.constexpr):
        # ASE _integrate_p_eta_j; ke2 = sum(p**2 / m) of the current p (J == 0)
        pj = pe[J]
        if J < M - 1:
            s = libdevice.exp(_qdiv(neg_delta4 * pe[J + 1], Q, J + 1, FAST))
            pj = pj * s
        if J == 0:
            g = ke2 - ndof_kT
        else:
            g = _qdiv(pe[J - 1] * pe[J - 1], Q, J - 1, FAST) - kT
        pj = pj + delta2 * g
        if J < M - 1:
            pj = pj * s  # p_eta[j+1] unchanged in between -> identical factor
        return _tset(pe, J, pj)

    @triton.jit
    def _nhc_half(p, m, pe, eta, Q, c_ptr, M: tl.constexpr, TLOOP: tl.constexpr,
                  FAST: tl.constexpr):
        # ASE integrate_nhc with _integrate_nhc_loop
        kT = tl.load(c_ptr + 2)
        ndof_kT = tl.load(c_ptr + 3)
        # sum(p**2 / masses) only changes when p changes: once at entry and
        # after every p scaling (the backward sweep reuses the same value).
        ke2 = tl.sum(_mdiv(p * p, m, FAST), axis=0)
        for _t in tl.static_range(TLOOP):
            for k in tl.static_range(3):
                delta = tl.load(c_ptr + 4 + 4 * k)
                delta2 = tl.load(c_ptr + 5 + 4 * k)
                neg_delta4 = tl.load(c_ptr + 6 + 4 * k)
                neg_delta = tl.load(c_ptr + 7 + 4 * k)
                for jj in tl.static_range(M):
                    pe = _p_eta_j(ke2, pe, Q, kT, ndof_kT, delta2, neg_delta4,
                                  M - 1 - jj, M, FAST)
                for j in tl.static_range(M):
                    eta = _tset(eta, j, eta[j] + _qdiv(delta * pe[j], Q, j, FAST))
                p = p * libdevice.exp(_qdiv(neg_delta * pe[0], Q, 0, FAST))
                ke2 = tl.sum(_mdiv(p * p, m, FAST), axis=0)
                for j in tl.static_range(M):
                    pe = _p_eta_j(ke2, pe, Q, kT, ndof_kT, delta2, neg_delta4, j, M,
                                  FAST)
        return p, pe, eta

    @triton.jit
    def _nhc_pre_force_kernel(q_ptr, p_ptr, f_ptr, m_ptr, eta_ptr, pe_ptr, c_ptr,
                              n_elem, BLOCK: tl.constexpr, M: tl.constexpr,
                              TLOOP: tl.constexpr, F32_KICK: tl.constexpr,
                              FAST: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_elem
        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        m = tl.load(m_ptr + offs // 3, mask=mask, other=1.0)
        pe = _tload(pe_ptr, M)
        eta = _tload(eta_ptr, M)
        if FAST:
            Q = _tload(c_ptr + 16 + M, M)  # 1/Q
        else:
            Q = _tload(c_ptr + 16, M)
        # (1) p = integrate_nhc(p, dt/2)
        p, pe, eta = _nhc_half(p, m, pe, eta, Q, c_ptr, M, TLOOP, FAST)
        # (2) p += dt/2 * F(q_n) (cached forces)
        dt = tl.load(c_ptr + 0)
        dt2 = tl.load(c_ptr + 1)
        f = tl.load(f_ptr + offs, mask=mask, other=0.0)
        if F32_KICK:  # NumPy NEP50: python float * float32 array -> float32
            kick = (dt2.to(tl.float32) * f.to(tl.float32)).to(tl.float64)
        else:
            kick = dt2 * f
        p = p + kick
        # (3) q += dt * p / m
        q = tl.load(q_ptr + offs, mask=mask, other=0.0)
        q = q + _mdiv(dt * p, m, FAST)
        tl.store(q_ptr + offs, q, mask=mask)
        tl.store(p_ptr + offs, p, mask=mask)
        _tstore(pe_ptr, pe, M)
        _tstore(eta_ptr, eta, M)

    @triton.jit
    def _nhc_post_force_kernel(p_ptr, fin_ptr, fcache_ptr, m_ptr, eta_ptr, pe_ptr,
                               c_ptr, ein_ptr, eout_ptr, n_elem,
                               BLOCK: tl.constexpr, M: tl.constexpr,
                               TLOOP: tl.constexpr, F32_KICK: tl.constexpr,
                               HAS_E: tl.constexpr, FAST: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_elem
        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        m = tl.load(m_ptr + offs // 3, mask=mask, other=1.0)
        pe = _tload(pe_ptr, M)
        eta = _tload(eta_ptr, M)
        if FAST:
            Q = _tload(c_ptr + 16 + M, M)  # 1/Q
        else:
            Q = _tload(c_ptr + 16, M)
        dt2 = tl.load(c_ptr + 1)
        f_in = tl.load(fin_ptr + offs, mask=mask, other=0.0)
        f64 = f_in.to(tl.float64)
        # (4) p += dt/2 * F(q_{n+1})
        if F32_KICK:
            kick = (dt2.to(tl.float32) * f_in.to(tl.float32)).to(tl.float64)
        else:
            kick = dt2 * f64
        p = p + kick
        # (5) p = integrate_nhc(p, dt/2)
        p, pe, eta = _nhc_half(p, m, pe, eta, Q, c_ptr, M, TLOOP, FAST)
        tl.store(p_ptr + offs, p, mask=mask)
        tl.store(fcache_ptr + offs, f64, mask=mask)
        _tstore(pe_ptr, pe, M)
        _tstore(eta_ptr, eta, M)
        if HAS_E:
            tl.store(eout_ptr, tl.load(ein_ptr).to(tl.float64))

    # ---- protocol="decomposed": reduction, replicated chain, rows ----
    # n_elem (3 x owned rows) changes with every migration: a value-specialized
    # int (==1, %16==0) compiled a second variant inside the first refresh step
    # (+243 ms). Specialization only changes vectorization, never values.

    @triton.jit(do_not_specialize=["n_elem"])
    def _nhc_ke2_kernel(p_ptr, m_ptr, real_ptr, out_ptr, n_elem,
                        BLOCK: tl.constexpr, HAS_MASK: tl.constexpr):
        # local sum(p**2 / m) of the real rows; one program, fixed order
        acc = tl.zeros([BLOCK], dtype=tl.float64)
        for start in range(0, n_elem, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < n_elem
            if HAS_MASK:
                mask = mask & (tl.load(real_ptr + offs // 3, mask=mask, other=0) != 0)
            p = tl.load(p_ptr + offs, mask=mask, other=0.0)
            m = tl.load(m_ptr + offs // 3, mask=mask, other=1.0)
            acc += p * p / m
        tl.store(out_ptr, tl.sum(acc, axis=0))

    @triton.jit
    def _nhc_chain_kernel(ke2_ptr, x_ptr, c_ptr, M: tl.constexpr, TLOOP: tl.constexpr,
                          STAGE: tl.constexpr, ZERO: tl.constexpr):
        # x = [eta (M), p_eta (M), s (3*TLOOP)]; the chain of _nhc_half with a
        # given global ke2. STAGE < 0: the whole integrate_nhc, ke2 <- ke2*s*s
        # after each scaling. STAGE = i (recompute): forward sweep of sub-step
        # i-1 and backward sweep, eta and s of sub-step i (the caller scales p
        # and re-reduces ke2 in between). ZERO: contribute zeros to the
        # broadcast from rank 0.
        if ZERO:
            for i in tl.static_range(2 * M + 3 * TLOOP):
                tl.store(x_ptr + i, 0.0)
        else:
            ke2 = tl.load(ke2_ptr)
            kT = tl.load(c_ptr + 2)
            ndof_kT = tl.load(c_ptr + 3)
            eta = _tload(x_ptr, M)
            pe = _tload(x_ptr + M, M)
            Q = _tload(c_ptr + 16, M)
            for t in tl.static_range(TLOOP):
                for k in tl.static_range(3):
                    delta = tl.load(c_ptr + 4 + 4 * k)
                    delta2 = tl.load(c_ptr + 5 + 4 * k)
                    neg_delta4 = tl.load(c_ptr + 6 + 4 * k)
                    neg_delta = tl.load(c_ptr + 7 + 4 * k)
                    if STAGE < 0 or STAGE == 3 * t + k:
                        for jj in tl.static_range(M):
                            pe = _p_eta_j(ke2, pe, Q, kT, ndof_kT, delta2, neg_delta4,
                                          M - 1 - jj, M, False)
                        for j in tl.static_range(M):
                            eta = _tset(eta, j, eta[j] + _qdiv(delta * pe[j], Q, j, False))
                        s = libdevice.exp(_qdiv(neg_delta * pe[0], Q, 0, False))
                        tl.store(x_ptr + 2 * M + 3 * t + k, s)
                        if STAGE < 0:
                            ke2 = ke2 * s * s
                    if STAGE < 0 or STAGE == 3 * t + k + 1:
                        for j in tl.static_range(M):
                            pe = _p_eta_j(ke2, pe, Q, kT, ndof_kT, delta2, neg_delta4, j, M,
                                          False)
            _tstore(x_ptr, eta, M)
            _tstore(x_ptr + M, pe, M)

    @triton.jit(do_not_specialize=["n_elem"])
    def _nhc_rows_kernel(q_ptr, p_ptr, f_ptr, m_ptr, real_ptr, s_ptr, c_ptr, n_elem,
                         BLOCK: tl.constexpr, NSCALE: tl.constexpr, KICK: tl.constexpr,
                         DRIFT: tl.constexpr, F32_KICK: tl.constexpr,
                         HAS_MASK: tl.constexpr):
        # p *= s_1 ... s_NSCALE; p += dt/2*F; q += dt*p/m, per element in the
        # op order of _nhc_pre_force_kernel / _nhc_post_force_kernel
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        if HAS_MASK:
            mask = mask & (tl.load(real_ptr + offs // 3, mask=mask, other=0) != 0)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        for k in tl.static_range(NSCALE):
            p = p * tl.load(s_ptr + k)
        if KICK:
            dt2 = tl.load(c_ptr + 1)
            f = tl.load(f_ptr + offs, mask=mask, other=0.0)
            if F32_KICK:  # NumPy NEP50: python float * float32 array -> float32
                kick = (dt2.to(tl.float32) * f.to(tl.float32)).to(tl.float64)
            else:
                kick = dt2 * f.to(tl.float64)
            p = p + kick
        if DRIFT:
            dt = tl.load(c_ptr + 0)
            m = tl.load(m_ptr + offs // 3, mask=mask, other=1.0)
            q = tl.load(q_ptr + offs, mask=mask, other=0.0)
            q = q + dt * p / m
            tl.store(q_ptr + offs, q, mask=mask)
        tl.store(p_ptr + offs, p, mask=mask)


class _DecomposedChain:
    """One NHC step's thermostat for rows partitioned over ranks.

    The algorithm of ``protocol="decomposed"``: ASE ``NoseHooverChainNVT``
    for atoms partitioned over ranks. The per-atom rows are this rank's;
    ``eta``/``p_eta`` are replicated. DOF and Q use the owner's ``n_global``
    (the same :func:`nhc_constants` as ``protocol="single"``, so at P=1 the
    constants are identical). One NHC half step (ASE ``integrate_nhc(dt/2)``):

    1. local sum(p**2/m) of the real rows -> ``comm.allreduce_sum_`` (ke2);
    2. the chain in ASE order for tloop x 3 Suzuki-Yoshida sub-steps in one
       single-program kernel. ``ke2_update="scaled"`` (Q13 default) carries
       ke2 <- ke2*s*s across each scaling instead of re-reducing;
    3. ``scalar_sync="broadcast"``: rank 0's [eta, p_eta, s_1..s_K] reach all
       ranks through ONE ``allreduce_sum_`` to which the other ranks
       contribute zeros (x + 0 is exact), so the chain and the scaling are
       bitwise replicated. ``"local"``: every rank keeps its own result (the
       Q16 measurement). No broadcast at world size 1;
    4. elementwise p *= s_1; p *= s_2; ... (the P=1 kernel's op order).

    Two collectives per half step and no host syncs; every rank issues the
    same collectives in the same order (also with 0 rows).
    ``ke2_update="recompute"`` (diagnostic): ASE's algorithm verbatim, p is
    scaled and sum(p**2/m) re-reduced after every scaling (3*tloop extra
    collectives, scalars computed locally; ``"broadcast"`` then replicates
    only the final eta/p_eta). The row count is a runtime kernel argument
    that is not value-specialized, so after the owner's ``warmup_compile``
    no row count (migration) compiles again.

    Composed into its owner (:class:`GPUNoseHooverChainNVT`), which provides
    ``comm``, ``consts``, ``Q``, ``sub``, ``dt``, ``dt2``, ``tchain``,
    ``tloop``, ``f32_kick``, ``backend`` and ``device``; they are read at
    every call, so ``set_temperature`` and a temporarily swapped ``comm``
    (warmup) take effect. ``chain`` is the replicated
    ``[eta | p_eta | s_1..s_K]`` (K = 3*tloop), also the broadcast vector.

    Rows per call: positions/momenta contiguous float64 (n, 3), updated in
    place; masses float64 (n,); forces (n, 3) contiguous (float32 or
    float64); ``real``: optional bool (n,) (False rows are neither reduced
    nor updated).
    """

    _KE2_BLOCK = 2048
    _ROWS_BLOCK = 1024

    def __init__(self, owner, chain: Tensor, *, scalar_sync: str, ke2_update: str) -> None:
        self.owner = owner
        self.ke2_update = ke2_update
        self._n_sub = 3 * owner.tloop
        self.bind(chain)
        self._ke2 = torch.zeros(1, dtype=torch.float64, device=owner.device)
        comm = owner.comm
        self._broadcast = scalar_sync == "broadcast" and int(comm.world_size) > 1
        self._zero = self._broadcast and int(comm.rank) != 0

    def bind(self, chain: Tensor) -> None:
        """Use ``chain`` (a new buffer after slot growth) from now on."""
        M = self.owner.tchain
        self._chain = chain
        self.eta = chain[:M]
        self.p_eta = chain[M: 2 * M]
        self._s = chain[2 * M:]

    def pre_force(self, q: Tensor, p: Tensor, f: Tensor, m: Tensor,
                  real: Tensor | None = None) -> None:
        """NHC(dt/2); p += dt/2*F(q_n); q += dt*p/m (``f`` = F(q_n))."""
        pending = self._integrate_nhc(p, m, real)
        self._update_rows(q, p, f, m, real, 0, pending, kick=True, drift=True)

    def post_force(self, p: Tensor, f: Tensor, m: Tensor, real: Tensor | None = None) -> None:
        """p += dt/2*F(q_{n+1}); NHC(dt/2)."""
        self._update_rows(None, p, f, m, real, 0, 0, kick=True, drift=False)
        pending = self._integrate_nhc(p, m, real)
        if pending:
            self._update_rows(None, p, None, m, real, 0, pending, kick=False, drift=False)

    def _integrate_nhc(self, p: Tensor, m: Tensor, real: Tensor | None) -> int:
        """Collectives and chain of one integrate_nhc(dt/2); returns how many
        factors of ``_s`` the caller's elementwise pass still applies to p."""
        comm = self.owner.comm
        self._local_ke2(p, m, real)
        comm.allreduce_sum_(self._ke2)
        if self.ke2_update == "scaled":
            self._chain_update(-1, self._zero)
            if self._broadcast:
                comm.allreduce_sum_(self._chain)
            return self._n_sub
        for i in range(self._n_sub):
            self._chain_update(i, False)
            self._update_rows(None, p, None, m, real, i, 1, kick=False, drift=False)
            self._local_ke2(p, m, real)
            comm.allreduce_sum_(self._ke2)
        self._chain_update(self._n_sub, self._zero)
        if self._broadcast:
            comm.allreduce_sum_(self._chain)
        return 0

    def _local_ke2(self, p: Tensor, m: Tensor, real: Tensor | None) -> None:
        n_elem = p.numel()
        if self.owner.backend == "torch" or n_elem == 0:
            with torch.no_grad():
                x = p * p / m.reshape(-1, 1)
                if real is not None:
                    x = torch.where(real.reshape(-1, 1), x, 0.0)
                self._ke2.copy_(x.sum().reshape(1))
            return
        _nhc_ke2_kernel[(1,)](
            p, m, p if real is None else real.view(torch.uint8), self._ke2, n_elem,
            BLOCK=self._KE2_BLOCK, HAS_MASK=real is not None,
            num_warps=8, enable_fp_fusion=False,
        )

    def _chain_update(self, stage: int, zero: bool) -> None:
        o = self.owner
        if o.backend == "triton":
            _nhc_chain_kernel[(1,)](
                self._ke2, self._chain, o.consts, M=o.tchain, TLOOP=o.tloop,
                STAGE=stage, ZERO=zero, num_warps=1, enable_fp_fusion=False,
            )
            return
        with torch.no_grad():
            if zero:
                self._chain.zero_()
                return
            ke2, eta, pe, Q = self._ke2[0], self.eta, self.p_eta, o.Q
            M = o.tchain
            for i in range(self._n_sub):
                delta, delta2, neg_delta4, neg_delta = o.sub[i % 3]
                if stage < 0 or stage == i:
                    for j in range(M):
                        self._chain_p_eta_j(ke2, M - j - 1, delta2, neg_delta4)
                    eta.add_(pe * delta / Q)
                    s = torch.exp(pe[0] * neg_delta / Q[0])
                    self._s[i].copy_(s)
                    if stage < 0:
                        ke2 = ke2 * s * s
                if stage < 0 or stage == i + 1:
                    for j in range(M):
                        self._chain_p_eta_j(ke2, j, delta2, neg_delta4)

    def _chain_p_eta_j(self, ke2, j, delta2, neg_delta4) -> None:
        # GPUNoseHooverChainNVT._torch_p_eta_j with the given global ke2
        o = self.owner
        pe, Q, M = self.p_eta, o.Q, o.tchain
        if j < M - 1:
            s = torch.exp(pe[j + 1] * neg_delta4 / Q[j + 1])
            pe[j].mul_(s)
        if j == 0:
            g = ke2 - o.consts[3]
        else:
            g = pe[j - 1] * pe[j - 1] / Q[j - 1] - o.consts[2]
        pe[j].add_(g * delta2)
        if j < M - 1:
            pe[j].mul_(s)

    def _update_rows(self, q, p, f, m, real, first: int, nscale: int, *, kick: bool,
                     drift: bool) -> None:
        """p *= s[first:first+nscale] (sequentially); optional kick and drift."""
        n_elem = p.numel()
        if n_elem == 0 or not (nscale or kick or drift):
            return
        o = self.owner
        if o.backend == "triton":
            grid = (triton.cdiv(n_elem, self._ROWS_BLOCK),)
            _nhc_rows_kernel[grid](
                q if drift else p, p, f if kick else p, m,
                p if real is None else real.view(torch.uint8),
                self._s[first:] if nscale else self._s, o.consts, n_elem,
                BLOCK=self._ROWS_BLOCK, NSCALE=nscale, KICK=kick, DRIFT=drift,
                F32_KICK=o.f32_kick, HAS_MASK=real is not None,
                num_warps=4, enable_fp_fusion=False,
            )
            return
        with torch.no_grad():
            new_p = p
            for k in range(first, first + nscale):
                new_p = new_p * self._s[k]
            if kick:
                if o.f32_kick:  # NumPy NEP50 semantics for float32 forces
                    new_p = new_p + (f.to(torch.float32) * o.dt2).to(torch.float64)
                else:
                    new_p = new_p + f.to(torch.float64) * o.dt2
            keep = None if real is None else real.reshape(-1, 1)
            if drift:
                new_q = q + new_p * o.dt / m.reshape(-1, 1)
                q.copy_(new_q if keep is None else torch.where(keep, new_q, q))
            p.copy_(new_p if keep is None else torch.where(keep, new_p, p))


class GPUNoseHooverChainNVT(_TargetTemperature, StatefulIntegrator):
    """GPU/CUDA-graph friendly port of ``ase.md.nose_hoover_chain.NoseHooverChainNVT``.

    Parameters mirror ASE (ASE units): ``timestep`` and ``tdamp`` in ASE time
    units (e.g. ``0.25 * units.fs``), ``temperature_K`` in K, ``tchain``,
    ``tloop``. ``masses``: (N,) amu (``atoms.get_masses()``);
    ``positions``/``momenta``: (N, 3) (``atoms.get_positions()/get_momenta()``,
    use the float64 host arrays for ASE-exact trajectories). ``forces``:
    F(positions), may be given later via :meth:`set_forces`.
    ``force_dtype``: dtype of the model forces (MatRIS: float32, see the module
    docstring for the NumPy kick semantics). ``fast_math`` (triton only):
    reciprocal multiplies and FMA contraction (<= 1 ulp per op vs ASE).
    :meth:`set_temperature` changes the target between steps (heating ramp).

    Per step (eager or inside one CUDA graph)::

        q = nhc.step_pre_force()           # q_{n+1}, float64 (N, 3) view
        forces, energy = model(q)          # any float dtype
        nhc.step_post_force(forces, energy=energy)

    or ``positions, momenta, forces = nhc.step(force_fn)`` with
    ``force_fn(q) -> forces`` or ``(forces, energy)``.

    ``protocol``: ``"single"`` (P=1: the single-CTA kernels above, all N
    atoms) or ``"decomposed"`` (this rank's rows of atoms partitioned over
    ranks, the algorithm of :class:`_DecomposedChain`: local
    reductions, ``comm.allreduce_sum_`` and a replicated chain); ``"auto"``:
    single without a ``comm``, else decomposed. ``backend`` picks the kernels:
    ``"auto"`` is triton on CUDA (single: only for 3N <= 16384), else torch.
    Decomposed only: ``comm`` (the Communicator protocol; default a
    ``LocalCommunicator``), ``n_global`` (the global atom count for DOF, Q
    and the temperature; default the local count), ``capacity`` (rows per
    per-atom buffer, rounded up to an even count so every row view is 16-byte
    aligned; default n + max(64, n/8) rounded to 64), ``scalar_sync`` and
    ``ke2_update`` (as :class:`_DecomposedChain`) and ``fix_com``
    (the global momentum is removed after every step, one more
    ``allreduce_sum_``). Rows ``[0, n_owned)`` of the buffers are the owned
    atoms; ``positions``/``momenta``/``forces`` and the step halves use those
    row views, :meth:`positions32` is the float32 evaluator input,
    :meth:`migrating_fields` the payload that migrates with the atoms, and
    :meth:`reorder_` loads the rows an epoch refresh returns. Collective
    under decomposed (every rank calls): the step halves,
    :meth:`kinetic_energy`, :meth:`temperature`, :meth:`conserved_energy`.
    """

    ensemble = "nvt_nhc"

    def __init__(
        self,
        masses,
        positions,
        momenta,
        *,
        temperature_K: float,
        timestep: float,
        tdamp: float,
        tchain: int = 3,
        tloop: int = 1,
        forces=None,
        potential_energy=None,
        device="cuda",
        backend: str = "auto",
        force_dtype: torch.dtype = torch.float64,
        numpy_kick_semantics: bool = True,
        fast_math: bool = False,
        comm=None,
        n_global: int | None = None,
        capacity: int | None = None,
        protocol: str = "auto",
        scalar_sync: str = "broadcast",
        ke2_update: str = "scaled",
        fix_com: bool = False,
    ) -> None:
        if tchain < 1 or tloop < 1:
            raise ValueError("tchain and tloop must be >= 1")
        if tdamp <= 0 or timestep < 0:
            raise ValueError("tdamp must be positive and timestep non-negative")
        if scalar_sync not in ("broadcast", "local"):
            raise ValueError(f"unknown scalar_sync {scalar_sync!r}")
        if ke2_update not in ("scaled", "recompute"):
            raise ValueError(f"unknown ke2_update {ke2_update!r}")
        if protocol == "auto":
            protocol = "single" if comm is None else "decomposed"
        if protocol not in ("single", "decomposed"):
            raise ValueError(f"unknown protocol {protocol!r}")
        self.device = torch.device(device)
        masses_np = np.asarray(
            masses.detach().cpu().numpy() if torch.is_tensor(masses) else masses,
            dtype=np.float64,
        ).reshape(-1)
        self.n_atoms = n = int(masses_np.shape[0])
        self.tchain = int(tchain)
        self.tloop = int(tloop)
        self.temperature_K = float(temperature_K)
        self.dt = float(timestep)
        self.tdamp = float(tdamp)
        self.force_dtype = force_dtype
        self.f32_kick = bool(numpy_kick_semantics and force_dtype == torch.float32)
        self.protocol = protocol
        self.scalar_sync, self.ke2_update = scalar_sync, ke2_update  # decomposed only
        self.n_global = n if n_global is None else int(n_global)
        self.fix_com = bool(fix_com)
        self.generation = 0
        self.n_owned = n
        self._dc = None
        if protocol == "decomposed":
            self._init_decomposed(masses_np, positions, momenta, forces, potential_energy,
                                  comm=comm, capacity=capacity, backend=backend,
                                  fast_math=fast_math, scalar_sync=scalar_sync,
                                  ke2_update=ke2_update)
            return
        if comm is not None or self.fix_com or capacity not in (None, n) or self.n_global != n:
            raise ValueError("protocol='single' integrates all N atoms on one device: no comm, "
                             "capacity or n_global other than N, and no fix_com")
        self.comm = None
        self.capacity = n

        # scalars with the exact Python/NumPy expressions of ASE
        self._init_constants(n)
        self.masses = torch.tensor(
            masses_np, dtype=torch.float64, device=self.device
        ).reshape(n, 1).contiguous()
        self.inv_masses = (1.0 / self.masses).contiguous()  # fast_math only

        # state + transactional backup (single flat buffers)
        self.state = NHCState.allocate(n, self.tchain, self.device)
        self._backup = NHCState.allocate(n, self.tchain, self.device)
        self.set_state(positions=positions, momenta=momenta, forces=forces,
                       potential_energy=potential_energy)

        if backend == "auto":
            backend = (
                "triton"
                if _HAS_TRITON and self.device.type == "cuda" and 3 * n <= _TRITON_MAX_ELEMS
                else "torch"
            )
        if backend == "triton":
            if not _HAS_TRITON or self.device.type != "cuda":
                raise RuntimeError("triton backend needs triton + CUDA")
            if 3 * n > _TRITON_MAX_ELEMS:
                raise ValueError(f"triton backend supports 3N <= {_TRITON_MAX_ELEMS}")
        elif backend != "torch":
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.fast_math = bool(fast_math) and backend == "triton"
        self._block = max(16, triton.next_power_of_2(3 * n)) if backend == "triton" else 0
        # GB300 (fp64 latency bound): 4 warps best for 3N in [405, 1500]
        self._num_warps = (1 if self._block <= 64 else 2 if self._block <= 128
                           else 4 if self._block <= 2048 else 8 if self._block <= 8192
                           else 16)

    @classmethod
    def from_atoms(cls, atoms, *, temperature_K, timestep, tdamp, tchain=3, tloop=1,
                   **kwargs) -> "GPUNoseHooverChainNVT":
        """Build from ASE Atoms whose momenta are already set. Forces are not
        evaluated here."""
        return cls(atoms.get_masses(), atoms.get_positions(), atoms.get_momenta(),
                   temperature_K=temperature_K, timestep=timestep, tdamp=tdamp,
                   tchain=tchain, tloop=tloop, **kwargs)

    def _init_decomposed(self, masses_np, positions, momenta, forces, potential_energy, *,
                         comm, capacity, backend, fast_math, scalar_sync, ke2_update) -> None:
        if fast_math:
            raise ValueError("fast_math needs protocol='single'")
        if self.n_global < 1:
            raise ValueError("n_global must be >= 1")
        if comm is None:
            from fastmd.engine.comm import LocalCommunicator

            comm = LocalCommunicator()
        n = self.n_owned
        capacity = slot_capacity(n) if capacity is None else int(capacity)
        if capacity < n:
            raise ValueError(f"capacity {capacity} < {n} owned rows")
        capacity = max(capacity, 2)
        capacity += capacity % 2  # even: every row view of the flat buffer is 16-byte aligned
        if backend == "auto":
            backend = "triton" if _HAS_TRITON and self.device.type == "cuda" else "torch"
        if backend == "triton":
            if not _HAS_TRITON or self.device.type != "cuda":
                raise RuntimeError("triton backend needs triton + CUDA")
        elif backend != "torch":
            raise ValueError(f"unknown backend {backend!r}")
        self.comm = comm
        self.backend = backend
        self.fast_math = False
        self._block = self._num_warps = 0
        self.inv_masses = None
        self._init_constants(self.n_global)
        self._allocate_slots(capacity)
        self.masses[:n].copy_(torch.as_tensor(masses_np).reshape(n, 1))
        self._dc = _DecomposedChain(self, self.state.chain, scalar_sync=scalar_sync,
                                    ke2_update=ke2_update)
        self.set_state(positions=positions, momenta=momenta, forces=forces,
                       potential_energy=potential_energy)

    def _allocate_slots(self, capacity: int) -> None:
        """Decomposed: state and backup with ``capacity`` rows (chain scratch
        3*tloop), masses (capacity, 1) float64 (padding 1.0) and the float32
        positions mirror. Rows are zero; the chain and energy are copied over."""
        old = getattr(self, "state", None)
        options = {"dtype": torch.float64, "device": self.device}
        self.state = NHCState.allocate(capacity, self.tchain, self.device, scratch=3 * self.tloop)
        self._backup = NHCState.allocate(capacity, self.tchain, self.device,
                                         scratch=3 * self.tloop)
        self.masses = torch.ones(capacity, 1, **options)
        self._positions32 = torch.zeros(capacity, 3, dtype=torch.float32, device=self.device)
        self._real = torch.zeros(capacity, dtype=torch.bool, device=self.device)  # rows [0, n_owned)
        self._real[:min(self.n_owned, capacity)] = True
        self.capacity = capacity
        self.allocations += 1
        if old is not None:
            self.state.chain.copy_(old.chain)
            self.state.potential_energy.copy_(old.potential_energy)
            self._dc.bind(self.state.chain)

    # ------------------------------------------------------------------ state
    @property
    def positions(self) -> Tensor:
        if self._dc is not None:
            return self.state.positions[:self.n_owned]
        return self.state.positions

    @property
    def momenta(self) -> Tensor:
        if self._dc is not None:
            return self.state.momenta[:self.n_owned]
        return self.state.momenta

    @property
    def forces(self) -> Tensor:
        if self._dc is not None:
            return self.state.forces[:self.n_owned]
        return self.state.forces

    @property
    def potential_energy(self) -> Tensor:
        return self.state.potential_energy

    @property
    def eta(self) -> Tensor:
        return self.state.eta

    @property
    def p_eta(self) -> Tensor:
        return self.state.p_eta

    def _as_f64(self, x) -> Tensor:
        if torch.is_tensor(x):
            return x.detach().to(device=self.device, dtype=torch.float64)
        return torch.as_tensor(np.asarray(x, dtype=np.float64), device=self.device)

    def _rows_of(self, state: NHCState | None) -> tuple[NHCState, Tensor, Tensor, Tensor]:
        """(state, q, p, F) of the integrated rows: the whole buffers under
        "single", the owned rows ``[:n_owned]`` of the slots under "decomposed"."""
        if self._dc is None:
            st = self.state if state is None else state
            return st, st.positions, st.momenta, st.forces
        if state is not None and state is not self.state:
            raise ValueError("protocol='decomposed' integrates its own state only")
        st, n = self.state, self.n_owned
        return st, st.positions[:n], st.momenta[:n], st.forces[:n]

    @torch.no_grad()
    def set_state(self, *, positions=None, momenta=None, forces=None, eta=None,
                  p_eta=None, potential_energy=None, state: NHCState | None = None):
        """(Re)load state from host/device arrays (not for use inside capture)."""
        st, q, p, f = self._rows_of(state)
        n = q.shape[0]
        if positions is not None:
            q.copy_(self._as_f64(positions).reshape(n, 3))
            if self._dc is not None:
                self._positions32[:n].copy_(q)
        if momenta is not None:
            p.copy_(self._as_f64(momenta).reshape(n, 3))
        if forces is not None:
            f.copy_(self._as_f64(forces).reshape(n, 3))
        if eta is not None:
            st.eta.copy_(self._as_f64(eta).reshape(-1))
        if p_eta is not None:
            st.p_eta.copy_(self._as_f64(p_eta).reshape(-1))
        if potential_energy is not None:
            st.potential_energy.copy_(self._as_f64(potential_energy).reshape(1))

    @torch.no_grad()
    def set_forces(self, forces, energy=None, state: NHCState | None = None) -> None:
        """Set F(q) (and E(q)) for the current positions (capture-safe for
        device tensors)."""
        st, _, _, f = self._rows_of(state)
        n = f.shape[0]
        f.copy_(
            forces.reshape(n, 3) if torch.is_tensor(forces)
            else self._as_f64(forces).reshape(n, 3)
        )
        if energy is not None:
            st.potential_energy.copy_(
                energy.reshape(1) if torch.is_tensor(energy)
                else self._as_f64(energy).reshape(1)
            )

    # -------------------------------------------------------------- step halves
    def step_pre_force(self, state: NHCState | None = None) -> Tensor:
        """ASE ``step()`` first half: NHC(dt/2); p += dt/2*F(q_n); q += dt*p/m.

        Uses the cached ``state.forces`` = F(q_n). Returns ``state.positions``
        (float64 (N, 3) view, updated in place) for the force evaluation.
        Decomposed: the owned rows (collective), and :meth:`positions32`
        follows.
        """
        if self._dc is not None:
            _, q, p, f = self._rows_of(state)
            n = q.shape[0]
            self._dc.pre_force(q, p, f, self.masses[:n].reshape(-1))
            self._positions32[:n].copy_(q)
            return q
        st = self.state if state is None else state
        if self.backend == "triton":
            _nhc_pre_force_kernel[(1,)](
                st.positions, st.momenta, st.forces,
                self.inv_masses if self.fast_math else self.masses,
                st.eta, st.p_eta, self.consts, 3 * self.n_atoms,
                BLOCK=self._block, M=self.tchain, TLOOP=self.tloop,
                F32_KICK=self.f32_kick, FAST=self.fast_math,
                num_warps=self._num_warps, enable_fp_fusion=self.fast_math,
            )
        else:
            with torch.no_grad():
                self._torch_integrate_nhc(st)
                self._torch_kick(st, st.forces)
                st.positions.add_(st.momenta * self.dt / self.masses)
        return st.positions

    def step_post_force(self, forces: Tensor, energy: Tensor | None = None,
                        state: NHCState | None = None) -> None:
        """ASE ``step()`` second half: p += dt/2*F(q_{n+1}); NHC(dt/2).

        ``forces``: (N, 3) device tensor (float32/float64) = F(q_{n+1}); it is
        cached (as float64) for the next pre-force half. ``energy``: optional
        1-element device tensor stored as the potential energy. Decomposed:
        the owned rows' forces (collective); ``fix_com`` then removes the
        global momentum (as the step 3.3 driver did).
        """
        if self._dc is not None:
            self._decomposed_post_force(forces, energy, state)
            return
        st = self.state if state is None else state
        forces = forces.reshape(self.n_atoms, 3)
        if self.backend == "triton":
            if not forces.is_contiguous():
                forces = forces.contiguous()
            has_e = energy is not None
            e = energy.reshape(-1) if has_e else st.potential_energy
            _nhc_post_force_kernel[(1,)](
                st.momenta, forces, st.forces,
                self.inv_masses if self.fast_math else self.masses,
                st.eta, st.p_eta, self.consts, e, st.potential_energy,
                3 * self.n_atoms,
                BLOCK=self._block, M=self.tchain, TLOOP=self.tloop,
                F32_KICK=self.f32_kick, HAS_E=has_e, FAST=self.fast_math,
                num_warps=self._num_warps, enable_fp_fusion=self.fast_math,
            )
        else:
            with torch.no_grad():
                self._torch_kick(st, forces)
                self._torch_integrate_nhc(st)
                st.forces.copy_(forces)
                if energy is not None:
                    st.potential_energy.copy_(energy.reshape(1))

    @torch.no_grad()
    def step_pre_force_rows(self, rows: int) -> Tensor:
        """Decomposed :meth:`step_pre_force` on rows ``[0, rows)`` with the owned-row mask (capture-safe,
        collective; the HAS_MASK kernels leave the padding rows untouched); the float32 positions view."""
        st, real = self.state, self._real[:rows]
        q = st.positions[:rows]
        self._dc.pre_force(q, st.momenta[:rows], st.forces[:rows], self.masses[:rows].reshape(-1), real)
        self._positions32[:rows].copy_(q)
        return self._positions32[:rows]

    @torch.no_grad()
    def step_post_force_rows(self, forces: Tensor, energy: Tensor | None, rows: int) -> None:
        """Decomposed :meth:`step_post_force` on rows ``[0, rows)`` (capture-safe, collective); padding
        rows of ``forces`` are ignored."""
        st, real = self.state, self._real[:rows]
        forces = torch.where(real[:, None], forces.reshape(rows, 3), 0.0).contiguous()
        p = st.momenta[:rows]
        self._dc.post_force(p, forces, self.masses[:rows].reshape(-1), real)
        if self.fix_com:
            total = p.sum(dim=0)
            self.comm.allreduce_sum_(total)
            p.sub_(torch.where(real[:, None], total / self.n_global, 0.0))
        st.forces[:rows].copy_(forces)
        if energy is not None:
            st.potential_energy.copy_(energy.reshape(1))

    def nonfinite_rows(self, rows: int) -> Tensor:
        st = self.state
        finite = torch.isfinite(st.positions[:rows]).all() & torch.isfinite(st.momenta[:rows]).all()
        return (~finite).reshape(1).long()

    @torch.no_grad()
    def reserve(self, rows: int) -> bool:
        """Decomposed: at least ``rows`` slot rows (even); the owned rows, chain and energy are kept."""
        if self._dc is None:
            raise ValueError("reserve needs protocol='decomposed'")
        if int(rows) <= self.capacity:
            return False
        n, st = self.n_owned, self.state
        kept = [x[:n].clone() for x in (st.positions, st.momenta, st.forces, self.masses, self._positions32)]
        capacity = int(rows) + int(rows) % 2
        self._allocate_slots(capacity)
        st = self.state
        for dst, src in zip((st.positions, st.momenta, st.forces, self.masses, self._positions32), kept):
            dst[:n].copy_(src)
        self._backup.flat.copy_(st.flat)
        self.generation += 1
        return True

    @torch.no_grad()
    def _decomposed_post_force(self, forces: Tensor, energy: Tensor | None,
                               state: NHCState | None) -> None:
        st, _, p, f = self._rows_of(state)
        n = p.shape[0]
        forces = forces.reshape(n, 3)
        if not forces.is_contiguous():
            forces = forces.contiguous()
        self._dc.post_force(p, forces, self.masses[:n].reshape(-1))
        if self.fix_com:  # the step 3.3 driver's op sequence
            total = p.sum(dim=0)
            self.comm.allreduce_sum_(total)
            p.sub_(total / self.n_global)
        f.copy_(forces)
        if energy is not None:
            st.potential_energy.copy_(energy.reshape(1))

    def step(
        self, force_fn, state: NHCState | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full step; same conventions as :meth:`GPUIntegrator.step` and
        ``GPUResidentMolecularDynamics.evaluate``.

        ``force_fn(positions_f64)`` returns the forces, or ``(forces, energy)``
        with an optional 1-element ``energy`` (``None`` keeps the stored
        potential energy). Returns ``(positions, momenta, forces)``, the
        float64 (N, 3) views of the updated state (forces = F(positions)).
        """
        st = self.state if state is None else state
        q = self.step_pre_force(st)
        result = force_fn(q)
        forces, energy = result if isinstance(result, tuple) else (result, None)
        self.step_post_force(forces, energy=energy, state=st)
        if self._dc is not None:
            return self.positions, self.momenta, self.forces
        return st.positions, st.momenta, st.forces

    # ---------------------------------------------------------- torch reference
    def _torch_kick(self, st: NHCState, forces: Tensor) -> None:
        if self.f32_kick:  # NumPy NEP50 semantics for float32 forces
            st.momenta.add_((forces.to(torch.float32) * self.dt2).to(torch.float64))
        else:
            st.momenta.add_(forces.to(torch.float64) * self.dt2)

    def _torch_p_eta_j(self, st, j, delta2, neg_delta4) -> None:
        # kT and 3N*kT from the constants buffer (not Python floats baked into
        # a captured graph) so set_temperature reaches graph replays; same bits
        pe, Q, M = st.p_eta, self.Q, self.tchain
        if j < M - 1:
            s = torch.exp(pe[j + 1] * neg_delta4 / Q[j + 1])
            pe[j].mul_(s)
        if j == 0:
            g = (st.momenta * st.momenta / self.masses).sum() - self.consts[3]
        else:
            g = pe[j - 1] * pe[j - 1] / Q[j - 1] - self.consts[2]
        pe[j].add_(g * delta2)
        if j < M - 1:
            pe[j].mul_(s)

    def _torch_integrate_nhc(self, st: NHCState) -> None:
        M = self.tchain
        for _ in range(self.tloop):
            for delta, delta2, neg_delta4, neg_delta in self.sub:
                for j in range(M):
                    self._torch_p_eta_j(st, M - j - 1, delta2, neg_delta4)
                st.eta.add_(st.p_eta * delta / self.Q)
                st.momenta.mul_(torch.exp(st.p_eta[0] * neg_delta / self.Q[0]))
                for j in range(M):
                    self._torch_p_eta_j(st, j, delta2, neg_delta4)

    # -------------------------------------------------------------- observables
    def velocities(self, state: NHCState | None = None) -> Tensor:
        if self._dc is not None:
            _, _, p, _ = self._rows_of(state)
            return p / self.masses[:p.shape[0]]
        st = self.state if state is None else state
        return st.momenta / self.masses

    def kinetic_energy(self, state: NHCState | None = None) -> Tensor:
        """0.5 * vdot(p, p/m) (``Atoms.get_kinetic_energy``), 0-dim float64.
        Decomposed: the global value (one ``allreduce_sum_``, collective)."""
        if self._dc is not None:
            _, _, p, _ = self._rows_of(state)
            local = (0.5 * (p * (p / self.masses[:p.shape[0]])).sum()).reshape(1)
            self.comm.allreduce_sum_(local)
            return local[0]
        st = self.state if state is None else state
        return 0.5 * (st.momenta * (st.momenta / self.masses)).sum()

    def temperature(self, state: NHCState | None = None) -> Tensor:
        """``Atoms.get_temperature()``: 2*Ekin/(3N kB) (no constraints; N =
        ``n_global``)."""
        if self._dc is not None:
            return 2 * self.kinetic_energy(state) / (3 * self.n_global * units.kB)
        return 2 * self.kinetic_energy(state) / (3 * self.n_atoms * units.kB)

    def thermostat_energy(self, state: NHCState | None = None, constants: tuple | None = None) -> Tensor:
        """``NoseHooverChainThermostat.get_thermostat_energy``. ``constants``: the ``(kT, ndof_kT, Q)``
        of a snapshot (a staged frame's) instead of the current target's."""
        st = self.state if state is None else state
        kT, ndof_kT, Q = (self.kT, self.ndof_kT, self.Q) if constants is None else constants
        return (ndof_kT * st.eta[0] + kT * st.eta[1:].sum()
                + (0.5 * (st.p_eta * st.p_eta) / Q).sum())

    def conserved_energy(self, potential_energy=None,
                         state: NHCState | None = None, constants: tuple | None = None) -> Tensor:
        """``NoseHooverChainNVT.get_conserved_energy``: E_pot + E_kin +
        E_thermostat (uses the stored potential energy unless given;
        ``constants`` as in :meth:`thermostat_energy`)."""
        st = self.state if state is None else state
        epot = st.potential_energy[0] if potential_energy is None else potential_energy
        return epot + self.kinetic_energy(st) + self.thermostat_energy(st, constants=constants)

    # ------------------------------------------------------------ transactions
    def backup(self, state: NHCState | None = None) -> None:
        """Snapshot the full integrator state (one D2D copy, capture-safe)."""
        self._backup.flat.copy_((self.state if state is None else state).flat)

    def restore(self, state: NHCState | None = None) -> None:
        """Roll back to the last :meth:`backup` (one D2D copy, capture-safe)."""
        (self.state if state is None else state).flat.copy_(self._backup.flat)
        if self._dc is not None:
            self._sync_positions32()

    def snapshot(self) -> tuple:
        """Clones of state and backup (eager; capture warmup, trials).
        Decomposed: also the slot ``generation`` (see :meth:`load_snapshot`)."""
        if self._dc is not None:
            return self.state.flat.clone(), self._backup.flat.clone(), self.generation
        return self.state.flat.clone(), self._backup.flat.clone()

    def load_snapshot(self, snapshot: tuple) -> None:
        if self._dc is not None:
            state, backup, generation = snapshot
            if generation != self.generation:  # rows were reordered since
                raise ValueError("snapshot predates an epoch refresh")
            self.state.flat.copy_(state)
            self._backup.flat.copy_(backup)
            self._sync_positions32()
            return
        state, backup = snapshot
        self.state.flat.copy_(state)
        self._backup.flat.copy_(backup)

    def new_state(self) -> NHCState:
        """Allocate another state of identical shape (e.g. for replicas)."""
        if self._dc is not None:
            raise ValueError("replica states need protocol='single'")
        return NHCState.allocate(self.n_atoms, self.tchain, self.device)

    # ------------------------------------------------------- slots (decomposed)
    def _sync_positions32(self) -> None:
        n = self.n_owned
        self._positions32[:n].copy_(self.state.positions[:n])

    def positions32(self) -> Tensor:
        """The evaluator's input: float32 (n_owned, 3) positions (decomposed:
        a mirror written by :meth:`step_pre_force` and :meth:`reorder_`)."""
        if self._dc is None:
            return self.state.positions.to(torch.float32)
        return self._positions32[:self.n_owned]

    def migrating_fields(self) -> dict[str, Tensor]:
        """Views of the per-atom payload that migrates with the atoms, with
        the evaluator payload's names and shapes: masses (n,), momenta and
        positions64 (n, 3), float64."""
        n = self.n_owned
        return {"masses": self.masses[:n].reshape(-1), "momenta": self.state.momenta[:n],
                "positions64": self.state.positions[:n]}

    @torch.no_grad()
    def reorder_(self, state) -> bool:
        """Load the rows of ``state`` (the ``AtomState`` an epoch refresh
        returned; its payload holds :meth:`migrating_fields`' names) in place.

        positions64, momenta and masses fill rows ``[0, n')``; the padding
        rows are zeroed (masses 1.0); the float32 positions fill the mirror.
        Forces are not carried (F(q_n) was consumed by the pre-force half and
        :meth:`step_post_force` writes F(q_{n+1}) on the new rows) unless the
        payload has "forces" (the refresh of the P>1 in-graph step runs
        before the pre-force half, which reads them). The backup
        follows the reorder and ``generation`` increments (older snapshots
        are refused). If n' exceeds the capacity the buffers grow first (the
        chain and energy are kept). Returns True if they were reallocated.
        """
        if self._dc is None:
            raise ValueError("reorder_ needs protocol='decomposed'")
        payload = state.payload
        n = int(payload["momenta"].shape[0])
        grown = n > self.capacity
        if grown:
            self._allocate_slots(slot_capacity(n))
        st = self.state
        own = {x.untyped_storage().data_ptr()
               for x in (st.flat, self._backup.flat, self.masses, self._positions32)}
        rows = [(st.positions, payload["positions64"]), (st.momenta, payload["momenta"]),
                (self.masses, payload["masses"].reshape(n, 1)),
                (self._positions32, state.positions)]
        if "forces" in payload:
            rows.append((st.forces, payload["forces"]))
        sources = []
        for dst, src in rows:
            src = src.reshape(n, dst.shape[1])
            if src.untyped_storage().data_ptr() in own:  # defensive: our own rows
                src = src.clone()
            sources.append(src)
        for (dst, _), src in zip(rows, sources):
            dst[:n].copy_(src)
        st.positions[n:].zero_()
        st.momenta[n:].zero_()
        st.forces[n:].zero_()
        self.masses[n:].fill_(1.0)
        self._positions32[n:].zero_()
        self._real.zero_()
        self._real[:n] = True
        self.n_owned = self.n_atoms = n
        self._backup.flat.copy_(st.flat)
        self.generation += 1
        return grown

    def warmup_compile(self, force_dtypes=None, energy_dtype=torch.float32) -> None:
        """JIT-compile the Triton kernels outside of any graph capture for the
        given force dtypes (default ``self.force_dtype``), with and without
        energy. State is preserved. Decomposed: one step on row 0 of the
        slots (the step's pointers, so the same specializations) through a
        ``LocalCommunicator`` (no collective runs; a non-root rank warms its
        zero-contribution chain), covering the float64 cached-force kick."""
        if self.backend != "triton":
            return
        if self._dc is not None:
            self._warmup_decomposed(force_dtypes, energy_dtype)
            return
        force_dtypes = [self.force_dtype] if force_dtypes is None else force_dtypes
        saved = self.snapshot()
        for dtype in force_dtypes:
            for with_energy in (True, False):
                self.step_pre_force()
                forces = self.state.forces.to(dtype).clone()
                energy = (
                    self.state.potential_energy.to(energy_dtype).clone()
                    if with_energy else None
                )
                self.step_post_force(forces, energy=energy)
                self.load_snapshot(saved)
        torch.cuda.synchronize(self.device)

    def _warmup_decomposed(self, force_dtypes, energy_dtype) -> None:
        from fastmd.engine.comm import LocalCommunicator

        force_dtypes = [self.force_dtype] if force_dtypes is None else force_dtypes
        st = self.state
        saved = (st.flat.clone(), self._backup.flat.clone(), self._positions32.clone(),
                 self.masses.clone(), self._dc._ke2.clone(), self._real.clone())
        # _dc's broadcast/zero flags stay this rank's: a non-root rank warms its ZERO chain
        comm, n = self.comm, self.n_owned
        self.comm, self.n_owned = LocalCommunicator(), 1
        try:
            for dtype in force_dtypes:
                for with_energy in (True, False):
                    # The per-call halves, then the masked in-graph rows (HAS_MASK kernels).
                    for rows in (None, 1):
                        st.positions[:1].zero_()
                        st.momenta[:1].fill_(1.0)
                        st.forces[:1].zero_()
                        self.masses[:1].fill_(1.0)
                        self._real[:1] = True
                        forces = torch.zeros(1, 3, dtype=dtype, device=self.device)
                        energy = (torch.zeros(1, dtype=energy_dtype, device=self.device)
                                  if with_energy else None)
                        if rows is None:
                            self.step_pre_force()
                            self.step_post_force(forces, energy=energy)
                        else:
                            self.step_pre_force_rows(rows)
                            self.step_post_force_rows(forces, energy, rows)
        finally:
            self.comm, self.n_owned = comm, n
            for dst, src in zip((st.flat, self._backup.flat, self._positions32, self.masses,
                                 self._dc._ke2, self._real), saved):
                dst.copy_(src)
        torch.cuda.synchronize(self.device)
