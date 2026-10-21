"""Structure relaxation of different systems in the slots of a whole-step CUDA graph: ASE's FIRE per system,
with a fixed cell or a variable one.

The relaxation counterpart of the slot unions of :mod:`fastmd.execution.system_batch`, whose systems advance
by MD: here each slot's system takes one FIRE step per whole-step step, its optimizer state and its
convergence on the device, so a captured window of k steps relaxes every system by up to k steps without the
host:

* :class:`SystemFIRE`: ASE's ``FIRE`` (``ase.optimize.FIRE``, its defaults: :class:`FIREParameters`) per
  system: its own velocity, time step, mixing parameter and count of downhill steps, each dot product and the
  step-length limit over its own degrees of freedom; it converges when its largest force row is below ``fmax``
  (strictly, as ASE's optimizers), stops at a step limit, or fails on a non-finite state. A system stops in the
  step it finishes (its step count is ASE's ``nsteps``) and keeps the forces, energy and stress of its last
  state. With ``cell_filter``, ASE's ``FrechetCellFilter`` (its default arguments) per system: its degrees of
  freedom are the positions in its original cell and the logarithm of its cell's deformation scaled by its atom
  count; its forces are the filter's, from its stress.
* :class:`CellRelaxationBuilder`: the slot builder whose cells change in the graph
  (:meth:`~CellRelaxationBuilder.set_cells`); its skin test bounds the change of every pair distance since the
  candidate refresh by the atoms' moves (in fractional coordinates, times the refresh's cell) and each cell's
  deformation since then.
* :class:`SlotRelaxationRunner`: the slot runner over them; a window's record holds one phase per system, read
  with the window's flags in one copy.

``fastmd.interfaces.multi_system_relax`` drives these parts, with a queue of structures through the slots.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from fastmd.engine.entry_config import scoped_call
from fastmd.graph.geometry import small_matmul
from fastmd.execution.gpu_integrator import StatefulIntegrator
from fastmd.execution.gpu_nhc import _HAS_TRITON
from fastmd.execution.system_batch import (
    SlotLayout,
    SlotUnionGraphBuilder,
    SlotWholeStepRunner,
    segment_sums,
    system_energy_derivative,
    system_volumes,
)
from fastmd.execution.whole_step_cuda_graph import WholeStepGraphRunner

if _HAS_TRITON:
    from fastmd.graph.fixed_capacity_builder import allocate_cell_list, estimate_batch_cell_list_sizes

# The phase of each slot's system: one float64 per system on the device, in the window record.
IDLE, PENDING, ACTIVE, CONVERGED, STEP_LIMIT, FAILED = 0, 1, 2, 3, 4, 5
FINISHED = (CONVERGED, STEP_LIMIT, FAILED)
PHASES = {IDLE: "idle", PENDING: "pending", ACTIVE: "active", CONVERGED: "converged", STEP_LIMIT: "max_steps",
          FAILED: "failed"}

# The matrix exponential of the cell filter inside the graph: a degree-12 Taylor polynomial of A / 2**8 (Horner),
# squared 8 times, in float64; fixed shapes and no host decision (torch.linalg.matrix_exp chooses its scaling on
# the host). The polynomial and the squarings carry exp(A) - 1, so a small A keeps its relative accuracy instead of
# rounding at the identity's scale. Up to a norm of 10 the truncation stays below 1e-25 relative, under the
# squarings' float64 rounding; a relaxation's log-deformations and the normalised gradient blocks are far smaller.
_EXPM_SQUARINGS = 8
_EXPM_DEGREE = 12


def expm_small(matrices: Tensor) -> Tensor:
    """``exp`` of each ``[..., d, d]`` float64 matrix (see ``_EXPM_SQUARINGS``): device ops, capture-safe."""
    scaled = matrices / float(1 << _EXPM_SQUARINGS)
    eye = torch.eye(matrices.shape[-1], dtype=matrices.dtype, device=matrices.device).expand(matrices.shape)
    # Horner for exp(X) - 1 = X (1 + X/2 (1 + X/3 (...))), the bracket from the inside out, then times X
    bracket = eye + scaled / _EXPM_DEGREE
    for k in range(_EXPM_DEGREE - 1, 1, -1):
        bracket = eye + torch.matmul(scaled, bracket) / k
    deviation = torch.matmul(scaled, bracket)
    for _ in range(_EXPM_SQUARINGS):  # (1 + E)^2 - 1 = 2 E + E E
        deviation = 2.0 * deviation + torch.matmul(deviation, deviation)
    return eye + deviation


def expm_frechet_adjoint(log_deformation: Tensor, gradient: Tensor) -> Tensor:
    """``[S, 3, 3]``: ``L(Y^T, G)``, the Fréchet derivative of ``expm`` at ``Y^T`` in the direction ``G``, which is
    the adjoint of ``L(Y, .)`` in the Frobenius product: ASE's ``FrechetCellFilter`` cell force (before its
    division by ``exp_cell_factor``), whose loop over the nine directions ``E`` takes ``<L(Y, E), G>``. The top
    right block of ``expm([[Y^T, G], [0, Y^T]])``, with ``G`` scaled by its largest entry (``L`` is linear in
    ``G``)."""
    scale = gradient.abs().amax(dim=(1, 2)).clamp_min(1e-300)[:, None, None]
    top = log_deformation.transpose(1, 2)
    block = torch.cat((torch.cat((top, gradient / scale), dim=2),
                       torch.cat((torch.zeros_like(top), top), dim=2)), dim=1)
    return expm_small(block)[:, :3, 3:] * scale


def inverse33(matrices: Tensor) -> Tensor:
    """The inverse of each ``[S, 3, 3]`` matrix from its rows' cross products and determinant (elementwise: no
    solver, no host synchronization)."""
    m0, m1, m2 = matrices[:, 0], matrices[:, 1], matrices[:, 2]
    columns = torch.stack((torch.linalg.cross(m1, m2, dim=1), torch.linalg.cross(m2, m0, dim=1),
                           torch.linalg.cross(m0, m1, dim=1)), dim=2)
    return columns / (m0 * columns[:, :, 0]).sum(dim=1)[:, None, None]


def rows_times(rows: Tensor, matrices: Tensor) -> Tensor:
    """``rows[i] @ matrices[i]`` for ``(n, 3)`` rows and ``(n, 3, 3)`` matrices, row by row (no GEMM)."""
    return rows[:, 0:1] * matrices[:, 0] + rows[:, 1:2] * matrices[:, 1] + rows[:, 2:3] * matrices[:, 2]


def segment_max(values: Tensor, ceilings: Tensor, atoms: Tensor) -> Tensor:
    """``(S,)``: the largest of ``values`` (``(rows,)``) over each system's rows, the first ``atoms[s]`` of its
    slot of ``ceilings[s]`` rows (:func:`~fastmd.execution.system_batch.segment_sums`' segments)."""
    lengths = torch.stack((atoms, ceilings - atoms), dim=1).reshape(-1)
    return torch.segment_reduce(values, "max", lengths=lengths, unsafe=True)[::2]


@dataclass(frozen=True)
class FIREParameters:
    """ASE's ``FIRE`` parameters, its defaults: the initial time step ``dt``, the longest step of a system
    ``maxstep`` (the norm over all its degrees of freedom, with a cell filter its cell's too), ``dtmax``, the
    downhill steps ``Nmin`` before the time step grows, ``finc``, ``fdec``, the mixing parameter ``astart`` after
    an uphill step, ``fa`` and the initial mixing parameter ``a``."""

    dt: float = 0.1
    maxstep: float = 0.2
    dtmax: float = 1.0
    Nmin: int = 5
    finc: float = 1.1
    fdec: float = 0.5
    astart: float = 0.1
    fa: float = 0.99
    a: float = 0.1

    def __post_init__(self) -> None:
        for name in ("dt", "maxstep", "dtmax", "finc", "fdec", "astart", "fa", "a"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"FIREParameters: {name} must be finite and positive")
        if isinstance(self.Nmin, bool) or not isinstance(self.Nmin, int) or self.Nmin < 0:
            raise ValueError("FIREParameters: Nmin must be a nonnegative integer")


class SystemFIREState:
    """Views into one flat float64 buffer: per row ``x`` (the optimizer's positions: Cartesian, or with a cell
    filter in the system's original cell), ``v`` (velocity), ``g`` (the optimizer's forces: ASE's negative
    gradient) and ``f`` (the model's forces), ``(rows, 3)``; per system ``dt``, ``a``, ``npos`` (downhill steps),
    ``fresh`` (no step yet), ``phase``, ``steps``, ``fmax``, ``energy`` and ``factor`` (the cell filter's
    ``exp_cell_factor``, its atom count), ``(S,)``; with a cell filter per system ``log_deformation``,
    ``cell_velocity``, ``cell_gradient``, ``stress`` (eV/Å³) and ``original_cells``, ``(S, 3, 3)``."""

    ROWS = ("x", "v", "g", "f")
    SYSTEMS = ("dt", "a", "npos", "fresh", "phase", "steps", "fmax", "energy", "factor")
    CELLS = ("log_deformation", "cell_velocity", "cell_gradient", "stress", "original_cells")

    def __init__(self, rows: int, systems: int, cell_filter: bool, device) -> None:
        shapes = ([(name, (rows, 3)) for name in self.ROWS] + [(name, (systems,)) for name in self.SYSTEMS]
                  + ([(name, (systems, 3, 3)) for name in self.CELLS] if cell_filter else []))
        self.flat = torch.zeros(sum(math.prod(shape) for _, shape in shapes), dtype=torch.float64, device=device)
        offset = 0
        for name, shape in shapes:
            size = math.prod(shape)
            setattr(self, name, self.flat[offset: offset + size].view(shape))
            offset += size


class SystemFIRE(StatefulIntegrator):
    """ASE's ``FIRE`` for the systems in the slots of :class:`~fastmd.execution.system_batch.SlotLayout` (see
    the module docstring), one optimizer step per step half pair:

        positions = fire.step_pre_force()        # each active system's FIRE step from its cached forces
        forces, energies[, stresses] = the union's evaluation at positions (in fire.cells)
        fire.step_post_force(forces, energy=energies[, stress=stresses])   # forces, convergence, phase

    A written system (:meth:`write_system`) is pending: the next step evaluates it without moving it (ASE's
    forces at the start), checks its convergence there (0 steps) and makes it active. ``positions`` are the
    systems' Cartesian positions ``(rows, 3)`` float64 (a vacant row: the origin, never moved); ``forces`` the
    model's at them, ``potential_energy`` ``(S,)``; ``cells`` ``(S, 3, 3)`` float64 (a cell filter). The step
    halves, :meth:`backup` and :meth:`restore` are capture-safe; :meth:`sync` recomputes the positions and cells
    from the state (after :meth:`restore` or a write). No restart files.
    """

    ensemble = "fire"
    comm = None

    def __init__(self, layout: SlotLayout, *, fmax: float, max_steps: int, parameters: FIREParameters | None = None,
                 cell_filter: bool = False, device="cuda") -> None:
        if not math.isfinite(fmax) or fmax <= 0:
            raise ValueError("fmax must be finite and positive")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        self.device = torch.device(device)
        self.layout = layout
        self.systems = layout.count
        self.parameters = FIREParameters() if parameters is None else parameters
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        self.cell_filter = bool(cell_filter)
        self.generation = 0
        self.state = self.new_state()
        self._backup = self.new_state()
        self.system_of_row = layout.system_of_row(self.device).long()
        self.ceilings = torch.tensor(layout.ceilings, dtype=torch.int64, device=self.device)
        self.atoms = torch.ones(self.systems, dtype=torch.int64, device=self.device)
        self.occupied = torch.zeros((layout.rows, 1), dtype=torch.bool, device=self.device)
        self.host_atoms = [0] * self.systems
        # Derived from the state (sync() and every pre-force half): the positions the model sees and, with a cell
        # filter, each system's deformation, its inverse and its cell.
        self._positions = torch.zeros((layout.rows, 3), dtype=torch.float64, device=self.device)
        if self.cell_filter:
            eye = torch.eye(3, dtype=torch.float64, device=self.device).expand(self.systems, 3, 3)
            self._deformation = eye.clone()
            self._inverse_deformation = eye.clone()
            self._cells = eye.clone()

    # ------------------------------------------------------------------ systems
    @torch.no_grad()
    def write_system(self, slot: int, *, positions, cell=None) -> None:
        """A system of ``n`` atoms at ``positions`` (``(n, 3)`` Å; with a cell filter in ``cell``, ``(3, 3)``,
        its original cell) in ``slot``, pending, with a fresh optimizer (ASE's initial time step and mixing
        parameter, no velocity, no step). Eager, between windows; :meth:`sync` before the next window."""
        positions = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 3)
        n, ceiling = int(positions.shape[0]), self.layout.ceilings[slot]
        if not 1 <= n <= ceiling:
            raise ValueError(f"slot {slot} holds 1 to {ceiling} atoms, got {n}")
        if self.cell_filter and cell is None:
            raise ValueError("a cell filter needs each system's cell")
        st, p, start = self.state, self.parameters, self.layout.starts[slot]
        for field in (st.x, st.v, st.g, st.f):
            field[start:start + ceiling].zero_()
        st.x[start:start + n].copy_(positions)
        values = {"dt": p.dt, "a": p.a, "npos": 0.0, "fresh": 1.0, "phase": float(PENDING), "steps": 0.0,
                  "fmax": math.inf, "energy": 0.0, "factor": float(n)}
        for name, value in values.items():
            getattr(st, name)[slot] = value
        if self.cell_filter:
            for name in ("log_deformation", "cell_velocity", "cell_gradient", "stress"):
                getattr(st, name)[slot].zero_()
            st.original_cells[slot].copy_(torch.as_tensor(cell, dtype=torch.float64).reshape(3, 3))
        self.atoms[slot] = n
        self.occupied[start:start + ceiling] = False
        self.occupied[start:start + n] = True
        self.host_atoms[slot] = n

    def release(self, slot: int) -> None:
        """Slot ``slot``'s finished system stays where it is, frozen, and is no longer reported (idle)."""
        self.state.phase[slot] = float(IDLE)

    @torch.no_grad()
    def take_systems(self, source: "SystemFIRE", slots: list[int]) -> None:
        """Slot ``k`` takes the system of ``source``'s slot ``slots[k]`` with its optimizer state as it is (its
        positions, velocity, forces, time step, mixing parameter, steps, phase and, with a cell filter, its cell
        variables): it continues where it was. Eager; :meth:`sync` follows."""
        if source.cell_filter != self.cell_filter or len(slots) != self.systems:
            raise ValueError("take_systems needs one source slot per slot, and the same cell filter")
        destination_rows, source_rows = [], []
        for k, s in enumerate(slots):
            n = source.host_atoms[s]
            if n > self.layout.ceilings[k]:
                raise ValueError(f"slot {k} holds at most {self.layout.ceilings[k]} atoms, got {n}")
            destination_rows.append(torch.arange(self.layout.starts[k], self.layout.starts[k] + n))
            source_rows.append(torch.arange(source.layout.starts[s], source.layout.starts[s] + n))
            self.atoms[k] = n
            self.host_atoms[k] = n
            self.occupied[self.layout.starts[k]:self.layout.starts[k] + self.layout.ceilings[k]] = False
            self.occupied[self.layout.starts[k]:self.layout.starts[k] + n] = True
        destination = torch.cat(destination_rows).to(self.device)
        rows = torch.cat(source_rows).to(self.device)
        systems = torch.as_tensor(slots, dtype=torch.int64, device=self.device)
        for name in SystemFIREState.ROWS:
            rows_of_source = getattr(source.state, name).index_select(0, rows)
            getattr(self.state, name).zero_().index_copy_(0, destination, rows_of_source)
        for name in SystemFIREState.SYSTEMS + (SystemFIREState.CELLS if self.cell_filter else ()):
            getattr(self.state, name).copy_(getattr(source.state, name).index_select(0, systems))

    @torch.no_grad()
    def sync(self, state: SystemFIREState | None = None) -> None:
        """The positions (and cells) derived from the state: after a write, a restore or a snapshot load."""
        self._derive(self.state if state is None else state)

    def _derive(self, st: SystemFIREState) -> None:
        if not self.cell_filter:
            self._positions.copy_(st.x)
            return
        deformation = expm_small(st.log_deformation)
        self._deformation.copy_(deformation)
        self._inverse_deformation.copy_(expm_small(-st.log_deformation))
        transposed = deformation.transpose(1, 2)
        self._cells.copy_(torch.matmul(st.original_cells, transposed))
        # r = u F^T, row by row; vacant rows stay at the origin
        self._positions.copy_(rows_times(st.x, transposed.index_select(0, self.system_of_row)))

    # ------------------------------------------------------------------ the StatefulIntegrator state
    @property
    def n_owned(self) -> int:
        return self.layout.rows

    @property
    def positions(self) -> Tensor:
        return self._positions

    @property
    def momenta(self) -> Tensor:
        """The optimizer's velocities (unit masses), ``(rows, 3)``."""
        return self.state.v

    @property
    def forces(self) -> Tensor:
        return self.state.f

    @property
    def potential_energy(self) -> Tensor:
        return self.state.energy

    @property
    def cells(self) -> Tensor | None:
        """``(S, 3, 3)`` float64: each system's current cell (a cell filter; None otherwise)."""
        return self._cells if self.cell_filter else None

    @property
    def phase(self) -> Tensor:
        return self.state.phase

    @torch.no_grad()
    def set_forces(self, forces, energy=None, state: SystemFIREState | None = None) -> None:
        """The model's forces ``(rows, 3)`` (and energies ``(S,)``) at the current state, stored as they are; the
        optimizer's forces and the phases change only in :meth:`step_post_force` (capture-safe)."""
        st = self.state if state is None else state
        st.f.copy_(torch.as_tensor(forces, device=self.device).reshape(-1, 3))
        if energy is not None:
            st.energy.copy_(torch.as_tensor(energy, device=self.device).reshape(self.systems))

    # -------------------------------------------------------------- step halves
    def _sums(self, rows: list[Tensor], cells: list[Tensor]) -> Tensor:
        """``(S, k)`` float64: each system's sums of the ``k`` per-row values ``rows`` (``(rows,)``), plus the cell
        filter's per-system ``cells`` (``(S,)``)."""
        sums = segment_sums(torch.stack(rows, dim=1), self.ceilings, self.atoms)
        return sums + torch.stack(cells, dim=1) if self.cell_filter else sums

    def step_pre_force(self, state: SystemFIREState | None = None) -> Tensor:
        """Every active system's ASE ``FIRE.step`` from its cached forces (a pending, finished or idle system does
        not move); returns the float64 ``(rows, 3)`` positions to evaluate (and sets ``cells``)."""
        st = self.state if state is None else state
        p = self.parameters
        rows = self.system_of_row
        moving = st.phase == ACTIVE
        first = moving & (st.fresh != 0)
        later = moving & (st.fresh == 0)
        dot_rows = [(st.g * st.v).sum(dim=1), (st.g * st.g).sum(dim=1), (st.v * st.v).sum(dim=1)]
        dot_cells = []
        if self.cell_filter:
            gc, vc = st.cell_gradient, st.cell_velocity
            dot_cells = [(gc * vc).sum(dim=(1, 2)), (gc * gc).sum(dim=(1, 2)), (vc * vc).sum(dim=(1, 2))]
        vf, g2, v2 = self._sums(dot_rows, dot_cells).unbind(dim=1)
        downhill = later & (vf > 0.0)
        uphill = later & ~(vf > 0.0)
        reset = first | uphill
        grow = downhill & (st.npos > p.Nmin)
        # The time step first (ASE's step uses the new one), then the velocity: mixed with the forces downhill (ASE's
        # arithmetic order, the mixing parameter before its update), zero uphill and on the first step, unchanged
        # where the system does not move; then the move, its norm limited per system.
        dt = torch.where(grow, torch.clamp(st.dt * p.finc, max=p.dtmax), torch.where(uphill, st.dt * p.fdec, st.dt))
        a, root_g2, root_v2 = st.a.clone(), g2.sqrt(), v2.sqrt()
        per_row = torch.stack((a, root_g2, root_v2, dt), dim=1).index_select(0, rows)
        a_row, g2_row, v2_row, dt_row = per_row[:, 0:1], per_row[:, 1:2], per_row[:, 2:3], per_row[:, 3:4]
        down_row, reset_row = downhill[rows][:, None], reset[rows][:, None]
        move_row = moving[rows][:, None] & self.occupied
        v = torch.where(down_row, (1.0 - a_row) * st.v + a_row * st.g / g2_row * v2_row,
                        torch.where(reset_row, torch.zeros_like(st.v), st.v))
        v = torch.where(move_row, v + dt_row * st.g, v)
        dr = dt_row * v
        if self.cell_filter:
            a_cell, g2_cell, v2_cell, dt_cell = (value[:, None, None] for value in (a, root_g2, root_v2, dt))
            move_cell = moving[:, None, None]
            vc = torch.where(downhill[:, None, None], (1.0 - a_cell) * vc + a_cell * gc / g2_cell * v2_cell,
                             torch.where(reset[:, None, None], torch.zeros_like(vc), vc))
            vc = torch.where(move_cell, vc + dt_cell * gc, vc)
            dr_cell = dt_cell * vc
            norm = self._sums([(dr * dr).sum(dim=1)], [(dr_cell * dr_cell).sum(dim=(1, 2))])[:, 0].sqrt()
        else:
            norm = self._sums([(dr * dr).sum(dim=1)], [])[:, 0].sqrt()
        limited = norm > p.maxstep
        norm_row = norm[:, None].index_select(0, rows)
        dr = torch.where(limited[rows][:, None], p.maxstep * dr / norm_row, dr)
        st.v.copy_(v)
        st.x.copy_(torch.where(move_row, st.x + dr, st.x))
        if self.cell_filter:
            dr_cell = torch.where(limited[:, None, None], p.maxstep * dr_cell / norm[:, None, None], dr_cell)
            st.cell_velocity.copy_(vc)
            st.log_deformation.copy_(torch.where(move_cell, st.log_deformation + dr_cell / st.factor[:, None, None],
                                                 st.log_deformation))
        st.dt.copy_(dt)
        st.a.copy_(torch.where(grow, a * p.fa, torch.where(uphill, torch.full_like(a, p.astart), a)))
        st.npos.copy_(torch.where(downhill, st.npos + 1.0, torch.where(uphill, torch.zeros_like(st.npos), st.npos)))
        st.fresh.copy_(torch.where(moving, torch.zeros_like(st.fresh), st.fresh))
        st.steps.copy_(st.steps + moving.to(torch.float64))
        self._derive(st)
        return self._positions

    def step_post_force(self, forces: Tensor, energy: Tensor | None = None, state: SystemFIREState | None = None,
                        stress: Tensor | None = None) -> None:
        """The model's forces ``(rows, 3)``, energies ``(S,)`` (and, with a cell filter, stresses ``(S, 3, 3)`` in
        eV/Å³) at the positions of :meth:`step_pre_force`: each evaluated (active or pending) system's optimizer
        forces, largest force row and phase (converged below ``fmax``, else the step limit, else active; failed
        where its state is not finite); finished and idle systems keep theirs."""
        st = self.state if state is None else state
        rows = self.system_of_row
        evaluated = (st.phase == ACTIVE) | (st.phase == PENDING)
        evaluated_row = evaluated[rows][:, None] & self.occupied
        model_forces = forces.reshape(-1, 3).to(torch.float64)
        st.f.copy_(torch.where(evaluated_row, model_forces, st.f))
        if energy is not None:
            st.energy.copy_(torch.where(evaluated, energy.reshape(-1).to(torch.float64), st.energy))
        if self.cell_filter:
            # ASE's FrechetCellFilter.get_forces: the atoms' f F, and the cell's L(Y^T, virial F^-T) / factor with
            # the virial -V sigma of the current cell
            gradient = rows_times(model_forces, self._deformation.index_select(0, rows))
            sigma = stress.to(torch.float64)
            virial = -system_volumes(self._cells)[:, None, None] * sigma
            cell_gradient = expm_frechet_adjoint(
                st.log_deformation, torch.matmul(virial, self._inverse_deformation.transpose(1, 2)))
            evaluated_cell = evaluated[:, None, None]
            st.stress.copy_(torch.where(evaluated_cell, sigma, st.stress))
            st.cell_gradient.copy_(torch.where(evaluated_cell, cell_gradient / st.factor[:, None, None],
                                               st.cell_gradient))
        else:
            gradient = model_forces
        st.g.copy_(torch.where(evaluated_row, gradient, st.g))
        norms = torch.linalg.vector_norm(st.g, dim=1)
        largest = segment_max(torch.where(torch.isfinite(norms), norms, torch.full_like(norms, math.inf)),
                              self.ceilings, self.atoms)
        if self.cell_filter:
            cell_norms = torch.linalg.vector_norm(st.cell_gradient, dim=2).amax(dim=1)
            largest = torch.maximum(largest, torch.where(torch.isfinite(cell_norms), cell_norms,
                                                         torch.full_like(cell_norms, math.inf)))
        finite = torch.isfinite(largest) & torch.isfinite(st.energy)
        phase = torch.where(~finite, FAILED, torch.where(largest < self.fmax, CONVERGED,
                                                         torch.where(st.steps >= self.max_steps, STEP_LIMIT, ACTIVE)))
        st.fmax.copy_(torch.where(evaluated, largest, st.fmax))
        st.phase.copy_(torch.where(evaluated, phase.to(torch.float64), st.phase))

    # ------------------------------------------------------------ transactions
    def backup(self, state: SystemFIREState | None = None) -> None:
        self._backup.flat.copy_((self.state if state is None else state).flat)

    def restore(self, state: SystemFIREState | None = None) -> None:
        (self.state if state is None else state).flat.copy_(self._backup.flat)

    def snapshot(self) -> tuple:
        return self.state.flat.clone(), self._backup.flat.clone()

    def load_snapshot(self, snapshot: tuple) -> None:
        state, backup = snapshot
        self.state.flat.copy_(state)
        self._backup.flat.copy_(backup)
        self.sync()

    def new_state(self) -> SystemFIREState:
        return SystemFIREState(self.layout.rows, self.layout.count, self.cell_filter, self.device)

    def state_dict(self) -> dict:
        raise NotImplementedError("restart files of the batched relaxation are not implemented")

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError("restart files of the batched relaxation are not implemented")


class CellRelaxationBuilder(SlotUnionGraphBuilder):
    """:class:`~fastmd.execution.system_batch.SlotUnionGraphBuilder` whose systems' cells change inside the
    captured graph (a variable-cell relaxation).

    :meth:`set_cells` writes every cell table on the device. The skin test bounds how far any pair distance has
    moved since the candidate refresh: a pair within the cutoff now was within ``cutoff / (1 - e) + 2 delta`` of
    it at the refresh, ``delta`` the largest move of an atom in fractional coordinates times the refresh's cell
    and ``e = sqrt(|A|_1 |A|_inf)`` (``A = C_ref^-1 C - 1``) a bound on the spectral norm of its cell's
    deformation since then (``1 - e`` bounds the deformation's smallest singular value). The candidates hold
    while ``2 delta + cutoff e / (1 - e) <= skin`` and ``e < 1/2`` (the far self-loop padding, at least twice the
    cutoff long at the refresh, then stays beyond the cutoff), so the reported displacement is
    ``delta + cutoff e / (2 (1 - e))``, infinite from ``e = 1/2``. Fractional coordinates do not change with the
    cell, so the candidates' images and the wraps stay exact. A candidate refresh takes the current cells as the
    reference (the cell list's bins, the far self-loop images and the sort keys' image bound with them; eager)."""

    def __init__(self, *, cells, **kwargs) -> None:
        device = torch.device(kwargs.get("device", "cuda"))
        self.current_cells = torch.as_tensor(cells, dtype=torch.float64).reshape(-1, 3, 3).to(device).clone()
        super().__init__(cells=self.current_cells.clone(), **kwargs)
        self.reference_cells = self.current_cells.clone()
        self.reference_inverse = inverse33(self.reference_cells)
        self.row_reference_cells = self.row_cells.clone()
        self._eye = torch.eye(3, dtype=torch.float64, device=self.device)

    def _set_cell(self, slot: int, cell64: Tensor) -> None:
        super()._set_cell(slot, cell64)
        self.current_cells[slot].copy_(cell64)

    def set_cells(self, cells: Tensor) -> None:
        """Every system's cell ``cells`` (``[S, 3, 3]`` float64 on the device) into the tables, in place (device
        ops only, capture-safe): the float32 search cells and their inverses, each row's copies, the model's cells
        (``systems.cells``) and the batch's cell (the first system's)."""
        self.current_cells.copy_(cells)
        self.cell.copy_(cells)
        self.inv_cells.copy_(inverse33(cells))
        self.systems.cells.copy_(cells)
        rows = self.systems.system_of_row
        self.row_cells.copy_(self.cell.index_select(0, rows))
        self.row_inv_cells.copy_(self.inv_cells.index_select(0, rows))
        if self.row_model_cells is not None:
            self.row_model_cells.copy_(self.systems.cells.index_select(0, rows))
        self.batch.cell.copy_(self.systems.cells[0])

    def _displacement(self, positions: Tensor) -> Tensor:
        fractional = small_matmul(positions.unsqueeze(1), self.row_inv_cells).squeeze(1)
        moved = small_matmul(fractional.unsqueeze(1), self.row_reference_cells).squeeze(1)
        delta = torch.linalg.vector_norm(moved - self.reference_positions, dim=1).max()
        change = (torch.matmul(self.reference_inverse, self.current_cells) - self._eye).abs()
        deformation = torch.sqrt(change.sum(dim=1).amax(dim=1) * change.sum(dim=2).amax(dim=1))
        strain = torch.where(deformation < 0.5, self.atom_graph_cutoff * deformation / (1.0 - deformation),
                             torch.full_like(deformation, math.inf)).max()
        return delta + (0.5 * strain).to(delta.dtype)

    @torch.no_grad()
    def _take_reference_cells(self) -> None:
        """The current cells become the skin test's reference, with the cell list's bins, the far self-loop
        images and the sort keys' image bound (host work: a candidate refresh, eager)."""
        if self.sorted_edges and self._image_bound(self.cell) > self.sort_key_max_image:
            raise ValueError(f"a relaxed cell became thinner than the union's sort keys allow (image bound "
                             f"{self._image_bound(self.cell)} > {self.sort_key_max_image})")
        cells, radius = estimate_batch_cell_list_sizes(self.cells, self.pbc, self.candidate_cutoff)
        if cells > self._cell_list_cells:  # the cell list is eager only: a new one is no recapture
            self._cell_list_cells = int(cells)
            self.cell_list_cache = allocate_cell_list(self.n_real, int(cells), radius, self.device)
        else:
            self.cell_list_cache[1].copy_(radius)
        if self._loop_images is not None:  # self_loop_padding's far image (its longest lattice vector) of every cell
            lengths = self.cell.double().norm(dim=2)
            longest, axis = lengths.max(dim=1)
            multiple = torch.ceil(max(self.padding.min_length, 2.0 * self.atom_graph_cutoff) / longest).clamp(min=1)
            self._loop_images.zero_().scatter_(1, axis[:, None], multiple[:, None].to(torch.int32))
        self.reference_cells.copy_(self.current_cells)
        self.reference_inverse.copy_(inverse33(self.current_cells))
        self.row_reference_cells.copy_(self.row_cells)

    @torch.no_grad()
    def refresh_candidates(self, positions: Tensor) -> None:
        self._take_reference_cells()
        super().refresh_candidates(positions)

    @torch.no_grad()
    def build(self, positions: Tensor, *, rebuild_candidates: bool = True, predicate=None):
        if rebuild_candidates:
            self._take_reference_cells()
        return super().build(positions, rebuild_candidates=rebuild_candidates, predicate=predicate)


class SlotRelaxationRunner(SlotWholeStepRunner):
    """:class:`~fastmd.execution.system_batch.SlotWholeStepRunner` relaxing its systems with :class:`SystemFIRE`
    (a :class:`CellRelaxationBuilder` with a cell filter): each step the integrator's FIRE step, its cells into
    the builder (a cell filter), the union's evaluation (with each system's stress for a cell filter) and the
    integrator's convergence. The window record holds each system's phase (read with the window's flags in one
    copy: :attr:`phases` after :meth:`settle`); a system whose state turns non-finite is a failed phase, not an
    error. The mirrors are the integrator's state (positions, velocities, the stored forces and energies)."""

    def __init__(self, **kwargs) -> None:
        self._stress = None
        self.phases: tuple = ()
        super().__init__(**kwargs)

    @staticmethod
    def _union_shape(builder, integrator) -> tuple[int, int]:
        if not isinstance(builder, SlotUnionGraphBuilder) or not isinstance(integrator, SystemFIRE):
            raise TypeError("SlotRelaxationRunner needs a SlotUnionGraphBuilder and a SystemFIRE")
        if integrator.layout != builder.layout:
            raise ValueError("builder and integrator must hold the same slots")
        if integrator.cell_filter != isinstance(builder, CellRelaxationBuilder):
            raise ValueError("a cell filter relaxes with a CellRelaxationBuilder, a fixed cell without one")
        return builder.layout.count, 0

    def _health(self) -> Tensor:
        """``(S,)`` float64: each system's phase (``SystemFIRE.phase``)."""
        return self.integrator.phase

    @scoped_call
    def _evaluate(self, positions: Tensor, *, rebuild_candidates: bool, stress: bool = False):
        if stress:
            raise ValueError("SlotRelaxationRunner records no stress window")
        graph, status = self.builder.build(positions, rebuild_candidates=rebuild_candidates)
        if not self.integrator.cell_filter:
            energies, forces = system_energy_derivative(self.adapter, self.builder.batch, graph)
            return forces, energies, status
        energies, forces, stresses = system_energy_derivative(self.adapter, self.builder.batch, graph, stress=True)
        if self._stress is None:  # the first evaluation is eager (construction)
            self._stress = torch.zeros_like(stresses)
        self._stress.copy_(stresses.detach())
        return forces, energies, status

    def _stateful_step_body(self, *, rebuild_candidates: bool, stress: bool = False, mirror: bool = True) -> object:
        integrator = self.stateful_integrator
        positions = integrator.step_pre_force()
        if integrator.cell_filter:
            self.builder.set_cells(integrator.cells)
        forces, energy, status = self._evaluate(positions, rebuild_candidates=rebuild_candidates,
                                                **({"stress": True} if stress else {}))
        with torch.no_grad():
            integrator.step_post_force(forces.detach(), energy=energy.detach(), stress=self._stress)
            if not mirror:
                return status
            self.positions.copy_(positions)
            self.momenta.copy_(integrator.momenta)
            self.forces.copy_(integrator.forces)
            self.potential_energy.copy_(integrator.potential_energy)
        return status

    def _sync_cells(self) -> None:
        self.integrator.sync()
        if self.integrator.cell_filter:
            self.builder.set_cells(self.integrator.cells)

    def _restore_backup(self) -> None:
        super()._restore_backup()
        self._sync_cells()

    def _capture(self) -> None:
        try:
            super()._capture()
        finally:  # the warmup windows moved the builder's cells; the integrator's state is the saved one
            self._sync_cells()

    def rewind(self) -> None:
        super().rewind()
        self._sync_cells()

    def settle(self, *, record: bool = False, flags: tuple | None = None) -> None:
        """:meth:`WholeStepGraphRunner.settle` (an overflow's fallback, the counts, the proactive candidate
        refresh), and the systems' phases (:attr:`phases`) from the same copy, or from the integrator after a
        fallback window (whose eager steps write no record)."""
        packed = tuple(self._window_record.tolist()) if flags is None else tuple(flags)
        fallbacks = self.fallback_windows
        WholeStepGraphRunner.settle(self, record=record, flags=packed[:4])
        self.phases = (packed[4:] if self.fallback_windows == fallbacks
                       else tuple(self.integrator.phase.tolist()))

    @torch.no_grad()
    def reset(self, *, positions: Tensor, momenta: Tensor, forces: Tensor, potential_energy: Tensor,
              reset_counters: bool = True) -> None:
        self._sync_cells()
        super().reset(positions=positions, momenta=momenta, forces=forces, potential_energy=potential_energy,
                      reset_counters=reset_counters)


def relaxation_builder(adapter, *, cell_filter: bool, layout: SlotLayout, cells, atomic_numbers: Tensor, atoms,
                       u_capacity: int, candidate_skin: float, device, admit_cells=None) -> SlotUnionGraphBuilder:
    """The slot union builder of ``adapter`` (:func:`~fastmd.execution.system_batch.slot_union_builder`'s
    options), a :class:`CellRelaxationBuilder` for a cell filter; ``admit_cells``: the cells of the systems
    written later (the sort keys hold them)."""
    from fastmd.execution.replica_batch import graph_builder_options

    cutoff = float(adapter.topology_spec().cutoff)
    kind = CellRelaxationBuilder if cell_filter else SlotUnionGraphBuilder
    return kind(layout=layout, cells=cells, atomic_numbers=atomic_numbers, atoms=atoms, u_capacity=int(u_capacity),
                atom_graph_cutoff=cutoff, candidate_cutoff=cutoff + candidate_skin, device=device,
                padding=adapter.padding_rule(), prepare=adapter.prepare, admit_cells=admit_cells,
                **graph_builder_options(adapter))
