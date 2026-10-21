"""The engine-owned derivative: forces and stress of a model that returns its energy only.

A model with a model-owned derivative (MatRIS) computes forces and stress in
its own call, ``adapter.evaluate``. A model with an energy tap
(:class:`~fastmd.adapters.protocol.EnergyTap`) returns per-atom energies
only, one with a whole-system energy tap
(:class:`~fastmd.adapters.protocol.SystemEnergyTap`) the energy of the
system only; the runtime differentiates them here, with one
``torch.autograd.grad(create_graph=False)`` per evaluation:

* the leaf is ``batch.positions`` (a detached view of the builder's buffer,
  no copy), so F = -dE/dx;
* for a stress, a zero ``[3, 3]`` strain leaf deforms the positions and the
  cell by its symmetric part, ``x (1 + e)`` and ``cell (1 + e)``; the stress is
  dE/de / volume, in eV/Å³;
* the energy is the FP64 sum of the real rows' energies (rows past
  ``n_real`` are sink atoms that add nothing and get no gradient), or the
  system energy in FP64 for a whole-system tap.

Nothing here synchronizes the host, so the whole-step and model-graph
runners capture it together with the model's call. The runners call
:class:`EnergyDerivative` where they call a model-owned ``evaluate`` and read
the same mapping.
"""
from __future__ import annotations

import torch
from torch import Tensor

from fastmd.adapters.protocol import EnergyTap
from fastmd.adapters.protocol import GraphBatch
from fastmd.adapters.protocol import ModelOwnedDerivative
from fastmd.adapters.protocol import SystemEnergyTap


def stress_unit_scale(adapter) -> float:
    """Convert the declared model stress to ASE units; energy taps already use ASE units."""
    if not isinstance(adapter, ModelOwnedDerivative):
        return 1.0
    contract = getattr(adapter, "capture_contract", None)
    unit = contract().stress_unit if contract is not None else "GPa"
    if unit == "eV/A^3":
        return 1.0
    if unit == "GPa":
        from ase import units

        return units.GPa
    raise ValueError(f"Unsupported stress unit {unit!r}")


def engine_derivative(adapter) -> "EnergyDerivative | None":
    """The derivative the runtime runs for ``adapter``: None for a model-owned derivative (the
    runtime calls ``adapter.evaluate``), else the engine's own over the adapter's energy tap (per
    atom, or of the whole system)."""
    if isinstance(adapter, ModelOwnedDerivative):
        return None
    if isinstance(adapter, (EnergyTap, SystemEnergyTap)):
        return EnergyDerivative(adapter)
    raise TypeError(f"model adapter {getattr(adapter, 'name', adapter)!r} has neither a model-owned derivative "
                    "(evaluate, extensive_energy) nor an energy tap (node_energy or system_energy)")


def edges_only_builder_options(adapter) -> dict:
    """The options of an edges-only fixed-capacity builder for ``adapter`` beyond the defaults: float64
    positions and cell for a float64 model the engine differentiates, and no per-build sort for a model
    that needs its edge rows grouped by centre only (``TopologySpec.edge_order``)."""
    topology = adapter.topology_spec()
    options = {}
    if engine_derivative(adapter) is not None and topology.model_dtype == "float64":
        options["positions_dtype"] = torch.float64
    if topology.edge_order != "sorted":
        options["sorted_edges"] = False
    return options


def _times(rows: Tensor, matrix: Tensor) -> Tensor:
    """``rows @ matrix`` for a ``[3, 3]`` matrix, row by row (no GEMM: no TF32, and an edge's or
    atom's arithmetic does not depend on the row count)."""
    return rows[:, 0:1] * matrix[0] + rows[:, 1:2] * matrix[1] + rows[:, 2:3] * matrix[2]


def cell_volume(cell: Tensor) -> Tensor:
    """|det(cell)| as the triple product (elementwise: no solver, no host synchronization)."""
    return torch.dot(cell[0], torch.linalg.cross(cell[1], cell[2])).abs()


class EnergyDerivative:
    """The engine-owned derivative of an energy-tap model (see the module docstring).

    ``derivative(batch, inputs, task=..., n_real=...)`` evaluates
    ``adapter.node_energy(inputs, positions, cell)`` (or, for a whole-system
    tap, ``adapter.system_energy(inputs, positions, cell)``) over ``inputs =
    adapter.prepare(batch)`` and returns ``{"e": [1] FP64 energy, "f":
    [forces [rows, 3]], "s": [stress [3, 3] in eV/Å³]}`` (``"s"`` for a task
    with ``"s"``), the layout of a model-owned ``evaluate``. The forces and the
    stress have the dtype of ``batch.positions``.
    """

    stress_unit = "eV/A^3"

    def __init__(self, adapter: EnergyTap | SystemEnergyTap) -> None:
        self.adapter = adapter
        # Per-atom energies (EnergyTap, which wins if both are present), else the system's energy.
        self._per_atom = isinstance(adapter, EnergyTap)

    def extensive_energy(self, energy: Tensor, atom_counts) -> Tensor:
        """The energy as returned: the sum of the atoms' energies is the system's."""
        return energy

    def __call__(self, batch: GraphBatch, inputs, *, task: str = "ef", n_real: int | None = None) -> dict:
        stress = "s" in task
        with torch.enable_grad():
            leaf = batch.positions.detach().requires_grad_(True)
            positions, cell = leaf, batch.cell
            leaves = (leaf,)
            if stress:
                strain = torch.zeros((3, 3), dtype=leaf.dtype, device=leaf.device, requires_grad=True)
                symmetric = 0.5 * (strain + strain.transpose(0, 1))
                positions = leaf + _times(leaf, symmetric)
                cell = cell + _times(cell, symmetric)
                leaves = (leaf, strain)
            if self._per_atom:
                energies = self.adapter.node_energy(inputs, positions, cell)
                rows = energies.shape[0] if n_real is None else n_real
                energy = energies[:rows].to(torch.float64).sum()
            else:
                energy = self.adapter.system_energy(inputs, positions, cell).to(torch.float64).sum()
            grads = torch.autograd.grad(energy, leaves)
        out = {"e": energy.detach().reshape(1), "f": [-grads[0]]}
        if stress:
            out["s"] = [grads[1] / cell_volume(batch.cell)]
        return out
