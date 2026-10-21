"""A test-only edges-only model behind the adapter protocol.

``SmoothPairPotential`` is a tiny pair potential with an exact cutoff: every
directed edge adds ``0.5 * s_i * s_j * (1 - (r / rc)^2)^power`` for
``r < rc`` and exactly nothing (value and gradient) at and beyond ``rc``.
With ``power`` 12 an edge within ~1e-5 of the cutoff also adds exactly 0 in
FP32 (the power underflows). ``PairPotentialAdapter`` puts it behind
:class:`~fastmd.adapters.protocol.ModelAdapter` with a model-owned
derivative: it reads the builders' :class:`~fastmd.adapters.protocol.GraphBatch`
directly and differentiates its own energy, so the edges-only builders,
padding rules, skin reuse and the eager, model-graph and whole-step paths
run without a foreign package. It declares its capture contract (sync-free,
exact cutoff), so the capability negotiation plans it like another model.
``SystemTapPairAdapter`` is the same potential behind a whole-system energy
tap (:class:`~fastmd.adapters.protocol.SystemEnergyTap`): the engine
differentiates its energy.
"""
from __future__ import annotations

import torch
from ase import units
from torch import Tensor, nn

from fastmd.adapters.protocol import CaptureContract
from fastmd.adapters.protocol import GraphBatch
from fastmd.adapters.protocol import PaddingRule
from fastmd.adapters.protocol import TopologySpec

# Fixed shapes and no host reads (capturable), exactly 0 at and beyond the cutoff, FP32 arithmetic.
CONTRACT = CaptureContract(sync_free=True, exact_cutoff=True, accuracy_class="fp32_noise",
                           force_mode="model_owned", stress_unit="GPa")


class SmoothPairPotential(nn.Module):
    def __init__(self, cutoff: float = 5.0, power: int = 12, strength: float = 0.05) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.power = int(power)
        # Per-element strength s_Z: a parameter, so the module has weights like a model.
        self.species_strength = nn.Parameter(strength * (1.0 + torch.arange(119, dtype=torch.float32) / 20.0))

    def edge_energies(self, vectors: Tensor, center: Tensor, neighbor: Tensor, species: Tensor) -> Tensor:
        """Per-edge energy of the directed edges with these vectors (exactly 0 at and beyond the cutoff)."""
        r = torch.linalg.vector_norm(vectors, dim=1)
        x2 = (r / self.cutoff) ** 2
        strength = self.species_strength[species.long()]
        pair = strength[center] * strength[neighbor]
        inside = r < self.cutoff
        base = torch.where(inside, 1.0 - x2, torch.zeros_like(x2))
        return torch.where(inside, 0.5 * pair * base**self.power, torch.zeros_like(base))

    def forward(self, batch: GraphBatch, n_real: int | None = None, stress: bool = False):
        """(energy [1] float32, forces [rows, 3], stress [3, 3] in GPa or None) of ``batch``."""
        with torch.enable_grad():
            positions = batch.positions.detach().requires_grad_(True)
            strain = torch.zeros((3, 3), dtype=positions.dtype, device=positions.device, requires_grad=stress)
            deformation = torch.eye(3, dtype=positions.dtype, device=positions.device) + strain
            center = batch.edge_index[:, 0].long()
            neighbor = batch.edge_index[:, 1].long()
            # The shift row by row (no GEMM): an edge's arithmetic does not depend on the row count.
            image, cell = batch.edge_image, batch.cell
            shift = image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]
            vectors = positions.index_select(0, neighbor) + shift - positions.index_select(0, center)
            if stress:
                vectors = vectors @ deformation
            if batch.edge_vec_override is not None:
                rows = torch.arange(vectors.shape[0], device=vectors.device)
                vectors = torch.where((rows < batch.n_edges)[:, None], vectors, batch.edge_vec_override)
            energies = self.edge_energies(vectors, center, neighbor, batch.species)
            # Per-atom energies in FP64: the scatter's grouping (which depends on where padding rows
            # sit) then cannot show in the FP32 total.
            atom_energy = torch.zeros(positions.shape[0], dtype=torch.float64, device=energies.device)
            atom_energy = atom_energy.index_add(0, center, energies.double())
            rows = positions.shape[0] if n_real is None else n_real
            energy = atom_energy[:rows].sum().float().reshape(1)
            inputs = (positions, strain) if stress else (positions,)
            grads = torch.autograd.grad(energy, inputs)
        forces = -grads[0]
        virial = None
        if stress:
            volume = torch.linalg.det(batch.cell).abs()
            virial = grads[1] / volume / units.GPa
        return energy.detach(), forces.detach(), virial


class PairPotentialAdapter:
    """``SmoothPairPotential`` as a :class:`~fastmd.adapters.protocol.ModelAdapter` (edges only).

    ``evaluate`` takes the runtime's call of a model-owned derivative:
    ``evaluate([inputs], task=..., is_training=False, n_real=...)`` with
    ``inputs = prepare(batch)``, and returns ``{"e": [1] extensive energy,
    "f": [forces], "s": [stress in GPa]}`` (per-system lists) like MatRIS.
    """

    name = "pair_potential"
    compiled = False

    def __init__(self, module: SmoothPairPotential | None = None, padding: PaddingRule | None = None,
                 contract: CaptureContract = CONTRACT) -> None:
        self.module = SmoothPairPotential() if module is None else module
        self._padding = PaddingRule("far_self_loops") if padding is None else padding
        self._contract = contract

    def topology_spec(self) -> TopologySpec:
        return TopologySpec(cutoff=self.module.cutoff, num_interactions=1, triplet_cutoff=None)

    def padding_rule(self) -> PaddingRule:
        return self._padding

    def prepare(self, batch: GraphBatch) -> GraphBatch:
        return batch

    def capture_contract(self) -> CaptureContract:
        return self._contract

    def isolated_atom_policy(self) -> str:
        return "reference"  # an atom without neighbours has energy 0, its (zero) reference

    def reference_energies(self):
        return None

    def extensive_energy(self, energy, atom_counts):
        return energy

    def evaluate(self, graphs, *, task: str = "ef", is_training: bool = False, n_real: int | None = None,
                 handle_isolated_atoms: bool = True) -> dict:
        (batch,) = graphs
        energy, forces, stress = self.module(batch, n_real=n_real, stress="s" in task)
        out = {"e": energy, "f": [forces]}
        if stress is not None:
            out["s"] = [stress]
        return out


class SystemTapPairAdapter:
    """``SmoothPairPotential`` behind a whole-system energy tap: ``system_energy`` returns the potential's
    energy as one FP64 number and the engine-owned derivative computes forces and stress (eV/Å³). The
    per-edge arithmetic and sums are those of :class:`PairPotentialAdapter`'s call."""

    name = "pair_potential_system_tap"
    compiled = False

    def __init__(self, module: SmoothPairPotential | None = None) -> None:
        self.module = SmoothPairPotential() if module is None else module

    def topology_spec(self) -> TopologySpec:
        return TopologySpec(cutoff=self.module.cutoff, num_interactions=1, triplet_cutoff=None)

    def padding_rule(self) -> PaddingRule:
        return PaddingRule("far_self_loops")

    def prepare(self, batch: GraphBatch) -> GraphBatch:
        return batch

    def capture_contract(self) -> CaptureContract:
        return CaptureContract(sync_free=True, exact_cutoff=True, accuracy_class="fp32_noise",
                               force_mode="engine_autograd", stress_unit="eV/A^3")

    def isolated_atom_policy(self) -> str:
        return "reference"

    def reference_energies(self):
        return None

    def system_energy(self, inputs: GraphBatch, positions: Tensor, cell: Tensor) -> Tensor:
        batch = inputs
        center = batch.edge_index[:, 0].long()
        neighbor = batch.edge_index[:, 1].long()
        image = batch.edge_image
        shift = image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]
        vectors = positions.index_select(0, neighbor) + shift - positions.index_select(0, center)
        energies = self.module.edge_energies(vectors, center, neighbor, batch.species)
        atom_energy = torch.zeros(positions.shape[0], dtype=torch.float64, device=energies.device)
        atom_energy = atom_energy.index_add(0, center, energies.double())
        return atom_energy[:batch.n_real].sum().reshape(1)


class UnionPairAdapter:
    """``SmoothPairPotential`` behind a per-atom energy tap: ``node_energy`` returns each atom's FP64 share
    of the potential (half of each of its pair terms; the per-edge arithmetic of :class:`SystemTapPairAdapter`)
    and the engine differentiates their sum. ``system_batching`` returns ``declare`` ("union": an atom's
    energy depends on its own system's atoms only, so replicas of a system may run as one disjoint union;
    None: not declared). ``tether`` (eV/Å²) adds to each atom's energy the squared distance from the centroid
    of all rows of the batch, a term that is not local: in a union the centroid is shared by its systems.
    ``batch_energy`` (eV) adds that energy shared out over all rows of the batch: a constant, so no force
    changes, but in a union each system gets only its share. ``triplet_cutoff`` (Å) gives the topology a
    line graph (padded with far self-loops), over whose triplets ``angle`` (eV) adds to each centre atom
    ``angle * w(r_1) * w(r_2) * cos(theta)^2`` of its two bonds, ``w(r) = (1 - (r / triplet_cutoff)^2)^2``
    (exactly 0 at and beyond the triplet cutoff: a padded triplet, on two far self-loops, adds nothing).
    ``edge_shifts`` (the default) declares that ``node_energy`` takes the engine's per-edge shifts
    (``EdgeShiftInput``: systems with cells of their own); ``read_cell`` makes it compute them from ``cell`` all
    the same (a false declaration)."""

    name = "pair_potential_union"
    compiled = False

    def __init__(self, module: SmoothPairPotential | None = None, *, declare: str | None = "union",
                 tether: float = 0.0, batch_energy: float = 0.0, triplet_cutoff: float | None = None,
                 angle: float = 0.0, edge_shifts: bool = True, read_cell: bool = False) -> None:
        self.module = SmoothPairPotential() if module is None else module
        self.declare = declare
        self.edge_shifts = bool(edge_shifts)
        self.read_cell = bool(read_cell)
        self.tether = float(tether)
        self.batch_energy = float(batch_energy)
        self.triplet_cutoff = None if triplet_cutoff is None else float(triplet_cutoff)
        self.angle = float(angle)

    def topology_spec(self) -> TopologySpec:
        return TopologySpec(cutoff=self.module.cutoff, num_interactions=1, triplet_cutoff=self.triplet_cutoff)

    def padding_rule(self) -> PaddingRule:
        return PaddingRule("far_self_loops")

    def prepare(self, batch: GraphBatch) -> GraphBatch:
        return batch

    def capture_contract(self) -> CaptureContract:
        return CaptureContract(sync_free=True, exact_cutoff=True, accuracy_class="fp32_noise",
                               force_mode="engine_autograd", stress_unit="eV/A^3")

    def isolated_atom_policy(self) -> str:
        return "reference"

    def reference_energies(self):
        return None

    def system_batching(self) -> str | None:
        return self.declare

    def edge_shift_input(self) -> bool:
        return self.edge_shifts

    def node_energy(self, inputs: GraphBatch, positions: Tensor, cell: Tensor,
                    edge_shifts: Tensor | None = None) -> Tensor:
        batch = inputs
        center = batch.edge_index[:, 0].long()
        neighbor = batch.edge_index[:, 1].long()
        image = batch.edge_image
        shift = (image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]
                 if edge_shifts is None or self.read_cell else edge_shifts)
        vectors = positions.index_select(0, neighbor) + shift - positions.index_select(0, center)
        energies = self.module.edge_energies(vectors, center, neighbor, batch.species)
        atom_energy = torch.zeros(positions.shape[0], dtype=torch.float64, device=energies.device)
        atom_energy = atom_energy.index_add(0, center, energies.double())
        if self.triplet_cutoff is not None and self.angle:
            # The triplet rows: (centre atom, first pair, first directed edge, second pair, second directed edge).
            triplets = batch.triplets
            first = vectors.index_select(0, triplets[:, 2].long())
            second = vectors.index_select(0, triplets[:, 4].long())
            r1, r2 = (torch.linalg.vector_norm(bond, dim=1) for bond in (first, second))
            weights = []
            for r in (r1, r2):
                inside = r < self.triplet_cutoff
                base = torch.where(inside, 1.0 - (r / self.triplet_cutoff) ** 2, torch.zeros_like(r))
                weights.append(torch.where(inside, base * base, torch.zeros_like(r)))
            cosine = (first * second).sum(dim=1) / (r1 * r2)
            term = self.angle * weights[0] * weights[1] * cosine * cosine
            atom_energy = atom_energy.index_add(0, triplets[:, 0].long(), term.double())
        if self.tether:
            offset = positions - positions.mean(dim=0)
            atom_energy = atom_energy + self.tether * (offset * offset).sum(dim=1).double()
        if self.batch_energy:
            atom_energy = atom_energy + self.batch_energy / positions.shape[0]
        return atom_energy
