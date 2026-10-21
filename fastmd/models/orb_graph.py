"""Capturable ORB direct/v2 and conservative models with exact nearest-K graphs.

Every evaluation selects the nearest K per sender with fixed tensor shapes.
Rejected edges become far self-loops; sigmoid cutoff gates and ZBL make their
contribution exactly zero. Checkpoint weights and neighbor caps are unchanged.
Only one nonsingular periodic cell is supported, as in the GPU MD builders.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import torch
from fastmd.models.orb import ORBAdapter, ORBInput, ORB_VERSIONS, _sync_free_aggregation
from fastmd.adapters.patches import PatchRecord, apply_patches, revert_patches
from fastmd.adapters.protocol import GraphBatch, KernelChoice

_DIRECT_PINS = tuple(
    PatchRecord(name=name, target=target, distribution='orb-models', versions=ORB_VERSIONS,
                source_sha256=digest, reason=reason)
    for name, target, digest, reason in (
        ('orb.direct_predict', 'orb_models.forcefield.direct_regressor.DirectForcefieldRegressor.predict',
         '89e3f31e23f2a526c582b3d429e6271bf08493e512ee54f09d4fb9635228534f',
         'direct energy, force, stress heads and additive ZBL; omit unused confidence'),
        ('orb.direct_force_forward', 'orb_models.forcefield.forcefield_heads.ForceHead.forward',
         '1af970141eefc2723bb59b0258c753bf78215112698e2065e55ea8e56cdfac22',
         'periodic force head: MLP then subtract mean, with no torque correction'),
        ('orb.direct_force_predict', 'orb_models.forcefield.forcefield_heads.ForceHead.predict',
         '8b18804291d5dcdb40d2cce1fc940a2d990f55cf232636c00e829c79b19e5f7f',
         'force denormalization follows mean subtraction'),
        ('orb.direct_force_mean', 'orb_models.forcefield.forcefield_utils.maybe_remove_net_force_and_torque',
         'f7f62f0f64a63a26f259e747a9a4bf93c5c4a128d9c1d4319a69fad21b4faede',
         'single graph mean without dynamic repeat_interleave or device PBC check'),
    )
)


@dataclass
class CappedORBInput:
    base: ORBInput
    edge_rows: torch.Tensor
    far_vector: torch.Tensor
    stress_index: torch.Tensor
    compact_senders: torch.Tensor | None = None
    compact_n_edge: torch.Tensor | None = None

    @property
    def batch(self):
        return self.base.batch


def nearest_k_mask(senders, vectors, rows, n_atoms, max_neighbors, cutoff):
    """Stable nearest-K selection with fixed outputs, including empty lists.

    Selection is discrete; retained vectors still carry position/strain gradients.
    No boolean indexing, dynamic bincount, nonzero, or host reads are used.
    """
    distances = vectors.detach().square().sum(-1)
    order = torch.argsort(distances, stable=True)
    order = order[torch.argsort(senders[order], stable=True)]
    sorted_senders = senders[order]
    starts = torch.full((n_atoms,), len(rows), dtype=torch.long, device=senders.device)
    starts.scatter_reduce_(0, sorted_senders, rows, reduce='amin', include_self=True)
    selected = (rows - starts[sorted_senders] < max_neighbors) & (distances[order] < cutoff * cutoff)
    return torch.zeros_like(selected).scatter_(0, order, selected)


class ORBGraphAdapter:
    """Factory/base separating direct-head and conservative energy capabilities."""
    name = 'orb_graph'

    def __new__(cls, module, *, checkpoint=None):
        if cls is ORBGraphAdapter:
            cls = (ORBConservativeGraphAdapter if hasattr(module, 'grad_forces_name')
                   else ORBDirectGraphAdapter)
        return object.__new__(cls)

    def __init__(self, module, *, checkpoint=None):
        self._base = ORBAdapter(module, checkpoint=checkpoint)
        self.module = self._base.module
        self.checkpoint = self._base.checkpoint
        self.cutoff = float(module.system_config.radius)
        self.max_neighbors = int(module.system_config.max_num_neighbors)
        self._direct_patches = []
        self._energy_installation = None
        if self.cutoff != 6.0 or self.max_neighbors < 1 or any(
            not layer._distance_cutoff or layer._attention_gate != 'sigmoid'
            for layer in module.model.gnn_stacks
        ):
            self.close()
            raise ValueError('ORB capped capture requires the validated 6 A sigmoid cutoff architecture')
        if not hasattr(module, 'grad_forces_name'):
            self._direct_patches = apply_patches(_DIRECT_PINS)
            declined = [p for p in self._direct_patches if not p.applied]
            if declined:
                self.close()
                raise RuntimeError('ORB direct inference source changed: ' + '; '.join(p.reason for p in declined))

    @property
    def compiled(self):
        if self._energy_installation is not None:
            return self._energy_installation.backend == 'inductor'
        return self._base.compiled

    @property
    def upstream(self):
        return self._base.upstream

    def topology_spec(self):
        # The builder retains every radius edge; this adapter applies the cap.
        return replace(self._base.topology_spec(), max_neighbors=None)

    def padding_rule(self):
        return self._base.padding_rule()

    def capacity_step(self):
        return self._base.capacity_step()

    def isolated_atom_policy(self):
        return 'model'

    def reference_energies(self):
        return self._base.reference_energies()

    def capture_contract(self):
        contract = replace(self._base.capture_contract(), force_mode=(
            'engine_autograd' if hasattr(self.module, 'grad_forces_name') else 'direct_head'))
        if self._energy_installation is not None:
            contract = replace(contract, compiled='capture_safe' if self.compiled else 'none')
        return contract

    def system_inputs(self):
        inputs = {'max_neighbors': self.max_neighbors, 'checkpoint': self.checkpoint,
                'neighbor_selection': ('nearest_k_fixed_shape_v1' if self._base.operator_backend == 'upstream'
                                       else 'nearest_k_compact_v2'),
                **self._base.system_inputs()}
        if self._energy_installation is not None:
            handle = self._energy_installation
            inputs['energy_compile'] = {'backend': handle.backend,
                'options': dict(handle.options), 'pooling': 'single_system_mean_sum',
                'source_pins': dict(handle.source_pins)}
        return inputs

    def enable_operator_rewrites(self, backend='torch', *, steady_options=None):
        if self._energy_installation is not None:
            raise RuntimeError('Disable energy compilation before changing ORB operator rewrites')
        self._base.enable_operator_rewrites(backend, steady_options=steady_options)

    def enable_energy_compile(self, *, autotune=False):
        """Explicit conservative FP32 energy compilation; select before prepare.

        Requires Torch operator rewrites. ``autotune`` considers ATEN/Triton GEMM
        implementations with the same FP32 precision. Compilation is lazy; first
        evaluation may fail and is deliberately not replaced by an eager fallback.
        """
        if self._energy_installation is not None:
            raise RuntimeError('ORB energy compilation is already enabled')
        from fastmd.models.orb_ops.orb_energy import install
        self._energy_installation = install(self, autotune=autotune)

    def disable_energy_compile(self):
        """Restore the previous backbone and energy implementation before prepare."""
        if self._energy_installation is None:
            return
        if self._base._in_use:
            raise RuntimeError('Disable ORB energy compilation before prepare/capture; close after graph use')
        self._energy_installation.restore()
        self._energy_installation = None

    def kernel_backends(self):
        if self._energy_installation is not None:
            return (KernelChoice(site='energy', options=('inductor', 'eager'), reference='eager'),)
        return self._base.kernel_backends()

    def select_kernel(self, site, option):
        if self._energy_installation is not None:
            if site != 'energy':
                raise ValueError('Whole-energy compilation owns the energy kernel; disable it before selecting backbone')
            if option not in ('inductor', 'eager'):
                raise ValueError('ORB energy kernel options are inductor/eager')
            if option == self._energy_installation.backend:
                return
            if self._base._in_use:
                raise RuntimeError('Select ORB energy kernels before prepare/capture')
            self._energy_installation.select_backend(option)
            return
        self._base.select_kernel(site, option)

    def upstream_patches(self):
        return (self._base.upstream_patches() + self._direct_patches +
                ([] if self._energy_installation is None else self._energy_installation.outcomes))

    def close(self):
        if self._energy_installation is not None:
            self._energy_installation.restore()
            self._energy_installation = None
        revert_patches(self._direct_patches)
        self._base.close()

    def system_batching(self):
        """Capped graphs and direct force heads retain their single-system contract."""
        return None

    def prepare(self, batch):
        if batch.padding is not None and batch.padding.kind != 'far_self_loops':
            raise ValueError('ORB capped graphs require far_self_loops padding')
        # Outside capture. Do not silently bypass nonperiodic torque correction.
        if not bool(torch.isfinite(batch.cell).all()) or float(torch.linalg.det(batch.cell).abs()) == 0:
            raise ValueError('ORBGraphAdapter requires a nonsingular periodic cell')
        base = self._base.prepare(batch)
        device = batch.species.device
        compact = (self._base.operator_backend != 'upstream'
                   and batch.n_real * self.max_neighbors < len(batch.edge_index))
        compact_senders = (torch.arange(batch.n_real, device=device).repeat_interleave(self.max_neighbors)
                           if compact else None)
        return CappedORBInput(
            base, torch.arange(batch.edge_index.shape[0], device=device),
            torch.tensor([2 * self.cutoff, 0., 0.], dtype=batch.positions.dtype, device=device),
            torch.tensor([[0, 5, 4], [5, 1, 3], [4, 3, 2]], device=device),
            compact_senders,
            torch.tensor([len(compact_senders)], device=device) if compact else None)

    def _graph(self, inputs, positions, cell):
        from orb_models.forcefield.base import AtomGraphs
        base, batch = inputs.base, inputs.batch
        senders, receivers = batch.edge_index.long().unbind(1)
        images = batch.edge_image.to(positions.dtype)
        shifts = images[:, 0:1] * cell[0] + images[:, 1:2] * cell[1] + images[:, 2:3] * cell[2]
        vectors = positions[receivers] - positions[senders] + shifts
        if inputs.compact_senders is not None:
            if self._base.operator_backend == 'triton':
                from fastmd.models.orb_ops.triton_orb_inference import compact_topk
            else:
                from fastmd.models.orb_ops.orb_inference import compact_topk_torch as compact_topk
            selected = compact_topk(senders, vectors.detach().square().sum(-1), len(positions),
                                    self.max_neighbors, self.cutoff, batch.n_edges).reshape(-1)
            keep = selected >= 0
            indices = selected.clamp_min(0)
            vectors, receivers, images = vectors[indices], receivers[indices], images[indices]
            senders = inputs.compact_senders
            n_edge = inputs.compact_n_edge
        else:
            keep = nearest_k_mask(senders, vectors, inputs.edge_rows, len(positions),
                                  self.max_neighbors, self.cutoff)
            if batch.n_edges is not None:
                keep = keep & (inputs.edge_rows < batch.n_edges.reshape(()))
            n_edge = base.n_edge
        # Rejected rows have zero position/strain derivative and zero messages/ZBL.
        vectors = torch.where(keep[:, None], vectors, inputs.far_vector)
        receivers = torch.where(keep, receivers, senders)
        images = torch.where(keep[:, None], images, 0)
        numbers = batch.species.long()
        return AtomGraphs(
            senders=senders, receivers=receivers, n_node=base.n_node, n_edge=n_edge,
            node_features={'positions': positions, 'atomic_numbers': numbers,
                           'atomic_numbers_embedding': base.one_hot.index_select(0, numbers)},
            edge_features={'vectors': vectors, 'unit_shifts': images},
            system_features={'cell': cell.unsqueeze(0)},
            node_targets=None, edge_targets=None, system_targets=None, system_id=None,
            fix_atoms=None, tags=None, radius=self.cutoff, max_num_neighbors=base.max_num_neighbors)


class ORBConservativeGraphAdapter(ORBGraphAdapter):
    """The engine differentiates the energy on retained KNN edges."""
    def system_energy(self, inputs, positions, cell):
        graph = self._graph(inputs, positions, cell)
        with _sync_free_aggregation():
            features = self.module.model(graph)['node_features']
            head = self.module.heads[self.module.energy_name]
            energy = head.denormalize(head(features, graph), graph)
            if self.module.pair_repulsion:
                energy = energy + self.module.pair_repulsion_fn(graph)['energy']
        return energy


class ORBDirectGraphAdapter(ORBGraphAdapter):
    """Preserve the checkpoint's direct force and stress heads."""
    def extensive_energy(self, energy, atom_counts):
        return energy

    def evaluate(self, inputs, *, task='efs', **kwargs):
        from orb_models.forcefield import segment_ops
        if len(inputs) != 1:
            raise ValueError('ORBGraphAdapter supports one MD system')
        inputs = inputs[0]
        if isinstance(inputs, GraphBatch):
            inputs = self.prepare(inputs)
        graph = self._graph(inputs, inputs.batch.positions, inputs.batch.cell)
        with torch.no_grad(), _sync_free_aggregation():
            features = self.module.model(graph)['node_features']
            energy = self.module.heads['energy'].predict(features, graph)
            # Derived from orb-models 0.5.4 ForceHead.forward/predict and
            # maybe_remove_net_force_and_torque, Apache-2.0, Orbital Materials.
            # A single periodic graph needs only a broadcast of its mean; its
            # upstream non-PBC torque branch has no effect for this cell.
            head = self.module.heads['forces']
            forces = head.mlp(features)
            if head.remove_mean:
                forces = forces - segment_ops.aggregate_nodes(forces, graph.n_node, reduction='mean')
            forces = head.normalizer.inverse(forces)
            stress = self.module.heads['stress'].predict(features, graph) if 's' in task else None
            if self.module.pair_repulsion:
                repulsion = self.module.pair_repulsion_fn(graph)
                energy = energy + repulsion['energy']
                forces = forces + repulsion['forces']
                if stress is not None:
                    stress = stress + repulsion['stress']
        out = {'e': energy.reshape(1), 'f': [forces], 'atoms_per_graph': graph.n_node}
        if stress is not None:
            out['s'] = stress.reshape(-1)[inputs.stress_index].unsqueeze(0)
        return out
