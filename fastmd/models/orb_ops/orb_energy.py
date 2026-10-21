"""Explicit FP32 whole-energy compilation for a single conservative ORB graph.

EnergyHead.forward and ZBL's energy lines follow orb-models 0.5.4, copyright
Orbital Materials, Inc., Apache-2.0. Single-system mean pooling avoids the
scatter count mutation that was misordered by the tested GPU Inductor version.
No upstream class, process precision flag, checkpoint parameter, or unrelated
adapter is changed. Select before prepare/capture; restore after graph use ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import inspect
from types import MethodType
from typing import Any

import torch

from fastmd.adapters.patches import PatchRecord, apply_patches, revert_patches


_PINS = {
    "EnergyHead.forward": "228928b3e1a2a4dcbbcd4971c816cf061e99fdca433d6dfc12ef99a0f35f6f03",
    "ZBLBasis.forward": "88e5f6a3164959d15912686634fecbde0df10e398107ec1ecaa324844a427a19",
    "ORBAdapter.prepare": "c842201abd5f987be402e27ced86d13527ee3d4c3720e75520bcf24775f7baf0",
    "ORBGraphAdapter._graph": "e224300a3a82ae447bbe4843170b982b153a3ed22acc13ea04a983e08ac015e6",
    "ORBConservativeGraphAdapter.system_energy": "84fa52e40482d60a7d30c93677ef26733e6625988eae57178c84f07b857c723b",
    "nearest_k_mask": "d0a6e1c4fb1058bb032f05a53801bf1e1fcf3e25899243b2d2b08fb51debca5e",
}


def _check_precision():
    if (torch.get_float32_matmul_precision() != "highest"
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")):
        raise ValueError("ORB whole-energy requires highest FP32 precision, TF32 off and autocast off")


def _single_system_aggregate(values, n_node, reduction):
    # These are shape guards, with no tensor-value reads or device sync. The
    # pinned ORBGraphAdapter._graph constructs n_node from its one real system.
    if n_node.ndim != 1 or n_node.shape[0] != 1 or values.shape[0] < 1:
        raise ValueError("Safe energy pooling requires one nonempty atomic system")
    if reduction == "mean":
        return values.mean(dim=0, keepdim=True)
    if reduction == "sum":
        return values.sum(dim=0, keepdim=True)
    raise ValueError("Safe energy pooling supports only mean/sum")


def energy_head_forward(self, node_features, batch):
    pooled = _single_system_aggregate(node_features, batch.n_node, self.node_aggregation)
    return self.mlp(pooled).squeeze(-1)


def zbl_energy_forward(self, batch):
    from orb_models.forcefield import segment_ops

    senders = batch.senders
    receivers = batch.receivers
    one_hot = batch.node_features["atomic_numbers_embedding"]
    atomic_numbers = torch.argmax(one_hot, dim=1)
    Z_u = atomic_numbers[senders] + 1
    Z_v = atomic_numbers[receivers] + 1
    a = self.a_prefactor * 0.529 / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
    vectors = batch.edge_features["vectors"]
    x = vectors.norm(dim=1)
    r_over_a = x / a
    exp_term = torch.exp(-self.d * r_over_a.unsqueeze(0))
    phi = torch.sum(self.c * exp_term, dim=0)
    coulomb_term = 14.3996 * Z_u * Z_v / x
    v_edges_raw = coulomb_term * phi
    r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
    envelope, _ = self._polynomial_cutoff_with_derivative(x, r_max, self.p)
    v_edges = 0.5 * v_edges_raw * envelope
    V_ZBL = segment_ops.segment_sum(v_edges, senders, one_hot.shape[0])
    energy = _single_system_aggregate(V_ZBL, batch.n_node, self.node_aggregation)
    return {"energy": energy}


@dataclass
class Installation:
    options: dict[str, Any]
    source_pins: dict[str, str] = field(default_factory=lambda: dict(_PINS))
    outcomes: list = field(default_factory=list)
    backend: str = "inductor"
    restored: bool = False
    _undo: list = field(default_factory=list, repr=False)
    _compiled: Any = field(default=None, repr=False)

    def set(self, owner, name, value):
        self._undo.append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, value)

    def patch(self, instance, target, key, implementation):
        record = PatchRecord(name="orb.energy." + key, target=target,
            distribution="orb-models", versions=("0.5.4",), source_sha256=_PINS[key],
            reason="explicit conservative whole-energy compilation with safe single-system pooling",
            replacement=MethodType(implementation, instance), per_instance=True)
        added = apply_patches([record], instance=instance)
        self.outcomes.extend(added)
        if not all(outcome.applied for outcome in added):
            raise RuntimeError("ORB energy installation refused: " +
                               "; ".join(outcome.reason for outcome in added))

    def select_backend(self, option):
        if option not in ("inductor", "eager"):
            raise ValueError("ORB energy kernel options are inductor/eager")
        if self.restored:
            raise RuntimeError("ORB energy installation has been restored")
        self.backend = option

    def restore(self):
        if self.restored:
            return
        # Undo adapter.system_energy first, then the private head/ZBL methods,
        # then restore the original backbone callable. Graph use must be over.
        self._compiled = None
        revert_patches(self.outcomes)
        for owner, name, own, old in reversed(self._undo):
            if own:
                setattr(owner, name, old)
            else:
                delattr(owner, name)
        self.restored = True

    __call__ = restore
    close = restore


def install(adapter, *, autotune=False):
    """Install an explicitly selected, independently reported energy kernel.

    Compilation is lazy and fullgraph, so unsupported operations fail during
    first evaluation rather than silently running outside the requested graph.
    The adapter owns backend selection, disable timing, and lifecycle reporting.
    """
    from fastmd.models.orb import ORBAdapter, _sync_free_aggregation
    from fastmd.models.orb_graph import ORBConservativeGraphAdapter, ORBGraphAdapter, nearest_k_mask
    from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor
    from orb_models.forcefield.forcefield_heads import EnergyHead
    from orb_models.forcefield.pair_repulsion import ZBLBasis

    if type(autotune) is not bool:
        raise TypeError("autotune must be bool")
    if importlib.metadata.version("orb-models") != "0.5.4":
        raise RuntimeError("ORB whole-energy requires orb-models 0.5.4")
    if type(adapter) is not ORBConservativeGraphAdapter:
        raise ValueError("ORB whole-energy supports only the conservative graph adapter")
    base, module = adapter._base, adapter.module
    if base._in_use:
        raise RuntimeError("Enable ORB whole-energy before prepare/capture")
    if not isinstance(module, ConservativeForcefieldRegressor):
        raise ValueError("Expected the conservative ORB regressor")
    if base.operator_backend != "torch":
        raise ValueError("First enable the ORB Torch operator rewrites")
    if not base._sync_free or any(not outcome.applied for outcome in base.patches):
        raise RuntimeError("Required ORB source pins/capture-safe patches are unavailable")
    if module.training or any(p.requires_grad or p.dtype != torch.float32 for p in module.parameters()):
        raise ValueError("ORB whole-energy requires frozen FP32 eval weights")
    _check_precision()
    for key, function in (("EnergyHead.forward", EnergyHead.forward),
                          ("ZBLBasis.forward", ZBLBasis.forward),
                          ("ORBAdapter.prepare", ORBAdapter.prepare),
                          ("ORBGraphAdapter._graph", ORBGraphAdapter._graph),
                          ("ORBConservativeGraphAdapter.system_energy", ORBConservativeGraphAdapter.system_energy),
                          ("nearest_k_mask", nearest_k_mask)):
        digest = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        if digest != _PINS[key]:
            raise RuntimeError(f"Refusing changed {key}: source SHA256 {digest}")
    if (getattr(base.prepare, "__func__", None) is not ORBAdapter.prepare
            or getattr(adapter._graph, "__func__", None) is not ORBGraphAdapter._graph
            or getattr(adapter.system_energy, "__func__", None) is not ORBConservativeGraphAdapter.system_energy):
        raise RuntimeError("Unexpected instance graph/energy override")
    head = module.heads[module.energy_name]
    if (type(head) is not EnergyHead or getattr(head.forward, "__func__", None) is not EnergyHead.forward
            or head.node_aggregation not in ("mean", "sum")):
        raise ValueError("Expected the original mean/sum EnergyHead")
    zbl = module.pair_repulsion_fn if module.pair_repulsion else None
    if zbl is not None and (type(zbl) is not ZBLBasis
            or getattr(zbl.forward, "__func__", None) is not ZBLBasis.forward
            or zbl.compute_gradients or zbl.node_aggregation not in ("mean", "sum")):
        raise ValueError("Expected the original energy-only mean/sum ZBL")

    options = {"force_same_precision": True, "triton.cudagraphs": False}
    if autotune:
        options.update(max_autotune=True, max_autotune_gemm_backends="ATEN,TRITON",
                       max_autotune_gemm_search_space="DEFAULT", epilogue_fusion=True)
    handle = Installation(options)
    original_graph = adapter._graph

    def energy(inputs, positions, cell):
        graph = original_graph(inputs, positions, cell)
        features = module.model(graph)["node_features"]
        energy_head = module.heads[module.energy_name]
        result = energy_head.denormalize(energy_head(features, graph), graph)
        if module.pair_repulsion:
            result = result + module.pair_repulsion_fn(graph)["energy"]
        return result

    def system_energy(self, inputs, positions, cell):
        _check_precision()
        if positions.dtype != torch.float32 or cell.dtype != torch.float32:
            raise ValueError("ORB whole-energy requires FP32 positions/cell")
        with _sync_free_aggregation():
            if handle.backend == "eager":
                return energy(inputs, positions, cell)
            return handle._compiled(inputs, positions, cell)

    try:
        handle.set(module.model, "_compiled_call_impl", None)
        handle.patch(head, "orb_models.forcefield.forcefield_heads.EnergyHead.forward",
                     "EnergyHead.forward", energy_head_forward)
        if zbl is not None:
            handle.patch(zbl, "orb_models.forcefield.pair_repulsion.ZBLBasis.forward",
                         "ZBLBasis.forward", zbl_energy_forward)
        handle._compiled = torch.compile(energy, dynamic=True, fullgraph=True, options=options)
        handle.patch(adapter, "fastmd.models.orb_graph.ORBConservativeGraphAdapter.system_energy",
                     "ORBConservativeGraphAdapter.system_energy", system_energy)
    except BaseException:
        handle.restore()
        raise
    return handle
