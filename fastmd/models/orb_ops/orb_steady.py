"""Explicit, source-pinned FP32 ORB conservative inference optimizations.

Install after ``adapter.enable_operator_rewrites('torch')`` and before any
``prepare``/compilation/warm-up. The returned handle is callable and restores
all candidate attributes and methods (it leaves the pre-existing FastMD Torch
rewrite installed). No upstream class, global precision flag, or weight is
changed. The handle must outlive captured CUDA graphs and be restored only
after those graphs are no longer used.

The node-only forward follows orb-models 0.5.4 MoleculeGNS.forward, copyright
Orbital Materials, Inc., Apache-2.0; it omits the decoder and unconsumed output
dictionary entries. Projection ordering follows FastMD's source-pinned Torch
rewrite. Concatenation of frozen weights occurs once at installation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import inspect
from types import MethodType
from typing import Any

import torch
from torch.nn import functional as F

from fastmd.adapters.patches import PatchRecord, apply_patches, revert_patches


_PINS = {
    "MoleculeGNS.forward": "a02ec27dc12566dc38198e09fca6ce486d898895ced0746c7ef4206f2cc6f026",
    "AttentionInteractionNetwork.forward": "5c3b6f57cdca3abafc47869a6d363dae2582c26a863d9e5a34507b39074f90d0",
    "interaction": "a7a0ac45ff3ad391fb76519bc3fcf4251665b4d89c1a0d465e7b6dc9eb2a4601",
}


def _pin(function, name):
    found = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    if found != _PINS[name]:
        raise RuntimeError(f"Refusing changed {name}: source SHA256 {found}")


def node_features_only(self, batch):
    """Conservative energy tap consumes only the final node features."""
    from orb_models.forcefield.nn_util import get_cutoff

    edge_features = self.featurize_edges(batch)
    node_features = self.featurize_nodes(batch)
    nodes, edges = self._encoder(node_features, edge_features)
    cutoff = get_cutoff(batch.edge_features["vectors"].norm(dim=-1))
    for gnn in self.gnn_stacks:
        nodes, edges = gnn(nodes, edges, batch.senders, batch.receivers, cutoff,
                           cond_nodes=None, cond_edges=None)
    return {"node_features": nodes}


def merged_interaction(self, nodes, edges, senders, receivers, cutoff,
                       cond_nodes=None, cond_edges=None):
    """Independent, frozen-weight node/attention concatenations in FP32."""
    from fastmd.models.orb_ops.orb_inference import gated_aggregate_torch

    if cond_nodes is not None:
        nodes = nodes + self._cond_node_proj(cond_nodes)
    if cond_edges is not None:
        edges = edges + self._cond_edge_proj(cond_edges)

    edge_weight, send_weight, receive_weight = self._fastmd_projection_weights
    updated = F.linear(edges, edge_weight, self._edge_mlp.mlp[0].bias)
    if self._orb_steady_merge_node:
        both = F.linear(nodes, self._orb_steady_node_weight)
        sent, received = both.split(self._orb_steady_node_width, dim=-1)
    else:
        sent = F.linear(nodes, send_weight)
        received = F.linear(nodes, receive_weight)
    # Keep the original addition order; changing it would add another FP32
    # rounding difference unrelated to the projection-merge hypothesis.
    updated = updated + sent[senders] + received[receivers]
    for layer in list(self._edge_mlp.mlp.children())[1:]:
        updated = layer(updated)
    updated = self._edge_mlp.layer_norm(updated)

    if self._orb_steady_merge_attention:
        both_logits = F.linear(edges, self._orb_steady_attention_weight,
                               self._orb_steady_attention_bias)
        send_logits, receive_logits = both_logits.split(1, dim=-1)
    else:
        send_logits = self._send_attn(edges)
        receive_logits = self._receive_attn(edges)
    sent, received = gated_aggregate_torch(
        updated, send_logits, receive_logits, cutoff, senders, receivers,
        nodes.shape[0])
    updated_nodes = self._node_mlp(torch.cat((nodes, received, sent), dim=1))
    return nodes + updated_nodes, edges + updated


@dataclass
class Installation:
    options: dict[str, bool]
    source_pins: dict[str, str] = field(default_factory=lambda: dict(_PINS))
    outcomes: list = field(default_factory=list)
    _undo: list[tuple[Any, str, bool, Any]] = field(default_factory=list, repr=False)
    restored: bool = False

    def set(self, owner, name, value):
        own = name in vars(owner)
        self._undo.append((owner, name, own, vars(owner).get(name)))
        setattr(owner, name, value)

    def replace_forward(self, owner, target, implementation, name, source):
        """Record a per-instance patch above the existing Torch rewrite."""
        record = PatchRecord(
            name=name, target="orb_models.forcefield.gns." + target,
            distribution="orb-models", versions=("0.5.4",),
            source_sha256=_PINS[source],
            reason="explicit FP32 conservative-only projection/output optimization",
            replacement=MethodType(implementation, owner), per_instance=True)
        added = apply_patches([record], instance=owner)
        self.outcomes.extend(added)
        if not all(outcome.applied for outcome in added):
            raise RuntimeError("ORB steady optimization refused: " +
                               "; ".join(outcome.reason for outcome in added))

    def restore(self):
        if self.restored:
            return
        # Drop the compiled callable before reverting any methods. The old
        # callable is restored last by the attribute undo list below.
        if self._undo:
            self._undo[0][0]._compiled_call_impl = None
        revert_patches(self.outcomes)
        for owner, name, own, original in reversed(self._undo):
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.restored = True

    __call__ = restore
    close = restore

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()


def install(adapter, *, prune=False, merge_node=False, merge_attention=False):
    """Install independently selectable candidates on one conservative adapter.

    Weight, architecture, installed-forward and version guards deliberately
    reject unsupported models instead of modifying a potentially shared class.
    This is inference-only: concatenated weights are constant detached copies,
    while input gradients (positions/cell through the network) remain enabled.
    """
    from fastmd.models.orb_ops.orb_inference import interaction
    from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor
    from orb_models.forcefield.gns import AttentionInteractionNetwork, MoleculeGNS

    if importlib.metadata.version("orb-models") != "0.5.4":
        raise RuntimeError("Candidates are pinned to orb-models 0.5.4")
    _pin(MoleculeGNS.forward, "MoleculeGNS.forward")
    _pin(AttentionInteractionNetwork.forward, "AttentionInteractionNetwork.forward")
    _pin(interaction, "interaction")
    base = getattr(adapter, "_base", adapter)
    module = adapter.module
    backbone = module.model
    if not isinstance(module, ConservativeForcefieldRegressor):
        raise ValueError("Candidates are restricted to the conservative energy tap")
    if type(backbone) is not MoleculeGNS:
        raise ValueError("Unsupported ORB backbone type")
    if base._in_use:
        raise RuntimeError("Install candidates before prepare or MD construction")
    if base.operator_backend != "torch":
        raise RuntimeError("First call adapter.enable_operator_rewrites('torch')")
    if module.training or any(p.requires_grad for p in module.parameters()):
        raise ValueError("Candidates require eval mode and frozen weights")
    if any(p.dtype != torch.float32 for p in backbone.parameters()):
        raise ValueError("Candidates require existing FP32 weights")
    if "_orb_steady_installed" in vars(backbone):
        raise RuntimeError("Candidates already installed on this instance")
    if getattr(backbone.forward, "__func__", None) is not MoleculeGNS.forward:
        raise RuntimeError("Unexpected pre-existing backbone forward override")

    # Validate every layer before mutating even the first one.
    for layer in backbone.gnn_stacks:
        if type(layer) is not AttentionInteractionNetwork:
            raise ValueError("Unsupported ORB interaction type")
        if getattr(layer.forward, "__func__", None) is not interaction:
            raise RuntimeError("Expected the source-pinned FastMD Torch interaction")
        if layer._attention_gate != "sigmoid" or not layer._distance_cutoff:
            raise ValueError("Expected sigmoid attention with distance cutoff")
        if layer._fastmd_operator_backend != "torch":
            raise ValueError("Mixed operator backends are unsupported")
        linear = layer._edge_mlp.mlp[0]
        weights = layer._fastmd_projection_weights
        width = layer._send_attn.in_features
        if (not isinstance(linear, torch.nn.Linear) or linear.in_features != 3 * width
                or len(weights) != 3
                or any(w.shape != (linear.out_features, width) for w in weights)
                or any(w.dtype != torch.float32 or w.requires_grad for w in weights)
                or layer._receive_attn.in_features != width
                or layer._send_attn.out_features != 1
                or layer._receive_attn.out_features != 1
                or layer._send_attn.bias is None
                or layer._receive_attn.bias is None):
            raise ValueError("Unexpected ORB projection shapes")

    handle = Installation(dict(prune=bool(prune), merge_node=bool(merge_node),
                               merge_attention=bool(merge_attention)))
    try:
        # Restore the exact old callable on undo; compiling is left to the
        # adapter/runtime after all variants have been selected.
        handle.set(backbone, "_compiled_call_impl", None)
        handle.set(backbone, "_orb_steady_installed", dict(handle.options))
        if prune:
            handle.replace_forward(backbone, "MoleculeGNS.forward", node_features_only,
                                   "orb.steady.node_features_only", "MoleculeGNS.forward")
        if merge_node or merge_attention:
            for index, layer in enumerate(backbone.gnn_stacks):
                handle.set(layer, "_orb_steady_merge_node", bool(merge_node))
                handle.set(layer, "_orb_steady_merge_attention", bool(merge_attention))
                if merge_node:
                    _, send_weight, receive_weight = layer._fastmd_projection_weights
                    handle.set(layer, "_orb_steady_node_weight",
                               torch.cat((send_weight, receive_weight), dim=0).detach())
                    handle.set(layer, "_orb_steady_node_width", send_weight.shape[0])
                if merge_attention:
                    handle.set(layer, "_orb_steady_attention_weight", torch.cat(
                        (layer._send_attn.weight, layer._receive_attn.weight), dim=0).detach())
                    handle.set(layer, "_orb_steady_attention_bias", torch.cat(
                        (layer._send_attn.bias, layer._receive_attn.bias), dim=0).detach())
                handle.replace_forward(layer, "AttentionInteractionNetwork.forward",
                    merged_interaction, f"orb.steady.interaction.{index}",
                    "AttentionInteractionNetwork.forward")
        if base.kernel == "inductor":
            backbone.compile(mode="default", dynamic=True)
    except BaseException:
        handle.restore()
        raise
    return handle
