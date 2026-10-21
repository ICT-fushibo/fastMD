"""Inference rewrites for ORB 0.5.4, with an ordinary Torch reference path.

Algebra and interaction ordering follow orb-models (Orbital Materials, Inc.,
Apache-2.0). Weights are never changed: only the first edge projection is
factored into edge and node projections. Nonlinear layers remain per edge.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def projected_edge_input(nodes, edges, senders, receivers, weights, bias):
    edge_weight, send_weight, receive_weight = weights
    edge = F.linear(edges, edge_weight, bias)
    sent = F.linear(nodes, send_weight)
    received = F.linear(nodes, receive_weight)
    return edge + sent[senders] + received[receivers]


def gated_aggregate_torch(values, send_logits, receive_logits, cutoff, senders, receivers, n_nodes):
    send = torch.sigmoid(send_logits) * cutoff
    receive = torch.sigmoid(receive_logits) * cutoff
    shape = (n_nodes, values.shape[1])
    return (values.new_zeros(shape).index_add(0, senders, values * send),
            values.new_zeros(shape).index_add(0, receivers, values * receive))


def interaction(self, nodes, edges, senders, receivers, cutoff, cond_nodes=None, cond_edges=None):
    if cond_nodes is not None:
        nodes = nodes + self._cond_node_proj(cond_nodes)
    if cond_edges is not None:
        edges = edges + self._cond_edge_proj(cond_edges)
    updated = projected_edge_input(nodes, edges, senders, receivers,
                                   self._fastmd_projection_weights, self._edge_mlp.mlp[0].bias)
    for layer in list(self._edge_mlp.mlp.children())[1:]:
        updated = layer(updated)
    updated = self._edge_mlp.layer_norm(updated)
    # ORB's attention reads the OLD edges, before the edge residual update.
    send_logits, receive_logits = self._send_attn(edges), self._receive_attn(edges)
    if self._fastmd_operator_backend == 'triton':
        from fastmd.models.orb_ops.triton_orb_inference import gated_aggregate
        sent, received = gated_aggregate(updated, send_logits, receive_logits, cutoff,
                                         senders, receivers, len(nodes))
    else:
        sent, received = gated_aggregate_torch(updated, send_logits, receive_logits, cutoff,
                                               senders, receivers, len(nodes))
    updated_nodes = self._node_mlp(torch.cat((nodes, received, sent), dim=1))
    return nodes + updated_nodes, edges + updated


def compact_topk_torch(senders, distances, n_nodes, k, cutoff, n_edges=None):
    """CPU/reference stable top-K, returning original edge rows (invalid = -1).

    Ties use original row order, matching the existing stable double sort.
    This reference deliberately supports unsorted senders and arbitrary padding.
    """
    rows = torch.arange(len(senders), device=senders.device)
    valid = distances < cutoff * cutoff
    if n_edges is not None:
        valid = valid & (rows < n_edges.reshape(()))
    matrix = torch.where((torch.arange(n_nodes, device=senders.device)[:, None] == senders)
                         & valid, distances, float('inf'))
    # Append K invalid entries: even an empty input has a fixed N*K output.
    matrix = torch.cat((matrix, distances.new_full((n_nodes, k), float('inf'))), dim=1)
    order = torch.argsort(matrix, dim=1, stable=True)[:, :k]
    live = torch.gather(matrix, 1, order).isfinite()
    return torch.where(live, order, -1)
