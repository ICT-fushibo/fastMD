"""Local hooks for MatRIS interaction layers.

The layer schedule calls these methods directly; routes are identity and
there is no communication or distributed gradient backend.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import torch

from fastmd.engine.comm import LocalCommunicator

if os.environ.get("MATRIS_RUNTIME_ENGINE", "unified") != "unified":
    raise ValueError("MATRIS_RUNTIME_ENGINE: the legacy runtimes were removed in step 4.2; "
                     f"only 'unified' remains, got {os.environ['MATRIS_RUNTIME_ENGINE']!r}")

# The model's forward entry points: the layer methods these hooks call and the model methods the forwards call
# (MatRIS's admission and the run boundary's model stamp in fastmd.distributed.execution read them).
# Shared forward entrypoints are executable model state, just like forward.
# Reject instance replacements at public admission and stamp internal callers'
# bindings so prepare/run cannot execute two different model programs.
MODEL_FORWARD_METHODS = frozenset({
    "forward", "forward_features", "embed_atoms", "normalize_readout",
    "atomic_energy", "atomic_energy_rows", "atomic_energies", "_graph_sum", "_per_atom_reference",
    "forward_lengths", "forward_indexed",
    "forward_vectors", "forward_basis", "attention_rows", "attention_nodes",
    "attention_edges", "refinement_rows", "refinement_payload",
    "refinement_nodes", "refinement_edges", "_fused_edge_ffn", "wrapper_attn_layer",
    "wrapper_refine_layer", "_attn_layer_call", "_refine_layer_call", "_forward", "_normalize_real_counts",
    "_can_fuse_indexed_cat_first_linear", "_first_linear_modules",
    "_split_first_linear_params", "aligned_first_linear_block",
    "forward_aligned_gather_cat3", "forward_aligned_gather_cat4",
    "_run_after_first_activation", "_finish_gate", "_merged_attention_projection",
    "_combined_swish_linear_weight",
})




class EngineRuntime:
    """Runtime hooks of the shared model forward, at root, block or layer level.

    ``for_block(i, block)`` and ``for_layer(name)`` return views bound to one
    interaction block / layer; the eight hooks are called on layer views,
    ``select_line_nodes`` on block views. A view is built once per block and
    layer of an evaluation (as the pre-3.1b block/layer runtimes were).
    """

    def __init__(self, comm: Any = None, backend: Any = None, routes: Any = None):
        if backend is not None or routes is not None:
            raise ValueError("FastMD supports local execution only")
        # ``comm``: the context's Communicator (Local at P=1, DomainCommunicator at P>1).
        self.comm, self.backend, self.routes = comm, backend, routes
        self.index = self.block = self.name = self.line_layer = None

    def _view(self, index, block, name) -> EngineRuntime:
        # Plain attribute stores: touching ``__dict__`` would materialize it
        # and slow every later attribute read of the view (CPython 3.12).
        view = object.__new__(type(self))
        view.comm, view.backend, view.routes = self.comm, self.backend, self.routes
        view.index, view.block, view.name, view.line_layer = index, block, name, None
        return view

    def for_block(self, index: int, block) -> EngineRuntime:
        return self._view(index, block, None)

    def for_layer(self, name: str) -> EngineRuntime:
        view = self._view(self.index, self.block, name)
        routes = self.routes
        if routes is not None:
            raise TypeError("FastMD supports local execution only")
        return view

    def local(self, fn: Callable, *args, name: str = ""):
        """A rank-local stage: ``fn(*args)`` itself with identity routes."""
        if self.routes is None:
            return fn(*args)
        return self.backend.local(fn, *args, name=name)

    def select_line_nodes(self, new, old, mask):
        """Block level: the line-graph result where ``mask``, else the input edge
        features (the P=1 ``torch.where`` of ``Interaction_Block.forward``). With
        exchange routes ``mask`` must be collectively admitted (INV-7)."""
        if self.routes is None:
            return torch.where(mask, new, old)
        return self.backend.local(lambda values, original: torch.where(mask, values, original),
                                  new, old, name=f"block_{self.index}.{self.line_layer}.select")

    # --- the eight layer hooks; the identity branch keeps the pre-3.1b local runtime's bodies ---

    def attention_rows(self, layer, nodes, edges, graph, directed2undirected):
        if self.routes is None:
            return layer.attention_rows(nodes, edges, graph, directed2undirected)
        raise TypeError("FastMD supports local execution only")

    def attention_route(self, rows, graph):
        if self.routes is None:
            return rows
        raise TypeError("FastMD supports local execution only")

    def attention_nodes(self, layer, nodes, rows, graph):
        if self.routes is None:
            message, source_logits, target_logits = rows
            return layer.attention_nodes(
                nodes, message, source_logits, graph["source_index"],
                message, target_logits, graph["target_index"], graph,
            )
        raise TypeError("FastMD supports local execution only")

    def attention_edges(self, layer, edges, rows, graph, directed2undirected):
        if self.routes is None:
            return layer.attention_edges(edges, rows[0], graph, directed2undirected)
        raise TypeError("FastMD supports local execution only")

    def refinement_rows(self, layer, nodes, edges, radial, graph,
                        directed2undirected, atom_feat):
        if self.routes is None:
            raw, weight, residual = layer.refinement_rows(
                nodes, edges, radial, graph, directed2undirected, atom_feat,
            )
            if layer.graph_type == "atom graph" and layer.use_smoothed_for_delta_edge:
                weighted, edge_input = layer.refinement_payload(raw, weight)
                return weighted, None, edge_input, residual
            # Keep weight separate so the local fused weighted-segment reduction
            # never needs to materialize the full weighted message tensor.
            return raw, weight, raw, residual
        raise TypeError("FastMD supports local execution only")

    def refinement_route(self, rows, graph):
        if self.routes is None:
            return rows
        raise TypeError("FastMD supports local execution only")

    def refinement_nodes(self, layer, nodes, rows, graph):
        if self.routes is None:
            return layer.refinement_nodes(
                nodes, rows[0], rows[1], graph["target_index"], graph,
            )
        raise TypeError("FastMD supports local execution only")

    def refinement_edges(self, layer, edges, rows, graph,
                         directed2undirected, skip_edge_update):
        if self.routes is None:
            # A merged projection may return a residual alias whose shared VJP
            # eliminates a separate accumulation.  Keep that alias alive here.
            residual = edges if skip_edge_update else rows[3]
            return layer.refinement_edges(
                residual, rows[2], graph, directed2undirected, skip_edge_update,
            )
        raise TypeError("FastMD supports local execution only")




# P=1: the layer-level singleton (identity routes); blocks keep runtime=None.
LOCAL_RUNTIME = EngineRuntime(LocalCommunicator())
