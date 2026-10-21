from __future__ import annotations
from typing import Literal, Tuple

from fastmd.config import env_value
from fastmd.engine import policy as _policy
from fastmd.engine import recorder as _recorder

import torch
from torch import Tensor, nn
from typing import Dict
from fastmd.models.matris.functions import MLP
from fastmd.models.matris.functions import FusedSiLU
from fastmd.models.matris.functions import GatedMLP
from fastmd.models.matris.functions import _apply_layer
from fastmd.models.matris.functions import _layer_precision
from fastmd.models.matris.functions import _routed
from fastmd.models.matris.functions import get_normalization
from torch.utils.checkpoint import checkpoint
from fastmd.engine import saved_recompute as _saved_recompute
from fastmd.engine.cache_provenance import CAPTURE as _CAPTURE, retain_under_capture
from fastmd.engine.kernels import triton_wide_gemm as _wide
from fastmd.models.matris.forward_runtime import LOCAL_FORWARD_RUNTIME

THRESHOLD_VALUE = 60000 # Safe value for MatRIS-10M (A100-80GB)
_FUSED_WEIGHTED_SEGMENT_EAGER = env_value("MATRIS_FUSED_WEIGHTED_SEGMENT_EAGER", "0") == "1"
# C4: the line-graph refinement's edge FFN in one kernel per direction (``refinement_edges``; the residual stays).
_FUSED_EDGE_FFN = env_value("MATRIS_FUSED_EDGE_FFN") == "1"
_FUSED_GATHER_CAT_EAGER = env_value("MATRIS_FUSED_GATHER_CAT_EAGER", "0") == "1"
_FUSED_DIRECTED_PAIR_AGGREGATE = env_value(
    "MATRIS_FUSED_DIRECTED_PAIR_AGGREGATE", "1"
) != "0"
_FUSED_DIRECTED_PAIR_EXPAND = env_value("MATRIS_FUSED_DIRECTED_PAIR_EXPAND", "1") != "0"
_FUSED_SEGMENT_ATTENTION_ENABLED = env_value("MATRIS_FUSED_SEGMENT_ATTENTION", "1") != "0"
_FUSED_SEGMENT_ATTENTION_EAGER = env_value("MATRIS_FUSED_SEGMENT_ATTENTION_EAGER", "0") == "1"
_FUSED_SEGMENT_ATTENTION_MIN_ROWS = int(env_value("MATRIS_FUSED_SEGMENT_ATTENTION_MIN_ROWS", "0"))
_FUSED_PAIRED_SEGMENT_ATTENTION = env_value("MATRIS_FUSED_PAIRED_SEGMENT_ATTENTION", "1") != "0"
_FUSED_SEGMENT_ATTENTION_REUSE_CSR = env_value(
    "MATRIS_FUSED_SEGMENT_ATTENTION_REUSE_CSR", "1"
) != "0"
_FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR = env_value(
    "MATRIS_FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR", "1"
) != "0"
_FUSED_LINE_ENVELOPE_ENABLED = env_value("MATRIS_FUSED_LINE_ENVELOPE", "1") != "0"
_FUSED_LINE_ENVELOPE_EAGER = env_value("MATRIS_FUSED_LINE_ENVELOPE_EAGER", "0") == "1"
_FUSED_RESIDUAL_ADD_ENABLED = env_value("MATRIS_FUSED_RESIDUAL_ADD", "1") != "0"
_FUSED_RESIDUAL_ADD_MIN_ROWS = int(env_value("MATRIS_FUSED_RESIDUAL_ADD_MIN_ROWS", "0"))
_MERGED_ATTN_PROJECTIONS = env_value(
    "MATRIS_MERGED_ATTENTION_PROJECTIONS", "1"
) != "0"
_MERGED_REFINEMENT_PROJECTION = env_value(
    "MATRIS_MERGED_REFINEMENT_PROJECTION", "1"
) != "0"
_SKIP_DEAD_THREEBODY_TAIL = env_value(
    "MATRIS_SKIP_DEAD_THREEBODY_TAIL", "1"
) != "0"


# Deprecated aliases of fastmd.engine.policy.set_fusion_scope, kept for one version.
# Since step 2.2 both families share one fusion scope: setting either sets both, and
# nested set/restore pairs must be restored in reverse (LIFO) order.
def set_fused_graph_optimizations(enabled: bool) -> tuple[bool, bool, bool, bool]:
    return (_policy.set_fusion_scope(enabled),) * 4


def restore_fused_graph_optimizations(state: tuple[bool, bool, bool, bool]) -> None:
    _policy.set_fusion_scope(state[0])


def _gather_rows(value: Tensor, index: Tensor) -> Tensor:
    return torch.index_select(value, 0, index)


def _alpha_projection(edge_feat: Tensor, weight: Tensor, split: int, precision: str | None,
                      wide: str | None = None) -> tuple[Tensor, Tensor]:
    """Recompute recipe of the merged projection's (source, target) alpha views, with the forward's GEMM
    (``precision``: the row GEMM's, whose column ranges are bitwise those of the wide product; None: ``wide``,
    the 3xTF32 wide GEMM's, or None cuBLAS)."""
    from fastmd.engine.kernels.merged_frozen_projections import project
    alphas = project(edge_feat, weight, precision, wide)
    return alphas[:, :split], alphas[:, split:]


def _module_forward(module, value):
    """A model module call at the layer sites' call depth (compiled-program shape); a frozen nn.Linear without
    hooks through the wide GEMM dispatch (``functions._routed``), at the same depth."""
    if type(module) is nn.Linear and _routed(module):
        return _wide._linear(value, module.weight, module.bias)
    return module(value)


def _projection_choice(module, aligned, gathered, indices, parts, *, local_enabled,
                       dispatcher):
    """Whether the site tries the indexed projection, and the executor policy.

    The dispatcher's executor policy applies (None at P=1: the module's own
    switches; strict at P>1)."""
    return local_enabled, dispatcher.projection_config


def _policy_residual(delta, residual, weight, graph):
    return graph["_ops"].residual(delta, residual, weight)


def _policy_pair_average(data, directed2undirected, graph):
    return graph["_ops"].pair_average(data, directed2undirected, graph)


def _record_merge_decline(rec, reason, edge_feat_0) -> None:
    rec.fallback("merged_attention_projection", "merged_frozen_projections", reason,
                 {"rows": edge_feat_0.shape[0], "width": edge_feat_0.shape[-1]})


def _record_refinement_merge_decline(rec, reason, edge_feat_0) -> None:
    rec.fallback("merged_refinement_projection", "projection_with_alias", reason,
                 {"rows": edge_feat_0.shape[0], "width": edge_feat_0.shape[-1]})


class Graph_Attention_Layer(nn.Module):
    
    def __init__(
        self,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        hidden_dim: int = 128,
        use_bias: bool = False,
        dropout: float = 0.0,
        mlp_type: str = "GateMLP", # MLP, GateMLP
        activation_type: str = "silu",
        norm_type: str = "layer",
        use_fp16: bool = False, 
    ):
        super().__init__()
        # C3: whether the merged projection's consumers may share one gradient
        # buffer; InteractionBlock turns it off in the last block (see there).
        self.merged_grad_buffer = True

        self.source_weight_linear = nn.Linear(
            in_features = edge_feat_dim, out_features = edge_feat_dim, bias = False
        )
        self.target_weight_linear = nn.Linear(
            in_features = edge_feat_dim, out_features = edge_feat_dim, bias = False
        )
        if mlp_type.lower() == "mlp":
            self.node_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=edge_feat_dim * 2 + node_feat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=node_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                ),
                get_normalization(name=norm_type, dim=node_feat_dim) 
            )
            self.edge_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=node_feat_dim * 2 + edge_feat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=edge_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                    use_fp16=use_fp16,
                ),
                get_normalization(name=norm_type, dim=edge_feat_dim) 
            )
        elif mlp_type.lower() == "gatemlp":
            self.node_nonlinear_update = GatedMLP(
                input_dim=edge_feat_dim * 2 + node_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=node_feat_dim,
                norm_type=norm_type,
                dropout=dropout,
                activation=activation_type,
            )
            self.edge_nonlinear_update = GatedMLP(
                input_dim=node_feat_dim * 2 + edge_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=edge_feat_dim,
                norm_type=norm_type,
                dropout=dropout,
                activation=activation_type,
                use_fp16=use_fp16,
            )
        else:
            raise NotImplementedError

        self.node_res_weight = torch.nn.Parameter(torch.ones(1, node_feat_dim), requires_grad=True)
        self.edge_res_weight = torch.nn.Parameter(torch.ones(1, edge_feat_dim), requires_grad=True)
        self._merged_attn_proj_key = None
        self._merged_attn_proj_weight = None
        self._merged_attn_proj_splits = None

    def _merged_attention_projection(self, edge_feat_0: Tensor, gathered_width=None, *,
                                     projection_config=None):
        """Concatenated frozen weight for {cat3-p0, source-alpha, target-alpha}.

        ``gathered_width``: width of the gathered node parts when it differs
        from the aligned one (D8, e.g. atom attention [32|128|128]).

        Returns (w_cat [2H+dim+dim, dim], splits) or None when the merge is
        not applicable (training, bias, non-decomposed backend, capture with
        a cold cache). Collapses three autograd consumers of edge_feat_0 into
        one, removing two gradient-accumulation adds per layer.
        """
        if (not _MERGED_ATTN_PROJECTIONS and projection_config is None
                or not edge_feat_0.is_contiguous()):
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_merge_decline(
                    rec, "disabled" if not _MERGED_ATTN_PROJECTIONS and projection_config is None
                    else "not_contiguous", edge_feat_0)
            return None
        source_linear = self.source_weight_linear
        target_linear = self.target_weight_linear
        if source_linear.bias is not None or target_linear.bias is not None:
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_merge_decline(rec, "projection_has_bias", edge_feat_0)
            return None
        if source_linear.weight.requires_grad or target_linear.weight.requires_grad:
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_merge_decline(rec, "trainable_projection", edge_feat_0)
            return None
        block_getter = getattr(
            self.edge_nonlinear_update, "aligned_first_linear_block", None
        )
        if block_getter is None:
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_merge_decline(rec, "no_aligned_first_linear_block", edge_feat_0)
            return None
        rows, dim = edge_feat_0.shape
        if gathered_width is not None:
            dim = (dim, gathered_width, gathered_width)
        w0 = block_getter(rows, dim, 3, edge_feat_0.device, edge_feat_0.dtype,
                         projection_config=projection_config)
        if w0 is None:
            rec = _recorder._ACTIVE
            if rec is not None:
                _record_merge_decline(rec, "first_linear_block_unavailable", edge_feat_0)
            return None
        key = (
            w0.data_ptr(),
            w0._version,
            # w0 is the split cache's tensor: a rebuilt one can reuse the address at
            # version 0, so the split cache's own key (its source weights) is part of this one.
            self.edge_nonlinear_update._indexed_cat_linear_split_cache_key,
            source_linear.weight.data_ptr(),
            source_linear.weight._version,
            target_linear.weight.data_ptr(),
            target_linear.weight._version,
            edge_feat_0.device,
            edge_feat_0.dtype,
        )
        if self._merged_attn_proj_key != key:
            if (
                edge_feat_0.device.type == "cuda"
                and torch.cuda.is_current_stream_capturing()
            ):
                rec = _recorder._ACTIVE
                if rec is not None:
                    _record_merge_decline(rec, "cold_cache_under_capture", edge_feat_0)
                return None
            with torch.no_grad():
                self._merged_attn_proj_weight = torch.cat(
                    [
                        w0.detach(),
                        source_linear.weight.detach(),
                        target_linear.weight.detach(),
                    ],
                    dim=0,
                ).contiguous()
                self._merged_attn_proj_splits = (
                    int(w0.shape[0]),
                    int(source_linear.weight.shape[0]),
                    int(target_linear.weight.shape[0]),
                )
                self._merged_attn_proj_key = key
                from fastmd.engine.cache_provenance import publish_projection_cache
                publish_projection_cache(self, "attention_merged", (
                    w0, source_linear.weight, target_linear.weight,
                ))
        if not torch.compiler.is_compiling() and _CAPTURE.holder is not None:  # a capture keeps what it reads
            # w0 is the split cache's, retained (with its sources) by its builder.
            retain_under_capture(self._merged_attn_proj_weight,
                                 (source_linear.weight, target_linear.weight))
        return self._merged_attn_proj_weight, self._merged_attn_proj_splits

    def forward(self,
        node_feat: Tensor,
        edge_feat: Tensor,
        graph: Dict, # atom graph or line graph
        directed2undirected: Tensor = None,
        skip_edge_update: bool = False,
        *,
        runtime=None,
    ):
        """Shared schedule; a runtime changes placement, never the equations.
        ``skip_edge_update``: the caller reads no edge output (its input edges are returned)."""
        runtime = LOCAL_FORWARD_RUNTIME if runtime is None else runtime
        rows = runtime.attention_rows(self, node_feat, edge_feat, graph, directed2undirected)
        routed = runtime.attention_route(rows, graph)
        nodes = runtime.attention_nodes(self, node_feat, routed, graph)
        if skip_edge_update:
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("attention_edge_update", "skipped_dead",
                           {"rows": edge_feat.shape[0], "width": edge_feat.shape[-1]})
            return nodes, edge_feat
        edges = runtime.attention_edges(self, edge_feat, rows, graph, directed2undirected)
        return nodes, edges

    def attention_rows(self, node_feat, edge_feat, graph,
                       directed2undirected=None):
        """Message and channel-wise logits on producer-owned directed rows."""
        source_node_index = graph['source_index']
        target_node_index = graph['target_index']
        attn_edge_feat_nonlinear = None
        merged_alphas = None
        # The OpDispatcher (process_graphs / the P>1 routes attach it) selects at the
        # former helpers' call depth (under torch.compile an extra frame level
        # changes the graph breaks). Module calls stay here; the dispatcher
        # supplies their policy.
        dispatcher = graph["_ops"]
        if directed2undirected is not None:
            # Atom Graph Update
            edge_feat_0 = dispatcher.pair_expand(
                edge_feat, directed2undirected, graph) # [edge, dim] -> [2*edge, dim]
        else:
            edge_feat_0 = edge_feat
        local_indexed = (directed2undirected is not None
                         or dispatcher.use_gather_cat(edge_feat, node_feat))
        use_indexed, projection_config = _projection_choice(
            self.edge_nonlinear_update, edge_feat_0, (node_feat,),
            (target_node_index, source_node_index), 3, local_enabled=local_indexed,
            dispatcher=dispatcher,
        )
        if use_indexed and hasattr(self.edge_nonlinear_update, "forward_aligned_gather_cat3"):
            precomputed_p0 = None
            merged = self._merged_attention_projection(
                edge_feat_0, node_feat.shape[1], projection_config=projection_config)
            if merged is not None:
                from fastmd.engine.kernels.merged_frozen_projections import merged_frozen_projections
                w_cat, splits = merged
                precomputed_p0, merged_source_alpha, merged_target_alpha = (
                    # Compiled blocks never use the buffer; reading the flag there would add
                    # a guard that recompiles the last block.
                    merged_frozen_projections(edge_feat_0, w_cat, splits,
                                              torch.compiler.is_compiling() or self.merged_grad_buffer)
                )
                scope = _saved_recompute.active()
                if scope is not None:
                    from fastmd.engine.kernels.merged_frozen_projections import project
                    from fastmd.engine.kernels.merged_frozen_projections import row_gemm_precision
                    from fastmd.engine.kernels.merged_frozen_projections import wide_precision
                    # The forward's GEMM (C4, else the wide GEMM or cuBLAS): its recipes rebuild with the same one.
                    precision = row_gemm_precision(edge_feat_0, w_cat, splits)
                    wide = None if precision is not None else wide_precision(edge_feat_0, w_cat, splits)
                    # Saved-tensor recompute: the attention saves the alpha views, which pin the
                    # whole [rows, 2H+2dim] product; rebuild the alphas alone
                    # from the input (one [rows, 2dim] GEMM, compact layout).
                    scope.register((merged_source_alpha, merged_target_alpha), _alpha_projection,
                                   scope.ref(edge_feat_0), w_cat[splits[0]:], splits[1], precision, wide,
                                   same_layout=False)
                    if scope.mode == "deep":
                        # deep: the cat3 preact is rebuilt too, from this p0.
                        scope.register((precomputed_p0,), project,
                                       scope.ref(edge_feat_0), w_cat[:splits[0]], precision, wide,
                                       same_layout=False)
                merged_alphas = (merged_source_alpha, merged_target_alpha)
            attn_edge_feat_nonlinear = self.edge_nonlinear_update.forward_aligned_gather_cat3(
                edge_feat_0, node_feat, target_node_index, source_node_index,
                # The single-card sorted contract does not describe compact
                # fetched rows on a distributed producer.
                dispatcher.sorted_vjp("atom") if directed2undirected is not None else False,
                precomputed_p0=precomputed_p0, projection_config=projection_config,
            )
        if attn_edge_feat_nonlinear is None and directed2undirected is None and local_indexed:
            attn_edge_feat = dispatcher.gather_cat(
                edge_feat_0, (node_feat,), (target_node_index, source_node_index))
        elif attn_edge_feat_nonlinear is None:
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("gather_cat3", "torch",
                           {"rows": edge_feat_0.shape[0], "width": edge_feat_0.shape[-1]})
            source_node_feat = dispatcher.gather(node_feat, source_node_index)
            target_node_feat = dispatcher.gather(node_feat, target_node_index)
            #======= combine feature =======
            attn_edge_feat = torch.cat([edge_feat_0, target_node_feat, source_node_feat], dim=1)
            scope = _saved_recompute.active()
            if scope is not None:
                scope.register_gather_cat(attn_edge_feat, edge_feat_0, node_feat, target_node_index,
                                          source_node_index)
        if attn_edge_feat_nonlinear is None:
            attn_edge_feat = _module_forward(self.edge_nonlinear_update, attn_edge_feat)
        else:
            attn_edge_feat = attn_edge_feat_nonlinear

        # ======= update atom feature =======
        if merged_alphas is not None:
            source_alpha_0, target_alpha_0 = merged_alphas
        else:
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("attention_projection", "torch_linear",
                           {"rows": edge_feat_0.shape[0], "width": edge_feat_0.shape[-1]})
            precisions = (_layer_precision(self.source_weight_linear, edge_feat_0),
                          _layer_precision(self.target_weight_linear, edge_feat_0))
            source_alpha_0 = _apply_layer(self.source_weight_linear, edge_feat_0, precisions[0])
            target_alpha_0 = _apply_layer(self.target_weight_linear, edge_feat_0, precisions[1])
            scope = _saved_recompute.active()
            if scope is not None:
                # Saved-tensor recompute: the logits are rebuilt from the input, with the forward's GEMM.
                scope.register((source_alpha_0,), _apply_layer, self.source_weight_linear, scope.ref(edge_feat_0),
                               precisions[0])
                scope.register((target_alpha_0,), _apply_layer, self.target_weight_linear, scope.ref(edge_feat_0),
                               precisions[1])
        return attn_edge_feat, source_alpha_0, target_alpha_0

    def attention_nodes(self, node_feat, source_rows, source_alpha_0,
                        source_node_index, target_rows, target_alpha_0,
                        target_node_index, graph):
        """Reduce complete owner segments and apply the common node update."""
        num_segment = node_feat.shape[0]
        dispatcher = graph["_ops"]
        # Compute Attention output
        attn_source_feat, attn_target_feat = dispatcher.segment_attention(
            source_alpha_0, source_rows, source_node_index,
            target_alpha_0, target_rows, target_node_index, num_segment, graph,
        )

        fusion_node_feat = torch.cat([node_feat, attn_target_feat, attn_source_feat], dim=1)
        # Module calls stay here, not in the dispatcher: their activations
        # graph-break under torch.compile and an extra frame would change the
        # compiled program. The dispatcher supplies the gated-LN residual policy.
        attn_node_feat = None
        node_residual_fused = False
        if hasattr(self.node_nonlinear_update, "forward_with_residual"):
            attn_node_feat = self.node_nonlinear_update.forward_with_residual(
                fusion_node_feat,
                node_feat,
                self.node_res_weight,
                dispatcher=dispatcher,
            )
            node_residual_fused = attn_node_feat is not None
        if attn_node_feat is None:
            attn_node_feat = self.node_nonlinear_update(fusion_node_feat)
        
        # Resdual
        if not node_residual_fused:
            attn_node_feat = dispatcher.residual(attn_node_feat, node_feat, self.node_res_weight)
        return attn_node_feat

    def attention_node_update(self, node_feat, attn_target_feat, attn_source_feat, graph):
        """The node update of ``attention_nodes`` (dispatcher branch) on reduced
        attention features, for a runtime that reduces the segments itself
        (the A2 bond-partial runtime merges two ranks' partials first).

        A separate method, not called by ``attention_nodes``: an extra frame
        around the GatedMLP call would change the P=1 compiled programs.
        """
        dispatcher = graph["_ops"]
        fusion_node_feat = torch.cat([node_feat, attn_target_feat, attn_source_feat], dim=1)
        attn_node_feat = None
        if hasattr(self.node_nonlinear_update, "forward_with_residual"):
            attn_node_feat = self.node_nonlinear_update.forward_with_residual(
                fusion_node_feat, node_feat, self.node_res_weight, dispatcher=dispatcher,
            )
        if attn_node_feat is not None:
            return attn_node_feat
        attn_node_feat = self.node_nonlinear_update(fusion_node_feat)
        return dispatcher.residual(attn_node_feat, node_feat, self.node_res_weight)

    def attention_edges(self, edge_feat, messages, graph,
                        directed2undirected=None):
        if directed2undirected is not None:
            messages = _policy_pair_average(messages, directed2undirected, graph)
        return _policy_residual(messages, edge_feat, self.edge_res_weight, graph)


class Refinement(nn.Module):
    # Set on the blocks InteractionBlock wraps with torch.compile (C4: no fused edge FFN there).
    _compiled_block = False

    def __init__(
        self,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        hidden_dim: int = 128,
        num_basis: int = 7,
        dropout: float = 0.0,
        mlp_type: str = "GateMLP",    
        activation_type: str = "silu",
        norm_type: str = "layer",
        use_bias: bool = False,
        graph_type: Literal["atom graph", "line graph"] = "atom graph",
        atom_feat_dim: int = 128,
        use_smoothed_for_delta_edge: bool = False,
        use_fp16: bool = False, 
    ):
        super().__init__()
        self.graph_type = graph_type
        self.use_smoothed_for_delta_edge = use_smoothed_for_delta_edge
        
        if graph_type == "atom graph":
            input_dim = 2 * node_feat_dim + edge_feat_dim
        else:
            input_dim = atom_feat_dim + 2 * node_feat_dim + edge_feat_dim 
        if mlp_type.lower() == "mlp":
            self.edge_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=edge_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                    use_fp16=use_fp16,
                ),
                get_normalization(name=norm_type, dim=edge_feat_dim)
            )
        elif mlp_type.lower() == "gatemlp":
            self.edge_nonlinear_update = GatedMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=edge_feat_dim,
                dropout=dropout,
                norm_type=norm_type,
                activation=activation_type,
                use_fp16=use_fp16,
            )
        else:
            raise NotImplementedError
        
        self.node_FFN = MLP(
            input_dim=edge_feat_dim,
            hidden_dim=node_feat_dim,
            output_dim=node_feat_dim,
            bias=use_bias,
        )
        self.edge_FFN = MLP(
            input_dim=edge_feat_dim,
            hidden_dim=edge_feat_dim,
            output_dim=edge_feat_dim,
            bias=use_bias,
            use_fp16=use_fp16,
        )
        self.learnable_envelope = nn.Linear(
            in_features = num_basis, out_features = edge_feat_dim, bias = False
        )
        
        self.node_res_weight = torch.nn.Parameter(torch.ones(1, node_feat_dim), requires_grad=True)
        self.edge_res_weight = torch.nn.Parameter(torch.ones(1, edge_feat_dim), requires_grad=True)

    def forward(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        smooth_weight: Tensor,
        graph: Dict,
        directed2undirected: Tensor = None,
        atom_feat: Tensor = None, # Line graph
        skip_edge_update: bool = False,
        *,
        runtime=None,
    ) -> Tensor:
        runtime = LOCAL_FORWARD_RUNTIME if runtime is None else runtime
        rows = runtime.refinement_rows(
            self, node_feat, edge_feat, smooth_weight, graph, directed2undirected, atom_feat,
        )
        routed = runtime.refinement_route(rows, graph)
        nodes = runtime.refinement_nodes(self, node_feat, routed, graph)
        edges = runtime.refinement_edges(
            self, edge_feat, rows, graph, directed2undirected, skip_edge_update,
        )
        return nodes, edges

    def refinement_rows(self, node_feat, edge_feat, smooth_weight, graph,
                        directed2undirected=None, atom_feat=None):
        """Nonlinear producer rows, envelope, and the projection residual alias."""
        # Gather
        # when graph=="line graph", make sure atom_deat is not None.
        is_atom_graph = (self.graph_type == "atom graph")
        edge_feat_residual = edge_feat
        # The OpDispatcher: see attention_rows.
        dispatcher = graph["_ops"]

        if is_atom_graph:
            edge_feat_0 = dispatcher.pair_expand(edge_feat, directed2undirected, graph)
        else:
            edge_feat_0 = edge_feat

        source_node_index = graph['source_index']
        target_node_index = graph['target_index']
        refine_fusion_feat_nonlinear = None
        # Envelope 
        if is_atom_graph:
            smooth_weight = dispatcher.pair_expand(smooth_weight, directed2undirected, graph)
            expanded = smooth_weight
            smooth_weight = _module_forward(self.learnable_envelope, smooth_weight)
            scope = _saved_recompute.active()
            if scope is not None and scope.mode == "deep":
                scope.register((smooth_weight,), self.learnable_envelope, expanded)
            use_indexed, projection_config = _projection_choice(
                self.edge_nonlinear_update, edge_feat_0, (node_feat,),
                (target_node_index, source_node_index), 3, local_enabled=True,
                dispatcher=dispatcher,
            )
            if use_indexed and hasattr(self.edge_nonlinear_update, "forward_aligned_gather_cat3"):
                refine_fusion_feat_nonlinear = self.edge_nonlinear_update.forward_aligned_gather_cat3(
                    edge_feat_0,
                    node_feat,
                    target_node_index,
                    source_node_index,
                    dispatcher.sorted_vjp("atom"),
                    projection_config=projection_config,
                )
            if refine_fusion_feat_nonlinear is None:
                rec = _recorder._ACTIVE
                if rec is not None:
                    rec.record("gather_cat3", "torch",
                               {"rows": edge_feat_0.shape[0], "width": edge_feat_0.shape[-1]})
                source_node_feat = dispatcher.gather(node_feat, source_node_index)
                target_node_feat = dispatcher.gather(node_feat, target_node_index)
                # Fusion feature
                refine_fusion_feat = torch.cat([edge_feat_0, target_node_feat, source_node_feat], dim=1)
                scope = _saved_recompute.active()
                if scope is not None:
                    scope.register_gather_cat(refine_fusion_feat, edge_feat_0, node_feat,
                                              target_node_index, source_node_index)
        else:
            base_envelope = _module_forward(self.learnable_envelope, smooth_weight)
            envelope_out = None
            # Kernels called here, not in the dispatcher: the compiled lowering
            # can graph-break, and a dispatcher frame around it would change the
            # compiled program.
            if dispatcher.line_incidence_direct and dispatcher.use_line_envelope(
                base_envelope, source_node_index, target_node_index
            ):
                from fastmd.models.matris.op.triton_line_envelope import line_envelope_product

                envelope_out = line_envelope_product(
                    base_envelope, source_node_index, target_node_index
                )
                scope = _saved_recompute.active()
                if scope is not None:
                    scope.register((envelope_out,), line_envelope_product,
                                   base_envelope, source_node_index, target_node_index)
            elif dispatcher.compiled_envelope:
                from fastmd.models.matris.op.compiled_lowerings import compiled_line_envelope

                envelope_out = compiled_line_envelope(
                    base_envelope, source_node_index, target_node_index
                )
            if envelope_out is not None:
                smooth_weight = envelope_out
            else:
                rec = _recorder._ACTIVE
                if rec is not None:
                    rec.record("line_envelope", "torch_index_select_product",
                               {"rows": source_node_index.numel(),
                                "width": base_envelope.shape[-1]})
                base_weights_i = torch.index_select(base_envelope, 0, source_node_index)
                base_weights_j = torch.index_select(base_envelope, 0, target_node_index)
                scope = _saved_recompute.active()
                if scope is not None:
                    # Saved-tensor recompute: gathers of the bond-level envelope.
                    scope.register((base_weights_i,), _gather_rows, base_envelope, source_node_index)
                    scope.register((base_weights_j,), _gather_rows, base_envelope, target_node_index)
                smooth_weight = base_weights_i * base_weights_j
                if scope is not None:
                    scope.register_product(smooth_weight, base_weights_i, base_weights_j)
            # Fusion feature
            use_indexed, projection_config = (False, None) if atom_feat is None else _projection_choice(
                self.edge_nonlinear_update, edge_feat_0, (atom_feat, node_feat),
                (graph['atom_list'], target_node_index, source_node_index), 4,
                local_enabled=dispatcher.use_gather_cat4(edge_feat_0, atom_feat, node_feat),
                dispatcher=dispatcher,
            )
            # D8: parts of different widths ([32|128|32|32]) are indexed-projection only.
            mixed_width = atom_feat is not None and not (
                edge_feat_0.shape[1] == atom_feat.shape[1] == node_feat.shape[1])
            if use_indexed:
                if hasattr(self.edge_nonlinear_update, "forward_aligned_gather_cat4"):
                    refine_p0 = None
                    if (_MERGED_REFINEMENT_PROJECTION or projection_config is not None) and edge_feat_0.is_contiguous():
                        block_getter = getattr(
                            self.edge_nonlinear_update,
                            "aligned_first_linear_block",
                            None,
                        )
                        if block_getter is None:
                            rec = _recorder._ACTIVE
                            if rec is not None:
                                _record_refinement_merge_decline(
                                    rec, "no_aligned_first_linear_block", edge_feat_0)
                        if block_getter is not None:
                            # _split_first_linear_params carries its own
                            # frozen-weight and capture-cold-cache guards.
                            w0 = block_getter(
                                edge_feat_0.shape[0],
                                (edge_feat_0.shape[1], atom_feat.shape[1],
                                 node_feat.shape[1], node_feat.shape[1]),
                                4,
                                edge_feat_0.device,
                                edge_feat_0.dtype,
                                projection_config=projection_config,
                            )
                            if w0 is not None:
                                from fastmd.engine.kernels.merged_frozen_projections import projection_with_alias

                                refine_p0, edge_feat_residual = (
                                    projection_with_alias(edge_feat_0, w0)
                                )
                            else:
                                rec = _recorder._ACTIVE
                                if rec is not None:
                                    _record_refinement_merge_decline(
                                        rec, "first_linear_block_unavailable", edge_feat_0)
                    else:
                        rec = _recorder._ACTIVE
                        if rec is not None:
                            _record_refinement_merge_decline(
                                rec, "disabled" if not _MERGED_REFINEMENT_PROJECTION
                                and projection_config is None else "not_contiguous", edge_feat_0)
                    refine_fusion_feat_nonlinear = self.edge_nonlinear_update.forward_aligned_gather_cat4(
                        edge_feat_0,
                        atom_feat,
                        graph['atom_list'],
                        node_feat,
                        target_node_index,
                        source_node_index,
                        dispatcher.sorted_vjp("line"),
                        precomputed_p0=refine_p0,
                        projection_config=projection_config,
                    )
                    if refine_fusion_feat_nonlinear is None or refine_p0 is None:
                        edge_feat_residual = edge_feat
                if refine_fusion_feat_nonlinear is None and not mixed_width:
                    refine_fusion_feat = dispatcher.gather_cat(
                        edge_feat_0,
                        (atom_feat, node_feat),
                        (graph['atom_list'], target_node_index, source_node_index),
                    )
            if refine_fusion_feat_nonlinear is None and (not use_indexed or mixed_width):
                rec = _recorder._ACTIVE
                if rec is not None:
                    rec.record("gather_cat4", "torch",
                               {"rows": edge_feat_0.shape[0], "width": edge_feat_0.shape[-1]})
                chunk = _saved_recompute.chunk_rows(edge_feat_0.shape[0])
                mlp = self.edge_nonlinear_update
                if (chunk and hasattr(mlp, "forward_gathered_chunks") and not mlp.use_fp16
                        and mlp._first_linear_modules() is not None
                        and not any(p.requires_grad for m in mlp._first_linear_modules() for p in m.parameters())
                        and not getattr(dispatcher, "private_sink", False)):
                    # Saved-tensor recompute: the [rows, 224] concatenation is the forward's widest
                    # transient; build and project it a row chunk at a time.
                    refine_fusion_feat_nonlinear = mlp.forward_gathered_chunks(
                        edge_feat_0,
                        ((atom_feat, graph['atom_list']), (node_feat, target_node_index),
                         (node_feat, source_node_index)),
                        chunk,
                    )
                else:
                    source_node_feat = dispatcher.gather(node_feat, source_node_index)
                    target_node_feat = dispatcher.gather(node_feat, target_node_index)
                    three_body_atom_feat = dispatcher.gather(atom_feat, graph['atom_list'])
                if refine_fusion_feat_nonlinear is None:
                    refine_fusion_feat = torch.cat([edge_feat_0, three_body_atom_feat, target_node_feat, source_node_feat], dim=1)
        
        # Nonlinear            
        if refine_fusion_feat_nonlinear is None:
            refine_fusion_feat_nonlinear = _module_forward(self.edge_nonlinear_update, refine_fusion_feat)
        return refine_fusion_feat_nonlinear, smooth_weight, edge_feat_residual

    def refinement_payload(self, raw, weight):
        """Shared weighted transport and optional smoothed edge-update input."""
        weighted = raw * weight
        edge_input = (
            weighted if self.graph_type == "atom graph" and self.use_smoothed_for_delta_edge
            else raw
        )
        return weighted, edge_input

    def refinement_nodes(self, node_feat, rows, weight, target_index, graph):
        dispatcher = graph["_ops"]
        # C10: compiled layers keep their traced segment sum (is_compiling first: no guard).
        if not torch.compiler.is_compiling() and dispatcher.segment_sum_csr_enabled:
            refine_node_feas = dispatcher.segment_sum_graph(
                rows, target_index, len(node_feat), graph, "target", weight=weight,
                bin_count=graph['target_bincount'],
            )
        else:
            refine_node_feas = dispatcher.segment_sum(
                rows, target_index, len(node_feat), weight=weight,
                bin_count=graph['target_bincount'],
            )
        delta_node_feat = self.node_FFN(refine_node_feas)
        return dispatcher.residual(delta_node_feat, node_feat, self.node_res_weight)

    def refinement_node_update(self, node_feat, reduced, graph):
        """The node update of ``refinement_nodes`` on an already reduced segment
        sum (see ``Graph_Attention_Layer.attention_node_update``)."""
        delta_node_feat = self.node_FFN(reduced)
        return graph["_ops"].residual(delta_node_feat, node_feat, self.node_res_weight)

    def _fused_edge_ffn(self, edge_input):
        """The edge FFN ``W2 silu(W1 edge_input)`` in one kernel per direction (C4), else None (recorded)."""
        reason = _edge_ffn_decline(self.edge_FFN, edge_input)
        if reason is not None:
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.fallback("edge_ffn", "triton", reason, {"rows": edge_input.shape[0], "width": edge_input.shape[-1]})
            return None
        from fastmd.models.matris.op.triton_edge_ffn import fused_edge_ffn
        layers = self.edge_FFN.layers
        return fused_edge_ffn(edge_input, layers[0].weight, layers[3].weight)

    def refinement_edges(self, edge_feat, edge_input, graph,
                         directed2undirected=None, skip_edge_update=False):
        if skip_edge_update:
            # Readout-dead update (the last block's line-graph three-body update;
            # at P=1 also its bond update under a conservative head): model.forward
            # returns it but nothing consumes it; skip the edge_FFN GEMMs, the pair
            # average and the residual entirely.
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("refinement_edge_update", "skipped_dead_tail",
                           {"rows": edge_feat.shape[0], "width": edge_feat.shape[-1]})
            return edge_feat

        if (not torch.compiler.is_compiling() and _FUSED_EDGE_FFN and self.graph_type != "atom graph"
                and not self._compiled_block):
            delta_edge_feat = self._fused_edge_ffn(edge_input)
            if delta_edge_feat is not None:
                return _policy_residual(delta_edge_feat, edge_feat, self.edge_res_weight, graph)

        delta_edge_feat = _module_forward(self.edge_FFN, edge_input)

        if self.graph_type == "atom graph":
            delta_edge_feat = _policy_pair_average(
                delta_edge_feat, directed2undirected, graph,
            ) # [2*edge, dim] -> [edge, dim]

        return _policy_residual(delta_edge_feat, edge_feat, self.edge_res_weight, graph)


def _edge_ffn_decline(edge_ffn, edge_input) -> str | None:
    """None if ``edge_ffn`` (Linear-SiLU-Dropout-Linear, no bias) runs fused (C4), else why not."""
    layers = edge_ffn.layers
    if edge_ffn.use_fp16:
        return "fp16"
    if not (len(layers) == 4 and type(layers[0]) is nn.Linear and type(layers[3]) is nn.Linear
            and isinstance(layers[1], (FusedSiLU, nn.SiLU)) and type(layers[2]) is nn.Dropout
            and (layers[2].p == 0.0 or not layers[2].training)):
        return "layers"
    if layers[0].bias is not None or layers[3].bias is not None:
        return "bias"
    from fastmd.models.matris.op.triton_edge_ffn import edge_ffn_supported
    return edge_ffn_supported(edge_input, layers[0].weight, layers[3].weight)


_BLOCK_LAYERS = (
    "attn_block_line_graph", "attn_block_atom_graph",
    "refine_block_line_graph", "refine_block_atom_graph",
)


def _record_checkpoint(rec, block, layer, use_checkpoint, runtime) -> None:
    """Name the execution of one interaction layer (checkpoint on/off/runtime)."""
    name = next((name for name in _BLOCK_LAYERS if getattr(block, name, None) is layer),
                type(layer).__name__)
    rec.record("layer_checkpoint", "runtime" if runtime is not None
               else "on" if use_checkpoint else "off", {
                   "layer": name, "enable_checkpoint": block.enable_checkpoint,
                   "compiled": bool(getattr(block, "_is_compiled", False)),
               })


class Interaction_Block(nn.Module):
    """
    Interaction Block for MatRIS that processes both atom graphs and line graphs.
    
    This block performs attention-based message passing and refinement on two hierarchical graph structures:
    1. Atom graph: Nodes represent atoms, edges represent bonds
    2. Line graph: Nodes represent bonds, edges represent three-body interactions (angles)
    
    Attributes:
        attn_block_atom_graph (Graph_Attention_Layer): Attention layer for atom graph
        attn_block_line_graph (Graph_Attention_Layer): Attention layer for line graph
        refine_block_atom_graph (Refinement): Refinement layer for atom graph  
        refine_block_line_graph (Refinement): Refinement layer for line graph
    """
    
    def __init__(self,
                 node_feat_dim: int = 128,
                 edge_feat_dim: int = 128,
                 three_body_feat_dim: int = 128,
                 num_radial: int = 7,
                 num_angular: int = 7,
                 dropout: float = 0.0, 
                 use_bias: bool = False,
                 use_smoothed_for_delta_edge: bool = False,
                 mlp_type: str = "GateMLP",
                 norm_type: str = "layer",
                 activation_type: str = "silu",
                 enable_compile: bool = False,
                 enable_checkpoint: bool | None = False,
                 last_block: bool = False,
                 ):
        """
        Initialize the Interaction Block.

        Args:
            node_feat_dim (int): Dimension of node features (atom features)
            edge_feat_dim (int): Dimension of edge features (bond features)  
            three_body_feat_dim (int): Dimension of three-body features (angle features)
            mlp_type (str): Type of MLP to use in the layers
            norm_type (str): Type of normalization to apply
            activation_type (str): Type of activation function to use
            enable_compile (bool): Whether to compile the inner attention and
                refinement blocks.
            enable_checkpoint (bool | None): False disables activation
                checkpointing, True forces checkpointing on, and None uses the
                automatic threshold.
        """
        super().__init__()
        self.enable_compile = enable_compile
        self.enable_checkpoint = enable_checkpoint
        self.last_block = bool(last_block)
        
        self.attn_block_atom_graph = Graph_Attention_Layer(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=node_feat_dim,
                use_bias=use_bias,
                mlp_type=mlp_type,
                norm_type=norm_type,
                activation_type=activation_type,
            )

        self.attn_block_line_graph = Graph_Attention_Layer(
                node_feat_dim=edge_feat_dim,
                edge_feat_dim=three_body_feat_dim,
                hidden_dim=edge_feat_dim,
                use_bias=use_bias,
                mlp_type=mlp_type,
                norm_type=norm_type,
                activation_type=activation_type,
                use_fp16=False,
            )
        if self.last_block:
            # C3: the last block's backward runs first, before the caching allocator holds a
            # free block of a merged projection's [rows, sum(splits)] gradient buffer size (the
            # earlier blocks' buffers reuse one), so a buffer here would take a new segment at
            # the peak: its per-output GEMMs keep the former peak reserved memory.
            self.attn_block_atom_graph.merged_grad_buffer = False
            self.attn_block_line_graph.merged_grad_buffer = False
        
        self.refine_block_atom_graph = Refinement(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=node_feat_dim,  
                num_basis=num_radial,      
                dropout=dropout,            
                activation_type=activation_type,
                norm_type=norm_type,
                use_bias=use_bias,
                mlp_type=mlp_type,
                graph_type="atom graph",
                use_smoothed_for_delta_edge=use_smoothed_for_delta_edge,
            )
        
        self.refine_block_line_graph = Refinement(
                node_feat_dim=edge_feat_dim,
                edge_feat_dim=three_body_feat_dim,
                hidden_dim=edge_feat_dim,  
                num_basis=num_angular,     
                dropout=dropout,          
                activation_type=activation_type,
                norm_type=norm_type,
                use_bias=use_bias,
                mlp_type=mlp_type,
                graph_type="line graph",
                atom_feat_dim=node_feat_dim,
                use_fp16=False, 
            )
    
    def forward(
        self,
        batch_graph: Dict,
        node_feat: Tensor, 
        edge_feat: Tensor, 
        threebody_feat: Tensor | None,
        smooth_weight: Tensor,
        *,
        runtime=None,
        edge_output: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the Interaction Block.
        
        Args:
            batch_graph: Graph object containing:
                - atom_graph_dict: Atom graph structure
                - line_graph_dict: Bond graph (line graph) structure  
                - directed2undirected: Mapping from directed to undirected edges
                - bond_bases_bg: Smooth weights for bond graph
                - bond_bases_ag: Smooth weights for atom graph
            node_feat (Tensor): Node features [num_atoms, node_feat_dim]
            edge_feat (Tensor): Edge features [num_bonds, edge_feat_dim] 
            threebody_feat (Tensor): Three-body features [num_angles, three_body_feat_dim] or None
            bincount_atom_graph (Dict): Bincount information for atom graph
            bincount_line_graph (Dict): Bincount information for line graph
            edge_output (bool): False when nothing reads the returned edge features
                (a conservative head after the last block).
        """
        # Initialize variables to handle both cases (with and without threebody features)
        attn_edge_feat = edge_feat 
        attn_threebody_feat = threebody_feat
        update_edge_feat = edge_feat
        update_threebody_feat = threebody_feat 
        has_real_line_graph = batch_graph.get('_has_real_line_graph')
        enable_checkpoint = self.enable_checkpoint
        if enable_checkpoint is None:
            use_checkpoint = (
                isinstance(threebody_feat, torch.Tensor)
                and threebody_feat.shape[0] > THRESHOLD_VALUE
            )
        else:
            use_checkpoint = enable_checkpoint
        if runtime is None and self.enable_compile and not getattr(self, "_is_compiled", False):
            # C4: the fused edge FFN declines in compiled blocks, also where one of their frames runs
            # eagerly under dynamo (there is_compiling() is False, and its callees would compile anew).
            self.refine_block_atom_graph._compiled_block = True
            self.refine_block_line_graph._compiled_block = True
            self.attn_block_atom_graph = torch.compile(
                self.attn_block_atom_graph, dynamic=True
            )
            self.attn_block_line_graph = torch.compile(
                self.attn_block_line_graph, dynamic=True
            )
            self.refine_block_atom_graph = torch.compile(
                self.refine_block_atom_graph, dynamic=True
            )
            self.refine_block_line_graph = torch.compile(
                self.refine_block_line_graph, dynamic=True
            )
            self._is_compiled = True

        # Dead edge updates (MATRIS_SKIP_DEAD_THREEBODY_TAIL), host-static so CUDA graphs capture the
        # pruned program; BondPartialRuntime prunes the same at P>1.
        skip_dead = _SKIP_DEAD_THREEBODY_TAIL
        last_block = getattr(self, "last_block", False)

        # Process line graph (bond graph) with attention if threebody features exist
        if threebody_feat is not None: 
            attn_edge_feat, attn_threebody_feat = self.wrapper_attn_layer(
                attn_layer=self.attn_block_line_graph,
                node_feat=edge_feat,
                edge_feat=threebody_feat,
                graph=batch_graph['line_graph_dict'],
                use_checkpoint=use_checkpoint, 
                runtime=None if runtime is None else runtime.for_layer("line_attention"),
            )
            if has_real_line_graph is not None:
                rec = _recorder._ACTIVE
                if rec is not None:
                    rec.record("line_graph_select", "torch_where" if runtime is None
                               else "runtime:select_line_nodes", {"stage": "attention"})
                attn_edge_feat = (
                    torch.where(has_real_line_graph, attn_edge_feat, edge_feat)
                    if runtime is None else runtime.select_line_nodes(
                        attn_edge_feat, edge_feat, has_real_line_graph,
                    )
                )

        # Process atom graph with attention
        attn_node_feat, attn_edge_feat = self.wrapper_attn_layer(
            attn_layer=self.attn_block_atom_graph,
            node_feat=node_feat, 
            edge_feat=attn_edge_feat, 
            graph=batch_graph['atom_graph_dict'], 
            directed2undirected=batch_graph['directed2undirected'],
            # No triplets: the atom refinement reads the block-input bonds.
            skip_edge_update=skip_dead and threebody_feat is None,
            runtime=None if runtime is None else runtime.for_layer("atom_attention"),
        ) 
        
        # Refine line graph features if threebody features exist
        if threebody_feat is not None:
            update_edge_feat, update_threebody_feat = self.wrapper_refine_layer(
                refine_layer=self.refine_block_line_graph,
                node_feat=attn_edge_feat,
                edge_feat=attn_threebody_feat,
                smooth_weight=smooth_weight['line graph'],
                graph=batch_graph['line_graph_dict'],
                atom_feat=attn_node_feat,
                use_checkpoint=use_checkpoint,
                skip_edge_update=skip_dead and last_block,
                runtime=None if runtime is None else runtime.for_layer("line_refinement"),
            )
            if has_real_line_graph is not None:
                rec = _recorder._ACTIVE
                if rec is not None:
                    rec.record("line_graph_select", "torch_where" if runtime is None
                               else "runtime:select_line_nodes", {"stage": "refinement"})
                # Match the no-line-graph branch's initial value, which is the
                # original edge feature, not the atom-attention edge output.
                update_edge_feat = (
                    torch.where(has_real_line_graph, update_edge_feat, edge_feat)
                    if runtime is None else runtime.select_line_nodes(
                        update_edge_feat, edge_feat, has_real_line_graph,
                    )
                )
        
        # Refine atom graph features
        update_node_feat, update_edge_feat = self.wrapper_refine_layer(
            refine_layer=self.refine_block_atom_graph,
            node_feat=attn_node_feat,
            edge_feat=update_edge_feat,
            smooth_weight=smooth_weight['atom graph'],
            graph=batch_graph['atom_graph_dict'],
            directed2undirected=batch_graph['directed2undirected'],
            skip_edge_update=skip_dead and last_block and not edge_output,
            runtime=None if runtime is None else runtime.for_layer("atom_refinement"),
        )
        
        return update_node_feat, update_edge_feat, update_threebody_feat
    
    def wrapper_attn_layer(self,
                            attn_layer: nn.Module,
                            node_feat: Tensor, 
                            edge_feat: Tensor, 
                            graph: Dict,
                            directed2undirected: Tensor = None,
                            use_checkpoint: bool = False,
                            skip_edge_update: bool = False,
                            runtime=None,
                       ):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_checkpoint(rec, self, attn_layer, use_checkpoint, runtime)
        if runtime is not None:
            return attn_layer(
                node_feat=node_feat, edge_feat=edge_feat, graph=graph,
                directed2undirected=directed2undirected, skip_edge_update=skip_edge_update, runtime=runtime,
            )
        if getattr(self, "_is_compiled", False):
            # torch.compile'd blocks: their GEMMs stay cuBLAS (Inductor's), also in frames that run eagerly.
            with _wide.compiled_region():
                return self._attn_layer_call(attn_layer, node_feat, edge_feat, graph, directed2undirected,
                                             use_checkpoint, skip_edge_update)
        return self._attn_layer_call(attn_layer, node_feat, edge_feat, graph, directed2undirected,
                                     use_checkpoint, skip_edge_update)

    def _attn_layer_call(self, attn_layer, node_feat, edge_feat, graph, directed2undirected, use_checkpoint,
                         skip_edge_update):
        if use_checkpoint:
            attn_node_feat, attn_edge_feat = checkpoint(
                attn_layer,
                node_feat, 
                edge_feat, 
                graph,
                directed2undirected,
                skip_edge_update,
                use_reentrant=False,
                # Eval never draws random numbers: do not stash/restore RNG
                # state (a CUDA-graph capture must not touch the generator).
                preserve_rng_state=self.training,
            ) 
        else:
            attn_node_feat, attn_edge_feat = attn_layer(
                node_feat=node_feat, 
                edge_feat=edge_feat, 
                graph=graph,
                directed2undirected=directed2undirected,
                skip_edge_update=skip_edge_update,
            )
        
        return attn_node_feat, attn_edge_feat 
        
    def wrapper_refine_layer(self, 
                            refine_layer: nn.Module,
                            node_feat: Tensor,
                            edge_feat: Tensor,
                            smooth_weight: Tensor,
                            graph: Dict,
                            directed2undirected: Tensor = None,
                            atom_feat: Tensor = None,
                            use_checkpoint: bool = False,
                            skip_edge_update: bool = False,
                            runtime=None,
                        ):
        rec = _recorder._ACTIVE
        if rec is not None:
            _record_checkpoint(rec, self, refine_layer, use_checkpoint, runtime)
        if runtime is not None:
            return refine_layer(
                node_feat=node_feat, edge_feat=edge_feat, smooth_weight=smooth_weight,
                graph=graph, directed2undirected=directed2undirected,
                atom_feat=atom_feat, skip_edge_update=skip_edge_update, runtime=runtime,
            )
        if getattr(self, "_is_compiled", False):
            with _wide.compiled_region():  # as in wrapper_attn_layer
                return self._refine_layer_call(refine_layer, node_feat, edge_feat, smooth_weight, graph,
                                               directed2undirected, atom_feat, use_checkpoint, skip_edge_update)
        return self._refine_layer_call(refine_layer, node_feat, edge_feat, smooth_weight, graph,
                                       directed2undirected, atom_feat, use_checkpoint, skip_edge_update)

    def _refine_layer_call(self, refine_layer, node_feat, edge_feat, smooth_weight, graph, directed2undirected,
                           atom_feat, use_checkpoint, skip_edge_update):
        if use_checkpoint:
            update_node_feat, update_edge_feat = checkpoint(
                refine_layer,
                node_feat,
                edge_feat,
                smooth_weight,
                graph,
                directed2undirected,
                atom_feat,
                skip_edge_update,
                use_reentrant=False,
                preserve_rng_state=self.training,
            )
        else:
            update_node_feat, update_edge_feat = refine_layer(
                    node_feat=node_feat,
                    edge_feat=edge_feat,
                    smooth_weight=smooth_weight,
                    graph=graph,
                    directed2undirected=directed2undirected,
                    atom_feat=atom_feat,
                    skip_edge_update=skip_edge_update,
                )
        return update_node_feat, update_edge_feat 
        
    
