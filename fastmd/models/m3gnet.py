"""MatGL M3GNet adapter for topology contracts and shared sparse lowerings.

This module intentionally uses duck typing and does not make MatGL a required
MatRIS dependency.  The only model-specific code maps PyG relations and the two
M3GNet message-update call sites; lowering selection and the indexed-affine
kernel are shared with MatRIS.
"""

from __future__ import annotations

from fastmd.config import env_value

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from fastmd.graph.topology_contract import BuilderEvidence
from fastmd.graph.topology_contract import ContractProperty
from fastmd.graph.topology_contract import DescriptorLifetime
from fastmd.graph.topology_contract import DomainExtent
from fastmd.graph.topology_contract import DomainSpec
from fastmd.graph.topology_contract import ExtentPolicy
from fastmd.graph.topology_contract import ExtentScope
from fastmd.graph.topology_contract import PropertyKind
from fastmd.graph.topology_contract import RelationSpec
from fastmd.graph.topology_contract import TopologyDomain
from fastmd.graph.topology_contract import TopologyInstance
from fastmd.graph.topology_contract import TopologySchema
from fastmd.graph.topology_contract import VerificationMode
from fastmd.graph.topology_contract import VerificationReport
from fastmd.graph.topology_contract import certify_topology
from fastmd.engine.cache_provenance import CAPTURE as _CAPTURE
from fastmd.engine.cache_provenance import retain_under_capture
from fastmd.engine.topology_lowering import LoweringPlan
from fastmd.engine.topology_lowering import LoweringRequest
from fastmd.engine.topology_lowering import OperatorKind
from fastmd.engine.topology_lowering import compile_lowering_plan


ATOM = "atom"
DIRECTED_EDGE = "directed_edge"
LINE_EDGE = "line_edge"
TRIPLET = "triplet"
GRAPH = "graph"
DUMMY = "dummy"

EDGE_SOURCE = "edge_source"
EDGE_TARGET = "edge_target"
ATOM_GRAPH_ID = "atom_graph_id"
LINE_TO_DIRECTED = "line_to_directed"
LINE_EDGE_OWNER = "line_edge_owner"
TRIPLET_OWNER = "triplet_owner"
TRIPLET_EDGE_A = "triplet_edge_a"
TRIPLET_EDGE_B = "triplet_edge_b"
TRIPLET_SEGMENT = "triplet_segment"


_RELATIONS = (
    RelationSpec(EDGE_SOURCE, DIRECTED_EDGE, ATOM),
    RelationSpec(EDGE_TARGET, DIRECTED_EDGE, ATOM),
    RelationSpec(
        ATOM_GRAPH_ID,
        ATOM,
        GRAPH,
        lifetime=DescriptorLifetime.CAPTURE,
        invalidated_by=("batch_changed",),
    ),
    RelationSpec(LINE_TO_DIRECTED, LINE_EDGE, DIRECTED_EDGE),
    RelationSpec(LINE_EDGE_OWNER, LINE_EDGE, ATOM),
    RelationSpec(TRIPLET_OWNER, TRIPLET, ATOM),
    RelationSpec(TRIPLET_EDGE_A, TRIPLET, LINE_EDGE),
    RelationSpec(TRIPLET_EDGE_B, TRIPLET, LINE_EDGE),
    RelationSpec(TRIPLET_SEGMENT, TRIPLET, LINE_EDGE),
)


def m3gnet_schema(
    *,
    edge_source_sorted: bool = True,
    triplet_segment_sorted: bool = True,
) -> TopologySchema:
    properties: list[ContractProperty] = [
        ContractProperty(PropertyKind.BOUNDED, relation.name) for relation in _RELATIONS
    ]
    properties.extend(
        [
            ContractProperty(PropertyKind.GRAPH_CONTIGUOUS, ATOM_GRAPH_ID),
            ContractProperty(
                PropertyKind.INCIDENCE,
                TRIPLET_OWNER,
                (TRIPLET_EDGE_A, TRIPLET_EDGE_B, LINE_EDGE_OWNER),
                ExtentScope.ACTIVE,
            ),
        ]
    )
    if edge_source_sorted:
        properties.append(
            ContractProperty(
                PropertyKind.SORTED_BY,
                EDGE_SOURCE,
                scope=ExtentScope.ACTIVE,
            )
        )
    if triplet_segment_sorted:
        properties.append(
            ContractProperty(
                PropertyKind.SORTED_BY,
                TRIPLET_SEGMENT,
                scope=ExtentScope.ACTIVE,
            )
        )
    domains = (
        DomainSpec(ATOM, TopologyDomain.ATOM),
        DomainSpec(DIRECTED_EDGE, TopologyDomain.DIRECTED_EDGE),
        DomainSpec(LINE_EDGE, TopologyDomain.DIRECTED_EDGE),
        DomainSpec(TRIPLET, TopologyDomain.TRIPLET),
        DomainSpec(GRAPH, TopologyDomain.GRAPH, ExtentPolicy.DYNAMIC),
        DomainSpec(DUMMY, TopologyDomain.DUMMY),
    )
    return TopologySchema(domains, _RELATIONS, tuple(properties))


@dataclass(frozen=True)
class M3GNetTopologyBinding:
    instance: TopologyInstance
    report: VerificationReport
    plan: LoweringPlan


def m3gnet_lowering_requests(feature_dim: int = 64) -> tuple[LoweringRequest, ...]:
    return (
        LoweringRequest.create(
            "edge_indexed_affine",
            OperatorKind.INDEXED_AFFINE,
            parameters={"feature_dim": feature_dim},
            index_a=EDGE_SOURCE,
            index_b=EDGE_TARGET,
        ),
        LoweringRequest.create(
            "node_indexed_affine",
            OperatorKind.INDEXED_AFFINE,
            parameters={"feature_dim": feature_dim},
            index_a=EDGE_SOURCE,
            index_b=EDGE_TARGET,
        ),
        LoweringRequest.create(
            "edge_segment_vjp",
            OperatorKind.SORTED_SEGMENT_VJP,
            segment=EDGE_SOURCE,
        ),
        LoweringRequest.create(
            "threebody_incidence",
            OperatorKind.LINE_INCIDENCE,
            owner=TRIPLET_OWNER,
            edge_a=TRIPLET_EDGE_A,
            edge_b=TRIPLET_EDGE_B,
            edge_owner=LINE_EDGE_OWNER,
        ),
        LoweringRequest.create(
            "threebody_segment_vjp",
            OperatorKind.SORTED_SEGMENT_VJP,
            segment=TRIPLET_SEGMENT,
        ),
    )


def _matgl_segment_ids_on_device(n_triple_ij: Tensor, triplet_count: int) -> Tensor:
    """Device implementation of MatGL's zero-count-preserving segment rule."""

    segments = torch.zeros(
        triplet_count,
        dtype=torch.int32,
        device=n_triple_ij.device,
    )
    if triplet_count == 0:
        return segments.long()
    boundaries = n_triple_ij.cumsum(0)[:-1].long()
    segments[boundaries] = 1
    return segments.cumsum(0)


def bind_m3gnet_topology(
    graph: Any,
    line_graph: dict[str, Tensor],
    *,
    mode: VerificationMode | str = VerificationMode.BUILDER,
    edge_source_sorted: bool = True,
    triplet_segment_sorted: bool = True,
    feature_dim: int = 64,
) -> M3GNetTopologyBinding:
    """Map a PyG M3GNet graph to the shared contract roles and certify it."""

    edge_index = graph.edge_index.long()
    line_edge_index = line_graph["line_edge_index"].long()
    line_count = int(line_graph["n_triple_ij"].shape[0])
    kept = line_graph.get("kept_edge_ids")
    if kept is None:
        kept = torch.arange(line_count, device=edge_index.device)
    kept = kept[:line_count].long()
    line_owner = edge_index[0].index_select(0, kept)
    triplet_a = line_edge_index[0]
    triplet_b = line_edge_index[1]
    triplet_owner = line_owner.index_select(0, triplet_a)
    triplet_segment = _matgl_segment_ids_on_device(
        line_graph["n_triple_ij"], int(line_edge_index.shape[1])
    )

    batch = getattr(graph, "batch", None)
    atom_count = int(graph.pos.shape[0])
    if batch is None:
        batch = torch.zeros(atom_count, dtype=torch.long, device=edge_index.device)
        graph_count = 1
    else:
        batch = batch.long()
        graph_count = int(getattr(graph, "num_graphs", int(batch.max().item()) + 1))
    edge_count = int(edge_index.shape[1])
    triplet_count = int(line_edge_index.shape[1])
    schema = m3gnet_schema(
        edge_source_sorted=edge_source_sorted,
        triplet_segment_sorted=triplet_segment_sorted,
    )
    relations = {
        EDGE_SOURCE: edge_index[0],
        EDGE_TARGET: edge_index[1],
        ATOM_GRAPH_ID: batch,
        LINE_TO_DIRECTED: kept,
        LINE_EDGE_OWNER: line_owner,
        TRIPLET_OWNER: triplet_owner,
        TRIPLET_EDGE_A: triplet_a,
        TRIPLET_EDGE_B: triplet_b,
        TRIPLET_SEGMENT: triplet_segment,
    }
    extents = {
        ATOM: DomainExtent(atom_count, atom_count, atom_count),
        DIRECTED_EDGE: DomainExtent(edge_count, edge_count, edge_count),
        LINE_EDGE: DomainExtent(line_count, line_count, line_count),
        TRIPLET: DomainExtent(triplet_count, triplet_count, triplet_count),
        GRAPH: DomainExtent(graph_count, graph_count, graph_count),
        DUMMY: DomainExtent(0, 0, 0),
    }
    evidence = BuilderEvidence("matgl.m3gnet_pyg", schema.properties)
    instance = TopologyInstance(schema, extents, relations, evidence.issuer)
    certified, report = certify_topology(instance, evidence=evidence, mode=mode)
    assert certified.certificate is not None
    plan = compile_lowering_plan(
        schema,
        certified.certificate,
        m3gnet_lowering_requests(feature_dim),
    )
    return M3GNetTopologyBinding(certified, report, plan)


class _ContractM3GNetGraphConv(nn.Module):
    """Drop-in graph-conv wrapper that reuses the shared indexed-affine kernel."""

    def __init__(
        self,
        base: nn.Module,
        plan: LoweringPlan,
        enabled_requests: frozenset[str],
    ) -> None:
        super().__init__()
        self.base = base
        self.plan = plan
        self.enabled_requests = enabled_requests
        self.specialized_calls = 0
        self.fallback_calls = 0
        self._parameter_cache: dict[tuple, tuple[tuple[Tensor, ...], Tensor | None]] = {}

    @staticmethod
    def _run_tail(sequence: nn.Sequential, value: Tensor) -> Tensor:
        for layer in list(sequence)[2:]:
            value = layer(value)
        return value

    def _parameters_for(
        self,
        module: nn.Module,
        chunk_dim: int,
        aligned_position: int,
    ) -> tuple[tuple[Tensor, ...], Tensor | None] | None:
        layers = getattr(module, "layers", None)
        gates = getattr(module, "gates", None)
        if layers is None or gates is None or len(layers) < 3 or len(gates) < 3:
            return None
        core = layers[0]
        gate = gates[0]
        if not isinstance(core, nn.Linear) or not isinstance(gate, nn.Linear):
            return None
        if core.in_features != 3 * chunk_dim or gate.in_features != 3 * chunk_dim:
            return None
        key = (
            id(module),
            core.weight.data_ptr(),
            core.weight._version,
            gate.weight.data_ptr(),
            gate.weight._version,
            chunk_dim,
            aligned_position,
        )
        cached = self._parameter_cache.get(key)
        if cached is not None:
            if _CAPTURE.holder is not None:
                retain_under_capture(cached, (core.weight, gate.weight))
            return cached
        logical_positions = [aligned_position] + [
            position for position in range(3) if position != aligned_position
        ]
        weights = []
        for position in logical_positions:
            start = position * chunk_dim
            end = start + chunk_dim
            weights.append(
                torch.cat(
                    [
                        core.weight.detach()[:, start:end],
                        gate.weight.detach()[:, start:end],
                    ],
                    dim=0,
                ).contiguous()
            )
        if core.bias is None:
            bias = None
        else:
            bias = torch.cat([core.bias.detach(), gate.bias.detach()]).contiguous()
        result = (tuple(weights), bias)
        self._parameter_cache = {key: result}
        if _CAPTURE.holder is not None:
            retain_under_capture(result, (core.weight, gate.weight))
        return result

    def _indexed_update(
        self,
        module: nn.Module,
        aligned: Tensor,
        gathered: Tensor,
        index_a: Tensor,
        index_b: Tensor,
        request_key: str,
    ) -> Tensor | None:
        if (
            request_key not in self.enabled_requests
            or not aligned.is_cuda
            or not gathered.is_cuda
        ):
            return None
        selection = self.plan.selection(request_key)
        if not selection.lowering.startswith("indexed_affine.decomposed"):
            return None
        if aligned.dim() != 2 or gathered.dim() != 2:
            return None
        chunk_dim = int(aligned.shape[1])
        if gathered.shape[1] != chunk_dim:
            return None
        params = self._parameters_for(module, chunk_dim, aligned_position=2)
        if params is None:
            return None
        weights, bias = params
        from fastmd.engine.kernels.decomposed_indexed_cat_silu_linear import decomposed_indexed_cat3_silu_linear

        fused = decomposed_indexed_cat3_silu_linear(
            aligned.contiguous(),
            gathered.contiguous(),
            index_a,
            index_b,
            weights,
            bias,
            selection.lowering == "indexed_affine.decomposed_sorted_vjp",
        )
        if fused is None:
            return None
        core, gate = fused
        core = self._run_tail(module.layers, core)
        gate = self._run_tail(module.gates, gate)
        self.specialized_calls += 1
        return core * gate

    def forward(
        self,
        edge_index: Tensor,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor | None,
        rbf: Tensor,
        node_batch: Tensor | None,
        edge_batch: Tensor | None,
        num_nodes: int,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if bool(getattr(self.base, "include_state", False)):
            self.fallback_calls += 1
            return self.base(
                edge_index,
                edge_feat,
                node_feat,
                state_feat,
                rbf,
                node_batch,
                edge_batch,
                num_nodes,
                num_graphs,
            )

        src, dst = edge_index[0], edge_index[1]
        edge_update = self._indexed_update(
            self.base.edge_update_func,
            edge_feat,
            node_feat,
            src,
            dst,
            "edge_indexed_affine",
        )
        if edge_update is None:
            edge_update = self.base.edge_update_(
                edge_index, edge_feat, node_feat, rbf, None
            )
            self.fallback_calls += 1
        else:
            edge_update = edge_update * self.base.edge_weight_func(rbf)
        edge_feat_new = edge_feat + edge_update

        node_update = self._indexed_update(
            self.base.node_update_func,
            edge_feat_new,
            node_feat,
            src,
            dst,
            "node_indexed_affine",
        )
        if node_update is None:
            node_update = self.base.node_update_(
                edge_index, edge_feat_new, node_feat, rbf, None, num_nodes
            )
            self.fallback_calls += 1
        else:
            from torch_geometric.utils import scatter

            messages = node_update * self.base.node_weight_func(rbf)
            node_update = scatter(messages, src, dim=0, dim_size=num_nodes, reduce="sum")
        return edge_feat_new, node_feat + node_update, state_feat


class _ContractThreeBodyInteractions(nn.Module):
    """Use the certified line relation directly as segment/incidence metadata."""

    def __init__(self, base: nn.Module, segment_ids: Tensor) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("segment_ids", segment_ids, persistent=False)
        self.direct_calls = 0

    def forward(
        self,
        edge_dst_atom: Tensor,
        line_edge_index: Tensor,
        n_triple_ij: Tensor,
        num_bonds: int,
        three_basis: Tensor,
        three_cutoff: Tensor,
        node_feat: Tensor,
        edge_feat: Tensor,
    ) -> Tensor:
        if self.segment_ids.numel() != line_edge_index.shape[1]:
            return self.base(
                edge_dst_atom,
                line_edge_index,
                n_triple_ij,
                num_bonds,
                three_basis,
                three_cutoff,
                node_feat,
                edge_feat,
            )
        if line_edge_index.shape[1] == 0:
            self.direct_calls += 1
            return edge_feat
        source_edge = line_edge_index[0]
        target_edge = line_edge_index[1]
        end_atom_indices = edge_dst_atom[target_edge].long()
        updated_atoms = self.base.update_network_atom(node_feat)
        basis = three_basis * updated_atoms[end_atom_indices]
        weights = three_cutoff[source_edge] * three_cutoff[target_edge]
        basis = (basis * weights[:, None]).to(torch.float32)
        new_bonds = torch.zeros(
            (num_bonds, basis.shape[1]),
            dtype=basis.dtype,
            device=basis.device,
        )
        segment_ids = self.segment_ids.view(-1, 1).expand_as(basis)
        new_bonds.scatter_add_(0, segment_ids, basis)
        self.direct_calls += 1
        return edge_feat + self.base.update_network_bond(new_bonds)


def install_m3gnet_topology_lowerings(
    model: nn.Module,
    plan: LoweringPlan,
    *,
    enable_indexed_affine: bool | None = None,
    enabled_requests: tuple[str, ...] | None = None,
    topology: TopologyInstance | None = None,
) -> int:
    """Install contract-selected wrappers after loading and moving a model."""

    if enable_indexed_affine is None:
        enable_indexed_affine = env_value("MATRIS_M3GNET_INDEXED_AFFINE", "1") != "0"
    if not enable_indexed_affine:
        request_set: frozenset[str] = frozenset()
    elif enabled_requests is not None:
        request_set = frozenset(enabled_requests)
    else:
        request_set = frozenset(
            request
            for request, env_name in (
                ("edge_indexed_affine", "MATRIS_M3GNET_INDEXED_EDGE"),
                ("node_indexed_affine", "MATRIS_M3GNET_INDEXED_NODE"),
            )
            if env_value(env_name, "1") != "0"
        )
    request_set = frozenset(
        request
        for request in request_set
        if plan.selection(request).lowering.startswith("indexed_affine.decomposed")
    )
    count = 0
    graph_layers = tuple(getattr(model, "graph_layers", ()))
    if request_set and not graph_layers:
        raise TypeError("model does not expose M3GNet graph_layers")
    for layer in graph_layers:
        conv = layer.conv
        if request_set:
            if isinstance(conv, _ContractM3GNetGraphConv):
                conv.plan = plan
                conv.enabled_requests = request_set
            else:
                layer.conv = _ContractM3GNetGraphConv(conv, plan, request_set)
            count += 1
        elif isinstance(conv, _ContractM3GNetGraphConv):
            layer.conv = conv.base

    enable_threebody = (
        topology is not None
        and plan.uses("threebody_incidence", "line_incidence.direct")
        and plan.uses("threebody_segment_vjp", "segment_vjp.contiguous")
    )
    if enable_threebody:
        assert topology is not None
        segment_ids = topology.relation(TRIPLET_SEGMENT)
    for index, interaction in enumerate(getattr(model, "three_body_interactions", ())):
        if enable_threebody:
            if isinstance(interaction, _ContractThreeBodyInteractions):
                interaction.segment_ids = segment_ids
            else:
                model.three_body_interactions[index] = _ContractThreeBodyInteractions(
                    interaction, segment_ids
                )
            count += 1
        elif isinstance(interaction, _ContractThreeBodyInteractions):
            model.three_body_interactions[index] = interaction.base
    return count


def uninstall_m3gnet_topology_lowerings(model: nn.Module) -> int:
    count = 0
    for layer in getattr(model, "graph_layers", ()):
        if isinstance(layer.conv, _ContractM3GNetGraphConv):
            layer.conv = layer.conv.base
            count += 1
    for index, interaction in enumerate(getattr(model, "three_body_interactions", ())):
        if isinstance(interaction, _ContractThreeBodyInteractions):
            model.three_body_interactions[index] = interaction.base
            count += 1
    return count


def m3gnet_lowering_stats(model: nn.Module) -> dict[str, int]:
    wrappers = [
        layer.conv
        for layer in getattr(model, "graph_layers", ())
        if isinstance(layer.conv, _ContractM3GNetGraphConv)
    ]
    threebody = [
        interaction
        for interaction in getattr(model, "three_body_interactions", ())
        if isinstance(interaction, _ContractThreeBodyInteractions)
    ]
    return {
        "wrapped_layers": len(wrappers),
        "specialized_calls": sum(wrapper.specialized_calls for wrapper in wrappers),
        "fallback_calls": sum(wrapper.fallback_calls for wrapper in wrappers),
        "wrapped_threebody_layers": len(threebody),
        "direct_incidence_calls": sum(layer.direct_calls for layer in threebody),
    }
