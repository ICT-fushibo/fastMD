"""MatRIS's topology bindings: the binding of its batched graph to the schema of ``fastmd.graph.matris_topology``,
certified once per graph (:func:`bind_matris_topology`), and the operator roles its sites request lowerings for from
the model-independent lowering registry (:func:`matris_lowering_requests`, ``fastmd.engine.topology_lowering``)."""

from __future__ import annotations

from fastmd.config import env_value

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from fastmd.engine.topology_lowering import LoweringRequest, OperatorKind
from fastmd.graph.matris_topology import (
    ATOM,
    ATOM_GRAPH_ID,
    DIRECTED_EDGE,
    DIRECTED_TO_PAIR,
    DUMMY,
    EDGE_SOURCE,
    EDGE_TARGET,
    GRAPH,
    PAIR_TO_DIRECTED,
    TRIPLET,
    TRIPLET_EDGE_A,
    TRIPLET_EDGE_B,
    TRIPLET_OWNER,
    TRIPLET_PAIR_A,
    TRIPLET_PAIR_B,
    UNDIRECTED_PAIR,
    matris_builder_evidence,
    matris_schema,
)
from fastmd.graph.radiusgraph import RadiusGraph
from fastmd.graph.topology_contract import (
    BuilderEvidence,
    ContractProperty,
    DomainExtent,
    ExtentScope,
    PropertyKind,
    TopologyCertificate,
    TopologyInstance,
    VerificationMode,
    VerificationReport,
    certify_topology,
)


_VERIFIED_BINDING_CACHE: dict[
    tuple, tuple[TopologyCertificate, VerificationReport]
] = {}


def _extent(capacity: int, real: int | None = None, active: int | None = None) -> DomainExtent:
    return DomainExtent(
        real=capacity if real is None else int(real),
        active=capacity if active is None else int(active),
        capacity=int(capacity),
    )


@dataclass(frozen=True)
class MatrisTopologyBinding:
    instance: TopologyInstance
    report: VerificationReport


def bind_matris_topology(
    batched_graph: dict,
    graphs: Sequence[RadiusGraph],
    *,
    mode: VerificationMode | str | None = None,
) -> MatrisTopologyBinding:
    """Bind and certify the descriptor produced by :func:`process_graphs`."""

    graph_evidence = []
    for graph in graphs:
        evidence = getattr(graph, "topology_evidence", None)
        if evidence is None:
            evidence = matris_builder_evidence(
                "fastmd.legacy_radius_graph",
                atom_target_sorted=bool(getattr(graph, "atom_target_sorted", False)),
                line_owner_sorted=bool(getattr(graph, "line_atom_sorted", False)),
            )
        graph_evidence.append(evidence)
    atom_sorted_claim = ContractProperty(
        PropertyKind.SORTED_BY, EDGE_TARGET, scope=ExtentScope.ACTIVE
    )
    line_sorted_claim = ContractProperty(
        PropertyKind.SORTED_BY, TRIPLET_OWNER, scope=ExtentScope.ACTIVE
    )
    atom_sorted = all(
        atom_sorted_claim in evidence.properties for evidence in graph_evidence
    )
    line_sorted = all(
        line_sorted_claim in evidence.properties for evidence in graph_evidence
    )
    incidence_claim = ContractProperty(
        PropertyKind.INCIDENCE,
        TRIPLET_OWNER,
        (TRIPLET_EDGE_A, TRIPLET_EDGE_B, EDGE_TARGET),
        ExtentScope.ACTIVE,
    )
    fixed_capacity_claim = ContractProperty(PropertyKind.FIXED_CAPACITY, ATOM)
    fixed_capacity = all(
        fixed_capacity_claim in evidence.properties for evidence in graph_evidence
    )
    incidence_consistent = all(
        incidence_claim in evidence.properties for evidence in graph_evidence
    )
    schema = matris_schema(
        atom_target_sorted=atom_sorted,
        line_owner_sorted=line_sorted,
        fixed_capacity=fixed_capacity,
        incidence_consistent=incidence_consistent,
    )
    atom_graph = batched_graph["atom_graph_dict"]
    line_graph = batched_graph["line_graph_dict"]
    atom_count = int(batched_graph["atomic_numbers"].shape[0])
    edge_count = int(atom_graph["target_index"].shape[0])
    pair_count = int(batched_graph["undirected2directed"].shape[0])
    triplet_count = int(line_graph["line_graph"].shape[0])
    graph_count = int(batched_graph["num_graphs"])
    device = batched_graph["atomic_numbers"].device
    empty = torch.empty(0, dtype=torch.long, device=device)
    relations = {
        EDGE_TARGET: atom_graph["target_index"],
        EDGE_SOURCE: atom_graph["source_index"],
        DIRECTED_TO_PAIR: batched_graph["directed2undirected"],
        PAIR_TO_DIRECTED: batched_graph["undirected2directed"],
        ATOM_GRAPH_ID: batched_graph["atom_segment"],
        TRIPLET_OWNER: line_graph.get("atom_list", empty),
        TRIPLET_PAIR_A: line_graph.get("target_index", empty),
        TRIPLET_PAIR_B: line_graph.get("source_index", empty),
        TRIPLET_EDGE_A: line_graph.get("target_DE_index", empty),
        TRIPLET_EDGE_B: line_graph.get("source_DE_index", empty),
    }
    extents = {
        ATOM: _extent(atom_count),
        DIRECTED_EDGE: _extent(edge_count),
        UNDIRECTED_PAIR: _extent(pair_count),
        TRIPLET: _extent(triplet_count),
        GRAPH: _extent(graph_count),
        DUMMY: _extent(0),
    }
    issuers = sorted(
        {
            getattr(evidence, "issuer", "legacy")
            for evidence in graph_evidence
        }
    )
    common_properties = set(graph_evidence[0].properties)
    for graph_proof in graph_evidence[1:]:
        common_properties.intersection_update(graph_proof.properties)
    missing_claims = set(schema.properties) - common_properties
    if missing_claims:
        missing = ", ".join(
            sorted(f"{prop.kind.value}({prop.subject})" for prop in missing_claims)
        )
        raise ValueError(f"MatRIS builders do not jointly prove schema claims: {missing}")
    evidence = BuilderEvidence("+".join(issuers), schema.properties)
    instance = TopologyInstance(
        schema=schema,
        extents=extents,
        relations=relations,
        producer=evidence.issuer,
    )
    verify_mode = VerificationMode(
        mode or env_value("MATRIS_TOPOLOGY_VERIFY", VerificationMode.BUILDER.value)
    )
    first_relation = next(iter(relations.values()))
    cache_key = (
        schema.fingerprint,
        instance.extent_signature,
        evidence.issuer,
        first_relation.device.type,
        first_relation.device.index,
        verify_mode.value,
        tuple(id(graph) for graph in graphs),
    )
    cached = _VERIFIED_BINDING_CACHE.get(cache_key)
    if cached is not None:
        certificate, report = cached
        certified = instance.with_certificate(certificate)
    else:
        if (
            verify_mode != VerificationMode.BUILDER
            and first_relation.is_cuda
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "topology verification must run once before CUDA Graph capture"
            )
        certified, report = certify_topology(
            instance, evidence=evidence, mode=verify_mode
        )
        if verify_mode != VerificationMode.BUILDER:
            _VERIFIED_BINDING_CACHE[cache_key] = (certified.certificate, report)
    return MatrisTopologyBinding(certified, report)


def matris_lowering_requests(feature_dim: int = 128) -> tuple[LoweringRequest, ...]:
    """Operator-role bindings used by MatRIS; the registry remains generic."""

    from fastmd.graph.matris_topology import (
        DIRECTED_TO_PAIR,
        EDGE_SOURCE,
        EDGE_TARGET,
        PAIR_TO_DIRECTED,
        TRIPLET_EDGE_A,
        TRIPLET_EDGE_B,
        TRIPLET_OWNER,
        TRIPLET_PAIR_A,
    )

    return (
        LoweringRequest.create(
            "atom_indexed_affine",
            OperatorKind.INDEXED_AFFINE,
            parameters={"feature_dim": feature_dim},
            index_a=EDGE_TARGET,
            index_b=EDGE_SOURCE,
        ),
        LoweringRequest.create(
            "line_indexed_affine",
            OperatorKind.INDEXED_AFFINE,
            parameters={"feature_dim": feature_dim},
            index_a=TRIPLET_OWNER,
            index_b=TRIPLET_PAIR_A,
        ),
        LoweringRequest.create(
            "target_segment_attention",
            OperatorKind.SEGMENT_ATTENTION,
            segment=EDGE_TARGET,
        ),
        LoweringRequest.create(
            "line_target_segment_attention",
            OperatorKind.SEGMENT_ATTENTION,
            segment=TRIPLET_PAIR_A,
        ),
        LoweringRequest.create(
            "pair_algebra",
            OperatorKind.PAIR_ALGEBRA,
            pair_map=DIRECTED_TO_PAIR,
            edge_source=EDGE_SOURCE,
            edge_target=EDGE_TARGET,
            pair_to_edge=PAIR_TO_DIRECTED,
        ),
        LoweringRequest.create(
            "line_incidence",
            OperatorKind.LINE_INCIDENCE,
            owner=TRIPLET_OWNER,
            edge_a=TRIPLET_EDGE_A,
            edge_b=TRIPLET_EDGE_B,
            edge_owner=EDGE_TARGET,
        ),
        LoweringRequest.create(
            "line_envelope",
            OperatorKind.LINE_INCIDENCE,
            parameters={
                # Provenance preference for the envelope-product site only:
                # when enabled, the registry selects the torch.compile-
                # generated lowering over the hand-written kernel for THIS
                # request; the shared line_incidence request keeps its
                # certificate-selected direct lowerings (the constraint
                # rejects the compiled spec when the parameter is absent).
                "compiled_lowerings": 1
                if env_value("MATRIS_COMPILED_LOWERINGS", "0") != "0"
                else 0,
            },
            owner=TRIPLET_OWNER,
            edge_a=TRIPLET_EDGE_A,
            edge_b=TRIPLET_EDGE_B,
            edge_owner=EDGE_TARGET,
        ),
        LoweringRequest.create(
            "sorted_segment_vjp",
            OperatorKind.SORTED_SEGMENT_VJP,
            segment=EDGE_TARGET,
        ),
    )
