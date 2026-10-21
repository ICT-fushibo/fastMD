"""MatRIS's graph topology for the model-independent topology contract IR: the relation names, the schema and the
evidence a graph builder emits for it. The binding of MatRIS's batched graph to the schema is in MatRIS's package
(``fastmd.models.matris.topology``)."""

from __future__ import annotations

from fastmd.graph.topology_contract import (
    BuilderEvidence,
    ContractProperty,
    DescriptorLifetime,
    DomainSpec,
    ExtentPolicy,
    ExtentScope,
    PropertyKind,
    RelationSpec,
    TopologyDomain,
    TopologySchema,
)


ATOM = "atom"
DIRECTED_EDGE = "directed_edge"
UNDIRECTED_PAIR = "undirected_pair"
TRIPLET = "triplet"
GRAPH = "graph"
DUMMY = "dummy"

EDGE_TARGET = "edge_target"
EDGE_SOURCE = "edge_source"
DIRECTED_TO_PAIR = "directed_to_pair"
PAIR_TO_DIRECTED = "pair_to_directed"
ATOM_GRAPH_ID = "atom_graph_id"
TRIPLET_OWNER = "triplet_owner"
TRIPLET_PAIR_A = "triplet_pair_a"
TRIPLET_PAIR_B = "triplet_pair_b"
TRIPLET_EDGE_A = "triplet_edge_a"
TRIPLET_EDGE_B = "triplet_edge_b"


_BASE_RELATIONS = (
    RelationSpec(EDGE_TARGET, DIRECTED_EDGE, ATOM),
    RelationSpec(EDGE_SOURCE, DIRECTED_EDGE, ATOM),
    RelationSpec(DIRECTED_TO_PAIR, DIRECTED_EDGE, UNDIRECTED_PAIR),
    RelationSpec(PAIR_TO_DIRECTED, UNDIRECTED_PAIR, DIRECTED_EDGE),
    RelationSpec(
        ATOM_GRAPH_ID,
        ATOM,
        GRAPH,
        lifetime=DescriptorLifetime.CAPTURE,
        invalidated_by=("batch_changed",),
    ),
    RelationSpec(TRIPLET_OWNER, TRIPLET, ATOM),
    RelationSpec(TRIPLET_PAIR_A, TRIPLET, UNDIRECTED_PAIR),
    RelationSpec(TRIPLET_PAIR_B, TRIPLET, UNDIRECTED_PAIR),
    RelationSpec(TRIPLET_EDGE_A, TRIPLET, DIRECTED_EDGE),
    RelationSpec(TRIPLET_EDGE_B, TRIPLET, DIRECTED_EDGE),
)


def _matris_properties(
    *,
    atom_target_sorted: bool,
    line_owner_sorted: bool,
    fixed_capacity: bool = False,
    real_dummy_closed: bool = False,
    incidence_consistent: bool = True,
) -> tuple[ContractProperty, ...]:
    properties: list[ContractProperty] = [
        ContractProperty(PropertyKind.BOUNDED, relation.name)
        for relation in _BASE_RELATIONS
    ]
    properties.extend(
        [
            ContractProperty(PropertyKind.GRAPH_CONTIGUOUS, ATOM_GRAPH_ID),
            ContractProperty(
                PropertyKind.PAIR_CARDINALITY,
                DIRECTED_TO_PAIR,
                (2,),
                ExtentScope.ACTIVE,
            ),
            ContractProperty(
                PropertyKind.REVERSE_PAIR_INVOLUTION,
                DIRECTED_TO_PAIR,
                (EDGE_SOURCE, EDGE_TARGET, PAIR_TO_DIRECTED),
                ExtentScope.ACTIVE,
            ),
        ]
    )
    if incidence_consistent:
        properties.append(
            ContractProperty(
                PropertyKind.INCIDENCE,
                TRIPLET_OWNER,
                (TRIPLET_EDGE_A, TRIPLET_EDGE_B, EDGE_TARGET),
                ExtentScope.ACTIVE,
            )
        )
    if atom_target_sorted:
        properties.append(
            ContractProperty(
                PropertyKind.SORTED_BY,
                EDGE_TARGET,
                scope=ExtentScope.ACTIVE,
            )
        )
    if line_owner_sorted:
        properties.append(
            ContractProperty(
                PropertyKind.SORTED_BY,
                TRIPLET_OWNER,
                scope=ExtentScope.ACTIVE,
            )
        )
    if fixed_capacity:
        properties.extend(
            ContractProperty(PropertyKind.FIXED_CAPACITY, domain)
            for domain in (ATOM, DIRECTED_EDGE, UNDIRECTED_PAIR, TRIPLET)
        )
    if real_dummy_closed:
        properties.extend(
            ContractProperty(PropertyKind.REAL_DUMMY_CLOSED, relation.name)
            for relation in _BASE_RELATIONS
        )
    return tuple(properties)


def matris_builder_evidence(
    issuer: str,
    *,
    atom_target_sorted: bool,
    line_owner_sorted: bool,
    fixed_capacity: bool = False,
    real_dummy_closed: bool = False,
    incidence_consistent: bool = True,
) -> BuilderEvidence:
    """Create static evidence emitted by a MatRIS graph builder."""

    return BuilderEvidence(
        issuer=issuer,
        properties=_matris_properties(
            atom_target_sorted=atom_target_sorted,
            line_owner_sorted=line_owner_sorted,
            fixed_capacity=fixed_capacity,
            real_dummy_closed=real_dummy_closed,
            incidence_consistent=incidence_consistent,
        ),
    )


def matris_schema(
    *,
    atom_target_sorted: bool,
    line_owner_sorted: bool,
    fixed_capacity: bool = False,
    real_dummy_closed: bool = False,
    incidence_consistent: bool = True,
) -> TopologySchema:
    policy = ExtentPolicy.FIXED_CAPACITY if fixed_capacity else ExtentPolicy.DYNAMIC
    domains = (
        DomainSpec(ATOM, TopologyDomain.ATOM, policy, has_real_prefix=fixed_capacity),
        DomainSpec(
            DIRECTED_EDGE,
            TopologyDomain.DIRECTED_EDGE,
            policy,
            has_real_prefix=fixed_capacity,
        ),
        DomainSpec(
            UNDIRECTED_PAIR,
            TopologyDomain.UNDIRECTED_PAIR,
            policy,
            has_real_prefix=fixed_capacity,
        ),
        DomainSpec(TRIPLET, TopologyDomain.TRIPLET, policy, has_real_prefix=fixed_capacity),
        DomainSpec(GRAPH, TopologyDomain.GRAPH, ExtentPolicy.DYNAMIC),
        DomainSpec(DUMMY, TopologyDomain.DUMMY, policy, has_real_prefix=fixed_capacity),
    )
    return TopologySchema(
        domains=domains,
        relations=_BASE_RELATIONS,
        properties=_matris_properties(
            atom_target_sorted=atom_target_sorted,
            line_owner_sorted=line_owner_sorted,
            fixed_capacity=fixed_capacity,
            real_dummy_closed=real_dummy_closed,
            incidence_consistent=incidence_consistent,
        ),
    )
