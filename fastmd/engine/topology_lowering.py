"""Contract-driven lowering selection for sparse forward and VJP operators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time

from fastmd.graph.topology_contract import ContractProperty
from fastmd.graph.topology_contract import ExtentScope
from fastmd.graph.topology_contract import PropertyKind
from fastmd.graph.topology_contract import TopologyCertificate
from fastmd.graph.topology_contract import TopologySchema


class OperatorKind(str, Enum):
    INDEXED_AFFINE = "indexed_affine"
    SEGMENT_ATTENTION = "segment_attention"
    PAIR_ALGEBRA = "pair_algebra"
    LINE_INCIDENCE = "line_incidence"
    SORTED_SEGMENT_VJP = "sorted_segment_vjp"


@dataclass(frozen=True)
class PropertyRequirement:
    kind: PropertyKind
    role: str
    arguments: tuple[str | int | float | bool, ...] = ()
    scope: ExtentScope = ExtentScope.CAPACITY

    def bind(self, bindings: dict[str, str]) -> ContractProperty:
        if self.role not in bindings:
            raise ValueError(f"missing binding for topology role {self.role!r}")
        arguments = tuple(bindings.get(str(argument), argument) for argument in self.arguments)
        return ContractProperty(self.kind, bindings[self.role], arguments, self.scope)


@dataclass(frozen=True)
class OperatorContract:
    kind: OperatorKind
    consumed_roles: tuple[str, ...]
    output_domain: str
    mutation_effects: tuple[str, ...]
    forward_preconditions: tuple[PropertyRequirement, ...]
    vjp_preconditions: tuple[PropertyRequirement, ...]
    fallback: str


@dataclass(frozen=True)
class LoweringSpec:
    name: str
    operator: OperatorKind
    priority: int
    forward_rule: str
    vjp_rule: str
    requirements: tuple[PropertyRequirement, ...] = ()
    constraints: tuple[StaticCostConstraint, ...] = ()


@dataclass(frozen=True)
class StaticCostConstraint:
    """A compile-time guard derived from integrated forward/VJP calibration."""

    parameter: str
    minimum: int | float | None = None
    maximum: int | float | None = None

    def accepts(self, parameters: dict[str, int | float | str]) -> bool:
        value = parameters.get(self.parameter)
        if not isinstance(value, (int, float)):
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        return self.maximum is None or value <= self.maximum


@dataclass(frozen=True)
class LoweringRequest:
    key: str
    operator: OperatorKind
    bindings: tuple[tuple[str, str], ...]
    parameters: tuple[tuple[str, int | float | str], ...] = ()

    @classmethod
    def create(
        cls,
        key: str,
        operator: OperatorKind,
        parameters: dict[str, int | float | str] | None = None,
        **bindings: str,
    ) -> LoweringRequest:
        return cls(
            key,
            operator,
            tuple(sorted(bindings.items())),
            tuple(sorted((parameters or {}).items())),
        )

    def binding_dict(self) -> dict[str, str]:
        return dict(self.bindings)

    def parameter_dict(self) -> dict[str, int | float | str]:
        return dict(self.parameters)


@dataclass(frozen=True)
class LoweringSelection:
    request_key: str
    operator: OperatorKind
    lowering: str
    forward_rule: str
    vjp_rule: str
    required_properties: tuple[ContractProperty, ...]
    fallback: str


@dataclass(frozen=True)
class LoweringPlan:
    schema_fingerprint: str
    certified_properties: tuple[ContractProperty, ...]
    selections: tuple[LoweringSelection, ...]
    compile_ms: float

    def selection(self, request_key: str) -> LoweringSelection:
        for selection in self.selections:
            if selection.request_key == request_key:
                return selection
        raise KeyError(request_key)

    def uses(self, request_key: str, lowering: str) -> bool:
        try:
            return self.selection(request_key).lowering == lowering
        except KeyError:
            return False

    def certifies(
        self,
        kind: PropertyKind,
        subject: str,
        arguments: tuple[str | int | float | bool, ...] = (),
        scope: ExtentScope = ExtentScope.CAPACITY,
    ) -> bool:
        return ContractProperty(kind, subject, arguments, scope) in self.certified_properties


class LoweringRegistry:
    def __init__(self) -> None:
        self._contracts: dict[OperatorKind, OperatorContract] = {}
        self._lowerings: dict[OperatorKind, list[LoweringSpec]] = {}

    def register_contract(self, contract: OperatorContract) -> None:
        if contract.kind in self._contracts:
            raise ValueError(f"operator contract already registered: {contract.kind.value}")
        self._contracts[contract.kind] = contract

    def register_lowering(self, lowering: LoweringSpec) -> None:
        self._lowerings.setdefault(lowering.operator, []).append(lowering)
        self._lowerings[lowering.operator].sort(key=lambda item: item.priority, reverse=True)

    def resolve(
        self,
        request: LoweringRequest,
        certified: frozenset[ContractProperty],
    ) -> LoweringSelection:
        contract = self._contracts[request.operator]
        bindings = request.binding_dict()
        missing_roles = set(contract.consumed_roles) - set(bindings)
        if missing_roles:
            raise ValueError(
                f"request {request.key!r} is missing roles {sorted(missing_roles)}"
            )
        base = tuple(
            requirement.bind(bindings)
            for requirement in contract.forward_preconditions + contract.vjp_preconditions
        )
        parameters = request.parameter_dict()
        for lowering in self._lowerings.get(request.operator, ()):
            if not all(
                constraint.accepts(parameters) for constraint in lowering.constraints
            ):
                continue
            required = tuple(dict.fromkeys(base + tuple(
                requirement.bind(bindings) for requirement in lowering.requirements
            )))
            if all(prop in certified for prop in required):
                return LoweringSelection(
                    request_key=request.key,
                    operator=request.operator,
                    lowering=lowering.name,
                    forward_rule=lowering.forward_rule,
                    vjp_rule=lowering.vjp_rule,
                    required_properties=required,
                    fallback=contract.fallback,
                )
        return LoweringSelection(
            request_key=request.key,
            operator=request.operator,
            lowering=contract.fallback,
            forward_rule=contract.fallback,
            vjp_rule=contract.fallback,
            required_properties=base,
            fallback=contract.fallback,
        )


def _bounded(role: str) -> PropertyRequirement:
    return PropertyRequirement(PropertyKind.BOUNDED, role)


def _default_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    registry.register_contract(
        OperatorContract(
            OperatorKind.INDEXED_AFFINE,
            ("index_a", "index_b"),
            "row",
            (),
            (_bounded("index_a"), _bounded("index_b")),
            (_bounded("index_a"), _bounded("index_b")),
            "indexed_affine.materialized",
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "indexed_affine.decomposed_sorted_vjp",
            OperatorKind.INDEXED_AFFINE,
            30,
            "decomposed_projection_gather_epilogue",
            "contiguous_first_index_vjp",
            (
                PropertyRequirement(
                    PropertyKind.SORTED_BY,
                    "index_a",
                    scope=ExtentScope.ACTIVE,
                ),
            ),
            (StaticCostConstraint("feature_dim", minimum=128, maximum=128),),
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "indexed_affine.decomposed_atomic_vjp",
            OperatorKind.INDEXED_AFFINE,
            20,
            "decomposed_projection_gather_epilogue",
            "atomic_index_vjp",
            constraints=(
                StaticCostConstraint("feature_dim", minimum=128, maximum=128),
            ),
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "indexed_affine.materialized",
            OperatorKind.INDEXED_AFFINE,
            0,
            "gather_cat_linear",
            "generic_autograd",
        )
    )

    registry.register_contract(
        OperatorContract(
            OperatorKind.SEGMENT_ATTENTION,
            ("segment",),
            "segment",
            (),
            (_bounded("segment"),),
            (_bounded("segment"),),
            "segment_attention.generic",
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "segment_attention.identity_csr",
            OperatorKind.SEGMENT_ATTENTION,
            20,
            "identity_csr_attention",
            "identity_csr_attention_vjp",
            (
                PropertyRequirement(
                    PropertyKind.SORTED_BY,
                    "segment",
                    scope=ExtentScope.ACTIVE,
                ),
            ),
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "segment_attention.sorted_csr",
            OperatorKind.SEGMENT_ATTENTION,
            10,
            "sort_then_csr_attention",
            "permuted_csr_attention_vjp",
        )
    )

    registry.register_contract(
        OperatorContract(
            OperatorKind.PAIR_ALGEBRA,
            ("pair_map", "edge_source", "edge_target", "pair_to_edge"),
            "undirected_pair",
            (),
            (_bounded("pair_map"), _bounded("pair_to_edge")),
            (_bounded("pair_map"), _bounded("pair_to_edge")),
            "pair_algebra.scatter",
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "pair_algebra.paired_rows",
            OperatorKind.PAIR_ALGEBRA,
            20,
            "paired_row_reduce_expand",
            "paired_row_reduce_expand_vjp",
            (
                PropertyRequirement(
                    PropertyKind.PAIR_CARDINALITY,
                    "pair_map",
                    (2,),
                    ExtentScope.ACTIVE,
                ),
                PropertyRequirement(
                    PropertyKind.REVERSE_PAIR_INVOLUTION,
                    "pair_map",
                    ("edge_source", "edge_target", "pair_to_edge"),
                    ExtentScope.ACTIVE,
                ),
            ),
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "pair_algebra.scatter",
            OperatorKind.PAIR_ALGEBRA,
            0,
            "scatter_reduce_index_select",
            "generic_autograd",
        )
    )

    registry.register_contract(
        OperatorContract(
            OperatorKind.LINE_INCIDENCE,
            ("owner", "edge_a", "edge_b", "edge_owner"),
            "triplet",
            (),
            (_bounded("owner"), _bounded("edge_a"), _bounded("edge_b")),
            (_bounded("owner"), _bounded("edge_a"), _bounded("edge_b")),
            "line_incidence.gather",
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "line_incidence.compiled",
            OperatorKind.LINE_INCIDENCE,
            25,
            "compiled_incidence_gather_mul",
            "compiled_incidence_scatter_vjp",
            (),
            (StaticCostConstraint("compiled_lowerings", minimum=1),),
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "line_incidence.direct",
            OperatorKind.LINE_INCIDENCE,
            20,
            "direct_incidence_fusion",
            "incidence_scatter_vjp",
            (
                PropertyRequirement(
                    PropertyKind.INCIDENCE,
                    "owner",
                    ("edge_a", "edge_b", "edge_owner"),
                    ExtentScope.ACTIVE,
                ),
            ),
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "line_incidence.direct_bounded",
            OperatorKind.LINE_INCIDENCE,
            10,
            "direct_bounded_incidence_fusion",
            "bounded_incidence_scatter_vjp",
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "line_incidence.gather",
            OperatorKind.LINE_INCIDENCE,
            0,
            "materialized_incidence_gather",
            "generic_autograd",
        )
    )

    registry.register_contract(
        OperatorContract(
            OperatorKind.SORTED_SEGMENT_VJP,
            ("segment",),
            "segment",
            (),
            (_bounded("segment"),),
            (_bounded("segment"),),
            "segment_vjp.atomic",
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "segment_vjp.contiguous",
            OperatorKind.SORTED_SEGMENT_VJP,
            20,
            "segment_forward",
            "contiguous_segment_vjp",
            (
                PropertyRequirement(
                    PropertyKind.SORTED_BY,
                    "segment",
                    scope=ExtentScope.ACTIVE,
                ),
            ),
        )
    )
    registry.register_lowering(
        LoweringSpec(
            "segment_vjp.atomic",
            OperatorKind.SORTED_SEGMENT_VJP,
            0,
            "segment_forward",
            "atomic_segment_vjp",
        )
    )
    return registry


DEFAULT_LOWERING_REGISTRY = _default_registry()
_PLAN_CACHE: dict[tuple, LoweringPlan] = {}


def compile_lowering_plan(
    schema: TopologySchema,
    certificate: TopologyCertificate,
    requests: tuple[LoweringRequest, ...],
    *,
    registry: LoweringRegistry = DEFAULT_LOWERING_REGISTRY,
) -> LoweringPlan:
    """Compile and cache a plan using schema/properties, never model names."""

    if certificate.schema_fingerprint != schema.fingerprint:
        raise ValueError("cannot lower with a certificate for another topology schema")
    key = (
        schema.fingerprint,
        tuple(prop.canonical() for prop in certificate.properties),
        requests,
        id(registry),
    )
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    started = time.perf_counter()
    certified = frozenset(certificate.properties)
    selections = tuple(registry.resolve(request, certified) for request in requests)
    plan = LoweringPlan(
        schema_fingerprint=schema.fingerprint,
        certified_properties=certificate.properties,
        selections=selections,
        compile_ms=(time.perf_counter() - started) * 1.0e3,
    )
    _PLAN_CACHE[key] = plan
    return plan
