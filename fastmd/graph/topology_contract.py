"""Versioned topology contracts for differentiable sparse operators.

The schema is tensor-free and hashable, while an instance binds the schema to
the index tensors for one graph shape.  Verification is deliberately separate
from lowering selection: production builders can issue a proof certificate
without synchronizing the device, and tests can exhaustively check the same
claims before a lowering plan is cached.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import time
from typing import Mapping

import torch
from torch import Tensor


IR_VERSION = "1.0"


class TopologyDomain(str, Enum):
    ATOM = "Atom"
    DIRECTED_EDGE = "DirectedEdge"
    UNDIRECTED_PAIR = "UndirectedPair"
    TRIPLET = "Triplet"
    GRAPH = "Graph"
    DUMMY = "Dummy"


class ExtentPolicy(str, Enum):
    DYNAMIC = "dynamic"
    FIXED_CAPACITY = "fixed_capacity"


class ExtentScope(str, Enum):
    REAL = "real"
    ACTIVE = "active"
    CAPACITY = "capacity"


class DescriptorLifetime(str, Enum):
    MODEL = "model"
    CAPTURE = "capture"
    CANDIDATE_LIST = "candidate_list"
    STEP = "step"


class PropertyKind(str, Enum):
    BOUNDED = "Bounded"
    SORTED_BY = "SortedBy"
    UNIQUE = "Unique"
    PAIR_CARDINALITY = "PairCardinality"
    REVERSE_PAIR_INVOLUTION = "ReversePairInvolution"
    INCIDENCE = "Incidence"
    REAL_DUMMY_CLOSED = "RealDummyClosed"
    GRAPH_CONTIGUOUS = "GraphContiguous"
    FIXED_CAPACITY = "FixedCapacity"


class VerificationMode(str, Enum):
    BUILDER = "builder"
    SAMPLED = "sampled"
    EXHAUSTIVE = "exhaustive"


ContractArgument = str | int | float | bool


@dataclass(frozen=True)
class ContractProperty:
    """A property claim over a domain or relation.

    ``arguments`` contains relation names or scalar parameters.  Keeping this
    representation declarative lets operator requirements bind symbolic roles
    without importing a model implementation.
    """

    kind: PropertyKind
    subject: str
    arguments: tuple[ContractArgument, ...] = ()
    scope: ExtentScope = ExtentScope.CAPACITY

    def canonical(self) -> tuple[str, str, tuple[ContractArgument, ...], str]:
        return (self.kind.value, self.subject, self.arguments, self.scope.value)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    kind: TopologyDomain
    extent_policy: ExtentPolicy = ExtentPolicy.DYNAMIC
    lifetime: DescriptorLifetime = DescriptorLifetime.STEP
    has_real_prefix: bool = False


@dataclass(frozen=True)
class RelationSpec:
    name: str
    row_domain: str
    value_domain: str
    lifetime: DescriptorLifetime = DescriptorLifetime.STEP
    invalidated_by: tuple[str, ...] = ("topology_changed",)
    restored_on_rollback: bool = True


@dataclass(frozen=True)
class TopologySchema:
    """Tensor-free semantic schema used as the compiler cache key."""

    domains: tuple[DomainSpec, ...]
    relations: tuple[RelationSpec, ...]
    properties: tuple[ContractProperty, ...]
    version: str = IR_VERSION

    def __post_init__(self) -> None:
        domain_names = [domain.name for domain in self.domains]
        relation_names = [relation.name for relation in self.relations]
        if len(domain_names) != len(set(domain_names)):
            raise ValueError("topology domain names must be unique")
        if len(relation_names) != len(set(relation_names)):
            raise ValueError("topology relation names must be unique")
        known_domains = set(domain_names)
        for relation in self.relations:
            if relation.row_domain not in known_domains:
                raise ValueError(
                    f"relation {relation.name!r} has unknown row domain "
                    f"{relation.row_domain!r}"
                )
            if relation.value_domain not in known_domains:
                raise ValueError(
                    f"relation {relation.name!r} has unknown value domain "
                    f"{relation.value_domain!r}"
                )
        known_subjects = known_domains | set(relation_names)
        for prop in self.properties:
            if prop.subject not in known_subjects:
                raise ValueError(
                    f"property {prop.kind.value} has unknown subject {prop.subject!r}"
                )

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": self.version,
            "domains": [
                {
                    "name": item.name,
                    "kind": item.kind.value,
                    "extent_policy": item.extent_policy.value,
                    "lifetime": item.lifetime.value,
                    "has_real_prefix": item.has_real_prefix,
                }
                for item in self.domains
            ],
            "relations": [
                {
                    "name": item.name,
                    "row_domain": item.row_domain,
                    "value_domain": item.value_domain,
                    "lifetime": item.lifetime.value,
                    "invalidated_by": item.invalidated_by,
                    "restored_on_rollback": item.restored_on_rollback,
                }
                for item in self.relations
            ],
            "properties": [
                item.canonical()
                for item in sorted(
                    self.properties,
                    key=lambda prop: json.dumps(prop.canonical(), default=str),
                )
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:20]

    def relation(self, name: str) -> RelationSpec:
        for relation in self.relations:
            if relation.name == name:
                return relation
        raise KeyError(name)

    def has_property(self, prop: ContractProperty) -> bool:
        return prop in self.properties


@dataclass(frozen=True)
class DomainExtent:
    """Real, active, and allocated sizes for one domain."""

    real: int
    active: int
    capacity: int

    def __post_init__(self) -> None:
        if not (0 <= self.real <= self.active <= self.capacity):
            raise ValueError(
                "domain extent must satisfy 0 <= real <= active <= capacity, got "
                f"{(self.real, self.active, self.capacity)}"
            )

    def size(self, scope: ExtentScope) -> int:
        return int(getattr(self, scope.value))


@dataclass(frozen=True)
class BuilderEvidence:
    """Static proof claims emitted by a graph construction algorithm."""

    issuer: str
    properties: tuple[ContractProperty, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "issuer": self.issuer,
            "properties": [
                {
                    "kind": prop.kind.value,
                    "subject": prop.subject,
                    "arguments": list(prop.arguments),
                    "scope": prop.scope.value,
                }
                for prop in self.properties
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BuilderEvidence:
        properties = value.get("properties", ())
        if not isinstance(properties, (list, tuple)):
            raise TypeError("builder evidence properties must be a sequence")
        if not all(isinstance(prop, Mapping) for prop in properties):
            raise TypeError("every builder evidence property must be a mapping")
        return cls(
            issuer=str(value["issuer"]),
            properties=tuple(
                ContractProperty(
                    PropertyKind(str(prop["kind"])),
                    str(prop["subject"]),
                    tuple(prop.get("arguments", ())),
                    ExtentScope(str(prop.get("scope", ExtentScope.CAPACITY.value))),
                )
                for prop in properties
            ),
        )


@dataclass(frozen=True)
class TopologyCertificate:
    schema_fingerprint: str
    extent_signature: tuple[tuple[str, int, int, int], ...]
    properties: tuple[ContractProperty, ...]
    issuer: str
    mode: VerificationMode
    sample_size: int | None = None

    def certifies(self, prop: ContractProperty) -> bool:
        return prop in self.properties


@dataclass(frozen=True)
class TopologyInstance:
    """Runtime binding from a schema to extents and index tensors."""

    schema: TopologySchema
    extents: Mapping[str, DomainExtent]
    relations: Mapping[str, Tensor]
    producer: str
    certificate: TopologyCertificate | None = None

    @property
    def extent_signature(self) -> tuple[tuple[str, int, int, int], ...]:
        return tuple(
            (name, extent.real, extent.active, extent.capacity)
            for name, extent in sorted(self.extents.items())
        )

    def relation(self, name: str) -> Tensor:
        return self.relations[name]

    def with_certificate(self, certificate: TopologyCertificate) -> TopologyInstance:
        if certificate.schema_fingerprint != self.schema.fingerprint:
            raise ValueError("certificate does not match topology schema")
        if certificate.extent_signature != self.extent_signature:
            raise ValueError("certificate does not match topology extents")
        return replace(self, certificate=certificate)


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class VerificationReport:
    mode: VerificationMode
    checks: tuple[VerificationCheck, ...]
    elapsed_ms: float
    certificate: TopologyCertificate | None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class TopologyVerificationError(ValueError):
    def __init__(self, report: VerificationReport):
        failed = [check.name for check in report.checks if not check.passed]
        super().__init__("topology contract verification failed: " + ", ".join(failed))
        self.report = report


def _sample_positions(length: int, mode: VerificationMode, sample_size: int) -> Tensor | None:
    if mode == VerificationMode.EXHAUSTIVE or length <= sample_size:
        return None
    if length <= 0:
        return torch.empty(0, dtype=torch.long)
    return torch.linspace(0, length - 1, sample_size, dtype=torch.long)


def _sample_rows(tensor: Tensor, mode: VerificationMode, sample_size: int) -> Tensor:
    positions = _sample_positions(tensor.shape[0], mode, sample_size)
    if positions is None:
        return tensor
    return tensor.index_select(0, positions.to(tensor.device))


def _as_bool(value: Tensor | bool) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value.all().item())


def _structural_checks(instance: TopologyInstance) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    domain_names = {domain.name for domain in instance.schema.domains}
    checks.append(
        VerificationCheck(
            "domain bindings",
            set(instance.extents) == domain_names,
            "every schema domain must have one extent",
        )
    )
    relation_names = {relation.name for relation in instance.schema.relations}
    checks.append(
        VerificationCheck(
            "relation bindings",
            set(instance.relations) == relation_names,
            "every schema relation must have one tensor",
        )
    )
    if not checks[-2].passed or not checks[-1].passed:
        return checks

    for relation in instance.schema.relations:
        tensor = instance.relations[relation.name]
        expected_rows = instance.extents[relation.row_domain].capacity
        checks.append(
            VerificationCheck(
                f"{relation.name}: integral 1D relation",
                tensor.dim() == 1
                and tensor.dtype
                in {
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                },
            )
        )
        checks.append(
            VerificationCheck(
                f"{relation.name}: row extent",
                tensor.dim() >= 1 and tensor.shape[0] == expected_rows,
                f"expected {expected_rows} rows",
            )
        )
    return checks


def _verify_property(
    instance: TopologyInstance,
    prop: ContractProperty,
    mode: VerificationMode,
    sample_size: int,
) -> VerificationCheck:
    relations = instance.relations
    schema = instance.schema
    name = f"{prop.kind.value}({prop.subject})"

    if prop.kind == PropertyKind.BOUNDED:
        relation = schema.relation(prop.subject)
        values = _sample_rows(relations[prop.subject], mode, sample_size)
        upper = instance.extents[relation.value_domain].size(prop.scope)
        passed = values.numel() == 0 or _as_bool((values >= 0) & (values < upper))
        return VerificationCheck(name, passed, f"range=[0,{upper})")

    if prop.kind in {PropertyKind.SORTED_BY, PropertyKind.GRAPH_CONTIGUOUS}:
        values = relations[prop.subject]
        limit = instance.extents[schema.relation(prop.subject).row_domain].size(prop.scope)
        values = values[:limit]
        if values.numel() < 2:
            return VerificationCheck(name, True)
        if mode == VerificationMode.EXHAUSTIVE or values.numel() - 1 <= sample_size:
            passed = _as_bool(values[1:] >= values[:-1])
        else:
            positions = torch.linspace(
                0, values.numel() - 2, sample_size, dtype=torch.long, device=values.device
            )
            passed = _as_bool(values[positions + 1] >= values[positions])
        return VerificationCheck(name, passed)

    if prop.kind == PropertyKind.UNIQUE:
        values = relations[prop.subject]
        limit = instance.extents[schema.relation(prop.subject).row_domain].size(prop.scope)
        values = values[:limit]
        passed = values.numel() == torch.unique(values).numel()
        return VerificationCheck(name, bool(passed))

    if prop.kind == PropertyKind.PAIR_CARDINALITY:
        cardinality = int(prop.arguments[0])
        relation_spec = schema.relation(prop.subject)
        row_count = instance.extents[relation_spec.row_domain].size(prop.scope)
        pair_count = instance.extents[relation_spec.value_domain].size(prop.scope)
        values = relations[prop.subject][:row_count].long()
        values_in_bounds = values.numel() == 0 or _as_bool(
            (values >= 0) & (values < pair_count)
        )
        if not values_in_bounds:
            passed = False
        elif values.numel() == 0:
            passed = pair_count == 0
        else:
            counts = torch.bincount(values, minlength=pair_count)[:pair_count]
            counts = _sample_rows(counts, mode, sample_size)
            passed = _as_bool(counts == cardinality)
        return VerificationCheck(name, passed, f"cardinality={cardinality}")

    if prop.kind == PropertyKind.REVERSE_PAIR_INVOLUTION:
        edge_source_name, edge_target_name, pair_to_edge_name = (
            str(value) for value in prop.arguments
        )
        pair_map_spec = schema.relation(prop.subject)
        edge_count = instance.extents[pair_map_spec.row_domain].size(prop.scope)
        pair_count = instance.extents[pair_map_spec.value_domain].size(prop.scope)
        pair_map = relations[prop.subject][:edge_count].long()
        pair_to_edge = relations[pair_to_edge_name][:pair_count].long()
        if edge_count != 2 * pair_count:
            return VerificationCheck(name, False, "edge extent is not twice pair extent")
        if pair_map.numel() and not _as_bool(
            (pair_map >= 0) & (pair_map < pair_count)
        ):
            return VerificationCheck(name, False, "pair map is out of bounds")
        if pair_to_edge.numel() and not _as_bool(
            (pair_to_edge >= 0) & (pair_to_edge < edge_count)
        ):
            return VerificationCheck(name, False, "pair representative is out of bounds")
        pair_rows = torch.argsort(pair_map).reshape(pair_count, 2)
        positions = _sample_positions(pair_count, mode, sample_size)
        if positions is not None:
            pair_rows = pair_rows.index_select(0, positions.to(pair_rows.device))
            pair_ids = positions.to(pair_map.device)
            representatives = pair_to_edge.index_select(0, positions.to(pair_to_edge.device))
        else:
            pair_ids = torch.arange(pair_count, device=pair_map.device)
            representatives = pair_to_edge
        source = relations[edge_source_name]
        target = relations[edge_target_name]
        first, second = pair_rows[:, 0], pair_rows[:, 1]
        passed = _as_bool(
            (source[first] == target[second])
            & (target[first] == source[second])
            & (pair_map[representatives] == pair_ids)
        )
        return VerificationCheck(name, passed)

    if prop.kind == PropertyKind.INCIDENCE:
        edge_a_name, edge_b_name, edge_owner_name = (
            str(value) for value in prop.arguments
        )
        owner = relations[prop.subject]
        limit = instance.extents[schema.relation(prop.subject).row_domain].size(prop.scope)
        positions = _sample_positions(limit, mode, sample_size)
        if positions is None:
            positions = torch.arange(limit, device=owner.device)
        else:
            positions = positions.to(owner.device)
        edge_a = relations[edge_a_name].index_select(0, positions).long()
        edge_b = relations[edge_b_name].index_select(0, positions).long()
        expected = owner.index_select(0, positions)
        edge_owner = relations[edge_owner_name]
        if edge_a.numel() and not _as_bool(
            (edge_a >= 0)
            & (edge_a < edge_owner.shape[0])
            & (edge_b >= 0)
            & (edge_b < edge_owner.shape[0])
        ):
            return VerificationCheck(name, False, "incidence reference is out of bounds")
        passed = _as_bool(
            (edge_owner[edge_a] == expected) & (edge_owner[edge_b] == expected)
        )
        return VerificationCheck(name, passed)

    if prop.kind == PropertyKind.REAL_DUMMY_CLOSED:
        relation_spec = schema.relation(prop.subject)
        values = relations[prop.subject]
        row_extent = instance.extents[relation_spec.row_domain]
        value_extent = instance.extents[relation_spec.value_domain]
        real_values = _sample_rows(values[: row_extent.real], mode, sample_size)
        dummy_values = _sample_rows(values[row_extent.real :], mode, sample_size)
        real_ok = real_values.numel() == 0 or _as_bool(real_values < value_extent.real)
        dummy_ok = dummy_values.numel() == 0 or _as_bool(dummy_values >= value_extent.real)
        return VerificationCheck(name, real_ok and dummy_ok)

    if prop.kind == PropertyKind.FIXED_CAPACITY:
        extent = instance.extents[prop.subject]
        domain = next(domain for domain in schema.domains if domain.name == prop.subject)
        passed = domain.extent_policy == ExtentPolicy.FIXED_CAPACITY and extent.capacity >= extent.active
        return VerificationCheck(name, passed)

    return VerificationCheck(name, False, "unsupported property kind")


def certify_topology(
    instance: TopologyInstance,
    *,
    evidence: BuilderEvidence | None = None,
    mode: VerificationMode | str = VerificationMode.BUILDER,
    sample_size: int = 4096,
) -> tuple[TopologyInstance, VerificationReport]:
    """Validate claims and bind a certificate to ``instance``.

    Builder mode checks only tensor-free structure and evidence compatibility;
    it never reads a CUDA value.  Sampled and exhaustive modes evaluate every
    claimed property and are intended for setup validation and tests.
    """

    mode = VerificationMode(mode)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    started = time.perf_counter()
    checks = _structural_checks(instance)
    claims = instance.schema.properties
    issuer = instance.producer
    if mode == VerificationMode.BUILDER and evidence is None:
        checks.append(
            VerificationCheck(
                "builder evidence supplied",
                False,
                "builder mode cannot certify tensor values without producer evidence",
            )
        )
    if evidence is not None:
        issuer = evidence.issuer
        undeclared = set(evidence.properties) - set(claims)
        missing = set(claims) - set(evidence.properties)
        checks.append(
            VerificationCheck(
                "builder evidence matches schema",
                not undeclared and not missing,
                f"undeclared={len(undeclared)} missing={len(missing)}",
            )
        )

    if all(check.passed for check in checks):
        if mode == VerificationMode.BUILDER:
            checks.extend(
                VerificationCheck(
                    f"builder proof: {prop.kind.value}({prop.subject})", True
                )
                for prop in claims
            )
        else:
            checks.extend(
                _verify_property(instance, prop, mode, sample_size) for prop in claims
            )

    certificate = None
    if all(check.passed for check in checks):
        certificate = TopologyCertificate(
            schema_fingerprint=instance.schema.fingerprint,
            extent_signature=instance.extent_signature,
            properties=claims,
            issuer=issuer,
            mode=mode,
            sample_size=sample_size if mode == VerificationMode.SAMPLED else None,
        )
    report = VerificationReport(
        mode=mode,
        checks=tuple(checks),
        elapsed_ms=(time.perf_counter() - started) * 1.0e3,
        certificate=certificate,
    )
    if not report.passed or certificate is None:
        raise TopologyVerificationError(report)
    return instance.with_certificate(certificate), report
