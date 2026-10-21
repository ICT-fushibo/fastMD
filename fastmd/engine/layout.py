"""Certified graph layout used for single-device operator selection.

Layout metadata records pair indexing, line incidence, sorted relations and
CSR structure established by the topology lowering plan. Retained layout
fields have their single-device defaults; exchange layouts are not supported."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json

# Layouts per lowering plan object (plans are cached by topology_lowering).
_LAYOUT_CACHE_SIZE = 64
_LAYOUTS: OrderedDict[int, tuple[object, ExecutionLayout]] = OrderedDict()


# P>1 fields (step 2.4) and their P=1 defaults; a field at its default is left
# out of the fingerprint payload, so P=1 layouts keep their step-2.3 fingerprints.
_P1_DEFAULTS = {"sink": "none", "owner_csr": frozenset(), "shared_value_rows": frozenset(),
                "required_keys": frozenset()}


@dataclass(frozen=True)
class ExecutionLayout:
    route: str = "identity"
    pair_layout: str = "pair_index"
    incidence: str = "strong"
    compiled_envelope: bool = False
    sorted_by: frozenset[str] = frozenset()
    identity_csr: frozenset[str] = frozenset()
    sink: str = "none"
    owner_csr: frozenset[str] = frozenset()
    shared_value_rows: frozenset[str] = frozenset()
    required_keys: frozenset[str] = frozenset()

    @cached_property
    def fingerprint(self) -> str:
        fields = {name: sorted(value) if isinstance(value, frozenset) else value
                  for name, value in asdict(self).items()
                  if name not in _P1_DEFAULTS or value != _P1_DEFAULTS[name]}
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


    def check_required_keys(self, kind: str, graph: dict) -> None:
        """Reject a graph dict that lacks a key a generic lowering will read."""
        missing = sorted(key for key in self.required_keys if key not in graph)
        if missing:
            raise KeyError(f"{kind} graph lacks the keys {missing} required by layout "
                           f"{self.fingerprint} (ExecutionLayout.required_keys)")

    @classmethod
    def from_plan(cls, plan) -> ExecutionLayout:
        """The P=1 layout certified by a ``LoweringPlan`` (cached per plan)."""
        key = id(plan)
        entry = _LAYOUTS.get(key)
        if entry is not None and entry[0] is plan:
            _LAYOUTS.move_to_end(key)
            return entry[1]
        uses = plan.uses
        sorted_by = set()
        if (uses("atom_indexed_affine", "indexed_affine.decomposed_sorted_vjp")
                and uses("sorted_segment_vjp", "segment_vjp.contiguous")):
            sorted_by.add("atom_indexed_affine")
        if uses("line_indexed_affine", "indexed_affine.decomposed_sorted_vjp"):
            sorted_by.add("line_indexed_affine")
        identity_csr = {kind for kind, request in (("atom", "target_segment_attention"),
                                                   ("line", "line_target_segment_attention"))
                        if uses(request, "segment_attention.identity_csr")}
        layout = cls(
            pair_layout="pair_index" if uses("pair_algebra", "pair_algebra.paired_rows") else "generic",
            incidence=("strong" if uses("line_incidence", "line_incidence.direct")
                       else "bounded" if uses("line_incidence", "line_incidence.direct_bounded")
                       else "none"),
            compiled_envelope=uses("line_envelope", "line_incidence.compiled"),
            sorted_by=frozenset(sorted_by),
            identity_csr=frozenset(identity_csr),
        )
        # Keep the plan alive with its entry so that its id cannot be reused.
        _LAYOUTS[key] = (plan, layout)
        if len(_LAYOUTS) > _LAYOUT_CACHE_SIZE:
            _LAYOUTS.popitem(last=False)
        return layout
