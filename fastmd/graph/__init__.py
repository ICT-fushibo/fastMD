"""Graph construction: the radius graph, its builders and the topology contract. The names below, and a module
accessed as an attribute of the package, are imported on first access: importing the package, or one of its modules,
imports what that module needs (the GPU builders import Triton and the neighbour-list library)."""
from fastmd import _lazy

# The names this package re-exports, per module.
_EXPORTS = {
    # MatRIS's graph converter moved to MatRIS's model package: its old path, kept for compatibility.
    "fastmd.models.matris.converter": ("GraphConverter",),
    "fastmd.graph.fixed_capacity_builder": (
        "BatchedCandidateRefresher",
        "FixedCapacityGraphBuilder",
        "FixedCapacityGraphStatus",
        "StaticizationInvariantReport",
        "validate_staticization_invariants",
    ),
    "fastmd.graph.gpu_graph_builder": ("TensorGraphBuilder", "atoms_to_graph_gpu", "tensors_to_graph_gpu"),
    "fastmd.graph.radiusgraph": ("RadiusGraph", "datatype"),
    "fastmd.graph.validation": ("raise_if_graph_has_isolated_atoms", "raise_if_isolated_atoms"),
    "fastmd.graph.topology_contract": (
        "BuilderEvidence",
        "ContractProperty",
        "DomainExtent",
        "DomainSpec",
        "PropertyKind",
        "RelationSpec",
        "TopologyCertificate",
        "TopologyDomain",
        "TopologyInstance",
        "TopologySchema",
        "TopologyVerificationError",
        "VerificationMode",
        "certify_topology",
    ),
}
__all__ = sorted(name for names in _EXPORTS.values() for name in names)
__getattr__, __dir__ = _lazy.lazy_attributes(__name__, {name: module for module, names in _EXPORTS.items()
                                                         for name in names})
