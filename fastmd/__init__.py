"""FastMD: single-device simulation with multiple machine-learning interatomic potentials."""
from importlib import import_module

__version__ = "0.1.0"
_EXPORTS = {
    "MatRIS": ("fastmd.models.matris", "MatRIS"),
    "ReplicaMolecularDynamics": ("fastmd.interfaces.replica_md", "ReplicaMolecularDynamics"),
    "MultiSystemMolecularDynamics": ("fastmd.interfaces.multi_system_md", "MultiSystemMolecularDynamics"),
    "MultiSystemRelaxation": ("fastmd.interfaces.multi_system_relax", "MultiSystemRelaxation"),
    "InferenceConfig": ("fastmd.config", "InferenceConfig"),
    "CapacityConfig": ("fastmd.config", "CapacityConfig"),
    "MLIPCalculator": ("fastmd.applications", "MLIPCalculator"),
    "MolecularDynamics": ("fastmd.applications", "MolecularDynamics"),
    "GPUResidentMolecularDynamics": ("fastmd.applications", "GPUResidentMolecularDynamics"),
    "StructOptimizer": ("fastmd.applications", "StructOptimizer"),
    "load_adapter": ("fastmd.adapters", "load_adapter"),
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
