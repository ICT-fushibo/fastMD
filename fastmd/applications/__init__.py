"""Compatibility exports for the public simulation interfaces and execution components."""
from fastmd._lazy import lazy_attributes

_EXPORTS = {
    'MLIPCalculator': 'fastmd.interfaces.calculator',
    'GPUResidentMolecularDynamics': 'fastmd.interfaces.gpu_md',
    'MolecularDynamics': 'fastmd.interfaces.md',
    'StructOptimizer': 'fastmd.interfaces.relax',
    'GPUNoseHooverChainNVT': 'fastmd.execution.gpu_nhc',
    'maxwell_boltzmann_momenta': 'fastmd.execution.gpu_nhc',
    'HeatingRamp': 'fastmd.execution.md_protocol',
    'BatchedWholeStepGraphRunner': 'fastmd.execution.whole_step_cuda_graph',
    'WholeStepGraphRunner': 'fastmd.execution.whole_step_cuda_graph',
    'ReplicaMolecularDynamics': 'fastmd.interfaces.replica_md',
    'MultiSystemMolecularDynamics': 'fastmd.interfaces.multi_system_md',
    'MultiSystemRelaxation': 'fastmd.interfaces.multi_system_relax',
}
__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_attributes(__name__, _EXPORTS)
