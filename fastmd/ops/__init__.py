"""Compatibility imports for kernels moved to the engine or their model package."""
from fastmd._lazy import lazy_attributes
_NAMES = {'fused_silu': 'fastmd.models.matris.op.fused_silu_op', 'fused_sigmoid': 'fastmd.models.matris.op.fuse_sigmoid_op'}
__all__ = list(_NAMES)
__getattr__, __dir__ = lazy_attributes(__name__, _NAMES)
