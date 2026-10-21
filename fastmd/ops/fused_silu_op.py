"""Moved to fastmd.models.matris.op.fused_silu_op; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.fused_silu_op'
sys.modules[__name__] = importlib.import_module(_NEW)
