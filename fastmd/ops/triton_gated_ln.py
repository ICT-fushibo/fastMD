"""Moved to fastmd.models.matris.op.triton_gated_ln; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.triton_gated_ln'
sys.modules[__name__] = importlib.import_module(_NEW)
