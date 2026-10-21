"""Moved to fastmd.models.esen_ops.triton_esen_nonlinear; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.esen_ops.triton_esen_nonlinear'
sys.modules[__name__] = importlib.import_module(_NEW)
