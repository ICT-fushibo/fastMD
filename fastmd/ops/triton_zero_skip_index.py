"""Moved to fastmd.engine.kernels.triton_zero_skip_index; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.kernels.triton_zero_skip_index'
sys.modules[__name__] = importlib.import_module(_NEW)
