"""Moved to fastmd.engine.kernels.triton_row_gather_scatter; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.kernels.triton_row_gather_scatter'
sys.modules[__name__] = importlib.import_module(_NEW)
