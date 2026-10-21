"""Moved to fastmd.engine.kernels.row_count; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.kernels.row_count'
sys.modules[__name__] = importlib.import_module(_NEW)
