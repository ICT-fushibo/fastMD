"""Moved to fastmd.engine.kernels.row_tiles; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.kernels.row_tiles'
sys.modules[__name__] = importlib.import_module(_NEW)
