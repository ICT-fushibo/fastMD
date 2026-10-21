"""Moved to fastmd.engine.kernels.merged_frozen_projections; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.kernels.merged_frozen_projections'
sys.modules[__name__] = importlib.import_module(_NEW)
