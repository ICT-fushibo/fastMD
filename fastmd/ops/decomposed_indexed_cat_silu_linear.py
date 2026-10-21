"""Moved to fastmd.engine.kernels.decomposed_indexed_cat_silu_linear; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.kernels.decomposed_indexed_cat_silu_linear'
sys.modules[__name__] = importlib.import_module(_NEW)
