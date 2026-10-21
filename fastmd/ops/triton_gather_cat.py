"""Moved to fastmd.models.matris.op.triton_gather_cat; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.triton_gather_cat'
sys.modules[__name__] = importlib.import_module(_NEW)
