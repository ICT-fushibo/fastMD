"""Moved to fastmd.models.matris.op.triton_directed_pair_aggregate; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.triton_directed_pair_aggregate'
sys.modules[__name__] = importlib.import_module(_NEW)
