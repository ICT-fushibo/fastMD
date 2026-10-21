"""Moved to fastmd.models.matris.op.triton_weighted_segment_sum; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.triton_weighted_segment_sum'
sys.modules[__name__] = importlib.import_module(_NEW)
