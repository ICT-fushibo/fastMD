"""Moved to fastmd.engine.kernels.triton_segment_softmax; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.kernels.triton_segment_softmax'
sys.modules[__name__] = importlib.import_module(_NEW)
