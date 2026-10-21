"""Moved to fastmd.execution.whole_step_cuda_graph; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.execution.whole_step_cuda_graph'
sys.modules[__name__] = importlib.import_module(_NEW)
