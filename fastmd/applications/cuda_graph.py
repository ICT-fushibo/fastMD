"""Moved to fastmd.execution.cuda_graph; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.execution.cuda_graph'
sys.modules[__name__] = importlib.import_module(_NEW)
