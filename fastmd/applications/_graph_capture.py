"""Moved to fastmd.engine._graph_capture; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine._graph_capture'
sys.modules[__name__] = importlib.import_module(_NEW)
