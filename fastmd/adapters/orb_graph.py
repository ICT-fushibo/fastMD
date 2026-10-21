"""Moved to fastmd.models.orb_graph; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.orb_graph'
sys.modules[__name__] = importlib.import_module(_NEW)
