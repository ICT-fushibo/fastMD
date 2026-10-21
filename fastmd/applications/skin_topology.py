"""Moved to fastmd.execution.skin_topology; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.execution.skin_topology'
sys.modules[__name__] = importlib.import_module(_NEW)
