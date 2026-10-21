"""Moved to fastmd.engine.topology_lowering; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.topology_lowering'
sys.modules[__name__] = importlib.import_module(_NEW)
