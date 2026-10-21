"""Moved to fastmd.engine.capacity_policy; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.capacity_policy'
sys.modules[__name__] = importlib.import_module(_NEW)
