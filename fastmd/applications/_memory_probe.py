"""Moved to fastmd.execution._memory_probe; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.execution._memory_probe'
sys.modules[__name__] = importlib.import_module(_NEW)
