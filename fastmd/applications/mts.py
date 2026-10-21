"""Moved to fastmd.interfaces.mts; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.interfaces.mts'
sys.modules[__name__] = importlib.import_module(_NEW)
