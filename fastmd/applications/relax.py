"""Moved to fastmd.interfaces.relax; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.interfaces.relax'
sys.modules[__name__] = importlib.import_module(_NEW)
