"""Moved to fastmd.interfaces.md; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.interfaces.md'
sys.modules[__name__] = importlib.import_module(_NEW)
