"""Moved to fastmd.interfaces.calculator; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.interfaces.calculator'
sys.modules[__name__] = importlib.import_module(_NEW)
