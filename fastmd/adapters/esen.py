"""Moved to fastmd.models.esen; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.esen'
sys.modules[__name__] = importlib.import_module(_NEW)
