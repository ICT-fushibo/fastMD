"""Moved to fastmd.models.mace; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.mace'
sys.modules[__name__] = importlib.import_module(_NEW)
