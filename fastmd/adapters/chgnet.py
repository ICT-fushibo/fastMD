"""Moved to fastmd.models.chgnet; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.chgnet'
sys.modules[__name__] = importlib.import_module(_NEW)
