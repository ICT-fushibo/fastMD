"""Moved to fastmd.models.orb; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.orb'
sys.modules[__name__] = importlib.import_module(_NEW)
