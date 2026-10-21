"""Moved to fastmd.models.sevennet; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.sevennet'
sys.modules[__name__] = importlib.import_module(_NEW)
