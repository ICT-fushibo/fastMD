"""Moved to fastmd.models.matris.converter; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.converter'
sys.modules[__name__] = importlib.import_module(_NEW)
