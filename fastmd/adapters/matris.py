"""Moved to fastmd.models.matris.adapter; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.adapter'
sys.modules[__name__] = importlib.import_module(_NEW)
