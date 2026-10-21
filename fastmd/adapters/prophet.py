"""Moved to fastmd.models.prophet; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.prophet'
sys.modules[__name__] = importlib.import_module(_NEW)
