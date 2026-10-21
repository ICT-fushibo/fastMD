"""Moved to fastmd.models.m3gnet; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.m3gnet'
sys.modules[__name__] = importlib.import_module(_NEW)
