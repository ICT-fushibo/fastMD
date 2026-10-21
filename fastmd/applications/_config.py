"""Moved to fastmd.engine.entry_config; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.entry_config'
sys.modules[__name__] = importlib.import_module(_NEW)
