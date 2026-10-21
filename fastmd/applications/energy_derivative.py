"""Moved to fastmd.engine.energy_derivative; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.energy_derivative'
sys.modules[__name__] = importlib.import_module(_NEW)
