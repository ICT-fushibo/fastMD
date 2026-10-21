"""Moved to fastmd.models.orb_ops.orb_energy; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.orb_ops.orb_energy'
sys.modules[__name__] = importlib.import_module(_NEW)
