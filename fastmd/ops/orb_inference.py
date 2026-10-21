"""Moved to fastmd.models.orb_ops.orb_inference; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.orb_ops.orb_inference'
sys.modules[__name__] = importlib.import_module(_NEW)
