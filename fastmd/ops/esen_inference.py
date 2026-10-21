"""Moved to fastmd.models.esen_ops.esen_inference; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.esen_ops.esen_inference'
sys.modules[__name__] = importlib.import_module(_NEW)
