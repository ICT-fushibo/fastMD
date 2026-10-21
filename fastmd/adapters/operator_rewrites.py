"""Moved to fastmd.models.operator_rewrites; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.operator_rewrites'
sys.modules[__name__] = importlib.import_module(_NEW)
