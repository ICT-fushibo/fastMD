"""Moved to fastmd.interfaces.gpu_md; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.interfaces.gpu_md'
sys.modules[__name__] = importlib.import_module(_NEW)
