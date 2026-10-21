"""Moved to fastmd.models.matris.op.compiled_lowerings; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.compiled_lowerings'
sys.modules[__name__] = importlib.import_module(_NEW)
