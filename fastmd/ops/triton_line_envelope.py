"""Moved to fastmd.models.matris.op.triton_line_envelope; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.triton_line_envelope'
sys.modules[__name__] = importlib.import_module(_NEW)
