"""Moved to fastmd.models.matris.op.triton_edge_ffn; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.triton_edge_ffn'
sys.modules[__name__] = importlib.import_module(_NEW)
