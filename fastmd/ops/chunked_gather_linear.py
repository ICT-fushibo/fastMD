"""Moved to fastmd.models.matris.op.chunked_gather_linear; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.chunked_gather_linear'
sys.modules[__name__] = importlib.import_module(_NEW)
