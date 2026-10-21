"""Moved to fastmd.models.matris.op.segment_partials; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.segment_partials'
sys.modules[__name__] = importlib.import_module(_NEW)
