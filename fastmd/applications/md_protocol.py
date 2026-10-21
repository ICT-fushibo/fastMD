"""Moved to fastmd.execution.md_protocol; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.execution.md_protocol'
sys.modules[__name__] = importlib.import_module(_NEW)
