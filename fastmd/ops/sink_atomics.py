"""Moved to fastmd.engine.kernels.sink_atomics; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.kernels.sink_atomics'
sys.modules[__name__] = importlib.import_module(_NEW)
