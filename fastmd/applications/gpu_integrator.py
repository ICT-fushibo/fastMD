"""Moved to fastmd.execution.gpu_integrator; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.execution.gpu_integrator'
sys.modules[__name__] = importlib.import_module(_NEW)
