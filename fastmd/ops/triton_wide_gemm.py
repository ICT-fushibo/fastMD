"""Moved to fastmd.engine.kernels.triton_wide_gemm; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.engine.kernels.triton_wide_gemm'
sys.modules[__name__] = importlib.import_module(_NEW)
