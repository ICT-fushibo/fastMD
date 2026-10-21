"""Compatibility alias for an engine kernel."""
import importlib, sys
sys.modules[__name__] = importlib.import_module('fastmd.engine.kernels.triton_segment_softmax')
