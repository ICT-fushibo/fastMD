"""Moved to fastmd.models.matris.op.fuse_sigmoid_op; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.models.matris.op.fuse_sigmoid_op'
sys.modules[__name__] = importlib.import_module(_NEW)
