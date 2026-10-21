"""Moved to fastmd.interfaces.torchsim_interface; compatibility alias of the same module."""
import importlib
import sys

_NEW = 'fastmd.interfaces.torchsim_interface'
sys.modules[__name__] = importlib.import_module(_NEW)
