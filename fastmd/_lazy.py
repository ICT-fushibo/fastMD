"""Lazy package attributes (PEP 562): a package re-exporting names imports each one from its module on first access,
so importing the package, or one of its modules, does not import every module the package re-exports from."""
from __future__ import annotations

import importlib
import sys


def lazy_attributes(package: str, names: dict[str, str]):
    """A module ``__getattr__`` and ``__dir__`` for ``package``. ``names`` maps each lazily re-exported name to the
    module defining it. Any other name is resolved as a submodule of the package, imported on first access, so
    ``package.submodule`` does not depend on another import having loaded it."""

    def __getattr__(name):
        if name in names:
            return getattr(importlib.import_module(names[name]), name)
        if not name.startswith("__") and "." not in name:
            try:
                return importlib.import_module(f"{package}.{name}")
            except ModuleNotFoundError as error:
                if error.name != f"{package}.{name}":
                    raise  # the submodule exists, but a module it imports does not
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    def __dir__():
        return sorted({*vars(sys.modules[package]), *names})

    return __getattr__, __dir__
