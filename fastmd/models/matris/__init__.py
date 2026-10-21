"""The MatRIS model. Importing this package does not import the model, and importing one of its modules (a kernel,
``functions``, ...) imports what that module needs: ``MatRIS``, and a module accessed as an attribute of the package,
are imported on first access."""
from fastmd import _lazy

__all__ = ["MatRIS"]
__getattr__, __dir__ = _lazy.lazy_attributes(__name__, {"MatRIS": "fastmd.models.matris.model"})
