"""Opt-in numerical-accuracy experiments; production precision stays explicit."""

from fastmd.accuracy.layered_precision import (
    LayeredPrecisionInstallation,
    apply_layered_precision,
    discover_linear_groups,
)

__all__ = [
    "LayeredPrecisionInstallation",
    "apply_layered_precision",
    "discover_linear_groups",
]
