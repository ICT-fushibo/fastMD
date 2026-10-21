"""The nvalchemiops neighbor list behind the GPU graph builders, resolved once at import.

nvalchemiops 0.3 moved its torch API to ``nvalchemiops.torch.neighbors``;
``nvalchemiops.neighborlist`` became a layer over the Warp launchers without
the torch functions MatRIS calls. This module imports the current torch API
first and the 0.2 API second, and exports what the ragged and fixed-capacity
builders and the multi-rank candidate search call, with the 0.2 behaviour they
rely on under either API:

- a query counts every pair within the cutoff in ``num_neighbors`` but stores
  only a row's first ``max_neighbors`` slots, so ``num_neighbors.max() >
  max_neighbors`` detects truncation exactly;
- ``estimate_max_neighbors`` is the 0.2 volume estimate (0.4 lowered the
  default density and deprecated ``safety_factor``);
- the cell grid only orders the candidate search: the current API keeps its
  own finer default (at least 4 cells per periodic dimension, at most 8192
  per system; 0.2: cells no smaller than the cutoff, at most 1000), which
  finds the same pairs faster;
- the query is the atom-centric one that writes both rows of a pair from one
  distance, with no host synchronization (0.4's automatic choice may pick a
  pair-centric query, which reads sizes back to the host);
- the Warp kernels run on Warp's current stream for the device, as before:
  callers that need them ordered with torch's stream wrap the calls in
  ``wp.ScopedStream(wp.stream_from_torch(...))``. The candidate refreshes run
  outside CUDA graph capture; the current API's cell-list build also
  allocates scan scratch through Warp's allocator.

With the current API, whose cell-list sizing reads the cell back to the host
twice, a fixed-cell caller sizes its cell list once (``cell_list_for_fixed_cell``)
and passes it to every ``neighbor_list`` call.

``API`` names the resolved API (``"torch"``, ``"neighborlist"`` for the 0.2
API, None when neither imports), ``VERSION`` the installed
nvalchemi-toolkit-ops version and ``UNAVAILABLE_REASON`` why neither API
imports; ``describe()`` collects them for diagnostics.
"""
from __future__ import annotations

import importlib
import inspect
import math
import re
from importlib import metadata

DISTRIBUTION = "nvalchemi-toolkit-ops"
SUPPORTED_VERSIONS = ">=0.4.1,<0.5 (nvalchemiops.torch.neighbors), or 0.2.0 (the old API, removed in a later version)"
_TORCH_API_RANGE = ((0, 4, 1), (0, 5))

# The current API's default cell grid (pinned): at most this many cells per system, at least
# this many per periodic dimension (0.2: 1000 cells, none smaller than the cutoff).
_MAX_CELLS_PER_SYSTEM = 8192
_MIN_CELLS_PER_DIMENSION = 4


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _version() -> str | None:
    """The nvalchemiops version: its package metadata, else the module's ``__version__``."""
    version = _installed_version(DISTRIBUTION)
    if version is None:
        try:
            version = getattr(importlib.import_module("nvalchemiops"), "__version__", None)
        except Exception:
            return None
    return version


def _release(version: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def estimate_max_neighbors(cutoff: float, atomic_density: float = 0.35, safety_factor: float = 5.0) -> int:
    """Per-atom neighbor slots, nvalchemiops 0.2's estimate under either API.

    ``safety_factor * atomic_density * (4/3) pi cutoff^3``, at least 1, rounded
    up to a multiple of 16; 0 for a non-positive cutoff.
    """
    if cutoff <= 0:
        return 0
    cutoff_sphere_volume = atomic_density * (4.0 / 3.0) * math.pi * (cutoff**3)
    expected_neighbors = max(1, safety_factor * cutoff_sphere_volume)
    return int(math.ceil(expected_neighbors / 16)) * 16


def _check_warp_requirement(version: str) -> None:
    """The warp-lang lower bound this nvalchemiops release declares, against the installed warp-lang."""
    try:
        requirements = metadata.requires(DISTRIBUTION) or []
    except metadata.PackageNotFoundError:
        return
    warp = _installed_version("warp-lang")
    for requirement in requirements:
        match = re.fullmatch(r"warp-lang\s*>=\s*([0-9.]+)", requirement.strip())
        if match and warp is not None and _release(warp) < _release(match.group(1)):
            raise ImportError(f"{DISTRIBUTION} {version} requires warp-lang>={match.group(1)}, found {warp}")


def _torch_api() -> dict:
    """The current torch API, pinned to the 0.2 behaviour (see the module docstring)."""
    version = _version()
    low, high = _TORCH_API_RANGE
    if version is None or not low <= _release(version) < high:
        raise ImportError(f"its torch API is used for {DISTRIBUTION} >=0.4.1,<0.5, found {version}")
    _check_warp_requirement(version)
    import warp as wp

    # nvalchemiops quiets Warp with config.quiet, which warp-lang 1.18 replaced with log_level.
    if hasattr(wp.config, "log_level") and wp.config.log_level == wp.LOG_INFO:
        wp.config.log_level = wp.LOG_WARNING
    module = importlib.import_module("nvalchemiops.torch.neighbors.batch_cell_list")
    from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError
    from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list

    needed = {
        module.estimate_batch_cell_list_sizes: ("max_nbins", "min_cells_per_dimension"),
        module.batch_build_cell_list: ("min_cells_per_dimension",),
        module.batch_query_cell_list: ("strategy", "atom_centric_path"),
    }
    for function, names in needed.items():
        missing = [name for name in names if name not in inspect.signature(function).parameters]
        if missing:
            raise ImportError(f"{module.__name__}.{function.__name__} has no {', '.join(missing)}")

    def estimate_batch_cell_list_sizes(cell, pbc, cutoff):
        return module.estimate_batch_cell_list_sizes(
            cell, pbc, cutoff, max_nbins=_MAX_CELLS_PER_SYSTEM,
            min_cells_per_dimension=_MIN_CELLS_PER_DIMENSION)

    def batch_build_cell_list(positions, cutoff, cell, pbc, batch_idx, cells_per_dimension,
                              neighbor_search_radius, atom_periodic_shifts, atom_to_cell_mapping,
                              atoms_per_cell_count, cell_atom_start_indices, cell_atom_list):
        module.batch_build_cell_list(
            positions, cutoff, cell, pbc, batch_idx, cells_per_dimension, neighbor_search_radius,
            atom_periodic_shifts, atom_to_cell_mapping, atoms_per_cell_count, cell_atom_start_indices,
            cell_atom_list, min_cells_per_dimension=_MIN_CELLS_PER_DIMENSION)

    def batch_query_cell_list(positions, cell, pbc, cutoff, batch_idx, cells_per_dimension,
                              neighbor_search_radius, atom_periodic_shifts, atom_to_cell_mapping,
                              atoms_per_cell_count, cell_atom_start_indices, cell_atom_list,
                              neighbor_matrix, neighbor_matrix_shifts, num_neighbors, half_fill=False):
        module.batch_query_cell_list(
            positions, cell, pbc, cutoff, batch_idx, cells_per_dimension, neighbor_search_radius,
            atom_periodic_shifts, atom_to_cell_mapping, atoms_per_cell_count, cell_atom_start_indices,
            cell_atom_list, neighbor_matrix, neighbor_matrix_shifts, num_neighbors, half_fill,
            strategy="atom_centric", atom_centric_path="direct")

    def cell_list_for_fixed_cell(cell, pbc, cutoff, total_atoms, device):
        max_total_cells, radius = estimate_batch_cell_list_sizes(cell, pbc, cutoff)
        return allocate_cell_list(total_atoms, max_total_cells, radius, device)

    def neighbor_list(positions, cutoff, cell, pbc, batch_idx, method="batch_cell_list", *,
                      neighbor_matrix, neighbor_matrix_shifts, num_neighbors, half_fill=False,
                      return_neighbor_list=False, cell_list=None):
        """0.2's ``neighbor_list(method="batch_cell_list")`` into caller buffers; ``cell_list``:
        one from ``cell_list_for_fixed_cell`` for this cell, pbc, cutoff and atom count."""
        if method != "batch_cell_list" or return_neighbor_list:
            raise ValueError("only the batch cell list into caller buffers is provided")
        total = positions.shape[0]
        if total <= 0 or cutoff <= 0:
            return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
        neighbor_matrix.fill_(total)
        neighbor_matrix_shifts.zero_()
        num_neighbors.zero_()
        if cell_list is None:
            cell_list = cell_list_for_fixed_cell(cell, pbc, cutoff, total, positions.device)
        batch_build_cell_list(positions, cutoff, cell, pbc, batch_idx, *cell_list)
        batch_query_cell_list(positions, cell, pbc, cutoff, batch_idx, *cell_list,
                              neighbor_matrix, neighbor_matrix_shifts, num_neighbors, half_fill)
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts

    return {
        "NeighborOverflowError": NeighborOverflowError,
        "allocate_cell_list": allocate_cell_list,
        "batch_build_cell_list": batch_build_cell_list,
        "batch_query_cell_list": batch_query_cell_list,
        "cell_list_for_fixed_cell": cell_list_for_fixed_cell,
        "estimate_batch_cell_list_sizes": estimate_batch_cell_list_sizes,
        "neighbor_list": neighbor_list,
    }


def _old_api() -> dict:
    """The nvalchemiops 0.2 API, unchanged. Old API: removed in a later version."""
    version = _version()
    if version is not None and _release(version)[:2] != (0, 2):
        raise ImportError(f"the old API is used for {DISTRIBUTION} 0.2, found {version}")
    from nvalchemiops.neighborlist.batch_cell_list import (
        batch_build_cell_list,
        batch_query_cell_list,
        estimate_batch_cell_list_sizes,
    )
    from nvalchemiops.neighborlist.neighbor_utils import NeighborOverflowError, allocate_cell_list
    from nvalchemiops.neighborlist.neighborlist import neighbor_list

    return {
        "NeighborOverflowError": NeighborOverflowError,
        "allocate_cell_list": allocate_cell_list,
        "batch_build_cell_list": batch_build_cell_list,
        "batch_query_cell_list": batch_query_cell_list,
        "estimate_batch_cell_list_sizes": estimate_batch_cell_list_sizes,
        "neighbor_list": neighbor_list,
    }


def resolve() -> tuple[str | None, dict, str | None]:
    """(API name, functions, reason neither imports): the torch API first, then 0.2's."""
    errors = []
    for name, load in (("torch", _torch_api), ("neighborlist", _old_api)):
        try:
            return name, load(), None
        except Exception as error:  # ImportError, or AttributeError from an incompatible warp
            errors.append(f"{name}: {type(error).__name__}: {error}")
    version = _version()
    if version is None:
        return None, {}, f"{DISTRIBUTION} is not installed (supported: {SUPPORTED_VERSIONS})"
    warp = _installed_version("warp-lang")
    return None, {}, (f"{DISTRIBUTION} {version} (warp-lang {warp}) is unsupported or does not import "
                      f"({'; '.join(errors)}); supported: {SUPPORTED_VERSIONS}")


class _UnavailableNeighborOverflowError(Exception):
    """Stands in for nvalchemiops' NeighborOverflowError when it does not import."""

    def __init__(self, max_neighbors: int, num_neighbors: int):
        super().__init__(f"The number of neighbors is larger than the maximum allowed: "
                         f"{num_neighbors} > {max_neighbors}.")


API, _functions, UNAVAILABLE_REASON = resolve()
VERSION = _version()
available = API is not None


def _unavailable(*args, **kwargs):
    raise ImportError(UNAVAILABLE_REASON)


def _sized_per_call(cell, pbc, cutoff, total_atoms, device):
    return None


NeighborOverflowError = _functions.get("NeighborOverflowError", _UnavailableNeighborOverflowError)
allocate_cell_list = _functions.get("allocate_cell_list", _unavailable)
batch_build_cell_list = _functions.get("batch_build_cell_list", _unavailable)
batch_query_cell_list = _functions.get("batch_query_cell_list", _unavailable)
# A cell list a fixed-cell caller sizes once and passes to every neighbor_list call
# (``cell_list=``); None with the 0.2 API, which sizes it per call as before.
cell_list_for_fixed_cell = _functions.get("cell_list_for_fixed_cell", _sized_per_call)
estimate_batch_cell_list_sizes = _functions.get("estimate_batch_cell_list_sizes", _unavailable)
neighbor_list = _functions.get("neighbor_list", _unavailable)
del _functions


def require(what: str | None = None) -> None:
    """Raise ImportError naming the missing or unsupported nvalchemiops version."""
    if not available:
        raise ImportError(UNAVAILABLE_REASON if what is None else f"{what}: {UNAVAILABLE_REASON}")


def describe() -> dict:
    """The resolved neighbor-list backend, for diagnostics and run records."""
    return {"package": DISTRIBUTION, "version": VERSION, "api": API,
            "warp": _installed_version("warp-lang"), "unavailable_reason": UNAVAILABLE_REASON}
