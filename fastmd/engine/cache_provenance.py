"""Provenance for frozen projection caches, independent of execution policy.

A model registers its cache groups (``register_projection_cache_groups``;
MatRIS's are in ``fastmd.models.matris.projection_caches``), and only their
builders publish these seals. A seal attests to metadata
and mutation versions, not tensor contents: no weights are copied, hashed, or
synchronized. This lets an immutable inference model acquire a legitimate
derived cache while rejecting injected/replaced/mutated cache tensors.

The builders also retain what a CUDA graph capture reads (``capture_retention``):
a captured graph reads a derived tensor by its address, while the module's
single-slot cache may later be rebuilt for another key (another engine sharing
the model under another matmul precision or autocast state, or new weights).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from weakref import WeakKeyDictionary, ref

import torch
from torch import Tensor


# The registered cache groups ({group: the module fields its builder assigns}) and all their fields.
PROJECTION_CACHE_GROUPS: dict[str, tuple[str, ...]] = {}
PROJECTION_CACHE_FIELDS: set[str] = set()


def register_projection_cache_groups(groups) -> None:
    """Register a model's projection cache groups (``{group: field names}``): the module fields one builder
    assigns and publishes together (``publish_projection_cache``), which the distributed model guard reads as
    cache state. Registering a group again with the same fields does nothing. A group registered with other
    fields, or a field of another group, is an error, and a call that raises registers nothing."""
    groups = {group: tuple(names) for group, names in groups.items()}
    owners = {name: group for group, names in PROJECTION_CACHE_GROUPS.items() for name in names}
    for group, names in groups.items():
        if PROJECTION_CACHE_GROUPS.get(group, names) != names:
            raise ValueError(f"projection cache group {group!r} is registered with other fields")
        for name in names:
            if owners.setdefault(name, group) != group:
                raise ValueError(f"projection cache field {name!r} belongs to the group {owners[name]!r}")
    for group, names in groups.items():
        PROJECTION_CACHE_GROUPS.setdefault(group, names)
        PROJECTION_CACHE_FIELDS.update(names)


# Stateless immutable scalar results, not a tensor/epoch or cache-key memo.
# Cold models contain many None fields; identity checks also avoid invoking
# user equality/hash hooks on malformed values.
_NONE_STAMP = ("value", None)
_TRUE_STAMP = ("value", True)
_FALSE_STAMP = ("value", False)


def _tensor_metadata(value):
    return (id(value), value._version, tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device, value.requires_grad, value.data_ptr())


def cache_value_stamp(value, tensor_stamp=None):
    """Freeze the deliberately small, known cache metadata vocabulary."""
    if value is None:
        return _NONE_STAMP
    if value is True:
        return _TRUE_STAMP
    if value is False:
        return _FALSE_STAMP
    if isinstance(value, Tensor):
        return ("tensor", (_tensor_metadata if tensor_stamp is None else tensor_stamp)(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(cache_value_stamp(part, tensor_stamp) for part in value))
    if isinstance(value, (bool, int, float, str, type(None), torch.dtype, torch.device)):
        return ("value", value)
    # Never call user-provided equality/repr hooks on malformed cache values.
    return ("unsupported", type(value), id(value))


@dataclass(frozen=True)
class _ProjectionSeal:
    sources: tuple
    source_stamps: tuple
    fields: tuple


# Weak module keys and weak source references do not keep a model or GPU weight
# alive after its owner is released. Cache fields below contain metadata only.
_SEALS = WeakKeyDictionary()


def publish_projection_cache(module, group, sources):
    """Publish immediately after one official builder assigns all its fields."""
    names = PROJECTION_CACHE_GROUPS[group]
    sources = tuple(sources)
    if not sources or any(not isinstance(value, Tensor) for value in sources):
        raise TypeError("projection cache sources must be nonempty tensors")
    namespace = vars(module)
    fields = tuple((name, cache_value_stamp(namespace.get(name))) for name in names)
    if all(namespace.get(name) is None for name in names):
        raise ValueError("cannot publish an empty projection cache")
    seal = _ProjectionSeal(tuple(ref(value) for value in sources),
                           tuple(_tensor_metadata(value) for value in sources), fields)
    _SEALS.setdefault(module, {})[group] = seal


def validate_projection_caches(module, namespace=None, *, field_stamps=None, tensor_metadata=None):
    """Return errors for untrusted caches; empty never-built groups are legal.

    Calling this from a boundary's existing namespace walk avoids any extra
    model traversal. A previously published cache may not be silently removed
    or reset to cold state: that is also a mutation of an immutable program.

    A caller may supply canonical field stamps and a metadata callback from
    its *current* boundary inspection. Both must use the same eight-field
    tensor format as _tensor_metadata, including data_ptr, not an executor's
    different internal tensor-stamp format. Neither input is stored here.
    Missing supplied fields are inspected normally, including absent/None
    fields of a previously sealed group. Standalone callers need no memo.
    """
    namespace = vars(module) if namespace is None else namespace
    metadata = _tensor_metadata if tensor_metadata is None else tensor_metadata
    seals = _SEALS.get(module, {})
    errors = []
    for group, names in PROJECTION_CACHE_GROUPS.items():
        seal = seals.get(group)
        materialized = any(namespace.get(name) is not None for name in names)
        if not materialized and seal is None:
            continue
        if seal is None:
            errors.append(f"untrusted projection cache: {group}")
            continue
        current = tuple(
            (name, field_stamps[name] if field_stamps is not None and name in field_stamps
             else cache_value_stamp(namespace.get(name), metadata))
            for name in names)
        if current != seal.fields:
            errors.append(f"projection cache was mutated: {group}")
            continue
        sources = tuple(source() for source in seal.sources)
        if (any(value is None for value in sources)
                or tuple(metadata(value) for value in sources) != seal.source_stamps):
            errors.append(f"projection cache source was mutated: {group}")
    return tuple(errors)


# Derived caches under CUDA graph capture ------------------------------------

class StaleCaptureError(RuntimeError):
    """A captured graph would read a projection cache derived from weights that
    changed after the capture (in place, or rebound with the old tensor released)."""


class _CaptureState(threading.local):
    holder = None  # the capture in progress on this thread: [(derived, sources), ...]


# Builders test ``CAPTURE.holder is not None`` before calling retain_under_capture:
# an attribute read, not a call, so frames that run under torch.compile's frame
# hook (after a graph break) meet no new code object outside a capture.
CAPTURE = _CaptureState()


def retain_under_capture(derived, sources) -> None:
    """Called by a builder that returns ``derived`` (tensors, or a tuple of them),
    derived from the parameter tensors ``sources``, while ``CAPTURE.holder`` is set
    (inside ``capture_retention``): the capture keeps ``derived`` alive and records
    the sources' versions."""
    CAPTURE.holder.append((derived, tuple(sources)))


def _check_sources(stamps) -> None:
    for source, version, pointer in stamps:
        tensor = source()
        if tensor is None or tensor._version != version or tensor.data_ptr() != pointer:
            raise StaleCaptureError(
                "the model's weights changed after this engine captured its CUDA graphs (e.g. an "
                "in-place training step or load_state_dict): its captured projection caches are "
                "stale and the graph was not replayed. Create a new MLIPCalculator / "
                "GPUResidentMolecularDynamics for the updated model (or use execution='eager'); an "
                "MD step that raised this may have advanced its state before the force call, so "
                "restart from your own atoms, not from this engine's state.")


@contextmanager
def capture_retention(graph, *, check_replay: bool):
    """Around one capture into ``graph``: every derived cache the builders return
    meanwhile stays alive as long as ``graph`` (``graph._matris_derived``), so a
    rebuild of a module's single-slot cache never frees memory the graph reads.

    ``check_replay``: ``graph.replay()`` first checks that the weights those caches
    were derived from are unchanged (the same live tensor, version and address) and raises
    ``StaleCaptureError`` instead of replaying stale caches. Distributed engines
    pass False: their model guard rejects a model edit collectively, while a
    one-rank raise before a replay that holds collectives would stall its peers.

    Compiled layers are covered too: a builder compares data pointers, which
    dynamo cannot trace, so its frame runs eagerly inside a compiled layer.

    Not detected: writes through ``param.data`` (no version bump, eager caches miss
    them as well), a rebinding while the old weight stays referenced elsewhere (the
    graph keeps replaying the old weights), and the rebinding of a weight that no
    derived cache comes from (the graph reads its old storage).
    """
    holder = []
    previous = CAPTURE.holder
    CAPTURE.holder = holder
    try:
        yield
    finally:
        CAPTURE.holder = previous
    graph._matris_derived = tuple(derived for derived, _ in holder)
    sources = {id(tensor): tensor for _, group in holder for tensor in group}
    if check_replay and sources:
        stamps = tuple((ref(tensor), tensor._version, tensor.data_ptr()) for tensor in sources.values())
        graph_ref, replay = ref(graph), type(graph).replay

        def checked_replay():
            _check_sources(stamps)
            return replay(graph_ref())

        # An instance attribute: the runners call graph.replay() unchanged. It holds
        # the graph weakly (no reference cycle that only a collection would free).
        graph.replay = checked_replay
