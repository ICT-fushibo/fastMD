"""Upstream patch records: version- and source-pinned changes to a model's upstream package.

An adapter that needs a change in its upstream code (a host synchronization
in a call the runtime captures, a table on the wrong device) does not fork or
edit the package: it declares a :class:`PatchRecord` and applies it when the
adapter is loaded (``load_adapter``), and reverts it when the adapter is
closed. A record applies only to the upstream versions it was validated on
and to the exact source it was written against (the sha256 of
``inspect.getsource`` of the original); otherwise :func:`apply_patches`
declines it with a reason and leaves upstream as it is. A mismatch is never
an error: an adapter must stay correct without its declined patches (slower,
e.g. not captured) and says why in its plan.

* A record with a ``replacement`` is a patch. By default it is process-wide:
  the attribute at ``target`` (a module global or a class attribute) and
  every alias in ``bindings`` (``from module import name`` copies, checked
  to be the same object first) are replaced; several adapters of one
  process share one installation of a record (reference-counted by name).
  With ``per_instance`` the replacement is set on the instance passed to
  :func:`apply_patches` (the target's class stays as it is); several
  adapters of one instance (e.g. of one module) share that installation
  (reference-counted by instance and name), and the last one reverts it.
* A record without a replacement is a pin: upstream code an adapter relies
  on unpatched, checked the same way, nothing installed.

:func:`patch_report` is the ``patches`` field of the engine plan record
(``{"applied": [...], "declined": {name: reason}}``); the applied records'
source hashes are part of the plan fingerprint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.metadata
import inspect
import re
from typing import Any, Iterable, Sequence
import weakref

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PatchRecord:
    """One change to upstream code (or, without ``replacement``, a pin of code relied on unchanged)."""

    name: str  # "<family>.<what>", e.g. "sevennet.onehot_table_on_device"
    # The import path of the replaced attribute: "<module>.<attribute>" or "<module>.<Class>.<attribute>"
    target: str
    distribution: str  # the installed distribution whose version is checked, e.g. "sevenn"
    versions: tuple[str, ...]  # the exact upstream versions the record was validated on
    # sha256 of inspect.getsource(<the original target>), or of ``source`` where given
    source_sha256: str
    reason: str  # one sentence: capture safety, host sync, device placement, ...
    replacement: Any = None  # what is installed; None: a pin (checked, nothing installed)
    # Other import paths bound to the target's original object (``from module import name``), replaced too
    bindings: tuple[str, ...] = ()
    # The import path of the code the record was written against, where the target has no source of its
    # own (a tensor attribute, a module global read by that code); None: the target itself
    source: str | None = None
    # Set the replacement on the instance given to apply_patches (an instance of the target's class)
    per_instance: bool = False

    def __post_init__(self) -> None:
        if not self.name or "." not in self.target:
            raise ValueError(f"patch record {self.name!r}: a name and a dotted target path are required")
        if not self.versions or not all(isinstance(version, str) for version in self.versions):
            raise ValueError(f"patch record {self.name!r}: versions must be a non-empty tuple of version strings")
        if not _SHA256.fullmatch(self.source_sha256):
            raise ValueError(f"patch record {self.name!r}: source_sha256 must be 64 lowercase hex digits")
        if self.replacement is None and (self.bindings or self.per_instance):
            raise ValueError(f"patch record {self.name!r}: a pin installs nothing (no bindings, not per instance)")
        if self.per_instance and self.bindings:
            raise ValueError(f"patch record {self.name!r}: a per-instance patch has no module bindings")

    @property
    def pin(self) -> bool:
        return self.replacement is None


@dataclass
class PatchOutcome:
    """What :func:`apply_patches` did with one record."""

    record: PatchRecord
    applied: bool  # installed (a pin: checked)
    reason: str  # why it was declined ("" when applied)
    version: str | None = None  # the installed version of the record's distribution
    reverted: bool = False
    _shared: bool = field(default=False, repr=False)  # a process-wide installation counted in _INSTALLED
    _instance: Any = field(default=None, repr=False)  # a per-instance installation's instance (_INSTANCE_INSTALLED)

    def as_dict(self) -> dict:
        record = self.record
        out = {"name": record.name, "target": record.target, "distribution": record.distribution,
               "version": self.version, "source_sha256": record.source_sha256, "reason": record.reason,
               "kind": "pin" if record.pin else ("per_instance" if record.per_instance else "process")}
        if record.source is not None:
            out["source"] = record.source
        if record.bindings:
            out["bindings"] = list(record.bindings)
        return out


# Process-wide installations by record name: [record, count, undo list].
_INSTALLED: dict[str, list] = {}
# Per-instance installations by (id of the instance, record name): [record, count, (attribute, kept, original),
# weak reference to the instance]; an entry goes when its instance does.
_INSTANCE_INSTALLED: dict[tuple[int, str], list] = {}


def source_sha256(path: str) -> str:
    """sha256 of ``inspect.getsource`` of the object at the import path ``path`` (what a record pins)."""
    return hashlib.sha256(inspect.getsource(_resolve(path)[2]).encode()).hexdigest()


def _resolve(path: str, *, required: bool = True) -> tuple[Any, str, Any]:
    """(owner, attribute, value) of ``path``: its longest importable module prefix, then attributes
    (the value None for a missing last attribute that is not ``required``)."""
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:split])
        try:
            owner = importlib.import_module(module)
        except ModuleNotFoundError as error:
            if error.name is not None and module != error.name and not module.startswith(error.name + "."):
                raise  # the module exists but a module it imports does not
            continue
        for name in parts[split:-1]:
            owner = getattr(owner, name)
        return owner, parts[-1], getattr(owner, parts[-1]) if required else getattr(owner, parts[-1], None)
    raise ModuleNotFoundError(f"no importable module in {path!r}")


def _own(owner, attribute: str) -> tuple[bool, Any]:
    """Whether ``owner`` holds ``attribute`` itself (a class's or an instance's own, a torch module's
    parameter, buffer or submodule; not inherited) and that value (unwrapped static/class methods kept)."""
    own = vars(owner) if hasattr(owner, "__dict__") else {}
    if attribute in own:
        return True, own[attribute]
    for store in ("_parameters", "_buffers", "_modules"):
        if attribute in own.get(store, {}):
            return True, getattr(owner, attribute)
    return False, None


def _decline(record: PatchRecord, version: str | None, reason: str) -> PatchOutcome:
    return PatchOutcome(record, applied=False, reason=reason, version=version)


def _same(a: PatchRecord, b: PatchRecord) -> bool:
    """The same record: equal declarations and the same replacement object."""
    fields = ("name", "target", "distribution", "versions", "source_sha256", "bindings", "source", "per_instance")
    return all(getattr(a, name) == getattr(b, name) for name in fields) and a.replacement is b.replacement


def _check(record: PatchRecord) -> tuple[str | None, str | None, tuple]:
    """(installed version, reason to decline or None, resolved target) of a record."""
    try:
        version = importlib.metadata.version(record.distribution)
    except importlib.metadata.PackageNotFoundError:
        return None, f"{record.distribution} is not installed", ()
    if version not in record.versions:
        return version, (f"{record.distribution} {version} is not a validated version "
                         f"({', '.join(record.versions)})"), ()
    try:
        # (a per-instance target may be an attribute only instances have, e.g. a tensor set in __init__)
        target = _resolve(record.target, required=not record.per_instance)
        pinned = target[2] if record.source is None else _resolve(record.source)[2]
        found = hashlib.sha256(inspect.getsource(pinned).encode()).hexdigest()
    except (ImportError, AttributeError, OSError, TypeError) as error:
        return version, f"cannot read {record.source or record.target}: {type(error).__name__}: {error}", ()
    if found != record.source_sha256:
        return version, (f"the source of {record.source or record.target} changed (sha256 {found[:16]}..., "
                         f"pinned {record.source_sha256[:16]}...)"), ()
    return version, None, target


def apply_patches(records: Iterable[PatchRecord], *, plan=None, instance=None) -> list[PatchOutcome]:
    """Apply each record whose installed version and source hash match; otherwise decline it with a reason.

    Never raises for a mismatch (version, source, an alias that is not the target's object, a missing
    module): the outcome says why and upstream stays as it is. A per-instance record is set on
    ``instance`` (required, an instance of the target's class). Fills ``plan.patches`` (an
    :class:`~fastmd.adapters.negotiate.EnginePlanRecord`) when given. Call outside any CUDA graph.
    """
    outcomes = []
    for record in records:
        if record.per_instance and instance is None:
            raise ValueError(f"patch record {record.name!r} is per instance: pass instance=")
        # A process-wide record installed already had its source checked when it was installed (its source
        # is the replacement's now): its version and its installation are checked; another record under its
        # name is declined below.
        version, why, target = _check(record) if not (
            record.name in _INSTALLED and not record.per_instance) else (*_check_installed(record), ())
        if why is not None:
            outcomes.append(_decline(record, version, why))
            continue
        if record.pin:
            outcomes.append(PatchOutcome(record, applied=True, reason="", version=version))
        elif record.per_instance:
            outcomes.append(_apply_to_instance(record, version, target, instance))
        else:
            outcomes.append(_apply_process_wide(record, version, target))
    if plan is not None:
        plan.patches = patch_report(outcomes)
    return outcomes


def _version(record: PatchRecord) -> str | None:
    try:
        return importlib.metadata.version(record.distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_installed(record: PatchRecord) -> tuple[str | None, str | None]:
    """(installed version, reason to decline or None) of a record under the name of a process-wide
    installation: the installed record itself, a validated version, and the target still holding the installed
    replacement (looked up statically: a static or class method as it was set)."""
    installed = _INSTALLED[record.name][0]
    version = _version(record)
    if not _same(installed, record):
        return version, f"another record is installed as {record.name!r}"
    if version not in record.versions:
        return version, (f"{record.distribution} {version} is not a validated version "
                         f"({', '.join(record.versions)})")
    try:
        owner, attribute, _ = _resolve(record.target, required=False)
    except (ImportError, AttributeError) as error:
        return version, f"cannot read {record.target}: {type(error).__name__}: {error}"
    if _own(owner, attribute)[1] is not installed.replacement:
        return version, f"{record.target} is not the installed replacement any more (set outside the patch records)"
    return version, None


def _apply_process_wide(record: PatchRecord, version: str | None, target: tuple) -> PatchOutcome:
    installed = _INSTALLED.get(record.name)
    if installed is not None:
        if not _same(installed[0], record):
            return _decline(record, version, f"another record is installed as {record.name!r}")
        installed[1] += 1
        return PatchOutcome(record, applied=True, reason="", version=version, _shared=True)
    owner, attribute, original = target
    sites = [(owner, attribute)]
    for path in record.bindings:
        try:
            alias_owner, alias, value = _resolve(path)
        except (ImportError, AttributeError) as error:
            return _decline(record, version, f"cannot read the binding {path}: {type(error).__name__}: {error}")
        if value is not original:
            return _decline(record, version, f"the binding {path} is not the target's object (re-bound elsewhere)")
        sites.append((alias_owner, alias))
    undo = []
    try:
        for site_owner, site in sites:
            undo.append((site_owner, site, *_own(site_owner, site)))
            setattr(site_owner, site, record.replacement)
    except Exception as error:  # noqa: BLE001 - an attribute that cannot be set declines, all or nothing
        _restore(undo)
        return _decline(record, version, f"cannot set {record.target}: {type(error).__name__}: {error}")
    _INSTALLED[record.name] = [record, 1, undo]
    return PatchOutcome(record, applied=True, reason="", version=version, _shared=True)


def _apply_to_instance(record: PatchRecord, version: str | None, target: tuple, instance) -> PatchOutcome:
    owner, attribute, _ = target
    if inspect.isclass(owner) and not isinstance(instance, owner):
        return _decline(record, version, f"{type(instance).__name__} is not an instance of {owner.__name__}")
    key = (id(instance), record.name)
    installed = _INSTANCE_INSTALLED.get(key)
    if installed is not None and installed[3]() is instance:
        # Another adapter of this instance applying the same record (the same replacement object): one
        # installation, reverted by the last owner.
        if not _same(installed[0], record):
            return _decline(record, version, f"another record is installed as {record.name!r} on this instance")
        if _own(instance, attribute)[1] is not record.replacement:
            return _decline(record, version, f"{attribute} is not the installed replacement any more (set outside "
                            "the patch records)")
        installed[1] += 1
        return PatchOutcome(record, applied=True, reason="", version=version, _instance=instance)
    kept, original = _own(instance, attribute)
    try:
        setattr(instance, attribute, record.replacement)
    except Exception as error:  # noqa: BLE001 - an attribute that cannot be set declines
        _restore([(instance, attribute, kept, original)])
        return _decline(record, version, f"cannot set {attribute}: {type(error).__name__}: {error}")
    # (No strong reference to the instance here: an instance dropped unreverted drops its entry.)
    _INSTANCE_INSTALLED[key] = [record, 1, (attribute, kept, original),
                                weakref.ref(instance, lambda _, key=key: _INSTANCE_INSTALLED.pop(key, None))]
    return PatchOutcome(record, applied=True, reason="", version=version, _instance=instance)


def _restore(undo: list) -> None:
    for owner, attribute, kept, original in reversed(undo):
        if kept:
            setattr(owner, attribute, original)
        else:
            try:
                delattr(owner, attribute)
            except AttributeError:
                pass


def revert_patches(outcomes: Sequence[PatchOutcome]) -> None:
    """Undo the applied outcomes, last first: an installation (process-wide, or on one instance) when the
    last adapter that applied it reverts. Idempotent; declined outcomes and pins are skipped."""
    for outcome in reversed(list(outcomes)):
        if not outcome.applied or outcome.reverted or outcome.record.pin:
            continue
        if outcome._instance is not None:
            key = (id(outcome._instance), outcome.record.name)
            installed = _INSTANCE_INSTALLED.get(key)
            if installed is not None and installed[3]() is outcome._instance:
                installed[1] -= 1
                if installed[1] == 0:
                    del _INSTANCE_INSTALLED[key]
                    _restore([(outcome._instance, *installed[2])])
            outcome.reverted = True
            continue
        installed = _INSTALLED.get(outcome.record.name)
        if installed is not None and installed[1] == 1:
            _restore(installed[2])
            del _INSTALLED[outcome.record.name]
        elif installed is not None:
            installed[1] -= 1
        outcome.reverted = True


def patch_report(outcomes: Sequence[PatchOutcome]) -> dict:
    """The plan record's ``patches``: the applied records (patches and pins) and the declined ones with
    their reasons; a reverted record (its adapter closed) is listed as declined, with that reason."""
    return {"applied": [outcome.as_dict() for outcome in outcomes if outcome.applied and not outcome.reverted],
            "declined": {outcome.record.name: outcome.reason if not outcome.applied else "reverted: the adapter was "
                         "closed" for outcome in outcomes if not outcome.applied or outcome.reverted}}


def applied_pins(outcomes: Sequence[PatchOutcome]) -> list[list[str]]:
    """(name, source sha256) of each applied, unreverted record, sorted: the plan fingerprint's part."""
    return sorted([outcome.record.name, outcome.record.source_sha256] for outcome in outcomes
                  if outcome.applied and not outcome.reverted)
