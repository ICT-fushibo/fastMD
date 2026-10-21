"""Kernel-entry operator-selection recorder (design layer L1, ``OpPlanRecorder``).

Records what actually executed, not what a branch predicted (design §4.7):
every instrumented kernel entry (custom autograd Function forward/backward,
the CPU branch of an op wrapper), every dispatch site that takes a generic
lowering, and every run-time decline of a fused or cached implementation.

Instrumentation idiom; the only cost while no recorder is active is one
module-global read and a ``None`` compare (no call, allocation or sync)::

    from fastmd.engine import recorder as _recorder

    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("residual_add", "triton", {"rows": n, "width": d, "BM": bm})

Nothing is registered while no recorder is active. While one is active:

* global ``nn.Module`` forward pre/post hooks maintain a process-global site
  stack. Names come from ``root.named_modules()`` (the root itself is named
  by its class; ``torch.compile`` ``_orig_mod`` components are dropped so
  eager and compiled runs share site names); other modules use their class
  name. The stack is shared by all threads: it is read by forward records
  and during non-reentrant checkpoint recomputation, which runs on the
  autograd thread while the forward thread waits. Code that rejects custom
  global hooks may exempt the recorder's own (``recorder_hook_ids``).
* with ``aten=True`` dispatcher ops are counted per context: the module site
  in forward, ``"bwd:" + <autograd node name>`` while an autograd node
  executes (this wins over a site, e.g. during checkpoint recomputation),
  ``"<outside>"`` without either. Counting uses the autograd profiler's CPU
  op events (RecordFunction observers: ``aten::<name>`` including ops nested
  in composite ops, and custom-namespace ops such as ``matris::`` or
  ``c10d::``), not a ``TorchDispatchMode``: a dispatch mode makes every
  tensor "subclass-like", which switches autograd to out-of-place backward
  formulas (extra copies and allocations, also in CUDA graphs captured
  meanwhile) and makes dynamo skip compiled frames for good. The profiler
  observes the entering thread and the autograd work it launches, not other
  Python threads; it must not already be enabled (use ``aten=False`` inside
  another profiler). Counts are collected when the recorder exits.

Phases: ``"fwd"``, ``"bwd"`` and ``"recompute"``. A forward-entry record made
while an autograd node executes (non-reentrant checkpoint recomputation, or
a Function applied inside another Function's VJP) is stored as
``"recompute"``. Backward records pass the site stashed by their forward
(``ctx._opsel_site``); without one they use ``"bwd:" + <node name>``.
Backward runs on autograd worker threads: appends are serialized by a lock
and no thread-local state is used.

torch.compile: with the recorder off, instrumentation points constant-fold
(dynamo guards ``_ACTIVE is None``) and dynamo traces exactly as before.
While a recorder is active its public methods and hooks are wrapped with
``torch._dynamo.disable``, so they run at execution time, never only at
trace time: dynamo re-traces the compiled frames with a graph break at every
record (and may give up on frames that break inside a loop). That program is
what a recording of a compiled model executes. So that it never outlives the
recording, a recorder during which dynamo compiled anything resets dynamo's
code caches on exit (``torch._dynamo.reset_code_caches()``, process-wide):
the next call recompiles exactly as a process that never recorded. CUDA
graphs captured from compiled code while recording keep the recording-time
program; recapture them after the recorder exits.

CUDA graphs: records are made on the host during warmup and capture; a
replay executes no Python and records nothing.

The snapshot's ``policy_fingerprints`` lists the ``KernelPolicy.from_globals()``
fingerprints in effect at the top-level module forwards (empty site stack, not
a recomputation) made while recording (see ``fastmd.engine.policy``).

``params`` must be JSON scalars (bool/int/float/str/None, or lists/tuples of
them); ``torch.dtype``/``torch.device`` are stored as strings. Anything else is
stored as ``"<TypeName>"`` and counted in the snapshot's ``invalid_params``
(the recorder never raises into the model). Callers must never read device
data to build params.
"""
from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate
import json
import re
import threading
import time

import torch

from fastmd.engine.policy import policy_summary

SCHEMA = "fastmd.op_selection/2"
OUTSIDE = "<outside>"
PHASES = ("fwd", "bwd", "recompute")

# The one active recorder, or None. Instrumentation points read this directly.
_ACTIVE = None
_INSTALL_LOCK = threading.Lock()
_SCALARS = (bool, int, float, str, type(None))
# Methods replaced by dynamo-disabled wrappers while a recorder is active.
_PUBLIC = ("record", "fallback", "site", "mark")
# Profiler event names: our site ranges and section marks, autograd nodes.
_SITE_EVENT = "opsel.site:"
_MARK_EVENT = "opsel.mark:"
_NODE_EVENT = "autograd::engine::evaluate_function: "
# Dispatcher op events are "<namespace>::<op>"; autograd/profiler internals are not ops.
_OP_EVENT = re.compile(r"(?!autograd::|profiler::)[A-Za-z_]\w*::\w+\Z")


def canonical_json(snapshot: dict) -> str:
    """Deterministic text form of a snapshot (for golden files and diffs)."""
    return json.dumps(snapshot, sort_keys=True, indent=1) + "\n"


def recorder_hook_ids() -> frozenset:
    """Handle ids of the active recorder's global module hooks (empty if none).

    They are keys of ``torch.nn.modules.module._global_forward_pre_hooks`` /
    ``_global_forward_hooks`` (and ``_global_forward_hooks_always_called``).
    The hooks only maintain the recorder's site stack, so code that rejects
    or stamps custom global hooks may ignore these keys.
    """
    rec = _ACTIVE
    return frozenset() if rec is None else rec._hook_ids


def _class_name(module) -> str:
    # A torch.compile wrapper is named after the module it wraps.
    return type(getattr(module, "_orig_mod", module)).__name__


def _new_section(label: str) -> dict:
    return {"label": label, "entries": [], "fallbacks": [], "aten": {}}


def _dynamo_frames() -> int:
    from torch._dynamo.utils import counters

    return counters["frames"]["total"]


class OpPlanRecorder:
    """Context manager recording the executed operator plan; see module doc."""

    def __init__(self, aten: bool = True, root=None) -> None:
        if type(aten) is not bool:
            raise TypeError("aten must be bool")
        if root is None:
            roots = ()
        elif isinstance(root, torch.nn.Module):
            roots = (root,)
        else:
            roots = tuple(root)
            if not all(isinstance(module, torch.nn.Module) for module in roots):
                raise TypeError("root must be an nn.Module or a sequence of nn.Modules")
        self.aten = aten
        self._roots = roots
        self._lock = threading.Lock()
        self._sections = [_new_section("default")]
        self._marked = False
        self._invalid_params = 0
        self._names: dict = {}
        self._outsiders: dict = {}
        self._stack: list = []
        self._handles: list = []
        self._hook_ids = frozenset()
        self._profiler = None
        self._mark_times: list = []
        self._dynamo_frames = None
        self._policies: set = set()

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> "OpPlanRecorder":
        global _ACTIVE
        with _INSTALL_LOCK:
            if _ACTIVE is not None:
                raise RuntimeError(
                    "an OpPlanRecorder is already active; recorders do not nest")
            try:
                self._install()
            except BaseException:
                self._uninstall()
                raise
            _ACTIVE = self
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        global _ACTIVE
        with _INSTALL_LOCK:
            try:
                self._uninstall()
            finally:
                if _ACTIVE is self:
                    _ACTIVE = None
        return False

    def _install(self) -> None:
        import torch._dynamo
        from torch.nn.modules.module import (
            register_module_forward_hook,
            register_module_forward_pre_hook,
        )

        disable = torch._dynamo.disable
        for name in _PUBLIC:
            setattr(self, name, disable(getattr(type(self), name).__get__(self)))
        self._names = self._scan()
        self._outsiders = {}
        self._stack = []
        self._mark_times = []
        self._dynamo_frames = _dynamo_frames()
        if self.aten:
            if torch.autograd._profiler_enabled():
                raise RuntimeError(
                    "OpPlanRecorder(aten=True) counts aten ops with the autograd profiler, "
                    "which is already enabled; use aten=False")
            profiler = torch.autograd.profiler.profile(use_cpu=True)
            profiler.__enter__()
            self._profiler = profiler
        self._handles.append(register_module_forward_pre_hook(disable(self._pre_hook)))
        self._handles.append(
            register_module_forward_hook(disable(self._post_hook), always_call=True))
        self._hook_ids = frozenset(handle.id for handle in self._handles)

    def _uninstall(self) -> None:
        try:
            handles, self._handles = self._handles, []
            for handle in handles:
                handle.remove()
            self._hook_ids = frozenset()
            for name in _PUBLIC:
                self.__dict__.pop(name, None)
            # Close site ranges left open (a recorder exited inside a forward).
            for _, _, scope in reversed(self._stack):
                if scope is not None:
                    scope.__exit__(None, None, None)
            self._stack = []
            self._names = {}
            self._outsiders = {}
        finally:
            profiler, self._profiler = self._profiler, None
            try:
                if profiler is not None:
                    profiler.__exit__(None, None, None)
                    self._collect(profiler.kineto_results.events())
            finally:
                frames, self._dynamo_frames = self._dynamo_frames, None
                if frames is not None and _dynamo_frames() != frames:
                    # Dynamo (re)compiled while recording: those frames carry
                    # the recorder's graph breaks. Drop them (see module doc).
                    import torch._dynamo

                    torch._dynamo.reset_code_caches()

    # ---------------------------------------------------------------- sites

    def _scan(self) -> dict:
        names = {}
        for root in self._roots:
            for qualified, module in root.named_modules():
                parts = [part for part in qualified.split(".") if part and part != "_orig_mod"]
                names.setdefault(module, ".".join(parts) or _class_name(root))
        return names

    def _name(self, module) -> str:
        name = self._names.get(module)
        if name is not None:
            return name
        name = self._outsiders.get(module)
        if name is not None:
            return name
        with self._lock:
            # A module attached after entry (e.g. a lazily created
            # torch.compile wrapper): rescan the roots once for it.
            names = self._scan()
            name = names.get(module)
            if name is None:
                name = _class_name(module)
                self._outsiders[module] = name
            self._names = names
        return name

    def _pre_hook(self, module, args) -> None:
        if not self._stack and torch._C._current_autograd_node() is None:
            fingerprint = policy_summary()["fingerprint"]
            with self._lock:
                self._policies.add(fingerprint)
        name = self._name(module)
        scope = None
        if self._profiler is not None:
            # A profiler range attributes the aten ops of this forward.
            scope = torch._C._profiler._RecordFunctionFast(_SITE_EVENT + name)
            scope.__enter__()
        self._stack.append((module, name, scope))

    def _post_hook(self, module, args, output) -> None:
        stack = self._stack
        # Pop this module's innermost frame (and anything an exception left
        # above it); a module whose pre-hook never ran is ignored.
        for index in range(len(stack) - 1, -1, -1):
            if stack[index][0] is module:
                popped = stack[index:]
                del stack[index:]
                for _, _, scope in reversed(popped):
                    if scope is not None:
                        scope.__exit__(None, None, None)
                return

    def _site(self) -> str:
        top = self._stack[-1:]
        return top[0][1] if top else OUTSIDE

    def site(self) -> str:
        """Innermost module site of the calling forward, or ``"<outside>"``."""
        return self._site()

    # ------------------------------------------------------------- records

    def _params(self, params) -> dict:
        if not params:
            return {}
        invalid = []

        def value(item):
            if isinstance(item, _SCALARS):
                return item
            if isinstance(item, (tuple, list)):
                return [value(element) for element in item]
            if isinstance(item, (torch.dtype, torch.device)):
                return str(item)
            invalid.append(item)
            return f"<{type(item).__name__}>"

        clean = {str(key): value(params[key]) for key in sorted(params, key=str)}
        if invalid:
            with self._lock:
                self._invalid_params += len(invalid)
        return clean

    def _resolve(self, phase: str, site):
        if phase not in PHASES:
            phase = str(phase)
        node = torch._C._current_autograd_node() if phase == "fwd" or site is None else None
        if phase == "fwd" and node is not None:
            phase = "recompute"
        if site is None:
            site = ("bwd:" + node.name()) if phase == "bwd" and node is not None else self._site()
        return phase, str(site)

    def record(self, op, impl, params=None, *, phase="fwd", site=None) -> None:
        """One executed kernel or implementation choice."""
        phase, site = self._resolve(phase, site)
        entry = {"impl": str(impl), "op": str(op), "params": self._params(params),
                 "phase": phase, "site": site}
        with self._lock:
            self._sections[-1]["entries"].append(entry)

    def fallback(self, op, impl, reason, params=None, *, phase="fwd") -> None:
        """A fused/cached implementation declined at run time."""
        phase, site = self._resolve(phase, None)
        entry = {"impl": str(impl), "op": str(op), "params": self._params(params),
                 "phase": phase, "reason": str(reason), "site": site}
        with self._lock:
            self._sections[-1]["fallbacks"].append(entry)

    def mark(self, label) -> None:
        """Start a new section (e.g. "cold", "warm", "capture", "replay")."""
        with self._lock:
            self._marked = True
            self._sections.append(_new_section(str(label)))
            index = len(self._sections) - 1
            if self._profiler is not None:
                # The profiler's own event orders this mark against the aten
                # ops of every observed thread; the wall clock is the fallback
                # for a mark made on a thread the profiler does not observe.
                self._mark_times.append((index, time.time_ns()))
                with torch._C._profiler._RecordFunctionFast(f"{_MARK_EVENT}{index}"):
                    pass

    def _collect(self, events) -> None:
        """Profiler events -> per-section dispatcher op counts by context."""
        marks = dict(self._mark_times)
        threads: dict = {}
        for event in events:
            name = event.name()
            if name.startswith(_MARK_EVENT):
                marks[int(name[len(_MARK_EVENT):])] = event.start_ns()
            elif name.startswith((_SITE_EVENT, _NODE_EVENT)) or _OP_EVENT.match(name):
                start = event.start_ns()
                threads.setdefault(event.start_thread_id(), []).append(
                    (start, -event.duration_ns(), start + event.duration_ns(), name))
        # Section k (k >= 1) starts at mark k; ops before the first mark belong
        # to the initial section. Marks are sequential: keep their times sorted
        # even if a wall-clock fallback is slightly off the profiler's clock.
        order = sorted(marks)
        times = list(accumulate((marks[index] for index in order), max))
        with self._lock:
            sections = self._sections
            for rows in threads.values():
                rows.sort()
                open_ranges = []  # (end, context) of enclosing site/node ranges
                for start, _, end, name in rows:
                    while open_ranges and open_ranges[-1][0] <= start:
                        open_ranges.pop()
                    if name.startswith(_SITE_EVENT):
                        open_ranges.append((end, name[len(_SITE_EVENT):]))
                        continue
                    if name.startswith(_NODE_EVENT):
                        open_ranges.append((end, "bwd:" + name[len(_NODE_EVENT):]))
                        continue
                    context = OUTSIDE
                    for _, label in reversed(open_ranges):
                        if label.startswith("bwd:"):
                            context = label
                            break
                        if context == OUTSIDE:
                            context = label
                    position = bisect_right(times, start)
                    section = sections[order[position - 1] if position else 0]
                    counts = section["aten"].setdefault(context, {})
                    counts[name] = counts.get(name, 0) + 1

    # ------------------------------------------------------------ snapshot

    def snapshot(self) -> dict:
        """Deterministic JSON-serializable record (sorted keys, no ids/timings).

        Aten counts are complete once the recorder has exited.
        """
        with self._lock:
            sections = self._sections
            if (self._marked and len(sections) > 1 and not sections[0]["entries"]
                    and not sections[0]["fallbacks"] and not sections[0]["aten"]):
                sections = sections[1:]
            result = []
            for section in sections:
                entry_counts, fallback_counts = {}, {}
                for entry in section["entries"]:
                    key = f"{entry['op']}/{entry['impl']}/{entry['phase']}"
                    entry_counts[key] = entry_counts.get(key, 0) + 1
                for entry in section["fallbacks"]:
                    key = f"{entry['op']}/{entry['impl']}/{entry['reason']}"
                    fallback_counts[key] = fallback_counts.get(key, 0) + 1
                result.append({
                    "aten_counts": {context: dict(sorted(counts.items()))
                                    for context, counts in sorted(section["aten"].items())},
                    "entries": section["entries"],
                    "entry_counts": dict(sorted(entry_counts.items())),
                    "fallback_counts": dict(sorted(fallback_counts.items())),
                    "fallbacks": section["fallbacks"],
                    "label": section["label"],
                })
            snapshot = {"aten": self.aten, "invalid_params": self._invalid_params,
                        "policy_fingerprints": sorted(self._policies),
                        "schema": SCHEMA, "sections": result}
            # Detached deep copy: callers may mutate it freely.
            return json.loads(json.dumps(snapshot))
