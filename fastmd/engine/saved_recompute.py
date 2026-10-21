"""Selective recompute of saved activations for inference (``InferenceConfig.saved_tensor_recompute``).

Inside a recompute scope (inference only; ``MatRIS._forward`` opens one
around the interaction blocks) model code registers tensors that are cheap to
rebuild together with a recipe: a function that recomputes them from narrow
inputs with the same kernels. While the scope's ``saved_tensors_hooks`` are
active, autograd saves a handle instead of a registered storage. The backward
rebuilds that storage once, on the first unpack of any of its handles, and
keeps it until each handle of the recipe was unpacked. Other tensors are
saved as usual.

Levels: "on" (internal "narrow") keeps the first-layer outputs (the
pre-activations) of the line and atom layers, which the other recipes start
from. "deep" rebuilds those too: the line attention's gather-add preact by
re-launching its epilogue on a rebuilt projection, and the first layers on a
torch gather-concatenation by parts (``gather_cat_parts``: a narrow GEMM on
the aligned rows plus the node projections gathered and added), plus the
atom envelope.

Invariants:

* Forward values are untouched: a recipe runs only in the backward, under
  ``no_grad``, so energies are bitwise those of the off mode (the chunked
  refinement first layer runs the same GEMM per row chunk; tested). The
  recipes need not rebuild bitwise values: the line tails use 3xTF32 dots,
  the alpha projection one N=2*dim GEMM and deep's first layers their parts,
  so forces agree with the off mode within the noise floor, not bitwise.
* Plain saves are packed detached with their version and checked at unpack,
  as autograd does without hooks.
* A recipe holds only its narrow inputs. An input that is itself registered
  is passed as a ``Ref`` (weak until needed), never as a strong reference,
  which would keep the tensor alive and save nothing.
* A registration is used only while one registered tensor object is alive
  (so its storage cannot have been reused) and its version is unchanged;
  otherwise the tensor is saved as usual. A recipe input changed in place
  after registration is an error at recompute.
* Recipes are local (no collective; ``check_no_collective`` in the P>1
  exchange and communicator entry points raises inside one). They run on the
  autograd device thread while the caller waits in the backward, inside the
  entry point's configuration scope (module-global options, as a checkpoint
  recompute), and record nothing.
* Model code reaches a scope only through ``active()`` (then its methods),
  so the off path, also under torch.compile, calls nothing new.
* Inference only: the recomputed tensors carry no autograd history, so a
  backward with ``create_graph=True`` must not run in a scope (training uses
  ``plain_torch_scope`` and opens none).
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import weakref

import torch
from torch import Tensor

# Internal recipe sets: "narrow" is the public "on"; "deep" also rebuilds the
# first-layer outputs of the line attention and the atom layers.
MODES = ("off", "narrow", "deep")
_LOG = logging.getLogger(__name__)
# The scope of the running forward; a module global like the fusion scope, so
# that recipes registered in a CUDA-graph capture or a P>1 stage see it.
_ACTIVE: "RecomputeScope | None" = None
# Recipes running now (the backward); collectives must not run inside one.
_RECOMPUTING = 0


def check_no_collective(what: str) -> None:
    """Raise if a recipe runs: a P>1 recompute must stay rank-local (design §5.6)."""
    if _RECOMPUTING:
        raise RuntimeError(f"saved_recompute: collective {what!r} inside a recompute")


# Rows per chunk of a wide first-layer concatenation built in a scope (0:
# whole); 2^20 measured 7% faster than 2^18 at 5,376 atoms, same peak.
CHUNK_ROWS = 1 << 20


def chunk_rows(rows: int) -> int:
    """The chunk of a wide gather-concatenation of ``rows`` rows (0: unchunked).

    Balanced: ceil(rows / CHUNK_ROWS) chunks of equal size (a multiple of
    256), so that no chunk is a small tail GEMM that cuBLAS could run with
    another kernel (the chunked forward is bitwise only while every chunk
    takes the full GEMM's kernel)."""
    if active() is None or not 0 < CHUNK_ROWS < rows:
        return 0
    return balanced_chunk(rows, CHUNK_ROWS)


def balanced_chunk(rows: int, limit: int) -> int:
    """ceil(rows / ceil(rows / limit)) rounded up to a multiple of 256."""
    count = -(-rows // limit)
    return -(-(-(-rows // count)) // 256) * 256


def active() -> "RecomputeScope | None":
    """The open scope, or None (also while torch.compile traces)."""
    scope = _ACTIVE
    if scope is None or torch.compiler.is_compiling():
        return None
    return scope


class _Recipe:
    __slots__ = ("fn", "inputs", "versions", "layouts", "same_layout", "values", "pending",
                 "__weakref__")

    def __init__(self, fn, inputs, layouts, same_layout):
        self.fn = fn
        self.inputs = inputs
        self.versions = tuple(x._version if isinstance(x, Tensor) else None for x in inputs)
        self.layouts = layouts
        self.same_layout = same_layout
        self.values = None
        self.pending = 0

    def materialize(self) -> tuple[Tensor, ...]:
        if self.values is None:
            args = []
            for value, version in zip(self.inputs, self.versions):
                if isinstance(value, Ref):
                    value = value.get()
                elif isinstance(value, Tensor) and value._version != version:
                    raise RuntimeError("saved_recompute: a recipe input was modified in place "
                                       "after registration")
                args.append(value)
            from fastmd.engine import recorder as _recorder
            global _RECOMPUTING
            recorder, _recorder._ACTIVE = _recorder._ACTIVE, None
            _RECOMPUTING += 1
            try:
                with torch.no_grad():
                    out = self.fn(*args)
            finally:
                _RECOMPUTING -= 1
                _recorder._ACTIVE = recorder
            out = tuple(out) if isinstance(out, (tuple, list)) else (out,)
            for value in self.inputs:
                # An input recipe with no handle of its own was rebuilt only for this one.
                if isinstance(value, Ref) and value.recipe.pending <= 0:
                    value.recipe.values = None
            layouts = tuple(_layout(t) if self.same_layout else _layout(t)[:1] + _layout(t)[3:4]
                            for t in out)
            if layouts != self.layouts:
                raise RuntimeError("saved_recompute: a recipe returned tensors of another layout "
                                   f"({layouts} != {self.layouts})")
            self.values = out
        return self.values


class _View:
    """Where a saved tensor sits in its recipe's outputs: ``exact`` when it is
    the registered output itself (any recomputed layout), else a view of its
    storage (the recipe then rebuilds the same storage layout)."""
    __slots__ = ("recipe", "index", "exact", "size", "stride", "offset")

    def __init__(self, recipe, index, exact, tensor):
        self.recipe, self.index, self.exact = recipe, index, exact
        self.size, self.stride, self.offset = tensor.size(), tensor.stride(), tensor.storage_offset()

    def value(self) -> Tensor:
        base = self.recipe.materialize()[self.index]
        return base if self.exact else base.as_strided(self.size, self.stride, self.offset)


class Ref(_View):
    """A recipe input that is itself registered: the live tensor while it
    exists unchanged, else its recomputed value."""
    __slots__ = ("tensor", "version")

    def __init__(self, recipe, index, exact, tensor):
        super().__init__(recipe, index, exact, tensor)
        self.tensor, self.version = weakref.ref(tensor), tensor._version

    def get(self) -> Tensor:
        tensor = self.tensor()
        return tensor if tensor is not None and tensor._version == self.version else self.value()


class _Handle(_View):
    """A saved registered tensor: the tensor itself if something else keeps
    it alive (e.g. a P>1 stage output), else its recomputed value."""
    __slots__ = ("tensor", "version")

    def __init__(self, recipe, index, exact, tensor):
        super().__init__(recipe, index, exact, tensor)
        self.tensor, self.version = weakref.ref(tensor), tensor._version


def silu_halves_source(core, gate):
    """The preact behind two Refs to the Cat3/Cat4 epilogue activations, else None."""
    if not (isinstance(core, Ref) and isinstance(gate, Ref) and core.recipe is gate.recipe
            and (core.index, gate.index) == (0, 1) and core.exact and gate.exact):
        return None
    if getattr(core.recipe.fn, "__name__", "") != "_silu_halves":
        return None
    preact = core.recipe.inputs[0]
    return None if isinstance(preact, Ref) else preact  # deep: the preact is rebuilt itself


def _gather_cat3_rows(aligned: Tensor, nodes: Tensor, target: Tensor, source: Tensor) -> Tensor:
    return torch.cat([aligned, torch.index_select(nodes, 0, target), torch.index_select(nodes, 0, source)], dim=1)


def gather_cat_parts(source):
    """The recipe inputs (aligned, nodes, target, source) behind a Ref to a
    registered gather-concatenation, else None."""
    if isinstance(source, Ref) and source.exact and source.recipe.fn is _gather_cat3_rows:
        return source.recipe.inputs
    return None


def _int64_contiguous(index: Tensor) -> Tensor:
    return index.to(torch.int64).contiguous()


def _contiguous(tensor: Tensor) -> Tensor:
    return tensor.contiguous()


def register_copy(copy: Tensor, source: Tensor, *, int64: bool = False) -> None:
    """Register an op's own converted copy of an index (``source.contiguous()``,
    with ``int64``: ``source.to(int64).contiguous()``) that it saves: the
    source is kept anyway, the copy is rebuilt in the backward."""
    scope = active()
    if (scope is not None and copy is not source
            and copy.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()):
        scope.register((copy,), _int64_contiguous if int64 else _contiguous, source)


def _layout(tensor: Tensor):
    return (tuple(tensor.shape), tensor.stride(), tensor.storage_offset(), tensor.dtype,
            tensor.untyped_storage().nbytes())


class RecomputeScope:
    """Registrations of one forward; see the module docstring."""

    def __init__(self, mode: str = "narrow"):
        self.mode = mode
        # storage data_ptr -> [[recipe, [(index, size, stride, offset, weakref)], version, nbytes], ...]
        # (several recipes may cover views of one storage, e.g. a projection's
        # alpha and p0 columns)
        self._entries: dict[int, list] = {}
        self.registered = 0
        self.packed = 0

    def ref(self, tensor):
        """``tensor`` as a recipe input: a Ref if it is registered, else itself."""
        found = self._find(tensor)
        return tensor if found is None else Ref(*found, tensor)

    def register(self, outputs, fn, *inputs, same_layout: bool = True) -> None:
        """``fn(*inputs)`` recomputes ``outputs``. Inputs are tensors (held),
        Refs from ``ref`` (weak) or other values. ``same_layout``: the results
        have the outputs' shapes, strides, storage offsets and storage sizes
        (views of one storage share one), so that any view of a registered
        storage can be rebuilt; otherwise only the registered tensors
        themselves are (same shape and dtype, any layout)."""
        outputs = tuple(outputs)
        layouts = tuple(_layout(t) if same_layout else _layout(t)[:1] + _layout(t)[3:4] for t in outputs)
        recipe = _Recipe(fn, tuple(inputs), layouts, same_layout)
        entries = self._entries
        for index, tensor in enumerate(outputs):
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            if key == 0:
                continue
            bucket = entries.setdefault(key, [])
            entry = next((e for e in bucket if e[0] is recipe), None)
            if entry is None:
                entry = [recipe, [], tensor._version, storage.nbytes()]
                bucket.append(entry)
            views = entry[1]

            def forget(ref, key=key, entry=entry):
                entry[1][:] = [view for view in entry[1] if view[4] is not ref]
                bucket = entries.get(key)
                if not entry[1] and bucket is not None and entry in bucket:
                    bucket.remove(entry)
                    if not bucket:
                        del entries[key]

            views.append((index, tensor.size(), tensor.stride(), tensor.storage_offset(),
                          weakref.ref(tensor, forget)))
        self.registered += len(outputs)

    def register_product(self, product: Tensor, left: Tensor, right: Tensor) -> None:
        """Register ``product = left * right`` (an elementwise product that a
        reduction saves) when a factor is rebuilt itself (the other one is
        then kept anyway: the multiply saved it)."""
        a, b = self.ref(left), self.ref(right)
        if a is not left or b is not right:
            self.register((product,), torch.mul, a, b)

    def register_gather_cat(self, cat: Tensor, aligned: Tensor, nodes: Tensor, target: Tensor,
                            source: Tensor) -> None:
        """deep: register a torch gather-concatenation ``cat([aligned,
        nodes[target], nodes[source]], 1)`` that feeds a GatedMLP, which then
        rebuilds its first-layer outputs from the parts (``gather_cat_parts``)."""
        if self.mode == "deep":
            self.register((cat,), _gather_cat3_rows, self.ref(aligned), nodes, target, source)

    def _find(self, tensor):
        """(recipe, index, exact) of a registered tensor or storage view, else None."""
        if not isinstance(tensor, Tensor):
            return None
        bucket = self._entries.get(tensor.untyped_storage().data_ptr())
        if not bucket:
            return None
        where = (tensor.size(), tensor.stride(), tensor.storage_offset())
        fallback = None
        for entry in bucket:
            if not any(view[4]() is not None for view in entry[1]):
                continue
            if tensor._version != entry[2] or tensor.untyped_storage().nbytes() != entry[3]:
                continue
            for index, size, stride, offset, _ in entry[1]:
                if (size, stride, offset) == where:
                    return entry[0], index, True
            if entry[0].same_layout and fallback is None:
                fallback = entry[0], entry[1][0][0], False
        return fallback

    def pack(self, tensor):
        found = self._find(tensor)
        if found is None:
            # A plain save, detached (a Function saving its own output would
            # otherwise hold a cycle through its grad_fn) with its version, so
            # that an in-place write after the save still fails the backward.
            return tensor.detach(), tensor._version
        found[0].pending += 1
        self.packed += 1
        _STATS["packed"] += 1
        return _Handle(*found, tensor)


def _unpack(saved):
    if isinstance(saved, tuple):
        tensor, version = saved
        if tensor._version != version:
            raise RuntimeError(
                "one of the variables needed for gradient computation has been modified by an "
                f"inplace operation (saved at version {version}, now {tensor._version}; "
                "saved_recompute scope)")
        return tensor
    if not isinstance(saved, _Handle):
        return saved
    if torch.is_grad_enabled():
        # The rebuilt tensors carry no history: a create_graph backward would
        # silently drop second-order terms.
        raise RuntimeError("saved_recompute: a backward with create_graph=True reached a tensor "
                           "saved under MATRIS_SAVED_RECOMPUTE (inference only)")
    recipe = saved.recipe
    live = saved.tensor()
    if live is not None and live._version == saved.version:
        value = live
    else:
        value = saved.value()
        _STATS["rebuilt"] += 1
    _STATS["unpacked"] += 1
    recipe.pending -= 1
    if recipe.pending <= 0:
        # Every handle was unpacked: release the rebuilt storage (a later
        # Ref.get() of a dependent recipe recomputes it).
        recipe.values = None
    return value


# Process counters of handles (tests: every packed handle is unpacked once).
_STATS = {"packed": 0, "unpacked": 0, "rebuilt": 0}


def stats() -> dict[str, int]:
    return dict(_STATS)


def mode_for(model, triplet_rows: int = 0) -> str:
    """The recompute mode of a P=1 forward: the active entry point's
    ``saved_tensor_recompute`` ("auto" decided here, see ``AutoRecompute``),
    else the model's ``saved_recompute`` (set by the P>1 evaluator)."""
    from fastmd.config import current_config
    config = current_config()
    if config is None:
        mode = getattr(model, "saved_recompute", "off")
        return current_distributed_mode() if mode == "off" else mode
    requested = config.saved_tensor_recompute
    bound = _BOUND_ADMISSION
    global _GATED_TAIL_AUTO
    if requested == "auto":
        if bound is not None:
            return bound.recompute  # an evaluator's collective admission decided
        state = _single_auto(model, config)
        mode = state.decide_local(model, config, triplet_rows)
        _GATED_TAIL_AUTO = state.gated_tail
        return mode
    mode = explicit_mode(requested)
    _GATED_TAIL_AUTO = True
    from fastmd.engine import memory_admission
    tail = memory_admission.gated_tail_active(config.kernel_options)
    if bound is None and (tail or memory_admission.single_bwd_mode(config.kernel_options) == "auto"):
        # An explicit level: only C3 auto and C2 are decided (the same rules).
        state = _single_auto(model, config)
        state.decide_local(model, config, triplet_rows, fixed=mode)
        _GATED_TAIL_AUTO = state.gated_tail
    return mode


# --- auto --------------------------------------------------------------------
# "auto" climbs the ladder off -> on -> deep: it uses the cheapest (fastest)
# level whose peak the admission model (fastmd.engine.memory_admission,
# K6) predicts within this share of the budget (device free + reserved -
# pre-run allocations - margin). K6 predicts eager peaks within +2..+15%
# above the measured ones and model_graph within +2..+47%, and at 98.5% of
# the device the allocator already failed once from fragmentation: 85% keeps
# every configuration that fits comfortably on "off" (bitwise today's path)
# and switches before the K6 rejection would.
AUTO_THRESHOLD = 0.85
# InferenceConfig.recompute_policy="fast" (the default) raises the share, so that "off" (and C3) run up
# to about where they fit in practice (FAST_THRESHOLD, by where the decision is made):
# - P=1 eager (the entry points' decide_local): 0.98, on allocator expandable segments wherever
#   "fast" keeps a cheaper level or C3 than "safe" (0.85) would. With the default caching allocator,
#   GPU-resident MD in eager fills the device with cached blocks as the triplet rows grow (reserved
#   147.5 -> 182.8 GiB at 18,816 atoms over 1,000 NHC steps, 19 allocator retries, allocated flat);
#   with expandable segments the reserved peak stays within 1.8 GiB of the allocated one (0 retries,
#   same speed; 21,504 atoms off+C3 from 3000 K: 167.6 GiB). Where "fast" does not relax a decision
#   the allocator stays unchanged, although the growth happens at any level (the 10M model at 1,344
#   atoms "on": 65 GiB allocated, reserved 87 -> 166-181.5 GiB, 0-1 retries, no OOM): expandable
#   segments remove it (66.5 GiB) but cost +0.5-1.1% per eager evaluation (RDX 10,752 atoms, the
#   10M model at 1,344; model_graph unchanged), so long eager MD of large models sets them itself.
# - P=1 model_graph: 0.95, allocator unchanged (a failed allocation inside a capture cannot be
#   recovered, and a capture's pool is not a K6 point of its own).
# - Both P=1 cases: an uncaptured evaluation (also the warmups before each capture) that leaves less
#   than the admission margin free while "fast" keeps a cheaper level falls back to "safe".
# - P>1 bond_partial on the topology epoch (capacity-padded, fixed shapes): 0.95, prediction only
#   (the ranks cannot retry; the decision stays collective).
# - Everything else: "safe" (whole_step, whose growth captures a new runner while the old pool lives;
#   the evaluator's single engine), and so is every decision whose predictions do not all come from a
#   measured table (memory_admission.MemoryModel.calibrated and entry_calibrated: the RDX
#   tables, including a model the derived bound places within them, or the model's own measured
#   table); a derived-only prediction keeps the memory-safe rules.
FAST_THRESHOLD = {"eager": 0.98, "model_graph": 0.95, "distributed_epoch": 0.95}


def policy_threshold(policy: str, kind: str) -> float:
    """The share of the budget the memory decisions may use under a recompute policy; ``kind``: a key
    of FAST_THRESHOLD (else "safe"'s share)."""
    return FAST_THRESHOLD.get(kind, AUTO_THRESHOLD) if policy == "fast" else AUTO_THRESHOLD


# Whether the process's CUDA caching allocator uses expandable segments (None: not decided yet).
_EXPANDABLE: bool | None = None


def ensure_expandable_segments() -> bool:
    """Switch the caching allocator to expandable segments for new segments (recompute_policy
    "fast" at P=1 eager, where it keeps a cheaper level than "safe"; never inside a capture), unless
    the allocator is configured (``PYTORCH_CUDA_ALLOC_CONF``): then whether it already asks for them.
    False in a P>1 process, with another allocator backend, or where the running torch cannot switch
    at run time."""
    global _EXPANDABLE
    if _EXPANDABLE is None:
        getter = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
        conf = getter() if getter is not None else ",".join(
            os.environ.get(name, "") for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"))
        conf = conf.replace(" ", "").lower()
        if conf.strip(","):
            _EXPANDABLE = "expandable_segments:true" in conf
        elif ((torch.distributed.is_available() and torch.distributed.is_initialized()
               and torch.distributed.get_world_size() > 1)
              or torch.cuda.memory.get_allocator_backend() != "native"):
            # Not in a P>1 process (ranks must not diverge) nor with another allocator backend.
            return False
        else:
            setting = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
            try:
                if setting is None:
                    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
                else:
                    setting("expandable_segments:True")
                _EXPANDABLE = True
                torch.cuda.empty_cache()  # the cached regular segments go back to the device
            except (AttributeError, RuntimeError):
                _EXPANDABLE = False
            if _EXPANDABLE:
                _LOG.warning("recompute_policy=fast: the CUDA caching allocator uses expandable segments from now "
                             "on in this process (PYTORCH_CUDA_ALLOC_CONF unset)")
    return _EXPANDABLE
# C3 (MATRIS_MERGED_PROJECTION_SINGLE_BWD=auto): the single-GEMM merged-projection
# backward holds more memory. The ladder picks its level from the predictions
# without it; the single-GEMM backward then runs only if the prediction with it
# fits the same share of the budget at that level. So it falls back before the
# ladder would climb: off+C3 -> off -> on+C3 -> on -> deep+C3 -> deep, and the
# levels (and the largest system per level) are those without C3. Within a level
# it never switches back on (like the ladder, which never steps down); a climb
# decides it again at the new level.
# The P=1 decision of the running forward (``auto_single_bwd``); off until a prediction
# admits it (no admission model, e.g. MATRIS_MEMORY_ADMISSION=off: off).
_SINGLE_BWD_AUTO = False
# Internal levels in the order auto tries them (public off, on, deep).
LADDER = ("off", "narrow", "deep")
_PUBLIC = {"off": "off", "narrow": "on", "deep": "deep"}


def auto_level(peaks: dict, budgets, strict: bool = False, threshold: float = AUTO_THRESHOLD) -> str:
    """The auto rule on gathered per-rank data (identical on every rank, so
    every rank decides the same): the first level of ``LADDER`` whose
    predicted peak (``peaks[level]``, one per rank) is within
    ``threshold`` of every rank's budget. An uncalibrated (None)
    prediction counts as fitting, so it never climbs; ``strict`` (a model
    other than K6's RDX calibration, see memory_admission.MemoryModel): it
    does not, and with no prediction at all the deepest level is taken. If none is, the level
    with the smallest worst-rank share of the budget (the "on" and "deep"
    predictions cross at small P>1 row counts, where the fixed part of
    "deep" dominates)."""
    levels = [level for level in LADDER if level in peaks]
    for level in levels:
        if all((peak is None and not strict) or (peak is not None and peak <= threshold * budget)
               for peak, budget in zip(peaks[level], budgets)):
            return level
    predicted = [level for level in levels if any(peak is not None for peak in peaks[level])]
    if not predicted or (strict and len(predicted) < len(levels)):
        return levels[-1]  # strict, a level without a prediction: the deepest level
    share = lambda level: max((peak / budget if budget > 0 else float("inf"))
                              for peak, budget in zip(peaks[level], budgets) if peak is not None)
    return min(predicted, key=share)


class AutoRecompute:
    """One entry point's (or evaluator's) auto state. Decides at the first
    evaluation and again only when the rows grow past the last decision (a
    capture/growth boundary under CUDA graphs: a replay runs no Python), and
    never steps down the ladder. Logs each decision once."""

    def __init__(self, label: str, threshold: float = AUTO_THRESHOLD, memory=None):
        self.label, self.mode, self.rows, self.baseline, self.budget = label, "off", -1, None, None
        # The model's calibration (memory_admission.memory_model; the P=1 state resolves it at its first
        # decision): outside every measured envelope C3 stays off; strict, an unpredicted level does not fit.
        self.memory = memory
        # Of the budget (policy_threshold); a decision on a prediction that is not from a measured table
        # (``update``'s ``calibrated``) uses AUTO_THRESHOLD.
        self.threshold = threshold
        # recompute_policy "fast" at P=1: whether the current decision is cheaper than "safe"'s (it then
        # runs on expandable segments and is checked after each evaluation), and whether it fell back.
        self.fast_only = False
        self.fast_fallback = False
        self.single_bwd = True  # C3 auto: the single-GEMM merged backward (see AUTO_THRESHOLD)
        self.c3 = False  # whether this state decides C3 (``update`` got ``peaks_single_bwd``)
        self.gated_tail = True  # C2's memory decision at ``mode`` (decide_local; see memory_admission)
        self.decisions = []

    def decide(self, peaks: dict, budgets, peaks_single_bwd: dict | None = None, fixed: str | None = None,
               threshold: float | None = None, strict: bool = False, calibrated: bool = True) -> tuple[str, bool]:
        """The (level, C3) that ``update`` would choose from this state at ``threshold``, without
        committing it."""
        threshold = self.threshold if threshold is None else threshold
        mode = fixed or max(auto_level(peaks, budgets, strict, threshold), self.mode, key=LADDER.index)
        # Unlike the ladder, an uncalibrated (None) prediction does not admit C3, nor does a model outside
        # every measured envelope.
        single_bwd = (self.single_bwd or mode != self.mode) and (peaks_single_bwd is None or (
            calibrated and all(
                peak is not None and peak <= threshold * budget
                for peak, budget in zip(peaks_single_bwd[mode], budgets))))
        return mode, single_bwd

    def update(self, rows: int, peaks: dict, budgets, event: str, peaks_single_bwd: dict | None = None,
               fixed: str | None = None, strict: bool | None = None, calibrated: bool | None = None,
               threshold: float | None = None) -> str:
        """``peaks``: level -> per-rank predicted peaks (bytes or None), without C3 when
        ``peaks_single_bwd`` (the same with C3) is given: then C3 is decided too. ``fixed``:
        an explicit level (only C3 is decided). ``threshold``: this decision's share (default the
        state's); not ``calibrated``, at most AUTO_THRESHOLD."""
        global _SINGLE_BWD_AUTO
        if rows <= self.rows or (fixed is None and peaks_single_bwd is None and self.mode == LADDER[-1]):
            if self.c3:
                _SINGLE_BWD_AUTO = self.single_bwd
            return fixed or self.mode
        if strict is None:
            strict = self.memory is not None and self.memory.strict
        if calibrated is None:
            calibrated = self.memory is None or self.memory.calibrated
        threshold = self.threshold if threshold is None else threshold
        if not calibrated:
            threshold = min(threshold, AUTO_THRESHOLD)  # "fast" only on predictions from a measured table
        mode, single_bwd = self.decide(peaks, budgets, peaks_single_bwd, fixed, threshold, strict, calibrated)
        gib = lambda values: [None if v is None else round(v / 2**30, 1) for v in values]
        record = {"event": event, "rows": rows, "mode": _PUBLIC[mode], "threshold": threshold,
                  **{f"predicted_{_PUBLIC[level]}_gib": gib(values) for level, values in peaks.items()},
                  "budget_gib": gib(budgets)}
        if peaks_single_bwd is not None:
            record["single_bwd"] = single_bwd
            record[f"predicted_{_PUBLIC[mode]}_single_bwd_gib"] = gib(peaks_single_bwd[mode])
        if mode != self.mode or single_bwd != self.single_bwd or not self.decisions:
            first = not self.decisions
            self.decisions.append(record)
            DECISIONS.append({"entry": self.label, **record})
            if fixed is None and (mode != self.mode or first):
                (_LOG.warning if mode != "off" else _LOG.info)(
                    "saved_tensor_recompute=auto -> %s (%s, %s): predicted %s GiB, budget %s GiB, threshold %.0f%%",
                    record["mode"], self.label, event,
                    ", ".join(f"{_PUBLIC[level]} {gib(values)}" for level, values in peaks.items()),
                    record["budget_gib"], 100 * threshold)
            if peaks_single_bwd is not None and not single_bwd:
                _LOG.info("merged-projection single-GEMM backward auto -> off (%s, %s): predicted %s GiB at %s, "
                          "budget %s GiB", self.label, event, gib(peaks_single_bwd[mode]), _PUBLIC[mode],
                          record["budget_gib"])
        self.mode, self.rows, self.single_bwd = mode, rows, single_bwd
        if peaks_single_bwd is not None:
            self.c3 = True
            _SINGLE_BWD_AUTO = single_bwd
        return mode

    def decide_local(self, model, config, rows: int, fixed: str | None = None) -> str:
        """P=1 entry points: predict from this device's budget (free + reserved, less the
        bytes allocated before the entry point was created, see ``note_entry_point``, and
        the margin). ``fixed``: an explicit level, for which only the C3 auto decision and
        C2's memory decision are made."""
        global _SINGLE_BWD_AUTO
        from fastmd.engine import memory_admission
        parameter = next(model.parameters(), None)
        device = None if parameter is None else parameter.device
        if self.fast_only and not torch.cuda.is_current_stream_capturing():
            # recompute_policy "fast" kept a cheaper level than "safe" would: the evaluations since must
            # have left the admission margin free (with expandable segments the reserved memory follows
            # the live peak), else fall back to "safe" and decide again now.
            free, total = torch.cuda.mem_get_info(device)
            if free < memory_admission.margin_bytes(total):
                self.fast_only, self.fast_fallback, self.threshold, self.rows = False, True, AUTO_THRESHOLD, -1
                record = {"event": "fast_fallback", "mode": _PUBLIC[self.mode], "free_gib": round(free / 2**30, 2),
                          "reserved_gib": round(torch.cuda.memory_reserved(device) / 2**30, 2),
                          "allocated_gib": round(torch.cuda.memory_allocated(device) / 2**30, 2)}
                self.decisions.append(record)
                DECISIONS.append({"entry": self.label, **record})
                _LOG.warning("recompute_policy=fast -> safe (%s): %.1f GiB free after an evaluation at %s, less than "
                             "the margin %.1f GiB", self.label, free / 2**30, _PUBLIC[self.mode],
                             memory_admission.margin_bytes(total) / 2**30)
        if rows <= self.rows:
            if self.c3:
                _SINGLE_BWD_AUTO = self.single_bwd
            return fixed or self.mode
        blocks = getattr(model, "interaction_block", ())
        c3 = memory_admission.single_bwd_mode(config.kernel_options)
        levels = LADDER if fixed is None else (fixed,)
        if (device is None or device.type != "cuda" or not memory_admission.enabled()
                or any(getattr(b, "enable_compile", False) or getattr(b, "enable_checkpoint", False) is not False
                       for b in blocks)):
            if fixed is not None and c3 != "auto":
                self.rows = rows
                return fixed
            none = {level: [None] for level in levels}
            return self.update(rows, none, [0], "no admission model (CPU, disabled or compile/checkpoint)",
                               peaks_single_bwd=none if c3 == "auto" else None, fixed=fixed, strict=False)
        execution = "eager" if config.execution == "eager" else "model_graph"
        mixed_width = memory_admission.mixed_width_active(config.kernel_options)
        tail = memory_admission.gated_tail_active(config.kernel_options)
        if self.memory is None:
            self.memory = memory_admission.memory_model(model)
            if self.memory is memory_admission.UNCALIBRATED_MEMORY_MODEL:
                _LOG.warning("memory admission (%s): no MatRIS architecture in the model's config (K6 uncalibrated): "
                             "no automatic choice relaxes memory", self.label)
        memory = self.memory
        predict = lambda level, single_bwd: memory_admission.predict(
            "single", config.optimization_profile, execution, rows, recompute=level, mixed_width=mixed_width,
            single_bwd=single_bwd, memory=memory)
        peaks = {level: [predict(level, c3 == "on")] for level in levels}
        peaks_single_bwd = {level: [predict(level, True)] for level in levels} if c3 == "auto" else None
        calibrated = memory.calibrated and all(memory_admission.entry_calibrated(
            memory, "single", config.optimization_profile, execution, level) for level in levels)
        if torch.cuda.is_current_stream_capturing():
            if self.budget is None:
                # No device query inside a capture, and nothing known yet: keep
                # off (today's path) and decide at the next uncaptured evaluation.
                _SINGLE_BWD_AUTO = self.single_bwd and self.c3
                return fixed or self.mode
            budget = self.budget  # the last uncaptured query's
        else:
            free, total = torch.cuda.mem_get_info(device)
            if self.baseline is None:
                self.baseline = _ENTRY_BASELINE.get(model)  # see note_entry_point
                if self.baseline is None:
                    self.baseline = torch.cuda.memory_allocated(device)
            budget = (free + torch.cuda.memory_reserved(device) - self.baseline
                      - memory_admission.margin_bytes(total))
            self.budget = budget
        if self.threshold > AUTO_THRESHOLD and calibrated and (fixed is None or c3 == "auto"):
            # recompute_policy "fast" (only on predictions from a measured table, see ``update``): where
            # it keeps a cheaper level (or C3) than "safe" would, the allocator must use expandable
            # segments, else "fast" decides as "safe" here.
            differs = (self.decide(peaks, [budget], peaks_single_bwd, fixed, strict=memory.strict)
                       != self.decide(peaks, [budget], peaks_single_bwd, fixed, AUTO_THRESHOLD, memory.strict))
            if differs and config.execution == "eager" and (torch.cuda.is_current_stream_capturing()
                                                             or not ensure_expandable_segments()):
                self.threshold, self.fast_fallback, differs = AUTO_THRESHOLD, True, False
                _LOG.info("recompute_policy=fast decides as safe (%s): the allocator does not use expandable "
                          "segments (PYTORCH_CUDA_ALLOC_CONF)", self.label)
            self.fast_only = differs
        if fixed is None or c3 == "auto":
            mode = self.update(rows, peaks, [budget], f"{execution}, {config.optimization_profile}",
                               peaks_single_bwd=peaks_single_bwd, fixed=fixed, calibrated=calibrated)
        else:
            mode, self.rows = fixed, rows
        # C2 after the level (and C3), from the prediction with its increment (the ladder read them without).
        gated_tail = not tail or memory_admission.gated_tail_fits(
            "single", config.optimization_profile, execution, rows, 0, recompute=mode, mixed_width=mixed_width,
            budget=budget, single_bwd=c3 == "on" or (c3 == "auto" and self.single_bwd), memory=memory,
            threshold=self.threshold if calibrated else AUTO_THRESHOLD)
        if gated_tail != self.gated_tail:
            _LOG.info("fused narrow tail (C2) %s (%s, %s, recompute %s): %d rows, budget %.1f GiB",
                      "on" if gated_tail else "off by memory", self.label, execution, _PUBLIC[mode], rows,
                      budget / 2**30)
        self.gated_tail = gated_tail
        return mode


# Every auto decision of the process (tests, reports).
DECISIONS: list[dict] = []
# Device bytes allocated when a P=1 entry point was created (``note_entry_point``), per model: the
# baseline of its decisions. The K6 peaks are process reserved peaks that include the entry point's
# own graph and state, so only what was allocated before it (the model, the caller's tensors) comes
# off the budget, as the evaluator's admission takes off what was allocated before its first
# evaluation. Without a note (a forward under a configuration scope alone), the allocations at the
# first decision come off, which include the graph (conservative).
_ENTRY_BASELINE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def entry_baseline(model, device) -> int:
    """The bytes a P=1 entry point's memory decisions take off the budget: those allocated when it was
    created (``note_entry_point``), else those allocated now."""
    baseline = _ENTRY_BASELINE.get(model)
    return torch.cuda.memory_allocated(device) if baseline is None else baseline


def note_entry_point(model) -> None:
    """Record the baseline of a P=1 entry point's decisions; called when it is created, before
    its graph and state exist."""
    parameter = next(iter(model.parameters()), None)
    if isinstance(parameter, torch.Tensor) and parameter.is_cuda:
        _ENTRY_BASELINE[model] = torch.cuda.memory_allocated(parameter.device)


_SINGLE: dict[int, AutoRecompute] = {}
# An evaluator's MemoryAdmission whose collective decision a forward uses.
_BOUND_ADMISSION = None
# C2's memory decision of the running P=1 forward (``mode_for``; see ``gated_tail_allowed``).
_GATED_TAIL_AUTO = True


def gated_tail_allowed() -> bool:
    """Whether the running forward may use the fused GatedMLP narrow tail (C2), by memory: the bound
    evaluator admission's decision, else the P=1 entry point's state at its level, auto or explicit
    (no admission model: yes). See memory_admission.GATED_TAIL_GIB_PER_MILLION_ROWS."""
    bound = _BOUND_ADMISSION
    return _GATED_TAIL_AUTO if bound is None else bound.gated_tail_decision


def _single_auto(model, config) -> AutoRecompute:
    # Per model and configuration (an entry point's runner may hold copies of
    # its ResolvedConfig): its warmups, captures and eager calls share a state.
    key = (id(model), config.fingerprint)
    state = _SINGLE.get(key)
    if state is None:
        # model_graph needs a second warmup: the free check of the fast policy then runs before the capture
        # ("fast" relaxes a decision only where its predictions come from a measured table, see decide_local).
        kind = "safe" if config.execution == "model_graph" and config.capacity.warmup < 2 else config.execution
        state = _SINGLE[key] = AutoRecompute(
            f"config {config.fingerprint}", policy_threshold(getattr(config, "recompute_policy", "safe"), kind))
        weakref.finalize(model, _SINGLE.pop, key, None)
    return state


def current_mode(model, config) -> str:
    """The recompute level the forwards of ``config``'s P=1 entry point use now, without deciding:
    a bound evaluator's, an explicit level, else the entry point's ``AutoRecompute`` state ("off"
    before its first decision)."""
    bound = _BOUND_ADMISSION
    if bound is not None:
        return bound.recompute
    requested = config.saved_tensor_recompute
    if requested != "auto":
        return explicit_mode(requested)
    state = _SINGLE.get((id(model), config.fingerprint))
    return "off" if state is None else state.mode


def auto_record(model, config) -> dict | None:
    """The auto state of ``config``'s P=1 entry point for reports (the engine plan record), without
    deciding: its level, C3 and C2 decisions, budget share and decision records; None before its
    first decision (or under a bound evaluator, whose admission decides)."""
    if _BOUND_ADMISSION is not None or config is None:
        return None
    state = _SINGLE.get((id(model), config.fingerprint))
    if state is None:
        return None
    return {"mode": _PUBLIC[state.mode], "single_bwd": state.single_bwd if state.c3 else None,
            "gated_tail": state.gated_tail, "threshold": state.threshold, "fast_only": state.fast_only,
            "fast_fallback": state.fast_fallback, "rows": state.rows,
            "decisions": [dict(record) for record in state.decisions]}


def capture_check_pending(model, config) -> bool:
    """Whether a P=1 capture of ``config`` needs one more uncaptured run first: recompute_policy "fast"
    keeps a cheaper level than "safe" would, and its free-memory check runs only at an uncaptured
    evaluation's decision (``AutoRecompute.decide_local``), so a later capture with a single warmup
    would otherwise go unchecked (``fastmd.engine.capacity_policy.CapturePolicy.warmups``)."""
    if _BOUND_ADMISSION is not None or config is None:
        return False
    # Also for an explicit level: its C3 auto decision keeps a "fast" share too.
    state = _SINGLE.get((id(model), config.fingerprint))
    return bool(state is not None and state.fast_only)


def auto_single_bwd() -> bool:
    """C3 auto: the decision for the running forward (a bound evaluator's collective
    admission, else the P=1 entry point's ``AutoRecompute``)."""
    bound = _BOUND_ADMISSION
    return _SINGLE_BWD_AUTO if bound is None else bound.single_bwd_decision


def current_distributed_mode() -> str:
    bound = _BOUND_ADMISSION
    return "off" if bound is None else bound.recompute


@contextmanager
def bound_admission(admission):
    """Bind an evaluator's MemoryAdmission (its ``recompute`` is the mode its
    collective auto decision chose, its ``gated_tail_decision`` C2's) for the
    forwards of one evaluation."""
    global _BOUND_ADMISSION
    previous, _BOUND_ADMISSION = _BOUND_ADMISSION, admission
    try:
        yield
    finally:
        _BOUND_ADMISSION = previous


def internal_mode(requested: str) -> str:
    """The model attribute / admission key of a public mode ("on" -> "narrow")."""
    return {"on": "narrow", "deep": "deep", "off": "off", "auto": "auto"}[requested]


def explicit_mode(requested: str | None) -> str:
    """The internal mode of an explicit public value; "auto"/None -> "off"."""
    return {"on": "narrow", "deep": "deep"}.get(requested, "off")


def check_model(model, mode: str, *, runtime: bool = False) -> None:
    """Reject model settings a recompute scope cannot serve (modes on/deep only)."""
    if mode not in MODES:
        raise ValueError(f"recompute must be one of {MODES}, got {mode!r}")
    if mode == "off" or runtime:
        # A runtime (P>1) path never checkpoints or compiles its layers.
        return
    blocks = getattr(model, "interaction_block", ())
    if any(getattr(block, "enable_compile", False) for block in blocks):
        raise ValueError("recompute requires enable_compile=False "
                         "(recipes are not registered inside compiled layers)")
    if any(getattr(block, "enable_checkpoint", False) is not False for block in blocks):
        raise ValueError("recompute requires checkpoint='off' "
                         "(checkpointed layers save through their own hooks)")


@contextmanager
def forward_scope(mode: str = "off"):
    """Open a recompute scope for this (inference) forward when ``mode`` is on."""
    global _ACTIVE
    if mode == "off" or _ACTIVE is not None or torch.compiler.is_compiling():
        yield None
        return
    scope = RecomputeScope(mode)
    _ACTIVE = scope
    try:
        with torch.autograd.graph.saved_tensors_hooks(scope.pack, _unpack):
            yield scope
    finally:
        _ACTIVE = None
