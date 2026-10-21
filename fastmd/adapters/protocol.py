"""The adapter protocol: what the single-GPU runtime reads from a model.

The single-GPU runtime (``fastmd.interfaces`` and ``fastmd.execution``:
GPU-resident MD, the calculator, the CUDA-graph runners, torch-sim) reaches a
model through an adapter. A :class:`ModelAdapter` names the module and
declares the graph the model consumes (:class:`TopologySpec`), what happens to
isolated atoms and whether the module runs compiled code. Capabilities a model
may or may not have are separate protocols, discovered with ``isinstance``
(which checks member names only):

* :class:`ModelOwnedDerivative`: the model computes forces and stress with its
  own autograd and returns them from its own call (MatRIS on one GPU).
* :class:`EnergyTap`: the model returns per-atom energies and the engine
  differentiates them (one backward pass for forces and stress).
* :class:`SystemEnergyTap`: the model returns the energy of the whole system
  (no per-atom energies) and the engine differentiates it the same way.

The capability negotiation (:mod:`fastmd.adapters.negotiate`) reads further
declarations, each optional: :class:`CaptureDeclaration` (the
:class:`CaptureContract`), :class:`KernelBackends` (:class:`KernelChoice`),
:class:`RecomputeParticipant` and
:class:`SystemInputs`. A model without one gets the conservative choice.
The distributed runtime runs a model at P>1 through the exchange strategy its
:class:`ShardDeclaration` provides, or through its generic node-halo exchange
for a model that declares a :class:`ShardSpec` (:class:`ShardSpecDeclaration`;
with :class:`ShardedTap` where the adapter runs its own layer loop and pooled
readout); replicas of a system run as one batch for a model that declares
:class:`SystemBatching`, different systems for one that also takes the engine's
per-edge shifts (:class:`EdgeShiftInput`), declares ``"union"`` and has no line
graph. A
model the runtime plans by rules of its own (:class:`RuntimePlanned`: MatRIS)
is not negotiated; its decisions are recorded. A request a model cannot serve
raises :class:`NegotiationError`.

``docs/model_adapters.md`` describes the layer and what comes next.
"""
from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from torch import Tensor, nn


@dataclass(frozen=True)
class TopologySpec:
    """The graph a model consumes (GPU-resident MD's graph builders and the calculator's skin-reuse runner
    take their cutoffs here)."""

    cutoff: float  # pair cutoff rc in Å: directed edges i -> j with d_ij < rc, both directions
    num_interactions: int  # message-passing layers; the receptive field is num_interactions * cutoff
    triplet_cutoff: float | None = None  # bond pairs at a centre atom within this cutoff; None: edges only
    undirected_pairs: bool = False  # the model reads the undirected pair of every directed edge
    model_dtype: Literal["float32", "float64"] = "float32"  # the dtype of the model's parameters
    # The order of the real edge rows the model needs: "sorted" by (centre, neighbour, image), the same
    # rows whether a candidate list is reused or fresh; or "by_centre", grouped by centre in ascending
    # order (the builder skips its per-build sort; a model whose sums are order-free up to rounding).
    edge_order: Literal["sorted", "by_centre"] = "sorted"
    # The most neighbours per atom the model was built for (a kNN cap: its own graph keeps the nearest
    # ones); None: every pair within the cutoff. The engine still lists every pair within the cutoff;
    # GPU-resident MD reports a build in which an atom had more neighbours than this (from a device maximum).
    max_neighbors: int | None = None


@dataclass(frozen=True)
class PaddingRule:
    """What the padded rows of a fixed-capacity edge list look like to the model.

    A captured graph replays a fixed number of edge rows; the rows past the
    real edges are padding, which must add exactly nothing to the energy and
    the forces. The adapter declares the form its model needs; the builders
    write it:

    * ``"sink_atoms"``: self-loops of extra sink atoms (rows ``n_real`` and
      up, atomic number 1) with image (3, 3, 3), spread over the sinks; the
      model must leave the sink rows out (MatRIS's ``n_real``).
    * ``"far_self_loops"``: padded row ``k`` is a self-loop of real atom
      ``k mod n_real`` with an integer image along the longest lattice vector,
      at least ``min_length`` and at least twice the cutoff long (a model
      with an exact cutoff sees nothing; a length of 0 would be NaN for a
      Bessel basis). With a line graph, padded pair ``p`` is the two
      opposite self-loops (images ``+m`` and ``-m``) of real atom
      ``p mod n_real``, and a padded triplet joins a padded pair to itself at
      that atom: no sink rows, for a model whose edge and triplet weights
      vanish beyond its cutoffs (a padded row then adds exactly zero to the
      real rows, in value and gradient).
    * ``"exact_rc_loops"``: padded row ``k`` is a self-loop of real atom
      ``k mod n_real`` with image 0 whose edge vector the model takes from
      ``GraphBatch.edge_vec_override``: ``pad_vector``, by default
      ``(cutoff, 0, 0)`` (exactly at the cutoff, which an integer image
      cannot express).

    Padding is spread over many rows, never all on one row (atomics into one
    row serialize): self-loops go round robin over the real atoms, sink
    loops in consecutive pairs over the sinks.
    """

    kind: Literal["sink_atoms", "far_self_loops", "exact_rc_loops"] = "far_self_loops"
    min_length: float = 0.0  # far_self_loops: the shortest padded edge, in Å (at least twice the cutoff anyway)
    pad_vector: tuple[float, float, float] | None = None  # exact_rc_loops: the padded rows' edge vector, in Å
    placement: Literal["round_robin"] = "round_robin"

    def __post_init__(self) -> None:
        if self.kind not in ("sink_atoms", "far_self_loops", "exact_rc_loops"):
            raise ValueError(f"PaddingRule: unknown kind {self.kind!r}")
        if self.placement != "round_robin":
            raise ValueError(f"PaddingRule: unknown placement {self.placement!r}")
        if not math.isfinite(self.min_length) or self.min_length < 0.0:
            raise ValueError("PaddingRule: min_length must be finite and >= 0")
        if self.min_length and self.kind != "far_self_loops":
            raise ValueError("PaddingRule: min_length applies to far_self_loops only")
        if self.pad_vector is not None:
            if self.kind != "exact_rc_loops":
                raise ValueError("PaddingRule: pad_vector applies to exact_rc_loops only")
            if len(self.pad_vector) != 3 or not all(math.isfinite(value) for value in self.pad_vector):
                raise ValueError("PaddingRule: pad_vector must be three finite numbers")


@dataclass(frozen=True)
class SystemSegments:
    """The systems of a graph batch that holds several: system ``s`` is the atom rows
    ``[row_ptr[s], row_ptr[s + 1])``; no edge joins two systems. ``row_ptr`` and ``system_of_row`` are
    persistent device buffers (the builder rewrites them in place, never reallocates them, so a model's
    input may alias them like the batch's other tensors).

    Systems with cells of their own (``fastmd.execution.system_batch``) also give ``cells``, and their rows
    may be slots: system ``s`` is then the first ``atoms[s]`` rows of its slot ``[row_ptr[s], row_ptr[s + 1])``,
    whose other rows are vacant (no candidate and no real edge: only padding self-loops, which add nothing; their
    energies belong to no system). Both are persistent device buffers as well."""

    count: int  # systems
    row_ptr: Tensor  # [count + 1] int64 on the device: the first row of each system, then the row count
    system_of_row: Tensor  # [rows] int32 on the device: the system of each atom row (of a vacant row: its slot's)
    # [count, 3, 3] on the device, in the model's dtype: each system's cell (rows are its lattice vectors); None:
    # every system has the batch's cell
    cells: Tensor | None = None
    # [count] int64 on the device: each system's atoms, the first rows of its slot; None: every row is occupied
    atoms: Tensor | None = None


@dataclass
class GraphBatch:
    """The graph a builder publishes: engine-owned buffers the model reads.

    For a fixed-capacity builder every tensor is a persistent buffer whose
    address stays fixed while CUDA graphs that read it are captured and
    replayed; each build writes the current graph into them. A ragged
    builder publishes a new batch per build, without padding. Rows of the
    atom tensors are the real atoms, then ``padding`` sink rows if any.

    Edges are directed and listed in both directions; the real ones, rows
    ``[0, n_edges)``, are sorted by centre atom (then neighbour and image;
    for a topology with ``edge_order="by_centre"`` only grouped by centre
    atom), and the rows after them are padding laid out by ``padding`` (self-loops
    go round robin, so the whole edge list is not sorted). The vector of
    edge ``e`` from its centre to its neighbour is
    ``positions[edge_index[e, 1]] + edge_image[e] @ cell - positions[edge_index[e, 0]]``.
    The pair and triplet tensors exist only when the topology asks for them
    (``TopologySpec.triplet_cutoff``). A model that differentiates the
    energy itself may take ``positions.detach().requires_grad_(True)`` (a
    view, no copy) as its leaf, and strain ``positions``, ``cell`` and the
    shifts ``edge_image @ cell`` for a stress.

    A fixed-capacity builder with a line graph (MatRIS's topology) sets
    ``n_edges`` and ``status`` at each build; they are None before the first.
    """

    # [rows, 3] wrapped Cartesian coordinates in Å (fractional coordinates in [0, 1) times the cell);
    # None where the model reads frac_coords only (MatRIS)
    positions: Tensor | None
    frac_coords: Tensor  # [rows, 3] fractional coordinates in [0, 1)
    cell: Tensor  # [3, 3], rows are the lattice vectors, in Å
    species: Tensor  # [rows] int32 atomic numbers (sink rows: 1)
    n_real: int  # real atoms: rows [0, n_real)
    edge_index: Tensor  # [edge rows, 2] int32: (centre, neighbour); real rows sorted by centre
    edge_image: Tensor  # [edge rows, 3] float32 integer lattice shift of the neighbour
    # [1] int64 on the device: the real edges, rows [0, n_edges); the host does not read it. More than the
    # rows is an overflow: the rows are then not the edges (edges only: the first ones; with a line graph: an
    # all-padding substitute) and the runtime discards the evaluation
    n_edges: Tensor | None
    cutoff: float  # the edges are the pairs closer than this, in Å
    padding: PaddingRule | None  # the form of the rows past n_edges; None: a ragged batch has none
    # [edge rows, 3] (exact_rc_loops): the padded rows' edge vector; zero on real rows
    edge_vec_override: Tensor | None = None
    triplet_cutoff: float | None = None  # the bond pairs below are those within this cutoff; None: edges only
    pair_d2u: Tensor | None = None  # [edge rows] int32: the undirected pair of each directed edge
    pair_u2d: Tensor | None = None  # [edge rows / 2] int32: a directed edge of each undirected pair
    # [triplet rows, 5] int32 bond pairs at a centre: (centre, pair 1, edge 1, pair 2, edge 2)
    triplets: Tensor | None = None
    # A fixed-capacity builder's device flags, rewritten by every build: overflow, candidate validity, counts
    # (fastmd.graph.fixed_capacity_builder.FixedCapacityGraphStatus, without the undirected count: None);
    # None for a ragged batch and for the model-graph runner's padded batches
    status: Any = None
    composition: str = ""  # chemical formula, for reports
    evidence: Any = None  # the builder's topology-contract evidence (fastmd.graph.topology_contract)
    # The systems of a batch of several (a union of replicas: fastmd.execution.replica_batch); None for the batch
    # of a single-system builder
    systems: SystemSegments | None = None

    @property
    def edge_dst(self) -> Tensor:
        """The centre (receiving) atom of each edge: a view of ``edge_index[:, 0]``."""
        return self.edge_index[:, 0]

    @property
    def edge_src(self) -> Tensor:
        """The neighbour (sending) atom of each edge: a view of ``edge_index[:, 1]``."""
        return self.edge_index[:, 1]


@runtime_checkable
class ModelAdapter(Protocol):
    """A model behind the runtime: the module itself (never a copy) and what the runtime reads from it."""

    name: str
    module: nn.Module
    # Whether the module runs part of its forward under torch.compile; execution "auto" then stays eager.
    compiled: bool

    def topology_spec(self) -> TopologySpec:
        """The graph the model consumes."""
        ...

    def padding_rule(self) -> PaddingRule:
        """The form of the padded edge rows the model needs."""
        ...

    def prepare(self, batch: GraphBatch) -> Any:
        """The model's own input over a builder's ``batch``: a view of the same tensors (no copy).

        Called outside any CUDA graph, once per batch: a fixed-capacity
        builder calls it at construction, before its first build, and every
        build then rewrites the tensors behind it; a ragged build gets its
        own call. So it may only alias the batch's tensors: a conversion
        that depends on their values (another dtype or layout of the edge
        index, the shifts) belongs in the model's call, which is what a
        CUDA graph captures. The runtime passes the result to the model's
        call in place of the batch."""
        ...

    def isolated_atom_policy(self) -> Literal["reference", "error", "model"]:
        """What an atom without neighbours means: "reference", it keeps its element's reference
        energy; "error", the runtime rejects isolated atoms; "model", the model evaluates isolated
        atoms itself (the runtime admits them and substitutes nothing). Host-only (no tensor work)."""
        ...

    def reference_energies(self) -> Tensor | None:
        """The per-element reference energies as a ``[1, elements]`` row (column ``Z - 1``), read-only,
        or None without them."""
        ...


@runtime_checkable
class ModelOwnedDerivative(Protocol):
    """Capability: the model differentiates its own energy; one call returns energy, forces and stress.

    ``evaluate`` is the model's own call, so the runtime passes the model's own
    inputs; for MatRIS ``evaluate(graphs, task=..., is_training=False,
    n_real=..., handle_isolated_atoms=...)`` returns the mapping of
    ``MatRIS.forward`` (``"e"``, ``"f"``, ``"s"``, ``"atoms_per_graph"``, ...).
    It is not wrapped in an adapter method: the call keeps the model's frames.
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """The model's own call: its inputs in, its outputs out (energy, forces, stress)."""
        ...

    def extensive_energy(self, energy: Any, atom_counts: Any) -> Any:
        """The system energy from the energy ``evaluate`` returned for systems of ``atom_counts`` atoms."""
        ...


@runtime_checkable
class EnergyTap(Protocol):
    """Capability: the model's per-atom energies, which the engine differentiates itself.

    The engine-owned derivative (``fastmd.engine.energy_derivative``)
    takes ``batch.positions`` as an autograd leaf and, for a stress, a strain
    leaf that deforms the positions and the cell; ``node_energy`` evaluates
    the model at those (differentiable) positions and cell, and the engine
    sums the real rows in FP64 and runs one ``torch.autograd.grad``:
    F = -dE/dx and the stress dE/d(strain) / volume in eV/Å³. The model's
    call must not differentiate itself, synchronize the host, or keep the
    graph (``create_graph`` is False).
    """

    def node_energy(self, inputs: Any, positions: Tensor, cell: Tensor) -> Tensor:
        """The energy of each atom row (``[rows]``, extensive, any float dtype) of the model's input
        ``inputs`` (``prepare(batch)``), at ``positions`` (``[rows, 3]``, in place of
        ``batch.positions``) and ``cell`` (``[3, 3]``, in place of ``batch.cell``). An edge's vector is
        ``positions[neighbour] + edge_image @ cell - positions[centre]``."""
        ...


# --- Whole-system energy tap: models whose energy is not a sum of per-atom energies ------------------

@runtime_checkable
class SystemEnergyTap(Protocol):
    """Capability: the energy of the whole system, which the engine differentiates itself.

    The variant of :class:`EnergyTap` for a model without per-atom energies (a
    readout that pools the atoms' features before its energy head, or a mean
    over atoms): the engine-owned derivative evaluates ``system_energy`` at the
    same differentiable positions and cell, takes its FP64 value as the energy
    (no row mask, no row sum) and runs the same single ``torch.autograd.grad``.
    The same rules apply: the call must not differentiate itself, synchronize
    the host, or keep the graph. A model has one tap; :class:`EnergyTap` wins
    if both are present.
    """

    def system_energy(self, inputs: Any, positions: Tensor, cell: Tensor) -> Tensor:
        """The energy of the system (one element, ``[1]`` or a scalar, any float dtype) of the model's
        input ``inputs`` (``prepare(batch)``) at ``positions`` (``[rows, 3]``) and ``cell`` (``[3, 3]``),
        as for :meth:`EnergyTap.node_energy`; for a batch of several systems (``GraphBatch.systems``) of a
        model that declares ``"segments"`` (:class:`SystemBatching`), one energy per system (``[count]``)."""
        ...


# --- Adapter-declared capacity step: the edge-capacity tiers of a model's captured graphs -------------

@dataclass(frozen=True)
class CapacityStep:
    """How a model's captured graphs round their edge capacity: tiers of ``u_step`` undirected pairs
    (``2 * u_step`` edge rows), at least ``min_pad_u`` pairs above the demand; for a model with a line
    graph also its triplet capacity: tiers of ``t_step`` triplets, at least ``min_pad_t`` above the demand
    (both None: the configuration's triplet tiers).

    The runtime's default (``CapacityConfig``: 512 pairs, at least 256 of padding; 8,192 triplets, at least
    4,096) suits MatRIS. A model whose padded rows cost real work (every padded edge runs its edge network
    in every layer, every padded triplet its angle network) declares finer steps, so that a small system
    does not evaluate a large share of padding."""

    u_step: int
    min_pad_u: int
    t_step: int | None = None
    min_pad_t: int | None = None
    # Thermal factors, between 1 and 2, where the declared tiers of their dimension are in force: every capture
    # holds this factor times its pair (triplet) demand. That is GPU-resident MD's whole-step capture and each
    # growth of it, and every bucket of a model-graph runner (the calculator's included; not the exact-capacity
    # buckets of GPU-resident MD's precapture walk). The counts of a heating system grow, and fine tiers would
    # otherwise capture again at every step of that growth. At a whole step's first capture,
    # MATRIS_CAPACITY_THERMAL_FACTOR set to another value than 1 scales the triplet demand in place of the factor.
    u_thermal_factor: float = 1.0
    t_thermal_factor: float = 1.0  # other than 1 only with triplet tiers
    # The relative headroom, in [0, 1], that a capacity growth of a union of replicas
    # (``fastmd.interfaces.replica_md``) adds over the overflowing pair (triplet) demand: at least the drift of
    # the union's count at its target temperature, which every replica of the union drifts alike. None: the
    # runtime's (``fastmd.execution.replica_batch``). Single systems do not read them.
    u_union_headroom: float | None = None
    t_union_headroom: float | None = None

    def __post_init__(self) -> None:
        if (self.t_step is None) != (self.min_pad_t is None):
            raise ValueError("CapacityStep: t_step and min_pad_t are given together (or neither)")
        for name in ("u_union_headroom", "t_union_headroom"):
            headroom = getattr(self, name)
            if headroom is not None and (isinstance(headroom, bool) or not isinstance(headroom, (int, float))
                                         or not 0.0 <= headroom <= 1.0):
                raise ValueError(f"CapacityStep.{name} must be None or in [0, 1]")
        for name in ("u_thermal_factor", "t_thermal_factor"):
            factor = getattr(self, name)
            if (isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 1.0 <= factor <= 2.0
                    or (name == "t_thermal_factor" and factor != 1.0 and self.t_step is None)):
                raise ValueError(f"CapacityStep.{name} must be in [1, 2] (the triplet factor other than 1 only "
                                 "with triplet tiers)")
        for name, minimum in (("u_step", 1), ("min_pad_u", 0), ("t_step", 1), ("min_pad_t", 0)):
            value = getattr(self, name)
            if name in ("t_step", "min_pad_t") and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"CapacityStep.{name} must be an integer >= {minimum}")


@runtime_checkable
class CapacityDeclaration(Protocol):
    """Capability: the model's :class:`CapacityStep`. GPU-resident MD and the calculator use its pair tiers
    where the configuration keeps the default pair step (``CapacityConfig.u_step`` and ``min_pad_u``), and
    its triplet tiers where it keeps the default triplet step (``t_step`` and ``min_pad_t``); an explicit
    step is kept as given."""

    def capacity_step(self) -> CapacityStep:
        ...


# --- Capability negotiation (fastmd.adapters.negotiate): declarations a model may make --------------

# How a runtime path's results are held to their reference, strictest first. "bitwise": bit for bit the
# reference implementation's same path in deterministic mode (MatRIS: against the previous revision);
# "fp32_noise": within the eager-vs-eager noise floor of the model's own arithmetic (its FP32, or FP64
# for an FP64 model; another model against its upstream call); "fp32_equivalent": another arithmetic
# of at least FP32 accuracy (3xTF32 GEMMs, an FP64 model evaluated in FP32 on request); "reduced":
# below FP32 (TF32, bf16). A timing trial chooses only among options of one class, and "reduced" is
# never a default.
AccuracyClass = Literal["bitwise", "fp32_noise", "fp32_equivalent", "reduced"]
ACCURACY_CLASSES: tuple[str, ...] = ("bitwise", "fp32_noise", "fp32_equivalent", "reduced")


@dataclass(frozen=True)
class CaptureContract:
    """What a model promises the runtime about capture and numerics."""

    sync_free: bool  # one evaluation makes no host synchronization (it may be captured in a CUDA graph)
    exact_cutoff: bool  # a pair at r >= cutoff contributes exactly 0 to the value and its gradient
    accuracy_class: AccuracyClass = "fp32_noise"  # of the runtime's paths against their reference
    # Max |dF| (eV/Å) over the pairs of K=5 eager evaluations (the G-S2 protocol); None: not measured.
    noise_floor_f: float | None = None
    tol_multiplier: float = 1.5  # force tolerance max(1e-6, tol_multiplier * noise_floor_f)
    # Who differentiates the energy: the model's own call ("model_owned", ModelOwnedDerivative), one
    # engine-owned backward pass of the model's energy ("engine_autograd"), or a direct force head
    # without a backward pass ("direct_head").
    force_mode: Literal["model_owned", "engine_autograd", "direct_head"] = "engine_autograd"
    stress_unit: Literal["GPa", "eV/A^3"] = "eV/A^3"  # of the stress the derivative returns
    # torch.compile in the forward: "dynamic" (traced at run time: no capture, execution auto stays
    # eager), "capture_safe" (static compiled modules a CUDA graph may capture), or "none".
    compiled: Literal["none", "dynamic", "capture_safe"] = "none"
    # In place of exact_cutoff: value and gradient exactly 0 at r == cutoff but not beyond it (a cutoff
    # function with a double root there). A padded row at the cutoff adds nothing, and the model needs the
    # exact r < cutoff list, which the engine's builders re-filter at every build (reused candidates
    # included); whole_step and the calculator's topology reuse accept it.
    exact_at_cutoff: bool = False

    def __post_init__(self) -> None:
        if self.accuracy_class not in ACCURACY_CLASSES:
            raise ValueError(f"accuracy_class must be one of {ACCURACY_CLASSES}, got {self.accuracy_class!r}")
        if self.force_mode not in ("model_owned", "engine_autograd", "direct_head"):
            raise ValueError(f"unknown force_mode {self.force_mode!r}")
        if self.stress_unit not in ("GPa", "eV/A^3"):
            raise ValueError(f"stress_unit must be 'GPa' or 'eV/A^3', got {self.stress_unit!r}")
        if self.compiled not in ("none", "dynamic", "capture_safe"):
            raise ValueError(f"compiled must be 'none', 'dynamic' or 'capture_safe', got {self.compiled!r}")
        if self.noise_floor_f is not None and not 0 <= self.noise_floor_f < float("inf"):
            raise ValueError("noise_floor_f must be None or finite and >= 0")
        if not 1 <= self.tol_multiplier < float("inf"):
            raise ValueError("tol_multiplier must be finite and >= 1")


@dataclass(frozen=True)
class KernelChoice:
    """The implementations of one kernel site, fastest first, behind the upstream's own switch.

    ``reference`` (one of ``options``) is the numerical reference that ``optimization_profile="generic"``
    selects; it cannot be "reduced". ``accuracy`` gives an option's class where it differs from the
    contract's (e.g. a 3xTF32 GEMM "fp32_equivalent", declared only after it passed the model's force
    accuracy gate). ``by_execution`` orders the options under an execution mode ("eager",
    "model_graph", "whole_step") where they rank differently there (e.g. a kernel fastest inside a CUDA
    graph but slower eager); the negotiation installs the first option of the decided mode's order."""

    site: str
    options: tuple[str, ...]
    reference: str
    accuracy: Mapping[str, AccuracyClass] = field(default_factory=dict)
    by_execution: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def order(self, execution: str | None = None) -> tuple[str, ...]:
        """The options fastest first under ``execution`` (None, or a mode without its own order:
        ``options``)."""
        return tuple(self.by_execution.get(execution, self.options))

    def __post_init__(self) -> None:
        if not self.options or len(set(self.options)) != len(self.options):
            raise ValueError(f"kernel site {self.site!r}: options must be distinct and not empty")
        if self.reference not in self.options:
            raise ValueError(f"kernel site {self.site!r}: the reference {self.reference!r} is not an option")
        unknown = set(self.accuracy) - set(self.options)
        if unknown or any(value not in ACCURACY_CLASSES for value in self.accuracy.values()):
            raise ValueError(f"kernel site {self.site!r}: accuracy classes must name options and classes")
        if self.accuracy.get(self.reference) == "reduced":
            raise ValueError(f"kernel site {self.site!r}: the reference {self.reference!r} cannot be 'reduced' "
                             "(reduced accuracy is never a default)")
        for mode, order in self.by_execution.items():
            if mode not in ("eager", "model_graph", "whole_step") or sorted(order) != sorted(self.options):
                raise ValueError(f"kernel site {self.site!r}: by_execution maps an execution mode to an order of "
                                 f"all the options, got {mode!r}: {order!r}")


@runtime_checkable
class CaptureDeclaration(Protocol):
    """Capability: the model declares its :class:`CaptureContract` (without it: not capturable)."""

    def capture_contract(self) -> CaptureContract:
        ...


@runtime_checkable
class KernelBackends(Protocol):
    """Capability: kernel sites with alternative implementations (the kernel decision)."""

    def kernel_backends(self) -> Sequence[KernelChoice]:
        """The sites and their options, fastest first."""
        ...

    def select_kernel(self, site: str, option: str) -> None:
        """Install ``option`` at ``site`` through the upstream's own switch, including any import and
        JIT warm-up, outside every graph; raise if it is unavailable (the next option is tried). All or
        nothing: an option that raises must leave the model as it was."""
        ...


@runtime_checkable
class RecomputeParticipant(Protocol):
    """Capability: the saved-tensor recompute levels the model implements (public names)."""

    recompute_levels: tuple[str, ...]




@runtime_checkable
class ShardDeclaration(Protocol):
    """Capability: the model's own multi-GPU exchange strategy, the object through which the distributed runtime
    runs it at P>1 (``fastmd.distributed.strategy.ExchangeStrategy``); ``custom_strategy()`` imports it when
    called (the single-GPU entry points never do). A model without it does not run on the distributed
    runtime."""

    def custom_strategy(self) -> Any:
        ...


@dataclass(frozen=True)
class ExchangePoint:
    """One exchange of node features in a model's P>1 forward (:class:`ShardSpec`).

    At P>1 a rank holds its owned atoms and, as ghost rows, every atom within
    the halo of its domain. Where a layer reads the features of its edges'
    senders, the ghost rows must first hold their owners' rows: the runtime
    hooks the module at ``module`` (a dotted path in ``adapter.module``, e.g.
    ``"interactions.1"``) and replaces the ghost rows of the hooked tensor by a
    COPY of the owners' rows (backward: the reverse SUM of the ghost rows'
    gradients into the owners'). ``hook="input"`` hooks the module's input
    ``arg`` (a keyword name or a positional index), ``"output"`` its output.
    An input that is a mapping is addressed by ``(argument, key)``, e.g.
    ``(0, "x")``: the entry ``key`` of the mapping passed as ``argument``
    (SevenNet's convolutions take one dict and read its ``"x"``); the module
    then receives a shallow copy of the mapping with that entry replaced (the
    caller's mapping is not written: the module's return value carries its
    result). ``width`` is the number of values per row (the exchange's bytes).
    ``module=None``: no hook; the adapter's own layer loop
    (:class:`ShardedTap`) calls the exchange itself.

    ``kind="global_sum"`` (in ``ShardSpec.global_reductions``, ``module=None``):
    not a COPY but the FP64 sum over the ranks of ``width`` values, which the
    adapter's pooled readout calls (backward: the sum over the ranks of the
    values' gradients).

    ``kind="pair_merge"`` (a model with a line graph,
    ``ShardSpec.message_flow="line_graph"``): the hooked tensor has one row per
    bond of the rank's line graph, row k being the batch's undirected pair k
    (``GraphBatch.pair_u2d`` / ``pair_d2u`` numbering), holding the rank's
    partial sum over its own triplets; a line slot (a boundary bond shorter
    than ``ShardSpec.line_slot_cutoff``, held by the ranks of both its ends)
    takes the sum of both ends' partials (each end adds the other's: one IEEE
    add, the same bits at both ends; backward: the same exchange of the
    gradients). ``fuse_into`` (a COPY only): the name of an earlier pair_merge
    point whose collective also carries this COPY's rows, with ``fuse_from``
    the module input (``(module, arg)``) where the COPY's tensor is first
    passed before that point, the same tensor object this point's module
    receives (one collective instead of two): an input of the same width as
    that point's, and one per pair_merge point.
    """

    name: str  # the stable name of the exchange, e.g. "interactions.1.node_feats"
    module: str | None  # the dotted path of the hooked module in adapter.module; None: called by a ShardedTap
    width: int  # values per row (a global sum: values)
    hook: Literal["input", "output"] = "input"
    # hook="input": the hooked input, a keyword name or a positional index, or (one of them, the key of an entry of
    # that mapping argument)
    arg: int | str | tuple[int | str, str] = 0
    kind: Literal["copy", "global_sum", "pair_merge"] = "copy"
    fuse_into: str | None = None  # a COPY carried by this earlier pair_merge point's collective
    fuse_from: tuple[str, int | str] | None = None  # (module, arg) where that COPY's tensor is first passed

    def __post_init__(self) -> None:
        if not self.name or self.module == "":
            raise ValueError("ExchangePoint: a name is required, and a module path is a dotted path or None")
        if self.hook not in ("input", "output"):
            raise ValueError(f"ExchangePoint {self.name!r}: hook must be 'input' or 'output', got {self.hook!r}")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 1:
            raise ValueError(f"ExchangePoint {self.name!r}: width must be a positive integer")
        arg = self.arg
        if isinstance(arg, tuple):
            if len(arg) != 2 or not isinstance(arg[1], str) or not arg[1] or self.hook != "input":
                raise ValueError(f"ExchangePoint {self.name!r}: a mapping entry is addressed by (argument, key) on "
                                 "an input hook")
            arg = arg[0]
        if isinstance(arg, bool) or not isinstance(arg, (int, str)):
            raise ValueError(f"ExchangePoint {self.name!r}: arg must be a keyword name, a positional index or "
                             "(one of them, the key of a mapping entry)")
        if self.kind not in ("copy", "global_sum", "pair_merge"):
            raise ValueError(f"ExchangePoint {self.name!r}: kind must be 'copy', 'global_sum' or 'pair_merge', got "
                             f"{self.kind!r}")
        if self.kind == "global_sum" and self.module is not None:
            raise ValueError(f"ExchangePoint {self.name!r}: a global sum is called by the adapter (module=None)")
        if self.kind == "pair_merge" and self.module is None:
            raise ValueError(f"ExchangePoint {self.name!r}: a pair_merge point hooks a module (module=None is a "
                             "ShardedTap's)")
        if (self.fuse_into is None) != (self.fuse_from is None) or (self.fuse_into is not None and (
                self.kind != "copy" or self.module is None or not isinstance(self.fuse_from, tuple)
                or len(self.fuse_from) != 2 or not isinstance(self.fuse_from[0], str)
                or isinstance(self.fuse_from[1], bool) or not isinstance(self.fuse_from[1], (int, str)))):
            raise ValueError(f"ExchangePoint {self.name!r}: a fused COPY hooks a module and names both fuse_into (a "
                             "pair_merge point) and fuse_from (module, arg)")


@dataclass(frozen=True)
class ShardSpec:
    """How a model runs at P>1 on the runtime's generic node-halo exchange (``fastmd.distributed.node_halo``).

    ``message_flow="receiver"``: every layer sums its messages at the receiving
    atom (the edge's centre) and reads only per-atom features of the senders,
    so a rank keeps the edges of its owned centres and the layers need, before
    each declared ``exchange_points`` (forward order), the owners' features on
    the ghost rows; the first layer's features are computed from the atoms'
    species on every row (no exchange). ``"both_ends"``: every layer sends its
    messages to both ends of an edge (an edge state of its own may be carried
    from layer to layer), so a rank keeps every edge with an owned end (an edge
    between two domains is computed on both ranks, each copy from the same
    COPY'd node features) and its ghost rows need the owners' features before
    each declared exchange point as well. ``readout="per_atom"``: the energy
    is the sum of the per-atom energies of the model's energy tap, summed over
    the owned atoms. ``"pooled"``: the energy pools the atoms' features before
    its head (no per-atom energies); the adapter runs its own layer loop and
    readout (:class:`ShardedTap`), pools its owned rows and sums the pooled
    values over the ranks through ``global_reductions`` (``ExchangePoint``\\ s
    of kind "global_sum"). ``pad_species``: the atomic number of the unused
    atom slots (one the model accepts). ``source_pins``: pins
    (:class:`~fastmd.adapters.patches.PatchRecord` without a replacement) of
    the upstream code the hooks (or the adapter's own layer loop) rely on,
    checked when an evaluator that exchanges is built (P>1, or CPU ranks; a
    mismatch refuses it with the reason); one CUDA rank, which exchanges
    nothing, never reads them. ``allow_tf32=False``: the P>1 forward is
    validated in IEEE FP32 only, and a TF32 request at P>1 raises
    :class:`NegotiationError` (P=1 keeps it). ``kernels``: the option each kernel
    site of the model runs at P>1 where that is fixed (``((site, option),
    ...)``, sites of the model's ``kernel_backends``; the adapter's own layer
    loop runs, e.g., eager modules), installed in place of the site's
    negotiated choice. ``capacity_step``: the edge-capacity tiers of
    the P>1 topology (and of one CUDA rank of the distributed entry point), a
    :class:`CapacityStep` (a first capacity at the model's tier of the demand,
    a growth with a small headroom) for a model whose padded edge rows cost as
    much as real ones; None: the topology's own (1/8 headroom).

    ``message_flow="line_graph"`` (``fastmd.distributed.line_partial``): a
    model with a line graph (``TopologySpec.triplet_cutoff``) whose bond
    updates sum the messages of the triplets at both ends of a bond (CHGNet).
    A rank keeps the edges with an owned end (a boundary bond on the ranks of
    both its ends) and the triplets of its owned centres, so a bond's sum
    over its triplets is partial at a boundary: the declared ``pair_merge``
    points complete it on the line slots, the boundary bonds shorter than
    ``line_slot_cutoff`` (the only ones with triplets). The padded rows are
    sink atoms (species 1, the builder's), which the model must accept.
    """

    exchange_points: tuple[ExchangePoint, ...]
    message_flow: Literal["receiver", "both_ends", "line_graph"] = "receiver"
    readout: Literal["per_atom", "pooled"] = "per_atom"
    pad_species: int = 1
    source_pins: tuple[Any, ...] = ()
    global_reductions: tuple[ExchangePoint, ...] = ()
    allow_tf32: bool = True
    kernels: tuple[tuple[str, str], ...] = ()
    capacity_step: CapacityStep | None = None
    line_slot_cutoff: float | None = None  # line_graph: the bonds with triplets are shorter than this (Å)

    def __post_init__(self) -> None:
        if self.message_flow not in ("receiver", "both_ends", "line_graph"):
            raise ValueError(f"ShardSpec: message_flow {self.message_flow!r} is not supported (receiver, both_ends or "
                             "line_graph)")
        if self.readout not in ("per_atom", "pooled"):
            raise ValueError(f"ShardSpec: readout {self.readout!r} is not supported (per_atom or pooled)")
        if not all(isinstance(point, ExchangePoint) and point.kind in ("copy", "pair_merge")
                   for point in self.exchange_points):
            raise ValueError("ShardSpec: exchange_points must be ExchangePoints of kind 'copy' or 'pair_merge'")
        if not all(isinstance(point, ExchangePoint) and point.kind == "global_sum"
                   for point in self.global_reductions):
            raise ValueError("ShardSpec: global_reductions must be ExchangePoints of kind 'global_sum'")
        if (self.readout == "pooled") != bool(self.global_reductions):
            raise ValueError("ShardSpec: a pooled readout sums its pooled values over the ranks through "
                             "global_reductions, and only a pooled readout declares them")
        names = [point.name for point in self.exchange_points + self.global_reductions]
        if len(set(names)) != len(names):
            raise ValueError("ShardSpec: the exchange points' names must be distinct")
        if isinstance(self.pad_species, bool) or not isinstance(self.pad_species, int) or self.pad_species < 1:
            raise ValueError("ShardSpec: pad_species must be a positive atomic number")
        if not isinstance(self.allow_tf32, bool):
            raise ValueError("ShardSpec: allow_tf32 must be a bool")
        if not all(isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(name, str) for name in pair)
                   for pair in self.kernels) or len({site for site, _ in self.kernels}) != len(self.kernels):
            raise ValueError("ShardSpec: kernels is a tuple of (site, option) pairs (strings), one per site")
        if self.capacity_step is not None and not isinstance(self.capacity_step, CapacityStep):
            raise ValueError("ShardSpec: capacity_step must be a CapacityStep or None")
        line = self.message_flow == "line_graph"
        if line and self.adapter_called:
            raise ValueError("ShardSpec: a line-graph shard reads per-atom energies through its hooks (no pooled "
                             "readout, no exchange point without a module)")
        if line != (self.line_slot_cutoff is not None) or (line and not (
                isinstance(self.line_slot_cutoff, (int, float)) and math.isfinite(self.line_slot_cutoff)
                and self.line_slot_cutoff > 0)):
            raise ValueError("ShardSpec: message_flow 'line_graph' needs a finite positive line_slot_cutoff, and "
                             "only it takes one")
        merges, fused = {}, set()
        for point in self.exchange_points:
            if point.kind == "pair_merge":
                if not line:
                    raise ValueError(f"ShardSpec: pair_merge point {point.name!r} needs message_flow 'line_graph'")
                merges[point.name] = point
            elif point.fuse_into is not None:
                merge = merges.get(point.fuse_into)
                if merge is None:
                    raise ValueError(f"ShardSpec: {point.name!r} fuses into {point.fuse_into!r}, which is not an "
                                     "earlier pair_merge point")
                if point.fuse_into in fused:
                    raise ValueError(f"ShardSpec: {point.fuse_into!r} carries one fused COPY; {point.name!r} is a "
                                     "second")
                # One fused buffer (the merge's width), the COPY's own input checked against the source tensor.
                if point.hook != "input" or point.width != merge.width:
                    raise ValueError(f"ShardSpec: the fused COPY {point.name!r} hooks an input of its pair_merge "
                                     f"point's width ({merge.width})")
                fused.add(point.fuse_into)

    @property
    def adapter_called(self) -> bool:
        """Whether the adapter's own layer loop calls the exchanges (:class:`ShardedTap`): a pooled readout, or an
        exchange point without a hooked module."""
        return self.readout == "pooled" or any(point.module is None for point in self.exchange_points)

    def check_pins(self, model: str) -> list:
        """The source pins checked against the installed upstream code (nothing is installed): their
        :class:`~fastmd.adapters.patches.PatchOutcome`; a mismatch raises :class:`NegotiationError` naming the
        pins and the reasons (P>1 is refused)."""
        from fastmd.adapters.patches import apply_patches

        if any(getattr(pin, "replacement", None) is not None for pin in self.source_pins):
            raise NegotiationError(f"model {model!r}: ShardSpec.source_pins must be pins (no replacement)")
        outcomes = apply_patches(self.source_pins)
        declined = {outcome.record.name: outcome.reason for outcome in outcomes if not outcome.applied}
        if declined:
            raise NegotiationError(f"model {model!r} cannot run on several GPUs: the upstream code its node-halo "
                                   f"hooks rely on differs from the pinned source ({declined})")
        return outcomes

    def digest(self) -> str:
        """sha256 of the declaration (the exchange points, the message flow, the readout, the pad species, the
        pins, and where declared the global reductions, the TF32 refusal, the P>1 kernels, the edge tiers and the
        line-graph fields): the rank contract's entry and the P>1 plan fingerprint's."""
        import hashlib
        import json

        pins = [(getattr(pin, "name", None), getattr(pin, "target", None), getattr(pin, "source_sha256", None),
                 tuple(getattr(pin, "versions", ()))) for pin in self.source_pins]
        # The line-graph fields only where declared (a receiver shard's digest predates them).
        points = [(p.name, p.module, p.width, p.hook, p.arg) + (
            () if p.kind == "copy" and p.fuse_into is None else (p.kind, p.fuse_into, p.fuse_from))
            for p in self.exchange_points]
        declaration = {"points": points, "message_flow": self.message_flow, "readout": self.readout,
                       "pad_species": self.pad_species, "pins": pins}
        # Only where declared: the digest of a declaration without them is unchanged.
        if self.global_reductions:
            declaration["global_reductions"] = [(p.name, p.width) for p in self.global_reductions]
        if not self.allow_tf32:
            declaration["allow_tf32"] = False
        if self.kernels:
            declaration["kernels"] = dict(self.kernels)
        if self.capacity_step is not None:
            step = self.capacity_step
            declaration["capacity_step"] = (step.u_step, step.min_pad_u, step.u_thermal_factor)
        if self.line_slot_cutoff is not None:
            declaration["line_slot_cutoff"] = self.line_slot_cutoff
        text = json.dumps(declaration, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()


@runtime_checkable
class ShardSpecDeclaration(Protocol):
    """Capability: the model's :class:`ShardSpec`, through which the distributed runtime runs it at P>1 with its
    generic node-halo exchange (``fastmd.distributed.ShardedEvaluator``); None: the model does not run at P>1.
    A model with an exchange strategy of its own declares :class:`ShardDeclaration` instead."""

    def shard_spec(self) -> ShardSpec | None:
        ...


@runtime_checkable
class ShardedTap(Protocol):
    """Capability: the model's own P>1 forward and readout over a rank's rows, for a :class:`ShardSpec` whose
    exchanges the adapter calls itself (a pooled readout: ORB).

    ``sharded_energy(inputs, positions, cell, ctx)`` evaluates the model on one
    rank's rows (``prepare`` of its batch: the owned slots, atoms then unused
    slots, then the ghost rows) at the differentiable ``positions`` and ``cell`` (as
    :meth:`EnergyTap.node_energy`) and returns this rank's share of the energy
    (one element; the runtime sums the shares over the ranks and differentiates
    each rank's share). ``ctx``, the runtime's context of the step:

    * ``ctx.exchange(point, x)``: ``x`` (``[rows, ...]``) with its ghost rows
      replaced by a COPY of their owners' rows (each declared exchange point
      once per forward, in the declared order);
    * ``ctx.global_sum(point, x)``: the FP64 sum over the ranks of ``x``
      (``[point.width]``; each global reduction once per forward);
    * ``ctx.counted_once(x)``: a value the same on every rank (computed from
      global sums) counted once in the sum of the shares: ``x`` on one rank, a
      zero on the others, through which every rank still differentiates (a
      pooled readout passes the head of its sums through it once per forward:
      the runtime counts the call, and a share that skips it is refused; one
      that counted the head on every rank would multiply the forces by the
      number of ranks);
    * ``ctx.owned``: ``[rows]`` bool, the rank's owned real rows (a pooled sum
      reads these rows only); ``ctx.n_total``: the atoms of the whole system
      (an FP64 scalar tensor).

    The call must not synchronize the host or differentiate itself, and every
    rank must make the same calls in the same order.
    """

    def sharded_energy(self, inputs: Any, positions: Tensor, cell: Tensor, ctx: Any) -> Tensor:
        ...


@runtime_checkable
class SystemBatching(Protocol):
    """Capability: how the model evaluates several systems in one graph batch (``GraphBatch.systems``).

    ``"union"``: each atom's energy depends on its own system's atoms only (message passing over edges and a
    per-atom readout), so the systems may be evaluated as one disjoint union and the model may ignore
    ``systems`` for its energies and forces (a per-atom energy tap). ``"segments"``: the model reads
    ``systems`` and keeps the systems apart itself where it pools over a system (a mean over its atoms
    before an energy head): its input maps the segments to the model's own batch in each call, without a
    host synchronization (e.g. the atoms per system as ``row_ptr[1:] - row_ptr[:-1]``), and a whole-system
    tap returns one energy per system. The declaration comes from the loaded module (a model family may hold
    variants that are not local, or that pool otherwise); None: not declared, and the runtime does not batch
    the model (``fastmd.interfaces.replica_md`` rejects it). The negotiation also checks the declaration on
    two copies of the system."""

    def system_batching(self) -> Literal["union", "segments"] | None:
        ...


# The values of SystemBatching.system_batching() the runtime batches.
SYSTEM_BATCHING_KINDS = ("union", "segments")


@runtime_checkable
class EdgeShiftInput(Protocol):
    """Capability: the model's energy tap takes the engine's per-edge shifts, so the systems of one graph batch
    may have cells of their own (``SystemSegments.cells``).

    True: :meth:`EnergyTap.node_energy` accepts the keyword ``edge_shifts``, ``[edge rows, 3]`` in the dtype of
    ``positions``: the Cartesian shift ``edge_image @ cell`` of each edge in its own system's cell, row by row
    (``image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]``, the arithmetic of the
    adapters' own float32 shifts), and takes an edge's vector from it in place of its own ``edge_image @ cell``;
    ``cell`` is then the first system's and the model reads nothing else from it. The engine passes the shifts
    only for systems with cells of their own (a model that declares ``"union"``: a whole-system tap is not
    admitted for them yet); a call without them is the single-system call. The declaration comes from the loaded
    module; False or None: one cell per batch."""

    def edge_shift_input(self) -> bool:
        ...


@runtime_checkable
class SystemInputs(Protocol):
    """Capability: static per-system inputs (a model head, a task, charge, spin), part of the plan
    fingerprint; a change means a new plan."""

    def system_inputs(self) -> Mapping[str, Any]:
        ...


@runtime_checkable
class RuntimePlanned(Protocol):
    """Capability: the runtime plans the model by rules of its own, named by ``runtime_rules``. The one set is
    ``"matris"`` (MatRIS: execution auto in GPU-resident MD and in the calculator's capacity policy, K6 memory
    admission, saved-tensor recompute, topology reuse, precision, and the kernel policy of the operator
    dispatch, whose profiles include "legacy"). The negotiation records these decisions and does not make them
    anew; a model without this capability is negotiated. :func:`runtime_planned` reads it."""

    runtime_rules: str


class NegotiationError(ValueError):
    """A request the adapter cannot serve: P>1 without a layer exchange, the MatRIS-only
    ``optimization_profile="legacy"`` for another model, a kernel site whose reference fails, runtime rules
    the runtime does not have (``RuntimePlanned``), or declarations that contradict each other
    (``exact_at_cutoff`` only with ``far_self_loops`` padding)."""


# The rules the runtime plans a model by (RuntimePlanned.runtime_rules): MatRIS's.
RUNTIME_RULES = ("matris",)


def runtime_planned(adapter) -> bool:
    """Whether the runtime plans ``adapter`` by rules of its own (:class:`RuntimePlanned`, MatRIS), so that
    the negotiation records its decisions and does not make them: the adapter's class declares
    ``runtime_rules`` other than None (looked up statically on ``adapter.__class__``, so a wrapper forwarding
    attributes or a plain mock declares nothing, and a mock with an adapter class as its spec declares that
    class's rules). Rules the runtime does not have are an error."""
    rules = inspect.getattr_static(adapter.__class__, "runtime_rules", None)
    if rules is None:
        return False
    if rules not in RUNTIME_RULES:
        name = getattr(adapter, "name", type(adapter).__name__)
        raise NegotiationError(f"model {name!r} asks for the runtime rules {rules!r}; the runtime has rules for "
                               f"{', '.join(map(repr, RUNTIME_RULES))} only")
    return True


def check_profile(adapter, profile: str | None) -> None:
    """``optimization_profile="legacy"`` (registry defaults plus MATRIS_* environment options) exists
    for MatRIS only."""
    if profile == "legacy" and not runtime_planned(adapter):
        raise NegotiationError(f"optimization_profile='legacy' is MatRIS-only; model {adapter.name!r} has the "
                               "profiles 'topology' (fastest kernels, the default) and 'generic' (its reference)")


# --- Upstream patch records (fastmd.adapters.patches) ------------------------------------------------

@runtime_checkable
class UpstreamPatches(Protocol):
    """Capability: the adapter applied patch records to its upstream code when it was loaded
    (:mod:`fastmd.adapters.patches`). The plan record lists them, applied and declined, and the plan
    fingerprint covers the applied ones; a declined patch leaves the adapter correct (e.g. not captured)."""

    def upstream_patches(self) -> Sequence[Any]:
        """The :class:`~fastmd.adapters.patches.PatchOutcome` of each record, applied or declined."""
        ...

    def close(self) -> None:
        """Revert the applied patches (the adapter is not used afterwards)."""
        ...
