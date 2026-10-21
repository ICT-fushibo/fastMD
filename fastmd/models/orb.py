"""ORB v3 behind the adapter protocol: upstream orb-models with a whole-system energy tap.

orb-models is optional: this module imports it only when a model is loaded
(``pip install fastmd[orb]``), so ``fastmd`` imports without it. The adapter
wraps the upstream ``ConservativeForcefieldRegressor`` as loaded; no ORB module
is vendored, and the blocks below that restate upstream code (the replacement
of ``segment_ops.aggregate_nodes``, ZBL's pair terms, the edge vectors, the
readout's energy lines and the multi-GPU layer loop) carry their provenance
and license. What the adapter declares and does:

* **Topology**: directed edges within ORB's 6 Å radius, five message-passing
  layers, no triplets (the builders run edges only). ORB keeps at most 120
  neighbours per atom (``max_neighbors``); the engine builds the full radius
  graph and flags on the device a build in which an atom has more.
* **Padding**: far self-loops. ORB gates every message by an envelope times
  ``r < 6`` and masks its ZBL pair repulsion by the covalent radii, so a loop
  at least twice the cutoff long adds exactly nothing; its length is never 0
  (the Bessel basis sin(x)/x is NaN there).
* **Energy tap**: ORB has no per-atom energies (its energy head pools the
  atoms' features before its MLP, and the ZBL term is a mean over atoms), so
  the adapter is a whole-system tap: ``system_energy`` runs the energy lines of
  upstream's ``ConservativeForcefieldRegressor.forward`` (backbone, energy
  head, ``denormalize``, ZBL) on the engine's edges and returns the raw energy;
  the engine differentiates it (``fastmd.engine.energy_derivative``). The
  readout's sums (the node features it pools, the linear reference, ZBL's pair
  terms) and the energy's tail (the head's per-atom value times the atom
  count, plus the reference and ZBL's mean) are FP64; the head's MLP runs on
  the mean in the model's dtype. Upstream's own featurisation (its rotation
  generator and per-call determinant), its normaliser, the confidence head
  and the rotational gradient are not run: molecular dynamics reads none of
  them.
* **Inputs**: ``prepare(batch)`` keeps the batch and two shape constants; each
  call maps the engine's edges to ORB's ``AtomGraphs``: senders = centre and
  receivers = neighbour (int64), unit shifts = ``edge_image``, so ORB's vector
  ``x[receiver] - x[sender] + shift`` is the engine's edge.
* **Several systems** (a union of replicas, ``GraphBatch.systems``):
  ``"segments"`` for upstream's conservative regressor as loaded. ORB pools
  per graph, so each system becomes its own ORB graph: each call reads the
  atoms per system from ``row_ptr`` and the edge rows per system (contiguous
  blocks, the padding with the last system), on the device, and the tap
  returns one energy per system.
* **Kernels**: the backbone under upstream's own ``torch.compile`` switch
  (``model.compile(mode="default", dynamic=True)``, as ORB's loader does) by
  default, eager (the reference) for ``optimization_profile="generic"``. The
  compiled module is captured inside the engine's CUDA graphs after the
  runners' warm-ups ("capture_safe").
* **Patches** (``fastmd.adapters.patches``, applied at load, reverted by
  ``close``): two host synchronizations of upstream's forward are removed
  (segment aggregation sizes, inside the adapter's own calls only, where the
  restated readout no longer calls upstream's poolings; the ZBL cutoff power
  read from the device); the upstream code the energy tap restates or relies
  on is pinned by source hash (the linear reference's forward, which the
  readout restates, among the multi-GPU pins only). A declined patch leaves
  the adapter correct but not capturable.
* **Precision**: IEEE FP32 (the loader asks upstream for "float32-highest" and
  restores the process's FP32 GEMM flags and default dtype afterwards); TF32
  only on request (``InferenceConfig(matmul_precision="high")``, accuracy
  class "reduced").
* **Isolated atoms**: ``"model"``: an atom without neighbours enters ORB's mean
  pooling and ZBL mean like any other.


Checkpoints by name (``orb_v3``: orb-v3-conservative-inf-mpa) come from
upstream's own downloader and are checked against a pinned sha256; a
``model_path`` loads that file (the same architecture).
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import threading
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fastmd.adapters.patches import PatchRecord, apply_patches, revert_patches
from fastmd.adapters.protocol import (
    CapacityStep, CaptureContract, GraphBatch, KernelChoice, PaddingRule, TopologySpec,
)

# Pretrained checkpoints by name: (upstream's ORB_PRETRAINED_MODELS key, download URL, sha256 of the file).
ORB_CHECKPOINTS = {
    "orb_v3": ("orb-v3-conservative-inf-mpa",
               "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/"
               "orb-v3-conservative-inf-mpa-20250404.ckpt",
               "bd1840d1488f3cd01d213970d9618a63f971590ad45e5e83bf5404d4b95d216a"),
}
# The orb-models versions the patch records (and the energy lines below) were validated on.
ORB_VERSIONS = ("0.5.4",)
# The kernel site and its options, fastest first; eager is the reference.
KERNEL_SITE = "backbone"
KERNEL_OPTIONS = ("inductor", "eager")

_FAR_SELF_LOOPS = PaddingRule("far_self_loops")
# Edge-capacity tiers of 64 rows, at least 24 rows above the demand: every padded row runs ORB's edge MLP
# (768 -> 1024 -> 1024 -> 256) in each of its five layers.
_CAPACITY_STEP = CapacityStep(u_step=32, min_pad_u=12)
_ELEMENTS = 118  # ORB's one-hot classes (atomic numbers 0-117)


def is_orb_model(model, model_path: str | None = None) -> bool:
    """Whether ``load_adapter`` loads ``model`` as ORB: a name above (with or without ``model_path``)."""
    return isinstance(model, (str, os.PathLike)) and os.fspath(model) in ORB_CHECKPOINTS


def check_load_options(model_dtype: str | None, optimization_profile: str | None, model_path: str | None) -> None:
    """The loading options ``load_adapter`` rejects for ORB."""
    from fastmd.adapters import NegotiationError

    if optimization_profile == "legacy":
        raise NegotiationError("optimization_profile='legacy' is MatRIS-only; ORB has the profiles 'topology' "
                               "(compiled backbone, the default) and 'generic' (eager, its reference)")
    if model_dtype == "float64":
        raise ValueError("ORB runs float32 on this runtime: model_dtype must be None or 'float32'")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(name: str, model_path: str | None) -> dict:
    """The file to load: the pinned pretrained checkpoint (fetched by upstream's downloader,
    sha256-checked), or the given path."""
    _, url, expected = ORB_CHECKPOINTS[name]
    if model_path is None:
        from cached_path import cached_path  # orb-models' own downloader

        path = str(cached_path(url))
        found = _sha256(path)
        if found != expected:
            raise ValueError(f"ORB checkpoint {name!r} at {path}: sha256 {found} is not the pinned {expected}")
        return {"name": name, "path": path, "sha256": found}
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ORB checkpoint not found: {model_path!r}")
    found = _sha256(model_path)
    if found != expected:
        # Another checkpoint of the same architecture may have been trained for another neighbour cap (the
        # "-20" models: 20) or another graph: only the pinned one's declarations are known to hold.
        raise ValueError(f"ORB checkpoint {model_path!r}: sha256 {found} is not the pinned {name!r} checkpoint "
                         f"{expected}")
    return {"name": name, "path": str(model_path), "sha256": found}


# --- patch records -----------------------------------------------------------------------------------

# Upstream's aggregate_nodes (saved before the first adapter patches it), and whether this thread is inside an
# adapter's own call: only there does the patched function take its sync-free path.
_UPSTREAM: dict[str, Any] = {}
_OWN_CALL = threading.local()


@contextlib.contextmanager
def _sync_free_aggregation():
    previous = getattr(_OWN_CALL, "active", False)
    _OWN_CALL.active = True
    try:
        yield
    finally:
        _OWN_CALL.active = previous


# Derived from orb-models 0.5.4, orb_models/forcefield/segment_ops.py:8-53 (aggregate_nodes). Copyright 2024 Orbital
# Materials, Inc. Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Changed: inside the adapter's own calls the segment ids get their output size and the scatters theirs (no host
# synchronization); every other caller gets upstream's function.
def _aggregate_nodes(tensor, n_node, reduction="mean", deterministic=False):
    """``orb_models.forcefield.segment_ops.aggregate_nodes`` without host synchronization inside the
    adapter's own calls: the segment ids get their output size (the rows of ``tensor``, which the graphs'
    sizes sum to there) and the scatters theirs (the graph count), so neither is read back from the device.
    Every other caller gets upstream's function (which sizes its output by the last non-empty graph)."""
    if not getattr(_OWN_CALL, "active", False):
        return _UPSTREAM["aggregate_nodes"](tensor, n_node, reduction, deterministic)
    from orb_models.forcefield import segment_ops

    count = len(n_node)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    segments = torch.arange(count, device=tensor.device).repeat_interleave(n_node, output_size=tensor.shape[0])
    if reduction == "sum":
        return segment_ops.scatter_sum(tensor, segments, dim=0, dim_size=count)
    if reduction == "mean":
        return segment_ops.scatter_mean(tensor, segments, dim=0, dim_size=count)
    if reduction == "max":
        return segment_ops.segment_max(tensor, segments, count)
    raise ValueError("Invalid reduction argument. Use sum, mean or max.")


def _cutoff_with_host_power(original, power, r, r_max, p):
    """``ZBLBasis._polynomial_cutoff_with_derivative`` with its power read once on the host (``power``)
    instead of ``float(p)`` of a device buffer at every call."""
    return original(r, r_max, power)


AGGREGATE_NODES = PatchRecord(
    name="orb.aggregate_nodes_sizes_on_host",
    target="orb_models.forcefield.segment_ops.aggregate_nodes",
    distribution="orb-models", versions=ORB_VERSIONS,
    source_sha256="c97e0abf1626c9177855026914cab711c44a523406b37de7a0bf9b6e71e3dc2f",
    reason="host synchronization: repeat_interleave without output_size and scatters without dim_size read "
           "sizes back from the device in every pooling (energy head, reference energy, ZBL) of the adapter's calls",
    replacement=_aggregate_nodes)

# Upstream code the energy tap runs or mirrors unpatched (pins: checked, nothing installed).
PINS = tuple(PatchRecord(name=name, target=target, distribution="orb-models", versions=ORB_VERSIONS,
                         source_sha256=digest, reason=reason) for name, target, digest, reason in (
    ("orb.regressor_energy_lines", "orb_models.forcefield.conservative_regressor.ConservativeForcefieldRegressor"
     ".forward", "48bc2a2b59a7d25bc00067a78501a942e4ea6f2714ccf571d0ed53e737fb0e33",
     "the energy tap runs this forward's energy lines (backbone, energy head, denormalize, ZBL)"),
    ("orb.energy_head_pooling", "orb_models.forcefield.forcefield_heads.EnergyHead.forward",
     "228928b3e1a2a4dcbbcd4971c816cf061e99fdca433d6dfc12ef99a0f35f6f03",
     "the energy readout pools over the atoms (no per-atom energies: a whole-system tap)"),
    ("orb.energy_head_denormalize", "orb_models.forcefield.forcefield_heads.EnergyHead.denormalize",
     "675f61c362a3c16e9574f4f94589c0bc4fedda1d72c19d88627bb54705acdee4",
     "the raw energy: scale, atom count and the linear reference"),
    ("orb.zbl_energy", "orb_models.forcefield.pair_repulsion.ZBLBasis.forward",
     "88e5f6a3164959d15912686634fecbde0df10e398107ec1ecaa324844a427a19",
     "the pair repulsion reads the edge vectors and the one-hot elements the tap builds"),
    ("orb.exact_cutoff", "orb_models.forcefield.nn_util.get_cutoff",
     "cc9c9e7d6eccd52661a55aed8284293ddb16e99bf58d57361259a54301666002",
     "messages are gated by envelope * (r < 6): the far self-loop padding adds exactly nothing"),
    ("orb.backbone_inputs", "orb_models.forcefield.gns.MoleculeGNS.forward",
     "a02ec27dc12566dc38198e09fca6ce486d898895ced0746c7ef4206f2cc6f026",
     "the backbone reads the edge vectors, the senders and receivers and the element features only"),
    ("orb.edge_features_cutoff", "orb_models.forcefield.gns.MoleculeGNS.featurize_edges",
     "263fa380df96750fd852dce982af0b82eb45333c5603b526ef31377931b4211b",
     "the initial edge features are the cutoff times the basis: 0 for a padded loop"),
    ("orb.message_cutoff_gate", "orb_models.forcefield.gns.AttentionInteractionNetwork.forward",
     "5c3b6f57cdca3abafc47869a6d363dae2582c26a863d9e5a34507b39074f90d0",
     "every message is gated by the cutoff before the segment sums: a padded loop reaches no atom"),
))


# The upstream code ORB's P>1 forward (ORBAdapter.sharded_energy: its own layer loop and readout) restates or relies on
# row by row, checked when a multi-GPU evaluator is built (ShardSpec.source_pins; not part of the patches above, which
# the single-GPU plan reads): (name, target, sha256 of its source, reason).


# Derived from orb-models 0.5.4, orb_models/forcefield/pair_repulsion.py:60-120 (the energy lines of ZBLBasis.forward).
# Copyright 2024 Orbital Materials, Inc. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Changed: returns each directed edge's term, before upstream's sum by
# sender and mean over the atoms (which ORB's readout takes in FP64); no forces or stress.
def _zbl_pair_terms(zbl, graph) -> Tensor:
    """Each directed edge's ZBL term (``0.5 v(r) envelope(r)``, in the model's dtype) of ``ZBLBasis.forward``:
    upstream's energy is the mean over the atoms of these terms summed by sender."""
    senders, receivers = graph.senders, graph.receivers
    one_hot = graph.node_features["atomic_numbers_embedding"]
    atomic_numbers = torch.argmax(one_hot, dim=1)
    z_u = atomic_numbers[senders] + 1
    z_v = atomic_numbers[receivers] + 1
    a = zbl.a_prefactor * 0.529 / (torch.pow(z_u, zbl.a_exp) + torch.pow(z_v, zbl.a_exp))
    x = graph.edge_features["vectors"].norm(dim=1)
    r_over_a = x / a
    exp_term = torch.exp(-zbl.d * r_over_a.unsqueeze(0))
    phi = torch.sum(zbl.c * exp_term, dim=0)
    coulomb_term = 14.3996 * z_u * z_v / x
    v_edges_raw = coulomb_term * phi
    r_max = zbl.covalent_radii[z_u] + zbl.covalent_radii[z_v]
    envelope, _ = zbl._polynomial_cutoff_with_derivative(x, r_max, zbl.p)
    return 0.5 * v_edges_raw * envelope


def _zbl_power_record(zbl) -> PatchRecord:
    """The per-instance record of one ZBL module: its cutoff with the power read once here (outside any
    graph)."""
    original = zbl._polynomial_cutoff_with_derivative
    return PatchRecord(
        name="orb.zbl_cutoff_power_on_host",
        target="orb_models.forcefield.pair_repulsion.ZBLBasis._polynomial_cutoff_with_derivative",
        distribution="orb-models", versions=ORB_VERSIONS,
        source_sha256="4308142915ca37eed724baba3970c5ad12c5936d4b9b0d75b823367b138cdb23",
        reason="host synchronization: float(p) of the power, a device buffer, at every call",
        replacement=functools.partial(_cutoff_with_host_power, original, int(zbl.p)), per_instance=True)


# --- loading -------------------------------------------------------------------------------------------

def load_orb(model, *, model_path: str | None = None, device: str | None = None, model_dtype: str | None = None,
             optimization_profile: str | None = None) -> "ORBAdapter":
    """An :class:`ORBAdapter` of a pretrained name (see :func:`is_orb_model`), from ``model_path`` if given.

    Loads upstream's model with ``precision="float32-highest"`` and without compiling, restoring the
    process's FP32 GEMM flags and default dtype afterwards (upstream's loader sets them); applies the
    patch records; installs the backbone kernels: upstream's torch.compile switch unless
    ``optimization_profile`` is "generic" (eager, the reference) or the device is not CUDA.
    """
    from fastmd.adapters import NegotiationError

    if optimization_profile == "legacy":
        raise NegotiationError("optimization_profile='legacy' is MatRIS-only; ORB has the profiles 'topology' "
                               "(compiled backbone, the default) and 'generic' (eager, its reference)")
    if model_dtype not in (None, "float32"):
        raise ValueError(f"ORB runs float32 on this runtime: model_dtype must be None or 'float32', got "
                         f"{model_dtype!r}")
    from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS  # before any download: needs orb-models

    from fastmd.precision import restore_fp32_state

    name = os.fspath(model)
    upstream_name = ORB_CHECKPOINTS[name][0]
    device = torch.device("cuda" if device is None and torch.cuda.is_available() else (device or "cpu"))
    checkpoint = _checkpoint(name, model_path)

    default_dtype = torch.get_default_dtype()
    try:
        with restore_fp32_state(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Setting global torch default dtype")
            warnings.filterwarnings("ignore", message="Consider passing 'precision=float32-high'")
            module = ORB_PRETRAINED_MODELS[upstream_name](weights_path=checkpoint["path"], device=device,
                                                          precision="float32-highest", compile=False)
    finally:
        torch.set_default_dtype(default_dtype)
    adapter = ORBAdapter(module, checkpoint=checkpoint)
    if optimization_profile == "generic":
        adapter.kernel_reason = "optimization_profile='generic': eager, the reference"
    elif device.type != "cuda":
        adapter.kernel_reason = "not CUDA: eager"
    else:
        adapter.select_kernel(KERNEL_SITE, "inductor")
    return adapter


@dataclass
class ORBInput:
    """ORB's input over a graph batch: the batch, its shape constants (one system, graph 0) and the
    one-hot table. The value-dependent tensors (int64 edge indices, element features, edge vectors, and for
    a batch of several systems their atom and edge counts) are built in each call."""

    batch: GraphBatch
    n_node: Tensor  # [1] int64: the rows (one GPU: the atoms; at P>1 ORB's graph size only, see prepare)
    n_edge: Tensor  # [1] int64: every edge row (ORB's per-edge graph index would count rows)
    max_num_neighbors: Tensor  # [1] int64: ORB's neighbour cap (AtomGraphs field; not read by the tap)
    one_hot: Tensor  # [118, 118] identity in the model's dtype: row Z is the one-hot of atomic number Z


class ORBAdapter:
    """An upstream ORB v3 ``ConservativeForcefieldRegressor`` as a
    :class:`~fastmd.adapters.protocol.ModelAdapter` with a
    :class:`~fastmd.adapters.protocol.SystemEnergyTap` (see the module docstring).

    ``module`` is the regressor itself; ``kernel`` names the backbone's kernels ("inductor" or
    "eager"), ``checkpoint`` the loaded file (name, path, sha256) and ``patches`` the patch outcomes.
    """

    name = "orb_v3"

    def __init__(self, module, *, checkpoint: dict | None = None) -> None:
        from importlib.metadata import version

        self.module = module.eval()
        self._operator_patches = []
        self._steady_installation = None
        self.operator_backend = 'upstream'
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.checkpoint = dict(checkpoint or {})
        self.kernel = "eager"
        self.kernel_reason = ""
        self._version = version("orb-models")
        if self._version not in ORB_VERSIONS:
            raise RuntimeError(f"orb-models {self._version} is not supported (validated: {', '.join(ORB_VERSIONS)})")
        self._in_use = False  # prepare() was called: the module is bound to a runtime from now on
        head = module.heads[getattr(module, "energy_name", "energy")]
        # The readout restates upstream's poolings (_head_energy): the head's mean exactly when it predicts the
        # per-atom average (upstream's constructor ties the two), ZBL's mean or sum.
        if head.node_aggregation != ("mean" if head.atom_avg else "sum") or (
                module.pair_repulsion and module.pair_repulsion_fn.node_aggregation not in ("mean", "sum")):
            raise RuntimeError("ORB's energy head or pair repulsion pools otherwise than the readout restates "
                               f"(head {head.node_aggregation!r}, per-atom average {head.atom_avg})")
        system_config = module.system_config
        self._cutoff = float(system_config.radius)
        self._max_neighbors = int(system_config.max_num_neighbors)
        if getattr(self.module.model, "_compiled_call_impl", None) is not None:
            self.kernel = "inductor"  # compiled before it was wrapped (upstream's loader compiles by default)
        # The one-hot tables per device: built once and never freed, since captured graphs read them.
        self._one_hot: dict[str, Tensor] = {}
        from orb_models.forcefield import segment_ops

        if segment_ops.aggregate_nodes is not _aggregate_nodes:
            _UPSTREAM.setdefault("aggregate_nodes", segment_ops.aggregate_nodes)
        self.patches = apply_patches([AGGREGATE_NODES, *PINS])
        if self.module.pair_repulsion:
            self.patches += apply_patches([_zbl_power_record(self.module.pair_repulsion_fn)],
                                          instance=self.module.pair_repulsion_fn)
        pins = [outcome for outcome in self.patches if outcome.record.pin and not outcome.applied]
        if pins:
            self.close()
            raise RuntimeError("ORB code the energy tap relies on changed: "
                               + "; ".join(f"{o.record.name}: {o.reason}" for o in pins))
        self._sync_free = all(outcome.applied for outcome in self.patches)

    # --- ModelAdapter -----------------------------------------------------

    @property
    def compiled(self) -> bool:
        """Whether the backbone runs under torch.compile (capture-safe once warmed up: the contract)."""
        return self.kernel == "inductor"

    def topology_spec(self) -> TopologySpec:
        """Directed edges within ORB's radius, five layers, no triplets, at most 120 neighbours."""
        parameter = next(self.module.parameters())
        return TopologySpec(
            cutoff=self._cutoff,
            num_interactions=len(self.module.model.gnn_stacks),
            triplet_cutoff=None,
            undirected_pairs=False,
            model_dtype=str(parameter.dtype).removeprefix("torch."),
            edge_order="by_centre",  # ORB scatters messages to both ends; the order changes sums by rounding
            max_neighbors=self._max_neighbors,
        )

    def padding_rule(self) -> PaddingRule:
        """Far self-loops: exactly nothing under ORB's exact cutoff, never a zero length."""
        return _FAR_SELF_LOOPS

    def prepare(self, batch: GraphBatch) -> ORBInput:
        """The batch and its shape constants; rejects sink rows and elements ORB cannot encode (host
        check, outside any graph). At P>1 the batch is a rank's rows (its owned atoms, unused slots of
        hydrogen, ghost rows): ``sharded_energy`` evaluates every row and its readout reads the owned real rows
        and the system's atom count, so ``n_node`` (every row) is only the size of ORB's graph there."""
        rows = batch.species.shape[0]
        if batch.positions is None or batch.positions.shape[0] != rows or batch.n_real != rows:
            raise ValueError("ORB reads Cartesian positions: an edges-only graph batch without sink rows")
        species = batch.species
        if bool(((species < 1) | (species >= _ELEMENTS)).any()):
            raise ValueError(f"ORB encodes atomic numbers 1-{_ELEMENTS - 1}")
        device = species.device
        if str(device) not in self._one_hot:
            dtype = next(self.module.parameters()).dtype
            self._one_hot[str(device)] = torch.eye(_ELEMENTS, dtype=dtype, device=device)
        self._in_use = True
        return ORBInput(batch=batch,
                        n_node=torch.full((1,), rows, dtype=torch.int64, device=device),
                        n_edge=torch.full((1,), batch.edge_index.shape[0], dtype=torch.int64, device=device),
                        max_num_neighbors=torch.full((1,), self._max_neighbors, dtype=torch.int64, device=device),
                        one_hot=self._one_hot[str(device)])

    def capacity_step(self) -> CapacityStep:
        """Tiers of 32 undirected pairs (64 edge rows), at least 12 pairs above the demand: a padded row
        costs a full edge-network row in every layer."""
        return _CAPACITY_STEP

    def isolated_atom_policy(self) -> str:
        """"model": an atom without neighbours enters ORB's mean pooling and ZBL mean like any other."""
        return "model"

    def reference_energies(self) -> Tensor:
        """ORB's linear reference weights as a ``[1, 118]`` float64 row (column ``Z - 1``; NaN for
        Z = 118, which ORB cannot encode), read-only. A linear baseline of the energy head, not the
        energy of an isolated atom."""
        weight = self.module.heads["energy"].reference.linear.weight.detach().reshape(-1)
        row = torch.full((1, _ELEMENTS), float("nan"), dtype=torch.float64, device=weight.device)
        row[0, :_ELEMENTS - 1] = weight[1:].to(torch.float64)
        return row

    # --- SystemEnergyTap ------------------------------------------------------

    def system_energy(self, inputs: ORBInput, positions: Tensor, cell: Tensor) -> Tensor:
        """ORB's raw energy (``[1]``: backbone, energy head, ``denormalize``, + ZBL; ``[count]`` for a batch of
        several systems, one ORB graph each) at ``positions`` and ``cell``, with ORB's own derivatives and
        featurisation off; its tail in FP64 (:meth:`_head_energy`)."""
        graph = self._atom_graphs(inputs, positions, cell)
        with _sync_free_aggregation():
            nodes = self.module.model(graph)["node_features"]
            sums, reference = self._readout_sums(graph, nodes, systems=inputs.batch.systems)
            return self._head_energy(sums, graph.n_node.to(torch.float64)) + reference

    def _atom_graphs(self, inputs: ORBInput, positions: Tensor, cell: Tensor):
        """ORB's ``AtomGraphs`` of the engine's edges at ``positions`` and ``cell``: the graph of every call (one
        GPU, one ORB graph per system of a batch of several, and a rank's rows at P>1)."""
        from orb_models.forcefield.base import AtomGraphs

        batch = inputs.batch
        senders = batch.edge_index[:, 0].long()  # centre
        receivers = batch.edge_index[:, 1].long()  # neighbour
        numbers = batch.species.long()
        image = batch.edge_image.to(positions.dtype)
        n_node, n_edge, cells = inputs.n_node, inputs.n_edge, cell.unsqueeze(0)
        systems = batch.systems
        if systems is not None:
            # One ORB graph per system, on the device: its atoms from row_ptr (rewritten in place by the builder),
            # and its edge rows as contiguous blocks: the real rows of each system (grouped by centre in ascending
            # order), then the padded rows with the last one (ORB's energy path reads the atom counts and the rows'
            # systems, not these blocks: its poolings per graph).
            n_node = systems.row_ptr[1:] - systems.row_ptr[:-1]
            rows = senders.shape[0]
            real = (torch.ones_like(senders) if batch.n_edges is None
                    else (torch.arange(rows, device=senders.device) < batch.n_edges).long())
            n_edge = torch.zeros(systems.count, dtype=torch.int64, device=senders.device).index_add_(
                0, systems.system_of_row.index_select(0, senders), real)
            n_edge[-1:] += rows - n_edge.sum()
            cells = cells.expand(systems.count, 3, 3)
        # Derived from orb-models 0.5.4, orb_models/forcefield/base.py:164 (the edge vectors). Copyright 2024 Orbital
        # Materials, Inc. Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
        # Changed: the graph is built from the engine's edges and differentiable positions and cell (no
        # featurisation, stress displacement or rotation generator of ORB's own), the shifts row by row.
        # row by row: no TF32 GEMM for FP32
        shifts = image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]
        # ORB's own order of operations (x[receiver] - x[sender] + shift), not the engine's: the two round
        # differently on edges that cross the cell
        vectors = positions.index_select(0, receivers) - positions.index_select(0, senders) + shifts
        return AtomGraphs(
            senders=senders, receivers=receivers, n_node=n_node, n_edge=n_edge,
            node_features={"positions": positions, "atomic_numbers": numbers,
                           "atomic_numbers_embedding": inputs.one_hot.index_select(0, numbers)},
            edge_features={"vectors": vectors, "unit_shifts": image},
            system_features={"cell": cells},
            node_targets=None, edge_targets=None, system_targets=None, system_id=None, fix_atoms=None,
            tags=None, radius=self._cutoff, max_num_neighbors=inputs.max_num_neighbors)

    # Derived from orb-models 0.5.4: the energy lines of ConservativeForcefieldRegressor.forward
    # (conservative_regressor.py:136-143), EnergyHead.forward and denormalize with LinearReferenceEnergy
    # (forcefield_heads.py:36-51, 125-133, 176-181) and ZBLBasis.forward (pair_repulsion.py:60-127). Copyright 2024
    # Orbital Materials, Inc. Licensed under the Apache License, Version 2.0
    # (http://www.apache.org/licenses/LICENSE-2.0).
    # Changed: the pooled sums (the node features, the linear reference, ZBL's pair terms) are FP64 sums over the rows
    # (over a rank's owned rows at P>1, then summed over the ranks), the energy head's MLP runs on their mean in the
    # model's dtype, and the energy's tail (its per-atom value times the atom count, the reference, ZBL's mean) is
    # FP64; no normaliser round trip and no derivatives (the engine differentiates the energy).

    def _readout_sums(self, graph, nodes: Tensor, rows: Tensor | None = None, systems=None) -> tuple[Tensor, Tensor]:
        """The FP64 sums ORB's readout pools per graph, over every row of ``graph`` or the ``rows`` (``[rows]`` bool,
        a select), or per system of ``systems`` (``GraphBatch.systems``, one ORB graph each): ``[graphs, latent +
        1]``, the node features ``nodes`` and the ZBL pair terms of the edges whose sender (centre) is such a row
        (0 without ZBL); and the linear reference of those rows (``[graphs]``)."""
        module = self.module
        features = nodes.to(torch.float64)
        weights = module.heads[module.energy_name].reference.linear.weight.reshape(-1).to(torch.float64)
        reference = weights.index_select(0, graph.node_features["atomic_numbers"])
        pairs = (_zbl_pair_terms(module.pair_repulsion_fn, graph).to(torch.float64) if module.pair_repulsion
                 else features.new_zeros(graph.senders.shape[0]))
        if rows is not None:
            # Selects, not products: a ghost row or an unused slot is never part of a sum.
            features = torch.where(rows[:, None], features, 0.0)
            reference = torch.where(rows, reference, 0.0)
            pairs = torch.where(rows.index_select(0, graph.senders), pairs, 0.0)
        if systems is not None:
            row_system = systems.system_of_row.long()
            pair_system = row_system.index_select(0, graph.senders)
            features = features.new_zeros((systems.count, features.shape[1])).index_add_(0, row_system, features)
            pairs = pairs.new_zeros((systems.count, 1)).index_add_(0, pair_system, pairs[:, None])
            return torch.cat((features, pairs), dim=1), reference.new_zeros(systems.count).index_add_(
                0, row_system, reference)
        return (torch.cat((features.sum(0, keepdim=True), pairs.sum(0, keepdim=True)[None]), dim=1),
                reference.sum(0, keepdim=True))

    def _head_energy(self, sums: Tensor, atoms: Tensor) -> Tensor:
        """The energy (``[graphs]`` FP64) of the pooled sums ``sums`` (``[graphs, latent + 1]``,
        :meth:`_readout_sums`, summed over the ranks at P>1) of ``atoms`` atoms per graph (FP64), without the linear
        reference: upstream's energy head on the mean of the node features (its MLP and normaliser in the model's
        dtype), then in FP64 its per-atom value times ``atoms`` plus ZBL's mean."""
        module = self.module
        head = module.heads[getattr(module, "energy_name", "energy")]
        features, pairs = sums[:, :-1], sums[:, -1]
        # (upstream pools by "mean" exactly when it predicts the per-atom average)
        pooled = features / atoms[:, None] if head.atom_avg else features
        dtype = next(head.mlp.parameters()).dtype
        energy = head.normalizer.inverse(head.mlp(pooled.to(dtype)).squeeze(-1)).to(torch.float64)
        if head.atom_avg:
            energy = energy * atoms
        if module.pair_repulsion:
            energy = energy + (pairs / atoms if module.pair_repulsion_fn.node_aggregation == "mean" else pairs)
        return energy




    # --- capability negotiation -----------------------------------------------

    @property
    def upstream(self) -> str:
        """The package providing the model."""
        return f"orb-models {self._version}"

    def capture_contract(self) -> CaptureContract:
        """No host synchronization in a call while the patches are applied (captured whole), ORB's exact
        cutoff, held to upstream's own call within the eager noise floor (atomic scatters: not bitwise),
        the engine-owned derivative with the stress in eV/Å³, a compiled backbone a CUDA graph may capture
        after the runners' warm-ups."""
        return CaptureContract(sync_free=self._sync_free, exact_cutoff=True, accuracy_class="fp32_noise",
                               force_mode="engine_autograd", stress_unit="eV/A^3",
                               compiled="capture_safe" if self.compiled else "none")

    def kernel_backends(self) -> tuple[KernelChoice, ...]:
        """The backbone: upstream's torch.compile (fastest) or eager (the reference)."""
        return (KernelChoice(site=KERNEL_SITE, options=KERNEL_OPTIONS, reference="eager"),)

    def select_kernel(self, site: str, option: str) -> None:
        """Install ``option`` ("inductor" or "eager") for the backbone through upstream's own switch
        (``nn.Module.compile(mode="default", dynamic=True)``, as ORB's loader does; CUDA only). The
        compiled program is built at the first call (the runners warm up before any capture). The kernels
        are chosen before a runtime binds the module: once :meth:`prepare` has run, only the installed
        option is accepted."""
        if site != KERNEL_SITE or option not in KERNEL_OPTIONS:
            raise ValueError(f"ORB kernel site {KERNEL_SITE!r} has options {KERNEL_OPTIONS}, got {site!r}/{option!r}")
        if option == self.kernel:
            return
        if self._in_use:
            raise RuntimeError(f"ORB kernels are chosen when the model is loaded (installed: {self.kernel!r}); "
                               f"load it with the profile that selects {option!r}")
        backbone = self.module.model
        if option == "eager":
            backbone._compiled_call_impl = None
        else:
            if next(backbone.parameters()).device.type != "cuda":
                raise RuntimeError("the compiled ORB backbone is captured on CUDA only")
            backbone.compile(mode="default", dynamic=True)
        self.kernel = option
        self.kernel_reason = ""

    def system_batching(self) -> str | None:
        """"segments" for upstream's ``ConservativeForcefieldRegressor`` as loaded (its backbone, energy head and
        pair repulsion upstream's own classes): it pools over each graph (the energy head's mean over the atoms,
        the linear reference, the ZBL mean), and :meth:`system_energy` makes each system of a batch its own ORB
        graph and returns one energy per system. None for any other module (a subclass, another head): how it
        pools is not known."""
        module = self.module
        parts = [(module, "ConservativeForcefieldRegressor", "conservative_regressor"),
                 (module.model, "MoleculeGNS", "gns"),
                 (module.heads[module.energy_name], "EnergyHead", "forcefield_heads")]
        if module.pair_repulsion:
            parts.append((module.pair_repulsion_fn, "ZBLBasis", "pair_repulsion"))
        if any((type(part).__name__, type(part).__module__) != (name, f"orb_models.forcefield.{where}")
               for part, name, where in parts):
            return None
        return "segments"

    def upstream_patches(self) -> list:
        """The patch outcomes (applied and declined)."""
        steady = self._steady_installation
        return (list(self.patches) + list(self._operator_patches)
                + ([] if steady is None else list(steady.outcomes)))

    def enable_operator_rewrites(self, backend='torch', *, steady_options=None) -> None:
        """Select operators before warm-up; weights and FP32 precision stay unchanged.

        ``steady_options`` explicitly selects conservative-only output/projection
        optimizations with the Torch backend. Its supported boolean keys are
        ``prune``, ``merge_node`` and ``merge_attention``. ``None`` preserves the
        existing operator path and plan fingerprint. Installation is atomic:
        a refused option or source pin leaves this adapter available for retry.
        """
        if backend not in ('torch', 'triton'):
            raise ValueError("operator backend must be 'torch' or 'triton'")
        if self._in_use or self._operator_patches:
            raise RuntimeError('Select operator rewrites once, before preparing model inputs')
        if steady_options is not None:
            if not isinstance(steady_options, dict):
                raise TypeError('steady_options must be a dictionary of boolean options or None')
            allowed = {'prune', 'merge_node', 'merge_attention'}
            if any(key not in allowed for key in steady_options):
                raise ValueError('steady_options supports only prune, merge_node and merge_attention')
            if any(type(value) is not bool for value in steady_options.values()):
                raise TypeError('steady_options values must be bool')
            if backend != 'torch':
                raise ValueError('steady_options requires the torch operator backend')
            steady_options = dict(steady_options)
        from fastmd.models.operator_rewrites import install_orb

        backbone = self.module.model
        previous_compiled = backbone._compiled_call_impl
        previous_backend = self.operator_backend
        # The original rewrite caches projection tensors outside state_dict.
        # Preserve these too if either installation stage is refused.
        auxiliary = [(layer, name, name in vars(layer), vars(layer).get(name))
                     for layer in backbone.gnn_stacks
                     for name in ('_fastmd_projection_weights', '_fastmd_operator_backend')]
        try:
            self._operator_patches = install_orb(self.module, backend)
            self.operator_backend = backend
            if steady_options is not None:
                from fastmd.models.orb_ops.orb_steady import install
                self._steady_installation = install(self, **steady_options)
            else:
                # Invalidate any previously compiled callable; the new graph
                # traces the rewrites. The steady helper handles its own case.
                backbone._compiled_call_impl = None
                if self.kernel == 'inductor':
                    backbone.compile(mode='default', dynamic=True)
        except BaseException:
            if self._steady_installation is not None:
                self._steady_installation.restore()
                self._steady_installation = None
            revert_patches(self._operator_patches)
            self._operator_patches = []
            for layer, name, owned, value in reversed(auxiliary):
                if owned:
                    setattr(layer, name, value)
                elif name in vars(layer):
                    delattr(layer, name)
            self.operator_backend = previous_backend
            backbone._compiled_call_impl = previous_compiled
            raise

    def system_inputs(self):
        inputs = {'operator_backend': self.operator_backend}
        steady = self._steady_installation
        if steady is not None and not steady.restored:
            inputs.update(steady_options=dict(steady.options),
                          steady_source_pins=dict(steady.source_pins))
        return inputs

    def close(self) -> None:
        """Revert the applied patches (the adapter is not used afterwards)."""
        if self._steady_installation is not None:
            self._steady_installation.restore()
        if self._operator_patches:
            # A compiled callable must not retain an installed replacement.
            self.module.model._compiled_call_impl = None
        revert_patches(self._operator_patches)
        revert_patches(self.patches)
        self._sync_free = False
