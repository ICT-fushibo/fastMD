"""CHGNet behind the adapter protocol: the upstream model with an energy tap and a line graph.

chgnet is optional: this module imports it only when a model is loaded
(``pip install fastmd[chgnet]``), so ``fastmd`` imports without it. The adapter
runs the upstream ``CHGNet`` module as loaded (``CHGNet.load``, the bundled
0.3.0 weights); nothing of CHGNet's model is copied. What the adapter declares
and does:

* **Topology**: CHGNet's two graphs, the engine's line-graph topology: directed
  edges within the atom-graph cutoff (6 Å) with the undirected pair of each,
  and at every centre atom the ordered pairs of two distinct directed edges
  shorter than the bond-graph cutoff (3 Å) as triplets. That is upstream
  ``CrystalGraphConverter``'s rule; the engine's builders give the same edge,
  pair and angle sets.
* **Padding**: far self-loops (``PaddingRule("far_self_loops")``, no sink
  rows): a padded pair is the two opposite self-loops of a real atom, at
  least twice the 6 A cutoff long (never of length 0, where the radial basis
  sin(x)/x is NaN), and a padded triplet joins a padded pair to itself. Its
  smooth cutoffs make both radial bases exactly 0 there, so the bias-free
  bond weights are 0: a padded edge's message into its (real) centre atom is
  exactly 0, a padded triplet's update of its (padded) first bond too, and
  padded pairs and angles reach a real row only through those zero weights
  (in value and gradient). A module without that smooth cutoff (exponent 0)
  or with a basis cutoff beyond the loops pads with sink atoms instead.
* **Capacity tiers** (``capacity_step``): pairs and triplets in tiers of 128
  with at least 128 of padding, every capture holding 1.01 times the pair and
  1.06 times the triplet demand (a heating cell's counts grow; taken where the
  configuration keeps the engine's default tiers, which pad a 135-atom cell
  with hundreds of pairs and thousands of triplets); a union of replicas grows
  by 10 % of its pairs and 25 % of its triplets.
  FastMD also preserves the no-real-angle branch of older biased bond convolutions.
* **Energy tap**: ``node_energy`` runs upstream ``CHGNet._compute`` without
  forces or stress on a ``BatchedGraph`` built from the engine's batch at the
  given positions and cell (the arithmetic of ``BatchedGraph.from_graphs`` for
  one graph), takes the per-atom energies of its readout MLP (``mlp_first``:
  before pooling) and adds each atom's AtomRef reference energy, the weight
  ``w[Z - 1]``: their sum is the extensive energy upstream returns per atom
  times the atom count. The engine differentiates it
  (:mod:`fastmd.engine.energy_derivative`); CHGNet's own autograd
  (``create_graph=True``) never runs.
* **Patch records** (:mod:`fastmd.adapters.patches`, validated on chgnet 0.4.2):
  three replacements remove the call's host synchronizations so that a CUDA
  graph can capture it: ``aggregate`` sizes its output from the caller's
  ``num_owner`` or the evaluation's one graph instead of ``torch.bincount``,
  ``CHGNet._compute``'s ``torch.bincount(atom_owners)`` returns the
  evaluation's static row count, and ``Fourier.forward`` keeps its constant on
  the host. The first two act only inside the adapter's evaluation and defer to
  upstream outside it; the third computes upstream's values. Pins check the
  upstream code the adapter replicates or relies on (AtomRef, ``from_graphs``,
  ``_compute``'s per-atom readout, the bond and angle encoders and the radial
  basis, the line-graph rule). A declined replacement leaves the adapter correct
  but not capturable (its contract is then not sync-free, so the runtime
  evaluates it eagerly); a declined pin refuses to load.

* **Precision**: float32, as upstream (its AtomRef runs float32 only): IEEE
  FP32 under the plan's precision scope (TF32 changes CHGNet's forces by tenths
  of an eV/Å).
* **Isolated atoms**: ``"model"``: CHGNet evaluates an atom without neighbours
  itself (its energy is not the AtomRef weight).
* **Several systems** (``system_batching``): ``"union"`` for upstream's
  ``CHGNet`` itself: replicas of a system run as one disjoint graph with its
  line graph (``fastmd.interfaces.replica_md``).

Loading unpickles upstream's bundled checkpoint after checking its sha256.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastmd.adapters.patches import PatchRecord, apply_patches, revert_patches
from fastmd.adapters.protocol import CapacityStep, CaptureContract, GraphBatch, PaddingRule, \
    TopologySpec

# Pretrained weights by name: (CHGNet.load model name, file in the chgnet package, sha256 of the file).
CHGNET_CHECKPOINTS = {
    "chgnet_0_3_0": ("0.3.0", os.path.join("pretrained", "0.3.0", "chgnet_0.3.0_e29f68s314m37.pth.tar"),
                     "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"),
}
CHGNET_VERSIONS = ("0.4.2",)  # the chgnet releases the patch records were validated on

_FAR_SELF_LOOPS = PaddingRule("far_self_loops")
_SINK_ATOMS = PaddingRule("sink_atoms")
# Measured capacity tiers (GPU-resident MD, whole step, IEEE FP32, GB200; see capacity_step).
# A union of replicas grows by 10 % (pairs) and 25 % (triplets) over the overflowing demand: on the 135-atom
# CsSnI3 frame at 500 K the union's pair count drifts by up to 9.4 % within 1,000 steps of 0.25 fs, its triplets
# from 0 to 22-38 per replica on average and up to 76 (GPU-resident MD, IEEE FP32, GB200).
_CAPACITY_STEP = CapacityStep(u_step=128, min_pad_u=128, t_step=128, min_pad_t=128, u_thermal_factor=1.01,
                              t_thermal_factor=1.06, u_union_headroom=0.10, t_union_headroom=0.25)
_ELEMENTS = 94  # CHGNet's element tables (AtomRef columns, atom embedding rows): Z = 1..94


def _bond_has_angles(batch: GraphBatch) -> Tensor:
    """Device-only empty-angle branch, broadcast over each undirected bond's features."""
    triplets = batch.triplets
    # Padding joins a directed edge to itself. Real angles join distinct edges
    # (possibly the two images of the SAME undirected periodic pair).
    live = (triplets[:, 0] < batch.n_real) & (triplets[:, 2] != triplets[:, 4])
    if batch.n_edges is not None:
        live = live & (triplets[:, 2] < batch.n_edges) & (triplets[:, 4] < batch.n_edges)
    if batch.systems is None:
        return live.any()
    owners = batch.systems.system_of_row.long()
    centres = triplets[:, 0].long().clamp(max=owners.shape[0] - 1)
    counts = torch.zeros(batch.systems.count, dtype=torch.int64, device=owners.device)
    counts.index_add_(0, owners.index_select(0, centres), live.long())
    pair_centres = batch.edge_index.index_select(0, batch.pair_u2d.long())[:, 0].long()
    pair_systems = owners.index_select(0, pair_centres.clamp(max=owners.shape[0] - 1))
    return counts.index_select(0, pair_systems).gt(0).unsqueeze(1)


class _Evaluation(threading.local):
    """The adapter evaluation in progress in this thread (``node_energy`` around upstream's ``_compute``):
    its static atom owners (one graph) and per-graph row count, which the replacements answer with; None
    outside one."""

    owners: Tensor | None = None
    atoms_per_graph: Tensor | None = None


_ACTIVE = _Evaluation()
_UPSTREAM: dict[str, Any] = {}  # upstream's originals of the replaced functions (set before the first patch)


def _aggregate(data: Tensor, owners: Tensor, *, average: bool = True, num_owner: int | None = None) -> Tensor:
    """Upstream ``aggregate`` without its ``torch.bincount`` (a host synchronization): inside an
    evaluation the output has ``num_owner`` rows (the convolutions pass it) or, for the pooling of the
    evaluation's atoms, one row per graph; the averaging count is an ``index_add``. Upstream's own
    otherwise."""
    if _ACTIVE.owners is None or (num_owner is None and owners is not _ACTIVE.owners):
        return _UPSTREAM["aggregate"](data, owners, average=average, num_owner=num_owner)
    rows = _ACTIVE.atoms_per_graph.shape[0] if num_owner is None else int(num_owner)
    output = data.new_zeros([rows, data.shape[1]]).index_add_(0, owners, data)
    if average:
        count = torch.zeros(rows, dtype=torch.int64, device=owners.device).index_add_(
            0, owners, torch.ones_like(owners, dtype=torch.int64))
        count = count.where(count != 0, count.new_ones(1))
        output = (output.T / count).T
    return output


class _ModelTorch:
    """``torch`` as ``chgnet.model.model`` sees it: inside an evaluation, ``bincount`` of the
    evaluation's atom owners (``atoms_per_graph`` in ``CHGNet._compute``, a host synchronization) is
    its static per-graph row count; every other name and call is torch's."""

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    @staticmethod
    def bincount(input: Tensor, weights: Tensor | None = None, minlength: int = 0) -> Tensor:  # noqa: A002
        if _ACTIVE.owners is not None and input is _ACTIVE.owners and weights is None and minlength == 0:
            return _ACTIVE.atoms_per_graph
        return torch.bincount(input, weights=weights, minlength=minlength)


_MODEL_TORCH = _ModelTorch()
# Upstream's constant 1 / sqrt(2) as upstream computes it (float32 sqrt and division, IEEE-rounded), on
# the host without tensor work (importing this module runs no tensor operator).
_FOURIER_FIRST = float(np.float32(1) / np.sqrt(np.float32(2)))


def _fourier_forward(self, x: Tensor) -> Tensor:
    """Upstream ``Fourier.forward`` with its constant term as a host number (upstream builds it as a CPU
    tensor and copies it to the device in every call); the same values."""
    result = x.new_zeros(x.shape[0], 1 + 2 * self.order)
    result[:, 0] = _FOURIER_FIRST
    tmp = torch.outer(x, self.frequencies)
    result[:, 1 : self.order + 1] = torch.sin(tmp)
    result[:, self.order + 1 :] = torch.cos(tmp)
    return result / np.sqrt(np.pi)


def _record(name: str, target: str, sha256: str, reason: str, **fields) -> PatchRecord:
    return PatchRecord(name=name, target=target, distribution="chgnet", versions=CHGNET_VERSIONS,
                       source_sha256=sha256, reason=reason, **fields)


# The replacements (sync_free needs all three) and the pins of the code the adapter replicates.
SYNC_PATCHES = ("chgnet.aggregate_sized_by_caller", "chgnet.static_atoms_per_graph", "chgnet.fourier_constant_on_host")
PATCH_RECORDS = (
    _record("chgnet.aggregate_sized_by_caller", "chgnet.model.functions.aggregate",
            "9b1e44b40bd8b6e873bf6e3e43632051132089cc993018b683fadd7867af1942",
            "host synchronization: torch.bincount sizes the output; inside an evaluation the rows come "
            "from num_owner or the one graph", replacement=_aggregate, bindings=("chgnet.model.layers.aggregate",)),
    _record("chgnet.static_atoms_per_graph", "chgnet.model.model.torch",
            "5ae5d9693e397b39fce59ffce181a2ac6513144790ca1dacd4207ee1a70c788a",
            "host synchronization: CHGNet._compute counts the atoms per graph with torch.bincount; inside an "
            "evaluation the static row count", replacement=_MODEL_TORCH, source="chgnet.model.model.CHGNet._compute"),
    _record("chgnet.fourier_constant_on_host", "chgnet.model.basis.Fourier.forward",
            "5a4331bab16584633c19ba993fc4ef6ea74b33df7d1b42ab2247313fecf5580f",
            "capture safety: the constant term is a CPU tensor copied to the device per call; the same value as "
            "a host number", replacement=_fourier_forward),
    _record("chgnet.atom_reference_forward", "chgnet.model.composition_model.AtomRef.forward",
            "62f55bdf971e24895823fae07407e3e3261d78c73ad877b3e4275ce359d3a65d",
            "relied on: the per-atom reference energy is the AtomRef weight w[Z - 1]"),
    _record("chgnet.atom_reference_composition", "chgnet.model.composition_model.AtomRef._assemble_graphs",
            "293f6901f81880412912383ca345798441ffb56a608d128c2c5d33b44816ebe4",
            "relied on: intensive composition counts (divided by the atom count)"),
    _record("chgnet.atom_reference_energy", "chgnet.model.composition_model.AtomRef._get_energy",
            "ae4aae9b76dbb2681f54fe941042772f488d10062e070e8e2e72908245c28e8d",
            "relied on: a bias-free linear map of the composition"),
    _record("chgnet.batched_graph_featurization", "chgnet.model.model.BatchedGraph.from_graphs",
            "20f29e77412744560a71c10019a6c04070b38675c729d4a6aa2316886c3a98a7",
            "relied on: node_energy builds the BatchedGraph with this arithmetic"),
    _record("chgnet.per_atom_readout", "chgnet.model.model.CHGNet._compute",
            "5ae5d9693e397b39fce59ffce181a2ac6513144790ca1dacd4207ee1a70c788a",
            "relied on: with mlp_first the readout MLP's output is the per-atom energy the adapter taps"),
    _record("chgnet.bond_encoder", "chgnet.model.encoders.BondEncoder.forward",
            "9768a055cab3702e118d16207c32d1e2c1ad2bed600f0f9a14c2f62669badc21",
            "relied on: bond vectors centre - (neighbour + image @ lattice), bases from the pairs' lengths"),
    _record("chgnet.angle_encoder", "chgnet.model.encoders.AngleEncoder.forward",
            "ab3468f3dafa2146696ed53d0ab24016b836cc7669b5aa07e6fd8469e6d22a81",
            "relied on: angle bases from the two normalized directed bond vectors"),
    _record("chgnet.radial_basis", "chgnet.model.basis.RadialBessel.forward",
            "859d5ab9c8317807e3383e0827eea820b83ebcac6052d2130b7d39b8e7960891",
            "relied on: finite for every padded bond (sin(x)/x, nonzero lengths)"),
    _record("chgnet.smooth_cutoff", "chgnet.model.basis.CutoffPolynomial.forward",
            "03aa9a309e145889666959053633526bd54d63d806e9ae50642a5e5b47fca900",
            "relied on: an exact cutoff (0 at and beyond it), as the capture contract declares"),
    _record("chgnet.line_graph_rule", "chgnet.graph.graph.Graph.line_graph_adjacency_list",
            "a1a69b5b070d4a8e562238566481936df8eb6e65b4e450a76bfd0d90a1fac851",
            "relied on: the angle rule the engine's line-graph builders reproduce"),
)
# The upstream code the multi-GPU hooks rely on (``CHGNetAdapter.shard_spec``), checked where the hooks are installed
# (P>1 and CPU ranks of the distributed evaluator; never by one CUDA rank or the single-GPU entry points): the forward,
# which calls the layers with keywords (the same atom features into a bond convolution and the next atom
# convolution, every layer while the line graph has a row), and the three layers, whose bond convolution feeds its
# triplet sum to ``mlp_out`` alone. sha256 of inspect.getsource on chgnet 0.4.2.


def _loops_add_nothing(module) -> bool:
    """Whether far self-loops add exactly nothing to ``module``: both radial bases (atom and bond graph) end
    in a smooth cutoff (``CutoffPolynomial``, exponent not 0: exactly 0 at and beyond it) inside the loops,
    which are at least twice the atom-graph cutoff long."""
    from chgnet.model.basis import CutoffPolynomial

    expansion = module.bond_basis_expansion
    loop_length = 2.0 * float(module.graph_converter.atom_graph_cutoff)
    for basis in (expansion.rbf_expansion_ag, expansion.rbf_expansion_bg):
        cutoff = getattr(basis, "smooth_cutoff", None)
        if not isinstance(cutoff, CutoffPolynomial) or cutoff.p == 0 or not float(cutoff.cutoff) <= loop_length:
            return False
    return True


def is_chgnet_model(model, model_path: str | None = None) -> bool:
    """Whether ``load_adapter`` loads ``model`` as CHGNet: a name above."""
    return isinstance(model, str) and model in CHGNET_CHECKPOINTS


def check_load_options(model_dtype: str | None, optimization_profile: str | None, model_path: str | None) -> None:
    """The loading options ``load_adapter`` rejects for CHGNet."""
    from fastmd.adapters import NegotiationError

    if optimization_profile == "legacy":
        raise NegotiationError("optimization_profile='legacy' is MatRIS-only; CHGNet runs upstream's kernels "
                               "under 'topology' (the default) and 'generic'")
    if model_dtype == "float64":
        raise ValueError("CHGNet runs float32 (upstream's AtomRef is float32 only): model_dtype must be None "
                         "or 'float32'")
    if model_path is not None:
        raise ValueError("CHGNet loads its bundled pretrained weights by name; model_path is not supported")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_chgnet(model, *, model_path: str | None = None, device: str | None = None, model_dtype: str | None = None,
                optimization_profile: str | None = None) -> "CHGNetAdapter":
    """A :class:`CHGNetAdapter` of a pretrained name (``"chgnet_0_3_0"``): upstream's bundled checkpoint,
    sha256-checked, loaded with ``CHGNet.load`` on ``device`` (float32), parameters frozen, ``eval()``.
    ``optimization_profile`` selects nothing (CHGNet runs upstream's kernels); ``"legacy"`` is MatRIS's."""
    if optimization_profile == "legacy":
        raise ValueError("optimization_profile='legacy' is MatRIS's (CHGNet: None, 'topology' or 'generic')")
    if model_dtype not in (None, "float32"):
        raise ValueError(f"CHGNet runs float32 (upstream's AtomRef is float32 only), not model_dtype={model_dtype!r}")
    if model_path is not None:
        raise ValueError("CHGNet loads its bundled pretrained weights by name; model_path is not supported")
    import chgnet
    from chgnet.model import CHGNet

    name, relative, expected = CHGNET_CHECKPOINTS[model]
    path = os.path.join(os.path.dirname(chgnet.__file__), relative)
    found = _sha256(path)
    if found != expected:
        raise ValueError(f"CHGNet checkpoint {model!r} at {path}: sha256 {found} is not the pinned {expected}")
    device = torch.device("cuda" if device is None and torch.cuda.is_available() else (device or "cpu"))
    module = CHGNet.load(model_name=name, use_device=str(device), verbose=False)
    return CHGNetAdapter(module, checkpoint={"name": model, "path": path, "sha256": found})


@dataclass
class CHGNetInput:
    """CHGNet's input over a graph batch: the batch itself and its shape constants (every row is one
    system, graph 0). The value-dependent tensors (bond and angle bases) are built in each call."""

    batch: GraphBatch
    owners: Tensor  # [rows] int32 zeros: CHGNet's atom owners
    atoms_per_graph: Tensor  # [1] int64: the rows of graph 0


class CHGNetAdapter:
    """An upstream CHGNet module as a :class:`~fastmd.adapters.protocol.ModelAdapter` with an
    :class:`~fastmd.adapters.protocol.EnergyTap` (see the module docstring).

    ``module`` is the module itself; ``checkpoint`` names the loaded file (name, path, sha256).
    Constructing the adapter applies its patch records (``records``, default :data:`PATCH_RECORDS`);
    :meth:`close` reverts them.
    """

    name = "chgnet"
    compiled = False  # upstream CHGNet does not run torch.compile

    def __init__(self, module, *, checkpoint: dict | None = None,
                 records: tuple[PatchRecord, ...] = PATCH_RECORDS) -> None:
        if not getattr(module, "mlp_first", False):
            raise ValueError("the CHGNet energy tap reads per-atom energies before pooling (mlp_first=True)")
        self.module = module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.checkpoint = dict(checkpoint or {})
        # Far self-loops where the smooth cutoffs make them add nothing (the pretrained models); else sink rows.
        self._padding = _FAR_SELF_LOOPS if _loops_add_nothing(self.module) else _SINK_ATOMS
        # Shape constants per (rows, device): built once and never freed, since captured graphs read them.
        self._constants: dict[tuple, tuple[Tensor, Tensor]] = {}
        self._checked_species = None
        import chgnet.model.functions

        if chgnet.model.functions.aggregate is not _aggregate:  # upstream's, before a record replaces it
            _UPSTREAM["aggregate"] = chgnet.model.functions.aggregate
        self._patches = apply_patches(records)
        refused = {o.record.name: o.reason for o in self._patches if o.record.pin and not o.applied}
        if refused:
            revert_patches(self._patches)
            raise RuntimeError(f"CHGNet adapter: the upstream code it relies on is not the validated one ({refused})")
        applied = {o.record.name for o in self._patches if o.applied}
        # Captured only with every synchronization removed (else evaluated eagerly, still correct).
        self.sync_free = all(name in applied for name in SYNC_PATCHES)

    # --- ModelAdapter -----------------------------------------------------

    def topology_spec(self) -> TopologySpec:
        """Directed edges within the atom-graph cutoff, their pairs, and the triplets of bonds within the
        bond-graph cutoff (upstream's graph converter's cutoffs); one atom-graph layer per atom convolution."""
        converter = self.module.graph_converter
        return TopologySpec(
            cutoff=float(converter.atom_graph_cutoff),
            num_interactions=int(self.module.n_conv),
            triplet_cutoff=float(converter.bond_graph_cutoff),
            undirected_pairs=True,
            model_dtype=str(next(self.module.parameters()).dtype).removeprefix("torch."),
        )

    def padding_rule(self) -> PaddingRule:
        """Far self-loops: CHGNet's smooth cutoffs give the padded pairs and triplets zero weight, so the
        padding needs no sink rows (see the module docstring). Sink atoms for a module whose radial bases do
        not vanish within the loops (no smooth cutoff, or a basis cutoff beyond twice the atom-graph cutoff)."""
        return self._padding

    def prepare(self, batch: GraphBatch) -> CHGNetInput:
        """The batch and its shape constants; rejects a batch without a line graph or positions and
        elements outside CHGNet's tables (a host check, once per species tensor)."""
        if batch.triplets is None or batch.positions is None:
            raise ValueError("CHGNet reads a line graph with Cartesian positions (a graph batch with triplets "
                             "and positions)")
        if batch.species is not self._checked_species:
            species = batch.species
            if bool(((species < 1) | (species > _ELEMENTS)).any()):
                raise ValueError(f"CHGNet supports atomic numbers 1-{_ELEMENTS}")
            self._checked_species = species
        rows, device = batch.species.shape[0], batch.species.device
        key = (rows, str(device))
        if key not in self._constants:
            self._constants[key] = (torch.zeros(rows, dtype=torch.int32, device=device),
                                    torch.full((1,), rows, dtype=torch.int64, device=device))
        owners, atoms_per_graph = self._constants[key]
        return CHGNetInput(batch=batch, owners=owners, atoms_per_graph=atoms_per_graph)

    def isolated_atom_policy(self) -> str:
        """"model": CHGNet evaluates an atom without neighbours itself."""
        return "model"

    def reference_energies(self) -> Tensor:
        """The AtomRef weights as a ``[1, 94]`` float32 row (column ``Z - 1``), read-only: the reference
        part already inside each atom's energy."""
        return self.module.composition_model.fc.weight.detach()

    # --- EnergyTap ----------------------------------------------------------

    def node_energy(self, inputs: CHGNetInput, positions: Tensor, cell: Tensor) -> Tensor:
        """The energy of each row ([rows] float32): CHGNet's per-atom readout at ``positions`` and
        ``cell`` plus the atom's AtomRef weight, with CHGNet's own derivatives off."""
        from chgnet.model.model import BatchedGraph

        module, batch = self.module, inputs.batch
        edges, triplets = batch.edge_index, batch.triplets
        # BatchedGraph.from_graphs for one graph: bonds from the centre and neighbour positions and the
        # neighbour's image; angles from the two directed bond vectors of each triplet row.
        # The integer images in the cell's dtype (a float64 module's float64 cell; float32: the batch's own).
        image = batch.edge_image if batch.edge_image.dtype == cell.dtype else batch.edge_image.to(cell.dtype)
        bond_bases_ag, bond_bases_bg, bond_vectors = module.bond_basis_expansion(
            center=positions.index_select(0, edges[:, 0]), neighbor=positions.index_select(0, edges[:, 1]),
            undirected2directed=batch.pair_u2d, image=image, lattice=cell)
        angle_bases = module.angle_basis_expansion(bond_vectors.index_select(0, triplets[:, 2]),
                                                   bond_vectors.index_select(0, triplets[:, 4]))
        graph = BatchedGraph(
            atomic_numbers=batch.species, bond_bases_ag=bond_bases_ag, bond_bases_bg=bond_bases_bg,
            angle_bases=angle_bases, batched_atom_graph=edges,
            batched_bond_graph=torch.stack((triplets[:, 0], triplets[:, 1], triplets[:, 3]), dim=1),
            atom_owners=inputs.owners, directed2undirected=batch.pair_d2u, atom_positions=[positions],
            strains=[None], volumes=None)
        tapped, caller = [], threading.get_ident()

        def tap(_module, _args, output) -> None:
            # This call's thread only: concurrent evaluations (other threads) hook the same module for theirs.
            if threading.get_ident() == caller:
                tapped.append(output)
        hook = module.mlp.register_forward_hook(tap)
        bond_hooks = []
        try:
            if batch.padding is not None or batch.systems is not None:
                # CHGNet skips bond convolutions for an empty angle graph. The
                # biased 0.2.0 layers need this branch per system in a union,
                # including when a captured graph changes its live triplet count.
                has_angles = _bond_has_angles(batch)

                def preserve_empty_graph(_module, _args, kwargs, output):
                    if threading.get_ident() != caller:
                        return output
                    return torch.where(has_angles, output, kwargs["bond_feas"])

                bond_hooks = [layer.register_forward_hook(preserve_empty_graph, with_kwargs=True)
                              for layer in module.bond_conv_layers if layer is not None]
            with self._evaluation(inputs):
                module._compute(graph, compute_force=False, compute_stress=False)
        finally:
            hook.remove()
            for bond_hook in bond_hooks:
                bond_hook.remove()
        reference = module.composition_model.fc.weight[0]  # the AtomRef row (column Z - 1), where the module is
        return tapped[-1].squeeze(1) + reference.index_select(0, batch.species - 1)

    @contextlib.contextmanager
    def _evaluation(self, inputs: CHGNetInput):
        """The replacements' evaluation scope, only where this adapter's own patches applied (another
        adapter in the process may hold the process-wide replacements: outside a scope they are
        upstream's code)."""
        if not self.sync_free:
            yield
            return
        previous = (_ACTIVE.owners, _ACTIVE.atoms_per_graph)
        _ACTIVE.owners, _ACTIVE.atoms_per_graph = inputs.owners, inputs.atoms_per_graph
        try:
            yield
        finally:
            _ACTIVE.owners, _ACTIVE.atoms_per_graph = previous

    # --- capability negotiation ---------------------------------------------

    @property
    def upstream(self) -> str:
        """The package providing the model and the weights' file hash."""
        from importlib.metadata import version

        return f"chgnet {version('chgnet')}, weights {self.checkpoint.get('name', '')} sha256 " \
               f"{self.checkpoint.get('sha256', '')[:16]}"

    def capture_contract(self) -> CaptureContract:
        """No host synchronization in a call with the patches applied (captured whole), an exact
        cutoff, held to upstream's call within twice the eager noise floor (atomic scatters: not
        bitwise), the engine-owned derivative with the stress in eV/Å³, no torch.compile."""
        return CaptureContract(sync_free=self.sync_free, exact_cutoff=True, accuracy_class="fp32_noise",
                               tol_multiplier=2.0, force_mode="engine_autograd", stress_unit="eV/A^3",
                               compiled="none")

    def system_batching(self) -> str | None:
        """"union" for upstream's ``CHGNet`` itself: its per-atom energies come from convolutions over the
        edges and the line graph (each atom's and bond's own neighbours), row-wise layers and a per-atom readout
        before pooling (``mlp_first``, which the adapter requires), so systems evaluated as one disjoint graph
        give each atom its own system's energy (the per-graph pooling and counts, which the tap does not read,
        see the union as one graph). None for any other module (a subclass: its call is not known)."""
        if (type(self.module).__name__, type(self.module).__module__) != ("CHGNet", "chgnet.model.model"):
            return None
        return "union"

    def capacity_step(self) -> CapacityStep:
        """CHGNet's capacity tiers (``_CAPACITY_STEP``): every padded edge runs the atom convolution's gated
        MLP and every padded triplet the bond convolution's and the angle update's, so a small cell takes
        fine tiers of both; and a union's growth headrooms, above its counts' drift at 500 K."""
        return _CAPACITY_STEP

    def upstream_patches(self) -> list:
        """The outcome of each patch record (applied or declined)."""
        return list(self._patches)

    def close(self) -> None:
        """Revert the applied patches (idempotent); the adapter then runs upstream's code paths (not
        sync-free)."""
        revert_patches(self._patches)
        self.sync_free = False


