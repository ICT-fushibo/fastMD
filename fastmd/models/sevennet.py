"""SevenNet behind the adapter protocol: the upstream SevenNet-0 model with an energy tap.

sevenn is optional: this module imports it only when a model is loaded
(``pip install fastmd[sevennet]``), so ``fastmd`` imports without it. The
adapter wraps sevenn's ``AtomGraphSequential`` as sevenn's own checkpoint
builds it; nothing of SevenNet is copied here. What the adapter declares and
does:

* **Topology**: directed edges within the model cutoff (5 Å for 7net-0), no
  triplets, so the builders run edges only. SevenNet gathers messages at the
  centre and reads features at the neighbour: its ``edge_index`` is the
  engine's (centre, neighbour) as ``[2, E]`` and its edge vector the
  engine's, with no swap and no sign change. The cuEquivariance convolutions
  take the builders' int32 index as it is (they cast to int32 themselves);
  e3nn's take int64.
* **Padding**: exact-rc self-loops (``PaddingRule("exact_rc_loops")``, pad
  vector ``(rc, 0, 0)``). The XPLOR cutoff has a double root at rc, so the
  edge embedding and its gradient there are exactly 0, and the bias-free SiLU
  radial network maps it to zero weights: a padded row adds exactly nothing.
  Beyond rc the cutoff function is not 0, so the model must see the exact
  ``r < rc`` list, which the engine's builders re-filter at every build:
  the contract declares ``exact_at_cutoff``, not ``exact_cutoff``.
* **Capacity tiers** (``capacity_step``): 32 undirected pairs with at least 32
  of padding, every capture holding 1.01 times the pair demand (a heating
  cell's pair count grows; taken where the configuration keeps the engine's
  default tiers, which pad a 135-atom cell with hundreds of pairs that run
  every layer).
* **Energy tap**: the module's ``atomic_energy`` (the species-wise shift and
  scale included), with upstream's force head removed
  (``delete_module_by_key("force_output")``, as sevenn's own deploy does);
  the engine differentiates its sum (``fastmd.engine.energy_derivative``).
* **Upstream patches** (:mod:`fastmd.adapters.patches`): the one-hot table
  on the model's device (sevenn copies a CPU table to the device on every
  call, a host synchronization that prevents capture). If it is declined
  the adapter stays correct but is not declared sync-free (no capture). The
  code the padding and the tap rely on is pinned by source hash; a sevenn
  whose pinned source differs is refused (fail closed).
* **Kernels**: cuEquivariance through sevenn's own build switch, its
  tensor-product method uniform_1d in every execution mode (fused_tp, the
  second option, is faster only eager on systems of a few thousand edges and
  several times slower captured or at a thousand atoms); e3nn is the
  reference (``optimization_profile="generic"``, or without CUDA).
* **Precision**: float32 only (7net-0's parameters and one-hot features are
  float32 upstream); the engine sums the per-atom energies in FP64. The edge
  vectors are passed in the module's dtype, so a float64 copy of the module
  made outside ``load_adapter`` (``double()``, with sevenn's float32 one-hot
  features cast to float64: what the multi-GPU exactness tests run on their
  CPU ranks) runs in float64.
* **Isolated atoms**: ``"model"``: SevenNet evaluates an atom without
  neighbours itself (its energy is not the species shift).


Checkpoints by name (``sevennet_0``, ``7net-0``) ship inside the sevenn wheel
and are checked against a pinned sha256; ``"sevennet"`` with ``model_path``
loads that file. Loading unpickles the file (``torch.load(weights_only=False)``
inside sevenn): load only checkpoints you trust.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fastmd.adapters.patches import PatchRecord, apply_patches, revert_patches
from fastmd.adapters.protocol import (
    CapacityStep, CaptureContract, GraphBatch, KernelChoice, PaddingRule, TopologySpec,
)

# Pretrained checkpoints by name: (sevenn's pretrained name, sha256 of the file).
SEVENNET_CHECKPOINTS = {
    "sevennet_0": ("7net-0", "7052cb42b7b3be42b40b97fa0d21077a48c54b5548948fc4dcf346629f813c36"),
    "7net-0": ("7net-0", "7052cb42b7b3be42b40b97fa0d21077a48c54b5548948fc4dcf346629f813c36"),
}
# The kernel site: the tensor products of the interaction blocks, fastest first in every execution mode; e3nn is
# the reference. Measured on GB200 (IEEE FP32, ms/step): whole_step uniform_1d 1.96 vs fused_tp 3.43 on 135 atoms
# and 5.69 vs 52.9 on 1,344; eager fused_tp is faster on 135 atoms (24.0 vs 25.9) but uniform_1d on 1,344 (26.4 vs
# 56.2: fused_tp's cost grows much faster with the edges), so no execution mode ranks fused_tp first at every size.
KERNEL_SITE = "convolution"
KERNEL_OPTIONS = ("cueq_uniform_1d", "cueq_fused_tp", "e3nn")
_TP_METHODS = {"cueq_uniform_1d": "uniform_1d", "cueq_fused_tp": "fused_tp"}
# Measured capacity tiers (GPU-resident MD, whole step, GB200; see capacity_step).
_CAPACITY_STEP = CapacityStep(u_step=32, min_pad_u=32, u_thermal_factor=1.01)
# The cuEquivariance implementation class of each method (a method whose ops are missing falls back to
# "naive" silently: checked after every install).
_TP_CLASSES = {"uniform_1d": "SegmentedPolynomialFromUniform1dJit", "fused_tp": "SegmentedPolynomialFusedTP"}

# Upstream code the adapter relies on, validated on these versions (sha256 of inspect.getsource).
SEVENN_VERSIONS = ("0.13.0",)
_SEQUENTIAL = "sevenn.nn.sequential.AtomGraphSequential"
ONEHOT_SOURCE_SHA256 = "686a21f4eb41f97fb3f39726504be4b66945b2afa0345cedd7089eb8a2b844c6"
_PINS = (
    # The call: _preprocess (one-hot types, the gradient key), then every module in order.
    ("sevennet.preprocess", f"{_SEQUENTIAL}._preprocess",
     "5dce9cd60972f5574bd5f05cb0856653fcbbd423569edbeb2654153830796540",
     "the call's preprocessing: element one-hots from atomic numbers, the edge vector's gradient key"),
    ("sevennet.forward", f"{_SEQUENTIAL}.forward",
     "da893bdaf75cd0994ec952986437647f94bf611547995034a5aa59eff7074cce",
     "the call runs the remaining modules in order: the force head removed, atomic_energy is the last output"),
    ("sevennet.delete_module_by_key", f"{_SEQUENTIAL}.delete_module_by_key",
     "1bffdb714366a263774ad91cdbfebe09c3de997cdfc23e00890a679f2b18a42f",
     "removes only the named module (the force head)"),
    ("sevennet.onehot_embedding", "sevenn.nn.node_embedding.OnehotEmbedding.forward",
     "5391e05c7a717707e91d1da1c98b87a51a7e597e39195baef16b0c7bc8d7daa1",
     "the element one-hot features of the type indices"),
    ("sevennet.xplor_cutoff", "sevenn.nn.edge_embedding.XPLORCutoff.forward",
     "9cee0b76bc2533060c4f40fdbf588304839cde1ff65b933752b0d56062c3895b",
     "the double root at the cutoff that makes an exact-rc padded row add exactly nothing"),
    ("sevennet.edge_embedding", "sevenn.nn.edge_embedding.EdgeEmbedding.forward",
     "aff687b0c331867c936175df1f31b011f782a8deecec594c8a318976f485addc",
     "the edge embedding is the radial basis times the cutoff function (zero at the cutoff)"),
)
PINNED_RECORDS = tuple(PatchRecord(name, target, "sevenn", SEVENN_VERSIONS, digest, reason)
                       for name, target, digest, reason in _PINS)
# The upstream code the multi-GPU node-halo hooks rely on (``SevenNetAdapter.shard_spec``): each convolution reads
# its input's node features only at the senders and sums its messages at the receivers (sevenn's cuEquivariance
# fused convolution gathers and scatters inside its tensor product), and the interaction block builder makes the
# convolutions the block's only modules that read other atoms. Pinned to sevenn 0.13.0 (sha256 of
# inspect.getsource); another version or source refuses P>1 with the reason.
_INTERACTION_BLOCK = "sevenn.nn.interaction_blocks.NequIP_interaction_block"

def is_sevennet_model(model, model_path: str | None = None) -> bool:
    """Whether ``load_adapter`` loads ``model`` as SevenNet: a name above, or ``"sevennet"`` (with
    ``model_path``, a sevenn checkpoint file)."""
    if not isinstance(model, (str, os.PathLike)):
        return False
    name = os.fspath(model)
    return name in SEVENNET_CHECKPOINTS or name == "sevennet"


def check_load_options(model_dtype: str | None, optimization_profile: str | None, model_path: str | None) -> None:
    """The loading options ``load_adapter`` rejects for SevenNet."""
    from fastmd.adapters import NegotiationError

    if optimization_profile == "legacy":
        raise NegotiationError("optimization_profile='legacy' is MatRIS-only; SevenNet has the profiles "
                               "'topology' (cuEquivariance kernels, the default) and 'generic' (e3nn, its "
                               "reference)")
    if model_dtype == "float64":
        raise ValueError("SevenNet-0 is float32 upstream: model_dtype must be None or 'float32'")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(model, model_path: str | None) -> dict:
    """The file to load: a pinned pretrained checkpoint (inside the sevenn wheel, sha256-checked), or
    the given path."""
    name = os.fspath(model)
    if name in SEVENNET_CHECKPOINTS and model_path is None:
        from sevenn.util import pretrained_name_to_path

        upstream, expected = SEVENNET_CHECKPOINTS[name]
        path = pretrained_name_to_path(upstream)
        found = _sha256(path)
        if found != expected:
            raise ValueError(f"SevenNet checkpoint {name!r} at {path}: sha256 {found} is not the pinned {expected}")
        return {"name": name, "path": str(path), "sha256": found}
    path = model_path if model_path is not None else name
    if path == "sevennet" or not os.path.isfile(path):
        raise FileNotFoundError(f"SevenNet checkpoint not found: {path!r} (pass a checkpoint path or one of "
                                f"{sorted(SEVENNET_CHECKPOINTS)})")
    return {"name": os.path.basename(path), "path": str(path), "sha256": _sha256(path)}


def load_sevennet(model, *, model_path: str | None = None, device: str | None = None,
                  model_dtype: str | None = None, optimization_profile: str | None = None,
                  modal: str | None = None, allow_eager: bool = False) -> "SevenNetAdapter":
    """A :class:`SevenNetAdapter` of a pretrained name or checkpoint path (see :func:`is_sevennet_model`).

    Mirrors ``sevenn.calculator.SevenNetCalculator``'s build (``build_model``,
    ``set_is_batch_data(False)``, the device, ``eval()``), then removes the
    force head and freezes the parameters. The pins are checked first: a
    sevenn whose pinned source changed (or another version) is refused. The
    one-hot patch is applied to the module. Kernels: cuEquivariance
    (uniform_1d) unless ``optimization_profile`` is "generic" (e3nn, the
    reference) or it cannot be built (e3nn, with the reason in
    ``kernel_reason``).
    """
    if optimization_profile == "legacy":
        raise ValueError("optimization_profile='legacy' is MatRIS's (SevenNet: None, 'topology' or 'generic')")
    if model_dtype not in (None, "float32"):
        raise ValueError("SevenNet-0 is float32 upstream (its one-hot features are float32): model_dtype must be "
                         "None or 'float32'")
    from sevenn.checkpoint import SevenNetCheckpoint  # (ImportError: the sevennet extra is not installed)

    pins = _checked_pins()
    device = torch.device("cuda" if device is None and torch.cuda.is_available() else (device or "cpu"))
    checkpoint = _checkpoint(model, model_path)
    source = SevenNetCheckpoint(checkpoint["path"])
    # Keep the selected fidelity across rebuilds when negotiating e3nn/cuEquivariance.
    source.fastmd_modal = modal
    if modal is not None:
        checkpoint["modal"] = modal
    cueq = optimization_profile != "generic" and device.type == "cuda"
    adapter = SevenNetAdapter(_build(source, cueq, device), checkpoint=checkpoint, source=source, pins=pins,
                              allow_eager=allow_eager)
    adapter._loading = True  # the module may be rebuilt (e3nn <-> cuEquivariance) only here
    try:
        if not cueq:
            adapter.kernel_reason = ("optimization_profile='generic': e3nn, the reference" if device.type == "cuda"
                                     else "cuEquivariance runs on CUDA only: e3nn")
        else:
            failed = []
            for option in KERNEL_OPTIONS[:-1]:  # the cuEquivariance methods in order, then e3nn
                try:
                    adapter.select_kernel(KERNEL_SITE, option)
                    _warm_up(adapter.module, device)  # (the build's own method: compiled here, outside any graph)
                    break
                except Exception as error:  # noqa: BLE001 - the next option, with the reason
                    detail = str(error).splitlines()[0][:200] if str(error) else ""
                    failed.append(f"{option} ({type(error).__name__}: {detail})")
            else:
                adapter._install_e3nn()
            if failed:
                adapter.kernel_reason = f"{'; '.join(failed)} unavailable: {adapter.kernel}"
    finally:
        adapter._loading = False
    return adapter


class _FixedModalInput(torch.nn.Module):
    """Upstream's fixed scalar modality, allocated once outside CUDA capture."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.register_buffer("index", torch.tensor(index, dtype=torch.int64))

    def forward(self, data):
        from sevenn import _keys as key

        data[key.MODAL_TYPE] = self.index
        return data


def _checked_pins() -> list:
    """The outcomes of the pins (:data:`PINNED_RECORDS`); raises if one is declined (a sevenn other than the
    validated one, or a pinned source that changed)."""
    pins = apply_patches(PINNED_RECORDS)
    declined = {outcome.record.target: outcome.reason for outcome in pins if not outcome.applied}
    if declined:
        raise RuntimeError("SevenNet: the upstream code this adapter relies on is not the validated one ("
                           + "; ".join(f"{target}: {why}" for target, why in declined.items()) + ")")
    return pins


def _build(source, cueq: bool, device) -> torch.nn.Module:
    """sevenn's module of a checkpoint as its calculator builds it, without the force head, frozen (built
    under the float32 default dtype, the process's restored afterwards)."""
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        module = source.build_model(enable_cueq=cueq)
    finally:
        torch.set_default_dtype(default_dtype)
    module.set_is_batch_data(False)
    modal = getattr(source, "fastmd_modal", None)
    if getattr(module, "modal_map", None):
        if modal not in module.modal_map:
            raise ValueError(f"SevenNet checkpoint requires modal in {sorted(module.modal_map)}, got {modal!r}")
        # Upstream deployment fixes the modal index while retaining the original weights.
        module.prepare_modal_deploy(modal)
        module.replace_module("modal_input_prepare", _FixedModalInput(module.modal_map[modal]))
    elif modal is not None:
        raise ValueError("SevenNet modal was supplied for a single-modality checkpoint")
    module.delete_module_by_key("force_output")
    module = module.to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _convolutions(module) -> list:
    return [layer for name, layer in module._modules.items() if name.endswith("_convolution")]


def _fused(module) -> bool:
    """Whether every convolution is sevenn's cuEquivariance fused scatter convolution."""
    return all(type(layer).__name__ == "IrrepsScatterGatterFusedConvolution" for layer in _convolutions(module))


def _tp_method(module) -> str | None:
    """The cuEquivariance method all the convolutions run (None: e3nn or mixed)."""
    methods = set()
    for layer in _convolutions(module):
        polynomial = getattr(getattr(layer, "convolution", None), "f", None)
        if type(layer).__name__ != "IrrepsScatterGatterFusedConvolution" or polynomial is None:
            return None
        method = getattr(polynomial, "method", None)
        methods.add(method if type(getattr(polynomial, "m", None)).__name__ == _TP_CLASSES.get(method) else None)
    return methods.pop() if len(methods) == 1 else None


def _kernel_of(module) -> str:
    """The installed option: a cuEquivariance method every fused convolution runs, "e3nn" when every
    convolution runs e3nn's tensor product, else "unsupported" (e.g. cuEquivariance without sevenn's fused
    convolution, ``CUEQ_USE_SCATTER_FUSION=0``)."""
    method = _tp_method(module)
    if method is not None:
        return {"uniform_1d": "cueq_uniform_1d", "fused_tp": "cueq_fused_tp"}[method]
    e3nn = all(type(layer).__name__ == "IrrepsConvolution"
               and type(getattr(layer, "convolution", None)).__module__.startswith("e3nn.")
               for layer in _convolutions(module))
    return "e3nn" if e3nn else "unsupported"


def _check_module(module, *, require_exact: bool = True) -> bool:
    """What the adapter relies on in a built module: the force head removed, one modality, and the
    exact-rc padding exact (the edge embedding and its gradient exactly zero at (rc, 0, 0), every radial
    network zero at a zero embedding). Raises ValueError otherwise (another checkpoint's cutoff function or
    radial activation)."""
    if "force_output" in module._modules:
        raise ValueError("SevenNet: the force head must be removed (load the model with load_adapter)")
    if getattr(module, "modal_map", None) and "modal_input_prepare" not in module._modules:
        raise ValueError("SevenNet: select a fixed fidelity with load_sevennet(..., modal=...)")
    parameter = next(module.parameters())
    vector = torch.tensor([[float(module.cutoff), 0.0, 0.0]], dtype=parameter.dtype, device=parameter.device,
                          requires_grad=True)
    with torch.enable_grad():
        embedding = module._modules["edge_embedding"]({"edge_vec": vector})["edge_embedding"]
        (gradient,) = torch.autograd.grad(embedding.sum(), vector)
    zero = torch.zeros_like(embedding.detach())
    weights = [layer.weight_nn(zero) for layer in _convolutions(module)]
    exact = not (bool(embedding.detach().any()) or bool(gradient.any()) or any(bool(weight.any()) for weight in weights))
    if require_exact and not exact:
        raise ValueError("SevenNet: this checkpoint's edge embedding or radial networks are not exactly zero at the "
                         "cutoff, which the exact-rc padding needs")
    return exact


def _warm_up(module, device) -> None:
    """One energy and gradient evaluation on two atoms of the model's first element, 2 Å apart (the
    kernels compile on first use; raises if they cannot run)."""
    z = min(int(z) for z in module.type_map)
    # In the module's dtype, whatever the process default
    vectors = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=next(module.parameters()).dtype, device=device,
                           requires_grad=True)
    data = {"edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=device),
            "edge_vec": vectors, "atomic_numbers": torch.tensor([z, z], dtype=torch.int64, device=device)}
    with torch.enable_grad():
        energy = module(data)["atomic_energy"].sum()
        torch.autograd.grad(energy, vectors)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class SevenNetInput:
    """SevenNet's input over a graph batch: the batch itself and its row numbers (a shape constant); the
    value-dependent tensors are built in each call."""

    batch: GraphBatch
    rows: Tensor  # [edge rows] int64 arange: the rows past batch.n_edges are padding


class SevenNetAdapter:
    """An upstream SevenNet module (``AtomGraphSequential`` without its force head) as a
    :class:`~fastmd.adapters.protocol.ModelAdapter` with an
    :class:`~fastmd.adapters.protocol.EnergyTap` (see the module docstring).

    ``module`` is the module itself; ``kernel`` names the installed kernels
    (one of :data:`KERNEL_OPTIONS`), ``checkpoint`` the loaded file (name,
    path, sha256). :meth:`close` reverts the upstream patches.
    """

    name = "sevennet"
    compiled = False  # sevenn runs no torch.compile

    def __init__(self, module, *, checkpoint: dict | None = None, source=None, pins=None, allow_eager=False) -> None:
        if pins is None:
            pins = _checked_pins()
        self._allow_eager = allow_eager
        self._exact_rc_padding = _check_module(module, require_exact=not allow_eager)
        self.module = module
        self.checkpoint = dict(checkpoint or {})
        self.kernel = _kernel_of(module)
        self.kernel_reason = ""
        self._fused = _fused(module)  # cuEquivariance's fused convolutions: they take an int32 edge index
        self._source = source  # sevenn's checkpoint object: an e3nn <-> cuEquivariance switch rebuilds from it
        # load_sevennet is choosing the kernels: the only time the module may be replaced (an entry point binds
        # the module object it is given, so a later rebuild would leave it running the old one)
        self._loading = False
        # Tensor products replaced by an in-place method switch: kept alive, since a CUDA graph captured before
        # the switch still reads their buffers
        self._retired: list = []
        self._device = next(module.parameters()).device
        self._dtype = next(module.parameters()).dtype
        self._types = torch.tensor(sorted(int(z) for z in module.type_map), dtype=torch.int64)
        self._patches = list(pins) + self._onehot_patch(module)

    def _onehot_patch(self, module) -> list:
        """The one-hot table of ``module`` on its device: ``_atomic_numbers_to_onehot`` then indexes it in
        place (upstream copies its CPU table to the device on every call)."""
        record = PatchRecord(
            "sevennet.onehot_table_on_device", f"{_SEQUENTIAL}.z_to_onehot_tensor", "sevenn", SEVENN_VERSIONS,
            ONEHOT_SOURCE_SHA256, "upstream copies the CPU one-hot table to the device on every call, a host "
            "synchronization that prevents capture", replacement=module.z_to_onehot_tensor.to(self._device),
            source=f"{_SEQUENTIAL}._atomic_numbers_to_onehot", per_instance=True)
        return apply_patches([record], instance=module)

    def _rebind(self, module) -> None:
        """Make ``module`` (a new build, at load) the adapter's: the one-hot patch moves from the old module."""
        self._exact_rc_padding = _check_module(module, require_exact=not self._allow_eager)
        revert_patches(self._patches[-1:])  # the one-hot patch is the last outcome
        self._patches = self._patches[:-1] + self._onehot_patch(module)
        self.module = module
        self._fused = _fused(module)
        self._dtype = next(module.parameters()).dtype

    # --- ModelAdapter -----------------------------------------------------

    def topology_spec(self) -> TopologySpec:
        """Directed edges within the model cutoff, one layer per convolution, no triplets."""
        return TopologySpec(
            cutoff=float(self.module.cutoff),
            num_interactions=len(_convolutions(self.module)),
            triplet_cutoff=None,
            undirected_pairs=False,
            model_dtype=str(self._dtype).removeprefix("torch."),
            edge_order="by_centre",  # SevenNet sums messages per receiver; the order within one is free
        )

    def padding_rule(self) -> PaddingRule:
        """Self-loops whose edge vector is (rc, 0, 0): exactly nothing at the double root of the cutoff."""
        return PaddingRule("exact_rc_loops", pad_vector=(float(self.module.cutoff), 0.0, 0.0))

    def prepare(self, batch: GraphBatch) -> SevenNetInput:
        """The batch and its row numbers; rejects elements outside the model's type map and padding of
        another form (host checks, outside any graph)."""
        rows = batch.species.shape[0]
        if batch.positions is None or batch.positions.shape[0] != rows:
            raise ValueError("SevenNet reads Cartesian positions: an edges-only graph batch without sink rows")
        if batch.padding is not None and not self._exact_rc_padding:
            raise ValueError("This SevenNet checkpoint requires an unpadded eager graph")
        if batch.padding is not None and batch.padding.kind != "exact_rc_loops":
            raise ValueError(f"SevenNet needs exact_rc_loops padding, got {batch.padding.kind!r}")
        unknown = sorted(set(batch.species.tolist()) - set(self._types.tolist()))
        if unknown:
            raise ValueError(f"SevenNet model {self.checkpoint.get('name', '')!r} has no atomic numbers "
                             f"{unknown} (it supports {self._types.tolist()})")
        return SevenNetInput(batch=batch, rows=torch.arange(batch.edge_index.shape[0], device=batch.edge_index.device))

    def isolated_atom_policy(self) -> str:
        """"model": SevenNet evaluates an atom without neighbours itself."""
        return "model"

    def system_inputs(self) -> dict:
        """The fixed fidelity is part of the runtime plan fingerprint."""
        return {"modal": self.checkpoint.get("modal")}

    def reference_energies(self) -> None:
        """None: 7net-0 has no per-element reference table (the species-wise shift is in its energy)."""
        return None

    # --- EnergyTap ----------------------------------------------------------

    def node_energy(self, inputs: SevenNetInput, positions: Tensor, cell: Tensor,
                    edge_shifts: Tensor | None = None) -> Tensor:
        """SevenNet's ``atomic_energy`` ([rows] float32: the scaled atomic energy plus the species shift)
        at ``positions`` and ``cell`` (or each edge's shift ``edge_shifts`` in its own system's cell); padded
        rows take the batch's override vector."""
        batch = inputs.batch
        centre, neighbour = batch.edge_index[:, 0], batch.edge_index[:, 1]
        if edge_shifts is None:
            image = batch.edge_image.to(positions.dtype)
            # row by row: no TF32 GEMM, and an edge's arithmetic does not depend on the row count
            shift = image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]
        else:
            shift = edge_shifts
        vectors = positions.index_select(0, neighbour) + shift - positions.index_select(0, centre)
        if batch.edge_vec_override is not None:  # exact_rc_loops: the rows past n_edges are padding
            real = inputs.rows < batch.n_edges
            vectors = torch.where(real[:, None], vectors, batch.edge_vec_override.to(vectors.dtype))
        # [0] receives (centre), [1] sends (neighbour); e3nn's convolutions scatter with an int64 index
        edge_index = batch.edge_index.t() if self._fused else torch.stack((centre.long(), neighbour.long()))
        data = {
            "edge_index": edge_index,
            "edge_vec": vectors.to(self._dtype),
            "atomic_numbers": batch.species.long(),
        }
        return self.module(data)["atomic_energy"].reshape(-1)

    # --- capability negotiation ---------------------------------------------

    @property
    def upstream(self) -> str:
        """The package providing the model (and the kernels' package with cuEquivariance)."""
        from importlib.metadata import version

        cueq = f", cuequivariance {version('cuequivariance')}" if self.kernel.startswith("cueq") else ""
        return f"sevenn {version('sevenn')}{cueq}"

    def capacity_step(self) -> CapacityStep:
        """SevenNet's capacity tiers (``_CAPACITY_STEP``): every padded edge runs the convolutions' tensor
        products, so a small cell takes a fine pair step."""
        return _CAPACITY_STEP

    def capture_contract(self) -> CaptureContract:
        """Sync-free with the one-hot patch applied (else not captured), exact at the cutoff but not
        beyond (the builders' exact list), held to upstream's call within the eager noise floor (atomic
        scatters: not bitwise), the engine-owned derivative with the stress in eV/Å³, no torch.compile."""
        onehot = next(outcome for outcome in self._patches if outcome.record.name == "sevennet.onehot_table_on_device")
        return CaptureContract(sync_free=onehot.applied and not onehot.reverted and self._exact_rc_padding,
                               exact_cutoff=False, exact_at_cutoff=self._exact_rc_padding,
                               accuracy_class="fp32_noise", force_mode="engine_autograd",
                               stress_unit="eV/A^3", compiled="none")

    def kernel_backends(self) -> tuple[KernelChoice, ...]:
        """The convolutions' tensor products: cuEquivariance uniform_1d, then fused_tp, then e3nn (the
        reference), in every execution mode (see :data:`KERNEL_OPTIONS`)."""
        return (KernelChoice(site=KERNEL_SITE, options=KERNEL_OPTIONS, reference="e3nn"),)

    def system_batching(self) -> str:
        """"union": the module's input (:meth:`node_energy`) holds edges and atoms only, no graph index or
        cell, so every atom's energy comes from its own edges' convolutions and its per-atom readout, and
        replicas evaluated as one graph give each atom its own system's energy. Multi-fidelity
        checkpoints use the same fixed modality for every atom in the union."""
        return "union"

    def edge_shift_input(self) -> bool:
        """True: :meth:`node_energy` takes the engine's per-edge shifts (SevenNet reads edge vectors only)."""
        return True

    def select_kernel(self, site: str, option: str) -> None:
        """Install ``option`` at the convolutions through sevenn's own switches, then run it once (the
        kernels compile on first use) and check that every convolution runs it. A cuEquivariance method
        switch is in place: each convolution's tensor product is rebuilt with the method, the module object
        and its parameters stay, and the replaced tensor products are kept (a CUDA graph captured before
        the switch keeps running the old method). e3nn <-> cuEquivariance rebuilds the module, which is
        chosen when the model is loaded (``optimization_profile``): later it raises (under a profile that
        needs it, the negotiation falls back past it or, for the reference, raises). All or nothing: on an
        error the module stays as it was. Call outside any CUDA graph."""
        if site != KERNEL_SITE or option not in KERNEL_OPTIONS:
            raise ValueError(f"SevenNet kernel site {KERNEL_SITE!r} has options {KERNEL_OPTIONS}, got {site!r}/{option!r}")
        if option == self.kernel:
            return
        family = "cueq" if self._fused else self.kernel  # "e3nn", or "unsupported" (always rebuilt)
        rebuild = ("cueq" if option.startswith("cueq") else "e3nn") != family
        if rebuild and not self._loading:
            raise RuntimeError(f"SevenNet: {self.kernel!r} -> {option!r} rebuilds the module, which is chosen when "
                               "the model is loaded; load it with the profile that selects it")
        if option == "e3nn":
            self._install_e3nn()
            return
        if self._device.type != "cuda":
            raise RuntimeError("cuEquivariance kernels run on CUDA only")
        module = _build(self._source, True, self._device) if rebuild else self.module
        if not _fused(module):
            raise RuntimeError("sevenn built cuEquivariance without the fused scatter convolution (e3nn fallback, "
                               "or CUEQ_USE_SCATTER_FUSION=0)")
        method = _TP_METHODS[option]
        layers = _convolutions(module)
        kept = [layer.convolution for layer in layers]
        try:
            for layer in layers:
                if getattr(getattr(layer.convolution, "f", None), "method", None) != method:
                    layer.convolution = layer.convolution_cls(**{**layer.convolution_kwargs,
                                                                 "tp_method": method}).to(self._device)
            if _tp_method(module) != method:
                raise RuntimeError(f"cuEquivariance method {method!r} is not available (its ops fell back)")
            _warm_up(module, self._device)
        except Exception:
            for layer, convolution in zip(layers, kept):
                layer.convolution = convolution
            raise
        if module is not self.module:
            self._rebind(module)
        else:
            self._retired += [old for layer, old in zip(layers, kept) if layer.convolution is not old]
        self.kernel = option
        self.kernel_reason = ""

    def _install_e3nn(self) -> None:
        """The e3nn module (sevenn's reference build), at load."""
        if self.kernel == "e3nn":
            return
        module = _build(self._source, False, self._device)
        _warm_up(module, self._device)
        self._rebind(module)
        self.kernel = "e3nn"
        self.kernel_reason = ""



    # --- upstream patches ---------------------------------------------------

    def upstream_patches(self) -> list:
        """The outcomes of the pins and the one-hot patch (:mod:`fastmd.adapters.patches`)."""
        return list(self._patches)

    def close(self) -> None:
        """Revert the one-hot patch (the module copies its table per call again); idempotent."""
        revert_patches(self._patches)

