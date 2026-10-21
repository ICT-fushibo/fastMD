"""MACE behind the adapter protocol: the upstream mace-torch model with an energy tap.

mace-torch is optional: this module imports it only when a model is loaded
(``pip install fastmd[mace]``), so ``fastmd`` imports without it. The adapter
wraps the upstream ``ScaleShiftMACE`` as loaded (or as upstream's cuEquivariance
converter returns it); nothing of MACE is copied here. What the adapter
declares and does:

* **Topology**: directed edges within ``r_max``, no triplets, so the builders
  run edges only (``TopologySpec.triplet_cutoff`` None).
* **Padding**: far self-loops (``PaddingRule("far_self_loops")``): a padded row
  is a self-loop of a real atom, round robin, with an integer image at least
  twice ``r_max`` long. MACE's radial cutoff is exact (value and gradient 0 at
  ``r >= r_max``), and the length is never 0 (the Bessel basis sin(x)/x is NaN
  there). The integer image keeps the loop far under a strain too.
* **Capacity tiers** (``capacity_step``): 32 undirected pairs with at least 32
  of padding, every capture holding 1.02 times the pair demand (a heating
  cell's pair count grows; taken where the configuration keeps the engine's
  default tiers, which pad a 135-atom cell with hundreds of pairs that run
  every layer).
* **Energy tap**: ``node_energy`` returns MACE's own ``node_energy`` (float64 for
  every model dtype; it includes the per-element E0s and the scale/shift), and
  the engine differentiates its sum (``fastmd.engine.energy_derivative``):
  MACE's own force and stress autograd is off (``compute_force=False``).
* **Inputs**: ``prepare(batch)`` keeps the batch and its shape constants;
  each call maps the engine's edges to MACE's: ``edge_index = [neighbour,
  centre]`` as int64 ``[2, E]`` (sender, receiver: the receivers stay sorted
  over the real rows) and ``unit_shifts = -edge_image``, so MACE's vector
  ``x[receiver] - x[sender] + unit_shifts @ cell`` is the engine's edge reversed,
  from the neighbour's image to the centre.
* **Kernels**: cuEquivariance through upstream's converter
  (``mace.cli.convert_e3nn_cueq``) by default, e3nn (the reference) for
  ``optimization_profile="generic"`` or when cuEquivariance is unavailable.
  Serialized legacy CG basis phases are preserved by a one-time weight
  correction; unsupported basis changes reject conversion and keep e3nn.
* **Precision**: the module runs in ``model_dtype`` (float64 or float32).
* **Isolated atoms**: ``"model"``: MACE evaluates an atom without neighbours
  itself (MACE-MP-0's energy of an isolated atom is not its E0).

Checkpoints by name (``mace_mp_0``, ``mace_mpa_0``) come from mace-torch's own
downloader and are checked against a pinned sha256; a path loads that file.
Loading unpickles the file (``torch.load(weights_only=False)``, as mace-torch
does): load only checkpoints you trust.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fastmd.adapters.protocol import (
    CapacityStep, CaptureContract, GraphBatch, KernelChoice, PaddingRule, TopologySpec,
)

# Pretrained checkpoints by name: (mace-torch's download name, sha256 of the file).
MACE_CHECKPOINTS = {
    "mace_mp_0": ("medium", "01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845"),
    "mace_mpa_0": ("medium-mpa-0", "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"),
}
# The kernel site and its options, fastest first; e3nn is the reference.
KERNEL_SITE = "equivariant_products"
KERNEL_OPTIONS = ("cueq", "e3nn")

_FAR_SELF_LOOPS = PaddingRule("far_self_loops")
# Measured capacity tiers (GPU-resident MD, whole step, GB200; see capacity_step).
_CAPACITY_STEP = CapacityStep(u_step=32, min_pad_u=32, u_thermal_factor=1.02)
_ELEMENTS = 118  # columns of the reference-energy row (Z - 1)
# The upstream code the multi-GPU node-halo hooks rely on (``MACEAdapter.shard_spec``): the model's forward, which
# calls each interaction block with the node features as a keyword, and the interaction blocks of the pretrained
# checkpoints, which sum their messages at the receiver and read the senders' node features per edge. Pinned to
# mace-torch 0.3.16 (sha256 of inspect.getsource); another version or source refuses P>1 with the reason.
_SHARD_PIN_VERSIONS = ("0.3.16",)


def is_mace_model(model, model_path: str | None = None) -> bool:
    """Whether ``load_adapter`` loads ``model`` as MACE: a name above, ``"mace"`` (with
    ``model_path``), or a path to a ``.model`` file (mace-torch's checkpoint suffix)."""
    if not isinstance(model, (str, os.PathLike)):
        return False
    name = os.fspath(model)
    return name in MACE_CHECKPOINTS or name == "mace" or (name.endswith(".model") and model_path is None)


def check_load_options(model_dtype: str | None, optimization_profile: str | None, model_path: str | None) -> None:
    """The loading options ``load_adapter`` rejects for MACE (float32 and float64 load)."""
    from fastmd.adapters import NegotiationError

    if optimization_profile == "legacy":
        raise NegotiationError("optimization_profile='legacy' is MatRIS-only; MACE has the profiles 'topology' "
                               "(cuEquivariance kernels, the default) and 'generic' (e3nn, its reference)")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(model, model_path: str | None) -> dict:
    """The file to load: a pinned pretrained checkpoint (downloaded by mace-torch, sha256-checked), or
    the given path."""
    name = os.fspath(model)
    if name in MACE_CHECKPOINTS and model_path is None:
        from mace.calculators.foundations_models import download_mace_mp_checkpoint

        upstream, expected = MACE_CHECKPOINTS[name]
        path = download_mace_mp_checkpoint(upstream)
        found = _sha256(path)
        if found != expected:
            raise ValueError(f"MACE checkpoint {name!r} at {path}: sha256 {found} is not the pinned {expected}")
        return {"name": name, "path": str(path), "sha256": found}
    path = model_path if model_path is not None else name
    if path == "mace" or not os.path.isfile(path):
        raise FileNotFoundError(f"MACE checkpoint not found: {path!r} (pass a .model path or one of "
                                f"{sorted(MACE_CHECKPOINTS)})")
    return {"name": os.path.basename(path), "path": str(path), "sha256": _sha256(path)}


def load_mace(model, *, model_path: str | None = None, device: str | None = None, model_dtype: str | None = None,
              optimization_profile: str | None = None, head: str | None = None) -> "MACEAdapter":
    """A :class:`MACEAdapter` of a pretrained name or checkpoint path (see :func:`is_mace_model`).

    Mirrors ``mace.calculators.MACECalculator``'s load: the module on
    ``device`` in ``model_dtype`` (default float32), converted to
    cuEquivariance unless ``optimization_profile`` is "generic" (e3nn, the
    reference) or the conversion is unavailable (e3nn, with the reason in
    ``kernel_reason``), parameters frozen, ``eval()``. The process default
    dtype is restored afterwards (upstream's converter sets it).
    """
    if optimization_profile == "legacy":
        raise ValueError("optimization_profile='legacy' is MatRIS's (MACE: None, 'topology' or 'generic')")
    dtype = {None: torch.float32, "float32": torch.float32, "float64": torch.float64}[model_dtype]
    device = torch.device("cuda" if device is None and torch.cuda.is_available() else (device or "cpu"))
    checkpoint = _checkpoint(model, model_path)
    default_dtype = torch.get_default_dtype()
    try:
        module = torch.load(checkpoint["path"], map_location=device, weights_only=False)
        module = module.to(device=device, dtype=dtype)
        adapter = MACEAdapter(module, head=head, checkpoint=checkpoint)
        if optimization_profile == "generic":
            adapter.kernel_reason = "optimization_profile='generic': e3nn, the reference"
        else:
            try:
                adapter.select_kernel(KERNEL_SITE, "cueq")
            except Exception as error:  # noqa: BLE001 - any failure keeps e3nn, with the reason
                detail = str(error).splitlines()[0][:200] if str(error) else ""
                adapter.kernel_reason = f"cuEquivariance unavailable ({type(error).__name__}: {detail}); e3nn"
    finally:
        torch.set_default_dtype(default_dtype)
    return adapter


def _warm_up(module, head: int) -> None:
    """One energy and force evaluation of ``module`` on two atoms of its first element, 2 Å apart in a
    20 Å box (raises if the kernels cannot run)."""
    parameter = next(module.parameters())
    device, dtype = parameter.device, parameter.dtype
    elements = int(module.atomic_numbers.shape[0])
    node_attrs = torch.zeros((2, elements), dtype=dtype, device=device)
    node_attrs[:, 0] = 1.0
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype, device=device, requires_grad=True)
    data = {
        "positions": positions,
        "node_attrs": node_attrs,
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=device),
        "unit_shifts": torch.zeros((2, 3), dtype=dtype, device=device),
        "shifts": torch.zeros((2, 3), dtype=dtype, device=device),
        "cell": 20.0 * torch.eye(3, dtype=dtype, device=device),
        "batch": torch.zeros(2, dtype=torch.int64, device=device),
        "ptr": torch.tensor([0, 2], dtype=torch.int64, device=device),
        "head": torch.tensor([head], dtype=torch.int64, device=device),
    }
    with torch.enable_grad():
        out = module(data, training=False, compute_force=False)
        torch.autograd.grad(out["node_energy"].sum(), positions)
    if parameter.device.type == "cuda":
        torch.cuda.synchronize(device)


def _align_cueq_contraction_basis(source, target) -> list[dict]:
    """Preserve serialized CG phases when converting legacy MACE contractions.

    Upstream copies weights but rebuilds the original-MACE polynomial basis.
    Some old checkpoints (including large-0b2) store opposite signs for a few
    non-scalar, degree-three basis columns. Match those columns to cuEq's own
    basis and apply the signs to its weights once, before capture. Never alter
    the e3nn reference. An unsupported basis difference rejects conversion,
    letting load_mace keep the reference backend instead of changing the model.
    """
    from cuequivariance.group_theory.experimental.mace.symmetric_contractions import U_matrix_real

    corrections = []
    for layer, (original, converted) in enumerate(zip(source.products, target.products)):
        src, dst = original.symmetric_contractions, converted.symmetric_contractions
        # The reduced-CG conversion has a different, upstream-owned weight map.
        if getattr(dst, "projection", None) is None:
            continue
        if len(src.contractions) != len(dst.irreps_out):
            raise ValueError("MACE/cuEq contraction irreps differ")
        signs = []
        for contraction, (_, irrep) in zip(src.contractions, dst.irreps_out):
            degree_max = contraction.correlation
            for degree in range(degree_max, 0, -1):
                suffix = "max" if degree == degree_max else str(degree_max - degree - 1)
                if bool(getattr(contraction, f"weights_{suffix}_zeroed", False)):
                    continue  # same omitted weight block as upstream's converter
                stored = contraction.U_tensors(degree).detach().cpu()
                basis = torch.from_numpy(U_matrix_real(dst.irreps_in.set_mul(1), irrep, degree))
                if irrep.dim == 1:
                    basis = basis.squeeze(0)
                if basis.shape != stored.shape:
                    raise ValueError("MACE/cuEq contraction basis shapes differ")
                old = stored.double().reshape(-1, stored.shape[-1])
                new = basis.double().reshape_as(old)
                phase = torch.where((old * new).sum(0) < 0, -1.0, 1.0)
                tolerance = max(1e-12, 32 * torch.finfo(stored.dtype).eps)
                if not torch.allclose(old, new * phase, atol=tolerance, rtol=tolerance):
                    raise ValueError("MACE/cuEq contraction basis differs beyond column signs")
                changed = torch.where(phase < 0)[0].tolist()
                if changed:
                    corrections.append(dict(layer=layer, irrep=str(irrep), degree=degree, columns=changed))
                signs.append(phase)
        phase = torch.cat(signs).to(dst.weight)
        if phase.numel() != dst.weight.shape[1]:
            raise ValueError("MACE/cuEq contraction weight layout differs")
        with torch.no_grad():
            dst.weight.mul_(phase[None, :, None])
    return corrections


@dataclass
class MACEInput:
    """MACE's input over a graph batch: the batch itself and its shape constants (every row is one
    system, graph 0, read with one head). The value-dependent tensors are built in each call."""

    batch: GraphBatch
    graph_index: Tensor  # [rows] int64 zeros
    ptr: Tensor  # [2] int64: (0, rows)
    head: Tensor  # [1] int64: the head of the system
    z_index: Tensor  # [119] int64: atomic number -> row of the one-hot table (-1: not in the model)
    one_hot: Tensor  # [elements, elements] identity in the model's dtype
    # [119, elements] in the model's dtype: row Z is the one-hot row of Z (NaN outside the model), so a
    # call reads the node attributes with one lookup of the int32 species
    node_attrs_by_z: Tensor
    minus_one: Tensor  # [1] -1 in the model's dtype: unit_shifts = -edge_image in one kernel


class MACEAdapter:
    """An upstream MACE module (``ScaleShiftMACE``) as a :class:`~fastmd.adapters.protocol.ModelAdapter`
    with an :class:`~fastmd.adapters.protocol.EnergyTap` (see the module docstring).

    ``module`` is the module itself (the cuEquivariance one after
    :meth:`select_kernel`); ``kernel`` names the installed kernels ("cueq" or
    "e3nn") and ``checkpoint`` the loaded file (name, path, sha256).
    """

    name = "mace"
    compiled = False  # upstream MACE does not run torch.compile

    def __init__(self, module, *, head: str | None = None, checkpoint: dict | None = None,
                 kernel: str = "e3nn") -> None:
        self.module = module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.checkpoint = dict(checkpoint or {})
        self.kernel = kernel
        self.kernel_reason = ""
        self.kernel_basis_corrections: list[dict] = []
        self._e3nn_module = self.module if kernel == "e3nn" else None
        self._in_use = False  # prepare() was called: the module is bound to a runtime from now on
        heads = [str(name) for name in (getattr(module, "heads", None) or ["Default"])]
        if head is None:
            defaults = [name for name in heads if name.lower() == "default"]
            if len(heads) > 1 and not defaults:
                raise ValueError(f"MACE model has heads {heads}: pass head=")
            head = heads[0] if len(heads) == 1 else defaults[0]
        if head not in heads:
            raise ValueError(f"MACE head {head!r} is not one of {heads}")
        self.head = head
        self._head_index = heads.index(head)
        # The element lookups (atomic number -> one-hot row of MACE's z table) and the shape constants,
        # per (rows, device): built once and never freed, since captured graphs read them.
        self._constants: dict[tuple, tuple[Tensor, ...]] = {}

    # --- ModelAdapter -----------------------------------------------------

    def topology_spec(self) -> TopologySpec:
        """Directed edges within ``r_max``, ``num_interactions`` layers, no triplets."""
        parameter = next(self.module.parameters())
        return TopologySpec(
            cutoff=float(self.module.r_max),
            num_interactions=int(self.module.num_interactions),
            triplet_cutoff=None,
            undirected_pairs=False,
            model_dtype=str(parameter.dtype).removeprefix("torch."),
            edge_order="by_centre",  # MACE sums messages per receiver; the order within one is free
        )

    def padding_rule(self) -> PaddingRule:
        """Far self-loops: exactly nothing for MACE's exact cutoff, never a zero length."""
        return _FAR_SELF_LOOPS

    def prepare(self, batch: GraphBatch) -> MACEInput:
        """The batch and its shape constants; rejects elements outside the model's z table (host check,
        outside any graph)."""
        rows = batch.species.shape[0]
        if batch.positions is None or batch.positions.shape[0] != rows:
            raise ValueError("MACE reads Cartesian positions: an edges-only graph batch without sink rows")
        device = batch.species.device
        key = (rows, str(device))
        if key not in self._constants:
            numbers = self.module.atomic_numbers.to(device=device, dtype=torch.int64)
            z_index = torch.full((_ELEMENTS + 1,), -1, dtype=torch.int64, device=device)
            z_index[numbers] = torch.arange(numbers.shape[0], device=device)
            dtype = next(self.module.parameters()).dtype
            one_hot = torch.eye(numbers.shape[0], dtype=dtype, device=device)
            # NaN rows for elements outside the model: numbers changed in place after this host check (a
            # captured graph's new system) give non-finite results, never a silently wrong energy
            node_attrs_by_z = torch.full((_ELEMENTS + 1, numbers.shape[0]), float("nan"), dtype=dtype, device=device)
            node_attrs_by_z[numbers] = one_hot
            self._constants[key] = (torch.zeros(rows, dtype=torch.int64, device=device),
                                    torch.tensor([0, rows], dtype=torch.int64, device=device),
                                    torch.tensor([self._head_index], dtype=torch.int64, device=device),
                                    z_index, one_hot, node_attrs_by_z,
                                    torch.full((1,), -1.0, dtype=dtype, device=device))
        graph_index, ptr, head, z_index, one_hot, node_attrs_by_z, minus_one = self._constants[key]
        species = batch.species.long()
        if bool(((species < 1) | (species > _ELEMENTS) | (z_index[species.clamp(1, _ELEMENTS)] < 0)).any()):
            supported = sorted(int(z) for z in self.module.atomic_numbers.tolist())
            raise ValueError(f"MACE model {self.checkpoint.get('name', '')!r} supports atomic numbers {supported}")
        self._in_use = True
        return MACEInput(batch=batch, graph_index=graph_index, ptr=ptr, head=head, z_index=z_index,
                         one_hot=one_hot, node_attrs_by_z=node_attrs_by_z, minus_one=minus_one)

    def isolated_atom_policy(self) -> str:
        """"model": MACE evaluates an atom without neighbours itself."""
        return "model"

    def reference_energies(self) -> Tensor:
        """The E0 of each element as a ``[1, 118]`` float64 row (column ``Z - 1``; NaN for elements
        outside the model), read-only."""
        energies = torch.atleast_2d(self.module.atomic_energies_fn.atomic_energies.detach())[self._head_index]
        numbers = self.module.atomic_numbers.long().to(energies.device)
        row = torch.full((1, _ELEMENTS), float("nan"), dtype=torch.float64, device=energies.device)
        row[0, numbers - 1] = energies.to(torch.float64)
        return row

    # --- EnergyTap ----------------------------------------------------------

    def node_energy(self, inputs: MACEInput, positions: Tensor, cell: Tensor,
                    edge_shifts: Tensor | None = None) -> Tensor:
        """MACE's ``node_energy`` ([rows] float64: E0 + scaled interaction energy per atom) at
        ``positions`` and ``cell`` (or each edge's shift ``edge_shifts`` in its own system's cell), with MACE's
        own derivatives off."""
        batch = inputs.batch
        dtype = positions.dtype
        # one_hot[z_index[species]] in one lookup, and -edge_image in the model's dtype in one kernel
        node_attrs = inputs.node_attrs_by_z.index_select(0, batch.species)
        unit_shifts = torch.mul(batch.edge_image, inputs.minus_one if inputs.minus_one.dtype == dtype
                                else inputs.minus_one.to(dtype))
        if edge_shifts is not None:
            shifts = -edge_shifts  # MACE's edge is the engine's reversed (the row-by-row bits, negated exactly)
        elif dtype == torch.float64:
            shifts = unit_shifts @ cell  # an FP64 GEMM (no TF32)
        else:
            # row by row: no TF32 GEMM for FP32
            shifts = unit_shifts[:, 0:1] * cell[0] + unit_shifts[:, 1:2] * cell[1] + unit_shifts[:, 2:3] * cell[2]
        data = {
            "positions": positions,
            "node_attrs": node_attrs,
            # (sender, receiver) = (neighbour, centre), int64
            "edge_index": torch.stack((batch.edge_index[:, 1], batch.edge_index[:, 0])).to(torch.int64),
            "unit_shifts": unit_shifts,
            "shifts": shifts,
            "cell": cell,
            "batch": inputs.graph_index,
            "ptr": inputs.ptr,
            "head": inputs.head,
        }
        out = self.module(data, training=False, compute_force=False, compute_virials=False,
                          compute_stress=False, compute_displacement=False)
        return out["node_energy"]

    # --- capability negotiation ---------------------------------------------

    @property
    def upstream(self) -> str:
        """The package providing the model (and the kernels' package with cuEquivariance)."""
        from importlib.metadata import version

        cueq = f", cuequivariance {version('cuequivariance')}" if self.kernel == "cueq" else ""
        return f"mace-torch {version('mace-torch')}{cueq}"

    def capture_contract(self) -> CaptureContract:
        """No host synchronization in a call (captured whole), an exact radial cutoff, held to upstream's
        own call within the eager noise floor (atomic scatters: not bitwise), the engine-owned
        derivative with the stress in eV/Å³, no torch.compile."""
        return CaptureContract(sync_free=True, exact_cutoff=True, accuracy_class="fp32_noise",
                               force_mode="engine_autograd", stress_unit="eV/A^3", compiled="none")

    def capacity_step(self) -> CapacityStep:
        """MACE's capacity tiers (``_CAPACITY_STEP``): every padded edge runs the interactions' tensor
        products, so a small cell takes a fine pair step."""
        return _CAPACITY_STEP

    def kernel_backends(self) -> tuple[KernelChoice, ...]:
        """The equivariant products: cuEquivariance (fastest) or e3nn (the reference)."""
        return (KernelChoice(site=KERNEL_SITE, options=KERNEL_OPTIONS, reference="e3nn"),)

    def select_kernel(self, site: str, option: str) -> None:
        """Install ``option`` ("cueq" or "e3nn") at the kernel site through upstream's own switch
        (``mace.cli.convert_e3nn_cueq.run``, CUDA only), and run it once on a two-atom input (the kernels
        compile on first use). All or nothing: on an error the module stays. The kernels are chosen before
        a runtime binds the module: once :meth:`prepare` has run, only the installed option is accepted."""
        if site != KERNEL_SITE or option not in KERNEL_OPTIONS:
            raise ValueError(f"MACE kernel site {KERNEL_SITE!r} has options {KERNEL_OPTIONS}, got {site!r}/{option!r}")
        if option == self.kernel:
            return
        if self._in_use:
            raise RuntimeError(f"MACE kernels are chosen when the model is loaded (installed: {self.kernel!r}); "
                               f"load it with the profile that selects {option!r}")
        if option == "e3nn":
            if self._e3nn_module is None:
                raise RuntimeError("the e3nn module of this adapter is not kept (it was loaded with cuEquivariance)")
            module = self._e3nn_module
            corrections = []
        else:
            parameter = next(self.module.parameters())
            if parameter.device.type != "cuda":
                raise RuntimeError("cuEquivariance kernels run on CUDA only")
            import cuequivariance_torch  # noqa: F401  (the kernels; ImportError: not installed)
            from mace.cli.convert_e3nn_cueq import run

            default_dtype = torch.get_default_dtype()
            try:
                with torch.cuda.device(parameter.device):
                    # The literal "cuda": upstream fuses the convolution only for that string.
                    module = run(self.module, device="cuda").to(parameter.device)
                    corrections = _align_cueq_contraction_basis(self.module, module)
            finally:
                torch.set_default_dtype(default_dtype)
            module = module.eval()
            for weight in module.parameters():
                weight.requires_grad_(False)
            _warm_up(module, self._head_index)
        self.module = module
        self.kernel = option
        self.kernel_reason = ""
        self.kernel_basis_corrections = corrections

    def system_inputs(self) -> dict[str, Any]:
        """The static per-system input: MACE's head."""
        return {"head": self.head}

    def edge_shift_input(self) -> bool:
        """True: :meth:`node_energy` takes the engine's per-edge shifts (MACE reads ``shifts`` per edge; its
        ``cell`` and ``unit_shifts`` only for a stress, which the engine takes from its own strain)."""
        return True

    def system_batching(self) -> str | None:
        """"union" for upstream's ``ScaleShiftMACE`` itself (as loaded, or as upstream's cuEquivariance
        converter rebuilds it): its node energies come from messages over edges and a per-atom readout, so
        replicas evaluated as one graph give each atom its own system's energy. None for any other module: a
        subclass (e.g. a long-range term per graph) or a joint embedding (per-graph features read through the
        graph index, which the adapter sets to one graph)."""
        module = self.module
        if (type(module).__name__, type(module).__module__) != ("ScaleShiftMACE", "mace.modules.models"):
            return None
        if hasattr(module, "joint_embedding"):
            return None
        return "union"


