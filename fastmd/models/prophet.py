"""Kairos Materials' Prophet-OAME-MBD with FastMD's energy derivative.

The upstream ``node_energies`` method consumes the engine's radius graph.
Prophet names the neighbour ``sender`` and the centre ``receiver``; its
displacement is neighbour minus centre plus the neighbour's lattice shift.
Species are indices into the checkpoint's sorted atomic-number list.

The released multi-cutoff model selects edges with boolean masks in each
shell. Those selections have dynamic shapes and synchronize on CUDA, so this
adapter deliberately supports eager execution only. Far self-loop padding
is removed by those same masks. No upstream methods are patched.

``prophet-mlip`` is optional and imported only by ``load_prophet``. Named
models require an explicit local checkpoint (there are no implicit network
downloads). The official checkpoint contains a JSON header and state dict.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import warnings

import torch
from torch import Tensor

from fastmd.adapters.protocol import CaptureContract, GraphBatch, PaddingRule, TopologySpec


PROPHET_NAMES = frozenset({"prophet", "prophet-oame-mbd", "prophet_oame_mbd"})


def is_prophet_model(model, model_path=None) -> bool:
    return isinstance(model, (str, os.PathLike)) and os.fspath(model).lower() in PROPHET_NAMES


def load_prophet(model="prophet-oame-mbd", *, model_path=None, device=None,
                 model_dtype=None, optimization_profile=None, use_kernel=None):
    """Load the official module, optionally with its OpenEquivariance kernels.

    The default uses OpenEquivariance on CUDA when installed, and e3nn on
    CPU or with ``optimization_profile='generic'``. A failed automatic
    kernel build falls back to e3nn with a warning and ``kernel_reason``.
    Explicit ``use_kernel=True`` fails instead of changing the backend.
    """
    if model_path is None:
        raise ValueError("Prophet-OAME-MBD requires an explicit local model_path")
    if model_dtype not in (None, "float32"):
        raise ValueError("Prophet-OAME-MBD supports model_dtype=None or 'float32'")
    if optimization_profile == "legacy":
        raise ValueError("optimization_profile='legacy' is MatRIS-only")
    if use_kernel is not None and not isinstance(use_kernel, bool):
        raise TypeError("use_kernel must be None, True, or False")
    try:
        from prophet import load_model
    except ImportError as error:
        raise ImportError("Prophet requires the optional prophet-mlip package; install fastmd[prophet]") from error

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prophet checkpoint not found: {path}")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if use_kernel and target.type != "cuda":
        raise ValueError("Prophet OpenEquivariance kernels require a CUDA device")
    kernel = (target.type == "cuda" and optimization_profile != "generic"
              and importlib.util.find_spec("openequivariance") is not None) if use_kernel is None else use_kernel
    reason = "official OpenEquivariance tensor-product convolutions" if kernel else "official e3nn tensor products"
    try:
        module, config = load_model(path, use_kernel=kernel)
    except Exception as error:
        if not kernel or use_kernel is True:
            raise
        detail = str(error).splitlines()[0] if str(error) else "no error message"
        reason = f"OpenEquivariance unavailable ({type(error).__name__}: {detail}); e3nn"
        warnings.warn(reason, RuntimeWarning, stacklevel=2)
        kernel = False
        module, config = load_model(path, use_kernel=False)
    module = module.to(device=target, dtype=torch.float32).eval()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return ProphetAdapter(module, config=config,
                          checkpoint={"name": os.fspath(model), "path": str(path), "sha256": digest},
                          kernel="openequivariance" if kernel else "e3nn", kernel_reason=reason)


class ProphetAdapter:
    name = "prophet"
    compiled = False

    def __init__(self, module, *, config, checkpoint=None, kernel="e3nn", kernel_reason=None):
        self.module = module.eval()
        self.config = dict(config)
        self.checkpoint = dict(checkpoint or {})
        self.kernel = kernel
        self.kernel_reason = kernel_reason or f"official {kernel} tensor products"
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        numbers = sorted(int(z) for z in config["atomic_numbers"])
        if not numbers or numbers[0] < 1 or len(set(numbers)) != len(numbers):
            raise ValueError("Prophet checkpoint atomic_numbers must be distinct positive atomic numbers")
        if not config["cutoffs"] or min(config["cutoffs"]) <= 0:
            raise ValueError("Prophet checkpoint requires positive interaction cutoffs")
        # Config's model indexing agrees with the upstream graph preprocessor.
        target = next(module.parameters()).device
        self._species_indices = torch.full((max(numbers) + 1,), -1, dtype=torch.long, device=target)
        self._species_indices[numbers] = torch.arange(len(numbers), device=target)
        self._checked_species = None

    def topology_spec(self) -> TopologySpec:
        return TopologySpec(cutoff=float(max(self.config["cutoffs"])),
                            num_interactions=int(self.config["n_layers"]), model_dtype="float32")

    def padding_rule(self) -> PaddingRule:
        return PaddingRule("far_self_loops")

    def prepare(self, batch: GraphBatch) -> GraphBatch:
        if batch.species is not self._checked_species:
            numbers = batch.species.to(torch.long)
            if bool(((numbers < 1) | (numbers >= len(self._species_indices))).any()):
                raise ValueError("Atomic number outside the Prophet checkpoint species table")
            if bool((self._species_indices[numbers] < 0).any()):
                raise ValueError("Atomic number absent from the Prophet checkpoint species table")
            self._checked_species = batch.species
        return batch

    def node_energy(self, batch: GraphBatch, positions: Tensor, cell: Tensor) -> Tensor:
        senders = batch.edge_src.to(torch.long)
        receivers = batch.edge_dst.to(torch.long)
        species = self._species_indices[batch.species.to(torch.long)]
        # Match Prophet.forward's arithmetic and directed-edge convention.
        displacement = positions[senders] - positions[receivers]
        displacement = torch.baddbmm(displacement.unsqueeze(1),
                                    batch.edge_image.to(positions.dtype).unsqueeze(1),
                                    cell.unsqueeze(0).expand(len(senders), -1, -1)).squeeze(1)
        if batch.edge_vec_override is not None:
            displacement = displacement + batch.edge_vec_override.to(positions.dtype)
        return self.module.node_energies(displacement, species, senders, receivers).reshape(-1)

    def isolated_atom_policy(self):
        return "model"

    def reference_energies(self):
        # Isolated atoms still traverse the learned node network.
        return None

    def capture_contract(self) -> CaptureContract:
        return CaptureContract(sync_free=False, exact_cutoff=True,
                               force_mode="engine_autograd", stress_unit="eV/A^3")

    def close(self):
        """No upstream patches or external resources to release."""


def check_load_options(model_dtype, optimization_profile, model_path):
    """Validate registry options without loading the optional framework."""
    from fastmd.adapters.protocol import NegotiationError
    if optimization_profile == "legacy":
        raise NegotiationError("optimization_profile='legacy' is MatRIS-only")
