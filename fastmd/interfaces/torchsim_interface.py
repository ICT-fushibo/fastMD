"""torch-sim ModelInterface wrapper for MatRIS."""

from __future__ import annotations

import torch
from torch import Tensor
from pymatgen.io.ase import AseAtomsAdaptor

from fastmd.adapters import as_adapter, load_adapter
from fastmd.adapters.protocol import ModelAdapter, ModelOwnedDerivative
from fastmd.graph import RadiusGraph, atoms_to_graph_gpu
from fastmd.graph.gpu_graph_builder import op_available
from fastmd.config import InferenceConfig
from fastmd.engine.entry_config import (
    apply_checkpoint, configure_entrypoint, config_summary, current_adapter, release_adapter, scoped_call,
    use_adapter,
)
from fastmd.engine import saved_recompute

try:
    from torch_sim import units as _ts_units
    from torch_sim.models.interface import ModelInterface
    from torch_sim.state import SimState
except ImportError as exc:
    raise ImportError(
        "torch-sim is not installed. Please install torch-sim to use MLIPModel."
    ) from exc

_GPa = _ts_units.MetalUnits.pressure * 10000


class MLIPModel(ModelInterface):
    """torch-sim compatible wrapper for MatRIS.

    Converts torch-sim SimState inputs to RadiusGraph format, runs the
    MatRIS model, and returns outputs in torch-sim format.

    Output units:
        energy : eV  (total, per system)
        forces : eV/Å
        stress : eV/Å³

    Args:
        model: A :class:`~fastmd.models.matris.MatRIS` instance, a model adapter
            (``fastmd.adapters``), or the name of a pretrained checkpoint (e.g.
            ``"matris_10m_oam"``).
        compute_stress: Whether to compute the stress tensor.
        compute_magmom: Whether to compute per-atom magnetic moments.
        device: Device to run inference on. Defaults to the device of the
            loaded model parameters.
        dtype: Floating-point dtype exposed to torch-sim. Defaults to
            ``torch.float32``.
        enable_compile: Whether to compile interaction blocks.
        config: Unified inference configuration. Omission uses eager execution,
            strict isolated-atom rejection, and checkpointing off.

    Example::

        from fastmd.interfaces.torchsim_interface import MLIPModel

        model = MLIPModel(
            "matris_10m_oam",
            compute_stress=True,
            device="cuda",
            enable_compile=True,
        )
        results = model(sim_state)   # returns {"energy", "forces", "stress"}
    """

    @configure_entrypoint("torchsim")
    def __init__(
        self,
        model: torch.nn.Module | ModelAdapter | str = "matris_10m_oam",
        compute_stress: bool = True,
        compute_magmom: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        enable_compile: bool = True,
        config: InferenceConfig | None = None,
    ) -> None:
        """Initialize MatRIS torch-sim model.

        Args:
            enable_compile: Whether to compile interaction blocks.
            config: Unified inference settings. Only execution='eager' is
                supported here; enable_compile remains an independent option.
        """
        super().__init__()
        enable_checkpoint = self._inference_config.checkpoint_enabled
        if isinstance(model, str):
            self._adapter = load_adapter(
                model,
                device=str(device) if device is not None else None,
                enable_compile=enable_compile,
                enable_checkpoint=enable_checkpoint,
            )
            use_adapter(self, model)  # closed with its last user
            unsupported = _unsupported(self._adapter, repr(model))
            if unsupported:
                release_adapter(self)
                raise ValueError(unsupported)
            self.model = self._adapter.module
        else:
            self._adapter = as_adapter(model)
            unsupported = _unsupported(self._adapter, "the model") if self._adapter is model else None
            if unsupported:
                raise ValueError(unsupported)
            use_adapter(self, model)  # one another entry point loaded stays open while this one uses it
            self.model = self._adapter.module
            self.model.enable_compile = enable_compile
            for block in self.model.interaction_block:
                block.enable_compile = enable_compile
                block.enable_checkpoint = enable_checkpoint

        apply_checkpoint(self.model, enable_checkpoint)
        saved_recompute.check_model(
            self.model, saved_recompute.explicit_mode(self._inference_config.saved_tensor_recompute))
        saved_recompute.note_entry_point(self.model)
        self.handle_isolated_atoms = self._inference_config.handle_isolated_atoms

        self._compute_stress = compute_stress
        self._compute_magmom = compute_magmom
        self._dtype = dtype
        if device is not None:
            self._device = torch.device(device)
        else:
            try:
                self._device = next(self.model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")

        self.model = self.model.to(self._device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    adapter = property(current_adapter)

    def close(self) -> None:
        """Release the model adapter: one loaded by name (here, or by another entry point) is closed when its
        last user is; an adapter or a model the caller passed in stays open. Context exit closes too, and a
        dropped interface releases it when it is finalised."""
        release_adapter(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def config_summary(self) -> dict:
        """Requested/resolved configuration and observed execution state."""
        return config_summary(self)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    @property
    def compute_forces(self) -> bool:
        return True

    @property
    def compute_magmom(self) -> bool:
        return self._compute_magmom

    @property
    def memory_scales_with(self) -> str:
        return "n_atoms_x_density"

    @scoped_call
    def _state_to_graphs(self, state: SimState) -> list[RadiusGraph]:
        """Convert a SimState (or StateDict) to a list of RadiusGraph objects.

        The cell matrix is treated as row-major (each row is a lattice vector),
        matching the ASE / pymatgen convention.
        """

        # to_structures() copies the state to the host right after, so this
        # check adds no synchronization the call did not already have.
        if not bool(torch.isfinite(state.positions).all() & torch.isfinite(state.cell).all()):
            raise ValueError("MLIPModel: non-finite positions or cell in the SimState")
        structures = state.to_structures()
        if self._device.type == "cuda" and op_available:
            atoms = [AseAtomsAdaptor.get_atoms(structure) for structure in structures]
            graphs = atoms_to_graph_gpu(
                atoms,
                self.model.graph_converter.atom_graph_cutoff,
                self.model.graph_converter.line_graph_cutoff,
                device=str(self._device),
                check_isolated_atoms=(
                    not self.handle_isolated_atoms
                    or self.adapter.isolated_atom_policy() == "error"
                ),
            )
            return graphs if isinstance(graphs, list) else [graphs]
        return [
            self.model.graph_converter(
                structure,
                check_isolated_atoms=(
                    not self.handle_isolated_atoms
                    or self.adapter.isolated_atom_policy() == "error"
                ),
            ).to(str(self._device))
            for structure in structures
        ]

    @scoped_call
    def forward(self, state: SimState, **kwargs) -> dict[str, Tensor]:
        """Compute energy, forces, and optionally stress from a SimState.

        Args:
            state: A torch-sim ``SimState`` or a ``StateDict`` with keys
                ``positions``, ``cell``, ``atomic_numbers``, and
                ``system_idx``.

        Returns:
            Dictionary with:
                ``"energy"``  : Tensor of shape ``[n_systems]`` — total energy in eV.
                ``"forces"``  : Tensor of shape ``[n_atoms, 3]`` — forces in eV/Å.
                ``"stress"``  : Tensor of shape ``[n_systems, 3, 3]`` — stress in
                    eV/Å³ (only present when ``compute_stress=True``).
                ``"magmoms"`` : Tensor of shape ``[n_atoms]`` — per-atom magnetic
                    moments in μB (only present when ``compute_magmom=True``).
        """

        graphs = self._state_to_graphs(state)

        task = "ef"
        if self._compute_stress:
            task += "s"
        if self._compute_magmom:
            task += "m"

        result = self.adapter.evaluate(
            graphs, task=task, is_training=False,
            handle_isolated_atoms=self.handle_isolated_atoms,
        )
        energy: Tensor = result["e"]
        energy = self.adapter.extensive_energy(energy, result["atoms_per_graph"].to(dtype=energy.dtype))

        output: dict[str, Tensor] = {
            "energy": energy,
            "forces": torch.concat(result["f"], dim=0),
        }

        if self._compute_stress:
            stress = result["s"]
            if isinstance(stress, Tensor):
                output["stress"] = stress * _GPa
            else:
                output["stress"] = torch.stack(list(stress), dim=0) * _GPa

        if self._compute_magmom:
            output["magmoms"] = torch.concat(result["m"], dim=0)

        return output


def _unsupported(adapter, subject: str) -> str | None:
    """Why the torch-sim interface cannot run ``adapter`` (None: it can): it builds MatRIS's graph (with
    triplets) and calls the model's own ``evaluate`` (a model-owned derivative). A model that reads edges
    only, or one whose energy the engine differentiates (an energy tap, e.g. CHGNet), runs on
    GPU-resident MD and the calculator instead. ``subject``: the model as the message names it."""
    if adapter.topology_spec().triplet_cutoff is None:
        return (f"MLIPModel: the torch-sim interface builds MatRIS's graph (with triplets); {subject} reads "
                "edges only and is not supported here")
    if not isinstance(adapter, ModelOwnedDerivative):
        return (f"MLIPModel: the torch-sim interface calls the model's own evaluate (a model-owned "
                f"derivative); {subject} ({adapter.name!r}) has an energy the engine differentiates and is not "
                "supported here: use GPUResidentMolecularDynamics or MLIPCalculator")
    return None
