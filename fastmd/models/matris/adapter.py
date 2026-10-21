"""MatRIS behind the adapter protocol (wrap only: no MatRIS code moves here), and what the model registry
(``fastmd.models``) calls for MatRIS: its claim, its loader and its loading options' check."""
from __future__ import annotations

from torch import Tensor

from fastmd.adapters.protocol import CaptureContract
from fastmd.adapters.protocol import GraphBatch
from fastmd.adapters.protocol import PaddingRule
from fastmd.adapters.protocol import TopologySpec

_SINK_ATOMS = PaddingRule("sink_atoms")


def is_matris_model(model, model_path: str | None = None) -> bool:
    """Whether ``model`` is a MatRIS instance (``load_adapter`` wraps it as it is). MatRIS is also the
    registry's default: it loads what no model claims, a pretrained name or ``None`` with a ``model_path``."""
    from fastmd.models.matris.model import MatRIS

    return isinstance(model, MatRIS)


def load_matris(model, *, model_path: str | None = None, device: str | None = None, enable_compile: bool = False,
                enable_checkpoint: bool | None = False) -> "MatRISAdapter":
    """The adapter of a MatRIS instance (wrapped as it is: not moved, not reconfigured), else of
    :meth:`MatRIS.load`: the checkpoint at ``model_path`` if given, else the pretrained ``model`` name, on
    ``device`` with ``enable_compile`` and ``enable_checkpoint``."""
    from fastmd.models.matris.model import MatRIS

    if isinstance(model, MatRIS):
        return MatRISAdapter(model)
    return MatRISAdapter(MatRIS.load(
        model_path=model_path,
        model_name=model,
        device=device,
        enable_compile=enable_compile,
        enable_checkpoint=enable_checkpoint,
    ))


def check_load_options(model_dtype: str | None, optimization_profile: str | None, model_path: str | None) -> None:
    """MatRIS runs float32 on the single-GPU runtime; every profile is MatRIS's ("legacy" included)."""
    if model_dtype == "float64":
        raise ValueError("MatRIS runs float32 on the single-GPU runtime: model_dtype must be None or 'float32'")


class MatRISAdapter:
    """A MatRIS module as a :class:`~fastmd.adapters.protocol.ModelAdapter` with a model-owned derivative.

    ``module`` is the MatRIS instance itself: the adapter neither copies nor
    registers it (it is no ``nn.Module``), so ``state_dict`` keys, parameters
    and tensor addresses stay the module's. ``evaluate`` is the module too:
    ``adapter.evaluate(graphs, task=...)`` is ``module(graphs, task=...)``,
    ``MatRIS.__call__`` with no extra Python frame, so compiled programs,
    recorded operator sites and captured graphs are those of a direct call.
    Forces and stress come from MatRIS's own autograd (``ForceStressHead``).
    Like the runners before it, the adapter reads the module's attributes only
    when asked (duck typing; nothing is validated or cached).

    For the capability negotiation MatRIS declares its contract (bitwise
    against the previous revision, a model-owned derivative with the stress in
    GPa), its saved-tensor recompute levels, its multi-GPU exchange and that
    the runtime plans it by its own rules (``runtime_rules``); the negotiation
    records the decisions the runtime makes for MatRIS and does not make them
    anew.
    """

    name = "matris"
    recompute_levels = ("off", "on", "deep")  # InferenceConfig.saved_tensor_recompute
    runtime_rules = "matris"  # RuntimePlanned: the runtime plans MatRIS by its own rules

    def __init__(self, module) -> None:
        self.module = module
        self.evaluate = module

    @property
    def compiled(self) -> bool:
        """Whether interaction blocks compile with torch.compile (``enable_compile``)."""
        return any(getattr(block, "enable_compile", False) for block in self.module.interaction_block)

    def topology_spec(self) -> TopologySpec:
        """The graph converter's cutoffs: the atom graph and the line graph of bond pairs (triplets)."""
        converter = self.module.graph_converter
        parameter = next(iter(self.module.parameters()), None)
        return TopologySpec(
            cutoff=float(converter.atom_graph_cutoff),
            num_interactions=len(self.module.interaction_block),
            triplet_cutoff=float(converter.line_graph_cutoff),
            undirected_pairs=True,
            model_dtype="float32" if parameter is None else str(parameter.dtype).removeprefix("torch."),
        )

    def padding_rule(self) -> PaddingRule:
        """Sink atoms: padded edges and triplets sit on sink rows past ``n_real``, which MatRIS leaves out."""
        return _SINK_ATOMS

    def prepare(self, batch: GraphBatch):
        """MatRIS's ``RadiusGraph`` over the batch's tensors (the same objects, no copy): what the
        fixed-capacity builder returns from ``build`` for MatRIS's topology."""
        from fastmd.graph.fixed_capacity_builder import radius_graph_view

        return radius_graph_view(batch)

    @property
    def upstream(self) -> str:
        """The package providing the model: this one."""
        from fastmd import __version__
        return f"fastmd {__version__} (bundled MatRIS)"

    def capture_contract(self) -> CaptureContract:
        """Sync-free and captured today (model_graph, whole_step); every runtime path is held bit for
        bit to the previous revision's same path (deterministic mode); MatRIS differentiates itself and
        returns the stress in GPa."""
        return CaptureContract(sync_free=True, exact_cutoff=True, accuracy_class="bitwise",
                               force_mode="model_owned", stress_unit="GPa",
                               compiled="dynamic" if self.compiled else "none")



    def isolated_atom_policy(self) -> str:
        """``"reference"`` with an ``AtomRef`` reference-energy table, else ``"error"``."""
        return "error" if self.module.reference_energy is None else "reference"

    def reference_energies(self) -> Tensor | None:
        """The ``AtomRef`` weight itself (``[1, elements]``, column ``Z - 1``), or None without a table."""
        fc = getattr(getattr(self.module, "reference_energy", None), "fc", None)
        return None if fc is None else fc.weight

    def extensive_energy(self, energy, atom_counts):
        """``energy * atom_counts`` for an intensive MatRIS (``is_intensive``: ``"e"`` is per real atom),
        in the energy's dtype as the runners always multiplied (FP32 for an FP32 model); else ``energy``."""
        if self.module.is_intensive:
            return energy * atom_counts
        return energy
