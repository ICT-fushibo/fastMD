"""The models: MatRIS (``fastmd.models.matris``), one adapter module per foreign model (MACE, SevenNet, ORB v3,
CHGNet), and the MatGL M3GNet integration; and the one registry of the models an entry point loads.

:data:`MODELS` has a row per model: the module that loads it and, by name, its functions there (what it claims,
how it loads, which loading options it rejects). Adding a model means adding a row (and its module).
:func:`load_adapter` tries the rows in order; MatRIS is an explicit row, the last, and the default
(:data:`DEFAULT`): it loads what no row claims, and :func:`as_adapter` wraps any model object that is not an
adapter with the default row's ``wrap``. :func:`plan_recorder` names, per rule set of a model the runtime plans
by its own rules (MatRIS's), the module that records that plan for the negotiation. Importing this package imports no
model: a row's module is imported when the row is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmd.adapters.protocol import ModelAdapter


@dataclass(frozen=True)
class ModelEntry:
    """One model of the registry. ``module`` defines the functions the other fields name:

    * ``claims(model, model_path) -> bool``: whether :func:`load_adapter` loads ``model`` with this row;
    * ``load(model, *, model_path, device, ...)``: the adapter, given ``load_adapter``'s options named in
      ``load_options``;
    * ``check_options(model_dtype, optimization_profile, model_path)``: raises for the options the model rejects
      (before anything loads);
    * ``wrap(model)`` (the default row): the adapter of a model object as it is, never loaded.
    """

    name: str
    module: str
    claims: str
    load: str
    check_options: str = "check_load_options"
    load_options: tuple[str, ...] = ("model_dtype", "optimization_profile")
    wrap: str | None = None

    def function(self, field: str):
        """The function ``field`` names (e.g. "load"), from ``module`` (imported on first use)."""
        return getattr(importlib.import_module(self.module), getattr(self, field))


MODELS = (
    ModelEntry("prophet", "fastmd.models.prophet", claims="is_prophet_model", load="load_prophet"),
    ModelEntry("esen", "fastmd.models.esen", claims="is_esen_model", load="load_esen"),
    ModelEntry("mace", "fastmd.models.mace", claims="is_mace_model", load="load_mace"),
    ModelEntry("sevennet", "fastmd.models.sevennet", claims="is_sevennet_model", load="load_sevennet"),
    ModelEntry("orb", "fastmd.models.orb", claims="is_orb_model", load="load_orb"),
    ModelEntry("chgnet", "fastmd.models.chgnet", claims="is_chgnet_model", load="load_chgnet"),
    ModelEntry("matris", "fastmd.models.matris.adapter", claims="is_matris_model", load="load_matris",
               load_options=("enable_compile", "enable_checkpoint"), wrap="MatRISAdapter"),
)
# The row that loads what no row claims, and wraps the model objects as_adapter is given.
DEFAULT = "matris"
# The module that records the plan the runtime makes by its own rules, per rule set (RuntimePlanned.runtime_rules):
# its entry_record(instance) and dry_run_record(adapter, resolved, ...).
PLAN_RECORDERS = (("matris", "fastmd.models.matris.plan_record"),)


def _row(name: str) -> ModelEntry:
    return next(row for row in MODELS if row.name == name)


def _claiming(model, model_path: str | None) -> ModelEntry:
    """The row that loads ``model``: the first that claims it, else the default."""
    return next((row for row in MODELS if row.function("claims")(model, model_path)), _row(DEFAULT))


def plan_recorder(rules: str):
    """The module recording the plan of a model the runtime plans by the rule set ``rules`` (imported on first
    use)."""
    return importlib.import_module(dict(PLAN_RECORDERS)[rules])


def _is_adapter(model) -> bool:
    """An adapter object: an instance of the default row's adapter class (MatRIS's), or an object that is not a
    torch module and has every member of the adapter protocol (its annotated attributes and its methods) itself,
    looked up statically (a mock or a wrapper forwarding attributes stays a model)."""
    from torch import nn

    from fastmd.adapters.protocol import ModelAdapter

    if isinstance(model, _row(DEFAULT).function("wrap")):
        return True
    if isinstance(model, nn.Module):
        return False
    members = set(ModelAdapter.__annotations__) | {name for name in vars(ModelAdapter) if not name.startswith("_")}
    missing = object()
    return all(inspect.getattr_static(model, name, missing) is not missing for name in sorted(members))


def as_adapter(model) -> ModelAdapter:
    """``model`` if it is an adapter, else the default model's adapter of the model object (duck-typed like
    the runners' ``model`` argument, not checked, never loaded): MatRIS's."""
    return model if _is_adapter(model) else _row(DEFAULT).function("wrap")(model)


def load_adapter(
    model="matris_10m_oam",
    *,
    model_path: str | None = None,
    device: str | None = None,
    enable_compile: bool = False,
    enable_checkpoint: bool | None = False,
    model_dtype: str | None = None,
    optimization_profile: str | None = None,
) -> ModelAdapter:
    """The adapter of what an entry point was given.

    An adapter is returned as is and a MatRIS instance is wrapped as it is (not
    moved, not reconfigured; ``model_path`` and the loading options are then
    unused). Otherwise the first row of :data:`MODELS` that claims ``model``
    loads it: a MACE name (``"mace_mp_0"``, ``"mace_mpa_0"``), ``"mace"`` with a
    ``model_path``, or a path to a ``.model`` file loads MACE
    (:func:`fastmd.models.mace.load_mace`, which needs mace-torch) on
    ``device``; a SevenNet name (``"sevennet_0"``, ``"7net-0"``) or
    ``"sevennet"`` with a ``model_path`` loads SevenNet
    (:func:`fastmd.models.sevennet.load_sevennet`, which needs sevenn); an
    ORB name (``"orb_v3"``, with or without a ``model_path``) loads ORB
    (:func:`fastmd.models.orb.load_orb`, which needs orb-models); a CHGNet
    name (``"chgnet_0_3_0"``) loads upstream CHGNet's bundled weights
    (:func:`fastmd.models.chgnet.load_chgnet`, which needs chgnet), float32
    only. Anything else is loaded with :meth:`MatRIS.load`: the checkpoint
    at ``model_path`` if given, else the pretrained ``model`` name, on
    ``device`` with ``enable_compile`` and ``enable_checkpoint``.

    ``model_dtype`` ("float32" or "float64"; None: the model's default) and
    ``optimization_profile`` (None, "topology", "generic"; "legacy" is
    MatRIS's) apply to models loaded here that set their dtype or install
    their kernels at load time, before an entry point binds the module (MACE:
    float64 or float32, cuEquivariance unless "generic"; SevenNet: float32,
    cuEquivariance unless "generic"; ORB: float32, its compiled backbone
    unless "generic"). MatRIS runs float32
    on the single-GPU runtime and selects its kernels per call from the entry
    point's configuration: it accepts None or "float32", and the profile is
    only checked. Each row's ``check_options`` rejects the options its model
    does not take before anything loads; an adapter given as is is checked
    against its own profiles and dtype.
    """
    if model_dtype not in (None, "float32", "float64"):
        raise ValueError(f"model_dtype must be None, 'float32' or 'float64', got {model_dtype!r}")
    if optimization_profile not in (None, "legacy", "topology", "generic"):
        raise ValueError("optimization_profile must be None (topology), 'legacy', 'topology' or 'generic'")
    if _is_adapter(model):
        from fastmd.adapters.protocol import check_profile

        check_profile(model, optimization_profile)
        if model_dtype is not None and model.topology_spec().model_dtype != model_dtype:
            raise ValueError(f"model {model.name!r} is {model.topology_spec().model_dtype}, not model_dtype="
                             f"{model_dtype!r} (convert it when loading)")
        return model
    row = _claiming(model, model_path)
    row.function("check_options")(model_dtype, optimization_profile, model_path)
    options = {"model_dtype": model_dtype, "optimization_profile": optimization_profile,
               "enable_compile": enable_compile, "enable_checkpoint": enable_checkpoint}
    return row.function("load")(model, model_path=model_path, device=device,
                                **{name: options[name] for name in row.load_options})
