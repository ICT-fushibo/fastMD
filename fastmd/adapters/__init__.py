"""Model adapters: the contract between the single-GPU runtime and the models it runs.

The runtime calls a model, and reads what it needs to know about it, through
an adapter (:mod:`fastmd.adapters.protocol`: the protocol, its capabilities
and :class:`NegotiationError`). The models and their registry are in
:mod:`fastmd.models`; two of its functions resolve a model, and this package
forwards them:

* :func:`load_adapter`, for the entry points (calculator, GPU-resident MD,
  torch-sim): an adapter, a MatRIS instance, or a pretrained name or
  checkpoint path, as the entry points accepted before; each model of the
  registry claims its names and checkpoints (MACE, SevenNet, ORB and CHGNet,
  each with its optional extra), and MatRIS, the default, loads the rest;
* :func:`as_adapter`, for the runners: wraps the model object they are given,
  never loads.

The entry points negotiate the plan of a model other than MatRIS
(``negotiate.negotiate``, ``negotiate.renegotiate_execution``, with a
``memory.MeasuredMemoryModel`` measured on the model's own evaluation) and
return every model's plan record (:func:`entry_plan_record`).

:mod:`fastmd.adapters.negotiate` resolves (for MatRIS: records) the runtime
decisions of an adapter in an :class:`EnginePlanRecord`;
:func:`entry_plan_record` is an entry point's ``plan_record()``.
:mod:`fastmd.adapters.memory` holds the measured memory model of a model
without a K6 calibration. ``python -m fastmd.explain`` prints a dry-run plan.

This package only forwards: each name below, and a module accessed as an attribute of the package, is imported
on first access, so importing it (for example for :mod:`fastmd.adapters.protocol`) imports neither a model nor
the registry nor the negotiation. :class:`MatRISAdapter` is forwarded from MatRIS's package
(``fastmd.models.matris.adapter``) for compatibility.
"""
from __future__ import annotations

from fastmd import _lazy

__all__ = ["EnergyTap", "EnginePlanRecord", "GraphBatch", "MatRISAdapter", "ModelAdapter", "ModelOwnedDerivative",
           "NegotiationError", "PaddingRule", "SystemEnergyTap", "TopologySpec", "as_adapter", "entry_plan_record",
           "load_adapter"]

# The names imported on first access, and their modules.
_LAZY = {
    "EnergyTap": "fastmd.adapters.protocol", "GraphBatch": "fastmd.adapters.protocol",
    "ModelAdapter": "fastmd.adapters.protocol", "ModelOwnedDerivative": "fastmd.adapters.protocol",
    "NegotiationError": "fastmd.adapters.protocol", "PaddingRule": "fastmd.adapters.protocol",
    "SystemEnergyTap": "fastmd.adapters.protocol", "TopologySpec": "fastmd.adapters.protocol",
    "check_profile": "fastmd.adapters.protocol",
    "EnginePlanRecord": "fastmd.adapters.negotiate", "entry_plan_record": "fastmd.adapters.negotiate",
    "as_adapter": "fastmd.models", "load_adapter": "fastmd.models",
    "MatRISAdapter": "fastmd.models.matris.adapter",
}
__getattr__, __dir__ = _lazy.lazy_attributes(__name__, _LAZY)
