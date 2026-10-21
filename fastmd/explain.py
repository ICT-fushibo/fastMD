"""Dry-run negotiation: print the engine plan of a model for a structure and a configuration.

    python -m fastmd.explain --structure FILE [--model NAME | --model-path PATH] [--entry gpu_md|calculator]
                             [--device cpu|cuda] [--config JSON] [--world-size N] [--index I] [--json]

The model is loaded as the entry points load it, the configuration resolved with the entry point's
defaults (``--config``: the fields of ``InferenceConfig`` as a JSON object, ``capacity`` as an object),
and every decision is printed with its reason; no entry point is constructed and no MD runs. For MatRIS
the plan is the one the runtime's rules give (``fastmd.adapters.negotiate.dry_run``); on a CPU
execution auto resolves to eager, as the entry points do there, and a request the entry point rejects
(an unsupported execution mode, a CUDA graph without CUDA) fails with the entry point's error.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys


def config_from_json(text: str | None):
    """An ``InferenceConfig`` from a JSON object of its fields (None: the entry point's defaults)."""
    from fastmd.config import CapacityConfig
    from fastmd.config import InferenceConfig
    if not text:
        return None
    fields = json.loads(text)
    if not isinstance(fields, dict):
        raise ValueError("--config must be a JSON object of InferenceConfig fields")
    try:
        if "capacity" in fields:
            fields["capacity"] = CapacityConfig(**fields["capacity"])
        return InferenceConfig(**fields)
    except TypeError as error:  # an unknown field
        raise ValueError(f"--config: {error}") from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fastmd.explain", description=__doc__.split("\n\n")[0])
    parser.add_argument("--structure", required=True, help="a structure file ASE reads")
    parser.add_argument("--index", default="0", help="the frame of a multi-frame file (ASE index; default 0)")
    parser.add_argument("--model", default="matris_10m_oam", help="a pretrained model name (default matris_10m_oam)")
    parser.add_argument("--model-path", default=None, help="a checkpoint path (instead of --model)")
    parser.add_argument("--entry", choices=("gpu_md", "calculator"), default="gpu_md",
                        help="the entry point whose plan to explain (default gpu_md)")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: the entry point's: cuda when "
                        "available for gpu_md, cpu for the calculator)")
    parser.add_argument("--config", default=None, help="InferenceConfig fields as a JSON object")
    parser.add_argument("--world-size", type=int, choices=(1,), default=1, help="FastMD uses one device")
    parser.add_argument("--json", action="store_true", help="print the record as JSON")
    args = parser.parse_args(argv)

    import ase.io
    import torch
    from fastmd.adapters import load_adapter
    from fastmd.adapters.negotiate import NegotiationError
    from fastmd.adapters.negotiate import dry_run

    default = "cuda" if args.entry == "gpu_md" and torch.cuda.is_available() else "cpu"
    device = torch.device(args.device or default)
    try:
        atoms = ase.io.read(args.structure, index=args.index)
        if isinstance(atoms, list):
            if len(atoms) != 1:
                raise ValueError(f"--index {args.index!r} selects {len(atoms)} frames; give one")
            atoms = atoms[0]
        config = config_from_json(args.config)
        with contextlib.redirect_stdout(sys.stderr):  # loading messages stay out of the printed plan
            adapter = load_adapter(None if args.model_path else args.model, model_path=args.model_path,
                                   device=str(device),
                                   optimization_profile=None if config is None else config.optimization_profile)
        adapter.module.eval()
        for parameter in adapter.module.parameters():
            parameter.requires_grad_(False)
        record = dry_run(adapter, config, entry=args.entry, device=device, atoms=atoms,
                         world_size=args.world_size)

    except (NegotiationError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(record.to_json() if args.json else record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
