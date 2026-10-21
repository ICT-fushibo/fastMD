"""Optional real-model assets for GPU regression tests (skipped when absent).

``MATRIS_TEST_CHECKPOINT``: MatRIS-10M-OAM checkpoint (default: the
``MatRIS.load`` download cache, ``~/.cache/matris/MatRIS_10M_OAM.pth.tar``).
``MATRIS_TEST_S01_FRAMES``: npz with frames of the matbench-discovery MD
structure S01 (CsSnI3, 135 atoms; keys ``atomic_numbers``, ``positions``
[F, N, 3], ``cell`` [F, 3, 3], ``pbc``, ``temperature_kelvin``); there is no
default, the tests that need it are skipped unless it is set.
``MATRIS_TEST_ASSETS``: directory with the RDX assets of the op-selection
snapshots (G-S3, tests/op_selection_snapshots.py) and the RDX recorder test:
``rdx_checkpoint.pth.tar`` (a MatRIS-1M model fine-tuned on RDX, widths
128/32/32), ``rdx_1344_1x1x1.extxyz`` and ``rdx_10752_2x2x2.extxyz`` (RDX
crystal, 1,344 and 10,752 atoms). There is no default; the configurations and
tests that need a missing file are skipped.

This module imports numpy and ASE lazily: the snapshot harness parent imports it.
"""
from __future__ import annotations

import os
from pathlib import Path

CHECKPOINT = Path(
    os.environ.get("MATRIS_TEST_CHECKPOINT", "~/.cache/matris/MatRIS_10M_OAM.pth.tar")
).expanduser()
_S01_FRAMES_ENV = os.environ.get("MATRIS_TEST_S01_FRAMES", "")
S01_FRAMES = Path(_S01_FRAMES_ENV).expanduser() if _S01_FRAMES_ENV else None
_ASSETS_ENV = os.environ.get("MATRIS_TEST_ASSETS", "")
TEST_ASSETS = Path(_ASSETS_ENV).expanduser() if _ASSETS_ENV else None
RDX_CHECKPOINT = "rdx_checkpoint.pth.tar"
RDX_1344 = "rdx_1344_1x1x1.extxyz"
RDX_10752 = "rdx_10752_2x2x2.extxyz"


def test_asset(name: str) -> Path | None:
    """``$MATRIS_TEST_ASSETS/<name>``, or None when the variable is unset."""
    return None if TEST_ASSETS is None else TEST_ASSETS / name


def test_assets_available(*names: str) -> bool:
    return all(path is not None and path.is_file() for path in map(test_asset, names))


def real_model_assets_available() -> bool:
    return CHECKPOINT.is_file() and S01_FRAMES is not None and S01_FRAMES.is_file()


def load_s01_frames() -> tuple[list, float]:
    import numpy as np
    from ase import Atoms

    if S01_FRAMES is None:
        raise FileNotFoundError("set MATRIS_TEST_S01_FRAMES to the S01 frames npz")
    data = np.load(S01_FRAMES)
    frames = [
        Atoms(
            numbers=data["atomic_numbers"],
            positions=data["positions"][index],
            cell=data["cell"][index],
            pbc=data["pbc"],
        )
        for index in range(data["positions"].shape[0])
    ]
    return frames, float(data["temperature_kelvin"])

# Asset lookup is a helper, including when imported into a pytest module.
test_asset.__test__ = False
