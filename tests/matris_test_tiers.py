"""Opt-in test tiers (tests/README.md).

The default suite (``python -B -m unittest discover -s tests``) is tier 1 and
checks correctness only. The other tiers are opt-in:

* Tier 2, before merge: ``MATRIS_RUN_MULTI_GPU_TESTS=1`` runs the NCCL torchrun
  workers at 2/4 ranks (tests/test_multi_gpu_workers.py) and the multi-GPU
  checks inside them; ``MATRIS_RUN_LARGE_ROW_TESTS=1`` runs the large-row int64
  tests on an idle large-memory GPU.
* Tier 3, with the benchmarks: ``MATRIS_RUN_PERF_TIER=1`` runs the structural
  perf guards (kernel-selection snapshots, zero steady-state Gloo calls, JIT
  recompile, host-sync, scan and allocation counts). They time nothing.
  ``MATRIS_RUN_OP_SELECTION_SNAPSHOTS=1`` adds the full G-S3 snapshot matrix.

Timing lives only in the benchmark harnesses, never in unittest.
"""
from __future__ import annotations

import os
import unittest


def _enabled(*names: str) -> bool:
    return any(os.environ.get(name) == "1" for name in names)


# MATRIS_RUN_PERF_TESTS is the step-3.1a spelling of the perf tier; both enable it.
PERF_TIER = _enabled("MATRIS_RUN_PERF_TIER", "MATRIS_RUN_PERF_TESTS")
MULTI_GPU_TIER = _enabled("MATRIS_RUN_MULTI_GPU_TESTS")
LARGE_ROW_TIER = _enabled("MATRIS_RUN_LARGE_ROW_TESTS")

perf_tier = unittest.skipUnless(PERF_TIER, "structural perf guard: set MATRIS_RUN_PERF_TIER=1")
multi_gpu_tier = unittest.skipUnless(MULTI_GPU_TIER, "multi-GPU tier: set MATRIS_RUN_MULTI_GPU_TESTS=1")
large_row_tier = unittest.skipUnless(
    LARGE_ROW_TIER, "large-row tier: set MATRIS_RUN_LARGE_ROW_TESTS=1 on an idle large-memory GPU")
