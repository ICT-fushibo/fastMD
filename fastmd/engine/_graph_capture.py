"""CUDA-graph capture policy shared by every MatRIS capture site."""
from __future__ import annotations

import contextlib
import gc

import torch

from fastmd.engine.cache_provenance import capture_retention

# Captures run while other host threads keep using CUDA: AsyncGPUMDLogger's
# writer thread waits on its D2H-copy events (cudaEventSynchronize) while the
# MD thread (re)captures a whole-step or model graph after a capacity grow,
# tier switch or pool reset. In torch's default "global" mode CUDA rejects
# such calls from *any* thread while a capture is underway
# (cudaErrorStreamCaptureUnsupported) and invalidates the capture.
# "thread_local" keeps the prohibition of unsafe calls for the capturing
# thread, which issues all captured work and its allocations, and lets other
# threads' calls on non-captured streams/events proceed. The captured work is
# unchanged. torch.compile's CUDA-graph trees capture in this mode as well.
CAPTURE_ERROR_MODE = "thread_local"


# No automatic garbage collection inside a capture. A collection there can
# finalize an unreachable CUDA graph of an earlier runner (runners and their
# graphs sit in reference cycles), and CUDAGraph.reset() destroys its
# executable, which CUDA rejects on the capturing thread and which
# invalidates the capture. torch.cuda.graph no longer collects before a
# capture (torch.compiler.config.force_cudagraph_gc). Deferred cycles are
# collected at the next automatic pass after the capture.
@contextlib.contextmanager
def capture_graph(graph: torch.cuda.CUDAGraph, *, pool=None, stream=None, check_replay=True):
    """``torch.cuda.graph(graph, pool=pool, stream=stream)`` with the MatRIS
    capture error mode and the garbage collector paused. ``graph`` keeps the
    derived projection caches it read alive and, with ``check_replay``, checks
    their source weights before each replay
    (``fastmd.engine.cache_provenance.capture_retention``)."""
    collecting = gc.isenabled()
    gc.disable()
    try:
        with capture_retention(graph, check_replay=check_replay), torch.cuda.graph(
            graph, pool=pool, stream=stream, capture_error_mode=CAPTURE_ERROR_MODE
        ):
            yield
    finally:
        if collecting:
            gc.enable()
