from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
import functools

import torch


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Emit an NVTX range when CUDA profiling is available."""
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_ranged(name: str):
    """Decorator: run the function inside ``nvtx_range(name)`` (host side; never inside a capture body)."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with nvtx_range(name):
                return fn(*args, **kwargs)
        return wrapper
    return decorate
