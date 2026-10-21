"""Private-sink switch of the atomic backward kernels (design D3, Q15).

The distributed padded layout routes every padding row of a domain to ONE
private sink row. Those rows' cotangents are exactly zero, but an atomic VJP
still adds them, all to the same sink address, and the serialized adds
dominate the kernel (Q15: the line-attention cat3 VJP 4.97 vs 1.05 ms/call).

A kernel entry reads ``zero_skip_atomics()`` in its forward (under the scope
the P>1 forward opens, ``private_sink_atomics``) and keeps the value for its
backward, whose atomics then omit zero addends (``SKIP_ZERO`` constexpr).
Nonzero, NaN and infinite addends are still added, so the sums are unchanged
up to atomic ordering. Outside the scope (P=1, every single-GPU path) the flag
is False and the kernels compile to the code they had before. In deterministic
mode (``fastmd.precision.deterministic_mode``, also read in the forward) the
backward adds with ``triton_zero_skip_index.ordered_index_add_`` instead, in
and out of the scope.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

_ZERO_SKIP = False


def zero_skip_atomics() -> bool:
    return _ZERO_SKIP


@contextmanager
def private_sink_atomics(enabled: bool = True) -> Iterator[None]:
    """Forwards run in this scope record that their atomic VJPs skip zero addends."""
    global _ZERO_SKIP
    previous, _ZERO_SKIP = _ZERO_SKIP, bool(enabled)
    try:
        yield
    finally:
        _ZERO_SKIP = previous
