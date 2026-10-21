"""Row counts passed to K5 Triton kernels.

K5 kernels take their row/element count as a runtime int excluded from
Triton's value specialization, so eager steps with new counts reuse the
compiled kernel. Under torch.compile that count is a symbolic size; the
former ``tl.constexpr`` count made dynamo specialize it to its value, which
keeps the compiled graphs on static shapes. Without the specialization the
graphs stay symbolic, gain shape guards and dynamic-shape code, and the
enable_compile path gets slower. ``static_under_compile`` restores exactly
that specialization for traced code (``operator.index`` of a symbolic size
guards on its value) and is the identity on a Python int, i.e. in eager
execution and CUDA graph capture.
"""
from __future__ import annotations

import operator


def static_under_compile(count):
    """``count`` for a K5 kernel argument; specialized when traced by dynamo."""
    return operator.index(count)
