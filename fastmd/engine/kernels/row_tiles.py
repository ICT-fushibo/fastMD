"""Row tiles (BM) of the memory-bound row kernels, chosen by feature width (C1).

The residual add, the line envelope, the directed-pair average/expand and the
gather-add-SiLU epilogue of the decomposed indexed projections each move a
``[BM, cols]`` tile per program, ``cols`` being the kernel's padded feature
tile. Their time is set by the elements per program: the former fixed tiles
(BM 4, 4, 8 and 2) give 64-256 elements on the model's 32-wide line-graph
rows, and those kernels ran at a third of their bandwidth there. A GB200 sweep
of widths 32/64/128 at the row counts of 1,344 and 10,752 atoms and of a P=4
10,752 rank has a plateau from about 1,024 elements per program (512-2,048
within a few percent), and 256 elements per half for the SiLU epilogue, which
runs one warp below width 128. The rule keeps that element count:

  kernel           elements  width 32  width 64  width 128  former tile
  residual_add       1024       32        16         8          4
  line_envelope      1024       32        16         8          4
  directed_pair      1024       32        16         8          8
  indexed_silu        256        8         4         2          2

The SiLU epilogue backward keeps its own tile (MATRIS_DECOMPOSED_INDEXED_SILU_BWD_BM,
8): larger tiles were slower. A tile changes no value: these kernels are
elementwise per row (the envelope's backward atomics run in any order anyway).

``row_tile`` returns a pinned tile (the kernel's ``MATRIS_*_BM`` option, set
through expert_overrides or the legacy environment) for every width; unset
(None, the default) it applies the rule above, or with
``MATRIS_WIDTH_ROW_TILES=0`` (kept for one version) the former fixed tiles.
The module globals are bound by ``ResolvedConfig.scope()`` like every option.
"""
from __future__ import annotations

from fastmd.config import env_value

_WIDTH_ROW_TILES = env_value("MATRIS_WIDTH_ROW_TILES") != "0"

# kernel: (elements per program, largest tile, former fixed tile)
_RULES = {
    "residual_add": (1024, 64, 4),
    "line_envelope": (1024, 64, 4),
    "directed_pair": (1024, 64, 8),
    "indexed_silu": (256, 16, 2),
}


def row_tile(kernel: str, cols: int, pinned: int | None) -> int:
    """The row tile of a ``[BM, cols]`` program (``cols`` a power of two)."""
    if pinned is not None:
        return pinned
    elements, largest, fixed = _RULES[kernel]
    if not _WIDTH_ROW_TILES:
        return fixed
    return max(1, min(largest, elements // cols))
