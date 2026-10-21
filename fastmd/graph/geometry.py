"""TF32-immune small matrix products for lattice/fractional coordinate math."""
from __future__ import annotations

from torch import Tensor


def small_matmul(a: Tensor, b: Tensor) -> Tensor:
    """``a @ b`` for a tiny contraction dimension (3 for cell math), in full fp32.

    cuBLAS runs float32 GEMMs on TF32 tensor cores (10 mantissa bits) whenever
    ``torch.backends.cuda.matmul.allow_tf32`` is set (float32 matmul
    precision ``"high"``). NGC PyTorch containers export
    ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``, which makes that flag default to
    ``True`` at start-up (torch 2.12: only the default; setting it to
    ``False`` or ``set_float32_matmul_precision("highest")`` disables TF32
    again), so TF32 is active unless the script turns it off. For the
    geometry transforms ``pos @ inv(cell)``, ``frac @ cell`` and
    ``image @ lattice`` that rounds Cartesian positions and image shifts by
    ~1e-2 A in a 19 A cell, which moves atoms across the cutoffs; the stress
    VJP through the strain products loses ~1e-3 relative accuracy as well.

    The broadcast multiply + sum below never dispatches to cuBLAS, so it is
    independent of every TF32 setting. It supports ``torch.matmul`` batch
    broadcasting for inputs with at least two dimensions, is differentiable
    with respect to both operands, has no host synchronization and is CUDA
    graph capturable.
    """
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("small_matmul expects operands with at least 2 dimensions")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"small_matmul shape mismatch: {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(dim=-2)
