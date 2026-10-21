"""Segment helpers shared by the model code, the operator dispatch and the multi-GPU runtime.

``segment_count`` counts the rows of each segment without a host
synchronization (unlike ``torch.bincount``; CUDA-graph capturable);
``aggregate`` sums or averages the rows of each segment; ``Dimwise_softmax``
is the column-wise softmax over each segment's rows, on CUDA the fused kernel
of ``triton_segment_softmax`` unless switched off (the operator dispatch's
``fused`` argument, else ``MATRIS_FUSED_SEGMENT_SOFTMAX``, bound to this
module's ``_FUSED_SEGMENT_SOFTMAX``). ``IndexedProjectionConfig`` is the
lowering policy an executor hands to the indexed projections.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fastmd.config import env_value
from fastmd.engine import recorder as _recorder

_FUSED_SEGMENT_SOFTMAX = env_value("MATRIS_FUSED_SEGMENT_SOFTMAX", "1") != "0"


@dataclass(frozen=True)
class IndexedProjectionConfig:
    """Internal per-executor lowering policy; never changes global switches.

    The caller must qualify the tensor/module contract before supplying this
    policy. The shared GatedMLP still checks its own shape/frozen-weight/cache
    contract. Single-device callers omitting it retain their existing config.
    """

    enabled: bool = True
    backend: str = "decomposed_indexed_silu"
    min_rows: int = 20_000
    strict_errors: bool = True
    # D8: parts of different widths (the P>1 dispatcher sets it from its policy).
    mixed_width: bool = False


def segment_count(segment: Tensor, num_segment: int) -> Tensor:
    """Sync-free replacement for ``torch.bincount(segment, minlength=num_segment)``.

    ``torch.bincount`` reads ``max()`` on the host (a D2H sync / stall) on CUDA
    even when ``minlength`` is given. Counting via ``index_add_`` into a
    fixed-size buffer is sync-free and CUDA-graph capturable.
    """
    counts = torch.zeros(num_segment, dtype=torch.long, device=segment.device)
    counts.index_add_(0, segment.long(), torch.ones_like(segment, dtype=torch.long))
    return counts


def Dimwise_softmax(feas: Tensor, segment: Tensor, num_segment=None, *,
                    fused: bool | None = None) -> Tensor:
    """Computes a sparsely evaluated softmax.
    
    Args:
        feas: The source tensor. shape: [num, dim]
        segment: specify the segment of each row [num, 1] 
        fused: the operator dispatcher's segment_softmax switch; None reads
            MATRIS_FUSED_SEGMENT_SOFTMAX (legacy dispatch).
    """
    num, dim = feas.shape
    if num_segment is None:
        num_segment = int(segment.max()) + 1
    if feas.is_cuda and (_FUSED_SEGMENT_SOFTMAX if fused is None else fused):
        # Fusion A: single fused Triton segment-softmax (3 fwd + 2 bwd kernels)
        # replacing ~8 eager ops; validated exact (FP64 gradcheck) and 2-2.6x faster.
        from fastmd.engine.kernels.triton_segment_softmax import fused_segment_softmax
        return fused_segment_softmax(feas, segment, num_segment)

    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("segment_softmax", "torch", {"rows": num, "width": dim, "segments": num_segment})
    segment_expanded = segment.unsqueeze(1).expand(-1, dim) # [num, dim]
    
    feas_max = torch.empty( num_segment, dim, dtype=feas.dtype, device=feas.device )
    feas_max.fill_(float("-inf"))
    feas_max = feas_max.scatter_reduce(
        0, segment_expanded, feas, reduce='amax', include_self=False,
    ) #[num_segment, dim]
    # Gather: [num_segment, dim] -> [num, dim]
    feas_max = feas_max[segment]
    out = (feas - feas_max).exp()
    
    # =========== scatter sum ============
    out_sum = torch.zeros(num_segment, dim, device=feas.device, dtype=feas.dtype)
    out_sum = out_sum.scatter_reduce(
        0, segment_expanded, out, reduce='sum', include_self=False
    )
    # Gather: [num_segment, dim] -> [num, dim]
    out_sum = out_sum[segment]
    score = out / out_sum
    return score

def aggregate(data: torch.Tensor, 
              segment: torch.Tensor, 
              bin_count: torch.Tensor = None, 
              average=True, 
              num_segment=None) -> torch.Tensor:
    """Aggregate rows in data by specifying the segment.

    Args:
        data (Tensor): data tensor to aggregate [n_row, feature_dim]
        segment (Tensor): specify the owner of each row [n_row, 1]
        average (bool): if True, average the rows, if False, sum the rows.
            Default = True
        num_owner (int, optional): the number of owners, this is needed if the
            max idx of owner is not presented in owners tensor
            Default = None

    Returns:
        output (Tensor): [num_owner, feature_dim]
    """
    if bin_count is None:
        # torch.bincount syncs (reads max() on host) even with minlength; use a
        # sync-free index_add_ count when the segment size is known.
        if num_segment is not None:
            bin_count = segment_count(segment, num_segment)
        else:
            bin_count = torch.bincount(segment)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))

    if (num_segment is not None) and (bin_count.shape[0] != num_segment):
        difference = num_segment - bin_count.shape[0]
        bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])
    # make sure this operation is done on the same device of data and owners
    output = data.new_zeros([bin_count.shape[0], data.shape[1]])
    output = output.index_add_(0, segment, data)
    if average:
        output = (output.T / bin_count).T
    return output
