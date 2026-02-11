"""Triton kernel for batched min/max reduction."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _minmax_reduce_kernel(
    input_ptr,
    mins_ptr,
    maxs_ptr,
    N_per_group: tl.constexpr,
    stride_group,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute min and max over N_per_group elements for each group.

    Grid: (n_groups,).  Each program reduces one group using a block-strided
    loop, then writes one min and one max value.

    Args:
        input_ptr: Pointer to contiguous input, shape (n_groups, N_per_group).
        mins_ptr: (n_groups,) output min values (must be pre-initialized to +inf).
        maxs_ptr: (n_groups,) output max values (must be pre-initialized to -inf).
        N_per_group: Elements per reduction group.
        stride_group: Stride between groups (== N_per_group for contiguous).
        BLOCK_SIZE: Tile size.
    """
    group_id = tl.program_id(0)
    base = group_id * stride_group

    local_min = tl.full([BLOCK_SIZE], value=float("inf"), dtype=tl.float32)
    local_max = tl.full([BLOCK_SIZE], value=float("-inf"), dtype=tl.float32)

    for start in range(0, N_per_group, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_per_group
        vals = tl.load(input_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        local_min = tl.where(mask, tl.minimum(local_min, vals), local_min)
        local_max = tl.where(mask, tl.maximum(local_max, vals), local_max)

    result_min = tl.min(local_min, axis=0)
    result_max = tl.max(local_max, axis=0)

    tl.store(mins_ptr + group_id, result_min)
    tl.store(maxs_ptr + group_id, result_max)


def batched_minmax(
    tensor: torch.Tensor, n_groups: int, N_per_group: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute min/max over the last ``N_per_group`` elements per group.

    The tensor is viewed as ``(n_groups, N_per_group)`` (must be contiguous).

    Returns:
        (mins, maxs) each of shape ``(n_groups,)`` in float32.
    """
    assert tensor.is_contiguous()
    mins = torch.full((n_groups,), float("inf"), device=tensor.device, dtype=torch.float32)
    maxs = torch.full((n_groups,), float("-inf"), device=tensor.device, dtype=torch.float32)

    BLOCK_SIZE = 1024
    grid = (n_groups,)
    _minmax_reduce_kernel[grid](
        tensor,
        mins,
        maxs,
        N_per_group,
        N_per_group,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return mins, maxs
