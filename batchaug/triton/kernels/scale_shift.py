"""Triton kernels for scale/shift intensity transforms."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Batched sum + sum-of-squares reduction (for std/mean computation)
# ---------------------------------------------------------------------------

@triton.jit
def _batched_sum_sumsq_kernel(
    input_ptr,
    sum_ptr,       # (n_groups,) output
    sumsq_ptr,     # (n_groups,) output
    N_per_group: tl.constexpr,
    stride_group,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute sum and sum-of-squares over N_per_group elements per group.

    Grid: (n_groups,). Each program reduces one group.
    """
    group_id = tl.program_id(0)
    base = group_id * stride_group

    local_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    local_sumsq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, N_per_group, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_per_group
        vals = tl.load(input_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        local_sum += tl.where(mask, vals, 0.0)
        local_sumsq += tl.where(mask, vals * vals, 0.0)

    tl.store(sum_ptr + group_id, tl.sum(local_sum, axis=0))
    tl.store(sumsq_ptr + group_id, tl.sum(local_sumsq, axis=0))


def batched_sum_sumsq(
    tensor: torch.Tensor, n_groups: int, N_per_group: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group sum and sum-of-squares.

    Tensor must be contiguous and viewed as ``(n_groups, N_per_group)``.

    Returns:
        ``(sums, sumsqs)`` each of shape ``(n_groups,)`` in float32.
    """
    assert tensor.is_contiguous()
    sums = torch.zeros(n_groups, device=tensor.device, dtype=torch.float32)
    sumsqs = torch.zeros(n_groups, device=tensor.device, dtype=torch.float32)
    BLOCK_SIZE = 1024
    _batched_sum_sumsq_kernel[(n_groups,)](
        tensor, sums, sumsqs,
        N_per_group, N_per_group,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return sums, sumsqs


# ---------------------------------------------------------------------------
# RandScaleIntensity: output = input * (1 + factor)
# ---------------------------------------------------------------------------

@triton.jit
def _rand_scale_intensity_kernel(
    input_ptr,
    output_ptr,
    factor_ptr,    # (B,) scale factors
    mask_ptr,      # (B,) bool
    N_per_batch,   # C*H*W*D
    BLOCK_SIZE: tl.constexpr,
):
    """Fused scale: output[i] = mask[b] ? input[i] * (1 + factor[b]) : input[i].

    Grid: (B, cdiv(N_per_batch, BLOCK_SIZE)).
    """
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    m = tl.load(mask_ptr + batch_id)
    base = batch_id * N_per_batch
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N_per_batch

    vals = tl.load(input_ptr + base + offsets, mask=valid, other=0.0).to(tl.float32)

    if m != 0:
        factor = tl.load(factor_ptr + batch_id)
        vals = vals * (1.0 + factor)

    tl.store(output_ptr + base + offsets, vals, mask=valid)


# ---------------------------------------------------------------------------
# RandShiftIntensity: output = input + offset
# ---------------------------------------------------------------------------

@triton.jit
def _rand_shift_intensity_kernel(
    input_ptr,
    output_ptr,
    offset_ptr,    # (B,) offsets
    mask_ptr,      # (B,) bool
    N_per_batch,   # C*H*W*D
    SAFE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused shift: output[i] = mask[b] ? input[i] + offset[b] : input[i].

    Grid: (B, cdiv(N_per_batch, BLOCK_SIZE)).
    """
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    m = tl.load(mask_ptr + batch_id)
    base = batch_id * N_per_batch
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N_per_batch

    vals = tl.load(input_ptr + base + offsets, mask=valid, other=0.0).to(tl.float32)

    if m != 0:
        shift = tl.load(offset_ptr + batch_id)
        vals = vals + shift
        if SAFE:
            vals = tl.clamp(vals, 0.0, 1.0)

    tl.store(output_ptr + base + offsets, vals, mask=valid)


# ---------------------------------------------------------------------------
# RandStdShiftIntensity: output = input + factor * std(input)
# ---------------------------------------------------------------------------

@triton.jit
def _rand_std_shift_kernel(
    input_ptr,
    output_ptr,
    factor_ptr,    # (B,)
    mask_ptr,      # (B,) bool
    N_per_batch,   # C*H*W*D
    BLOCK_SIZE: tl.constexpr,
):
    """output[i] = mask[b] ? input[i] + factor[b] * std_batch[b] : input[i].

    Std is computed *inline* (two-pass within the same program by reading
    twice), which avoids a separate kernel launch at the cost of a second
    memory pass.

    Grid: (B, cdiv(N_per_batch, BLOCK_SIZE)).
    """
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    m = tl.load(mask_ptr + batch_id)
    base = batch_id * N_per_batch
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N_per_batch

    vals = tl.load(input_ptr + base + offsets, mask=valid, other=0.0).to(tl.float32)

    if m == 0:
        tl.store(output_ptr + base + offsets, vals, mask=valid)
        return

    factor = tl.load(factor_ptr + batch_id)
    # std and mean must be pre-computed and passed in
    tl.store(output_ptr + base + offsets, vals + factor, mask=valid)


@triton.jit
def _rand_std_shift_apply_kernel(
    input_ptr,
    output_ptr,
    factor_ptr,    # (B,) scale factors
    std_ptr,       # (B,) pre-computed std values
    mask_ptr,      # (B,) bool
    N_per_batch,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply step: output[i] = input[i] + factor[b] * std[b].

    Grid: (B, cdiv(N_per_batch, BLOCK_SIZE)).
    """
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    m = tl.load(mask_ptr + batch_id)
    base = batch_id * N_per_batch
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N_per_batch

    vals = tl.load(input_ptr + base + offsets, mask=valid, other=0.0).to(tl.float32)

    if m != 0:
        factor = tl.load(factor_ptr + batch_id)
        std = tl.load(std_ptr + batch_id)
        vals = vals + factor * std

    tl.store(output_ptr + base + offsets, vals, mask=valid)


# ---------------------------------------------------------------------------
# RandScaleIntensityFixedMean: output = mean + (input - mean) * (1 + factor)
# ---------------------------------------------------------------------------

@triton.jit
def _rand_scale_fixed_mean_kernel(
    input_ptr,
    output_ptr,
    factor_ptr,    # (B,)
    mean_ptr,      # (B,) pre-computed means
    mask_ptr,      # (B,) bool
    N_per_batch,
    BLOCK_SIZE: tl.constexpr,
):
    """output[i] = mean[b] + (input[i] - mean[b]) * (1 + factor[b]).

    Grid: (B, cdiv(N_per_batch, BLOCK_SIZE)).
    """
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    m = tl.load(mask_ptr + batch_id)
    base = batch_id * N_per_batch
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N_per_batch

    vals = tl.load(input_ptr + base + offsets, mask=valid, other=0.0).to(tl.float32)

    if m != 0:
        factor = tl.load(factor_ptr + batch_id)
        mean = tl.load(mean_ptr + batch_id)
        vals = mean + (vals - mean) * (1.0 + factor)

    tl.store(output_ptr + base + offsets, vals, mask=valid)
