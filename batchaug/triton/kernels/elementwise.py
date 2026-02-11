"""Triton fused elementwise kernels for ScaleIntensity and RandAdjustContrast."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_intensity_kernel(
    input_ptr,
    output_ptr,
    mins_ptr,          # (n_groups,)
    maxs_ptr,          # (n_groups,)
    mask_ptr,          # (B,) bool
    minv: tl.constexpr,
    maxv: tl.constexpr,
    N_per_batch,       # C*H*W*D
    N_per_channel,     # H*W*D
    C,
    CHANNEL_WISE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused normalize + scale intensity.

    Grid: (B, cdiv(N_per_batch, BLOCK_SIZE)).
    """
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # Check mask
    m = tl.load(mask_ptr + batch_id)
    base = batch_id * N_per_batch
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N_per_batch

    vals = tl.load(input_ptr + base + offsets, mask=valid, other=0.0).to(tl.float32)

    if m == 0:
        # Mask is False — pass through
        tl.store(output_ptr + base + offsets, vals, mask=valid)
        return

    if CHANNEL_WISE:
        # Group index is (batch_id * C + channel_id) — vector
        channel_id = offsets // N_per_channel
        group_id = batch_id * C + channel_id
        lo = tl.load(mins_ptr + group_id, mask=valid)
        hi = tl.load(maxs_ptr + group_id, mask=valid)
    else:
        # Scalar group_id — broadcast to all lanes
        lo = tl.load(mins_ptr + batch_id)
        hi = tl.load(maxs_ptr + batch_id)
    denom = hi - lo
    is_constant = denom == 0.0
    safe_denom = tl.where(is_constant, 1.0, denom)

    norm = (vals - lo) / safe_denom
    result = norm * (maxv - minv) + minv

    # MONAI edge case: when min==max, return tensor * minv
    result = tl.where(is_constant, vals * minv, result)

    tl.store(output_ptr + base + offsets, result, mask=valid)


@triton.jit
def _adjust_contrast_kernel(
    input_ptr,
    output_ptr,
    mins_ptr,          # (B,)
    maxs_ptr,          # (B,)
    gamma_ptr,         # (B,)
    mask_ptr,          # (B,) bool
    N_per_batch,       # C*H*W*D
    BLOCK_SIZE: tl.constexpr,
):
    """Fused normalize + pow(gamma) + denormalize.

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

    lo = tl.load(mins_ptr + batch_id)
    hi = tl.load(maxs_ptr + batch_id)
    gamma = tl.load(gamma_ptr + batch_id)
    img_range = hi - lo

    normalized = (vals - lo) / (img_range + 1e-7)
    normalized = tl.maximum(normalized, 0.0)
    # pow(x, gamma) = exp(gamma * log(x)); clamp log input to avoid -inf
    log_norm = tl.math.log(tl.maximum(normalized, 1e-10))
    powered = tl.math.exp(gamma * log_norm)
    # Where normalized was exactly 0, pow should be 0
    powered = tl.where(normalized == 0.0, 0.0, powered)
    result = powered * img_range + lo

    tl.store(output_ptr + base + offsets, result, mask=valid)
