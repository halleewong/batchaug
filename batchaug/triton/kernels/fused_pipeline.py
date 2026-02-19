"""Fused Triton kernel for elementwise intensity transforms.

Combines ScaleIntensity + RandAdjustContrast + RandBiasField + RandGaussianNoise
into a single kernel launch (plus one reduction pre-pass for min/max).

Grid: (B*C, cdiv(HWD, BLOCK_SIZE)) — fully parallelizes over batch AND channels.
Noise is generated in-kernel via tl.randn, eliminating the (B,C,H,W,D) allocation.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _fused_intensity_kernel(
    input_ptr,
    output_ptr,
    coeffs_ptr,            # (B, 20) Legendre coefficients
    gamma_ptr,             # (B,) gamma values
    mins_ptr,              # (B*C,) or (B,) from reduction
    maxs_ptr,              # (B*C,) or (B,) from reduction
    contrast_mask_ptr,     # (B,) bool
    bias_mask_ptr,         # (B,) bool
    noise_mask_ptr,        # (B,) bool
    noise_seed_ptr,        # (B,) int32 seeds for tl.randn
    noise_std_ptr,         # (B,) noise std
    noise_mean_ptr,        # (B,) noise mean
    minv: tl.constexpr,    # target min for scale intensity
    maxv: tl.constexpr,    # target max for scale intensity
    B, C, H, W, D,
    DO_SCALE: tl.constexpr,
    DO_CONTRAST: tl.constexpr,
    DO_BIAS: tl.constexpr,
    DO_NOISE: tl.constexpr,
    CHANNEL_WISE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused elementwise intensity kernel.

    Order: scale_intensity -> adjust_contrast -> bias_field -> noise.

    Grid: (B*C, cdiv(H*W*D, BLOCK_SIZE)).
    """
    bc_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    b = bc_id // C
    c = bc_id % C

    HWD = H * W * D

    offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < HWD

    # Load per-batch runtime masks (only when constexpr flag is True)
    rt_contrast = tl.load(contrast_mask_ptr + b) if DO_CONTRAST else 0
    rt_bias = tl.load(bias_mask_ptr + b) if DO_BIAS else 0
    rt_noise = tl.load(noise_mask_ptr + b) if DO_NOISE else 0

    # Check if ANY transform is active for this batch element
    any_active = DO_SCALE
    if DO_CONTRAST:
        any_active = any_active | rt_contrast
    if DO_BIAS:
        any_active = any_active | rt_bias
    if DO_NOISE:
        any_active = any_active | rt_noise

    addr = b * (C * HWD) + c * HWD + offsets

    if not any_active:
        # Nothing to do — copy through
        vals = tl.load(input_ptr + addr, mask=valid, other=0.0)
        tl.store(output_ptr + addr, vals, mask=valid)
        return

    x_val = tl.load(input_ptr + addr, mask=valid, other=0.0).to(tl.float32)

    # 1. Scale intensity: normalize to [minv, maxv]
    if DO_SCALE:
        if CHANNEL_WISE:
            lo = tl.load(mins_ptr + bc_id)
            hi = tl.load(maxs_ptr + bc_id)
        else:
            lo = tl.load(mins_ptr + b)
            hi = tl.load(maxs_ptr + b)
        denom = hi - lo
        is_constant = denom == 0.0
        safe_denom = tl.where(is_constant, 1.0, denom)
        norm = (x_val - lo) / safe_denom
        scaled = norm * (maxv - minv) + minv
        # MONAI edge case: when min==max, return tensor * minv
        x_val = tl.where(is_constant, x_val * minv, scaled)

    # 2. Adjust contrast: gamma correction
    if DO_CONTRAST:
        if rt_contrast:
            gamma = tl.load(gamma_ptr + b)
            if DO_SCALE:
                # After scale, range is [minv, maxv] — no reduction needed
                target_range = maxv - minv
                x_norm = (x_val - minv) / (target_range + 1e-7)
            else:
                # Use reduction min/max
                if CHANNEL_WISE:
                    c_lo = tl.load(mins_ptr + bc_id)
                    c_hi = tl.load(maxs_ptr + bc_id)
                else:
                    c_lo = tl.load(mins_ptr + b)
                    c_hi = tl.load(maxs_ptr + b)
                target_range = c_hi - c_lo
                x_norm = (x_val - c_lo) / (target_range + 1e-7)

            x_norm = tl.maximum(x_norm, 0.0)
            log_norm = tl.math.log(tl.maximum(x_norm, 1e-10))
            powered = tl.math.exp(gamma * log_norm)
            powered = tl.where(x_norm == 0.0, 0.0, powered)

            if DO_SCALE:
                x_val = powered * target_range + minv
            else:
                x_val = powered * target_range + c_lo

    # 3. Bias field: multiply by exp(Legendre polynomial)
    if DO_BIAS:
        # Compute spatial indices
        h_idx = offsets // (W * D)
        wd = offsets % (W * D)
        w_idx = wd // D
        d_idx = wd % D

        # Normalized coordinates in [-1, 1]
        x_coord = (2.0 * h_idx.to(tl.float32)) / (H - 1) - 1.0 if H > 1 else tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        y_coord = (2.0 * w_idx.to(tl.float32)) / (W - 1) - 1.0 if W > 1 else tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        z_coord = (2.0 * d_idx.to(tl.float32)) / (D - 1) - 1.0 if D > 1 else tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Legendre polynomials: L0=1, L1=x, L2=(3x^2-1)/2, L3=(5x^3-3x)/2
        Lx0 = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
        Lx1 = x_coord
        Lx2 = (3.0 * x_coord * x_coord - 1.0) * 0.5
        Lx3 = (5.0 * x_coord * x_coord * x_coord - 3.0 * x_coord) * 0.5

        Ly0 = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
        Ly1 = y_coord
        Ly2 = (3.0 * y_coord * y_coord - 1.0) * 0.5
        Ly3 = (5.0 * y_coord * y_coord * y_coord - 3.0 * y_coord) * 0.5

        Lz0 = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
        Lz1 = z_coord
        Lz2 = (3.0 * z_coord * z_coord - 1.0) * 0.5
        Lz3 = (5.0 * z_coord * z_coord * z_coord - 3.0 * z_coord) * 0.5

        # Dot product: 20 coefficients for degree=3
        coeff_base = b * 20
        bias_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        bias_val += tl.load(coeffs_ptr + coeff_base + 0) * Lx0 * Ly0 * Lz0
        bias_val += tl.load(coeffs_ptr + coeff_base + 1) * Lx0 * Ly0 * Lz1
        bias_val += tl.load(coeffs_ptr + coeff_base + 2) * Lx0 * Ly0 * Lz2
        bias_val += tl.load(coeffs_ptr + coeff_base + 3) * Lx0 * Ly0 * Lz3
        bias_val += tl.load(coeffs_ptr + coeff_base + 4) * Lx0 * Ly1 * Lz0
        bias_val += tl.load(coeffs_ptr + coeff_base + 5) * Lx0 * Ly1 * Lz1
        bias_val += tl.load(coeffs_ptr + coeff_base + 6) * Lx0 * Ly1 * Lz2
        bias_val += tl.load(coeffs_ptr + coeff_base + 7) * Lx0 * Ly2 * Lz0
        bias_val += tl.load(coeffs_ptr + coeff_base + 8) * Lx0 * Ly2 * Lz1
        bias_val += tl.load(coeffs_ptr + coeff_base + 9) * Lx0 * Ly3 * Lz0
        bias_val += tl.load(coeffs_ptr + coeff_base + 10) * Lx1 * Ly0 * Lz0
        bias_val += tl.load(coeffs_ptr + coeff_base + 11) * Lx1 * Ly0 * Lz1
        bias_val += tl.load(coeffs_ptr + coeff_base + 12) * Lx1 * Ly0 * Lz2
        bias_val += tl.load(coeffs_ptr + coeff_base + 13) * Lx1 * Ly1 * Lz0
        bias_val += tl.load(coeffs_ptr + coeff_base + 14) * Lx1 * Ly1 * Lz1
        bias_val += tl.load(coeffs_ptr + coeff_base + 15) * Lx1 * Ly2 * Lz0
        bias_val += tl.load(coeffs_ptr + coeff_base + 16) * Lx2 * Ly0 * Lz0
        bias_val += tl.load(coeffs_ptr + coeff_base + 17) * Lx2 * Ly0 * Lz1
        bias_val += tl.load(coeffs_ptr + coeff_base + 18) * Lx2 * Ly1 * Lz0
        bias_val += tl.load(coeffs_ptr + coeff_base + 19) * Lx3 * Ly0 * Lz0

        # Compute multiplier; if mask is off, use 1.0 (identity)
        bias_multiplier = tl.where(rt_bias != 0, tl.math.exp(bias_val), 1.0)
        x_val = x_val * bias_multiplier

    # 4. Noise: generated in-kernel via tl.randn (no pre-allocated tensor)
    if DO_NOISE:
        if rt_noise:
            seed = tl.load(noise_seed_ptr + b)
            noise_offset = (c * HWD + offsets).to(tl.int32)
            noise_val = tl.randn(seed, noise_offset)
            std = tl.load(noise_std_ptr + b)
            mean = tl.load(noise_mean_ptr + b)
            x_val = x_val + noise_val * std + mean

    tl.store(output_ptr + addr, x_val, mask=valid)
