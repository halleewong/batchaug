"""Triton fused bias field kernel: Legendre polynomial eval + exp + multiply."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bias_field_kernel_deg3(
    input_ptr,       # (B, C, H, W, D)
    output_ptr,      # (B, C, H, W, D)
    coeffs_ptr,      # (B, 20) for degree=3
    mask_ptr,        # (B,) bool
    B, C, H, W, D,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused bias field: compute Legendre basis on-the-fly, dot with coeffs, exp, multiply.

    Specialized for degree=3 (20 coefficients). Grid: (B, cdiv(H*W*D, BLOCK_SIZE)).
    """
    batch_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    m = tl.load(mask_ptr + batch_id)
    HWD = H * W * D
    N_per_batch = C * HWD

    offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < HWD

    if m == 0:
        # Copy input to output for all channels
        for c in range(C):
            addr = batch_id * N_per_batch + c * HWD + offsets
            vals = tl.load(input_ptr + addr, mask=valid, other=0.0)
            tl.store(output_ptr + addr, vals, mask=valid)
        return

    # Compute spatial indices
    h_idx = offsets // (W * D)
    wd = offsets % (W * D)
    w_idx = wd // D
    d_idx = wd % D

    # Normalized coordinates in [-1, 1]
    x = (2.0 * h_idx.to(tl.float32)) / (H - 1) - 1.0 if H > 1 else tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    y = (2.0 * w_idx.to(tl.float32)) / (W - 1) - 1.0 if W > 1 else tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    z = (2.0 * d_idx.to(tl.float32)) / (D - 1) - 1.0 if D > 1 else tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Legendre polynomials via recurrence: L0=1, L1=x, L2=(3x^2-1)/2, L3=(5x^3-3x)/2
    Lx0 = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    Lx1 = x
    Lx2 = (3.0 * x * x - 1.0) * 0.5
    Lx3 = (5.0 * x * x * x - 3.0 * x) * 0.5

    Ly0 = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    Ly1 = y
    Ly2 = (3.0 * y * y - 1.0) * 0.5
    Ly3 = (5.0 * y * y * y - 3.0 * y) * 0.5

    Lz0 = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    Lz1 = z
    Lz2 = (3.0 * z * z - 1.0) * 0.5
    Lz3 = (5.0 * z * z * z - 3.0 * z) * 0.5

    # Enumerate all (i,j,k) with i+j+k <= 3, same order as _legendre_basis_3d
    # (0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,3,0),
    # (1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,1),(1,2,0),
    # (2,0,0),(2,0,1),(2,1,0),
    # (3,0,0)
    coeff_base = batch_id * 20
    bias = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    c0 = tl.load(coeffs_ptr + coeff_base + 0)
    bias += c0 * Lx0 * Ly0 * Lz0    # (0,0,0)
    c1 = tl.load(coeffs_ptr + coeff_base + 1)
    bias += c1 * Lx0 * Ly0 * Lz1    # (0,0,1)
    c2 = tl.load(coeffs_ptr + coeff_base + 2)
    bias += c2 * Lx0 * Ly0 * Lz2    # (0,0,2)
    c3 = tl.load(coeffs_ptr + coeff_base + 3)
    bias += c3 * Lx0 * Ly0 * Lz3    # (0,0,3)
    c4 = tl.load(coeffs_ptr + coeff_base + 4)
    bias += c4 * Lx0 * Ly1 * Lz0    # (0,1,0)
    c5 = tl.load(coeffs_ptr + coeff_base + 5)
    bias += c5 * Lx0 * Ly1 * Lz1    # (0,1,1)
    c6 = tl.load(coeffs_ptr + coeff_base + 6)
    bias += c6 * Lx0 * Ly1 * Lz2    # (0,1,2)
    c7 = tl.load(coeffs_ptr + coeff_base + 7)
    bias += c7 * Lx0 * Ly2 * Lz0    # (0,2,0)
    c8 = tl.load(coeffs_ptr + coeff_base + 8)
    bias += c8 * Lx0 * Ly2 * Lz1    # (0,2,1)
    c9 = tl.load(coeffs_ptr + coeff_base + 9)
    bias += c9 * Lx0 * Ly3 * Lz0    # (0,3,0)
    c10 = tl.load(coeffs_ptr + coeff_base + 10)
    bias += c10 * Lx1 * Ly0 * Lz0   # (1,0,0)
    c11 = tl.load(coeffs_ptr + coeff_base + 11)
    bias += c11 * Lx1 * Ly0 * Lz1   # (1,0,1)
    c12 = tl.load(coeffs_ptr + coeff_base + 12)
    bias += c12 * Lx1 * Ly0 * Lz2   # (1,0,2)
    c13 = tl.load(coeffs_ptr + coeff_base + 13)
    bias += c13 * Lx1 * Ly1 * Lz0   # (1,1,0)
    c14 = tl.load(coeffs_ptr + coeff_base + 14)
    bias += c14 * Lx1 * Ly1 * Lz1   # (1,1,1)
    c15 = tl.load(coeffs_ptr + coeff_base + 15)
    bias += c15 * Lx1 * Ly2 * Lz0   # (1,2,0)
    c16 = tl.load(coeffs_ptr + coeff_base + 16)
    bias += c16 * Lx2 * Ly0 * Lz0   # (2,0,0)
    c17 = tl.load(coeffs_ptr + coeff_base + 17)
    bias += c17 * Lx2 * Ly0 * Lz1   # (2,0,1)
    c18 = tl.load(coeffs_ptr + coeff_base + 18)
    bias += c18 * Lx2 * Ly1 * Lz0   # (2,1,0)
    c19 = tl.load(coeffs_ptr + coeff_base + 19)
    bias += c19 * Lx3 * Ly0 * Lz0   # (3,0,0)

    # exp(bias_field)
    multiplier = tl.math.exp(bias)

    # Apply to all channels
    for c in range(C):
        addr = batch_id * N_per_batch + c * HWD + offsets
        vals = tl.load(input_ptr + addr, mask=valid, other=0.0).to(tl.float32)
        result = vals * multiplier
        tl.store(output_ptr + addr, result, mask=valid)
