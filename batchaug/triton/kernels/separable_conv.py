"""Triton kernel for separable 1D convolution along a single axis of a 5D tensor."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _separable_conv1d_axis0_kernel(
    input_ptr,     # (B*C, H, W, D) contiguous
    output_ptr,    # same shape
    kernel_ptr,    # (B, K) — per-batch-element 1D kernels
    BC,            # B*C
    C,
    H, W, D,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Convolve along axis 0 (H) for one (batch, channel) pair per program.

    Grid: (BC, cdiv(W*D, BLOCK_SIZE)).
    """
    bc_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    b = bc_id // C
    WD = W * D
    half_k = K // 2

    wd_offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    wd_valid = wd_offsets < WD

    kern_base = b * K
    base = bc_id * H * WD

    for h_out in range(H):
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for ki in range(K):
            h_in = h_out + ki - half_k
            in_bounds = (h_in >= 0) & (h_in < H)
            addr = base + h_in * WD + wd_offsets
            vals = tl.load(input_ptr + addr, mask=wd_valid & in_bounds, other=0.0)
            w = tl.load(kernel_ptr + kern_base + ki)
            acc += vals * w
        tl.store(output_ptr + base + h_out * WD + wd_offsets, acc, mask=wd_valid)


@triton.jit
def _separable_conv1d_axis1_kernel(
    input_ptr,
    output_ptr,
    kernel_ptr,    # (B, K)
    BC, C,
    H, W, D,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Convolve along axis 1 (W).

    Grid: (BC, cdiv(H*D, BLOCK_SIZE)).
    """
    bc_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    b = bc_id // C
    WD = W * D
    HD = H * D
    half_k = K // 2

    hd_offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hd_valid = hd_offsets < HD
    h_idx = hd_offsets // D
    d_idx = hd_offsets % D

    kern_base = b * K
    base = bc_id * H * WD

    for w_out in range(W):
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for ki in range(K):
            w_in = w_out + ki - half_k
            in_bounds = (w_in >= 0) & (w_in < W)
            addr = base + h_idx * WD + w_in * D + d_idx
            vals = tl.load(input_ptr + addr, mask=hd_valid & in_bounds, other=0.0)
            w = tl.load(kernel_ptr + kern_base + ki)
            acc += vals * w
        out_addr = base + h_idx * WD + w_out * D + d_idx
        tl.store(output_ptr + out_addr, acc, mask=hd_valid)


@triton.jit
def _separable_conv1d_axis2_kernel(
    input_ptr,
    output_ptr,
    kernel_ptr,    # (B, K)
    BC, C,
    H, W, D,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Convolve along axis 2 (D).

    Grid: (BC, cdiv(H*W, BLOCK_SIZE)).
    """
    bc_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    b = bc_id // C
    WD = W * D
    HW = H * W
    half_k = K // 2

    hw_offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hw_valid = hw_offsets < HW
    h_idx = hw_offsets // W
    w_idx = hw_offsets % W

    kern_base = b * K
    base = bc_id * H * WD

    for d_out in range(D):
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for ki in range(K):
            d_in = d_out + ki - half_k
            in_bounds = (d_in >= 0) & (d_in < D)
            addr = base + h_idx * WD + w_idx * D + d_in
            vals = tl.load(input_ptr + addr, mask=hw_valid & in_bounds, other=0.0)
            w = tl.load(kernel_ptr + kern_base + ki)
            acc += vals * w
        out_addr = base + h_idx * WD + w_idx * D + d_out
        tl.store(output_ptr + out_addr, acc, mask=hw_valid)


def separable_gaussian_conv3d_triton(
    tensor: torch.Tensor,
    kernel_h: torch.Tensor,
    kernel_w: torch.Tensor,
    kernel_d: torch.Tensor,
) -> torch.Tensor:
    """Triton separable 3D Gaussian blur — drop-in replacement for PyTorch version.

    Args:
        tensor: (B, C, H, W, D).
        kernel_h, kernel_w, kernel_d: (B, K_*) per-element 1D kernels.

    Returns:
        (B, C, H, W, D) blurred tensor.
    """
    B, C, H, W, D = tensor.shape
    out_dtype = tensor.dtype
    x = tensor.float().contiguous()
    BC = B * C

    BLOCK_SIZE = 512

    for kernel, axis_fn in [
        (kernel_h, _separable_conv1d_axis0_kernel),
        (kernel_w, _separable_conv1d_axis1_kernel),
        (kernel_d, _separable_conv1d_axis2_kernel),
    ]:
        K = kernel.shape[1]
        kern = kernel.float().contiguous()
        out = torch.empty_like(x)

        if axis_fn is _separable_conv1d_axis0_kernel:
            n_tiles = triton.cdiv(W * D, BLOCK_SIZE)
        elif axis_fn is _separable_conv1d_axis1_kernel:
            n_tiles = triton.cdiv(H * D, BLOCK_SIZE)
        else:
            n_tiles = triton.cdiv(H * W, BLOCK_SIZE)

        grid = (BC, n_tiles)
        axis_fn[grid](
            x, out, kern,
            BC, C, H, W, D,
            K=K,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        x = out

    return x.reshape(tensor.shape).to(out_dtype)
