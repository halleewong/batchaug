"""Triton-accelerated ScaleIntensity and RandAdjustContrast."""
from __future__ import annotations

import torch
import triton

from ...base import BatchDictTransform
from ...pytorch.intensity.contrast import RandAdjustContrast as _PTRandAdjustContrast
from ...pytorch.intensity.contrast import ScaleIntensity as _PTScaleIntensity
from ..kernels.elementwise import _adjust_contrast_kernel, _scale_intensity_kernel
from ..kernels.reduce import batched_minmax

_MAX_GRID_DIM = 65535


def _safe_block_size(n_elements: int, min_block: int = 1024) -> int:
    """Choose BLOCK_SIZE so grid dimension stays within CUDA limits."""
    bs = min_block
    while triton.cdiv(n_elements, bs) > _MAX_GRID_DIM:
        bs *= 2
    return bs


class ScaleIntensity(_PTScaleIntensity):
    """Triton-accelerated ScaleIntensity (same API as PyTorch version)."""

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        if self.factor is not None:
            return tensor * (1.0 + self.factor)

        if self.minv is None and self.maxv is None:
            return tensor

        B, C = tensor.shape[:2]
        N_spatial = tensor[0, 0].numel()  # H*W*D
        N_per_batch = C * N_spatial

        flat = tensor.contiguous()

        if self.channel_wise:
            n_groups = B * C
            N_per_group = N_spatial
        else:
            n_groups = B
            N_per_group = N_per_batch

        # Pass 1: min/max reduction
        if self.channel_wise:
            # View as (B*C, H*W*D) for channel-wise reduction
            mins, maxs = batched_minmax(flat.reshape(B * C, N_spatial), n_groups, N_per_group)
        else:
            mins, maxs = batched_minmax(flat.reshape(B, N_per_batch), n_groups, N_per_group)

        # Pass 2: fused elementwise
        output = torch.empty_like(tensor)
        # mask is always True for ScaleIntensity (prob=1.0)
        mask = torch.ones(B, device=tensor.device, dtype=torch.bool)

        minv = self.minv if self.minv is not None else 0.0
        maxv = self.maxv if self.maxv is not None else 1.0

        BLOCK_SIZE = _safe_block_size(N_per_batch)
        grid = (B, triton.cdiv(N_per_batch, BLOCK_SIZE))
        _scale_intensity_kernel[grid](
            flat, output, mins, maxs, mask,
            minv, maxv,
            N_per_batch, N_spatial, C,
            CHANNEL_WISE=self.channel_wise,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output


class ScaleIntensityd(BatchDictTransform):
    """Dictionary wrapper for Triton ScaleIntensity."""

    def __init__(
        self,
        keys: list[str],
        minv: float | None = 0.0,
        maxv: float | None = 1.0,
        factor: float | None = None,
        channel_wise: bool = True,
    ):
        transform = ScaleIntensity(
            minv=minv, maxv=maxv, factor=factor, channel_wise=channel_wise
        )
        super().__init__(keys=keys, transform=transform)


class RandAdjustContrast(_PTRandAdjustContrast):
    """Triton-accelerated RandAdjustContrast (same API as PyTorch version)."""

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        gamma = params["gamma"]

        B = tensor.shape[0]
        N_per_batch = tensor[0].numel()  # C*H*W*D

        flat = tensor.contiguous()

        # Pass 1: min/max per batch element (across all channels)
        mins, maxs = batched_minmax(flat.reshape(B, N_per_batch), B, N_per_batch)

        # Pass 2: fused normalize + pow + denormalize
        output = torch.empty_like(tensor)

        BLOCK_SIZE = _safe_block_size(N_per_batch)
        grid = (B, triton.cdiv(N_per_batch, BLOCK_SIZE))
        _adjust_contrast_kernel[grid](
            flat, output, mins, maxs, gamma, mask,
            N_per_batch,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Cast back to original dtype (kernel works in float32 internally)
        return output.to(tensor.dtype)


class RandAdjustContrastd(BatchDictTransform):
    """Dictionary wrapper for Triton RandAdjustContrast."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        gamma: tuple[float, float] = (0.5, 4.5),
    ):
        transform = RandAdjustContrast(prob=prob, gamma=gamma)
        super().__init__(keys=keys, transform=transform)
