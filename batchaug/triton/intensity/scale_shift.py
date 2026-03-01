"""Triton-accelerated scale/shift intensity transforms."""
from __future__ import annotations

import torch
import triton

from ...base import BatchDictTransform
from ...pytorch.intensity.scale_shift import (
    RandScaleIntensity as _PTRandScaleIntensity,
    RandScaleIntensityFixedMean as _PTRandScaleIntensityFixedMean,
    RandShiftIntensity as _PTRandShiftIntensity,
    RandStdShiftIntensity as _PTRandStdShiftIntensity,
)
from ..kernels.scale_shift import (
    _rand_scale_fixed_mean_kernel,
    _rand_scale_intensity_kernel,
    _rand_shift_intensity_kernel,
    _rand_std_shift_apply_kernel,
    batched_sum_sumsq,
)

_MAX_GRID_DIM = 65535


def _safe_block_size(n_elements: int, min_block: int = 1024) -> int:
    """Choose BLOCK_SIZE so the grid dimension stays within CUDA limits."""
    bs = min_block
    while triton.cdiv(n_elements, bs) > _MAX_GRID_DIM:
        bs *= 2
    return bs


# ---------------------------------------------------------------------------
# RandScaleIntensity
# ---------------------------------------------------------------------------

class RandScaleIntensity(_PTRandScaleIntensity):
    """Triton-accelerated RandScaleIntensity (same API as PyTorch version)."""

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        factor = params["factor"]

        B = tensor.shape[0]
        N_per_batch = tensor[0].numel()

        output = torch.empty_like(tensor)
        BLOCK_SIZE = _safe_block_size(N_per_batch)
        grid = (B, triton.cdiv(N_per_batch, BLOCK_SIZE))
        _rand_scale_intensity_kernel[grid](
            tensor.contiguous(), output,
            factor, mask,
            N_per_batch,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output.to(tensor.dtype)


class RandScaleIntensityd(BatchDictTransform):
    """Dictionary wrapper for Triton RandScaleIntensity."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        factors=( 0.0, 0.5),
    ):
        transform = RandScaleIntensity(prob=prob, factors=factors)
        super().__init__(keys=keys, transform=transform)


# ---------------------------------------------------------------------------
# RandShiftIntensity
# ---------------------------------------------------------------------------

class RandShiftIntensity(_PTRandShiftIntensity):
    """Triton-accelerated RandShiftIntensity (same API as PyTorch version)."""

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        offset = params["offset"]

        B = tensor.shape[0]
        N_per_batch = tensor[0].numel()

        output = torch.empty_like(tensor)
        BLOCK_SIZE = _safe_block_size(N_per_batch)
        grid = (B, triton.cdiv(N_per_batch, BLOCK_SIZE))
        _rand_shift_intensity_kernel[grid](
            tensor.contiguous(), output,
            offset, mask,
            N_per_batch,
            SAFE=self.safe,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output.to(tensor.dtype)


class RandShiftIntensityd(BatchDictTransform):
    """Dictionary wrapper for Triton RandShiftIntensity."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        offsets=(-0.1, 0.1),
        safe: bool = False,
    ):
        transform = RandShiftIntensity(prob=prob, offsets=offsets, safe=safe)
        super().__init__(keys=keys, transform=transform)


# ---------------------------------------------------------------------------
# RandStdShiftIntensity
# ---------------------------------------------------------------------------

class RandStdShiftIntensity(_PTRandStdShiftIntensity):
    """Triton-accelerated RandStdShiftIntensity.

    Falls back to PyTorch for ``nonzero=True`` or ``channel_wise=True``
    (non-trivial reduction patterns). Otherwise: Triton two-pass
    (sum/sumsq reduction → elementwise apply).
    """

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        if self.nonzero or self.channel_wise:
            # Delegate to PyTorch for complex reduction patterns
            return super().apply(tensor, params)

        mask = params["mask"]
        factor = params["factor"]

        B = tensor.shape[0]
        N_per_batch = tensor[0].numel()

        flat = tensor.contiguous().float().reshape(B, N_per_batch)

        # Pass 1: compute sum and sum-of-squares per batch element
        sums, sumsqs = batched_sum_sumsq(flat, B, N_per_batch)
        means = sums / N_per_batch
        # Variance with ddof=1 to match PyTorch's tensor.std() default
        variance = (sumsqs / N_per_batch - means * means) * (
            N_per_batch / (N_per_batch - 1)
        )
        stds = variance.clamp(min=0).sqrt()

        # Pass 2: elementwise apply
        output = torch.empty_like(tensor)
        BLOCK_SIZE = _safe_block_size(N_per_batch)
        grid = (B, triton.cdiv(N_per_batch, BLOCK_SIZE))
        _rand_std_shift_apply_kernel[grid](
            tensor.contiguous(), output,
            factor, stds, mask,
            N_per_batch,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output.to(tensor.dtype)


class RandStdShiftIntensityd(BatchDictTransform):
    """Dictionary wrapper for Triton RandStdShiftIntensity."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        factors=(-3.0, 3.0),
        nonzero: bool = False,
        channel_wise: bool = False,
    ):
        transform = RandStdShiftIntensity(
            prob=prob, factors=factors, nonzero=nonzero, channel_wise=channel_wise
        )
        super().__init__(keys=keys, transform=transform)


# ---------------------------------------------------------------------------
# RandScaleIntensityFixedMean
# ---------------------------------------------------------------------------

class RandScaleIntensityFixedMean(_PTRandScaleIntensityFixedMean):
    """Triton-accelerated RandScaleIntensityFixedMean.

    Falls back to PyTorch for ``channel_wise=True``.
    """

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        if self.channel_wise:
            return super().apply(tensor, params)

        mask = params["mask"]
        factor = params["factor"]

        B = tensor.shape[0]
        N_per_batch = tensor[0].numel()

        flat = tensor.contiguous().float().reshape(B, N_per_batch)

        # Pass 1: compute mean per batch element
        sums, _ = batched_sum_sumsq(flat, B, N_per_batch)
        means = sums / N_per_batch

        # Pass 2: elementwise apply
        output = torch.empty_like(tensor)
        BLOCK_SIZE = _safe_block_size(N_per_batch)
        grid = (B, triton.cdiv(N_per_batch, BLOCK_SIZE))
        _rand_scale_fixed_mean_kernel[grid](
            tensor.contiguous(), output,
            factor, means, mask,
            N_per_batch,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output.to(tensor.dtype)


class RandScaleIntensityFixedMeand(BatchDictTransform):
    """Dictionary wrapper for Triton RandScaleIntensityFixedMean."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        factors=(-0.5, 0.5),
        channel_wise: bool = False,
    ):
        transform = RandScaleIntensityFixedMean(
            prob=prob, factors=factors, channel_wise=channel_wise
        )
        super().__init__(keys=keys, transform=transform)
