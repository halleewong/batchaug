"""Triton-accelerated RandGaussianSmooth."""
from __future__ import annotations

import torch

from ...base import BatchDictTransform
from ...pytorch.intensity.smooth import RandGaussianSmooth as _PTRandGaussianSmooth
from ..kernels.separable_conv import separable_gaussian_conv3d_triton


class RandGaussianSmooth(_PTRandGaussianSmooth):
    """Triton-accelerated RandGaussianSmooth (same API as PyTorch version)."""

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        result = separable_gaussian_conv3d_triton(
            tensor, params["kernel_h"], params["kernel_w"], params["kernel_d"]
        )
        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandGaussianSmoothd(BatchDictTransform):
    """Dictionary wrapper for Triton RandGaussianSmooth."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        sigma_x: tuple[float, float] = (0.25, 1.5),
        sigma_y: tuple[float, float] = (0.25, 1.5),
        sigma_z: tuple[float, float] = (0.25, 1.5),
    ):
        transform = RandGaussianSmooth(
            prob=prob, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z
        )
        super().__init__(keys=keys, transform=transform)
