"""Triton-accelerated RandGaussianSharpen."""
from __future__ import annotations

import torch

from ...base import BatchDictTransform
from ...pytorch.intensity.sharpen import RandGaussianSharpen as _PTRandGaussianSharpen
from ..kernels.separable_conv import separable_gaussian_conv3d_triton
from .smooth import _check_padding_mode


class RandGaussianSharpen(_PTRandGaussianSharpen):
    """Triton-accelerated RandGaussianSharpen (same API as PyTorch version).

    Uses Triton separable convolution for both blur passes.

    Note: the Triton kernel only supports zero-padding. Unlike the PyTorch
    version (which defaults to ``padding_mode="reflect"``), this class defaults
    to zero-padding and raises ``NotImplementedError`` for any other mode.
    """

    def __init__(
        self,
        prob: float = 0.1,
        sigma1_x: tuple[float, float] = (0.5, 1.0),
        sigma1_y: tuple[float, float] = (0.5, 1.0),
        sigma1_z: tuple[float, float] = (0.5, 1.0),
        sigma2_x: float | tuple[float, float] = 0.5,
        sigma2_y: float | tuple[float, float] = 0.5,
        sigma2_z: float | tuple[float, float] = 0.5,
        alpha: tuple[float, float] = (10.0, 30.0),
        padding_mode: str | None = None,
    ):
        _check_padding_mode("RandGaussianSharpen", padding_mode)
        super().__init__(
            prob=prob,
            sigma1_x=sigma1_x,
            sigma1_y=sigma1_y,
            sigma1_z=sigma1_z,
            sigma2_x=sigma2_x,
            sigma2_y=sigma2_y,
            sigma2_z=sigma2_z,
            alpha=alpha,
            padding_mode=padding_mode,
        )

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        alpha = params["alpha"][:, None, None, None, None]

        blurred = separable_gaussian_conv3d_triton(
            tensor,
            params["kernel1_h"],
            params["kernel1_w"],
            params["kernel1_d"],
        )
        double_blurred = separable_gaussian_conv3d_triton(
            blurred,
            params["kernel2_h"],
            params["kernel2_w"],
            params["kernel2_d"],
        )

        result = blurred + alpha.to(blurred.dtype) * (blurred - double_blurred)

        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandGaussianSharpend(BatchDictTransform):
    """Dictionary wrapper for Triton RandGaussianSharpen.

    Like the non-dict version, only zero-padding is supported; any other
    ``padding_mode`` raises ``NotImplementedError``.
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        sigma1_x: tuple[float, float] = (0.5, 1.0),
        sigma1_y: tuple[float, float] = (0.5, 1.0),
        sigma1_z: tuple[float, float] = (0.5, 1.0),
        sigma2_x: float | tuple[float, float] = 0.5,
        sigma2_y: float | tuple[float, float] = 0.5,
        sigma2_z: float | tuple[float, float] = 0.5,
        alpha: tuple[float, float] = (10.0, 30.0),
        padding_mode: str | None = None,
    ):
        transform = RandGaussianSharpen(
            prob=prob,
            sigma1_x=sigma1_x,
            sigma1_y=sigma1_y,
            sigma1_z=sigma1_z,
            sigma2_x=sigma2_x,
            sigma2_y=sigma2_y,
            sigma2_z=sigma2_z,
            alpha=alpha,
            padding_mode=padding_mode,
        )
        super().__init__(keys=keys, transform=transform)
