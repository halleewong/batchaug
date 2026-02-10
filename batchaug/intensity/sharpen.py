from __future__ import annotations

import torch

from ..base import BatchDictTransform, BatchTransform
from .smooth import gaussian_1d_batch, separable_gaussian_conv3d


class RandGaussianSharpen(BatchTransform):
    """Unsharp masking: blurred + alpha * (blurred - double_blurred).

    Two rounds of Gaussian smoothing are applied:
      1. ``sigma1`` → ``blurred``
      2. ``sigma2`` on ``blurred`` → ``double_blurred``

    The high-frequency detail ``blurred - double_blurred`` is then added
    back with strength ``alpha``.

    If ``sigma2_*`` is a **scalar**, its upper bound is clamped to the
    per-element sampled ``sigma1`` value (matching MONAI behaviour).
    If it is a **tuple**, the range is used directly.

    Input shape: (B, C, H, W, D).
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
    ):
        super().__init__(prob=prob)
        self.sigma1_x = sigma1_x
        self.sigma1_y = sigma1_y
        self.sigma1_z = sigma1_z
        self.sigma2_x = sigma2_x
        self.sigma2_y = sigma2_y
        self.sigma2_z = sigma2_z
        self.alpha = alpha

    @staticmethod
    def _sample_sigma(
        sigma_range: tuple[float, float],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        low, high = sigma_range
        return torch.rand(batch_size, device=device) * (high - low) + low

    @staticmethod
    def _sample_sigma2(
        sigma2_param: float | tuple[float, float],
        sigma1_values: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample sigma2.  Scalar → range [scalar, sigma1]; tuple → fixed range."""
        if isinstance(sigma2_param, (int, float)):
            low = sigma2_param
            return (
                torch.rand(batch_size, device=device) * (sigma1_values - low)
                + low
            )
        low, high = sigma2_param
        return torch.rand(batch_size, device=device) * (high - low) + low

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)

        # sigma1
        s1x = self._sample_sigma(self.sigma1_x, batch_size, device)
        s1y = self._sample_sigma(self.sigma1_y, batch_size, device)
        s1z = self._sample_sigma(self.sigma1_z, batch_size, device)
        params["sigma1_x"] = s1x
        params["sigma1_y"] = s1y
        params["sigma1_z"] = s1z

        # sigma2 (may depend on sigma1)
        params["sigma2_x"] = self._sample_sigma2(
            self.sigma2_x, s1x, batch_size, device
        )
        params["sigma2_y"] = self._sample_sigma2(
            self.sigma2_y, s1y, batch_size, device
        )
        params["sigma2_z"] = self._sample_sigma2(
            self.sigma2_z, s1z, batch_size, device
        )

        # alpha
        low, high = self.alpha
        params["alpha"] = (
            torch.rand(batch_size, device=device) * (high - low) + low
        )

        # Pre-compute kernels for sharing across dict keys
        params["kernel1_h"] = gaussian_1d_batch(s1x)
        params["kernel1_w"] = gaussian_1d_batch(s1y)
        params["kernel1_d"] = gaussian_1d_batch(s1z)
        params["kernel2_h"] = gaussian_1d_batch(params["sigma2_x"])
        params["kernel2_w"] = gaussian_1d_batch(params["sigma2_y"])
        params["kernel2_d"] = gaussian_1d_batch(params["sigma2_z"])

        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        alpha = params["alpha"][:, None, None, None, None]

        blurred = separable_gaussian_conv3d(
            tensor,
            params["kernel1_h"],
            params["kernel1_w"],
            params["kernel1_d"],
        )
        double_blurred = separable_gaussian_conv3d(
            blurred,
            params["kernel2_h"],
            params["kernel2_w"],
            params["kernel2_d"],
        )

        result = blurred + alpha.to(blurred.dtype) * (blurred - double_blurred)

        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandGaussianSharpend(BatchDictTransform):
    """Dictionary wrapper for RandGaussianSharpen.

    All keys receive the same sharpening parameters.
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
        )
        super().__init__(keys=keys, transform=transform)
