from __future__ import annotations

import torch
import torch.nn.functional as F

from ..base import BatchDictTransform, BatchTransform


def gaussian_1d_batch(
    sigmas: torch.Tensor, truncated: float = 4.0
) -> torch.Tensor:
    """Batch of 1D Gaussian kernels using MONAI's erf approximation.

    Args:
        sigmas: (B,) sigma values.
        truncated: Truncation factor for kernel half-width.

    Returns:
        (B, K) normalised kernels where K = 2 * max_tail + 1.
    """
    device = sigmas.device
    sigmas = sigmas.float().clamp(min=1e-6)

    max_sigma = sigmas.max().item()
    max_tail = int(max(max_sigma * truncated, 0.5) + 0.5)

    x = torch.arange(
        -max_tail, max_tail + 1, dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, K)

    t = 0.70710678 / sigmas.unsqueeze(1)  # (B, 1)
    out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())  # (B, K)
    out = out.clamp(min=0)
    return out / out.sum(dim=1, keepdim=True)


def separable_gaussian_conv3d(
    tensor: torch.Tensor,
    kernel_h: torch.Tensor,
    kernel_w: torch.Tensor,
    kernel_d: torch.Tensor,
) -> torch.Tensor:
    """Apply separable 3D Gaussian blur using grouped conv3d.

    Args:
        tensor: (B, C, H, W, D).
        kernel_h, kernel_w, kernel_d: (B, K_*) per-element 1D kernels.

    Returns:
        (B, C, H, W, D) blurred tensor.
    """
    B, C = tensor.shape[:2]
    out_dtype = tensor.dtype
    # Always convolve in float32 for numerical stability
    x = tensor.float().reshape(1, B * C, *tensor.shape[2:])

    for kernel, axis in [(kernel_h, 0), (kernel_w, 1), (kernel_d, 2)]:
        K = kernel.shape[1]
        # Repeat each element's kernel C times → (B*C, K)
        w = kernel.repeat_interleave(C, dim=0)
        weight_shape = [B * C, 1, 1, 1, 1]
        weight_shape[axis + 2] = K
        w = w.reshape(weight_shape)  # already float32

        padding = [0, 0, 0]
        padding[axis] = K // 2

        x = F.conv3d(x, w, padding=padding, groups=B * C)

    return x.reshape(tensor.shape).to(out_dtype)


class RandGaussianSmooth(BatchTransform):
    """Random Gaussian smoothing with per-element sigma.

    Sigma is sampled independently per spatial axis per batch element.
    Kernels are pre-computed in ``sample_params`` so dict transforms
    share the same blur across keys.

    Input shape: (B, C, H, W, D).
    """

    def __init__(
        self,
        prob: float = 0.1,
        sigma_x: tuple[float, float] = (0.25, 1.5),
        sigma_y: tuple[float, float] = (0.25, 1.5),
        sigma_z: tuple[float, float] = (0.25, 1.5),
    ):
        super().__init__(prob=prob)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z

    @staticmethod
    def _sample_sigma(
        sigma_range: tuple[float, float],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        low, high = sigma_range
        return torch.rand(batch_size, device=device) * (high - low) + low

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        params["sigma_x"] = self._sample_sigma(self.sigma_x, batch_size, device)
        params["sigma_y"] = self._sample_sigma(self.sigma_y, batch_size, device)
        params["sigma_z"] = self._sample_sigma(self.sigma_z, batch_size, device)
        params["kernel_h"] = gaussian_1d_batch(params["sigma_x"])
        params["kernel_w"] = gaussian_1d_batch(params["sigma_y"])
        params["kernel_d"] = gaussian_1d_batch(params["sigma_z"])
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        result = separable_gaussian_conv3d(
            tensor, params["kernel_h"], params["kernel_w"], params["kernel_d"]
        )
        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandGaussianSmoothd(BatchDictTransform):
    """Dictionary wrapper for RandGaussianSmooth.

    All keys receive the same blur (same sigmas, same mask).
    """

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
