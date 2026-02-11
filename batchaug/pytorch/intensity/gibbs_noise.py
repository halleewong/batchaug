from __future__ import annotations

import torch

from ...base import BatchDictTransform, BatchTransform


class RandGibbsNoise(BatchTransform):
    """Simulate Gibbs ringing artifact via k-space truncation.

    Applies FFT, masks high-frequency components with a spherical mask,
    then applies inverse FFT. The ``alpha`` parameter controls truncation
    intensity: 0 = identity, 1 = maximum truncation.

    Alpha is sampled independently per batch element.
    The same artifact is applied to all channels within a batch element.

    Input shape: (B, C, H, W, D).
    """

    def __init__(
        self,
        prob: float = 0.1,
        alpha: tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(prob=prob)
        self.alpha = alpha

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        low, high = self.alpha
        params["alpha"] = torch.rand(batch_size, device=device) * (high - low) + low

        # Precompute spherical mask per batch element
        spatial_shape = shape[2:]  # (H, W, D)
        max_dim = max(spatial_shape)

        # Distance grid: (H, W, D)
        coords = [
            torch.arange(s, dtype=torch.float32, device=device) - (s - 1) / 2.0
            for s in spatial_shape
        ]
        dist_sq = (
            coords[0][:, None, None] ** 2
            + coords[1][None, :, None] ** 2
            + coords[2][None, None, :] ** 2
        )
        dist = torch.sqrt(dist_sq)  # (H, W, D)

        # Radius per batch element: (B,)
        r = (1.0 - params["alpha"]) * max_dim * (2**0.5) / 2.0

        # Mask: (B, 1, H, W, D) — broadcast over channels
        params["k_mask"] = (dist.unsqueeze(0) <= r[:, None, None, None]).unsqueeze(1)

        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        k_mask = params["k_mask"]  # (B, 1, H, W, D)

        # Work in float32 for FFT precision
        work = tensor.float()
        spatial_dims = (-3, -2, -1)

        # FFT → fftshift
        k = torch.fft.fftshift(torch.fft.fftn(work, dim=spatial_dims), dim=spatial_dims)

        # Apply k-space mask
        k_masked = k * k_mask

        # ifftshift → iFFT → real
        result = torch.fft.ifftn(
            torch.fft.ifftshift(k_masked, dim=spatial_dims),
            dim=spatial_dims,
        ).real.to(tensor.dtype)

        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandGibbsNoised(BatchDictTransform):
    """Dictionary wrapper for RandGibbsNoise.

    All keys receive the same Gibbs artifact (same alpha, same mask).
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        alpha: tuple[float, float] = (0.0, 1.0),
    ):
        transform = RandGibbsNoise(prob=prob, alpha=alpha)
        super().__init__(keys=keys, transform=transform)
