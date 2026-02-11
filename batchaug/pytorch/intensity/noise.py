from __future__ import annotations

import torch

from ...base import BatchDictTransform, BatchTransform


class RandGaussianNoise(BatchTransform):
    """Add random Gaussian noise per batch element.

    Args:
        prob: Probability of applying noise to each batch element.
        mean: Either a fixed scalar or a (low, high) tuple.
            - scalar: every batch element uses this mean.
            - tuple:  mean ~ U(low, high) sampled per batch element.
        std: Either a fixed scalar or a (low, high) tuple.
            - scalar: every batch element uses this std.
            - tuple:  std ~ U(low, high) sampled per batch element.

    The same mean/std are used for all channels within a batch element.
    Noise values are independent across channels and spatial locations.

    Input shape: (B, C, H, W, D).
    """

    def __init__(
        self,
        prob: float = 0.1,
        mean: float | tuple[float, float] = 0.0,
        std: float | tuple[float, float] = 0.1,
    ):
        super().__init__(prob=prob)
        self.mean = mean
        self.std = std

    def _sample_scalar(
        self,
        param: float | tuple[float, float],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return (B,) tensor: fixed value or sampled from U(low, high)."""
        if isinstance(param, (list, tuple)):
            low, high = param
            return torch.rand(batch_size, device=device) * (high - low) + low
        return torch.full((batch_size,), param, device=device)

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        params["std"] = self._sample_scalar(self.std, batch_size, device)
        params["mean"] = self._sample_scalar(self.mean, batch_size, device)
        # Pre-generate noise so dict transforms can share it across keys
        std_expanded = params["std"][:, None, None, None, None]
        mean_expanded = params["mean"][:, None, None, None, None]
        params["noise"] = (
            torch.randn(shape, device=device) * std_expanded + mean_expanded
        )
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        noise = params["noise"]
        mask_5d = mask[:, None, None, None, None].to(tensor.dtype)
        return tensor + noise.to(tensor.dtype) * mask_5d


class RandGaussianNoised(BatchDictTransform):
    """Dictionary wrapper for RandGaussianNoise.

    All keys receive the same noise (same noise tensor, same mask).
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        mean: float | tuple[float, float] = 0.0,
        std: float | tuple[float, float] = 0.1,
    ):
        transform = RandGaussianNoise(prob=prob, mean=mean, std=std)
        super().__init__(keys=keys, transform=transform)
