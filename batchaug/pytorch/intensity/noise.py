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


class RandRicianNoise(BatchTransform):
    """Add Rician-distributed noise per batch element (MRI artifact model).

    Equivalent to MONAI's ``RandRicianNoise``.

    The Rician distribution models the magnitude of complex Gaussian noise,
    which occurs in MRI magnitude images. Formula:

        ``output = sqrt((input + n1)^2 + n2^2)``

    where ``n1, n2 ~ N(mean, noise_std^2)``.

    Args:
        prob:       Probability of applying to each batch element.
        mean:       Mean of both Gaussian noise components.
        std:        Noise standard deviation. If ``sample_std=True``,
                    std is the *upper bound* of a ``U(0, std)`` sample.
        relative:   If True, multiply ``noise_std`` by ``std(input)``
                    per batch element.
        sample_std: If True (default), sample ``noise_std ~ U(0, std)``
                    per batch element. If False, use ``std`` directly.

    Input shape: ``(B, C, H, W, D)``.
    """

    def __init__(
        self,
        prob: float = 0.1,
        mean: float = 0.0,
        std: float = 0.1,
        relative: bool = False,
        sample_std: bool = True,
    ):
        super().__init__(prob=prob)
        self.mean = mean
        self.std = std
        self.relative = relative
        self.sample_std = sample_std

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)

        if self.sample_std:
            noise_std = torch.rand(batch_size, device=device) * self.std
        else:
            noise_std = torch.full((batch_size,), self.std, device=device)
        params["noise_std"] = noise_std

        # Pre-generate unit Gaussians — scaled by actual_std in apply().
        # Storing unit noise lets dict transforms share the same random
        # draw across all keys (consistent noise pattern).
        params["u1"] = torch.randn(shape, device=device)  # N(0,1)
        params["u2"] = torch.randn(shape, device=device)  # N(0,1)
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        noise_std = params["noise_std"]  # (B,)
        u1 = params["u1"].float()
        u2 = params["u2"].float()

        work = tensor.float()

        if self.relative:
            # Scale noise_std by per-element signal std
            signal_stds = work.flatten(1).std(dim=1)  # (B,)
            actual_std = noise_std * signal_stds
        else:
            actual_std = noise_std

        std_5d = actual_std[:, None, None, None, None]
        n1 = self.mean + std_5d * u1
        n2 = self.mean + std_5d * u2

        result = ((work + n1).pow(2) + n2.pow(2)).sqrt().to(tensor.dtype)
        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandRicianNoised(BatchDictTransform):
    """Dictionary wrapper for RandRicianNoise.

    All keys receive the same Rician noise (same unit noise tensors, same mask).
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        mean: float = 0.0,
        std: float = 0.1,
        relative: bool = False,
        sample_std: bool = True,
    ):
        transform = RandRicianNoise(
            prob=prob, mean=mean, std=std, relative=relative, sample_std=sample_std
        )
        super().__init__(keys=keys, transform=transform)
