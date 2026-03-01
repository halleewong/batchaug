from __future__ import annotations

import torch

from ...base import BatchDictTransform, BatchTransform


def _parse_range(param) -> tuple[float, float]:
    """Convert scalar or (low, high) to a (low, high) tuple.

    Scalar ``v`` → ``(-v, v)`` (symmetric, matching MONAI convention).
    """
    if isinstance(param, (int, float)):
        v = float(param)
        return (-v, v)
    return (float(param[0]), float(param[1]))


class RandScaleIntensity(BatchTransform):
    """Multiply each batch element by a random factor.

    Equivalent to MONAI's ``RandScaleIntensity``.

    Formula per element: ``output = input * (1 + factor)``
    where ``factor ~ U(factors[0], factors[1])``.

    Args:
        prob:    Probability of applying to each batch element.
        factors: ``(low, high)`` range for the scale factor, or a single
                 float ``f`` treated as ``(-f, f)``.

    Input shape: ``(B, C, H, W, D)``.
    """

    def __init__(
        self,
        prob: float = 0.1,
        factors: float | tuple[float, float] = (0.0, 0.5),
    ):
        super().__init__(prob=prob)
        self.factors = _parse_range(factors)

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        low, high = self.factors
        params["factor"] = torch.rand(batch_size, device=device) * (high - low) + low
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        factor = params["factor"][:, None, None, None, None].to(tensor.dtype)
        result = tensor * (1.0 + factor)
        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandScaleIntensityd(BatchDictTransform):
    """Dictionary wrapper for RandScaleIntensity."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        factors: float | tuple[float, float] = (0.0, 0.5),
    ):
        transform = RandScaleIntensity(prob=prob, factors=factors)
        super().__init__(keys=keys, transform=transform)


class RandShiftIntensity(BatchTransform):
    """Add a random offset to each batch element.

    Equivalent to MONAI's ``RandShiftIntensity``.

    Formula: ``output = input + offset``
    where ``offset ~ U(offsets[0], offsets[1])``.

    Args:
        prob:    Probability of applying.
        offsets: ``(low, high)`` range, or scalar ``f`` → ``(-f, f)``.
        safe:    If True, clamp result to ``[0, 1]``.

    Input shape: ``(B, C, H, W, D)``.
    """

    def __init__(
        self,
        prob: float = 0.1,
        offsets: float | tuple[float, float] = (-0.1, 0.1),
        safe: bool = False,
    ):
        super().__init__(prob=prob)
        self.offsets = _parse_range(offsets)
        self.safe = safe

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        low, high = self.offsets
        params["offset"] = torch.rand(batch_size, device=device) * (high - low) + low
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        offset = params["offset"][:, None, None, None, None].to(tensor.dtype)
        result = tensor + offset
        if self.safe:
            result = result.clamp(0.0, 1.0)
        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandShiftIntensityd(BatchDictTransform):
    """Dictionary wrapper for RandShiftIntensity."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        offsets: float | tuple[float, float] = (-0.1, 0.1),
        safe: bool = False,
    ):
        transform = RandShiftIntensity(prob=prob, offsets=offsets, safe=safe)
        super().__init__(keys=keys, transform=transform)


class RandStdShiftIntensity(BatchTransform):
    """Shift each batch element by ``factor * std(element)``.

    Equivalent to MONAI's ``RandStdShiftIntensity``.

    Formula: ``output = input + factor * std(input)``
    where ``factor ~ U(factors[0], factors[1])`` and std is computed
    per batch element (per channel if ``channel_wise=True``).

    Args:
        prob:         Probability of applying.
        factors:      ``(low, high)`` range, or scalar ``f`` → ``(-f, f)``.
        nonzero:      Compute std only from non-zero voxels.
        channel_wise: Compute std independently per channel.

    Input shape: ``(B, C, H, W, D)``.
    """

    def __init__(
        self,
        prob: float = 0.1,
        factors: float | tuple[float, float] = (-3.0, 3.0),
        nonzero: bool = False,
        channel_wise: bool = False,
    ):
        super().__init__(prob=prob)
        self.factors = _parse_range(factors)
        self.nonzero = nonzero
        self.channel_wise = channel_wise

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        low, high = self.factors
        params["factor"] = torch.rand(batch_size, device=device) * (high - low) + low
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        factor = params["factor"]  # (B,)

        work = tensor.float()

        if self.channel_wise:
            flat = work.flatten(2)  # (B, C, N)
            if self.nonzero:
                B, C = work.shape[:2]
                stds = torch.zeros(B, C, device=work.device)
                for b in range(B):
                    for c in range(C):
                        nz = flat[b, c][flat[b, c] != 0]
                        stds[b, c] = nz.std() if nz.numel() > 1 else flat[b, c].std()
                stds = stds[:, :, None, None, None]  # (B, C, 1, 1, 1)
            else:
                stds = flat.std(dim=2)[..., None, None, None]  # (B, C, 1, 1, 1)
            factor_5d = factor[:, None, None, None, None]
        else:
            flat = work.flatten(1)  # (B, N)
            if self.nonzero:
                B = work.shape[0]
                stds = torch.zeros(B, device=work.device)
                for b in range(B):
                    nz = flat[b][flat[b] != 0]
                    stds[b] = nz.std() if nz.numel() > 1 else flat[b].std()
                stds = stds[:, None, None, None, None]  # (B, 1, 1, 1, 1)
            else:
                stds = flat.std(dim=1)[:, None, None, None, None]  # (B, 1, 1, 1, 1)
            factor_5d = factor[:, None, None, None, None]

        result = (work + factor_5d * stds).to(tensor.dtype)
        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandStdShiftIntensityd(BatchDictTransform):
    """Dictionary wrapper for RandStdShiftIntensity."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        factors: float | tuple[float, float] = (-3.0, 3.0),
        nonzero: bool = False,
        channel_wise: bool = False,
    ):
        transform = RandStdShiftIntensity(
            prob=prob, factors=factors, nonzero=nonzero, channel_wise=channel_wise
        )
        super().__init__(keys=keys, transform=transform)


class RandScaleIntensityFixedMean(BatchTransform):
    """Scale intensity while preserving the mean of each batch element.

    Equivalent to MONAI's ``RandScaleIntensityFixedMean``.

    Formula: ``output = mean + (input - mean) * (1 + factor)``
    where ``factor ~ U(factors[0], factors[1])`` and mean is computed
    per batch element (per channel if ``channel_wise=True``).

    Args:
        prob:         Probability of applying.
        factors:      ``(low, high)`` range, or scalar ``f`` → ``(-f, f)``.
        channel_wise: Compute mean independently per channel.

    Input shape: ``(B, C, H, W, D)``.
    """

    def __init__(
        self,
        prob: float = 0.1,
        factors: float | tuple[float, float] = (-0.5, 0.5),
        channel_wise: bool = False,
    ):
        super().__init__(prob=prob)
        self.factors = _parse_range(factors)
        self.channel_wise = channel_wise

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        low, high = self.factors
        params["factor"] = torch.rand(batch_size, device=device) * (high - low) + low
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        factor = params["factor"]  # (B,)

        work = tensor.float()

        if self.channel_wise:
            flat = work.flatten(2)  # (B, C, N)
            means = flat.mean(dim=2)[..., None, None, None]  # (B, C, 1, 1, 1)
            factor_5d = factor[:, None, None, None, None]
        else:
            flat = work.flatten(1)  # (B, N)
            means = flat.mean(dim=1)[:, None, None, None, None]  # (B, 1, 1, 1, 1)
            factor_5d = factor[:, None, None, None, None]

        result = (means + (work - means) * (1.0 + factor_5d)).to(tensor.dtype)
        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandScaleIntensityFixedMeand(BatchDictTransform):
    """Dictionary wrapper for RandScaleIntensityFixedMean."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        factors: float | tuple[float, float] = (-0.5, 0.5),
        channel_wise: bool = False,
    ):
        transform = RandScaleIntensityFixedMean(
            prob=prob, factors=factors, channel_wise=channel_wise
        )
        super().__init__(keys=keys, transform=transform)
