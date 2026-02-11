from __future__ import annotations

import torch

from ...base import BatchDictTransform, BatchTransform


class ScaleIntensity(BatchTransform):
    """Rescale tensor intensity to [minv, maxv].

    Min/max are always computed **per batch element** (never pooled across
    the batch dimension).  The ``channel_wise`` flag controls whether
    channels within each batch element are treated independently or pooled:

    - channel_wise=True  (default): min/max per (batch element, channel)
      pair — each channel in each batch element is rescaled independently.
    - channel_wise=False: min/max per batch element across all its
      channels — every channel in a batch element shares one min/max.
    - factor is not None: multiply by (1 + factor), ignores channel_wise.
    """

    def __init__(
        self,
        minv: float | None = 0.0,
        maxv: float | None = 1.0,
        factor: float | None = None,
        channel_wise: bool = True,
    ):
        super().__init__(prob=1.0)
        self.minv = minv
        self.maxv = maxv
        self.factor = factor
        self.channel_wise = channel_wise

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        if self.factor is not None:
            return tensor * (1.0 + self.factor)

        if self.minv is None and self.maxv is None:
            return tensor

        if self.channel_wise:
            # Min/max per (batch, channel): flatten spatial dims only
            flat = tensor.flatten(2)  # (B, C, N)
            mins = flat.min(dim=2).values[..., None, None, None]  # (B, C, 1, 1, 1)
            maxs = flat.max(dim=2).values[..., None, None, None]
        else:
            # Min/max per batch element: flatten all non-batch dims
            flat = tensor.flatten(1)  # (B, N)
            mins = flat.min(dim=1).values[:, None, None, None, None]  # (B,1,1,1,1)
            maxs = flat.max(dim=1).values[:, None, None, None, None]

        denom = maxs - mins
        is_constant = denom == 0
        denom = torch.where(is_constant, torch.ones_like(denom), denom)

        norm = (tensor - mins) / denom

        if self.minv is None or self.maxv is None:
            result = norm
        else:
            result = norm * (self.maxv - self.minv) + self.minv

        # MONAI edge case: when min == max, returns arr * minv
        if self.minv is not None:
            result = torch.where(is_constant, tensor * self.minv, result)

        return result


class ScaleIntensityd(BatchDictTransform):
    """Dictionary wrapper for ScaleIntensity."""

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


class RandAdjustContrast(BatchTransform):
    """Random gamma correction per batch element.

    Formula per element: ((x - min) / (range + eps)) ^ gamma * range + min
    where gamma ~ U(gamma[0], gamma[1]) and min/max are computed per batch
    element across all channels.

    Input shape: (B, C, H, W, D).
    """

    def __init__(
        self,
        prob: float = 0.1,
        gamma: tuple[float, float] = (0.5, 4.5),
    ):
        super().__init__(prob=prob)
        self.gamma = gamma

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        low, high = self.gamma
        params["gamma"] = (
            torch.rand(batch_size, device=device) * (high - low) + low
        )
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        gamma = params["gamma"][:, None, None, None, None]

        # Work in float32 — pow() is numerically fragile in half-precision
        work = tensor.float()
        flat = work.flatten(1)
        mins = flat.min(dim=1).values[:, None, None, None, None]
        maxs = flat.max(dim=1).values[:, None, None, None, None]
        img_range = maxs - mins

        normalized = ((work - mins) / (img_range + 1e-7)).clamp(min=0)
        result = (normalized.pow(gamma) * img_range + mins).to(tensor.dtype)

        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandAdjustContrastd(BatchDictTransform):
    """Dictionary wrapper for RandAdjustContrast.

    All keys receive the same gamma correction (same gamma, same mask).
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        gamma: tuple[float, float] = (0.5, 4.5),
    ):
        transform = RandAdjustContrast(prob=prob, gamma=gamma)
        super().__init__(keys=keys, transform=transform)
