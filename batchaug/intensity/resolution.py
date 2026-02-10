from __future__ import annotations

import torch
import torch.nn.functional as F

from ..base import BatchDictTransform, BatchTransform


class RandSimulateLowResolution(BatchTransform):
    """Simulate low resolution by downsampling then upsampling.

    A scalar zoom factor is sampled per batch element from
    ``zoom_range``.  The spatial dimensions are scaled uniformly,
    downsampled with ``downsample_mode``, and upsampled back to the
    original size with ``upsample_mode``.

    Batch elements with the same integer target size are grouped and
    processed together via ``F.interpolate``.

    Input shape: (B, C, H, W, D).
    """

    _ALIGN_CORNERS_MODES = {"linear", "bilinear", "trilinear", "bicubic"}

    def __init__(
        self,
        prob: float = 0.1,
        zoom_range: tuple[float, float] = (0.5, 1.0),
        downsample_mode: str = "nearest",
        upsample_mode: str = "trilinear",
        align_corners: bool = False,
    ):
        super().__init__(prob=prob)
        self.zoom_range = zoom_range
        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        low, high = self.zoom_range
        params["zoom_factor"] = (
            torch.rand(batch_size, device=device) * (high - low) + low
        )
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        zoom_factors = params["zoom_factor"]
        spatial_shape = list(tensor.shape[2:])

        result = tensor.clone()

        # Integer target sizes per batch element — (B, ndim_spatial)
        targets = torch.stack(
            [
                (s * zoom_factors).round().long().clamp(min=1)
                for s in spatial_shape
            ],
            dim=1,
        )

        active = (mask).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return tensor

        active_targets = targets[active]
        unique_targets, inverse = torch.unique(
            active_targets, dim=0, return_inverse=True
        )

        for i in range(unique_targets.shape[0]):
            group_local = (inverse == i).nonzero(as_tuple=True)[0]
            group_global = active[group_local]

            ts = unique_targets[i].tolist()
            subset = tensor[group_global]

            down = F.interpolate(subset, size=ts, mode=self.downsample_mode)

            up_kwargs: dict = {"size": spatial_shape, "mode": self.upsample_mode}
            if self.upsample_mode in self._ALIGN_CORNERS_MODES:
                up_kwargs["align_corners"] = self.align_corners
            up = F.interpolate(down, **up_kwargs)

            result[group_global] = up

        return result


class RandSimulateLowResolutiond(BatchDictTransform):
    """Dictionary wrapper for RandSimulateLowResolution.

    All keys receive the same zoom factor so paired data stays aligned.
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        zoom_range: tuple[float, float] = (0.5, 1.0),
        downsample_mode: str = "nearest",
        upsample_mode: str = "trilinear",
        align_corners: bool = False,
    ):
        transform = RandSimulateLowResolution(
            prob=prob,
            zoom_range=zoom_range,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
            align_corners=align_corners,
        )
        super().__init__(keys=keys, transform=transform)
