from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import BatchDictTransform, BatchTransform


class RandSimulateLowResolution(BatchTransform):
    """Simulate low resolution by downsampling then upsampling.

    A scalar zoom factor is sampled per batch element from
    ``zoom_range``.  The spatial dimensions are scaled uniformly,
    downsampled with ``downsample_mode``, and upsampled back to the
    original size with ``upsample_mode``.

    Batch elements with the same integer target size are grouped and
    processed together via ``F.interpolate``.

    Input shape: (B, C, H, W, D).

    Note:
        ``downsample_mode`` and ``upsample_mode`` are independent and accept any
        mode supported by ``torch.nn.functional.interpolate`` (``"nearest"``,
        ``"nearest-exact"``, ``"trilinear"``, ``"area"``, …). ``align_corners``
        is automatically applied to whichever step uses a mode that supports it.

        The defaults (``downsample_mode="nearest"``, ``upsample_mode="trilinear"``,
        ``align_corners=False``) match MONAI's ``RandSimulateLowResolution``, but
        produce a spatial shift of roughly +½ pixel toward +H/+W/+D. The cause is
        that ``mode="nearest"`` uses a corner-aligned indexing convention while
        ``trilinear`` with ``align_corners=False`` uses a cell-centered one, so
        the two steps don't compose cleanly.

        Recommended overrides:

        * ``downsample_mode="nearest-exact"`` — also cell-centered, so the
          conventions match. Reduces the shift to <0.02 px for zoom factors of
          0.6–0.9. Worst case is exactly zoom=0.5, where nearest quantization
          still leaves a -0.5 px residual.
        * ``downsample_mode="trilinear"`` — true cell-centered downsample, ~0
          shift at all zoom factors. Smoother result, so it no longer behaves
          like a true "nearest" resample.
        * ``downsample_mode="area"`` — area-averaged (anti-aliased) downsample,
          ~0 shift at all zoom factors. Similar trade-off to ``"trilinear"``.
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

    def _interp_kwargs(self, size, mode) -> dict:
        kwargs: dict = {"size": size, "mode": mode}
        if mode in self._ALIGN_CORNERS_MODES:
            kwargs["align_corners"] = self.align_corners
        return kwargs

    def _lowres_one(self, sub, ts, spatial_shape):
        """Down- then up-sample one (n, C, *spatial) subset back to spatial_shape."""
        down = F.interpolate(sub, **self._interp_kwargs(ts, self.downsample_mode))
        return F.interpolate(down, **self._interp_kwargs(spatial_shape, self.upsample_mode))

    def _lowres_indexed(self, tensor, active, targets, spatial_shape):
        """Fast path: gather elements sharing a target size, resample the group in one
        interpolate call, scatter back."""
        result = tensor.clone()
        active_targets = targets[active]
        unique_targets, inverse = torch.unique(active_targets, dim=0, return_inverse=True)
        for i in range(unique_targets.shape[0]):
            group_global = active[(inverse == i).nonzero(as_tuple=True)[0]]
            ts = unique_targets[i].tolist()
            result[group_global] = self._lowres_one(tensor[group_global], ts, spatial_shape)
        return result

    def _lowres_looped(self, tensor, active, targets, spatial_shape):
        """Fallback: resample each selected element via basic slicing (no advanced indexing,
        so no gather-kernel grid limit). Touches only selected elements; same result as
        _lowres_indexed."""
        result = tensor.clone()
        for idx in active.tolist():
            ts = targets[idx].tolist()
            result[idx:idx + 1] = self._lowres_one(tensor[idx:idx + 1], ts, spatial_shape)
        return result

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        zoom_factors = params["zoom_factor"]
        spatial_shape = list(tensor.shape[2:])

        # Integer target sizes per batch element — (B, ndim_spatial)
        targets = torch.stack(
            [
                (s * zoom_factors).round().long().clamp(min=1)
                for s in spatial_shape
            ],
            dim=1,
        )

        active = mask.nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return tensor

        # Group elements by target size and resample each group via boolean-index
        # gather/scatter. On some CUDA builds (torch 2.8/cu128, not 2.6/cu124) the multi-index
        # gather kernel's grid height scales with the per-element volume and overflows CUDA's
        # 65535 grid-dim cap above ~2**26 voxels/element, raising "invalid configuration
        # argument". That synchronous launch error leaves the context healthy, so we fall back
        # to a per-element loop (basic slicing only — works at any size). Both paths give
        # identical results; normal-sized batches always take the fast path.
        try:
            return self._lowres_indexed(tensor, active, targets, spatial_shape)
        except RuntimeError:
            return self._lowres_looped(tensor, active, targets, spatial_shape)


class RandSimulateLowResolutiond(BatchDictTransform):
    """Dictionary wrapper for RandSimulateLowResolution.

    All keys receive the same zoom factor so paired data stays aligned.

    See ``RandSimulateLowResolution`` for notes on the spatial shift produced
    by the MONAI-matching defaults and the recommended overrides.
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
