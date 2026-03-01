from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import BatchTransform
from .affine import monai_affine_to_theta


class RandZoom(BatchTransform):
    """Random isotropic zoom per batch element.

    Equivalent to MONAI's ``RandZoom`` (with ``keep_size=True``).

    A zoom factor ``z ~ U(min_zoom, max_zoom)`` is sampled per element.
    ``z > 1`` zooms **in** (content appears larger, border regions are
    cropped/padded); ``z < 1`` zooms **out** (content appears smaller).

    The inverse-mapping convention used by ``F.grid_sample`` means the
    affine diagonal is ``1/z``: to zoom in, we sample from a smaller
    region of the input.

    Args:
        prob:         Probability of applying to each batch element.
        min_zoom:     Lower bound of zoom factor (must be > 0).
        max_zoom:     Upper bound of zoom factor.
        mode:         Interpolation mode for ``F.grid_sample``.
        padding_mode: Padding mode for ``F.grid_sample``.

    Input shape: ``(B, C, H, W, D)``.
    """

    def __init__(
        self,
        prob: float = 0.1,
        min_zoom: float = 0.9,
        max_zoom: float = 1.1,
        mode: str = "bilinear",
        padding_mode: str = "border",
    ):
        super().__init__(prob=prob)
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.mode = mode
        self.padding_mode = padding_mode

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        spatial_shape = shape[2:]

        # Sample zoom factor: z ~ U(min_zoom, max_zoom) per element
        zoom = (
            torch.rand(batch_size, device=device, dtype=torch.float32)
            * (self.max_zoom - self.min_zoom)
            + self.min_zoom
        )
        params["zoom"] = zoom

        # Build MONAI-space affine (inverse mapping): scale = 1/zoom
        scale = 1.0 / zoom  # (B,)
        affine = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        affine[:, 0, 0] = scale
        affine[:, 1, 1] = scale
        affine[:, 2, 2] = scale

        params["affine"] = affine
        params["theta"] = monai_affine_to_theta(affine, spatial_shape, device)
        params["spatial_shape"] = spatial_shape
        return params

    def to_affine(self, params: dict) -> torch.Tensor:
        """Return (B, 4, 4) MONAI-convention affine (identity for masked-out)."""
        affine = params["affine"]
        mask = params["mask"]
        B = mask.shape[0]
        eye = (
            torch.eye(4, device=affine.device, dtype=affine.dtype)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        return torch.where(mask[:, None, None], affine, eye)

    def apply(
        self,
        tensor: torch.Tensor,
        params: dict,
        mode: str | None = None,
        padding_mode: str | None = None,
    ) -> torch.Tensor:
        mask = params["mask"]
        theta = params["theta"]
        spatial_shape = params["spatial_shape"]

        if mode is None:
            mode = self.mode
        if padding_mode is None:
            padding_mode = self.padding_mode

        theta_34 = theta[:, :3, :]
        grid = F.affine_grid(
            theta_34,
            [tensor.shape[0], tensor.shape[1], *spatial_shape],
            align_corners=False,
        )
        result = F.grid_sample(
            tensor.float(),
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        ).to(tensor.dtype)

        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandZoomd:
    """Dictionary wrapper for RandZoom with per-key interpolation modes.

    Args:
        keys:         Dictionary keys to transform.
        mode:         Interpolation mode — single string or per-key dict.
        padding_mode: Padding mode — single string or per-key dict.
        **kwargs:     Forwarded to ``RandZoom``.
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        min_zoom: float = 0.9,
        max_zoom: float = 1.1,
        mode: str | dict[str, str] = "bilinear",
        padding_mode: str | dict[str, str] = "border",
    ):
        self.keys = keys
        self.transform = RandZoom(
            prob=prob, min_zoom=min_zoom, max_zoom=max_zoom
        )
        if isinstance(mode, str):
            self._mode = {k: mode for k in keys}
        else:
            self._mode = mode
        if isinstance(padding_mode, str):
            self._padding_mode = {k: padding_mode for k in keys}
        else:
            self._padding_mode = padding_mode

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        first_tensor = d[self.keys[0]]
        params = self.transform.sample_params(
            first_tensor.shape[0], first_tensor.shape, first_tensor.device
        )
        for key in self.keys:
            if key in d:
                d[key] = self.transform.apply(
                    d[key],
                    params,
                    mode=self._mode.get(key, "bilinear"),
                    padding_mode=self._padding_mode.get(key, "border"),
                )
        return d
