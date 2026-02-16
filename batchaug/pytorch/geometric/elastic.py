from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import BatchTransform
from ..intensity.smooth import gaussian_1d_batch, separable_gaussian_conv3d


class Rand3DElastic(BatchTransform):
    """Random elastic deformation via smoothed displacement fields.

    Steps:
        1. Sample random displacement field (B, 3, H, W, D) in [-1, 1]
        2. Smooth with Gaussian filter (truncated=3.0 to match MONAI)
        3. Scale by sampled magnitude
        4. Add to coordinate grid
        5. Resample via F.grid_sample

    Input shape: (B, C, H, W, D).
    """

    def __init__(
        self,
        prob: float = 0.1,
        sigma_range: tuple[float, float] = (3.0, 3.0),
        magnitude_range: tuple[float, float] = (0.0, 0.1),
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ):
        super().__init__(prob=prob)
        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.mode = mode
        self.padding_mode = padding_mode

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        H, W, D = shape[2:]

        # Random displacement field (B, 3, H, W, D) in [-1, 1]
        displacement = torch.rand(batch_size, 3, H, W, D, device=device) * 2 - 1

        # Sample sigma per element
        s_lo, s_hi = self.sigma_range
        sigma = torch.rand(batch_size, device=device) * (s_hi - s_lo) + s_lo

        # Sample magnitude per element
        m_lo, m_hi = self.magnitude_range
        magnitude = torch.rand(batch_size, device=device) * (m_hi - m_lo) + m_lo

        # Smooth each displacement component with Gaussian (truncated=3.0 matches MONAI)
        kernel = gaussian_1d_batch(sigma, truncated=3.0)  # (B, K)
        for i in range(3):
            comp = displacement[:, i : i + 1, :, :, :]  # (B, 1, H, W, D)
            comp = separable_gaussian_conv3d(comp, kernel, kernel, kernel)
            displacement[:, i : i + 1, :, :, :] = comp

        # Scale by magnitude
        displacement = displacement * magnitude[:, None, None, None, None]

        # Build coordinate grid in MONAI convention: linspace(-(d-1)/2, (d-1)/2, d)
        coords = []
        for dim_size in [H, W, D]:
            c = torch.linspace(
                -(dim_size - 1) / 2.0,
                (dim_size - 1) / 2.0,
                dim_size,
                device=device,
                dtype=torch.float32,
            )
            coords.append(c)
        grid_h, grid_w, grid_d = torch.meshgrid(coords, indexing="ij")
        # (3, H, W, D) → (1, 3, H, W, D) → (B, 3, H, W, D)
        grid = (
            torch.stack([grid_h, grid_w, grid_d], dim=0)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1, -1)
        )

        # Add displacement to grid
        grid = grid + displacement

        # Normalize to [-1, 1] for grid_sample
        sizes = torch.tensor([H, W, D], device=device, dtype=torch.float32)
        half = (sizes - 1) / 2.0
        # Avoid division by zero for size-1 dims
        half = half.clamp(min=1e-6)
        grid = grid / half.view(1, 3, 1, 1, 1)

        # Convert from MONAI (H, W, D) order to grid_sample (x=D, y=W, z=H) order
        # grid_sample expects (B, H, W, D, 3) with last dim = (x, y, z) = (D, W, H)
        grid = grid.permute(0, 2, 3, 4, 1)  # (B, H, W, D, 3)
        grid = grid.flip(-1)  # reverse axis order: (H, W, D) → (D, W, H)

        params["grid"] = grid
        return params

    def apply(
        self,
        tensor: torch.Tensor,
        params: dict,
        mode: str | None = None,
        padding_mode: str | None = None,
    ) -> torch.Tensor:
        mask = params["mask"]
        grid = params["grid"]

        if mode is None:
            mode = self.mode
        if padding_mode is None:
            padding_mode = self.padding_mode

        result = F.grid_sample(
            tensor.float(),
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        ).to(tensor.dtype)

        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class Rand3DElasticd:
    """Dictionary wrapper for Rand3DElastic with per-key interpolation modes.

    Samples one set of elastic parameters and applies to all keys,
    but allows different interpolation modes per key (e.g., bilinear
    for volumes, nearest for segmentations).
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        sigma_range: tuple[float, float] = (3.0, 3.0),
        magnitude_range: tuple[float, float] = (0.0, 0.1),
        mode: str | dict[str, str] = "bilinear",
        padding_mode: str | dict[str, str] = "zeros",
    ):
        self.keys = keys
        self.transform = Rand3DElastic(
            prob=prob,
            sigma_range=sigma_range,
            magnitude_range=magnitude_range,
        )
        if isinstance(mode, str):
            self._mode = {k: mode for k in keys}
        else:
            self._mode = mode
        if isinstance(padding_mode, str):
            self._padding_mode = {k: padding_mode for k in keys}
        else:
            self._padding_mode = padding_mode

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
                    padding_mode=self._padding_mode.get(key, "zeros"),
                )
        return d
