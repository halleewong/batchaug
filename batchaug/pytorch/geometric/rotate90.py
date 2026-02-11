from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import BatchDictTransform, BatchTransform
from .affine import monai_affine_to_theta


class RandRotate90(BatchTransform):
    """Randomly rotate by 90-degree increments per batch element.

    Matches MONAI's RandRotate90: samples k in {1, ..., max_k} and
    applies torch.rot90(tensor, k, axes). Different batch elements
    may receive different k values. All channels are rotated identically.

    Input shape: (B, C, H, W, D).
    spatial_axes: which two spatial axes define the rotation plane.
        (0, 1) = H-W plane, (0, 2) = H-D plane, (1, 2) = W-D plane.
    """

    def __init__(
        self,
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axes: tuple[int, int] = (0, 1),
    ):
        super().__init__(prob=prob)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        # MONAI samples k from {1, ..., max_k} (never 0)
        params["k"] = torch.randint(1, self.max_k + 1, (batch_size,), device=device)
        return params

    def to_affine(self, params: dict) -> torch.Tensor:
        """Return (B, 4, 4) MONAI-convention affine for the 90-degree rotation.

        Rotation by k*90 degrees in the (ax0, ax1) plane uses exact
        cos/sin values {0, +/-1}. Masked-out elements get identity.
        """
        mask = params["mask"]
        k_values = params["k"]
        B = mask.shape[0]
        device = mask.device
        ax0, ax1 = self.spatial_axes

        affine = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone()
        )

        # Affine uses the resampling (inverse) convention:
        # k=1: cos=0,sin=-1  k=2: cos=-1,sin=0  k=3: cos=0,sin=1
        cos_table = [0.0, -1.0, 0.0]
        sin_table = [-1.0, 0.0, 1.0]

        for k in range(1, self.max_k + 1):
            sel = mask & (k_values == k)
            if sel.any():
                c = cos_table[k - 1]
                s = sin_table[k - 1]
                affine[sel, ax0, ax0] = c
                affine[sel, ax0, ax1] = -s
                affine[sel, ax1, ax0] = s
                affine[sel, ax1, ax1] = c
        return affine

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        k_values = params["k"]
        # Map spatial axes to tensor dims: spatial 0 -> tensor dim 2, etc.
        tensor_axes = [a + 2 for a in self.spatial_axes]

        # Check if rotation plane is square
        dim0 = tensor.shape[tensor_axes[0]]
        dim1 = tensor.shape[tensor_axes[1]]

        if dim0 == dim1:
            # Fast path: rot90 preserves shape for square planes
            result = tensor.clone()
            for k in range(1, self.max_k + 1):
                batch_mask = mask & (k_values == k)
                if batch_mask.any():
                    result[batch_mask] = torch.rot90(
                        result[batch_mask], k, tensor_axes
                    )
            return result

        # Non-square plane: rot90 would change shape, so fall back to
        # affine_grid + grid_sample (consistent with Compose lazy fusion).
        # 90-degree rotations map to exact voxel centers, so bilinear
        # gives exact values within the valid region.
        affine = self.to_affine(params)
        spatial_shape = tensor.shape[2:]
        theta = monai_affine_to_theta(affine, spatial_shape, tensor.device)
        theta_34 = theta[:, :3, :]

        grid = F.affine_grid(theta_34, list(tensor.shape), align_corners=False)
        result = F.grid_sample(
            tensor.float(), grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        ).to(tensor.dtype)

        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandRotate90d(BatchDictTransform):
    """Dictionary wrapper for RandRotate90.

    All keys receive the same rotation (same k, same mask).
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axes: tuple[int, int] = (0, 1),
    ):
        transform = RandRotate90(prob=prob, max_k=max_k, spatial_axes=spatial_axes)
        super().__init__(keys=keys, transform=transform)
