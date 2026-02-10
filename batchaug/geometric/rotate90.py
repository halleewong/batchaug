from __future__ import annotations

import torch

from ..base import BatchDictTransform, BatchTransform


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

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        k_values = params["k"]
        # Map spatial axes to tensor dims: spatial 0 -> tensor dim 2, etc.
        tensor_axes = [a + 2 for a in self.spatial_axes]

        result = tensor.clone()
        for k in range(1, self.max_k + 1):
            batch_mask = mask & (k_values == k)
            if batch_mask.any():
                result[batch_mask] = torch.rot90(
                    result[batch_mask], k, tensor_axes
                )
        return result


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
