from __future__ import annotations

import torch

from ...base import BatchDictTransform, BatchTransform


class RandAxisFlip(BatchTransform):
    """Randomly flip along one spatial axis per batch element.

    Matches MONAI's RandAxisFlip: randomly selects one of the spatial
    dimensions and flips along it. Different batch elements may flip
    along different axes. All channels within a batch element are
    flipped identically.

    Input shape: (B, C, H, W, D) where spatial dims are H, W, D.
    """

    def __init__(self, prob: float = 0.1):
        super().__init__(prob=prob)

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        num_spatial = len(shape) - 2  # exclude B, C
        # Random axis per batch element: 0, 1, or 2 for 3D
        params["axes"] = torch.randint(num_spatial, (batch_size,), device=device)
        return params

    def to_affine(self, params: dict) -> torch.Tensor:
        """Return (B, 4, 4) MONAI-convention affine for the flip.

        Flipping spatial axis i negates coordinate i (diagonal -1).
        Masked-out elements get identity.
        """
        mask = params["mask"]
        axes = params["axes"]
        B = mask.shape[0]
        device = mask.device

        affine = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone()
        )
        for ax in range(3):
            sel = mask & (axes == ax)
            if sel.any():
                affine[sel, ax, ax] = -1.0
        return affine

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        axes = params["axes"]
        num_spatial = tensor.ndim - 2

        result = tensor.clone()
        for ax in range(num_spatial):
            # Batch elements that should be flipped on this specific axis
            batch_mask = mask & (axes == ax)
            if batch_mask.any():
                # Spatial dim ax maps to tensor dim ax + 2 (skip B, C)
                result[batch_mask] = torch.flip(result[batch_mask], dims=[ax + 2])
        return result


class RandAxisFlipd(BatchDictTransform):
    """Dictionary wrapper for RandAxisFlip.

    All keys receive the same flip (same axis, same mask).
    """

    def __init__(self, keys: list[str], prob: float = 0.1):
        transform = RandAxisFlip(prob=prob)
        super().__init__(keys=keys, transform=transform)


class RandFlip(BatchTransform):
    """Randomly flip along specified spatial axes.

    Equivalent to MONAI's ``RandFlip``.

    Unlike ``RandAxisFlip`` (which picks one random axis per element),
    ``RandFlip`` flips ALL specified axes simultaneously when activated.

    Args:
        prob:         Probability of applying the flip.
        spatial_axis: Which spatial axes to flip.
                      - ``None``: flip all spatial axes (H, W, D).
                      - ``int``: flip a single axis.
                      - ``list[int]``: flip each listed axis.
                      Axis 0 → H, axis 1 → W, axis 2 → D.

    Input shape: ``(B, C, H, W, D)``.
    """

    def __init__(
        self,
        prob: float = 0.1,
        spatial_axis: int | list[int] | None = None,
    ):
        super().__init__(prob=prob)
        if spatial_axis is None:
            self._spatial_axes = None  # flip all
        elif isinstance(spatial_axis, int):
            self._spatial_axes = [spatial_axis]
        else:
            self._spatial_axes = list(spatial_axis)

    def _flip_dims(self, ndim: int) -> list[int]:
        """Return tensor dims to flip (accounting for B, C offset)."""
        num_spatial = ndim - 2
        if self._spatial_axes is None:
            return list(range(2, 2 + num_spatial))
        return [ax + 2 for ax in self._spatial_axes]

    def to_affine(self, params: dict) -> torch.Tensor:
        """Return (B, 4, 4) MONAI-convention affine for the flip.

        Activated elements get -1 on the diagonal for each flipped axis;
        masked-out elements get identity.
        """
        mask = params["mask"]
        B = mask.shape[0]
        device = mask.device

        affine = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone()
        )
        spatial_axes = self._spatial_axes if self._spatial_axes is not None else [0, 1, 2]
        for ax in spatial_axes:
            affine[mask, ax, ax] = -1.0
        return affine

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        if not mask.any():
            return tensor

        dims = self._flip_dims(tensor.ndim)
        result = tensor.clone()
        result[mask] = torch.flip(result[mask], dims=dims)
        return result


class RandFlipd(BatchDictTransform):
    """Dictionary wrapper for RandFlip.

    All keys receive the same flip (same axes, same mask).
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        spatial_axis: int | list[int] | None = None,
    ):
        transform = RandFlip(prob=prob, spatial_axis=spatial_axis)
        super().__init__(keys=keys, transform=transform)
