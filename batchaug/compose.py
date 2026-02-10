from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometric.affine import monai_affine_to_theta


class Compose:
    """Compose multiple dict transforms with optional lazy geometric fusion.

    In eager mode (``lazy=False``), transforms are applied sequentially.

    In lazy mode (``lazy=True``), consecutive geometric transforms
    (those whose ``.transform`` has a ``to_affine`` method) are fused
    into a single affine matrix per key and materialized via one
    ``F.grid_sample`` call.  This preserves image quality (single
    interpolation pass) and reduces compute.

    Intensity transforms are applied eagerly at their position in the
    pipeline.  When an intensity transform appears between two groups
    of geometric transforms, the first group is materialized before
    the intensity transform runs, and the second group starts a new
    accumulation.

    Args:
        transforms: List of dict-level transforms.
        lazy: Enable lazy geometric fusion.
        mode: Interpolation mode for materialization.  A single string
            applies to all keys; a ``dict`` maps key → mode.
        padding_mode: Padding mode for materialization.  Same format
            as *mode*.
    """

    def __init__(
        self,
        transforms: list,
        lazy: bool = False,
        mode: str | dict[str, str] = "bilinear",
        padding_mode: str | dict[str, str] = "zeros",
    ):
        self.transforms = transforms
        self.lazy = lazy
        self._mode = mode
        self._padding_mode = padding_mode

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_geometric(t) -> bool:
        """Check if *t* is a dict transform wrapping a geometric BatchTransform."""
        inner = getattr(t, "transform", None)
        return inner is not None and hasattr(inner, "to_affine")

    def _get_mode(self, key: str) -> str:
        if isinstance(self._mode, dict):
            return self._mode.get(key, "bilinear")
        return self._mode

    def _get_padding_mode(self, key: str) -> str:
        if isinstance(self._padding_mode, dict):
            return self._padding_mode.get(key, "zeros")
        return self._padding_mode

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _materialize(
        self,
        data: dict[str, torch.Tensor],
        per_key_affines: dict[str, torch.Tensor],
        per_key_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply accumulated affine matrices via a single grid_sample per key."""
        d = dict(data)

        for key in per_key_affines:
            if key not in d:
                continue

            mask = per_key_masks[key]
            if not mask.any():
                continue  # nothing to resample for this key

            tensor = d[key]
            affine = per_key_affines[key]
            spatial_shape = tensor.shape[2:]

            theta = monai_affine_to_theta(affine, spatial_shape, tensor.device)
            theta_34 = theta[:, :3, :]  # (B, 3, 4)

            grid = F.affine_grid(
                theta_34, list(tensor.shape), align_corners=False,
            )

            mode = self._get_mode(key)
            padding_mode = self._get_padding_mode(key)

            result = F.grid_sample(
                tensor.float(), grid,
                mode=mode, padding_mode=padding_mode, align_corners=False,
            ).to(tensor.dtype)

            mask_5d = mask[:, None, None, None, None]
            d[key] = torch.where(mask_5d, result, tensor)

        return d

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(
        self, data: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.lazy:
            # Eager mode — simple sequential application.
            d = dict(data)
            for t in self.transforms:
                d = t(d)
            return d

        # Lazy mode — accumulate geometric affines per key.
        d = dict(data)

        # Infer batch info from first tensor in the dict.
        first_key = next(iter(d))
        first_tensor = d[first_key]
        B = first_tensor.shape[0]
        shape = first_tensor.shape
        device = first_tensor.device

        per_key_affines: dict[str, torch.Tensor] = {}
        per_key_masks: dict[str, torch.Tensor] = {}

        for t in self.transforms:
            if self._is_geometric(t):
                inner = t.transform
                params = inner.sample_params(B, shape, device)
                affine = inner.to_affine(params)  # (B, 4, 4)

                for key in t.keys:
                    if key in per_key_affines:
                        per_key_affines[key] = per_key_affines[key] @ affine
                        per_key_masks[key] = per_key_masks[key] | params["mask"]
                    else:
                        per_key_affines[key] = affine.clone()
                        per_key_masks[key] = params["mask"].clone()
            else:
                # Non-geometric: materialize pending, then apply eagerly.
                if per_key_affines:
                    d = self._materialize(d, per_key_affines, per_key_masks)
                    per_key_affines.clear()
                    per_key_masks.clear()
                d = t(d)

        # Materialize any remaining geometric transforms.
        if per_key_affines:
            d = self._materialize(d, per_key_affines, per_key_masks)

        return d
