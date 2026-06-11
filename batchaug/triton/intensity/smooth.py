"""Triton-accelerated RandGaussianSmooth."""
from __future__ import annotations

import torch

from ...base import BatchDictTransform
from ...pytorch.intensity.smooth import RandGaussianSmooth as _PTRandGaussianSmooth
from ..kernels.separable_conv import separable_gaussian_conv3d_triton

# The Triton separable-conv kernel only does zero-padding. None (implicit zero
# via conv3d) and "constant" (explicit zero) are numerically identical to it;
# any other mode would need a real padded buffer the kernel doesn't build.
_SUPPORTED_PADDING_MODES = (None, "constant")


def _check_padding_mode(cls_name: str, padding_mode: str | None) -> None:
    if padding_mode not in _SUPPORTED_PADDING_MODES:
        raise NotImplementedError(
            f"The Triton {cls_name} kernel only supports zero-padding "
            f'(padding_mode=None or "constant"); got padding_mode={padding_mode!r}. '
            f"Use the PyTorch backend (batchaug.{cls_name}) for "
            f'"reflect"/"replicate" padding.'
        )


class RandGaussianSmooth(_PTRandGaussianSmooth):
    """Triton-accelerated RandGaussianSmooth (same API as PyTorch version).

    Note: the Triton kernel only supports zero-padding. Unlike the PyTorch
    version (which defaults to ``padding_mode="reflect"``), this class defaults
    to zero-padding and raises ``NotImplementedError`` for any other mode.
    """

    def __init__(
        self,
        prob: float = 0.1,
        sigma_x: tuple[float, float] = (0.25, 1.5),
        sigma_y: tuple[float, float] = (0.25, 1.5),
        sigma_z: tuple[float, float] = (0.25, 1.5),
        padding_mode: str | None = None,
    ):
        _check_padding_mode("RandGaussianSmooth", padding_mode)
        super().__init__(
            prob=prob, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z,
            padding_mode=padding_mode,
        )

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        result = separable_gaussian_conv3d_triton(
            tensor, params["kernel_h"], params["kernel_w"], params["kernel_d"]
        )
        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandGaussianSmoothd(BatchDictTransform):
    """Dictionary wrapper for Triton RandGaussianSmooth.

    Like the non-dict version, only zero-padding is supported; any other
    ``padding_mode`` raises ``NotImplementedError``.
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        sigma_x: tuple[float, float] = (0.25, 1.5),
        sigma_y: tuple[float, float] = (0.25, 1.5),
        sigma_z: tuple[float, float] = (0.25, 1.5),
        padding_mode: str | None = None,
    ):
        transform = RandGaussianSmooth(
            prob=prob, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z,
            padding_mode=padding_mode
        )
        super().__init__(keys=keys, transform=transform)
