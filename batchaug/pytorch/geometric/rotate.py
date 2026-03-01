from __future__ import annotations

from .affine import RandAffine


class RandRotate(RandAffine):
    """Random rotation around each axis.

    Simplified API wrapping ``RandAffine`` with only rotation enabled.
    Equivalent to MONAI's ``RandRotate``.

    Args:
        prob:         Probability of applying to each batch element.
        range_x:      Rotation range around X (H-W plane), radians.
                      Scalar ``r`` → ``U(-r, r)``;
                      tuple ``(lo, hi)`` → ``U(lo, hi)``.
        range_y:      Rotation range around Y (H-D plane).
        range_z:      Rotation range around Z (W-D plane).
        mode:         Interpolation mode for ``F.grid_sample``.
        padding_mode: Padding mode for ``F.grid_sample``.

    Input shape: ``(B, C, H, W, D)``.
    """

    def __init__(
        self,
        prob: float = 0.1,
        range_x: float | tuple[float, float] = 0.0,
        range_y: float | tuple[float, float] = 0.0,
        range_z: float | tuple[float, float] = 0.0,
        mode: str = "bilinear",
        padding_mode: str = "border",
    ):
        super().__init__(
            prob=prob,
            rotate_range=[range_x, range_y, range_z],
            mode=mode,
            padding_mode=padding_mode,
        )


class RandRotated:
    """Dictionary wrapper for RandRotate with per-key interpolation modes.

    All keys receive the same rotation (same angles, same mask).
    Different interpolation modes can be specified per key (e.g.
    ``bilinear`` for volumes, ``nearest`` for segmentations).

    Args:
        keys:         Dictionary keys to transform.
        mode:         Interpolation mode — single string or per-key dict.
        padding_mode: Padding mode — single string or per-key dict.
        **kwargs:     Forwarded to ``RandRotate``.
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        range_x: float | tuple[float, float] = 0.0,
        range_y: float | tuple[float, float] = 0.0,
        range_z: float | tuple[float, float] = 0.0,
        mode: str | dict[str, str] = "bilinear",
        padding_mode: str | dict[str, str] = "border",
    ):
        from .affine import RandAffined

        self._inner = RandAffined(
            keys=keys,
            prob=prob,
            rotate_range=[range_x, range_y, range_z],
            mode=mode,
            padding_mode=padding_mode,
        )

    def __call__(self, data):
        return self._inner(data)
