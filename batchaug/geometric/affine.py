from __future__ import annotations

import torch
import torch.nn.functional as F

from ..base import BatchTransform


def _build_rotation_matrices(
    angles: torch.Tensor,
) -> torch.Tensor:
    """Build batched 4x4 rotation matrices from Euler angles.

    Composes Rx(a0) @ Ry(a1) @ Rz(a2) — same order as MONAI.

    Args:
        angles: (B, 3) tensor of Euler angles in radians.

    Returns:
        (B, 4, 4) rotation matrices.
    """
    B = angles.shape[0]
    device = angles.device
    eye = torch.eye(4, device=device, dtype=angles.dtype).unsqueeze(0).expand(B, -1, -1)

    result = eye.clone()

    # Rx — rotation about X axis (Y-Z plane)
    a = angles[:, 0]
    c, s = torch.cos(a), torch.sin(a)
    Rx = eye.clone()
    Rx[:, 1, 1] = c
    Rx[:, 1, 2] = -s
    Rx[:, 2, 1] = s
    Rx[:, 2, 2] = c
    result = result @ Rx

    # Ry — rotation about Y axis (X-Z plane)
    a = angles[:, 1]
    c, s = torch.cos(a), torch.sin(a)
    Ry = eye.clone()
    Ry[:, 0, 0] = c
    Ry[:, 0, 2] = s
    Ry[:, 2, 0] = -s
    Ry[:, 2, 2] = c
    result = result @ Ry

    # Rz — rotation about Z axis (X-Y plane)
    a = angles[:, 2]
    c, s = torch.cos(a), torch.sin(a)
    Rz = eye.clone()
    Rz[:, 0, 0] = c
    Rz[:, 0, 1] = -s
    Rz[:, 1, 0] = s
    Rz[:, 1, 1] = c
    result = result @ Rz

    return result


def _build_shear_matrices(shear: torch.Tensor) -> torch.Tensor:
    """Build batched 4x4 shear matrices.

    Args:
        shear: (B, 6) shear coefficients placed in the off-diagonal
               positions of the 3x3 upper-left block.

    Returns:
        (B, 4, 4) shear matrices.
    """
    B = shear.shape[0]
    device = shear.device
    mat = torch.eye(4, device=device, dtype=shear.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    mat[:, 0, 1] = shear[:, 0]
    mat[:, 0, 2] = shear[:, 1]
    mat[:, 1, 0] = shear[:, 2]
    mat[:, 1, 2] = shear[:, 3]
    mat[:, 2, 0] = shear[:, 4]
    mat[:, 2, 1] = shear[:, 5]
    return mat


def _build_translation_matrices(shift: torch.Tensor) -> torch.Tensor:
    """Build batched 4x4 translation matrices.

    Args:
        shift: (B, 3) translation vector.

    Returns:
        (B, 4, 4) translation matrices.
    """
    B = shift.shape[0]
    device = shift.device
    mat = torch.eye(4, device=device, dtype=shift.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    mat[:, 0, 3] = shift[:, 0]
    mat[:, 1, 3] = shift[:, 1]
    mat[:, 2, 3] = shift[:, 2]
    return mat


def _build_scale_matrices(scale: torch.Tensor) -> torch.Tensor:
    """Build batched 4x4 scale matrices.

    Args:
        scale: (B, 3) scale factors (already includes the +1.0 offset).

    Returns:
        (B, 4, 4) scale matrices.
    """
    B = scale.shape[0]
    device = scale.device
    mat = torch.eye(4, device=device, dtype=scale.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    mat[:, 0, 0] = scale[:, 0]
    mat[:, 1, 1] = scale[:, 1]
    mat[:, 2, 2] = scale[:, 2]
    return mat


def _sample_range(
    param_range, ndim: int, batch_size: int, device: torch.device,
    add_scalar: float = 0.0,
) -> torch.Tensor:
    """Sample per-axis parameters from MONAI-style range specs.

    Args:
        param_range: float, tuple, or tuple-of-tuples.
        ndim: Number of spatial dimensions (3).
        batch_size: B.
        device: Torch device.
        add_scalar: Offset added to sampled values (1.0 for scale).

    Returns:
        (B, ndim) tensor.
    """
    if isinstance(param_range, (int, float)):
        param_range = [param_range] * ndim

    result = torch.zeros(batch_size, ndim, device=device)
    for i, spec in enumerate(param_range):
        if spec is None:
            result[:, i] = add_scalar
        elif isinstance(spec, (int, float)):
            result[:, i] = (
                torch.rand(batch_size, device=device) * 2 * spec - spec + add_scalar
            )
        else:
            low, high = spec
            result[:, i] = (
                torch.rand(batch_size, device=device) * (high - low) + low + add_scalar
            )
    return result


class RandAffine(BatchTransform):
    """Random affine transformation using F.affine_grid + F.grid_sample.

    Composes rotation, shear, translation, and scaling into a single
    4x4 affine matrix per batch element, then resamples via
    ``F.grid_sample``.

    Matrix composition order (same as MONAI):
        Identity @ Rotate @ Shear @ Translate @ Scale

    Input shape: (B, C, H, W, D).
    """

    def __init__(
        self,
        prob: float = 0.1,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ):
        super().__init__(prob=prob)
        self.rotate_range = rotate_range
        self.shear_range = shear_range
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.mode = mode
        self.padding_mode = padding_mode

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        spatial_shape = shape[2:]  # (H, W, D)
        ndim = len(spatial_shape)

        # Build 4x4 affine: I @ R @ Sh @ T @ Sc
        affine = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )

        if self.rotate_range is not None:
            angles = _sample_range(self.rotate_range, ndim, batch_size, device)
            affine = affine @ _build_rotation_matrices(angles)

        if self.shear_range is not None:
            n_shear = ndim * (ndim - 1)  # 6 for 3D
            shear_spec = self.shear_range
            if isinstance(shear_spec, (int, float)):
                shear_spec = [shear_spec] * n_shear
            shear = _sample_range(shear_spec, n_shear, batch_size, device)
            affine = affine @ _build_shear_matrices(shear)

        if self.translate_range is not None:
            shift = _sample_range(self.translate_range, ndim, batch_size, device)
            affine = affine @ _build_translation_matrices(shift)

        if self.scale_range is not None:
            scale = _sample_range(
                self.scale_range, ndim, batch_size, device, add_scalar=1.0
            )
            affine = affine @ _build_scale_matrices(scale)

        # Convert MONAI centered-coordinate affine to F.affine_grid theta.
        #
        # MONAI grid: coords in [-(S-1)/2, (S-1)/2] with order (dim0, dim1, dim2).
        # grid_sample: coords in [-1, 1] with order (x=last_dim, y, z=first_dim).
        # align_corners=False normalization: x_norm = x_monai * 2 / S.
        #
        # theta = S_norm @ P @ affine @ P @ S_inv
        # where P reverses dim order (swap 0↔2), S_norm = diag(2/H, 2/W, 2/D, 1),
        # S_inv = diag(H/2, W/2, D/2, 1).
        H, W, D_dim = spatial_shape
        P = torch.eye(4, device=device, dtype=torch.float32)
        P[0, 0] = 0; P[0, 2] = 1
        P[2, 2] = 0; P[2, 0] = 1

        S_norm = torch.diag(torch.tensor(
            [2.0 / H, 2.0 / W, 2.0 / D_dim, 1.0],
            device=device, dtype=torch.float32,
        ))
        S_inv = torch.diag(torch.tensor(
            [H / 2.0, W / 2.0, D_dim / 2.0, 1.0],
            device=device, dtype=torch.float32,
        ))

        params["theta"] = S_norm @ P @ affine @ P @ S_inv  # (B, 4, 4)
        params["spatial_shape"] = spatial_shape

        return params

    def apply(
        self,
        tensor: torch.Tensor,
        params: dict,
        mode: str | None = None,
        padding_mode: str | None = None,
    ) -> torch.Tensor:
        mask = params["mask"]
        theta = params["theta"]  # (B, 4, 4)
        spatial_shape = params["spatial_shape"]

        if mode is None:
            mode = self.mode
        if padding_mode is None:
            padding_mode = self.padding_mode

        # F.affine_grid expects (B, 3, 4) and produces (B, H, W, D, 3)
        theta_34 = theta[:, :3, :]  # (B, 3, 4)
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


class RandAffined:
    """Dictionary wrapper for RandAffine with per-key interpolation modes.

    Samples one set of affine parameters and applies to all keys,
    but allows different interpolation modes per key (e.g., bilinear
    for volumes, nearest for segmentations).

    Args:
        keys: List of dictionary keys to transform.
        mode: Per-key interpolation mode. Either a single string
            (applied to all keys) or a dict mapping key → mode.
        padding_mode: Per-key padding mode. Same format as ``mode``.
        **kwargs: Forwarded to ``RandAffine``.
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        mode: str | dict[str, str] = "bilinear",
        padding_mode: str | dict[str, str] = "zeros",
    ):
        self.keys = keys
        self.transform = RandAffine(
            prob=prob,
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
        )
        # Resolve per-key modes
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
