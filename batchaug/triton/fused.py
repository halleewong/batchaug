"""FusedAugment: fused augmentation pipeline with Triton-accelerated elementwise fusion.

Combines all batchaug transforms into a single class with a fixed pipeline order
optimized for kernel fusion:

  Phase 1: Geometric fusion → single grid_sample
    RandAxisFlip → RandRotate90 → RandAffine

  Phase 2: Elastic deformation → separate grid_sample
    Rand3DElastic

  Phase 3: Spatial intensity (separate cuDNN/cuFFT kernels)
    RandGaussianSmooth → RandGaussianSharpen → RandSimulateLowResolution → RandGibbsNoise

  Phase 4: Fused elementwise → 2 Triton kernels (reduction + fused apply)
    ScaleIntensity + RandAdjustContrast + RandBiasField + RandGaussianNoise

Performance notes:
  - Kernel grid is (B*C, tiles) — fully parallelizes over batch AND channels.
  - Noise generated in-kernel via tl.randn — no (B,C,H,W,D) tensor allocation.
  - Affine grid computed once and reused across dict keys in FusedAugmentd.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from ..pytorch.geometric.affine import RandAffine, monai_affine_to_theta
from ..pytorch.geometric.elastic import Rand3DElastic
from ..pytorch.geometric.flip import RandAxisFlip
from ..pytorch.geometric.rotate90 import RandRotate90
from ..pytorch.intensity.gibbs_noise import RandGibbsNoise
from ..pytorch.intensity.resolution import RandSimulateLowResolution
from ..pytorch.intensity.sharpen import RandGaussianSharpen
from ..pytorch.intensity.smooth import RandGaussianSmooth
from .kernels.fused_pipeline import _fused_intensity_kernel
from .kernels.reduce import batched_minmax

_MAX_GRID_DIM = 65535


def _safe_block_size(n_elements: int, min_block: int = 1024) -> int:
    """Choose BLOCK_SIZE so grid dimension stays within CUDA limits."""
    bs = min_block
    while triton.cdiv(n_elements, bs) > _MAX_GRID_DIM:
        bs *= 2
    return bs


class FusedAugment:
    """Fused augmentation pipeline combining all transforms.

    Geometric transforms are composed into a single affine → one grid_sample.
    Elementwise intensity transforms are fused into a single Triton kernel.
    Spatial intensity transforms use existing cuDNN/cuFFT implementations.

    Args:
        flip_prob: Probability of random axis flip.
        rotate90_prob: Probability of random 90-degree rotation.
        max_k: Maximum rotation count for rotate90 (k in {1..max_k}).
        spatial_axes: Rotation plane for rotate90.
        affine_prob: Probability of random affine transform.
        rotate_range: Rotation range for affine (radians).
        shear_range: Shear range for affine.
        translate_range: Translation range for affine (voxels).
        scale_range: Scale range for affine.
        elastic_prob: Probability of elastic deformation.
        sigma_range: Gaussian sigma range for elastic smoothing.
        magnitude_range: Displacement magnitude range for elastic.
        smooth_prob: Probability of Gaussian smoothing.
        smooth_sigma_x/y/z: Sigma ranges per axis for smoothing.
        sharpen_prob: Probability of Gaussian sharpening (unsharp mask).
        sharpen_sigma1_x/y/z: First smooth sigma ranges.
        sharpen_sigma2_x/y/z: Second smooth sigma (scalar or range).
        sharpen_alpha: Sharpening strength range.
        low_res_prob: Probability of low-resolution simulation.
        zoom_range: Zoom factor range for low-res simulation.
        gibbs_prob: Probability of Gibbs noise.
        gibbs_alpha: Alpha range for Gibbs noise.
        scale_intensity: Whether to apply ScaleIntensity (deterministic).
        minv: Target minimum for ScaleIntensity.
        maxv: Target maximum for ScaleIntensity.
        channel_wise: Per-channel or per-element min/max for ScaleIntensity.
        contrast_prob: Probability of gamma correction.
        gamma: Gamma range for contrast adjustment.
        bias_field_prob: Probability of bias field.
        degree: Polynomial degree for bias field (only 3 supported in Triton).
        coeff_range: Coefficient range for bias field.
        noise_prob: Probability of Gaussian noise.
        noise_mean: Mean for Gaussian noise (scalar or range).
        noise_std: Std for Gaussian noise (scalar or range).
        mode: Interpolation mode for geometric transforms.
        padding_mode: Padding mode for geometric transforms.
    """

    def __init__(
        self,
        # Geometric
        flip_prob: float = 0.0,
        rotate90_prob: float = 0.0,
        max_k: int = 3,
        spatial_axes: tuple[int, int] = (0, 1, 2),
        affine_prob: float = 0.0,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        elastic_prob: float = 0.0,
        sigma_range: tuple[float, float] = (3.0, 3.0),
        magnitude_range: tuple[float, float] = (0.0, 0.1),
        # Spatial intensity
        smooth_prob: float = 0.0,
        smooth_sigma_x: tuple[float, float] = (0.25, 1.5),
        smooth_sigma_y: tuple[float, float] = (0.25, 1.5),
        smooth_sigma_z: tuple[float, float] = (0.25, 1.5),
        sharpen_prob: float = 0.0,
        sharpen_sigma1_x: tuple[float, float] = (0.5, 1.0),
        sharpen_sigma1_y: tuple[float, float] = (0.5, 1.0),
        sharpen_sigma1_z: tuple[float, float] = (0.5, 1.0),
        sharpen_sigma2_x: float | tuple[float, float] = 0.5,
        sharpen_sigma2_y: float | tuple[float, float] = 0.5,
        sharpen_sigma2_z: float | tuple[float, float] = 0.5,
        sharpen_alpha: tuple[float, float] = (10.0, 30.0),
        low_res_prob: float = 0.0,
        zoom_range: tuple[float, float] = (0.8, 1.0),
        gibbs_prob: float = 0.0,
        gibbs_alpha: tuple[float, float] = (0.0, 1.0),
        # Elementwise intensity (fused in Triton)
        scale_intensity: bool = True,
        minv: float = 0.0,
        maxv: float = 1.0,
        channel_wise: bool = True,
        contrast_prob: float = 0.0,
        gamma: tuple[float, float] = (0.5, 4.5),
        bias_field_prob: float = 0.0,
        degree: int = 3,
        coeff_range: tuple[float, float] = (0.0, 0.1),
        noise_prob: float = 0.0,
        noise_mean: float | tuple[float, float] = 0.0,
        noise_std: float | tuple[float, float] = 0.1,
        # Geometric settings
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ):
        # --- Geometric transforms ---
        self._flip = RandAxisFlip(prob=flip_prob) if flip_prob > 0 else None
        self._rot90 = (
            RandRotate90(prob=rotate90_prob, max_k=max_k, spatial_axes=spatial_axes)
            if rotate90_prob > 0 else None
        )
        self._affine = (
            RandAffine(
                prob=affine_prob, rotate_range=rotate_range,
                shear_range=shear_range, translate_range=translate_range,
                scale_range=scale_range,
            )
            if affine_prob > 0 else None
        )
        self._elastic = (
            Rand3DElastic(
                prob=elastic_prob, sigma_range=sigma_range,
                magnitude_range=magnitude_range,
            )
            if elastic_prob > 0 else None
        )
        self._geo_transforms = [
            ("flip", self._flip),
            ("rot90", self._rot90),
            ("affine", self._affine),
        ]

        # --- Spatial intensity transforms ---
        self._smooth = (
            RandGaussianSmooth(
                prob=smooth_prob, sigma_x=smooth_sigma_x,
                sigma_y=smooth_sigma_y, sigma_z=smooth_sigma_z,
            )
            if smooth_prob > 0 else None
        )
        self._sharpen = (
            RandGaussianSharpen(
                prob=sharpen_prob,
                sigma1_x=sharpen_sigma1_x, sigma1_y=sharpen_sigma1_y,
                sigma1_z=sharpen_sigma1_z,
                sigma2_x=sharpen_sigma2_x, sigma2_y=sharpen_sigma2_y,
                sigma2_z=sharpen_sigma2_z,
                alpha=sharpen_alpha,
            )
            if sharpen_prob > 0 else None
        )
        self._low_res = (
            RandSimulateLowResolution(prob=low_res_prob, zoom_range=zoom_range)
            if low_res_prob > 0 else None
        )
        self._gibbs = (
            RandGibbsNoise(prob=gibbs_prob, alpha=gibbs_alpha)
            if gibbs_prob > 0 else None
        )
        self._spatial_transforms = [
            ("smooth", self._smooth),
            ("sharpen", self._sharpen),
            ("low_res", self._low_res),
            ("gibbs", self._gibbs),
        ]

        # --- Elementwise intensity (fused in Triton kernel) ---
        self._do_scale = scale_intensity
        self._minv = minv
        self._maxv = maxv
        self._channel_wise = channel_wise

        self._do_contrast = contrast_prob > 0
        self._contrast_prob = contrast_prob
        self._gamma = gamma

        self._do_bias = bias_field_prob > 0
        self._bias_prob = bias_field_prob
        self._degree = degree
        self._coeff_range = coeff_range
        if degree != 3 and self._do_bias:
            raise ValueError("FusedAugment Triton kernel only supports degree=3 for bias field")

        self._do_noise = noise_prob > 0
        self._noise_prob = noise_prob
        self._noise_mean = noise_mean
        self._noise_std = noise_std

        self._has_elementwise = (
            self._do_scale or self._do_contrast
            or self._do_bias or self._do_noise
        )

        # Geometric settings
        self._mode = mode
        self._padding_mode = padding_mode

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_scalar(
        self,
        param: float | tuple[float, float],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(param, (list, tuple)):
            low, high = param
            return torch.rand(batch_size, device=device) * (high - low) + low
        return torch.full((batch_size,), param, device=device)

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device,
    ) -> dict:
        """Sample all random parameters for all enabled transforms."""
        params = {}

        # Geometric
        for name, t in self._geo_transforms:
            if t is not None:
                params[name] = t.sample_params(batch_size, shape, device)

        # Elastic
        if self._elastic is not None:
            params["elastic"] = self._elastic.sample_params(batch_size, shape, device)

        # Spatial intensity
        for name, t in self._spatial_transforms:
            if t is not None:
                params[name] = t.sample_params(batch_size, shape, device)

        # Elementwise: sample params directly
        if self._do_contrast:
            params["contrast_mask"] = torch.rand(batch_size, device=device) < self._contrast_prob
            low, high = self._gamma
            params["gamma"] = torch.rand(batch_size, device=device) * (high - low) + low

        if self._do_bias:
            params["bias_mask"] = torch.rand(batch_size, device=device) < self._bias_prob
            low, high = self._coeff_range
            params["coeffs"] = torch.rand(batch_size, 20, device=device) * (high - low) + low

        if self._do_noise:
            params["noise_mask"] = torch.rand(batch_size, device=device) < self._noise_prob
            params["noise_std"] = self._sample_scalar(self._noise_std, batch_size, device)
            params["noise_mean"] = self._sample_scalar(self._noise_mean, batch_size, device)
            params["noise_seed"] = torch.randint(
                0, 2**31 - 1, (batch_size,), device=device, dtype=torch.int32,
            )

        return params

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(
        self,
        tensor: torch.Tensor,
        params: dict,
        mode: str | None = None,
        padding_mode: str | None = None,
    ) -> torch.Tensor:
        """Apply the full augmentation pipeline."""
        if mode is None:
            mode = self._mode
        if padding_mode is None:
            padding_mode = self._padding_mode

        x = tensor
        B = x.shape[0]
        device = x.device
        spatial_shape = x.shape[2:]

        # Phase 1: Geometric fusion (compose affines → single grid_sample)
        affine = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone()
        )
        geo_mask = torch.zeros(B, dtype=torch.bool, device=device)
        has_geo = False

        for name, t in self._geo_transforms:
            if name in params:
                p = params[name]
                affine = affine @ t.to_affine(p)
                geo_mask = geo_mask | p["mask"]
                has_geo = True

        if has_geo and geo_mask.any():
            theta = monai_affine_to_theta(affine, spatial_shape, device)
            theta_34 = theta[:, :3, :]
            grid = F.affine_grid(theta_34, list(x.shape), align_corners=False)
            resampled = F.grid_sample(
                x.float(), grid,
                mode=mode, padding_mode=padding_mode, align_corners=False,
            ).to(x.dtype)
            mask_5d = geo_mask[:, None, None, None, None]
            x = torch.where(mask_5d, resampled, x)

        # Phase 2: Elastic deformation
        if "elastic" in params:
            x = self._elastic.apply(x, params["elastic"], mode=mode, padding_mode=padding_mode)

        # Phase 3: Spatial intensity
        for name, t in self._spatial_transforms:
            if name in params:
                x = t.apply(x, params[name])

        # Phase 4: Fused elementwise (Triton kernel)
        if self._has_elementwise:
            x = self._fused_elementwise(x, params)

        return x

    def _fused_elementwise(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply fused elementwise intensity transforms via Triton."""
        B, C, H, W, D = tensor.shape
        HWD = H * W * D
        N_per_batch = C * HWD
        device = tensor.device

        flat = tensor.contiguous()
        output = torch.empty_like(flat)

        # Min/max reduction (needed if scale or contrast is enabled)
        if self._do_scale or self._do_contrast:
            if self._channel_wise:
                mins, maxs = batched_minmax(
                    flat.reshape(B * C, HWD), B * C, HWD,
                )
            else:
                mins, maxs = batched_minmax(
                    flat.reshape(B, N_per_batch), B, N_per_batch,
                )
        else:
            mins = torch.empty(1, device=device, dtype=torch.float32)
            maxs = torch.empty(1, device=device, dtype=torch.float32)

        # Prepare elementwise params — small dummy tensors for disabled transforms
        gamma = params["gamma"] if self._do_contrast else torch.empty(1, device=device)
        coeffs = params["coeffs"] if self._do_bias else torch.empty(1, device=device)

        contrast_mask = (
            params["contrast_mask"] if self._do_contrast
            else torch.empty(1, device=device, dtype=torch.bool)
        )
        bias_mask = (
            params["bias_mask"] if self._do_bias
            else torch.empty(1, device=device, dtype=torch.bool)
        )
        noise_mask = (
            params["noise_mask"] if self._do_noise
            else torch.empty(1, device=device, dtype=torch.bool)
        )
        noise_seed = (
            params["noise_seed"] if self._do_noise
            else torch.empty(1, device=device, dtype=torch.int32)
        )
        noise_std = (
            params["noise_std"] if self._do_noise
            else torch.empty(1, device=device)
        )
        noise_mean = (
            params["noise_mean"] if self._do_noise
            else torch.empty(1, device=device)
        )

        # Launch fused kernel — grid over B*C (full channel parallelism)
        BLOCK_SIZE = _safe_block_size(HWD)
        grid = (B * C, triton.cdiv(HWD, BLOCK_SIZE))

        _fused_intensity_kernel[grid](
            flat, output,
            coeffs.contiguous(), gamma.contiguous(),
            mins, maxs,
            contrast_mask.contiguous(), bias_mask.contiguous(), noise_mask.contiguous(),
            noise_seed.contiguous(), noise_std.contiguous(), noise_mean.contiguous(),
            self._minv, self._maxv,
            B, C, H, W, D,
            DO_SCALE=self._do_scale,
            DO_CONTRAST=self._do_contrast,
            DO_BIAS=self._do_bias,
            DO_NOISE=self._do_noise,
            CHANNEL_WISE=self._channel_wise,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output.to(tensor.dtype)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        params = self.sample_params(tensor.shape[0], tensor.shape, tensor.device)
        return self.apply(tensor, params)


class FusedAugmentd:
    """Dictionary version of FusedAugment.

    Geometric transforms apply to all keys; intensity transforms apply
    only to ``intensity_keys``.

    Args:
        keys: All dictionary keys (geometric transforms applied to all).
        intensity_keys: Keys for intensity transforms. Defaults to all keys.
        mode: Interpolation mode. String or dict mapping key → mode.
        padding_mode: Padding mode. String or dict mapping key → mode.
        **kwargs: All FusedAugment parameters.
    """

    def __init__(
        self,
        keys: list[str],
        intensity_keys: list[str] | None = None,
        mode: str | dict[str, str] = "bilinear",
        padding_mode: str | dict[str, str] = "zeros",
        **kwargs,
    ):
        self.keys = keys
        self.intensity_keys = intensity_keys if intensity_keys is not None else keys

        # Resolve per-key modes
        if isinstance(mode, str):
            self._mode = {k: mode for k in keys}
        else:
            self._mode = mode
        if isinstance(padding_mode, str):
            self._padding_mode = {k: padding_mode for k in keys}
        else:
            self._padding_mode = padding_mode

        # Build the inner FusedAugment (without mode/padding_mode — those are per-key)
        self._transform = FusedAugment(**kwargs)

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        d = dict(data)
        first_tensor = d[self.keys[0]]
        B = first_tensor.shape[0]
        shape = first_tensor.shape
        device = first_tensor.device

        params = self._transform.sample_params(B, shape, device)

        # Phase 1: Geometric — apply to all keys with shared grid
        spatial_shape = shape[2:]
        affine = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone()
        )
        geo_mask = torch.zeros(B, dtype=torch.bool, device=device)
        has_geo = False

        for name, t in self._transform._geo_transforms:
            if name in params:
                p = params[name]
                affine = affine @ t.to_affine(p)
                geo_mask = geo_mask | p["mask"]
                has_geo = True

        if has_geo and geo_mask.any():
            theta = monai_affine_to_theta(affine, spatial_shape, device)
            theta_34 = theta[:, :3, :]
            mask_5d = geo_mask[:, None, None, None, None]

            # Compute grid once and reuse across same-shape keys
            grid = F.affine_grid(theta_34, list(shape), align_corners=False)

            for key in self.keys:
                if key not in d:
                    continue
                tensor = d[key]
                key_mode = self._mode.get(key, "bilinear")
                key_pad = self._padding_mode.get(key, "zeros")
                resampled = F.grid_sample(
                    tensor.float(), grid,
                    mode=key_mode, padding_mode=key_pad, align_corners=False,
                ).to(tensor.dtype)
                d[key] = torch.where(mask_5d, resampled, tensor)

        # Phase 2: Elastic — apply to all keys with per-key mode
        if "elastic" in params:
            for key in self.keys:
                if key not in d:
                    continue
                key_mode = self._mode.get(key, "bilinear")
                key_pad = self._padding_mode.get(key, "zeros")
                d[key] = self._transform._elastic.apply(
                    d[key], params["elastic"],
                    mode=key_mode, padding_mode=key_pad,
                )

        # Phase 3 + 4: Spatial + elementwise intensity — apply to intensity_keys
        for key in self.intensity_keys:
            if key not in d:
                continue
            tensor = d[key]
            # Phase 3: Spatial intensity
            for name, t in self._transform._spatial_transforms:
                if name in params:
                    tensor = t.apply(tensor, params[name])
            # Phase 4: Fused elementwise
            if self._transform._has_elementwise:
                tensor = self._transform._fused_elementwise(tensor, params)
            d[key] = tensor

        return d
