# BatchAug Transforms Reference

## Overview

BatchAug provides 22 transforms (each with a dict variant `*d`) and a `Compose` class with optional lazy geometric fusion.

**Input shape**: `(B, C, H, W, D)` — batch, channels, 3 spatial dims.

## Transform Summary

| Transform | Type | Fusible | Applies to | Key Parameters |
|---|---|:---:|---|---|
| `RandRotate90d` | Geometric | Yes | vol + seg | `prob`, `max_k`, `spatial_axes` |
| `RandAxisFlipd` | Geometric | Yes | vol + seg | `prob` |
| `RandFlipd` | Geometric | Yes | vol + seg | `prob`, `spatial_axis` |
| `RandRotated` | Geometric | Yes | vol + seg | `prob`, `range_x`, `range_y`, `range_z`, `mode` |
| `RandAffined` | Geometric | Yes | vol + seg | `prob`, `rotate_range`, `shear_range`, `translate_range`, `scale_range`, `mode` |
| `RandZoomd` | Geometric | Yes | vol + seg | `prob`, `min_zoom`, `max_zoom`, `mode` |
| `Rand3DElasticd` | Geometric | No | vol + seg | `prob`, `sigma_range`, `magnitude_range`, `mode` |
| `DivisiblePadd` | Utility | No | vol + seg | `k`, `method`, `mode` |
| `RandSimulateLowResolutiond` | Intensity | No | vol only | `prob`, `zoom_range` |
| `RandConvd` | Intensity | No | vol only | `prob`, `kernel_sizes`, `mixing`, `distribution`, `rand_bias` |
| `RandGaussianNoised` | Intensity | No | vol only | `prob`, `mean`, `std` |
| `RandRicianNoised` | Intensity | No | vol only | `prob`, `mean`, `std`, `relative`, `sample_std` |
| `RandBiasFieldd` | Intensity | No | vol only | `prob`, `degree`, `coeff_range` |
| `RandGibbsNoised` | Intensity | No | vol only | `prob`, `alpha` |
| `RandAdjustContrastd` | Intensity | No | vol only | `prob`, `gamma` |
| `RandScaleIntensityd` | Intensity | No | vol only | `prob`, `factors` |
| `RandScaleIntensityFixedMeand` | Intensity | No | vol only | `prob`, `factors`, `channel_wise` |
| `RandShiftIntensityd` | Intensity | No | vol only | `prob`, `offsets`, `safe` |
| `RandStdShiftIntensityd` | Intensity | No | vol only | `prob`, `factors`, `nonzero`, `channel_wise` |
| `RandGaussianSmoothd` | Intensity | No | vol only | `prob`, `sigma_x`, `sigma_y`, `sigma_z` |
| `RandGaussianSharpend` | Intensity | No | vol only | `prob`, `sigma1_*`, `sigma2_*`, `alpha` |
| `ScaleIntensityd` | Intensity | No | vol only | `minv`, `maxv`, `factor`, `channel_wise` |

**Fusible** = can be composed into a single `grid_sample` call via `Compose(lazy=True)`.

## Geometric Transforms (Fusible)

These transforms remap spatial coordinates without modifying voxel values. In lazy mode, `Compose` fuses consecutive geometric transforms into a single 4x4 affine matrix and applies one `grid_sample` — reducing interpolation artifacts and saving compute.

Each fusible transform implements `to_affine(params) -> (B, 4, 4)` in MONAI convention.

### RandRotate90 / RandRotate90d

Rotate by 90-degree increments in a chosen spatial plane.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `max_k` | int | 3 | k sampled from {1, ..., max_k} (never 0) |
| `spatial_axes` | tuple(int,int) | (0, 1) | Two spatial axes defining the rotation plane |

### RandAxisFlip / RandAxisFlipd

Flip along a randomly chosen spatial axis.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |

### RandFlip / RandFlipd

Flip along specified spatial axes (all at once). Unlike `RandAxisFlip` which picks one random axis, `RandFlip` flips ALL specified axes simultaneously when activated.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `spatial_axis` | int / list / None | None | Axes to flip. None = all spatial axes |

### RandRotate / RandRotated

Arbitrary-angle rotation. Simplified API wrapping `RandAffine` with only rotation enabled.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `range_x` | float / tuple | 0.0 | Rotation range around X (H-W plane), radians |
| `range_y` | float / tuple | 0.0 | Rotation range around Y (H-D plane) |
| `range_z` | float / tuple | 0.0 | Rotation range around Z (W-D plane) |
| `mode` | str / dict | "bilinear" | Interpolation mode |
| `padding_mode` | str / dict | "border" | Padding mode |

### RandAffine / RandAffined

Full affine: rotation + shear + translation + scale via `F.affine_grid` + `F.grid_sample`. Composition order: `I @ R @ Sh @ T @ Sc` (same as MONAI).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `rotate_range` | float / tuple | None | Euler angle range (radians). Scalar = symmetric for all 3 axes |
| `shear_range` | float / tuple | None | 6 shear coefficients. Scalar = symmetric for all |
| `translate_range` | float / tuple | None | Voxel translation range per axis |
| `scale_range` | float / tuple | None | Scale factor range (added to 1.0) |
| `mode` | str / dict | "bilinear" | Interpolation mode. Dict variant supports per-key (e.g. bilinear for vol, nearest for seg) |
| `padding_mode` | str / dict | "zeros" | Padding mode. Same per-key support |

### RandZoom / RandZoomd

Random isotropic zoom (keep_size=True). A zoom factor `z ~ U(min_zoom, max_zoom)` is sampled per element. `z > 1` zooms in, `z < 1` zooms out.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `min_zoom` | float | 0.9 | Lower bound of zoom factor (must be > 0) |
| `max_zoom` | float | 1.1 | Upper bound of zoom factor |
| `mode` | str / dict | "bilinear" | Interpolation mode |
| `padding_mode` | str / dict | "border" | Padding mode |

### Rand3DElastic / Rand3DElasticd

Random elastic deformation via smoothed displacement fields. Generates a random displacement field, smooths it with a Gaussian filter, scales by magnitude, and resamples via `grid_sample`. **Not fusible** — elastic deformations are non-linear and cannot be represented as a 4x4 affine matrix.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `sigma_range` | tuple(float,float) | (3.0, 3.0) | Gaussian smoothing sigma range for displacement field |
| `magnitude_range` | tuple(float,float) | (0.0, 0.1) | Displacement magnitude range |
| `mode` | str / dict | "bilinear" | Interpolation mode. Dict variant supports per-key |
| `padding_mode` | str / dict | "zeros" | Padding mode. Same per-key support |

## Utility Transforms

### DivisiblePad / DivisiblePadd

Deterministic padding to make spatial dimensions divisible by k. Always applied (no probability). Output shape may differ from input.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | int / tuple(int,int,int) | — | Divisor per spatial axis. Scalar = same for all axes |
| `method` | str | "symmetric" | Padding placement: "symmetric" (both sides) or "end" (right/bottom only) |
| `mode` | str | "constant" | Padding mode: "constant", "reflect", "replicate", "circular" |

## Intensity Transforms (Not Fusible)

These modify voxel values and cannot be represented as coordinate remappings.

### RandSimulateLowResolution / RandSimulateLowResolutiond

Simulates low scanner resolution by downsampling then upsampling. Requires two resampling steps (down + up) so cannot be expressed as a single affine.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `zoom_range` | tuple(float,float) | (0.5, 1.0) | Scalar zoom factor range. 0.33 = 3x downsampling |
| `downsample_mode` | str | "nearest" | `F.interpolate` mode for downsampling |
| `upsample_mode` | str | "trilinear" | `F.interpolate` mode for upsampling |

### RandConv / RandConvd

Random convolution augmentation. Perturbs low-level texture and colour statistics while preserving spatial structure by passing each volume through a conv layer with freshly randomized weights. Based on Xu et al. "Robust and Generalizable Visual Representation Learning via Random Convolutions" (ICLR 2021).

Each batch element receives its own independent random kernel. The kernel is applied to each channel separately with no cross-channel mixing (depthwise-style): the same scalar-input kernel is shared across all channels within a batch element. The grouped convolution trick (`groups=B*C`) processes all batch elements and channels in a single CUDA call.

Two modes:
- **RC_img** (`mixing=False`): replace input with conv output entirely.
- **RC_mix** (`mixing=True`): blend input and conv output — `alpha * randconv(x) + (1 - alpha) * x` where `alpha ~ U(0, 1)` per element.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.5 | Per-element probability |
| `kernel_sizes` | int / list[int] | 3 | Kernel size(s) to sample from uniformly each call |
| `mixing` | bool | False | Enable RC_mix blending mode |
| `distribution` | str | "kaiming_normal" | Weight init: `"kaiming_normal"`, `"kaiming_uniform"`, or `"xavier_normal"` |
| `rand_bias` | bool | False | Also randomise the conv bias |

### RandGaussianNoise / RandGaussianNoised

Additive Gaussian noise. Noise tensor is pre-generated in `sample_params` so dict variants share the same noise across keys.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `mean` | float / tuple | 0.0 | Fixed mean or (low, high) range sampled per element |
| `std` | float / tuple | 0.1 | Fixed std or (low, high) range sampled per element |

### RandRicianNoise / RandRicianNoised

Rician-distributed noise (MRI magnitude artifact model). Formula: `output = sqrt((input + n1)^2 + n2^2)` where `n1, n2 ~ N(mean, std^2)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `mean` | float | 0.0 | Mean of both Gaussian noise components |
| `std` | float | 0.1 | Noise std (or upper bound if `sample_std=True`) |
| `relative` | bool | False | If True, multiply std by per-element signal std |
| `sample_std` | bool | True | If True, sample `noise_std ~ U(0, std)` per element |

### RandBiasField / RandBiasFieldd

Multiplicative polynomial bias field (Legendre basis), simulating MRI inhomogeneity. Applied as `out = img * exp(field)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `degree` | int | 3 | Max polynomial degree |
| `coeff_range` | tuple(float,float) | (0.0, 0.1) | Coefficient range per element |

### RandGibbsNoise / RandGibbsNoised

Gibbs ringing artifact via k-space truncation: FFT, apply spherical mask, iFFT.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `alpha` | tuple(float,float) | (0.0, 1.0) | Truncation intensity (0 = identity, 1 = max) |

### RandAdjustContrast / RandAdjustContrastd

Random gamma correction: `((x - min) / range) ^ gamma * range + min`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `gamma` | tuple(float,float) | (0.5, 4.5) | Gamma range sampled per element |

### RandScaleIntensity / RandScaleIntensityd

Multiply each batch element by a random factor. Formula: `output = input * (1 + factor)` where `factor ~ U(factors[0], factors[1])`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `factors` | float / tuple | (0.0, 0.5) | Scale factor range. Scalar `f` → `(-f, f)` |

### RandScaleIntensityFixedMean / RandScaleIntensityFixedMeand

Scale intensity while preserving the mean. Formula: `output = mean + (input - mean) * (1 + factor)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `factors` | float / tuple | (-0.5, 0.5) | Scale factor range. Scalar `f` → `(-f, f)` |
| `channel_wise` | bool | False | Compute mean independently per channel |

### RandShiftIntensity / RandShiftIntensityd

Add a random offset. Formula: `output = input + offset` where `offset ~ U(offsets[0], offsets[1])`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `offsets` | float / tuple | (-0.1, 0.1) | Offset range. Scalar `f` → `(-f, f)` |
| `safe` | bool | False | If True, clamp result to [0, 1] |

### RandStdShiftIntensity / RandStdShiftIntensityd

Shift by a random multiple of per-element standard deviation. Formula: `output = input + factor * std(input)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `factors` | float / tuple | (-3.0, 3.0) | Factor range. Scalar `f` → `(-f, f)` |
| `nonzero` | bool | False | Compute std only from non-zero voxels |
| `channel_wise` | bool | False | Compute std independently per channel |

### RandGaussianSmooth / RandGaussianSmoothd

Gaussian blur via separable 3D grouped convolution. Per-axis sigma sampled independently.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `sigma_x` | tuple(float,float) | (0.25, 1.5) | Sigma range for axis 0 |
| `sigma_y` | tuple(float,float) | (0.25, 1.5) | Sigma range for axis 1 |
| `sigma_z` | tuple(float,float) | (0.25, 1.5) | Sigma range for axis 2 |

### RandGaussianSharpen / RandGaussianSharpend

Unsharp masking: `blurred + alpha * (blurred - double_blurred)`. Two rounds of Gaussian smoothing with independently sampled sigmas.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prob` | float | 0.1 | Per-element probability |
| `sigma1_x/y/z` | tuple(float,float) | (0.5, 1.0) | First blur sigma ranges |
| `sigma2_x/y/z` | float / tuple | 0.5 | Second blur sigma. Scalar: upper bound = sampled sigma1 |
| `alpha` | tuple(float,float) | (10.0, 30.0) | Sharpening strength |

### ScaleIntensity / ScaleIntensityd

Deterministic intensity rescaling to [minv, maxv]. Always applied (no `prob` parameter / prob=1.0).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `minv` | float | 0.0 | Target minimum |
| `maxv` | float | 1.0 | Target maximum |
| `factor` | float | None | If set, multiplies by (1 + factor) instead of min-max scaling |
| `channel_wise` | bool | True | Per-(batch, channel) vs per-batch min/max |

## Compose

Chains transforms with optional lazy geometric fusion.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `transforms` | list | — | Ordered list of transforms |
| `lazy` | bool | False | Enable lazy fusion of consecutive geometric transforms |
| `mode` | dict | None | Per-key interpolation mode for fused grid_sample (e.g. `{"vol": "bilinear", "seg": "nearest"}`) |

### Lazy Fusion

When `lazy=True`, consecutive geometric transforms (those with `to_affine`) are composed into a single `(B, 4, 4)` affine matrix and materialized with one `grid_sample` call when an intensity transform is encountered or the pipeline ends.

**Recommended pipeline order** (group all geometric transforms together):

```
Rot90 → Flip → Affine → Elastic → LowRes → Noise → Bias → Gibbs → Contrast → Smooth → Sharpen → Scale
├── geometric (fused) ──┤      ├   ├──────────────── intensity (sequential) ───────────────────────┤
                               └── not fusible, breaks lazy chain
```

This gives **one interpolation pass** for all geometric transforms instead of three separate ones.
