# Parameter Sampling Across Batch and Channel Dimensions

Every `BatchTransform` samples parameters once per batch in `sample_params()`, then applies them identically to all dict keys via `BatchDictTransform`. This document describes what is sampled and at what granularity.

## Summary Table

| Transform | User Parameters | Sampled Per Batch Element | Sampled Per Channel | Shared Across Batch |
|-----------|----------------|--------------------------|--------------------|--------------------|
| **ScaleIntensity** | `minv`, `maxv`, `factor`, `channel_wise` | min/max (from data) | min/max (if `channel_wise=True`) | — |
| **RandAdjustContrast** | `gamma=(lo, hi)` | gamma, min, max | — | — |
| **RandGaussianNoise** | `mean`, `std` | mean, std | — | noise tensor `(B,C,H,W,D)` |
| **RandGaussianSmooth** | `sigma_x/y/z=(lo, hi)` | sigma_x, sigma_y, sigma_z | — | 1D kernels |
| **RandGaussianSharpen** | `sigma1_x/y/z`, `sigma2_x/y/z`, `alpha` | sigma1_x/y/z, sigma2_x/y/z, alpha | — | blur kernels |
| **RandSimulateLowResolution** | `zoom_range=(lo, hi)` | zoom_factor | — | — |
| **RandBiasField** | `degree`, `coeff_range=(lo, hi)` | coefficients `(n_coeff,)` | — | Legendre basis `(n_coeff,H,W,D)` |
| **RandGibbsNoise** | `alpha=(lo, hi)` | alpha, k-space mask | — | — |
| **RandAxisFlip** | — | axis (from {0,1,2}) | — | — |
| **RandRotate90** | `max_k`, `spatial_axes` | k (from {1,...,max_k}) | — | — |
| **RandAffine** | `rotate_range`, `shear_range`, `translate_range`, `scale_range` | angles `(3,)`, shear `(6,)`, shift `(3,)`, scale `(3,)` | — | composed 4x4 affine |
| **Rand3DElastic** | `sigma_range`, `magnitude_range` | sigma, magnitude, displacement `(3,H,W,D)` | — | smoothed grid `(H,W,D,3)` |
| **DivisiblePad** | `k`, `method`, `mode` | — | — | — |

**Key**: "—" means nothing is sampled at that level.

## Detailed Notes

### Channel-wise behavior

No random transform samples different *parameter values* per channel. All channels within a batch element receive the same augmentation parameters. The one exception is `ScaleIntensity` with `channel_wise=True`, which computes min/max statistics per channel (but this is data-dependent, not randomly sampled).

However, some transforms produce different *values* per channel despite sharing parameters:

- **RandGaussianNoise**: mean and std are shared across channels, but the noise tensor itself is `(B, C, H, W, D)` with independent random draws per channel and voxel.
- **All geometric transforms** (flip, rotate90, affine, elastic): the transformation is identical for all channels — every channel is warped/flipped in exactly the same way.

### Dict key sharing

For `BatchDictTransform` wrappers (the `d`-suffixed classes), `sample_params()` is called **once** and the same parameters are applied to every key. This ensures paired data (e.g. vol + seg) receives identical geometric transforms and identical noise/bias patterns.

Geometric dict transforms (`RandAffined`, `Rand3DElasticd`) allow **per-key interpolation modes** — e.g. `mode={"vol": "bilinear", "seg": "nearest"}` — while still sharing the same spatial transformation.

### Pre-generated tensors

Several transforms pre-generate large tensors in `sample_params()` so they are shared across dict keys:

| Transform | What's pre-generated | Why |
|-----------|---------------------|-----|
| RandGaussianNoise | noise tensor `(B,C,H,W,D)` | Same noise added to all keys |
| RandGaussianSmooth | 1D kernels per axis `(B,K)` | Same blur applied to all keys |
| RandGaussianSharpen | two sets of 1D kernels + alpha | Same sharpening for all keys |
| RandBiasField | Legendre basis `(n_coeff,H,W,D)` + coefficients | Same bias field for all keys |
| RandGibbsNoise | k-space mask `(B,1,H,W,D)` | Same truncation for all keys |
| Rand3DElastic | deformed grid `(B,H,W,D,3)` | Same deformation for all keys |

### Probability mask

Every `BatchTransform` samples a boolean mask `(B,)` where each element is independently drawn from Bernoulli(`prob`). Elements where `mask=False` are left unchanged via `torch.where`.

### DivisiblePad

`DivisiblePad` is **not** a `BatchTransform` — it is deterministic, has no probability mask, and applies uniformly to all elements. It only pads spatial dimensions to the nearest multiple of `k`.
