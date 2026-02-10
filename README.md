# BatchAug

Batched GPU augmentations for paired 3D medical imaging data (volumes + segmentations). Replaces per-sample MONAI augmentation loops with batched operations that run entirely on the GPU.

- **Same augmentation across channels** (C dimension) — all paired slices get the same transform
- **Independent augmentation across batch** (B dimension) — each batch element samples its own random parameters
- **GPU-native** — all operations stay on CUDA, no CPU roundtrips
- **MONAI-compatible** — when B=1, results match MONAI's output
- **dtype support** — works with both `float32` and `bfloat16`

## Comparison with Other Libraries

| Library | 2D | 3D | Batched | GPU |
|---------|----|----|---------|-----|
| torchvision | ✓ | ✗ | ✗ | ✗ |
| kornia | ✓ | ✗ | ✓ | ✓ |
| batchgenerators | ✓ | ✓ | ✗ | ✗ |
| monai | ✓ | ✓ | ✗ | ✓ |
| **batchaug** | **✗** | **✓** | **✓** | **✓** |

## Installation

```
cd batchaug
pip install -e .
```

## Usage

All inputs have shape `(B, C, H, W, D)` where B is the batch size and C is the number of channels (e.g. query/support slices in a few-shot task). The same augmentation parameters are applied to all channels within a batch element, but different parameters are sampled for each batch element.

### Task Augmentation

Apply the same augmentation to paired volumes and segmentations using dictionary transforms. Parameters are sampled once and applied to all keys, so paired data stays aligned.

```python
import torch
import batchaug

# Paired volume and segmentation on GPU
batch = {
    "vol": torch.randn(5, 4, 128, 128, 128, device="cuda"),
    "seg": torch.randn(5, 4, 128, 128, 128, device="cuda"),
}

# Compose a pipeline (planned API)
task_augs = batchaug.Compose(
    lazy=True,
    transforms=[
        batchaug.RandRotate90d(keys=["vol", "seg"], prob=0.15, max_k=3, spatial_axes=(0, 1)),
        batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=0.15),
        batchaug.RandGaussianNoised(keys=["vol"], prob=0.15, mean=0.0, std=0.5),
        batchaug.RandAffined(keys=["vol", "seg"], prob=0.15, mode=["bilinear", "nearest"],
                             rotate_range=0.785, shear_range=0.3, translate_range=5),
        batchaug.ScaleIntensityd(keys=["vol"]),
    ]
)

augmented_batch = task_augs(batch)
```

With `lazy=True`, geometric transforms (rotations, flips, affines) are fused into a single `grid_sample` call, avoiding redundant interpolation. Intensity transforms are applied eagerly at their position in the pipeline.

This replaces the slow per-sample loop:

```python
# Before (slow, sequential)
for i in range(bs):
    sample = {"vol": x_tensors[i], "seg": y_tensors[i]}
    aug_sample = train_gpu_task_augmentations(sample)
    augmented_vols.append(aug_sample["vol"])
    augmented_segs.append(aug_sample["seg"])
x_tensors = torch.stack(augmented_vols, dim=0)
y_tensors = torch.stack(augmented_segs, dim=0)
```

### Data Augmentation

Apply transforms directly to tensors (without dictionary wrapping) when you don't need paired augmentation across keys.

```python
import torch
import batchaug

vol = torch.randn(8, 1, 64, 64, 64, device="cuda")

# Individual transforms
t = batchaug.RandGaussianNoise(prob=0.5, mean=0.0, std=(0.1, 0.5))
noisy_vol = t(vol)

# Or use sample_params / apply for full control
params = t.sample_params(vol.shape[0], vol.shape, vol.device)
noisy_vol = t.apply(vol, params)
```

### Available Transforms

| Transform | Dict version | Description |
|-----------|-------------|-------------|
| `ScaleIntensity` | `ScaleIntensityd` | Rescale intensity to [minv, maxv], per element or per element x channel |
| `RandAxisFlip` | `RandAxisFlipd` | Random flip along a spatial axis |
| `RandRotate90` | `RandRotate90d` | Random 90-degree rotation |
| `RandGaussianNoise` | `RandGaussianNoised` | Additive Gaussian noise with per-element mean/std |

### Planned Transforms

- `RandAdjustContrast` / `RandAdjustContrastd` — Gamma correction
- `RandGaussianSmooth` / `RandGaussianSmoothd` — Separable Gaussian blur
- `RandGaussianSharpen` / `RandGaussianSharpend` — Unsharp masking
- `RandSimulateLowResolution` / `RandSimulateLowResolutiond` — Downsample/upsample
- `RandBiasField` / `RandBiasFieldd` — Polynomial bias field
- `RandGibbsNoise` / `RandGibbsNoised` — FFT-based Gibbs ringing
- `RandAffine` / `RandAffined` — Batched affine with per-key interpolation modes
- `Compose` — Sequential + lazy geometric fusion