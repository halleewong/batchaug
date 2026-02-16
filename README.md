# BatchAug

Batched GPU augmentations for 3D medical imaging. The API mirrors [MONAI](https://github.com/Project-MONAI/MONAI) but performs augmentations over an entire batch at once, sampling independent random parameters per batch element (similar to [Kornia](https://github.com/kornia/kornia) for 2D). Like MONAI, dictionary transforms apply the same augmentation to multiple volumes and segmentations, keeping paired data aligned. The package provides both PyTorch and Triton backends and automatically selects the fastest one for each transform.

- **MONAI-compatible API** — drop-in replacements with matching output when B=1
- **Independent augmentation across batch** (B dimension) — each batch element samples its own random parameters
- **Same augmentation across channels** (C dimension) — all paired slices get the same transform
- **GPU-native** — all operations stay on CUDA, no CPU roundtrips
- **Auto backend selection** — Triton fused kernels where faster, PyTorch/cuDNN elsewhere
- **dtype support** — works with both `float32` and `bfloat16`

## Comparison with Other Libraries

| Library | 2D | 3D | Batched | GPU |
|---------|----|----|---------|-----|
| [torchvision](https://github.com/pytorch/vision) | ✅ | ❌ | ❌ | ❌ |
| [kornia](https://github.com/kornia/kornia) | ✅ | ❌ | ✅ | ✅ |
| [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) | ✅ | ✅ | ❌ | ❌ |
| [torchio](https://github.com/TorchIO-project/torchio) | ❌ | ✅ | ❌ | ❌ |
| [monai](https://github.com/Project-MONAI/MONAI) | ✅ | ✅ | ❌ | ✅ |
| **batchaug** | ❌ | ✅ | ✅ | ✅ |

## Installation

```
cd batchaug
pip install -e .
```

## Usage

All inputs have shape `(B, C, H, W, D)` where B is the batch size and C is the number of channels (e.g. query/support slices in a few-shot task). The same augmentation parameters are applied to all channels within a batch element, but different parameters are sampled for each batch element.

### Task Augmentation

Sometimes it is useful to apply the same augmentations across a set of volumes and segmentations from the same dataset — i.e., *task augmentations* as in [UniverSeg](https://arxiv.org/abs/2304.06131), [Tyche](https://arxiv.org/abs/2401.13650), and [MultiverSeg](https://arxiv.org/abs/2412.15058). In this setting, the channel dimension stores different examples from the same task. 

`BatchAug` samples parameters independently for each entry in the batch, then applies the same transformation to every entry along the channel dimension. Dictionary transforms apply the same augmentation to all keys, so paired data (e.g. vol + seg) stays aligned. For more details on parameter sampling, see [docs/sampling.md](docs/sampling.md).

```python
import torch
import batchaug

# Paired volume and segmentation on GPU
batch = {
    "vol": torch.randn(5, 4, 128, 128, 128, device="cuda"),
    "seg": torch.randn(5, 4, 128, 128, 128, device="cuda"),
}

# Compose a pipeline
task_augs = batchaug.Compose(
    transforms=[
        batchaug.RandRotate90d(keys=["vol", "seg"], prob=0.15, max_k=3, spatial_axes=(0, 1)),
        batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=0.15),
        batchaug.RandGaussianNoised(keys=["vol"], prob=0.15, mean=0.0, std=0.5),
        batchaug.RandAffined(keys=["vol", "seg"], prob=0.15,
                             rotate_range=0.785, shear_range=0.3, translate_range=5),
        batchaug.ScaleIntensityd(keys=["vol"]),
    ],
    lazy=True,
    mode={"vol": "bilinear", "seg": "nearest"},
)

augmented_batch = task_augs(batch)
```

With `lazy=True`, geometric transforms (rotations, flips, affines) are fused into a single `grid_sample` call, avoiding redundant interpolation. Intensity transforms are applied eagerly at their position in the pipeline.

This replaces the slow per-sample loop required by MONAI:

```python
# Before: MONAI per-sample loop (slow, sequential)
monai_aug = monai.transforms.Compose([...])  # single-sample MONAI pipeline
augmented_vols, augmented_segs = [], []
for i in range(B):
    sample = {"vol": vol[i], "seg": seg[i]}
    out = monai_aug(sample)
    augmented_vols.append(out["vol"])
    augmented_segs.append(out["seg"])
vol = torch.stack(augmented_vols)
seg = torch.stack(augmented_segs)
```

The same parameters are used for each channel within a batch element. To perform data augmentation independently accross channels, simply reshape the data to merge the B and C dimensions, apply the augmentation, then reshape back:

```python
# For independent channel augmentation, merge B and C dims
B, C = vol.shape[:2]
vol = vol.view(B * C, 1, *vol.shape[2:])  # (B*C, 1, H, W, D)
aug_vol = t(vol)
aug_vol = aug_vol.view(B, C, *aug_vol.shape[2:])
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

### Backends: PyTorch and Triton

BatchAug includes two backends: a pure **PyTorch** backend and a **Triton** backend with custom fused kernels. When Triton is installed, the library auto-selects the fastest implementation for each transform:

| Transform | Default backend | Why |
|-----------|----------------|-----|
| `ScaleIntensity` | Triton | Fused min/max reduction + rescale (1.5–4.4x faster) |
| `RandAdjustContrast` | Triton | Fused normalize + pow + denormalize (1.2–3.0x faster) |
| `RandBiasField` | Triton | On-the-fly Legendre polynomial eval avoids large basis tensor (3–10x faster) |
| `RandGaussianSmooth` | PyTorch | cuDNN's conv3d is faster than custom Triton separable conv |
| `RandGaussianSharpen` | PyTorch | Uses smooth internally, same cuDNN advantage |
| All others | PyTorch | Already use optimized CUDA ops (cuFFT, grid_sample, etc.) |

This happens transparently — `batchaug.ScaleIntensity` resolves to the Triton version, while `batchaug.RandGaussianSmooth` resolves to the PyTorch version. You can override:

```python
import batchaug

# Force a specific backend
batchaug.set_backend("pytorch")  # always use PyTorch
batchaug.set_backend("triton")   # always use Triton
batchaug.set_backend("auto")     # auto-select (default)

# Or import from a specific backend directly
from batchaug.pytorch import ScaleIntensity   # PyTorch version
from batchaug.triton import ScaleIntensity    # Triton version
```

### Available Transforms

**Composition**

| Transform | Description |
|-----------|-------------|
| `Compose` | Sequential pipeline with optional lazy geometric fusion |

**Geometric**

| Transform | Dict version | Description |
|-----------|-------------|-------------|
| `RandAxisFlip` | `RandAxisFlipd` | Random flip along a spatial axis |
| `RandRotate90` | `RandRotate90d` | Random 90-degree rotation |
| `RandAffine` | `RandAffined` | Random affine (rotate, shear, translate, scale) with per-key interpolation modes |
| `Rand3DElastic` | `Rand3DElasticd` | Random elastic deformation via smoothed displacement fields |

**Intensity**

| Transform | Dict version | Description |
|-----------|-------------|-------------|
| `ScaleIntensity` | `ScaleIntensityd` | Rescale intensity to [minv, maxv], per element or per element x channel |
| `RandGaussianNoise` | `RandGaussianNoised` | Additive Gaussian noise with per-element mean/std |
| `RandAdjustContrast` | `RandAdjustContrastd` | Gamma correction |
| `RandGaussianSmooth` | `RandGaussianSmoothd` | Separable Gaussian blur |
| `RandGaussianSharpen` | `RandGaussianSharpend` | Unsharp masking |
| `RandSimulateLowResolution` | `RandSimulateLowResolutiond` | Downsample/upsample simulation |
| `RandBiasField` | `RandBiasFieldd` | Polynomial bias field |
| `RandGibbsNoise` | `RandGibbsNoised` | FFT-based Gibbs ringing |

**Utility**

| Transform | Dict version | Description |
|-----------|-------------|-------------|
| `DivisiblePad` | `DivisiblePadd` | Pad spatial dims to be divisible by k |


# Development

## Run Tests

```
python -m pytest tests/ -v
```
