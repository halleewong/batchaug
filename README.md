# BatchAug

Batched GPU augmentations for 3D medical imaging data. The API mirrors [MONAI](https://github.com/Project-MONAI/MONAI) but performs augmentations over an entire batch at once, sampling independent random parameters per batch element (similar to [Kornia](https://github.com/kornia/kornia) for 2D). Like MONAI, dictionary transforms apply the same augmentation to multiple volumes and segmentations, keeping paired data aligned. The package provides both PyTorch and Triton backends and automatically selects the fastest one for each transform.

- **MONAI-compatible API** — drop-in replacements with matching output when B=1
- **Independent augmentation across batch** (B dimension) — each batch element samples its own random parameters
- **Same augmentation across channels** (C dimension) — all paired volumes get the same transform
- **GPU-native** — all operations stay on CUDA, no CPU roundtrips
- **Fast** — 2–1000x faster than MONAI per-sample loops depending on the transform ([benchmarks](#benchmarks))
- **Auto backend selection** — Triton fused kernels where faster, PyTorch/cuDNN elsewhere
- **dtype support** — works with both `float32` and `bfloat16`

## Comparison with Other Libraries

| Library | 2D | 3D | Batched | GPU |
|---------|----|----|---------|-----|
| [torchvision](https://github.com/pytorch/vision) | ✅ | ❌ | ❌ | ○ |
| [kornia](https://github.com/kornia/kornia) | ✅ | ○ | ✅ | ✅ |
| [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) | ✅ | ✅ | ❌ | ❌ |
| [torchio](https://github.com/TorchIO-project/torchio) | ❌ | ✅ | ❌ | ❌ |
| [monai](https://github.com/Project-MONAI/MONAI) | ✅ | ✅ | ❌ | ✅ |
| **batchaug** | ❌ | ✅ | ✅ | ✅ |

○ = partial support

## Installation

Requires PyTorch with CUDA. Install PyTorch first following [pytorch.org](https://pytorch.org/get-started/locally/), then:

```
pip install batchaug
```

For development:

```
git clone https://github.com/halleewong/batchaug.git
cd batchaug
pip install -e ".[test]"
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
        batchaug.RandRotate90d(keys=["vol", "seg"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
        batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=0.5),
        batchaug.RandAffined(keys=["vol", "seg"], prob=0.5,
                             rotate_range=0.785, shear_range=0.3, translate_range=5),
        batchaug.RandGaussianNoised(keys=["vol"], prob=0.5, mean=0.0, std=0.5),
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
monai_aug = monai.transforms.Compose([...])  # same transforms, MONAI API

augmented = {"vol": [], "seg": []}
for i in range(B):
    sample = {"vol": batch["vol"][i], "seg": batch["seg"][i]}
    out = monai_aug(sample)
    augmented["vol"].append(out["vol"])
    augmented["seg"].append(out["seg"])
augmented = {k: torch.stack(v) for k, v in augmented.items()}
```

The same parameters are used for each channel within a batch element. To perform data augmentation independently across channels, simply reshape the data to merge the B and C dimensions, apply the augmentation, then reshape back:

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

### Fused Pipeline

`FusedAugment` and `FusedAugmentd` are an alternative to `Compose` that fuse multiple transforms into fewer kernel calls. The pipeline has a fixed order:

- **Phase 1 — Geometric**: flip + rotate90 + affine composed into a single `grid_sample`
- **Phase 2 — Elastic**: separate `grid_sample` (non-linear, cannot be fused with affine)
- **Phase 3 — Spatial intensity**: smooth, sharpen, low-res simulation, Gibbs noise (separate cuDNN/cuFFT kernels)
- **Phase 4 — Elementwise**: scale + contrast + bias field + Gaussian noise fused into a single Triton kernel

```python
from batchaug.triton import FusedAugmentd

aug = FusedAugmentd(
    keys=["vol", "seg"],
    flip_prob=0.5,
    rotate90_prob=0.5, max_k=3, spatial_axes=(0, 1),
    affine_prob=0.5, rotate_range=0.785, shear_range=0.3, translate_range=5,
    noise_prob=0.5, noise_std=(0.0, 0.5),
    bias_field_prob=0.5,
    scale_intensity=True,
    mode={"vol": "bilinear", "seg": "nearest"},
)

augmented_batch = aug(batch)
```

Performance is mixed and depends on volume size, batch size, and number of channels. On an NVIDIA L40S with a 12-transform pipeline, prob=0.5 (median of 10 runs):

| Size | C | B | Eager (ms) | Lazy (ms) | Fused (ms) | Fused vs Lazy |
|------|---|---|-----------|----------|-----------|--------------|
| 64^3 | 1 | 1 | 11.7 | 11.5 | 63.7 | 0.2x |
| 64^3 | 1 | 2 | 12.6 | 11.0 | 9.5 | **1.2x** |
| 64^3 | 1 | 5 | 12.9 | 13.6 | 79.4 | 0.2x |
| 64^3 | 4 | 1 | 5.5 | 5.4 | 9.5 | 0.6x |
| 64^3 | 4 | 2 | 54.3 | 10.4 | 9.4 | **1.1x** |
| 64^3 | 4 | 5 | 14.4 | 13.2 | 11.8 | **1.1x** |
| 128^3 | 1 | 1 | 13.1 | 13.7 | 14.5 | 0.9x |
| 128^3 | 1 | 2 | 17.8 | 17.2 | 13.2 | **1.3x** |
| 128^3 | 1 | 5 | 36.3 | 45.5 | 30.9 | **1.5x** |
| 128^3 | 4 | 1 | 18.0 | 16.4 | 15.1 | **1.1x** |
| 128^3 | 4 | 2 | 27.6 | 30.7 | 25.8 | **1.2x** |
| 128^3 | 4 | 5 | 75.0 | 67.3 | 89.2 | 0.8x |

`FusedAugment` tends to help at **larger volumes (128^3+)** where the elementwise Triton kernel amortizes its fixed launch overhead. At smaller volumes (64^3), the unfused spatial passes (Gibbs noise, low-res simulation, elastic) can dominate and make it slower. The gains are also less consistent at large batch sizes (B=5, C=4) where memory bandwidth becomes the bottleneck regardless of fusion.

`Compose` with `lazy=True` is simpler and performs well in most cases. Use `FusedAugment` as an option to try when working at larger spatial scales. To evaluate on your specific workload, run the benchmark script:

```
python examples/benchmark_fused.py --batch_sizes 1 2 5 --spatial_sizes 64 128
```

### Benchmarks

Per-transform speedup over MONAI's per-sample loop, using the default auto backend (Triton where faster, PyTorch elsewhere). Measured on NVIDIA L40S with B=5, C=4, 128^3:

| Transform | MONAI (ms) | BatchAug (ms) | Speedup |
|-----------|-----------|--------------|---------|
| RandBiasField | 660.5 | 0.6 | **1118x** |
| RandGaussianNoise | 744.0 | 2.6 | **282x** |
| Rand3DElastic | 223.2 | 9.9 | **23x** |
| RandAffine | 113.5 | 11.5 | **10x** |
| RandGibbsNoise | 49.0 | 12.4 | **4x** |
| ScaleIntensity | 2.6 | 0.9 | **3x** |
| RandGaussianSmooth | 5.8 | 3.6 | **2x** |

The Triton backend provides additional speedups over the PyTorch backend for select transforms:

| Transform | Triton vs PyTorch | Why |
|-----------|------------------|-----|
| RandBiasField | 3–10x | On-the-fly Legendre eval avoids large basis tensor |
| ScaleIntensity | 1.5–4.4x | Fused min/max reduction + rescale |
| RandAdjustContrast | 1.2–3.0x | Fused normalize + pow + denormalize |

### Available Transforms

**Composition**

| Transform | Description |
|-----------|-------------|
| `Compose` | Sequential pipeline with optional lazy geometric fusion |
| `FusedAugment` / `FusedAugmentd` | Fixed-order pipeline with additional kernel fusion (Triton backend) |

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
