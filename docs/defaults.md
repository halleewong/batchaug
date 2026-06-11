## Recommended Defaults vs MONAI Parity

BatchAug's API mirrors MONAI, and for almost every transform the default arguments match MONAI's defaults exactly. For the few transforms listed below, we believe MONAI's defaults have a subtle bug or undesirable behavior, so **BatchAug defaults to our recommended fix instead**. Each is a single argument away from exact MONAI parity:

| Transform | Argument | BatchAug default | Pass this for MONAI parity |
|---|---|---|---|
| `RandSimulateLowResolution` | `downsample_mode` | `"nearest-exact"` | `"nearest"` |
| `RandGaussianSmooth` | `padding_mode` | `"reflect"` | `None` |
| `RandGaussianSharpen` | `padding_mode` | `"reflect"` | `None` |

### `RandSimulateLowResolution` — spatial shift from convention mismatch

MONAI uses `downsample_mode="nearest"`, `upsample_mode="trilinear"`, `align_corners=False`, which together produce a spatial shift of roughly +½ pixel toward +H/+W/+D on every down/up roundtrip. This is because PyTorch's `mode="nearest"` uses a corner-aligned indexing convention while `trilinear` with `align_corners=False` uses a cell-centered one, so the two steps don't compose cleanly. **BatchAug defaults to `downsample_mode="nearest-exact"`** to fix this; pass `downsample_mode="nearest"` for exact MONAI parity.

`downsample_mode` and `upsample_mode` are independent and accept any [`F.interpolate`](https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html) mode (`"nearest"`, `"nearest-exact"`, `"trilinear"`, `"area"`, …). `align_corners` applies automatically to whichever step uses a mode that supports it. Measured center-of-mass shift of a 3D Gaussian on a 64³ grid (input centered at index 32.0):

| `downsample_mode` ↓ / `upsample_mode` ↑ | zoom=0.5 | zoom=0.6 | zoom=0.7 | zoom=0.8 | zoom=0.9 |
|---|---|---|---|---|---|
| `"nearest"` / `"trilinear"` (MONAI default) | 32.500 | 32.851 | 32.721 | 32.642 | 32.561 |
| `"nearest-exact"` / `"trilinear"` | 31.500 | 31.995 | 32.001 | 32.000 | 31.982 |
| `"trilinear"` / `"trilinear"` | 32.000 | 32.000 | 32.000 | 32.000 | 31.998 |
| `"area"` / `"trilinear"` | 32.000 | 32.004 | 31.996 | 32.002 | 32.008 |
| `"trilinear"` / `"nearest"` | 32.000 | 32.497 | 32.504 | 32.498 | 32.491 |
| `"trilinear"` / `"nearest-exact"` | 32.000 | 31.996 | 31.999 | 31.998 | 32.005 |
| `"nearest"` / `"nearest"` | 32.500 | 33.398 | 33.303 | 33.200 | 33.077 |

Options (BatchAug defaults to the first):

- **`downsample_mode="nearest-exact"` (BatchAug default)** — also cell-centered, so the conventions match. Brings the shift to <0.02 px for zoom 0.6–0.9. Worst case is exactly zoom=0.5, where nearest-style quantization still leaves a -0.5 px residual.
- **`downsample_mode="trilinear"`** (or `"area"`) — proper cell-centered resample, ~0 shift at all zoom factors. Smoother result though, so it no longer behaves like a true "nearest" downsample.
- **`downsample_mode="trilinear"`, `upsample_mode="nearest"`** — anti-aliased downsample with blocky pixel-replication upsample (closest visual analog to a real low-res acquisition shown on the original grid). Still ~+0.5 px shift at non-integer zooms because the nearest upsample uses corner convention.
- **`downsample_mode="trilinear"`, `upsample_mode="nearest-exact"`** — same blocky look as the previous combo but with cell-centered upsample, so the shift drops to <0.01 px at all zooms. Best of both worlds if you want zero shift AND the low-res-on-original-grid aesthetic.

### `RandGaussianSmooth` / `RandGaussianSharpen` — dark edge halo from zero-padding

MONAI's `GaussianSmooth` / `GaussianSharpen` use `F.conv3d` with implicit zero-padding, which treats every voxel beyond the volume boundary as 0. For voxels within `kernel_size // 2` of an edge, the convolution averages in those zeros and produces a dark halo (and on `RandGaussianSharpen`, an inverted bright halo from the unsharp-mask subtraction). The artifact is most visible on volumes that are tightly cropped around foreground anatomy, where the boundary intensity is far from zero.

The `padding_mode` argument is forwarded to `F.pad` before each separable 1D convolution. Accepted values: `"reflect"`, `"replicate"`, `"constant"` (explicit zero), or `None` (implicit zero via `conv3d`). **BatchAug defaults to `padding_mode="reflect"`**; pass `padding_mode=None` for exact MONAI parity.

Options (BatchAug defaults to the first):

- **`padding_mode="reflect"` (BatchAug default)** — mirrors the volume across each boundary. Removes the halo cleanly; safe default for medical images where the boundary tissue continues smoothly into the (unobserved) neighborhood.
- **`padding_mode="replicate"`** — repeats the edge voxel. Also removes the halo, but introduces a thin streak along the boundary if the edge voxel happens to be an outlier. Prefer `"reflect"` unless you have a reason to avoid mirroring (e.g. directional content right at the boundary).

Caveats:

- Only the PyTorch backend honors `padding_mode`. The Triton kernels (`batchaug.triton.RandGaussianSmooth`, `RandGaussianSharpen`) implement zero-padding only: they accept `padding_mode=None` / `"constant"` (default `None`) and raise `NotImplementedError` for `"reflect"` / `"replicate"`. This is academic in practice since the Triton smooth/sharpen kernels are slower than cuDNN and are not exported by default — `batchaug.RandGaussianSmooth` resolves to the PyTorch version (which defaults to `"reflect"`) regardless of backend.
- `RandGaussianSharpen` applies the padding to both blur passes, so the recommendation carries over directly.
