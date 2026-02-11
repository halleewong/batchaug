"""Benchmark full pipeline: MONAI vs PyTorch-only batchaug vs best-of-both (Triton+PyTorch).

The "best" pipeline uses Triton kernels for ScaleIntensity, RandAdjustContrast,
and RandBiasField (where Triton is faster), and PyTorch/cuDNN for everything
else (smooth, sharpen, geometric, noise, gibbs, resolution).

This is what `batchaug.*` auto-dispatches to when Triton is installed.

Usage:
    conda run -n interseg3d python examples/benchmark_triton_pipeline.py 2>&1 | tee examples/benchmark_triton_pipeline.log
    conda run -n interseg3d python examples/benchmark_triton_pipeline.py --batch_sizes 1 2 4 8 --spatial_sizes 64 128 256
"""

import argparse
import sys

import torch
import monai.transforms

import batchaug
from batchaug import pytorch as ba_pytorch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cuda_timer(fn, warmup=3, repeats=10):
    """Time a function using CUDA events. Returns median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def measure_peak_mb(fn):
    """Run fn and return peak GPU memory above baseline in MB."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / (1024 * 1024)


def try_allocate(B, C, S):
    """Check if we can allocate vol + seg without OOM."""
    nelems = B * C * S * S * S
    bytes_needed = nelems * 4 * 3  # vol + seg + headroom
    free, _ = torch.cuda.mem_get_info()
    return bytes_needed < free * 0.8


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def make_monai_pipeline():
    """Build the equivalent MONAI pipeline."""
    return monai.transforms.Compose([
        monai.transforms.RandRotate90d(
            keys=["vol", "seg"], prob=0.15, max_k=3, spatial_axes=(0, 1),
        ),
        monai.transforms.RandAxisFlipd(keys=["vol", "seg"], prob=0.15),
        monai.transforms.RandAffined(
            keys=["vol", "seg"], prob=0.15,
            rotate_range=0.7853981633974483, shear_range=0.3, translate_range=5,
            mode=("bilinear", "nearest"), padding_mode="zeros",
        ),
        monai.transforms.RandSimulateLowResolutiond(
            keys=["vol"], prob=0.15, zoom_range=(0.33, 1.0),
        ),
        monai.transforms.RandGaussianNoised(
            keys=["vol"], prob=0.15, mean=0.0, std=0.5,
        ),
        monai.transforms.RandBiasFieldd(
            keys=["vol"], prob=0.15, coeff_range=(0.0, 0.05),
        ),
        monai.transforms.RandGibbsNoised(
            keys=["vol"], prob=0.15, alpha=(0.0, 0.33),
        ),
        monai.transforms.RandAdjustContrastd(
            keys=["vol"], prob=0.15, gamma=(0.5, 2.5),
        ),
        monai.transforms.RandGaussianSmoothd(
            keys=["vol"], prob=0.15,
            sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
        ),
        monai.transforms.RandGaussianSharpend(
            keys=["vol"], prob=0.15,
        ),
        monai.transforms.ScaleIntensityd(keys=["vol"]),
    ])


def _build_batchaug_transforms(mod):
    """Build the standard transform list using classes from the given module."""
    return [
        mod.RandRotate90d(
            keys=["vol", "seg"], prob=0.15, max_k=3, spatial_axes=(0, 1),
        ),
        mod.RandAxisFlipd(keys=["vol", "seg"], prob=0.15),
        mod.RandAffined(
            keys=["vol", "seg"], prob=0.15,
            rotate_range=0.7853981633974483, shear_range=0.3, translate_range=5,
        ),
        mod.RandSimulateLowResolutiond(
            keys=["vol"], prob=0.15, zoom_range=(0.33, 1.0),
        ),
        mod.RandGaussianNoised(keys=["vol"], prob=0.15, mean=0.0, std=0.5),
        mod.RandBiasFieldd(
            keys=["vol"], prob=0.15, coeff_range=(0.0, 0.05),
        ),
        mod.RandGibbsNoised(
            keys=["vol"], prob=0.15, alpha=(0.0, 0.33),
        ),
        mod.RandAdjustContrastd(
            keys=["vol"], prob=0.15, gamma=(0.5, 2.5),
        ),
        mod.RandGaussianSmoothd(
            keys=["vol"], prob=0.15,
            sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
        ),
        mod.RandGaussianSharpend(keys=["vol"], prob=0.15),
        mod.ScaleIntensityd(keys=["vol"]),
    ]


def make_pytorch_pipeline(lazy=False):
    """Build pipeline using only PyTorch backend."""
    transforms = _build_batchaug_transforms(ba_pytorch)
    if lazy:
        return batchaug.Compose(
            transforms, lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )
    return batchaug.Compose(transforms, lazy=False)


def make_best_pipeline(lazy=False):
    """Build pipeline using auto-dispatched backend (Triton where faster)."""
    transforms = _build_batchaug_transforms(batchaug)
    if lazy:
        return batchaug.Compose(
            transforms, lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )
    return batchaug.Compose(transforms, lazy=False)


def monai_per_sample_loop(monai_pipeline, batch, keys):
    """Run MONAI pipeline in a per-sample loop."""
    B = batch[keys[0]].shape[0]
    results = {k: [] for k in keys}
    for i in range(B):
        sample = {k: batch[k][i] for k in keys}
        out = monai_pipeline(sample)
        for k in keys:
            results[k].append(out[k])
    return {k: torch.stack(v, dim=0) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_task(B, C, S, repeats):
    """Task aug: same params across C.
    Returns (monai_ms, pt_eager, best_eager, pt_lazy, best_lazy)."""
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    keys = ["vol", "seg"]

    monai_pipe = make_monai_pipeline()
    pt_eager = make_pytorch_pipeline(lazy=False)
    best_eager = make_best_pipeline(lazy=False)
    pt_lazy = make_pytorch_pipeline(lazy=True)
    best_lazy = make_best_pipeline(lazy=True)

    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_pipe, batch, keys), repeats=repeats,
    )
    pt_eager_ms = cuda_timer(lambda: pt_eager(batch), repeats=repeats)
    best_eager_ms = cuda_timer(lambda: best_eager(batch), repeats=repeats)
    pt_lazy_ms = cuda_timer(lambda: pt_lazy(batch), repeats=repeats)
    best_lazy_ms = cuda_timer(lambda: best_lazy(batch), repeats=repeats)

    return monai_ms, pt_eager_ms, best_eager_ms, pt_lazy_ms, best_lazy_ms


def bench_task_memory(B, C, S):
    """Task aug peak memory.
    Returns (monai_mb, pt_eager, best_eager, pt_lazy, best_lazy)."""
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    keys = ["vol", "seg"]

    monai_pipe = make_monai_pipeline()
    pt_eager = make_pytorch_pipeline(lazy=False)
    best_eager = make_best_pipeline(lazy=False)
    pt_lazy = make_pytorch_pipeline(lazy=True)
    best_lazy = make_best_pipeline(lazy=True)

    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_pipe, batch, keys),
    )
    pt_eager_mb = measure_peak_mb(lambda: pt_eager(batch))
    best_eager_mb = measure_peak_mb(lambda: best_eager(batch))
    pt_lazy_mb = measure_peak_mb(lambda: pt_lazy(batch))
    best_lazy_mb = measure_peak_mb(lambda: best_lazy(batch))

    return monai_mb, pt_eager_mb, best_eager_mb, pt_lazy_mb, best_lazy_mb


# -- Data Augmentation: independent transform per channel -------------------

def _data_aug(pipe, vol, seg):
    """Reshape to (B*C, 1, ...), run pipeline, reshape back."""
    B, C = vol.shape[:2]
    spatial = vol.shape[2:]
    vol_flat = vol.reshape(B * C, 1, *spatial)
    seg_flat = seg.reshape(B * C, 1, *spatial)
    out = pipe({"vol": vol_flat, "seg": seg_flat})
    out["vol"] = out["vol"].reshape(B, C, *spatial)
    out["seg"] = out["seg"].reshape(B, C, *spatial)
    return out


def bench_data(B, C, S, repeats):
    """Data aug: independent params per channel.
    Returns (monai_ms, pt_eager, best_eager, pt_lazy, best_lazy)."""
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    keys = ["vol", "seg"]

    monai_pipe = make_monai_pipeline()
    flat_batch = {
        "vol": vol.reshape(B * C, 1, S, S, S),
        "seg": seg.reshape(B * C, 1, S, S, S),
    }
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_pipe, flat_batch, keys), repeats=repeats,
    )

    pt_eager = make_pytorch_pipeline(lazy=False)
    best_eager = make_best_pipeline(lazy=False)
    pt_lazy = make_pytorch_pipeline(lazy=True)
    best_lazy = make_best_pipeline(lazy=True)

    pt_eager_ms = cuda_timer(lambda: _data_aug(pt_eager, vol, seg), repeats=repeats)
    best_eager_ms = cuda_timer(lambda: _data_aug(best_eager, vol, seg), repeats=repeats)
    pt_lazy_ms = cuda_timer(lambda: _data_aug(pt_lazy, vol, seg), repeats=repeats)
    best_lazy_ms = cuda_timer(lambda: _data_aug(best_lazy, vol, seg), repeats=repeats)

    return monai_ms, pt_eager_ms, best_eager_ms, pt_lazy_ms, best_lazy_ms


def bench_data_memory(B, C, S):
    """Data aug peak memory.
    Returns (monai_mb, pt_eager, best_eager, pt_lazy, best_lazy)."""
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    keys = ["vol", "seg"]

    monai_pipe = make_monai_pipeline()
    flat_batch = {
        "vol": vol.reshape(B * C, 1, S, S, S),
        "seg": seg.reshape(B * C, 1, S, S, S),
    }
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_pipe, flat_batch, keys),
    )

    pt_eager = make_pytorch_pipeline(lazy=False)
    best_eager = make_best_pipeline(lazy=False)
    pt_lazy = make_pytorch_pipeline(lazy=True)
    best_lazy = make_best_pipeline(lazy=True)

    pt_eager_mb = measure_peak_mb(lambda: _data_aug(pt_eager, vol, seg))
    best_eager_mb = measure_peak_mb(lambda: _data_aug(best_eager, vol, seg))
    pt_lazy_mb = measure_peak_mb(lambda: _data_aug(pt_lazy, vol, seg))
    best_lazy_mb = measure_peak_mb(lambda: _data_aug(best_lazy, vol, seg))

    return monai_mb, pt_eager_mb, best_eager_mb, pt_lazy_mb, best_lazy_mb


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def fmt_speedup(base_ms, other_ms):
    if other_ms <= 0:
        return "  inf"
    s = base_ms / other_ms
    return f"{s:5.1f}x"


PIPELINE_DESC = "Rot90 → Flip → Affine → LowRes → Noise → Bias → Gibbs → Contrast → Smooth → Sharpen → Scale"
TRITON_NOTE = "best = Triton(Scale,Contrast,Bias) + PyTorch/cuDNN(all others)"


def print_time_table(title, bench_fn, batch_sizes, spatial_sizes, channels_list, repeats):
    """Print timing table."""
    W = 145
    print("\n" + "=" * W)
    print(f"  {title} — TIMING (ms, median)")
    print(f"  Pipeline: {PIPELINE_DESC}")
    print(f"  {TRITON_NOTE}")
    print("=" * W)

    header = (f"{'S':>4}  {'C':>3}  {'B':>3}  "
              f"{'monai':>8}  {'pt eager':>9}  {'best eager':>10}  "
              f"{'pt lazy':>8}  {'best lazy':>9}  "
              f"{'monai/best_e':>12}  {'pt_e/best_e':>11}  {'pt_l/best_l':>11}")
    print(header)
    print("─" * W)

    for S in spatial_sizes:
        for C in channels_list:
            for B in batch_sizes:
                if not try_allocate(B, C, S):
                    print(f"{S:>4}³ {C:>3}  {B:>3}  {'OOM':>8}")
                    continue
                try:
                    monai, pt_e, best_e, pt_l, best_l = bench_fn(B, C, S, repeats)
                    row = (f"{S:>4}³ {C:>3}  {B:>3}  "
                           f"{monai:>8.1f}  {pt_e:>9.1f}  {best_e:>10.1f}  "
                           f"{pt_l:>8.1f}  {best_l:>9.1f}  "
                           f"{fmt_speedup(monai, best_e):>12}  "
                           f"{fmt_speedup(pt_e, best_e):>11}  "
                           f"{fmt_speedup(pt_l, best_l):>11}")
                    print(row)
                    sys.stdout.flush()
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{S:>4}³ {C:>3}  {B:>3}  {'OOM':>8}")


def print_memory_table(title, mem_fn, batch_sizes, spatial_sizes, channels_list):
    """Print memory table."""
    W = 145
    print("\n" + "=" * W)
    print(f"  {title} — PEAK MEMORY (MB above baseline)")
    print(f"  Pipeline: {PIPELINE_DESC}")
    print(f"  {TRITON_NOTE}")
    print("=" * W)

    header = (f"{'S':>4}  {'C':>3}  {'B':>3}  {'input':>8}  "
              f"{'monai':>8}  {'pt eager':>9}  {'best eager':>10}  "
              f"{'pt lazy':>8}  {'best lazy':>9}  "
              f"{'best_e/pt_e':>11}  {'best_l/pt_l':>11}")
    print(header)
    print("─" * W)

    for S in spatial_sizes:
        for C in channels_list:
            for B in batch_sizes:
                if not try_allocate(B, C, S):
                    print(f"{S:>4}³ {C:>3}  {B:>3}  {'OOM':>8}")
                    continue
                try:
                    input_mb = 2 * B * C * S * S * S * 4 / (1024 * 1024)  # vol + seg
                    monai, pt_e, best_e, pt_l, best_l = mem_fn(B, C, S)
                    e_ratio = f"{best_e / pt_e:.2f}x" if pt_e > 0 else "N/A"
                    l_ratio = f"{best_l / pt_l:.2f}x" if pt_l > 0 else "N/A"
                    row = (f"{S:>4}³ {C:>3}  {B:>3}  {input_mb:>8.1f}  "
                           f"{monai:>8.1f}  {pt_e:>9.1f}  {best_e:>10.1f}  "
                           f"{pt_l:>8.1f}  {best_l:>9.1f}  "
                           f"{e_ratio:>11}  {l_ratio:>11}")
                    print(row)
                    sys.stdout.flush()
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{S:>4}³ {C:>3}  {B:>3}  {'OOM':>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark full pipeline: MONAI vs PyTorch batchaug vs best-of-both",
    )
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+", default=[1, 2, 5],
    )
    parser.add_argument(
        "--spatial_sizes", type=int, nargs="+", default=[128, 256, 512],
    )
    parser.add_argument(
        "--channels", type=int, nargs="+", default=[1, 4],
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
    )
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Skip memory profiling",
    )
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0)
    free_gb, total_gb = torch.cuda.mem_get_info()
    print(f"GPU: {device_name} ({total_gb / 1e9:.0f} GB, {free_gb / 1e9:.1f} GB free)")
    print(f"Backend: {batchaug.resolve_backend()}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Spatial sizes: {args.spatial_sizes}")
    print(f"Channels: {args.channels}")
    print(f"Repeats: {args.repeats} (median ms reported)")

    # Task Augmentation
    print_time_table(
        "TASK AUGMENTATION (same aug across C channels)",
        bench_task, args.batch_sizes, args.spatial_sizes, args.channels, args.repeats,
    )
    if not args.no_memory:
        print_memory_table(
            "TASK AUGMENTATION (same aug across C channels)",
            bench_task_memory, args.batch_sizes, args.spatial_sizes, args.channels,
        )

    # Data Augmentation
    print_time_table(
        "DATA AUGMENTATION (independent aug per channel, reshape B*C)",
        bench_data, args.batch_sizes, args.spatial_sizes, args.channels, args.repeats,
    )
    if not args.no_memory:
        print_memory_table(
            "DATA AUGMENTATION (independent aug per channel, reshape B*C)",
            bench_data_memory, args.batch_sizes, args.spatial_sizes, args.channels,
        )

    print()


if __name__ == "__main__":
    main()
