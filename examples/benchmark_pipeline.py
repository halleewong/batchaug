"""Benchmark full augmentation pipeline: MONAI per-sample loop vs batchaug eager vs lazy.

Measures wall-clock time and peak GPU memory for the full pipeline from tests/pipeline.yml.

Usage:
    conda run -n interseg3d python examples/benchmark_pipeline.py 2>&1 | tee examples/benchmark_pipeline.log
    conda run -n interseg3d python examples/benchmark_pipeline.py --batch_sizes 1 2 4 8 --spatial_sizes 64 128
"""

import argparse
import sys

import torch
import monai.transforms

import batchaug


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
    """Build the equivalent MONAI pipeline matching pipeline.yml."""
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


def make_batchaug_pipeline(lazy=False):
    """Build the batchaug pipeline matching examples/pipeline.yml."""
    transforms = [
        batchaug.RandRotate90d(
            keys=["vol", "seg"], prob=0.15, max_k=3, spatial_axes=(0, 1),
        ),
        batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=0.15),
        batchaug.RandAffined(
            keys=["vol", "seg"], prob=0.15,
            rotate_range=0.7853981633974483, shear_range=0.3, translate_range=5,
        ),
        batchaug.RandSimulateLowResolutiond(
            keys=["vol"], prob=0.15, zoom_range=(0.33, 1.0),
        ),
        batchaug.RandGaussianNoised(keys=["vol"], prob=0.15, mean=0.0, std=0.5),
        batchaug.RandBiasFieldd(
            keys=["vol"], prob=0.15, coeff_range=(0.0, 0.05),
        ),
        batchaug.RandGibbsNoised(
            keys=["vol"], prob=0.15, alpha=(0.0, 0.33),
        ),
        batchaug.RandAdjustContrastd(
            keys=["vol"], prob=0.15, gamma=(0.5, 2.5),
        ),
        batchaug.RandGaussianSmoothd(
            keys=["vol"], prob=0.15,
            sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
        ),
        batchaug.RandGaussianSharpend(keys=["vol"], prob=0.15),
        batchaug.ScaleIntensityd(keys=["vol"]),
    ]
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

# -- Task Augmentation: same transform across channels (B, C, H, W, D) -----

def bench_task(B, C, S, repeats):
    """Task aug: same params across C. Returns (monai_ms, eager_ms, lazy_ms)."""
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    keys = ["vol", "seg"]

    monai_pipe = make_monai_pipeline()
    eager_pipe = make_batchaug_pipeline(lazy=False)
    lazy_pipe = make_batchaug_pipeline(lazy=True)

    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_pipe, batch, keys),
        repeats=repeats,
    )
    eager_ms = cuda_timer(lambda: eager_pipe(batch), repeats=repeats)
    lazy_ms = cuda_timer(lambda: lazy_pipe(batch), repeats=repeats)

    return monai_ms, eager_ms, lazy_ms


def bench_task_memory(B, C, S):
    """Task aug peak memory. Returns (monai_mb, eager_mb, lazy_mb)."""
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    keys = ["vol", "seg"]

    monai_pipe = make_monai_pipeline()
    eager_pipe = make_batchaug_pipeline(lazy=False)
    lazy_pipe = make_batchaug_pipeline(lazy=True)

    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_pipe, batch, keys),
    )
    eager_mb = measure_peak_mb(lambda: eager_pipe(batch))
    lazy_mb = measure_peak_mb(lambda: lazy_pipe(batch))

    return monai_mb, eager_mb, lazy_mb


# -- Data Augmentation: independent transform per channel -------------------
#    Reshape (B, C, H, W, D) → (B*C, 1, H, W, D), augment, reshape back.

def _batchaug_data_aug(pipe, vol, seg):
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
    """Data aug: independent params per channel. Returns (monai_ms, eager_ms, lazy_ms)."""
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    keys = ["vol", "seg"]

    # MONAI: per-sample loop over B*C samples, each (1, H, W, D)
    monai_pipe = make_monai_pipeline()
    flat_batch = {
        "vol": vol.reshape(B * C, 1, S, S, S),
        "seg": seg.reshape(B * C, 1, S, S, S),
    }
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_pipe, flat_batch, keys),
        repeats=repeats,
    )

    # batchaug: reshape → pipeline → reshape back
    eager_pipe = make_batchaug_pipeline(lazy=False)
    lazy_pipe = make_batchaug_pipeline(lazy=True)

    eager_ms = cuda_timer(
        lambda: _batchaug_data_aug(eager_pipe, vol, seg),
        repeats=repeats,
    )
    lazy_ms = cuda_timer(
        lambda: _batchaug_data_aug(lazy_pipe, vol, seg),
        repeats=repeats,
    )

    return monai_ms, eager_ms, lazy_ms


def bench_data_memory(B, C, S):
    """Data aug peak memory. Returns (monai_mb, eager_mb, lazy_mb)."""
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

    eager_pipe = make_batchaug_pipeline(lazy=False)
    lazy_pipe = make_batchaug_pipeline(lazy=True)

    eager_mb = measure_peak_mb(
        lambda: _batchaug_data_aug(eager_pipe, vol, seg),
    )
    lazy_mb = measure_peak_mb(
        lambda: _batchaug_data_aug(lazy_pipe, vol, seg),
    )

    return monai_mb, eager_mb, lazy_mb


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def fmt_speedup(base_ms, other_ms):
    if other_ms <= 0:
        return "   inf"
    s = base_ms / other_ms
    return f"{s:5.1f}x"


PIPELINE_DESC = "Rot90 → Flip → Affine → LowRes → Noise → Bias → Gibbs → Contrast → Smooth → Sharpen → Scale"


def print_time_table(title, bench_fn, batch_sizes, spatial_sizes, channels_list, repeats):
    """Print timing table for a given bench function."""
    print("\n" + "=" * 110)
    print(f"  {title} — TIMING (ms, median)")
    print(f"  Pipeline: {PIPELINE_DESC}")
    print("=" * 110)

    header = f"{'S':>4}  {'C':>3}  {'B':>3}  "
    header += f"{'monai':>8}  {'eager':>8}  {'lazy':>8}  "
    header += f"{'eager/monai':>11}  {'lazy/monai':>11}  {'lazy/eager':>11}"
    print(header)
    print("─" * 110)

    for S in spatial_sizes:
        for C in channels_list:
            for B in batch_sizes:
                if not try_allocate(B, C, S):
                    print(f"{S:>4}³ {C:>3}  {B:>3}  {'OOM':>8}  {'OOM':>8}  {'OOM':>8}")
                    continue
                try:
                    monai_ms, eager_ms, lazy_ms = bench_fn(B, C, S, repeats)
                    row = f"{S:>4}³ {C:>3}  {B:>3}  "
                    row += f"{monai_ms:>8.1f}  {eager_ms:>8.1f}  {lazy_ms:>8.1f}  "
                    row += f"{fmt_speedup(monai_ms, eager_ms):>11}  "
                    row += f"{fmt_speedup(monai_ms, lazy_ms):>11}  "
                    row += f"{fmt_speedup(eager_ms, lazy_ms):>11}"
                    print(row)
                    sys.stdout.flush()
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{S:>4}³ {C:>3}  {B:>3}  {'OOM':>8}  {'OOM':>8}  {'OOM':>8}")


def print_memory_table(title, mem_fn, batch_sizes, spatial_sizes, channels_list):
    """Print memory table for a given memory bench function."""
    print("\n" + "=" * 110)
    print(f"  {title} — PEAK MEMORY (MB above baseline)")
    print(f"  Pipeline: {PIPELINE_DESC}")
    print("=" * 110)

    header = f"{'S':>4}  {'C':>3}  {'B':>3}  {'input':>8}  "
    header += f"{'monai':>8}  {'eager':>8}  {'lazy':>8}  "
    header += f"{'eager/monai':>11}  {'lazy/monai':>11}"
    print(header)
    print("─" * 110)

    for S in spatial_sizes:
        for C in channels_list:
            for B in batch_sizes:
                if not try_allocate(B, C, S):
                    print(f"{S:>4}³ {C:>3}  {B:>3}  {'OOM':>8}")
                    continue
                try:
                    input_mb = 2 * B * C * S * S * S * 4 / (1024 * 1024)  # vol + seg
                    monai_mb, eager_mb, lazy_mb = mem_fn(B, C, S)
                    row = f"{S:>4}³ {C:>3}  {B:>3}  {input_mb:>8.1f}  "
                    row += f"{monai_mb:>8.1f}  {eager_mb:>8.1f}  {lazy_mb:>8.1f}  "
                    if monai_mb > 0:
                        row += f"{eager_mb / monai_mb:>10.2f}x  "
                        row += f"{lazy_mb / monai_mb:>10.2f}x"
                    else:
                        row += f"{'N/A':>11}  {'N/A':>11}"
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
        description="Benchmark full pipeline: MONAI loop vs batchaug eager vs lazy",
    )
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+", default=[1, 2, 5],
        help="Batch sizes to sweep (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--spatial_sizes", type=int, nargs="+", default=[128, 256, 512],
        help="Spatial sizes to sweep (default: 64 128)",
    )
    parser.add_argument(
        "--channels", type=int, nargs="+", default=[1, 4],
        help="Channel counts to sweep (default: 1 4)",
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
        help="Timed iterations per measurement (median reported)",
    )
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Skip memory profiling",
    )
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0)
    free_gb, total_gb = torch.cuda.mem_get_info()
    print(f"GPU: {device_name} ({total_gb / 1e9:.0f} GB, {free_gb / 1e9:.1f} GB free)")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Spatial sizes: {args.spatial_sizes}")
    print(f"Channels: {args.channels}")
    print(f"Repeats: {args.repeats} (median ms reported)")

    # Task Augmentation: same augmentation across channels
    print_time_table(
        "TASK AUGMENTATION (same aug across C channels)",
        bench_task, args.batch_sizes, args.spatial_sizes, args.channels, args.repeats,
    )
    if not args.no_memory:
        print_memory_table(
            "TASK AUGMENTATION (same aug across C channels)",
            bench_task_memory, args.batch_sizes, args.spatial_sizes, args.channels,
        )

    # Data Augmentation: independent augmentation per channel
    # Reshape (B, C, H, W, D) → (B*C, 1, H, W, D), augment, reshape back
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
