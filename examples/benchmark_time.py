"""Benchmark batchaug vs MONAI per-sample loop on GPU.

Sweeps over spatial sizes and channel counts for each transform.

Usage:
    conda run -n interseg3d python examples/benchmark_time.py 2>&1 | tee benchmark_time.log
    conda run -n interseg3d python examples/benchmark_time.py --batch_size 4
    conda run -n interseg3d python examples/benchmark_time.py --spatial_sizes 64 128 --channels 1 4
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


def monai_per_sample_loop(monai_transform, batch, keys):
    """Run MONAI transform in a per-sample loop (the old approach)."""
    B = batch[keys[0]].shape[0]
    results = {k: [] for k in keys}
    for i in range(B):
        sample = {k: batch[k][i] for k in keys}
        out = monai_transform(sample)
        for k in keys:
            results[k].append(out[k])
    return {k: torch.stack(v, dim=0) for k, v in results.items()}


def try_allocate(B, C, S):
    """Check if we can allocate the tensors without OOM."""
    nelems = B * C * S * S * S
    bytes_needed = nelems * 4 * 3  # vol + seg + headroom
    free, _ = torch.cuda.mem_get_info()
    return bytes_needed < free * 0.8


def fmt_speedup(monai_ms, ba_ms):
    if ba_ms <= 0:
        return "   inf"
    s = monai_ms / ba_ms
    return f"{s:5.1f}x"


# ---------------------------------------------------------------------------
# Per-transform benchmark runners
# ---------------------------------------------------------------------------

def bench_one_scale_intensity_cw(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}

    ba_t = batchaug.ScaleIntensityd(keys=["vol"], channel_wise=True)
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.ScaleIntensityd(keys=["vol"], channel_wise=True)
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_scale_intensity_global(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}

    ba_t = batchaug.ScaleIntensityd(keys=["vol"], channel_wise=False)
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.ScaleIntensityd(keys=["vol"], channel_wise=False)
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_rand_axis_flip(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}

    ba_t = batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0)
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.RandAxisFlipd(keys=["vol", "seg"], prob=1.0)
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_rand_rotate90(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}

    ba_t = batchaug.RandRotate90d(
        keys=["vol", "seg"], prob=1.0, max_k=3, spatial_axes=(0, 1)
    )
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.RandRotate90d(
        keys=["vol", "seg"], prob=1.0, max_k=3, spatial_axes=(0, 1)
    )
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_rand_gaussian_noise(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}

    ba_t = batchaug.RandGaussianNoised(
        keys=["vol"], prob=1.0, mean=0.0, std=0.5
    )
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.RandGaussianNoised(
        keys=["vol"], prob=1.0, mean=0.0, std=0.5
    )
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_rand_adjust_contrast(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}

    ba_t = batchaug.RandAdjustContrastd(keys=["vol"], prob=1.0, gamma=(0.5, 4.5))
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.RandAdjustContrastd(keys=["vol"], prob=1.0, gamma=(0.5, 4.5))
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_rand_gaussian_smooth(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}

    ba_t = batchaug.RandGaussianSmoothd(keys=["vol"], prob=1.0)
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.RandGaussianSmoothd(keys=["vol"], prob=1.0)
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_rand_gaussian_sharpen(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}

    ba_t = batchaug.RandGaussianSharpend(keys=["vol"], prob=1.0)
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.RandGaussianSharpend(keys=["vol"], prob=1.0)
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_rand_simulate_low_resolution(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}

    ba_t = batchaug.RandSimulateLowResolutiond(keys=["vol"], prob=1.0, zoom_range=(0.5, 1.0))
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.RandSimulateLowResolutiond(keys=["vol"], prob=1.0, zoom_range=(0.5, 1.0))
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_rand_bias_field(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}

    ba_t = batchaug.RandBiasFieldd(keys=["vol"], prob=1.0, degree=3, coeff_range=(0.0, 0.1))
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.RandBiasFieldd(keys=["vol"], prob=1.0, degree=3, coeff_range=(0.0, 0.1))
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_rand_gibbs_noise(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}

    ba_t = batchaug.RandGibbsNoised(keys=["vol"], prob=1.0, alpha=(0.0, 1.0))
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.RandGibbsNoised(keys=["vol"], prob=1.0, alpha=(0.0, 1.0))
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


def bench_one_rand_affine(B, C, S, repeats):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}

    ba_t = batchaug.RandAffined(
        keys=["vol", "seg"], prob=1.0,
        rotate_range=0.3, scale_range=0.2,
        mode={"vol": "bilinear", "seg": "nearest"},
    )
    ba_ms = cuda_timer(lambda: ba_t(batch), repeats=repeats)

    monai_t = monai.transforms.RandAffined(
        keys=["vol", "seg"], prob=1.0,
        rotate_range=0.3, scale_range=0.2,
        mode=("bilinear", "nearest"),
        padding_mode="zeros",
    )
    monai_ms = cuda_timer(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"]),
        repeats=repeats,
    )
    return monai_ms, ba_ms


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "ScaleIntensityd (per element x channel) [vol]": bench_one_scale_intensity_cw,
    "ScaleIntensityd (per element, all channels) [vol]": bench_one_scale_intensity_global,
    "RandAxisFlipd [vol+seg]": bench_one_rand_axis_flip,
    "RandRotate90d [vol+seg]": bench_one_rand_rotate90,
    "RandGaussianNoised [vol]": bench_one_rand_gaussian_noise,
    "RandAdjustContrastd [vol]": bench_one_rand_adjust_contrast,
    "RandGaussianSmoothd [vol]": bench_one_rand_gaussian_smooth,
    "RandGaussianSharpend [vol]": bench_one_rand_gaussian_sharpen,
    "RandSimulateLowResolutiond [vol]": bench_one_rand_simulate_low_resolution,
    "RandBiasFieldd [vol]": bench_one_rand_bias_field,
    "RandGibbsNoised [vol]": bench_one_rand_gibbs_noise,
    "RandAffined [vol+seg]": bench_one_rand_affine,
}


def fmt_cell(monai_ms, ba_ms):
    """Format one (monai, batchaug, speedup) cell with fixed width."""
    spd = fmt_speedup(monai_ms, ba_ms)
    return f"{monai_ms:>8.1f} {ba_ms:>8.1f} {spd:>7}"


# Each channel group is 8 + 1 + 8 + 1 + 7 = 25 chars, plus 3-char gap = 28
COL_W = 25  # width of one channel group (monai + ba + spd)
GAP = "   "  # gap between channel groups


def run_sweep(name, bench_fn, B, spatial_sizes, channels_list, repeats):
    """Run one transform across all (spatial_size, channels) combos."""
    total_w = 7 + len(channels_list) * (COL_W + len(GAP))
    print(f"\n{'=' * total_w}")
    print(f"  {name}   (B={B})")
    print(f"{'=' * total_w}")

    # Header: channel labels centered over each group
    header = " " * 7
    for C in channels_list:
        label = f"C={C}"
        header += GAP + label.center(COL_W)
    print(header)

    # Sub-header: column names
    sub = " " * 7
    for _ in channels_list:
        sub += GAP + f"{'monai':>8} {'batchaug':>8} {'speedup':>7}"
    print(sub)

    # Separator
    sep = "  " + "─" * 5
    for _ in channels_list:
        sep += GAP + "─" * COL_W
    print(sep)

    for S in spatial_sizes:
        row = f"  {S:>4}³"
        for C in channels_list:
            if not try_allocate(B, C, S):
                row += GAP + f"{'OOM':>8} {'OOM':>8} {'':>7}"
                continue
            try:
                monai_ms, ba_ms = bench_fn(B, C, S, repeats)
                row += GAP + fmt_cell(monai_ms, ba_ms)
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row += GAP + f"{'OOM':>8} {'OOM':>8} {'':>7}"
        print(row)
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark batchaug vs MONAI")
    parser.add_argument(
        "--batch_size", type=int, default=5,
        help="Batch size (default: 5)",
    )
    parser.add_argument(
        "--spatial_sizes", type=int, nargs="+", default=[64, 128, 256],
        help="Spatial sizes to sweep",
    )
    parser.add_argument(
        "--channels", type=int, nargs="+", default=[1, 4, 8],
        help="Channel counts to sweep",
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
        help="Timed iterations per measurement (median reported)",
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+",
        default=list(BENCHMARKS.keys()),
        choices=list(BENCHMARKS.keys()),
        help="Which transforms to benchmark",
    )
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0)
    free_gb, total_gb = torch.cuda.mem_get_info()
    print(f"GPU: {device_name} ({total_gb / 1e9:.0f} GB, {free_gb / 1e9:.1f} GB free)")
    print(f"B={args.batch_size}, spatial={args.spatial_sizes}, C={args.channels}")
    print(f"Repeats: {args.repeats} (median ms reported)")
    print(f"Columns: monai(ms)  batchaug(ms)  speedup")

    for name in args.transforms:
        bench_fn = BENCHMARKS[name]
        run_sweep(name, bench_fn, args.batch_size, args.spatial_sizes,
                  args.channels, args.repeats)

    print()


if __name__ == "__main__":
    main()
