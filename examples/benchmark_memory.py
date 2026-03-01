"""Benchmark peak GPU memory: batchaug vs MONAI per-sample loop.

Measures the peak memory *allocated* during a single forward pass
(excluding the input tensors themselves).

Usage:
    conda run -n batchaug python examples/benchmark_memory.py 2>&1 | tee examples/benchmark_memory.log
    conda run -n batchaug python examples/benchmark_memory.py --batch_size 4
    conda run -n batchaug python examples/benchmark_memory.py --spatial_sizes 64 128 --channels 1 4
"""

import argparse
import sys

import torch
import monai.transforms

import batchaug


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_peak_mb(fn):
    """Run fn once and return peak GPU memory allocated above baseline (MB)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / (1024 * 1024)


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


def input_size_mb(B, C, S, n_tensors=1):
    """Size of input tensor(s) in MB."""
    return n_tensors * B * C * S * S * S * 4 / (1024 * 1024)


# ---------------------------------------------------------------------------
# Per-transform benchmark runners
# Each returns (monai_peak_mb, ba_peak_mb, input_mb)
# ---------------------------------------------------------------------------

def bench_one_scale_intensity_cw(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.ScaleIntensityd(keys=["vol"], channel_wise=True)
    # Warmup
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.ScaleIntensityd(keys=["vol"], channel_wise=True)
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_scale_intensity_global(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.ScaleIntensityd(keys=["vol"], channel_wise=False)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.ScaleIntensityd(keys=["vol"], channel_wise=False)
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_axis_flip(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    in_mb = input_size_mb(B, C, S, n_tensors=2)

    ba_t = batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandAxisFlipd(keys=["vol", "seg"], prob=1.0)
    monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    )

    del vol, seg, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_rotate90(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    in_mb = input_size_mb(B, C, S, n_tensors=2)

    ba_t = batchaug.RandRotate90d(
        keys=["vol", "seg"], prob=1.0, max_k=3, spatial_axes=(0, 1)
    )
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandRotate90d(
        keys=["vol", "seg"], prob=1.0, max_k=3, spatial_axes=(0, 1)
    )
    monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    )

    del vol, seg, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_gaussian_noise(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandGaussianNoised(
        keys=["vol"], prob=1.0, mean=0.0, std=0.5
    )
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandGaussianNoised(
        keys=["vol"], prob=1.0, mean=0.0, std=0.5
    )
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_adjust_contrast(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandAdjustContrastd(keys=["vol"], prob=1.0, gamma=(0.5, 4.5))
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandAdjustContrastd(keys=["vol"], prob=1.0, gamma=(0.5, 4.5))
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_gaussian_smooth(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandGaussianSmoothd(keys=["vol"], prob=1.0)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandGaussianSmoothd(keys=["vol"], prob=1.0)
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_gaussian_sharpen(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandGaussianSharpend(keys=["vol"], prob=1.0)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandGaussianSharpend(keys=["vol"], prob=1.0)
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_simulate_low_resolution(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandSimulateLowResolutiond(keys=["vol"], prob=1.0, zoom_range=(0.5, 1.0))
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandSimulateLowResolutiond(keys=["vol"], prob=1.0, zoom_range=(0.5, 1.0))
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_bias_field(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandBiasFieldd(keys=["vol"], prob=1.0, degree=3, coeff_range=(0.0, 0.1))
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandBiasFieldd(keys=["vol"], prob=1.0, degree=3, coeff_range=(0.0, 0.1))
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_gibbs_noise(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandGibbsNoised(keys=["vol"], prob=1.0, alpha=(0.0, 1.0))
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandGibbsNoised(keys=["vol"], prob=1.0, alpha=(0.0, 1.0))
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_affine(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    in_mb = input_size_mb(B, C, S, n_tensors=2)

    ba_t = batchaug.RandAffined(
        keys=["vol", "seg"], prob=1.0,
        rotate_range=0.3, scale_range=0.2,
        mode={"vol": "bilinear", "seg": "nearest"},
    )
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandAffined(
        keys=["vol", "seg"], prob=1.0,
        rotate_range=0.3, scale_range=0.2,
        mode=("bilinear", "nearest"),
        padding_mode="zeros",
    )
    monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    )

    del vol, seg, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_3d_elastic(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    in_mb = input_size_mb(B, C, S, n_tensors=2)

    ba_t = batchaug.Rand3DElasticd(
        keys=["vol", "seg"], prob=1.0,
        sigma_range=(3.0, 5.0), magnitude_range=(0.1, 0.5),
        mode={"vol": "bilinear", "seg": "nearest"},
    )
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.Rand3DElasticd(
        keys=["vol", "seg"],
        sigma_range=(3.0, 5.0), magnitude_range=(0.1, 0.5),
        prob=1.0,
        mode=("bilinear", "nearest"),
        padding_mode="zeros",
    )
    monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    )

    del vol, seg, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_divisible_pad(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    in_mb = input_size_mb(B, C, S, n_tensors=2)

    ba_t = batchaug.DivisiblePadd(keys=["vol", "seg"], k=48)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.DivisiblePadd(keys=["vol", "seg"], k=48)
    monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    )

    del vol, seg, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_scale_intensity(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandScaleIntensityd(keys=["vol"], prob=1.0, factors=0.5)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandScaleIntensityd(keys=["vol"], prob=1.0, factors=0.5)
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_shift_intensity(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandShiftIntensityd(keys=["vol"], prob=1.0, offsets=0.3)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandShiftIntensityd(keys=["vol"], prob=1.0, offsets=0.3)
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_std_shift_intensity(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandStdShiftIntensityd(keys=["vol"], prob=1.0, factors=3.0)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandStdShiftIntensityd(keys=["vol"], prob=1.0, factors=3.0)
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_scale_intensity_fixed_mean(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandScaleIntensityFixedMeand(keys=["vol"], prob=1.0, factors=0.5)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandScaleIntensityFixedMeand(keys=["vol"], prob=1.0, factors=0.5)
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_rician_noise(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    batch = {"vol": vol}
    in_mb = input_size_mb(B, C, S, n_tensors=1)

    ba_t = batchaug.RandRicianNoised(keys=["vol"], prob=1.0, std=0.1)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandRicianNoised(keys=["vol"], prob=1.0, std=0.1)
    monai_per_sample_loop(monai_t, batch, ["vol"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol"])
    )

    del vol, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_flip(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    in_mb = input_size_mb(B, C, S, n_tensors=2)

    ba_t = batchaug.RandFlipd(keys=["vol", "seg"], prob=1.0)
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandFlipd(keys=["vol", "seg"], prob=1.0, spatial_axis=None)
    monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    )

    del vol, seg, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_rotate(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    in_mb = input_size_mb(B, C, S, n_tensors=2)

    ba_t = batchaug.RandRotated(
        keys=["vol", "seg"], prob=1.0, range_x=0.3,
        mode={"vol": "bilinear", "seg": "nearest"},
    )
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandRotated(
        keys=["vol", "seg"], prob=1.0, range_x=0.3,
        mode=("bilinear", "nearest"), padding_mode="border",
    )
    monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    )

    del vol, seg, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


def bench_one_rand_zoom(B, C, S):
    vol = torch.rand(B, C, S, S, S, device="cuda")
    seg = torch.randint(0, 5, (B, C, S, S, S), device="cuda").float()
    batch = {"vol": vol, "seg": seg}
    in_mb = input_size_mb(B, C, S, n_tensors=2)

    ba_t = batchaug.RandZoomd(
        keys=["vol", "seg"], prob=1.0, min_zoom=0.8, max_zoom=1.2,
        mode={"vol": "bilinear", "seg": "nearest"},
    )
    ba_t(batch)
    ba_mb = measure_peak_mb(lambda: ba_t(batch))

    monai_t = monai.transforms.RandZoomd(
        keys=["vol", "seg"], prob=1.0, min_zoom=0.8, max_zoom=1.2,
        mode=("bilinear", "nearest"), keep_size=True, padding_mode="edge",
    )
    monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    monai_mb = measure_peak_mb(
        lambda: monai_per_sample_loop(monai_t, batch, ["vol", "seg"])
    )

    del vol, seg, batch
    torch.cuda.empty_cache()
    return monai_mb, ba_mb, in_mb


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "ScaleIntensityd (per element×channel) [vol]": bench_one_scale_intensity_cw,
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
    "Rand3DElasticd [vol+seg]": bench_one_rand_3d_elastic,
    "DivisiblePadd [vol+seg]": bench_one_divisible_pad,
    "RandScaleIntensityd [vol]": bench_one_rand_scale_intensity,
    "RandShiftIntensityd [vol]": bench_one_rand_shift_intensity,
    "RandStdShiftIntensityd [vol]": bench_one_rand_std_shift_intensity,
    "RandScaleIntensityFixedMeand [vol]": bench_one_rand_scale_intensity_fixed_mean,
    "RandRicianNoised [vol]": bench_one_rand_rician_noise,
    "RandFlipd [vol+seg]": bench_one_rand_flip,
    "RandRotated [vol+seg]": bench_one_rand_rotate,
    "RandZoomd [vol+seg]": bench_one_rand_zoom,
}


def fmt_ratio(mb, in_mb):
    """Format overhead as ratio of input size."""
    if in_mb <= 0:
        return "    n/a"
    return f"{mb / in_mb:5.1f}x"


def fmt_cell(monai_mb, ba_mb, in_mb):
    """Format one (monai, batchaug, ratio) cell with fixed width."""
    return (
        f"{monai_mb:>8.1f} {ba_mb:>8.1f}"
        f" {fmt_ratio(monai_mb, in_mb):>7} {fmt_ratio(ba_mb, in_mb):>7}"
    )


# monai(8) + ba(8) + monai_ratio(7) + ba_ratio(7) + 3 spaces = 33
COL_W = 33
GAP = "   "


def run_sweep(name, bench_fn, B, spatial_sizes, channels_list):
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

    # Sub-header
    sub = " " * 7
    for _ in channels_list:
        sub += GAP + f"{'monai':>8} {'batchaug':>8} {'m/in':>7} {'b/in':>7}"
    print(sub)

    # Units
    units = " " * 7
    for _ in channels_list:
        units += GAP + f"{'(MB)':>8} {'(MB)':>8} {'':>7} {'':>7}"
    print(units)

    # Separator
    sep = "  " + "─" * 5
    for _ in channels_list:
        sep += GAP + "─" * COL_W
    print(sep)

    for S in spatial_sizes:
        row = f"  {S:>4}³"
        for C in channels_list:
            if not try_allocate(B, C, S):
                row += GAP + f"{'OOM':>8} {'OOM':>8} {'':>7} {'':>7}"
                continue
            try:
                monai_mb, ba_mb, in_mb = bench_fn(B, C, S)
                row += GAP + fmt_cell(monai_mb, ba_mb, in_mb)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row += GAP + f"{'OOM':>8} {'OOM':>8} {'':>7} {'':>7}"
        print(row)
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark peak GPU memory: batchaug vs MONAI"
    )
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
    print(f"Peak memory above baseline (MB) | m/in = monai/input, b/in = batchaug/input")

    for name in args.transforms:
        bench_fn = BENCHMARKS[name]
        run_sweep(name, bench_fn, args.batch_size, args.spatial_sizes,
                  args.channels)

    print()


if __name__ == "__main__":
    main()
