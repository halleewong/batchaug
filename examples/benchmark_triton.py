"""Benchmark per-transform: PyTorch vs Triton backends.

Measures wall-clock time for each Triton-accelerated transform against
its PyTorch counterpart. Only benchmarks apply() — sample_params excluded.

Usage:
    conda run -n interseg3d python examples/benchmark_triton.py 2>&1 | tee examples/benchmark_triton.log
    conda run -n interseg3d python examples/benchmark_triton.py --spatial_sizes 64 128 256
"""

import argparse
import sys

import torch

from batchaug.pytorch.intensity.contrast import (
    RandAdjustContrast as PTRandAdjustContrast,
    ScaleIntensity as PTScaleIntensity,
)
from batchaug.pytorch.intensity.smooth import (
    RandGaussianSmooth as PTRandGaussianSmooth,
)
from batchaug.pytorch.intensity.sharpen import (
    RandGaussianSharpen as PTRandGaussianSharpen,
)
from batchaug.pytorch.intensity.bias_field import (
    RandBiasField as PTRandBiasField,
)
from batchaug.triton.intensity.contrast import (
    RandAdjustContrast as TRRandAdjustContrast,
    ScaleIntensity as TRScaleIntensity,
)
from batchaug.triton.intensity.smooth import (
    RandGaussianSmooth as TRRandGaussianSmooth,
)
from batchaug.triton.intensity.sharpen import (
    RandGaussianSharpen as TRRandGaussianSharpen,
)
from batchaug.triton.intensity.bias_field import (
    RandBiasField as TRRandBiasField,
)


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


def bench_transform(name, pt_cls, tr_cls, vol, pt_kwargs=None, tr_kwargs=None,
                    pt_params_override=None, tr_params_override=None, repeats=10):
    """Benchmark a single transform. Returns (pt_ms, tr_ms)."""
    kwargs = pt_kwargs or {}
    pt = pt_cls(prob=1.0, **kwargs)
    tr = tr_cls(prob=1.0, **(tr_kwargs or kwargs))

    B = vol.shape[0]
    pt_params = pt.sample_params(B, vol.shape, vol.device)
    if pt_params_override:
        pt_params.update(pt_params_override)

    # For Triton bias field, sample_params skips basis precomputation
    tr_params = tr.sample_params(B, vol.shape, vol.device)
    # Share the same random params (mask + transform-specific)
    for k in pt_params:
        if k in tr_params:
            tr_params[k] = pt_params[k]
        elif k != "basis":
            tr_params[k] = pt_params[k]

    pt_ms = cuda_timer(lambda: pt.apply(vol, pt_params), repeats=repeats)
    tr_ms = cuda_timer(lambda: tr.apply(vol, tr_params), repeats=repeats)

    return pt_ms, tr_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark per-transform: PyTorch vs Triton")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--spatial_sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--channels", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0)
    free_gb, total_gb = torch.cuda.mem_get_info()
    print(f"GPU: {device_name} ({total_gb / 1e9:.0f} GB, {free_gb / 1e9:.1f} GB free)")
    print(f"Batch: {args.batch_sizes}, Spatial: {args.spatial_sizes}, Channels: {args.channels}")
    print(f"Repeats: {args.repeats}\n")

    transforms = [
        ("ScaleIntensity", PTScaleIntensity, TRScaleIntensity, {}),
        ("RandAdjustContrast", PTRandAdjustContrast, TRRandAdjustContrast,
         {"gamma": (0.5, 2.5)}),
        ("RandGaussianSmooth", PTRandGaussianSmooth, TRRandGaussianSmooth,
         {"sigma_x": (0.5, 1.5), "sigma_y": (0.5, 1.5), "sigma_z": (0.5, 1.5)}),
        ("RandGaussianSharpen", PTRandGaussianSharpen, TRRandGaussianSharpen, {}),
        ("RandBiasField", PTRandBiasField, TRRandBiasField,
         {"coeff_range": (0.0, 0.1)}),
    ]

    # Header
    print(f"{'Transform':<24} {'S':>4} {'C':>3} {'B':>3}  "
          f"{'PyTorch':>9} {'Triton':>9} {'Speedup':>8}")
    print("─" * 80)

    for S in args.spatial_sizes:
        for C in args.channels:
            for B in args.batch_sizes:
                # Check memory
                nelems = B * C * S * S * S
                free, _ = torch.cuda.mem_get_info()
                if nelems * 4 * 4 > free * 0.7:
                    print(f"{'(skipped)':>24} {S:>4}³ {C:>3} {B:>3}  OOM")
                    continue

                vol = torch.randn(B, C, S, S, S, device="cuda")

                for name, pt_cls, tr_cls, kwargs in transforms:
                    try:
                        # ScaleIntensity has no prob parameter in __init__,
                        # use a different path
                        if name == "ScaleIntensity":
                            pt = pt_cls()
                            tr = tr_cls()
                            pt_params = pt.sample_params(B, vol.shape, vol.device)
                            tr_params = dict(pt_params)
                            pt_ms = cuda_timer(
                                lambda: pt.apply(vol, pt_params), repeats=args.repeats
                            )
                            tr_ms = cuda_timer(
                                lambda: tr.apply(vol, tr_params), repeats=args.repeats
                            )
                        else:
                            pt_ms, tr_ms = bench_transform(
                                name, pt_cls, tr_cls, vol, kwargs, repeats=args.repeats,
                            )

                        speedup = pt_ms / tr_ms if tr_ms > 0 else float("inf")
                        print(f"{name:<24} {S:>4}³ {C:>3} {B:>3}  "
                              f"{pt_ms:>8.2f}ms {tr_ms:>8.2f}ms {speedup:>7.2f}x")
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"{name:<24} {S:>4}³ {C:>3} {B:>3}  OOM")

                del vol
                torch.cuda.empty_cache()
                print()  # blank line between configs

        sys.stdout.flush()

    print("Done.")


if __name__ == "__main__":
    main()
