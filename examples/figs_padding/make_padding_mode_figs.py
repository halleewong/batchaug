"""Illustrate the `padding_mode` option on RandGaussianSmooth / RandGaussianSharpen.

MONAI's GaussianSmooth/GaussianSharpen convolve with implicit zero-padding, so
voxels within `kernel_size // 2` of an edge average in zeros and pick up a dark
halo (and an inverted bright halo for the unsharp-mask sharpen). The new
`padding_mode` argument (default None = MONAI behaviour) lets you pad with
`"reflect"` or `"replicate"` instead, which removes the halo.

To make the halo obvious we use a synthetic *coarse checkerboard* that is bright
all the way to the volume boundary — a real medical volume cropped to background
would be near-zero at the edge and the artifact would be invisible.

For each transform this writes one PNG with the original checkerboard and the
result of each padding mode side by side. Sigma is fixed and the seed is reset
before every call, so the only difference between panels is the padding mode —
the boundary halo is present only in the zero-padded (MONAI default) panel.

    conda run -n batchaug python examples/figs_padding/make_padding_mode_figs.py

No cluster data required — everything is synthetic and runs on one GPU.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import batchaug as ba


# -------------------- synthetic volume --------------------

def make_checkerboard(size=96, cell=16, low=0.4, high=1.0, device="cuda"):
    """A coarse 3D checkerboard as a (1, 1, size, size, size) float tensor.

    Cells alternate between `low` and `high`; both are non-zero so the bright
    cells that touch the boundary reveal the zero-padding halo. `cell` is chosen
    so the pattern is coarse relative to the Gaussian kernel — fine enough cells
    would just be blurred away and you'd never see the edge effect in isolation.
    """
    idx = torch.arange(size, device=device) // cell           # cell index per axis
    parity = (idx[:, None, None] + idx[None, :, None] + idx[None, None, :]) % 2
    vol = torch.where(parity.bool(), torch.tensor(high, device=device),
                      torch.tensor(low, device=device))
    return vol.to(torch.float32)[None, None]                   # (1,1,H,W,D)


# -------------------- transforms per padding mode --------------------

# Fixed sigma so the only thing that varies between frames is the padding mode.
SIGMA = 3.0

# (frame label, padding_mode passed to the transform). None is the MONAI default.
MODES = [
    ("zero (MONAI default)", None),
    ("reflect", "reflect"),
    ("replicate", "replicate"),
]


def smooth_transform(padding_mode):
    return ba.RandGaussianSmoothd(
        keys=["vol"], prob=1.0,
        sigma_x=(SIGMA, SIGMA), sigma_y=(SIGMA, SIGMA), sigma_z=(SIGMA, SIGMA),
        padding_mode=padding_mode,
    )


def sharpen_transform(padding_mode):
    # Pin both blur sigmas and the sharpening weight so frames are comparable.
    return ba.RandGaussianSharpend(
        keys=["vol"], prob=1.0,
        sigma1_x=(SIGMA, SIGMA), sigma1_y=(SIGMA, SIGMA), sigma1_z=(SIGMA, SIGMA),
        sigma2_x=(SIGMA, SIGMA), sigma2_y=(SIGMA, SIGMA), sigma2_z=(SIGMA, SIGMA),
        alpha=(10.0, 10.0),
        padding_mode=padding_mode,
    )


TRANSFORMS = {
    "smooth": ("RandGaussianSmooth", smooth_transform),
    "sharpen": ("RandGaussianSharpen", sharpen_transform),
}


# -------------------- figure --------------------

def midslice(t):
    """Centre slice along the last axis of a (1,1,H,W,D) tensor → (H,W) numpy."""
    d = t.shape[-1] // 2
    return t[0, 0, :, :, d].detach().cpu().numpy()


def make_panel(kind, out_dir, size, cell, device="cuda", seed=0):
    """Single static PNG: original checkerboard + result of each padding mode,
    side by side. Sigma is fixed and the seed is reset before every call, so the
    only difference between panels is the padding mode."""
    title, factory = TRANSFORMS[kind]
    vol = make_checkerboard(size=size, cell=cell, device=device)

    panels = [("original", midslice(vol))]
    for label, pm in MODES:
        torch.manual_seed(seed); np.random.seed(seed)
        out = factory(pm)({"vol": vol.clone()})["vol"]
        panels.append((f"padding_mode={label}", midslice(out)))

    fig, axs = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4))
    fig.subplots_adjust(left=0.01, right=0.97, top=0.86, bottom=0.01, wspace=0.25)
    fig.suptitle(title, fontsize=14)
    for ax, (label, img) in zip(axs, panels):
        # No vmin/vmax: each panel autoscales to its own range so the unsharp
        # overshoot beyond [0, 1] is visible instead of being clipped to white.
        im = ax.imshow(img.T, cmap="gray", origin="upper",
                       interpolation="nearest")
        ax.axis("off")
        ax.set_title(label, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{title.lower()}_padding_modes.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--size", type=int, default=96, help="volume edge length (voxels)")
    ap.add_argument("--cell", type=int, default=16, help="checkerboard cell size (voxels)")
    ap.add_argument("--kind", choices=[*TRANSFORMS, "all"], default="all")
    args = ap.parse_args()

    kinds = list(TRANSFORMS) if args.kind == "all" else [args.kind]
    for kind in kinds:
        make_panel(kind, args.out_dir, size=args.size, cell=args.cell)


if __name__ == "__main__":
    main()
