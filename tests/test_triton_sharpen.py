"""Tests for Triton RandGaussianSharpen: verify Triton matches PyTorch implementation."""
from __future__ import annotations

import pytest
import torch

from batchaug.pytorch.intensity.sharpen import (
    RandGaussianSharpen as PTRandGaussianSharpen,
)
from batchaug.triton.intensity.sharpen import (
    RandGaussianSharpen as TRRandGaussianSharpen,
    RandGaussianSharpend as TRRandGaussianSharpend,
)

B, C, H, W, D = 4, 2, 16, 16, 16
DEVICE = "cuda"


def _compare_sharpen(vol, sigma1_x=(0.8, 0.8), sigma1_y=(0.8, 0.8), sigma1_z=(0.8, 0.8),
                     sigma2_x=0.4, sigma2_y=0.4, sigma2_z=0.4,
                     alpha=(15.0, 15.0), atol=1e-4):
    """Sample params from PyTorch, run both backends, compare."""
    kwargs = dict(
        prob=1.0,
        sigma1_x=sigma1_x, sigma1_y=sigma1_y, sigma1_z=sigma1_z,
        sigma2_x=sigma2_x, sigma2_y=sigma2_y, sigma2_z=sigma2_z,
        alpha=alpha,
    )
    pt = PTRandGaussianSharpen(**kwargs)
    tr = TRRandGaussianSharpen(**kwargs)

    params = pt.sample_params(B, vol.shape, vol.device)

    pt_out = pt.apply(vol, params)
    tr_out = tr.apply(vol, params)

    diff = (pt_out - tr_out).abs().max().item()
    assert torch.allclose(pt_out, tr_out, atol=atol), f"max diff {diff}"
    return diff


def test_fixed_sigma():
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    diff = _compare_sharpen(vol)
    print(f"fixed sigma: max diff = {diff}")


def test_different_sigmas():
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    diff = _compare_sharpen(
        vol,
        sigma1_x=(0.5, 1.0), sigma1_y=(0.6, 0.9), sigma1_z=(0.7, 1.2),
        sigma2_x=(0.3, 0.5), sigma2_y=(0.2, 0.4), sigma2_z=(0.3, 0.6),
        alpha=(10.0, 30.0),
    )
    print(f"different sigmas: max diff = {diff}")


def test_mask():
    """Check that masked-out elements are preserved."""
    tr = TRRandGaussianSharpen(prob=1.0)
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    params = tr.sample_params(B, vol.shape, vol.device)
    params["mask"][:2] = False
    out = tr.apply(vol, params)
    assert torch.equal(out[:2], vol[:2]), "masked-out elements should be unchanged"


def test_single_channel():
    vol = torch.randn(B, 1, H, W, D, device=DEVICE)
    diff = _compare_sharpen(vol)
    print(f"single channel: max diff = {diff}")


def test_nonsquare():
    vol = torch.randn(B, C, 12, 20, 16, device=DEVICE)
    diff = _compare_sharpen(vol)
    print(f"nonsquare: max diff = {diff}")


def test_bfloat16():
    vol = torch.randn(B, C, H, W, D, device=DEVICE, dtype=torch.bfloat16)
    kwargs = dict(
        prob=1.0,
        sigma1_x=(0.8, 0.8), sigma1_y=(0.8, 0.8), sigma1_z=(0.8, 0.8),
        sigma2_x=0.4, sigma2_y=0.4, sigma2_z=0.4,
        alpha=(15.0, 15.0),
    )
    pt = PTRandGaussianSharpen(**kwargs)
    tr = TRRandGaussianSharpen(**kwargs)

    params = pt.sample_params(B, vol.shape, vol.device)
    pt_out = pt.apply(vol, params)
    tr_out = tr.apply(vol, params)

    diff = (pt_out.float() - tr_out.float()).abs().max().item()
    assert diff < 0.05, f"bfloat16 max diff {diff}"
    print(f"bfloat16: max diff = {diff}")


def test_high_alpha():
    """High alpha should still match."""
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    diff = _compare_sharpen(vol, alpha=(50.0, 50.0), atol=5e-4)
    print(f"high alpha: max diff = {diff}")


def test_dict_transform():
    tr = TRRandGaussianSharpend(keys=["vol", "seg"], prob=1.0)
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    seg = torch.randn(B, 1, H, W, D, device=DEVICE)
    result = tr({"vol": vol, "seg": seg})
    assert result["vol"].shape == vol.shape
    assert result["seg"].shape == seg.shape
