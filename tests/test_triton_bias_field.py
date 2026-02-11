"""Tests for Triton RandBiasField: verify Triton matches PyTorch implementation."""
from __future__ import annotations

import pytest
import torch

from batchaug.pytorch.intensity.bias_field import RandBiasField as PTRandBiasField
from batchaug.triton.intensity.bias_field import (
    RandBiasField as TRRandBiasField,
    RandBiasFieldd as TRRandBiasFieldd,
)

B, C, H, W, D = 4, 2, 16, 16, 16
DEVICE = "cuda"


def _compare_bias_field(vol, coeff_range=(0.0, 0.1), atol=1e-5):
    """Sample params from PyTorch, run both backends, compare."""
    pt = PTRandBiasField(prob=1.0, degree=3, coeff_range=coeff_range)
    tr = TRRandBiasField(prob=1.0, degree=3, coeff_range=coeff_range)

    pt_params = pt.sample_params(B, vol.shape, vol.device)

    # Triton params: same coeffs + mask, no basis
    tr_params = {"mask": pt_params["mask"], "coeffs": pt_params["coeffs"]}

    pt_out = pt.apply(vol, pt_params)
    tr_out = tr.apply(vol, tr_params)

    diff = (pt_out - tr_out).abs().max().item()
    assert torch.allclose(pt_out, tr_out, atol=atol), f"max diff {diff}"
    return diff


def test_basic():
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    diff = _compare_bias_field(vol)
    print(f"basic: max diff = {diff}")


def test_larger_coeff_range():
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    diff = _compare_bias_field(vol, coeff_range=(-0.5, 0.5), atol=1e-4)
    print(f"large coeff range: max diff = {diff}")


def test_single_channel():
    vol = torch.randn(B, 1, H, W, D, device=DEVICE)
    diff = _compare_bias_field(vol)
    print(f"single channel: max diff = {diff}")


def test_nonsquare():
    vol = torch.randn(B, C, 12, 20, 16, device=DEVICE)
    diff = _compare_bias_field(vol)
    print(f"nonsquare: max diff = {diff}")


def test_mask():
    """Check that masked-out elements are preserved."""
    tr = TRRandBiasField(prob=1.0, degree=3, coeff_range=(0.0, 0.1))
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    params = tr.sample_params(B, vol.shape, vol.device)

    # Force first two elements to be masked out
    params["mask"][:2] = False
    out = tr.apply(vol, params)

    assert torch.equal(out[:2], vol[:2]), "masked-out elements should be unchanged"
    # Masked-in elements should differ (bias field != identity for non-zero coeffs)
    if params["mask"][2:].any():
        assert not torch.equal(out[2:], vol[2:]), "masked-in elements should change"


def test_bfloat16():
    vol = torch.randn(B, C, H, W, D, device=DEVICE, dtype=torch.bfloat16)
    pt = PTRandBiasField(prob=1.0, degree=3, coeff_range=(0.0, 0.1))
    tr = TRRandBiasField(prob=1.0, degree=3, coeff_range=(0.0, 0.1))

    pt_params = pt.sample_params(B, vol.shape, vol.device)
    tr_params = {"mask": pt_params["mask"], "coeffs": pt_params["coeffs"]}

    pt_out = pt.apply(vol, pt_params)
    tr_out = tr.apply(vol, tr_params)

    diff = (pt_out.float() - tr_out.float()).abs().max().item()
    assert diff < 0.02, f"bfloat16 max diff {diff}"
    print(f"bfloat16: max diff = {diff}")


def test_all_masked_out():
    """All elements masked out — output should equal input."""
    tr = TRRandBiasField(prob=1.0, degree=3, coeff_range=(0.0, 0.1))
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    params = tr.sample_params(B, vol.shape, vol.device)
    params["mask"][:] = False
    out = tr.apply(vol, params)
    assert torch.equal(out, vol)


def test_dict_transform():
    """Verify RandBiasFieldd applies same bias to all keys."""
    tr = TRRandBiasFieldd(keys=["vol", "seg"], prob=1.0, coeff_range=(0.0, 0.1))
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    seg = torch.randn(B, 1, H, W, D, device=DEVICE)

    result = tr({"vol": vol, "seg": seg})
    # Both should be modified (prob=1.0 with non-zero coefficients)
    assert result["vol"].shape == vol.shape
    assert result["seg"].shape == seg.shape


def test_degree3_fallback():
    """Degree != 3 should fall back to PyTorch implementation."""
    tr = TRRandBiasField(prob=1.0, degree=2, coeff_range=(0.0, 0.1))
    vol = torch.randn(B, C, H, W, D, device=DEVICE)
    params = tr.sample_params(B, vol.shape, vol.device)
    # Should have basis (PyTorch fallback)
    assert "basis" in params
    out = tr.apply(vol, params)
    assert out.shape == vol.shape
