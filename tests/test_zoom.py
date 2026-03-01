"""Tests for RandZoom."""
import pytest
import torch

import batchaug


class TestRandZoomIdentity:
    def test_zoom_one_is_identity(self, device):
        """zoom=1.0 should leave the output unchanged."""
        inp = torch.rand(1, 1, 16, 16, 16, device=device)
        t = batchaug.RandZoom(prob=1.0, min_zoom=1.0, max_zoom=1.0)
        out = t(inp)
        assert torch.allclose(out, inp, atol=1e-4)

    def test_mask_false_unchanged(self, device):
        inp = torch.rand(2, 1, 16, 16, 16, device=device)
        t = batchaug.RandZoom(prob=1.0, min_zoom=0.5, max_zoom=2.0)
        params = t.sample_params(2, inp.shape, device)
        params["mask"] = torch.zeros(2, dtype=torch.bool, device=device)
        out = t.apply(inp, params)
        assert torch.allclose(out, inp)


class TestRandZoomSampling:
    def test_zoom_factors_in_range(self, device):
        t = batchaug.RandZoom(prob=1.0, min_zoom=0.8, max_zoom=1.2)
        params = t.sample_params(200, (200, 1, 8, 8, 8), device)
        assert (params["zoom"] >= 0.8).all()
        assert (params["zoom"] <= 1.2).all()
        assert params["zoom"].std() > 0.01

    def test_affine_diagonal_is_inv_zoom(self, device):
        """Affine diagonal should be 1/zoom (inverse mapping convention)."""
        t = batchaug.RandZoom(prob=1.0, min_zoom=2.0, max_zoom=2.0)
        params = t.sample_params(4, (4, 1, 8, 8, 8), device)
        # zoom=2.0 → affine diagonal = 0.5
        assert torch.allclose(params["affine"][:, 0, 0], torch.tensor(0.5, device=device), atol=1e-5)


class TestRandZoomZoomIn:
    def test_zoom_in_changes_content(self, device):
        """zoom > 1 (zoom in) should produce a different output than identity."""
        inp = torch.rand(1, 1, 16, 16, 16, device=device)
        t = batchaug.RandZoom(prob=1.0, min_zoom=1.5, max_zoom=1.5)
        out = t(inp)
        assert not torch.allclose(out, inp, atol=1e-3)
        assert out.shape == inp.shape


class TestRandZoomToAffine:
    def test_to_affine_shape(self, device):
        t = batchaug.RandZoom(prob=1.0, min_zoom=0.9, max_zoom=1.1)
        params = t.sample_params(4, (4, 1, 8, 8, 8), device)
        affine = t.to_affine(params)
        assert affine.shape == (4, 4, 4)

    def test_to_affine_masked_out_is_identity(self, device):
        t = batchaug.RandZoom(prob=1.0, min_zoom=2.0, max_zoom=2.0)
        params = t.sample_params(2, (2, 1, 8, 8, 8), device)
        params["mask"] = torch.tensor([True, False], device=device)
        affine = t.to_affine(params)
        eye = torch.eye(4, device=device)
        assert torch.allclose(affine[1], eye, atol=1e-6)
        assert torch.allclose(affine[0, 0, 0], torch.tensor(0.5, device=device), atol=1e-5)


class TestRandZoomDict:
    def test_dict_different_modes(self, device):
        vol = torch.rand(2, 1, 16, 16, 16, device=device)
        seg = torch.randint(0, 4, (2, 1, 16, 16, 16), device=device).float()
        t = batchaug.RandZoomd(
            keys=["vol", "seg"],
            prob=1.0,
            min_zoom=0.9,
            max_zoom=1.1,
            mode={"vol": "bilinear", "seg": "nearest"},
        )
        out = t({"vol": vol, "seg": seg})
        assert out["vol"].shape == vol.shape
        assert out["seg"].shape == seg.shape


class TestRandZoomNonSquare:
    def test_nonsquare_shape_preserved(self, vol_nonsquare, device):
        t = batchaug.RandZoom(prob=1.0, min_zoom=0.9, max_zoom=1.1)
        out = t(vol_nonsquare)
        assert out.shape == vol_nonsquare.shape


class TestRandZoomBfloat16:
    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandZoom(prob=1.0, min_zoom=0.9, max_zoom=1.1)
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()
