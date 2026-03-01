"""Tests for RandFlip."""
import pytest
import torch
import monai.transforms

import batchaug


class TestRandFlipMatchesMONAI:
    """B=1: batchaug RandFlip matches MONAI RandFlip."""

    @pytest.mark.parametrize("spatial_axis", [0, 1, 2])
    def test_single_axis_matches_monai(self, device, spatial_axis):
        torch.manual_seed(0)
        inp = torch.rand(1, 2, 16, 16, 16, device=device)

        ba_t = batchaug.RandFlip(prob=1.0, spatial_axis=spatial_axis)
        params = {"mask": torch.tensor([True], device=device)}
        ba_out = ba_t.apply(inp, params)

        # MONAI operates on (C, H, W, D) — strip batch dim
        monai_t = monai.transforms.Flip(spatial_axis=spatial_axis)
        monai_out = monai_t(inp[0])  # (C, H, W, D)
        monai_out = monai_out.unsqueeze(0)

        assert torch.allclose(ba_out, monai_out, atol=1e-6)

    def test_all_axes_matches_monai(self, device):
        torch.manual_seed(0)
        inp = torch.rand(1, 2, 16, 16, 16, device=device)

        ba_t = batchaug.RandFlip(prob=1.0, spatial_axis=None)
        params = {"mask": torch.tensor([True], device=device)}
        ba_out = ba_t.apply(inp, params)

        monai_t = monai.transforms.Flip(spatial_axis=None)
        monai_out = monai_t(inp[0]).unsqueeze(0)

        assert torch.allclose(ba_out, monai_out, atol=1e-6)

    def test_multi_axis_matches_monai(self, device):
        torch.manual_seed(0)
        inp = torch.rand(1, 2, 16, 16, 16, device=device)

        ba_t = batchaug.RandFlip(prob=1.0, spatial_axis=[0, 2])
        params = {"mask": torch.tensor([True], device=device)}
        ba_out = ba_t.apply(inp, params)

        monai_t = monai.transforms.Flip(spatial_axis=[0, 2])
        monai_out = monai_t(inp[0]).unsqueeze(0)

        assert torch.allclose(ba_out, monai_out, atol=1e-6)


class TestRandFlipMaskBehavior:
    def test_mask_false_unchanged(self, device):
        inp = torch.rand(2, 2, 8, 8, 8, device=device)
        t = batchaug.RandFlip(prob=1.0, spatial_axis=0)
        params = {"mask": torch.tensor([False, False], device=device)}
        out = t.apply(inp, params)
        assert torch.allclose(out, inp)

    def test_partial_mask(self, device):
        inp = torch.rand(2, 1, 8, 8, 8, device=device)
        t = batchaug.RandFlip(prob=1.0, spatial_axis=0)
        params = {"mask": torch.tensor([True, False], device=device)}
        out = t.apply(inp, params)
        # Element 0 should be flipped, element 1 unchanged
        assert torch.allclose(out[1], inp[1])
        expected_0 = torch.flip(inp[0], dims=[1])  # spatial axis 0 → tensor dim 2
        assert torch.allclose(out[0], expected_0)


class TestRandFlipBatchIndependence:
    def test_each_element_masked_independently(self, device):
        """With prob=0.5 some elements should be flipped and some not."""
        inp = torch.rand(8, 1, 16, 16, 16, device=device)
        t = batchaug.RandFlip(prob=0.5, spatial_axis=0)
        # Sample many times and check at least one flip and one pass-through
        got_flip = False
        got_unchanged = False
        for _ in range(20):
            out = t(inp)
            for b in range(8):
                if torch.allclose(out[b], inp[b]):
                    got_unchanged = True
                elif torch.allclose(out[b], torch.flip(inp[b], dims=[1])):
                    got_flip = True
        assert got_flip and got_unchanged


class TestRandFlipDict:
    def test_dict_same_flip(self, device):
        vol = torch.rand(2, 1, 8, 8, 8, device=device)
        seg = vol.clone()
        t = batchaug.RandFlipd(keys=["vol", "seg"], prob=1.0, spatial_axis=1)
        out = t({"vol": vol, "seg": seg})
        assert torch.allclose(out["vol"], out["seg"])


class TestRandFlipNonSquare:
    def test_nonsquare_shape(self, vol_nonsquare, device):
        t = batchaug.RandFlip(prob=1.0, spatial_axis=2)
        out = t(vol_nonsquare)
        assert out.shape == vol_nonsquare.shape
        assert not torch.isnan(out).any()


class TestRandFlipToAffine:
    def test_to_affine_single_axis(self, device):
        B = 3
        t = batchaug.RandFlip(prob=1.0, spatial_axis=1)
        mask = torch.ones(B, dtype=torch.bool, device=device)
        params = {"mask": mask}
        affine = t.to_affine(params)
        assert affine.shape == (B, 4, 4)
        # axis 1 → diagonal element [1,1] should be -1
        assert (affine[:, 1, 1] == -1.0).all()
        # Other diagonal elements should be +1
        assert (affine[:, 0, 0] == 1.0).all()
        assert (affine[:, 2, 2] == 1.0).all()

    def test_to_affine_masked_out_is_identity(self, device):
        B = 2
        t = batchaug.RandFlip(prob=1.0, spatial_axis=0)
        mask = torch.tensor([True, False], device=device)
        params = {"mask": mask}
        affine = t.to_affine(params)
        eye = torch.eye(4, device=device)
        assert torch.allclose(affine[1], eye)
        assert affine[0, 0, 0] == -1.0
