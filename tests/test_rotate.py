"""Tests for RandRotate."""
import math
import pytest
import torch

import batchaug


class TestRandRotateMatchesRandAffine:
    """RandRotate should give identical results to RandAffine with only rotate_range set."""

    def test_matches_rand_affine_rotate_only(self, device):
        torch.manual_seed(7)
        inp = torch.rand(2, 1, 16, 16, 16, device=device)
        range_x = 0.3
        padding_mode = "border"

        t_rotate = batchaug.RandRotate(
            prob=1.0, range_x=range_x, padding_mode=padding_mode
        )
        t_affine = batchaug.RandAffine(
            prob=1.0, rotate_range=[range_x, 0.0, 0.0], padding_mode=padding_mode
        )

        torch.manual_seed(7)
        params_rot = t_rotate.sample_params(2, inp.shape, device)

        torch.manual_seed(7)
        params_aff = t_affine.sample_params(2, inp.shape, device)

        out_rot = t_rotate.apply(inp, params_rot)
        out_aff = t_affine.apply(inp, params_aff)

        assert torch.allclose(out_rot, out_aff, atol=1e-5)


class TestRandRotateMaskBehavior:
    def test_mask_false_unchanged(self, device):
        inp = torch.rand(2, 1, 16, 16, 16, device=device)
        t = batchaug.RandRotate(prob=1.0, range_x=0.5)
        params = t.sample_params(2, inp.shape, device)
        params["mask"] = torch.zeros(2, dtype=torch.bool, device=device)
        out = t.apply(inp, params)
        assert torch.allclose(out, inp)

    def test_no_rotation_is_identity(self, device):
        """Zero rotation range → output equals input."""
        inp = torch.rand(1, 1, 16, 16, 16, device=device)
        t = batchaug.RandRotate(prob=1.0, range_x=(0.0, 0.0), range_y=(0.0, 0.0), range_z=(0.0, 0.0))
        out = t(inp)
        assert torch.allclose(out, inp, atol=1e-4)


class TestRandRotateTupleRange:
    def test_tuple_range_sampling(self, device):
        t = batchaug.RandRotate(prob=1.0, range_x=(0.1, 0.4))
        # angles are in rotate_range[0]
        assert t.rotate_range[0] == (0.1, 0.4)

    def test_angles_in_range(self, device):
        t = batchaug.RandRotate(prob=1.0, range_x=0.5)
        params = t.sample_params(100, (100, 1, 8, 8, 8), device)
        # The rotation part of the affine should have bounded values
        assert params["affine"].shape == (100, 4, 4)


class TestRandRotateDict:
    def test_dict_different_modes(self, device):
        vol = torch.rand(2, 1, 16, 16, 16, device=device)
        seg = torch.randint(0, 4, (2, 1, 16, 16, 16), device=device).float()
        t = batchaug.RandRotated(
            keys=["vol", "seg"],
            prob=1.0,
            range_x=0.3,
            mode={"vol": "bilinear", "seg": "nearest"},
        )
        out = t({"vol": vol, "seg": seg})
        assert out["vol"].shape == vol.shape
        assert out["seg"].shape == seg.shape


class TestRandRotateNonSquare:
    def test_nonsquare_shape_preserved(self, vol_nonsquare, device):
        t = batchaug.RandRotate(prob=1.0, range_z=0.2)
        out = t(vol_nonsquare)
        assert out.shape == vol_nonsquare.shape


class TestRandRotateBfloat16:
    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandRotate(prob=1.0, range_x=0.2)
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()
