"""Tests for RandScaleIntensity, RandShiftIntensity,
RandStdShiftIntensity, and RandScaleIntensityFixedMean."""
import pytest
import torch

import batchaug


# ===========================================================================
# RandScaleIntensity
# ===========================================================================

class TestRandScaleIntensityFormula:
    """B=1: verify output = input * (1 + factor)."""

    def test_scale_formula(self, device):
        inp = torch.rand(1, 2, 8, 8, 8, device=device)
        t = batchaug.RandScaleIntensity(prob=1.0, factors=(0.5, 0.5))
        params = {
            "mask": torch.tensor([True], device=device),
            "factor": torch.tensor([0.5], device=device),
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, inp * 1.5, atol=1e-6)

    def test_mask_false_unchanged(self, device):
        inp = torch.rand(1, 2, 8, 8, 8, device=device)
        t = batchaug.RandScaleIntensity(prob=1.0, factors=(0.5, 0.5))
        params = {
            "mask": torch.tensor([False], device=device),
            "factor": torch.tensor([0.5], device=device),
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, inp)

    def test_negative_factor_shrinks(self, device):
        inp = torch.ones(1, 1, 4, 4, 4, device=device) * 2.0
        t = batchaug.RandScaleIntensity(prob=1.0, factors=(-0.5, -0.5))
        params = {
            "mask": torch.tensor([True], device=device),
            "factor": torch.tensor([-0.5], device=device),
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, inp * 0.5, atol=1e-6)


class TestRandScaleIntensitySampling:
    def test_scalar_factor_is_symmetric(self, device):
        t = batchaug.RandScaleIntensity(prob=1.0, factors=0.3)
        params = t.sample_params(200, (200, 1, 4, 4, 4), device)
        assert (params["factor"] >= -0.3).all()
        assert (params["factor"] <= 0.3).all()

    def test_tuple_factor_range(self, device):
        t = batchaug.RandScaleIntensity(prob=1.0, factors=(0.1, 0.8))
        params = t.sample_params(200, (200, 1, 4, 4, 4), device)
        assert (params["factor"] >= 0.1).all()
        assert (params["factor"] <= 0.8).all()
        assert params["factor"].std() > 0.01

    def test_batch_independence(self, device):
        inp = torch.ones(4, 1, 8, 8, 8, device=device)
        t = batchaug.RandScaleIntensity(prob=1.0, factors=(-0.5, 0.5))
        out = t(inp)
        # Different batch elements should have different scale factors
        vals = out.flatten(1).mean(dim=1)
        assert not torch.allclose(vals[0:1], vals[1:2], atol=1e-3)


class TestRandScaleIntensityDict:
    def test_dict_same_params(self, device):
        vol = torch.rand(2, 1, 8, 8, 8, device=device)
        seg = torch.rand(2, 1, 8, 8, 8, device=device)
        t = batchaug.RandScaleIntensityd(keys=["vol", "seg"], prob=1.0, factors=(0.3, 0.3))
        out = t({"vol": vol, "seg": seg})
        # Both should be scaled by the same factor
        ratio_vol = out["vol"] / vol.clamp(min=1e-8)
        ratio_seg = out["seg"] / seg.clamp(min=1e-8)
        assert torch.allclose(ratio_vol, ratio_seg, atol=1e-5)


class TestRandScaleIntensityBfloat16:
    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandScaleIntensity(prob=1.0, factors=(0.1, 0.5))
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()


# ===========================================================================
# RandShiftIntensity
# ===========================================================================

class TestRandShiftIntensityFormula:
    def test_shift_formula(self, device):
        inp = torch.rand(1, 2, 8, 8, 8, device=device)
        t = batchaug.RandShiftIntensity(prob=1.0, offsets=(0.2, 0.2))
        params = {
            "mask": torch.tensor([True], device=device),
            "offset": torch.tensor([0.2], device=device),
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, inp + 0.2, atol=1e-6)

    def test_mask_false_unchanged(self, device):
        inp = torch.rand(1, 1, 8, 8, 8, device=device)
        t = batchaug.RandShiftIntensity(prob=1.0)
        params = {
            "mask": torch.tensor([False], device=device),
            "offset": torch.tensor([0.5], device=device),
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, inp)

    def test_safe_clamps(self, device):
        inp = torch.ones(1, 1, 4, 4, 4, device=device) * 0.9
        t = batchaug.RandShiftIntensity(prob=1.0, offsets=(0.5, 0.5), safe=True)
        params = {
            "mask": torch.tensor([True], device=device),
            "offset": torch.tensor([0.5], device=device),
        }
        out = t.apply(inp, params)
        assert out.max() <= 1.0 + 1e-6
        assert out.min() >= 0.0 - 1e-6


class TestRandShiftIntensitySampling:
    def test_scalar_offset_is_symmetric(self, device):
        t = batchaug.RandShiftIntensity(prob=1.0, offsets=0.2)
        params = t.sample_params(200, (200, 1, 4, 4, 4), device)
        assert (params["offset"] >= -0.2).all()
        assert (params["offset"] <= 0.2).all()


class TestRandShiftIntensityBfloat16:
    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandShiftIntensity(prob=1.0, offsets=(-0.1, 0.1))
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()


# ===========================================================================
# RandStdShiftIntensity
# ===========================================================================

class TestRandStdShiftIntensityFormula:
    def test_std_shift_formula(self, device):
        """output = input + factor * std(input)."""
        inp = torch.rand(1, 2, 8, 8, 8, device=device)
        factor = 2.0
        expected_std = inp.flatten().std()
        expected = inp + factor * expected_std

        t = batchaug.RandStdShiftIntensity(prob=1.0, channel_wise=False)
        params = {
            "mask": torch.tensor([True], device=device),
            "factor": torch.tensor([factor], device=device),
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_mask_false_unchanged(self, device):
        inp = torch.rand(1, 1, 8, 8, 8, device=device)
        t = batchaug.RandStdShiftIntensity(prob=1.0)
        params = {
            "mask": torch.tensor([False], device=device),
            "factor": torch.tensor([3.0], device=device),
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, inp)

    def test_channel_wise_uses_per_channel_std(self, device):
        """With channel_wise=True each channel is shifted by its own std."""
        B, C = 1, 3
        inp = torch.rand(B, C, 8, 8, 8, device=device)
        # Scale channel 1 so it has a very different std
        inp[0, 1] = inp[0, 1] * 0.01
        factor = 1.0

        t = batchaug.RandStdShiftIntensity(prob=1.0, channel_wise=True)
        params = {
            "mask": torch.tensor([True], device=device),
            "factor": torch.tensor([factor], device=device),
        }
        out = t.apply(inp, params)

        # Channels 0 and 2 should have larger shift than channel 1
        shift_c0 = (out[0, 0] - inp[0, 0]).abs().mean()
        shift_c1 = (out[0, 1] - inp[0, 1]).abs().mean()
        assert shift_c0 > shift_c1 * 5


class TestRandStdShiftIntensityBfloat16:
    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandStdShiftIntensity(prob=1.0)
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()


# ===========================================================================
# RandScaleIntensityFixedMean
# ===========================================================================

class TestRandScaleIntensityFixedMeanFormula:
    def test_fixed_mean_formula(self, device):
        """output = mean + (input - mean) * (1 + factor)."""
        inp = torch.rand(1, 2, 8, 8, 8, device=device)
        factor = 0.5
        mean = inp.mean()
        expected = mean + (inp - mean) * (1.0 + factor)

        t = batchaug.RandScaleIntensityFixedMean(prob=1.0, channel_wise=False)
        params = {
            "mask": torch.tensor([True], device=device),
            "factor": torch.tensor([factor], device=device),
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, expected.to(out.dtype), atol=1e-5)

    def test_mean_is_preserved(self, device):
        """After transform the spatial mean should be unchanged."""
        inp = torch.rand(1, 1, 16, 16, 16, device=device)
        t = batchaug.RandScaleIntensityFixedMean(prob=1.0, factors=(0.8, 0.8))
        out = t(inp)
        assert torch.allclose(out.mean(), inp.mean(), atol=1e-5)

    def test_mask_false_unchanged(self, device):
        inp = torch.rand(1, 1, 8, 8, 8, device=device)
        t = batchaug.RandScaleIntensityFixedMean(prob=1.0)
        params = {
            "mask": torch.tensor([False], device=device),
            "factor": torch.tensor([0.5], device=device),
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, inp)

    def test_channel_wise_preserves_per_channel_mean(self, device):
        inp = torch.rand(1, 3, 16, 16, 16, device=device)
        t = batchaug.RandScaleIntensityFixedMean(prob=1.0, factors=(0.5, 0.5), channel_wise=True)
        out = t(inp)
        for c in range(3):
            assert torch.allclose(out[0, c].mean(), inp[0, c].mean(), atol=1e-5)


class TestRandScaleIntensityFixedMeanBfloat16:
    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandScaleIntensityFixedMean(prob=1.0)
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()
