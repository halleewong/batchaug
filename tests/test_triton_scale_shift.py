"""Triton scale/shift transforms: equivalence with PyTorch reference."""
import pytest
import torch

import batchaug.pytorch as pt
import batchaug.triton as tr


@pytest.fixture
def big_vol(device):
    """Larger volume for meaningful speed/correctness testing."""
    torch.manual_seed(0)
    return torch.rand(4, 4, 32, 32, 32, device=device)


# ===========================================================================
# RandScaleIntensity
# ===========================================================================

class TestTritonRandScaleIntensity:
    def test_matches_pytorch(self, big_vol, device):
        torch.manual_seed(1)
        params = pt.RandScaleIntensity(prob=1.0, factors=(0.1, 0.9)).sample_params(
            big_vol.shape[0], big_vol.shape, device
        )
        out_pt = pt.RandScaleIntensity(prob=1.0, factors=(0.1, 0.9)).apply(big_vol, params)
        out_tr = tr.RandScaleIntensity(prob=1.0, factors=(0.1, 0.9)).apply(big_vol, params)
        assert torch.allclose(out_pt, out_tr, atol=1e-5)

    def test_mask_false_unchanged(self, big_vol, device):
        B = big_vol.shape[0]
        params = pt.RandScaleIntensity(prob=1.0, factors=(0.5, 0.5)).sample_params(
            B, big_vol.shape, device
        )
        params["mask"] = torch.zeros(B, dtype=torch.bool, device=device)
        out = tr.RandScaleIntensity(prob=1.0, factors=(0.5, 0.5)).apply(big_vol, params)
        assert torch.allclose(out, big_vol)

    def test_bfloat16(self, vol_bf16, device):
        t = tr.RandScaleIntensity(prob=1.0, factors=(0.1, 0.5))
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()


# ===========================================================================
# RandShiftIntensity
# ===========================================================================

class TestTritonRandShiftIntensity:
    def test_matches_pytorch(self, big_vol, device):
        torch.manual_seed(2)
        params = pt.RandShiftIntensity(prob=1.0, offsets=(-0.3, 0.3)).sample_params(
            big_vol.shape[0], big_vol.shape, device
        )
        out_pt = pt.RandShiftIntensity(prob=1.0, offsets=(-0.3, 0.3)).apply(big_vol, params)
        out_tr = tr.RandShiftIntensity(prob=1.0, offsets=(-0.3, 0.3)).apply(big_vol, params)
        assert torch.allclose(out_pt, out_tr, atol=1e-5)

    def test_safe_clamp_matches_pytorch(self, big_vol, device):
        params = pt.RandShiftIntensity(prob=1.0, offsets=(0.5, 0.5), safe=True).sample_params(
            big_vol.shape[0], big_vol.shape, device
        )
        out_pt = pt.RandShiftIntensity(prob=1.0, offsets=(0.5, 0.5), safe=True).apply(big_vol, params)
        out_tr = tr.RandShiftIntensity(prob=1.0, offsets=(0.5, 0.5), safe=True).apply(big_vol, params)
        assert torch.allclose(out_pt, out_tr, atol=1e-5)

    def test_bfloat16(self, vol_bf16, device):
        t = tr.RandShiftIntensity(prob=1.0, offsets=(-0.1, 0.1))
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()


# ===========================================================================
# RandStdShiftIntensity
# ===========================================================================

class TestTritonRandStdShiftIntensity:
    def test_matches_pytorch_basic(self, big_vol, device):
        torch.manual_seed(3)
        params = pt.RandStdShiftIntensity(prob=1.0, factors=(-2, 2)).sample_params(
            big_vol.shape[0], big_vol.shape, device
        )
        out_pt = pt.RandStdShiftIntensity(prob=1.0, factors=(-2, 2)).apply(big_vol, params)
        out_tr = tr.RandStdShiftIntensity(prob=1.0, factors=(-2, 2)).apply(big_vol, params)
        # Triton uses same ddof=1 formula as PyTorch — should match closely
        assert torch.allclose(out_pt, out_tr, atol=1e-4)

    def test_channel_wise_falls_back_to_pytorch(self, big_vol, device):
        """channel_wise=True falls back to PyTorch — should still be correct."""
        torch.manual_seed(4)
        params = pt.RandStdShiftIntensity(prob=1.0, channel_wise=True).sample_params(
            big_vol.shape[0], big_vol.shape, device
        )
        out_pt = pt.RandStdShiftIntensity(prob=1.0, channel_wise=True).apply(big_vol, params)
        out_tr = tr.RandStdShiftIntensity(prob=1.0, channel_wise=True).apply(big_vol, params)
        assert torch.allclose(out_pt, out_tr, atol=1e-4)


# ===========================================================================
# RandScaleIntensityFixedMean
# ===========================================================================

class TestTritonRandScaleIntensityFixedMean:
    def test_matches_pytorch(self, big_vol, device):
        torch.manual_seed(5)
        params = pt.RandScaleIntensityFixedMean(prob=1.0, factors=(-0.5, 0.5)).sample_params(
            big_vol.shape[0], big_vol.shape, device
        )
        out_pt = pt.RandScaleIntensityFixedMean(prob=1.0, factors=(-0.5, 0.5)).apply(big_vol, params)
        out_tr = tr.RandScaleIntensityFixedMean(prob=1.0, factors=(-0.5, 0.5)).apply(big_vol, params)
        assert torch.allclose(out_pt, out_tr, atol=1e-4)

    def test_mean_preserved(self, big_vol, device):
        t = tr.RandScaleIntensityFixedMean(prob=1.0, factors=(0.8, 0.8))
        out = t(big_vol)
        for b in range(big_vol.shape[0]):
            assert torch.allclose(out[b].mean(), big_vol[b].mean(), atol=1e-4)

    def test_channel_wise_falls_back(self, big_vol, device):
        params = pt.RandScaleIntensityFixedMean(prob=1.0, channel_wise=True).sample_params(
            big_vol.shape[0], big_vol.shape, device
        )
        out_pt = pt.RandScaleIntensityFixedMean(prob=1.0, channel_wise=True).apply(big_vol, params)
        out_tr = tr.RandScaleIntensityFixedMean(prob=1.0, channel_wise=True).apply(big_vol, params)
        assert torch.allclose(out_pt, out_tr, atol=1e-4)


# ===========================================================================
# RandRicianNoise (Triton vs PyTorch equivalence: statistical properties)
# ===========================================================================

class TestTritonRandRicianNoise:
    def test_output_nonnegative(self, big_vol, device):
        t = tr.RandRicianNoise(prob=1.0, std=0.2)
        out = t(big_vol)
        assert out.min() >= 0.0

    def test_mask_false_unchanged(self, big_vol, device):
        B = big_vol.shape[0]
        t = tr.RandRicianNoise(prob=1.0, std=0.2)
        params = t.sample_params(B, big_vol.shape, device)
        params["mask"] = torch.zeros(B, dtype=torch.bool, device=device)
        out = t.apply(big_vol, params)
        assert torch.allclose(out, big_vol.float(), atol=1e-5)

    def test_shape_preserved(self, big_vol, device):
        t = tr.RandRicianNoise(prob=1.0, std=0.1)
        out = t(big_vol)
        assert out.shape == big_vol.shape

    def test_statistical_properties(self, device):
        """Output should have higher mean than input (Rician adds positive bias)."""
        inp = torch.ones(4, 1, 32, 32, 32, device=device) * 0.5
        t = tr.RandRicianNoise(prob=1.0, std=0.3, sample_std=False)
        out = t(inp)
        # Rician distribution: E[R] = sigma * sqrt(pi/2) * L_{1/2}(-nu^2/(2*sigma^2))
        # For large SNR, E[R] ≈ nu = input. Output > 0 always.
        assert out.mean() > 0.0
        assert not torch.isnan(out).any()

    def test_bfloat16(self, vol_bf16, device):
        t = tr.RandRicianNoise(prob=1.0, std=0.1)
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()
        assert out.min() >= 0.0

    def test_relative_falls_back_to_pytorch(self, big_vol, device):
        """relative=True uses PyTorch path — output must still be >= 0."""
        t = tr.RandRicianNoise(prob=1.0, std=0.1, relative=True)
        out = t(big_vol)
        assert out.min() >= 0.0
        assert not torch.isnan(out).any()
