"""Tests for RandRicianNoise."""
import pytest
import torch

import batchaug
import batchaug.pytorch as pt


class TestRandRicianNoiseFormula:
    """B=1: verify output = sqrt((input + n1)^2 + n2^2) using PyTorch backend."""

    def test_rician_formula(self, device):
        inp = torch.rand(1, 2, 8, 8, 8, device=device)
        # Unit Gaussians — apply() will scale by noise_std
        u1 = torch.randn_like(inp)
        u2 = torch.randn_like(inp)
        noise_std = 0.1
        # Expected: n1 = mean + std * u1, n2 = mean + std * u2
        n1 = noise_std * u1
        n2 = noise_std * u2
        expected = ((inp + n1).pow(2) + n2.pow(2)).sqrt()

        # Use PyTorch backend directly (unit noise tensors are used there)
        t = pt.RandRicianNoise(prob=1.0, mean=0.0, std=noise_std, sample_std=False)
        params = {
            "mask": torch.tensor([True], device=device),
            "noise_std": torch.tensor([noise_std], device=device),
            "u1": u1,
            "u2": u2,
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_mask_false_unchanged(self, device):
        inp = torch.rand(1, 1, 8, 8, 8, device=device)
        t = pt.RandRicianNoise(prob=1.0, std=0.1)
        params = {
            "mask": torch.tensor([False], device=device),
            "noise_std": torch.tensor([0.1], device=device),
            "u1": torch.randn_like(inp),
            "u2": torch.randn_like(inp),
        }
        out = t.apply(inp, params)
        assert torch.allclose(out, inp)

    def test_output_always_nonnegative(self, device):
        """Rician output is always >= 0 (it's a magnitude)."""
        inp = torch.rand(4, 2, 16, 16, 16, device=device)
        t = batchaug.RandRicianNoise(prob=1.0, std=0.5)
        out = t(inp)
        assert out.min() >= 0.0


class TestRandRicianNoiseSampling:
    def test_fixed_std_no_sampling(self, device):
        t = batchaug.RandRicianNoise(prob=1.0, std=0.2, sample_std=False)
        params = t.sample_params(8, (8, 1, 4, 4, 4), device)
        assert torch.allclose(params["noise_std"], torch.tensor(0.2, device=device))

    def test_sample_std_in_range(self, device):
        t = batchaug.RandRicianNoise(prob=1.0, std=0.5, sample_std=True)
        params = t.sample_params(200, (200, 1, 4, 4, 4), device)
        assert (params["noise_std"] >= 0.0).all()
        assert (params["noise_std"] <= 0.5).all()
        assert params["noise_std"].std() > 0.01

    def test_unit_noise_tensors_shape(self, device):
        B, C, H, W, D = 4, 2, 8, 8, 8
        t = batchaug.RandRicianNoise(prob=1.0, std=0.1)
        params = t.sample_params(B, (B, C, H, W, D), device)
        assert params["u1"].shape == (B, C, H, W, D)
        assert params["u2"].shape == (B, C, H, W, D)


class TestRandRicianNoiseRelative:
    def test_relative_scales_with_signal_std(self, device):
        """With relative=True, noisy samples have higher SNR for strong signals."""
        torch.manual_seed(42)
        B = 4
        # Half the batch has 10x larger signal amplitude
        inp_weak = torch.rand(B, 1, 16, 16, 16, device=device) * 0.1
        inp_strong = torch.rand(B, 1, 16, 16, 16, device=device) * 1.0

        t = batchaug.RandRicianNoise(prob=1.0, std=0.1, relative=True, sample_std=False)
        out_weak = t(inp_weak)
        out_strong = t(inp_strong)

        # Strong signal should have larger absolute shift (relative noise is proportional)
        diff_strong = (out_strong - inp_strong).abs().mean()
        diff_weak = (out_weak - inp_weak).abs().mean()
        assert diff_strong > diff_weak


class TestRandRicianNoiseBatchIndependence:
    def test_different_noise_per_element(self, device):
        inp = torch.ones(4, 1, 16, 16, 16, device=device)
        t = batchaug.RandRicianNoise(prob=1.0, std=0.5)
        out = t(inp)
        # Elements should not all be the same
        for i in range(1, 4):
            assert not torch.allclose(out[0], out[i], atol=1e-4)


class TestRandRicianNoiseDict:
    def test_dict_same_noise(self, device):
        """Both keys should use the same unit noise tensors."""
        torch.manual_seed(0)
        vol = torch.rand(2, 1, 8, 8, 8, device=device)
        seg = vol.clone()  # identical input

        t = batchaug.RandRicianNoised(keys=["vol", "seg"], prob=1.0, std=0.1, sample_std=False)
        out = t({"vol": vol, "seg": seg})
        assert torch.allclose(out["vol"], out["seg"], atol=1e-6)


class TestRandRicianNoiseBfloat16:
    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandRicianNoise(prob=1.0, std=0.1)
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert out.min() >= 0.0
