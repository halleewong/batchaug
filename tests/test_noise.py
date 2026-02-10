import pytest
import torch
import monai.transforms

import batchaug


class TestRandGaussianNoiseMatchesMONAI:
    """B=1: batchaug noise formula matches MONAI's additive noise."""

    def test_additive_formula(self, device):
        """Verify output = input + noise (same formula as MONAI)."""
        torch.manual_seed(0)
        input_5d = torch.rand(1, 3, 16, 16, 16, device=device)

        noise = torch.randn(1, 3, 16, 16, 16, device=device) * 0.5

        ba_t = batchaug.RandGaussianNoise(prob=1.0, std=0.5)
        params = {
            "mask": torch.tensor([True], device=device),
            "std": torch.tensor([0.5], device=device),
            "mean": torch.tensor([0.0], device=device),
            "noise": noise,
        }
        ba_out = ba_t.apply(input_5d, params)

        expected = input_5d + noise
        assert torch.allclose(ba_out, expected, atol=1e-6)

    def test_no_noise_when_mask_false(self, device):
        """When mask is False, output equals input."""
        torch.manual_seed(0)
        input_5d = torch.rand(1, 3, 16, 16, 16, device=device)

        ba_t = batchaug.RandGaussianNoise(prob=1.0, std=0.5)
        params = {
            "mask": torch.tensor([False], device=device),
            "std": torch.tensor([0.5], device=device),
            "mean": torch.tensor([0.0], device=device),
            "noise": torch.randn(1, 3, 16, 16, 16, device=device) * 0.5,
        }
        ba_out = ba_t.apply(input_5d, params)

        assert torch.allclose(ba_out, input_5d, atol=1e-6)


class TestRandGaussianNoiseScalarParams:
    """Scalar mean/std produce fixed values for all batch elements."""

    def test_fixed_std(self, device):
        ba_t = batchaug.RandGaussianNoise(prob=1.0, std=0.3)
        params = ba_t.sample_params(8, (8, 3, 8, 8, 8), device)
        assert torch.allclose(params["std"], torch.tensor(0.3, device=device))

    def test_fixed_mean(self, device):
        ba_t = batchaug.RandGaussianNoise(prob=1.0, mean=0.5, std=0.1)
        params = ba_t.sample_params(8, (8, 3, 8, 8, 8), device)
        assert torch.allclose(params["mean"], torch.tensor(0.5, device=device))


class TestRandGaussianNoiseTupleParams:
    """Tuple mean/std sample from U(low, high) per batch element."""

    def test_tuple_std_range(self, device):
        ba_t = batchaug.RandGaussianNoise(prob=1.0, std=(0.1, 0.5))
        params = ba_t.sample_params(200, (200, 3, 4, 4, 4), device)
        assert (params["std"] >= 0.1).all()
        assert (params["std"] <= 0.5).all()
        # Should not all be the same value
        assert params["std"].std() > 0.01

    def test_tuple_mean_range(self, device):
        ba_t = batchaug.RandGaussianNoise(prob=1.0, mean=(-1.0, 1.0), std=0.01)
        params = ba_t.sample_params(200, (200, 3, 4, 4, 4), device)
        assert (params["mean"] >= -1.0).all()
        assert (params["mean"] <= 1.0).all()
        assert params["mean"].std() > 0.1

    def test_tuple_mean_shifts_noise(self, device):
        """Tuple mean should produce per-element bias in the noise."""
        tensor = torch.zeros(4, 1, 32, 32, 32, device=device)

        ba_t = batchaug.RandGaussianNoise(prob=1.0, mean=(5.0, 5.0), std=0.01)
        result = ba_t(tensor)

        # All noise should be centered near 5.0
        for b in range(4):
            assert abs(result[b].mean().item() - 5.0) < 0.1


class TestRandGaussianNoiseBatchIndependence:
    """B>1: each batch element gets independent noise."""

    def test_different_noise_per_element(self, device):
        tensor = torch.zeros(4, 2, 16, 16, 16, device=device)
        t = batchaug.RandGaussianNoise(prob=1.0, std=1.0)
        result = t(tensor)
        for i in range(1, 4):
            assert not torch.equal(result[0], result[i])

    def test_prob_mask_independence(self, device):
        tensor = torch.zeros(8, 2, 8, 8, 8, device=device)
        t = batchaug.RandGaussianNoise(prob=0.5, std=1.0)
        result = t(tensor)
        element_norms = result.flatten(1).norm(dim=1)
        assert element_norms.max() > 0  # at least one noisy


class TestRandGaussianNoiseChannelConsistency:
    """Same std/mean parameters are used for all channels within a batch element."""

    def test_params_shape_is_per_batch(self, device):
        t = batchaug.RandGaussianNoise(prob=1.0, std=(0.0, 0.5), mean=(-1.0, 1.0))
        params = t.sample_params(4, (4, 3, 16, 16, 16), device)
        assert params["std"].shape == (4,)
        assert params["mean"].shape == (4,)

    def test_channels_share_std(self, device):
        tensor = torch.zeros(2, 8, 32, 32, 32, device=device)
        t = batchaug.RandGaussianNoise(prob=1.0, std=1.0)
        result = t(tensor)
        for b in range(2):
            channel_stds = [result[b, c].std().item() for c in range(8)]
            mean_std = sum(channel_stds) / len(channel_stds)
            for s in channel_stds:
                assert abs(s - mean_std) / mean_std < 0.15


class TestRandGaussianNoiseBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandGaussianNoise(prob=1.0, std=0.1)
        result = t(vol_bf16)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
