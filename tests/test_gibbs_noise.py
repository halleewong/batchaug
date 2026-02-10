import torch
import monai.transforms

import batchaug


class TestRandGibbsNoiseMatchesMONAI:
    """B=1: batchaug Gibbs noise matches MONAI's GibbsNoise."""

    def test_matches_monai_alpha_05(self, vol, device):
        alpha = 0.5
        input_4d = vol[0]  # (C, H, W, D)
        input_5d = vol[:1]  # (1, C, H, W, D)

        monai_t = monai.transforms.GibbsNoise(alpha=alpha)
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandGibbsNoise(prob=1.0, alpha=(alpha, alpha))
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-5), (
            f"max diff: {(ba_out[0] - monai_out).abs().max().item()}"
        )

    def test_matches_monai_alpha_02(self, vol, device):
        alpha = 0.2
        input_4d = vol[0]
        input_5d = vol[:1]

        monai_t = monai.transforms.GibbsNoise(alpha=alpha)
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandGibbsNoise(prob=1.0, alpha=(alpha, alpha))
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-5), (
            f"max diff: {(ba_out[0] - monai_out).abs().max().item()}"
        )

    def test_no_change_when_mask_false(self, vol, device):
        ba_t = batchaug.RandGibbsNoise(prob=1.0, alpha=(0.5, 0.5))
        params = ba_t.sample_params(1, vol[:1].shape, device)
        params["mask"] = torch.tensor([False], device=device)
        ba_out = ba_t.apply(vol[:1], params)
        assert torch.equal(ba_out, vol[:1])


class TestRandGibbsNoiseBatchIndependence:
    """B>1: each batch element gets different Gibbs artifact."""

    def test_different_artifact_per_element(self, vol, device):
        t = batchaug.RandGibbsNoise(prob=1.0, alpha=(0.1, 0.9))
        result = t(vol)
        # Different alphas → different amounts of ringing
        diffs = (result - vol).abs().flatten(1).mean(dim=1)
        assert diffs.max() - diffs.min() > 1e-4


class TestRandGibbsNoiseChannelConsistency:
    """Same k-space mask applied to all channels."""

    def test_identical_channels_stay_consistent(self, device):
        base = torch.rand(2, 1, 16, 16, 16, device=device)
        vol = base.repeat(1, 4, 1, 1, 1)

        t = batchaug.RandGibbsNoise(prob=1.0, alpha=(0.5, 0.5))
        result = t(vol)

        for b in range(2):
            for c in range(1, 4):
                assert torch.allclose(result[b, 0], result[b, c], atol=1e-6)


class TestRandGibbsNoiseAlphaMonotonicity:
    """Larger alpha → more distortion."""

    def test_higher_alpha_more_distortion(self, vol, device):
        low_t = batchaug.RandGibbsNoise(prob=1.0, alpha=(0.1, 0.1))
        high_t = batchaug.RandGibbsNoise(prob=1.0, alpha=(0.8, 0.8))

        diff_low = (low_t(vol) - vol).abs().mean().item()
        diff_high = (high_t(vol) - vol).abs().mean().item()
        assert diff_high > diff_low


class TestRandGibbsNoiseBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandGibbsNoise(prob=1.0, alpha=(0.3, 0.7))
        result = t(vol_bf16)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
