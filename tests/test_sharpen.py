import torch
import monai.transforms

import batchaug


class TestRandGaussianSharpenMatchesMONAI:
    """B=1: batchaug unsharp mask matches MONAI's GaussianSharpen."""

    def test_matches_monai(self, vol, device):
        sigma1 = (0.8, 0.7, 0.9)
        sigma2 = (0.5, 0.4, 0.6)
        alpha = 15.0

        input_4d = vol[0]
        input_5d = vol[:1]

        monai_t = monai.transforms.GaussianSharpen(
            sigma1=sigma1, sigma2=sigma2, alpha=alpha
        )
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandGaussianSharpen(
            prob=1.0,
            sigma1_x=(sigma1[0], sigma1[0]),
            sigma1_y=(sigma1[1], sigma1[1]),
            sigma1_z=(sigma1[2], sigma1[2]),
            sigma2_x=(sigma2[0], sigma2[0]),
            sigma2_y=(sigma2[1], sigma2[1]),
            sigma2_z=(sigma2[2], sigma2[2]),
            alpha=(alpha, alpha),
        )
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-3)

    def test_no_change_when_mask_false(self, vol, device):
        ba_t = batchaug.RandGaussianSharpen(prob=1.0)
        params = ba_t.sample_params(1, vol[:1].shape, device)
        params["mask"] = torch.tensor([False], device=device)
        ba_out = ba_t.apply(vol[:1], params)
        assert torch.equal(ba_out, vol[:1])


class TestRandGaussianSharpenBatchIndependence:
    """B>1: each batch element gets different sharpening."""

    def test_different_sharpen_per_element(self, vol, device):
        t = batchaug.RandGaussianSharpen(prob=1.0)
        result = t(vol)
        diffs = [
            not torch.equal(result[0], result[i]) for i in range(1, vol.shape[0])
        ]
        assert any(diffs)


class TestRandGaussianSharpenChannelConsistency:
    """Same sharpening is applied to all channels within a batch element."""

    def test_identical_channels_stay_identical(self, device):
        base = torch.rand(2, 1, 16, 16, 16, device=device)
        vol = base.repeat(1, 4, 1, 1, 1)

        t = batchaug.RandGaussianSharpen(
            prob=1.0,
            sigma1_x=(0.8, 0.8),
            sigma1_y=(0.8, 0.8),
            sigma1_z=(0.8, 0.8),
            sigma2_x=(0.5, 0.5),
            sigma2_y=(0.5, 0.5),
            sigma2_z=(0.5, 0.5),
            alpha=(15.0, 15.0),
        )
        result = t(vol)

        for b in range(2):
            for c in range(1, 4):
                assert torch.allclose(result[b, 0], result[b, c], atol=1e-5)


class TestRandGaussianSharpenSigma2DependsOnSigma1:
    """When sigma2 is a scalar, it uses [scalar, sigma1] as the range."""

    def test_sigma2_bounded_by_sigma1(self, device):
        t = batchaug.RandGaussianSharpen(
            prob=1.0,
            sigma1_x=(0.6, 0.6),
            sigma2_x=0.3,  # scalar → range [0.3, sigma1_x=0.6]
        )
        params = t.sample_params(200, (200, 1, 8, 8, 8), device)
        assert (params["sigma2_x"] >= 0.3 - 1e-6).all()
        assert (params["sigma2_x"] <= 0.6 + 1e-6).all()


class TestRandGaussianSharpenBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandGaussianSharpen(prob=1.0)
        result = t(vol_bf16)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
