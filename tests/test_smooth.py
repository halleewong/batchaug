import torch
import monai.transforms

import batchaug


class TestRandGaussianSmoothMatchesMONAI:
    """B=1: batchaug Gaussian blur matches MONAI's GaussianSmooth."""

    def test_matches_monai_uniform_sigma(self, vol, device):
        sigma = (0.8, 0.8, 0.8)
        input_4d = vol[0]
        input_5d = vol[:1]

        monai_t = monai.transforms.GaussianSmooth(sigma=sigma)
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandGaussianSmooth(
            prob=1.0,
            sigma_x=(sigma[0], sigma[0]),
            sigma_y=(sigma[1], sigma[1]),
            sigma_z=(sigma[2], sigma[2]),
            padding_mode=None,  # match MONAI's zero-padding (BatchAug defaults to "reflect")
        )
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-4)

    def test_matches_monai_different_sigmas(self, vol, device):
        sigma = (0.5, 1.0, 0.7)
        input_4d = vol[0]
        input_5d = vol[:1]

        monai_t = monai.transforms.GaussianSmooth(sigma=sigma)
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandGaussianSmooth(
            prob=1.0,
            sigma_x=(sigma[0], sigma[0]),
            sigma_y=(sigma[1], sigma[1]),
            sigma_z=(sigma[2], sigma[2]),
            padding_mode=None,  # match MONAI's zero-padding (BatchAug defaults to "reflect")
        )
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-4)

    def test_no_change_when_mask_false(self, vol, device):
        ba_t = batchaug.RandGaussianSmooth(prob=1.0)
        params = ba_t.sample_params(1, vol[:1].shape, device)
        params["mask"] = torch.tensor([False], device=device)
        ba_out = ba_t.apply(vol[:1], params)
        assert torch.equal(ba_out, vol[:1])


class TestRandGaussianSmoothBatchIndependence:
    """B>1: each batch element gets different blur."""

    def test_different_blur_per_element(self, vol, device):
        t = batchaug.RandGaussianSmooth(prob=1.0)
        result = t(vol)
        stds = result.flatten(1).std(dim=1)
        # Different sigmas → different amounts of smoothing
        assert stds.max() - stds.min() > 1e-4


class TestRandGaussianSmoothChannelConsistency:
    """Same blur kernel is applied to all channels within a batch element."""

    def test_identical_channels_stay_identical(self, device):
        base = torch.rand(2, 1, 16, 16, 16, device=device)
        vol = base.repeat(1, 4, 1, 1, 1)

        t = batchaug.RandGaussianSmooth(
            prob=1.0,
            sigma_x=(0.8, 0.8),
            sigma_y=(0.8, 0.8),
            sigma_z=(0.8, 0.8),
        )
        result = t(vol)

        for b in range(2):
            for c in range(1, 4):
                assert torch.allclose(result[b, 0], result[b, c], atol=1e-6)


class TestRandGaussianSmoothNonCubic:
    """Non-cubic spatial shapes (H != W != D)."""

    def test_matches_monai_nonsquare(self, device):
        """Gaussian blur matches MONAI on non-cubic (1, 12, 16, 20)."""
        sigma = (0.8, 0.8, 0.8)
        torch.manual_seed(0)
        input_4d = torch.rand(1, 12, 16, 20, device=device)
        input_5d = input_4d.unsqueeze(0)

        monai_t = monai.transforms.GaussianSmooth(sigma=sigma)
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandGaussianSmooth(
            prob=1.0,
            sigma_x=(sigma[0], sigma[0]),
            sigma_y=(sigma[1], sigma[1]),
            sigma_z=(sigma[2], sigma[2]),
            padding_mode=None,  # match MONAI's zero-padding (BatchAug defaults to "reflect")
        )
        ba_out = ba_t(input_5d)

        assert ba_out.shape == input_5d.shape
        assert torch.allclose(ba_out[0], monai_out, atol=1e-4)

    def test_nonsquare_preserves_shape(self, vol_nonsquare, device):
        t = batchaug.RandGaussianSmooth(prob=1.0)
        result = t(vol_nonsquare)
        assert result.shape == vol_nonsquare.shape
        assert not torch.isnan(result).any()


class TestRandGaussianSmoothBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandGaussianSmooth(prob=1.0)
        result = t(vol_bf16)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestRandGaussianSmoothDefaultPadding:
    """BatchAug defaults to padding_mode='reflect' (not MONAI's zero-padding)."""

    def test_default_is_reflect(self):
        assert batchaug.RandGaussianSmooth(prob=1.0).padding_mode == "reflect"

    def test_default_matches_explicit_reflect(self, vol, device):
        sigma = (0.8, 0.8, 0.8)
        kw = dict(
            prob=1.0,
            sigma_x=(sigma[0], sigma[0]),
            sigma_y=(sigma[1], sigma[1]),
            sigma_z=(sigma[2], sigma[2]),
        )
        out_default = batchaug.RandGaussianSmooth(**kw)(vol)
        out_reflect = batchaug.RandGaussianSmooth(padding_mode="reflect", **kw)(vol)
        assert torch.equal(out_default, out_reflect)

    def test_default_differs_from_monai_at_edges(self, vol, device):
        """Reflect default and MONAI's zero-padding must diverge near boundaries."""
        sigma = (0.8, 0.8, 0.8)
        kw = dict(
            prob=1.0,
            sigma_x=(sigma[0], sigma[0]),
            sigma_y=(sigma[1], sigma[1]),
            sigma_z=(sigma[2], sigma[2]),
        )
        out_default = batchaug.RandGaussianSmooth(**kw)(vol)
        out_monai = batchaug.RandGaussianSmooth(padding_mode=None, **kw)(vol)
        assert not torch.allclose(out_default, out_monai, atol=1e-4)
