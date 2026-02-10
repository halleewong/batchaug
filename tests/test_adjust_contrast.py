import torch
import monai.transforms

import batchaug


class TestRandAdjustContrastMatchesMONAI:
    """B=1: batchaug gamma correction matches MONAI's AdjustContrast."""

    def test_matches_monai_gamma_1_5(self, vol, device):
        gamma = 1.5
        input_4d = vol[0]  # (C, H, W, D)
        input_5d = vol[:1]  # (1, C, H, W, D)

        monai_t = monai.transforms.AdjustContrast(gamma=gamma)
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandAdjustContrast(prob=1.0, gamma=(gamma, gamma))
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-5)

    def test_matches_monai_gamma_0_7(self, vol, device):
        gamma = 0.7
        input_4d = vol[0]
        input_5d = vol[:1]

        monai_t = monai.transforms.AdjustContrast(gamma=gamma)
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandAdjustContrast(prob=1.0, gamma=(gamma, gamma))
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-5)

    def test_no_change_when_mask_false(self, vol, device):
        ba_t = batchaug.RandAdjustContrast(prob=1.0, gamma=(2.0, 2.0))
        params = ba_t.sample_params(1, vol[:1].shape, device)
        params["mask"] = torch.tensor([False], device=device)
        ba_out = ba_t.apply(vol[:1], params)
        assert torch.equal(ba_out, vol[:1])


class TestRandAdjustContrastBatchIndependence:
    """B>1: each batch element gets a different gamma."""

    def test_different_gamma_per_element(self, vol, device):
        t = batchaug.RandAdjustContrast(prob=1.0, gamma=(0.5, 4.5))
        result = t(vol)
        # At least two elements should differ
        diffs = [
            not torch.equal(result[0], result[i]) for i in range(1, vol.shape[0])
        ]
        assert any(diffs)


class TestRandAdjustContrastChannelConsistency:
    """Same gamma is applied to all channels within a batch element."""

    def test_identical_channels_stay_identical(self, device):
        base = torch.rand(2, 1, 16, 16, 16, device=device)
        vol = base.repeat(1, 4, 1, 1, 1)  # all channels identical

        t = batchaug.RandAdjustContrast(prob=1.0, gamma=(2.0, 2.0))
        result = t(vol)

        for b in range(2):
            for c in range(1, 4):
                assert torch.allclose(result[b, 0], result[b, c], atol=1e-6)


class TestRandAdjustContrastNonCubic:
    """Non-cubic spatial shapes (H != W != D)."""

    def test_matches_monai_nonsquare(self, device):
        """Gamma correction matches MONAI on non-cubic (1, 12, 16, 20)."""
        gamma = 1.5
        torch.manual_seed(0)
        input_4d = torch.rand(1, 12, 16, 20, device=device)
        input_5d = input_4d.unsqueeze(0)

        monai_t = monai.transforms.AdjustContrast(gamma=gamma)
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandAdjustContrast(prob=1.0, gamma=(gamma, gamma))
        ba_out = ba_t(input_5d)

        assert ba_out.shape == input_5d.shape
        assert torch.allclose(ba_out[0], monai_out, atol=1e-5)

    def test_nonsquare_preserves_shape(self, vol_nonsquare, device):
        t = batchaug.RandAdjustContrast(prob=1.0, gamma=(0.5, 2.0))
        result = t(vol_nonsquare)
        assert result.shape == vol_nonsquare.shape
        assert not torch.isnan(result).any()


class TestRandAdjustContrastBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandAdjustContrast(prob=1.0, gamma=(1.5, 1.5))
        result = t(vol_bf16)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
