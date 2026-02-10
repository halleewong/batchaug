import torch
import monai.transforms

import batchaug


class TestRandBiasFieldMatchesMONAI:
    """B=1: batchaug bias field matches MONAI's RandBiasField."""

    def test_matches_monai_fixed_coeffs(self, vol, device):
        """Force identical coefficients and compare outputs."""
        input_4d = vol[0]  # (C, H, W, D)
        input_5d = vol[:1]  # (1, C, H, W, D)

        degree = 3
        coeff_val = 0.3

        # --- MONAI ---
        monai_t = monai.transforms.RandBiasField(
            degree=degree, coeff_range=(coeff_val, coeff_val), prob=1.0
        )
        monai_t.randomize(img_size=input_4d.shape[1:])
        monai_out = monai_t(input_4d.cpu().numpy(), randomize=False)
        monai_out = torch.as_tensor(monai_out).to(device)

        # --- batchaug ---
        ba_t = batchaug.RandBiasField(
            prob=1.0, degree=degree, coeff_range=(coeff_val, coeff_val)
        )
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-4), (
            f"max diff: {(ba_out[0] - monai_out).abs().max().item()}"
        )

    def test_no_change_when_mask_false(self, vol, device):
        ba_t = batchaug.RandBiasField(prob=1.0, degree=3)
        params = ba_t.sample_params(1, vol[:1].shape, device)
        params["mask"] = torch.tensor([False], device=device)
        ba_out = ba_t.apply(vol[:1], params)
        assert torch.equal(ba_out, vol[:1])


class TestRandBiasFieldBatchIndependence:
    """B>1: each batch element gets different bias field."""

    def test_different_field_per_element(self, vol, device):
        t = batchaug.RandBiasField(prob=1.0, degree=3, coeff_range=(0.0, 0.5))
        result = t(vol)
        # Each element should differ from original by different amounts
        diffs = (result - vol).abs().flatten(1).mean(dim=1)
        assert diffs.max() - diffs.min() > 1e-4


class TestRandBiasFieldChannelConsistency:
    """Same bias field is applied to all channels within a batch element."""

    def test_identical_channels_get_same_field(self, device):
        base = torch.rand(2, 1, 16, 16, 16, device=device) + 0.5  # avoid near-zero
        vol = base.repeat(1, 4, 1, 1, 1)

        t = batchaug.RandBiasField(
            prob=1.0, degree=3, coeff_range=(0.3, 0.3)
        )
        result = t(vol)

        for b in range(2):
            # All channels should have same multiplicative factor
            ratios = result[b] / vol[b]
            for c in range(1, 4):
                assert torch.allclose(ratios[0], ratios[c], atol=1e-5)


class TestRandBiasFieldNonCubic:
    """Non-cubic spatial shapes (H != W != D)."""

    def test_nonsquare_preserves_shape(self, vol_nonsquare, device):
        t = batchaug.RandBiasField(prob=1.0, degree=3, coeff_range=(0.0, 0.3))
        result = t(vol_nonsquare)
        assert result.shape == vol_nonsquare.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_nonsquare_matches_monai(self, device):
        """Bias field matches MONAI on non-cubic (1, 12, 16, 20)."""
        torch.manual_seed(0)
        input_4d = torch.rand(1, 12, 16, 20, device=device)
        input_5d = input_4d.unsqueeze(0)

        degree = 3
        coeff_val = 0.3

        monai_t = monai.transforms.RandBiasField(
            degree=degree, coeff_range=(coeff_val, coeff_val), prob=1.0
        )
        monai_t.randomize(img_size=input_4d.shape[1:])
        monai_out = monai_t(input_4d.cpu().numpy(), randomize=False)
        monai_out = torch.as_tensor(monai_out).to(device)

        ba_t = batchaug.RandBiasField(
            prob=1.0, degree=degree, coeff_range=(coeff_val, coeff_val)
        )
        ba_out = ba_t(input_5d)

        assert ba_out.shape == input_5d.shape
        assert torch.allclose(ba_out[0], monai_out, atol=1e-4), (
            f"max diff: {(ba_out[0] - monai_out).abs().max().item()}"
        )


class TestRandBiasFieldBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandBiasField(prob=1.0, degree=3, coeff_range=(0.0, 0.1))
        result = t(vol_bf16)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
