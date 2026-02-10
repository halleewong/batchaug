import pytest
import torch
import monai.transforms

import batchaug


class TestScaleIntensityMatchesMONAI:
    """B=1: batchaug output matches MONAI with channel_wise=True."""

    def test_default_0_1(self, device):
        """Default minv=0, maxv=1 rescaling matches MONAI channel_wise."""
        torch.manual_seed(0)
        input_4d = torch.rand(3, 16, 16, 16, device=device)
        input_5d = input_4d.unsqueeze(0)

        monai_t = monai.transforms.ScaleIntensity(
            minv=0.0, maxv=1.0, channel_wise=True
        )
        monai_out = monai_t(input_4d)

        ba_t = batchaug.ScaleIntensity(minv=0.0, maxv=1.0)
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-6)

    def test_custom_range(self, device):
        """Rescaling to [-1, 1] matches MONAI channel_wise."""
        torch.manual_seed(1)
        input_4d = torch.rand(3, 16, 16, 16, device=device)
        input_5d = input_4d.unsqueeze(0)

        monai_t = monai.transforms.ScaleIntensity(
            minv=-1.0, maxv=1.0, channel_wise=True
        )
        monai_out = monai_t(input_4d)

        ba_t = batchaug.ScaleIntensity(minv=-1.0, maxv=1.0)
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-6)

    def test_factor(self, device):
        """Factor-based scaling matches MONAI (no min/max, no channel issue)."""
        torch.manual_seed(2)
        input_4d = torch.rand(3, 16, 16, 16, device=device)
        input_5d = input_4d.unsqueeze(0)

        monai_t = monai.transforms.ScaleIntensity(minv=None, maxv=None, factor=0.5)
        monai_out = monai_t(input_4d)

        ba_t = batchaug.ScaleIntensity(minv=None, maxv=None, factor=0.5)
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-6)

    def test_constant_input(self, device):
        """Constant tensor (min == max) matches MONAI edge case."""
        input_4d = torch.full((3, 16, 16, 16), 5.0, device=device)
        input_5d = input_4d.unsqueeze(0)

        monai_t = monai.transforms.ScaleIntensity(
            minv=0.0, maxv=1.0, channel_wise=True
        )
        monai_out = monai_t(input_4d)

        ba_t = batchaug.ScaleIntensity(minv=0.0, maxv=1.0)
        ba_out = ba_t(input_5d)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-6)


class TestScaleIntensityBatchIndependence:
    """B>1: each batch element is scaled independently."""

    def test_different_ranges_produce_different_scales(self, device):
        """Batch elements with different value ranges get different scaling."""
        tensor = torch.zeros(2, 1, 8, 8, 8, device=device)
        tensor[0] = torch.linspace(0, 10, 8 * 8 * 8, device=device).view(1, 8, 8, 8)
        tensor[1] = torch.linspace(0, 100, 8 * 8 * 8, device=device).view(1, 8, 8, 8)

        t = batchaug.ScaleIntensity(minv=0.0, maxv=1.0)
        result = t(tensor)

        # Both should be in [0, 1] after scaling
        for b in range(2):
            assert torch.allclose(
                result[b].min(), torch.tensor(0.0, device=device), atol=1e-5
            )
            assert torch.allclose(
                result[b].max(), torch.tensor(1.0, device=device), atol=1e-5
            )


class TestScaleIntensityChannelIndependence:
    """Each channel is rescaled independently within a batch element."""

    def test_per_channel_min_max(self, device):
        """Channels with different ranges each map to [0, 1] independently."""
        tensor = torch.zeros(1, 2, 8, 8, 8, device=device)
        tensor[0, 0] = torch.linspace(0.0, 0.5, 8 ** 3, device=device).view(8, 8, 8)
        tensor[0, 1] = torch.linspace(5.0, 10.0, 8 ** 3, device=device).view(8, 8, 8)

        t = batchaug.ScaleIntensity(minv=0.0, maxv=1.0)
        result = t(tensor)

        # Each channel independently maps to [0, 1]
        for c in range(2):
            assert torch.allclose(
                result[0, c].min(), torch.tensor(0.0, device=device), atol=1e-5
            )
            assert torch.allclose(
                result[0, c].max(), torch.tensor(1.0, device=device), atol=1e-5
            )


class TestScaleIntensityBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.ScaleIntensity(minv=0.0, maxv=1.0)
        result = t(vol_bf16)

        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert result.min() >= -1e-3
        assert result.max() <= 1.0 + 1e-3
