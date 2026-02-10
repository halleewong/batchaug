import pytest
import torch
import monai.transforms

import batchaug


class TestRandAxisFlipMatchesMONAI:
    """B=1: batchaug flip matches MONAI Flip for each axis."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_matches_monai_deterministic(self, device, axis):
        """Force a specific axis and compare against MONAI's Flip."""
        torch.manual_seed(0)
        input_4d = torch.rand(3, 16, 16, 16, device=device)
        input_5d = input_4d.unsqueeze(0)

        # MONAI deterministic flip (spatial_axis relative to spatial dims)
        monai_t = monai.transforms.Flip(spatial_axis=axis)
        monai_out = monai_t(input_4d)

        # batchaug with forced params
        ba_t = batchaug.RandAxisFlip(prob=1.0)
        params = {
            "mask": torch.tensor([True], device=device),
            "axes": torch.tensor([axis], device=device),
        }
        ba_out = ba_t.apply(input_5d, params)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-6)

    def test_no_flip_when_mask_false(self, device):
        """When mask is False, output equals input."""
        torch.manual_seed(0)
        input_5d = torch.rand(1, 3, 16, 16, 16, device=device)

        ba_t = batchaug.RandAxisFlip(prob=1.0)
        params = {
            "mask": torch.tensor([False], device=device),
            "axes": torch.tensor([0], device=device),
        }
        ba_out = ba_t.apply(input_5d, params)

        assert torch.equal(ba_out, input_5d)


class TestRandAxisFlipBatchIndependence:
    """B>1: batch elements receive independent flips."""

    def test_different_elements_can_differ(self, device):
        """With prob=1.0, at least some batch elements should differ."""
        torch.manual_seed(123)
        tensor = torch.rand(8, 3, 16, 16, 16, device=device)

        t = batchaug.RandAxisFlip(prob=1.0)
        result = t(tensor)

        # At least two batch elements should differ
        # (astronomically unlikely all 8 get same axis and data)
        diffs = 0
        for i in range(1, 8):
            if not torch.equal(result[0], result[i]):
                diffs += 1
        assert diffs > 0

    def test_forced_different_axes(self, device):
        """Force different axes per batch element, verify they differ."""
        tensor = torch.rand(3, 2, 8, 8, 8, device=device)

        t = batchaug.RandAxisFlip(prob=1.0)
        params = {
            "mask": torch.tensor([True, True, True], device=device),
            "axes": torch.tensor([0, 1, 2], device=device),
        }
        result = t.apply(tensor, params)

        # Each batch element was flipped on a different axis
        # They should not all be equal to each other
        assert not torch.equal(result[0], result[1])
        assert not torch.equal(result[0], result[2])


class TestRandAxisFlipChannelConsistency:
    """All channels in a batch element get the same flip."""

    def test_channels_flip_identically(self, device):
        """Place a marker in each channel, verify they all move to the same spot."""
        B, C, H, W, D = 2, 4, 8, 8, 8
        tensor = torch.zeros(B, C, H, W, D, device=device)
        # Place unique marker at position (0, 0, 0) in each channel
        for c in range(C):
            tensor[:, c, 0, 0, 0] = float(c + 1)

        t = batchaug.RandAxisFlip(prob=1.0)
        # Force flip on axis 0 (H) for all batch elements
        params = {
            "mask": torch.ones(B, dtype=torch.bool, device=device),
            "axes": torch.zeros(B, dtype=torch.long, device=device),
        }
        result = t.apply(tensor, params)

        # After flipping H: marker should be at (H-1, 0, 0) for all channels
        for b in range(B):
            for c in range(C):
                assert result[b, c, H - 1, 0, 0].item() == float(c + 1)
                assert result[b, c, 0, 0, 0].item() == 0.0


class TestRandAxisFlipBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandAxisFlip(prob=1.0)
        result = t(vol_bf16)

        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
