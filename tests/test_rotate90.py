import pytest
import torch
import monai.transforms

import batchaug


class TestRandRotate90MatchesMONAI:
    """B=1: batchaug rotate90 matches MONAI Rotate90 for each k."""

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_matches_monai_deterministic(self, device, k):
        """Force a specific k and compare against MONAI's Rotate90."""
        torch.manual_seed(0)
        input_4d = torch.rand(3, 16, 16, 16, device=device)
        input_5d = input_4d.unsqueeze(0)

        # MONAI deterministic rotate90 (spatial_axes=(0,1) maps to tensor axes (1,2))
        monai_t = monai.transforms.Rotate90(k=k, spatial_axes=(0, 1))
        monai_out = monai_t(input_4d)

        # batchaug with forced params
        ba_t = batchaug.RandRotate90(prob=1.0, max_k=3, spatial_axes=(0, 1))
        params = {
            "mask": torch.tensor([True], device=device),
            "k": torch.tensor([k], device=device),
        }
        ba_out = ba_t.apply(input_5d, params)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-6)

    @pytest.mark.parametrize("spatial_axes", [(0, 1), (0, 2), (1, 2)])
    def test_matches_monai_all_planes(self, device, spatial_axes):
        """Test rotation in each spatial plane matches MONAI."""
        torch.manual_seed(0)
        input_4d = torch.rand(3, 16, 16, 16, device=device)
        input_5d = input_4d.unsqueeze(0)
        k = 2

        monai_t = monai.transforms.Rotate90(k=k, spatial_axes=spatial_axes)
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandRotate90(prob=1.0, max_k=3, spatial_axes=spatial_axes)
        params = {
            "mask": torch.tensor([True], device=device),
            "k": torch.tensor([k], device=device),
        }
        ba_out = ba_t.apply(input_5d, params)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-6)

    def test_no_rotation_when_mask_false(self, device):
        """When mask is False, output equals input."""
        torch.manual_seed(0)
        input_5d = torch.rand(1, 3, 16, 16, 16, device=device)

        ba_t = batchaug.RandRotate90(prob=1.0, max_k=3)
        params = {
            "mask": torch.tensor([False], device=device),
            "k": torch.tensor([2], device=device),
        }
        ba_out = ba_t.apply(input_5d, params)

        assert torch.equal(ba_out, input_5d)


class TestRandRotate90BatchIndependence:
    """B>1: batch elements receive independent rotations."""

    def test_forced_different_k(self, device):
        """Force different k values per batch element, verify they differ."""
        torch.manual_seed(0)
        tensor = torch.rand(3, 2, 16, 16, 16, device=device)

        t = batchaug.RandRotate90(prob=1.0, max_k=3)
        params = {
            "mask": torch.tensor([True, True, True], device=device),
            "k": torch.tensor([1, 2, 3], device=device),
        }
        result = t.apply(tensor, params)

        # Different k values produce different results
        assert not torch.equal(result[0], result[1])
        assert not torch.equal(result[0], result[2])

    def test_statistical_independence(self, device):
        """With prob=1.0 over many elements, not all get the same k."""
        torch.manual_seed(456)
        tensor = torch.rand(16, 2, 8, 8, 8, device=device)

        t = batchaug.RandRotate90(prob=1.0, max_k=3)
        params = t.sample_params(16, tensor.shape, device)

        # k should be in {1, 2, 3} and not all the same
        k_values = params["k"]
        assert (k_values >= 1).all()
        assert (k_values <= 3).all()
        assert k_values.unique().numel() > 1


class TestRandRotate90ChannelConsistency:
    """All channels in a batch element get the same rotation."""

    def test_channels_rotate_identically(self, device):
        """Place markers in each channel, verify they all rotate the same way."""
        B, C, H, W, D = 2, 4, 8, 8, 8
        tensor = torch.zeros(B, C, H, W, D, device=device)
        # Place unique marker at (0, 0, 0) for each channel
        for c in range(C):
            tensor[:, c, 0, 0, 0] = float(c + 1)

        # Force k=1 rotation in H-W plane for all batch elements
        t = batchaug.RandRotate90(prob=1.0, max_k=3, spatial_axes=(0, 1))
        params = {
            "mask": torch.ones(B, dtype=torch.bool, device=device),
            "k": torch.ones(B, dtype=torch.long, device=device),
        }
        result = t.apply(tensor, params)

        # After rot90(k=1, axes=(2,3)) on (B,C,H,W,D):
        # position (0,0,0) in H,W,D -> (0, H-1, 0) in the H,W plane
        # (rot90 k=1 in axes (2,3): (h,w) -> (w, H-1-h), so (0,0) -> (0, H-1))
        for b in range(B):
            for c in range(C):
                # Find where the marker ended up
                marker_loc = (result[b, c] == float(c + 1)).nonzero(as_tuple=False)
                assert marker_loc.shape[0] == 1
            # All channels should have marker at the same position
            positions = []
            for c in range(C):
                pos = (result[b, c] == float(c + 1)).nonzero(as_tuple=False)
                positions.append(pos)
            for c in range(1, C):
                assert torch.equal(positions[0], positions[c])


class TestRandRotate90Bfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandRotate90(prob=1.0, max_k=3)
        result = t(vol_bf16)

        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
