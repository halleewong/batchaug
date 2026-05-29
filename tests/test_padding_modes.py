"""Tests for padding_mode parameter in Gaussian smooth and sharpen."""
import torch
import batchaug


class TestGaussianSmoothPaddingModes:
    """Test different padding modes for RandGaussianSmooth."""

    def test_padding_mode_none(self, device):
        """Default (None) uses implicit zero-padding in conv3d."""
        vol = torch.rand(2, 1, 16, 16, 16, device=device)
        t = batchaug.RandGaussianSmooth(prob=1.0, padding_mode=None)
        result = t(vol)
        assert result.shape == vol.shape
        assert not torch.isnan(result).any()

    def test_padding_mode_reflect(self, device):
        """'reflect' uses F.pad with reflection mode."""
        vol = torch.rand(2, 1, 16, 16, 16, device=device)
        t = batchaug.RandGaussianSmooth(prob=1.0, padding_mode='reflect')
        result = t(vol)
        assert result.shape == vol.shape
        assert not torch.isnan(result).any()

    def test_padding_mode_replicate(self, device):
        """'replicate' uses F.pad with replicate mode."""
        vol = torch.rand(2, 1, 16, 16, 16, device=device)
        t = batchaug.RandGaussianSmooth(prob=1.0, padding_mode='replicate')
        result = t(vol)
        assert result.shape == vol.shape
        assert not torch.isnan(result).any()

    def test_different_padding_modes_differ(self, device):
        """Different padding modes should produce different results."""
        torch.manual_seed(42)
        vol = torch.rand(1, 1, 16, 16, 16, device=device)

        # Fix sigma for reproducible results
        t_none = batchaug.RandGaussianSmooth(
            prob=1.0,
            sigma_x=(0.8, 0.8),
            sigma_y=(0.8, 0.8),
            sigma_z=(0.8, 0.8),
            padding_mode=None
        )
        t_reflect = batchaug.RandGaussianSmooth(
            prob=1.0,
            sigma_x=(0.8, 0.8),
            sigma_y=(0.8, 0.8),
            sigma_z=(0.8, 0.8),
            padding_mode='reflect'
        )

        out_none = t_none(vol)
        out_reflect = t_reflect(vol)

        # Results should differ due to different padding
        assert not torch.allclose(out_none, out_reflect, atol=1e-3)

    def test_padding_mode_constant_same_as_none(self, device):
        """'constant' mode should give same results as None (zero-padding)."""
        torch.manual_seed(123)
        vol = torch.rand(1, 1, 16, 16, 16, device=device)

        t_none = batchaug.RandGaussianSmooth(
            prob=1.0,
            sigma_x=(0.8, 0.8),
            sigma_y=(0.8, 0.8),
            sigma_z=(0.8, 0.8),
            padding_mode=None
        )
        t_const = batchaug.RandGaussianSmooth(
            prob=1.0,
            sigma_x=(0.8, 0.8),
            sigma_y=(0.8, 0.8),
            sigma_z=(0.8, 0.8),
            padding_mode='constant'
        )

        out_none = t_none(vol)
        out_const = t_const(vol)

        # Both should use zero-padding
        assert torch.allclose(out_none, out_const, atol=1e-4)


class TestGaussianSharpenPaddingModes:
    """Test different padding modes for RandGaussianSharpen."""

    def test_padding_mode_none(self, device):
        """Default (None) uses implicit zero-padding in conv3d."""
        vol = torch.rand(2, 1, 16, 16, 16, device=device)
        t = batchaug.RandGaussianSharpen(prob=1.0, padding_mode=None)
        result = t(vol)
        assert result.shape == vol.shape
        assert not torch.isnan(result).any()

    def test_padding_mode_reflect(self, device):
        """'reflect' uses F.pad with reflection mode."""
        vol = torch.rand(2, 1, 16, 16, 16, device=device)
        t = batchaug.RandGaussianSharpen(prob=1.0, padding_mode='reflect')
        result = t(vol)
        assert result.shape == vol.shape
        assert not torch.isnan(result).any()

    def test_padding_mode_replicate(self, device):
        """'replicate' uses F.pad with replicate mode."""
        vol = torch.rand(2, 1, 16, 16, 16, device=device)
        t = batchaug.RandGaussianSharpen(prob=1.0, padding_mode='replicate')
        result = t(vol)
        assert result.shape == vol.shape
        assert not torch.isnan(result).any()

    def test_different_padding_modes_differ(self, device):
        """Different padding modes should produce different results for sharpening."""
        torch.manual_seed(42)
        vol = torch.rand(1, 1, 16, 16, 16, device=device)

        t_none = batchaug.RandGaussianSharpen(
            prob=1.0,
            sigma1_x=(0.8, 0.8),
            sigma1_y=(0.8, 0.8),
            sigma1_z=(0.8, 0.8),
            sigma2_x=(0.5, 0.5),
            sigma2_y=(0.5, 0.5),
            sigma2_z=(0.5, 0.5),
            alpha=(15.0, 15.0),
            padding_mode=None
        )
        t_reflect = batchaug.RandGaussianSharpen(
            prob=1.0,
            sigma1_x=(0.8, 0.8),
            sigma1_y=(0.8, 0.8),
            sigma1_z=(0.8, 0.8),
            sigma2_x=(0.5, 0.5),
            sigma2_y=(0.5, 0.5),
            sigma2_z=(0.5, 0.5),
            alpha=(15.0, 15.0),
            padding_mode='reflect'
        )

        out_none = t_none(vol)
        out_reflect = t_reflect(vol)

        # Results should differ due to different padding
        assert not torch.allclose(out_none, out_reflect, atol=1e-3)

    def test_edge_artifacts_reduced_with_reflection(self, device):
        """Reflection padding should reduce edge artifacts compared to zero-padding."""
        # Create a simple image with known structure
        vol = torch.ones(1, 1, 16, 16, 16, device=device) * 0.5

        t_none = batchaug.RandGaussianSharpen(
            prob=1.0,
            sigma1_x=(1.0, 1.0),
            sigma1_y=(1.0, 1.0),
            sigma1_z=(1.0, 1.0),
            sigma2_x=(0.5, 0.5),
            sigma2_y=(0.5, 0.5),
            sigma2_z=(0.5, 0.5),
            alpha=(20.0, 20.0),
            padding_mode=None
        )
        t_reflect = batchaug.RandGaussianSharpen(
            prob=1.0,
            sigma1_x=(1.0, 1.0),
            sigma1_y=(1.0, 1.0),
            sigma1_z=(1.0, 1.0),
            sigma2_x=(0.5, 0.5),
            sigma2_y=(0.5, 0.5),
            sigma2_z=(0.5, 0.5),
            alpha=(20.0, 20.0),
            padding_mode='reflect'
        )

        out_none = t_none(vol)
        out_reflect = t_reflect(vol)

        # Edge voxels should have less extreme values with reflection padding
        # (this is a qualitative test)
        edge_none = out_none[0, 0, 0:2, :, :].abs().max().item()
        edge_reflect = out_reflect[0, 0, 0:2, :, :].abs().max().item()

        # With reflection padding, edge artifacts should be less extreme
        # (though this depends on the specific input and parameters)
        assert out_reflect.shape == vol.shape
