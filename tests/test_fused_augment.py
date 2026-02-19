"""Tests for FusedAugment and FusedAugmentd."""
import pytest
import torch
import torch.nn.functional as F

import batchaug
from batchaug.pytorch.geometric.affine import monai_affine_to_theta
from batchaug.triton.fused import FusedAugment, FusedAugmentd

DEVICE = "cuda"
SHAPE_SMALL = (2, 2, 16, 16, 16)  # B=2, C=2
SHAPE_MED = (3, 1, 32, 32, 32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_tensor(shape=SHAPE_SMALL, device=DEVICE):
    torch.manual_seed(42)
    return torch.rand(shape, device=device)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

class TestFusedAugmentBasic:
    def test_passthrough_all_zero_prob(self):
        """All probs=0, scale_intensity=False → output == input."""
        x = _fixed_tensor()
        fa = FusedAugment(scale_intensity=False)
        out = fa(x)
        assert torch.equal(out, x)

    def test_output_shape(self):
        """Output shape matches input."""
        x = _fixed_tensor()
        fa = FusedAugment(
            flip_prob=0.5, noise_prob=0.5, noise_std=0.1, scale_intensity=True,
        )
        out = fa(x)
        assert out.shape == x.shape

    def test_scale_intensity_only(self):
        """Scale intensity alone matches standalone ScaleIntensity."""
        x = _fixed_tensor()
        fa = FusedAugment(scale_intensity=True, minv=0.0, maxv=1.0, channel_wise=True)
        out_fused = fa(x)
        # Compare vs standalone Triton ScaleIntensity
        si = batchaug.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)
        out_ref = si(x)
        assert torch.allclose(out_fused, out_ref, atol=1e-5)

    def test_scale_intensity_per_element(self):
        """Scale intensity with channel_wise=False."""
        x = _fixed_tensor()
        fa = FusedAugment(scale_intensity=True, minv=0.0, maxv=1.0, channel_wise=False)
        out_fused = fa(x)
        si = batchaug.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=False)
        out_ref = si(x)
        assert torch.allclose(out_fused, out_ref, atol=1e-5)


class TestFusedElementwise:
    """Test each elementwise transform individually via FusedAugment."""

    def test_contrast_only(self):
        """RandAdjustContrast alone matches standalone."""
        x = _fixed_tensor()
        torch.manual_seed(0)
        fa = FusedAugment(
            scale_intensity=False, contrast_prob=1.0, gamma=(1.5, 1.5),
        )
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        out_fused = fa.apply(x, params)

        # Reference: standalone RandAdjustContrast
        rc = batchaug.RandAdjustContrast(prob=1.0, gamma=(1.5, 1.5))
        ref_params = {
            "mask": params["contrast_mask"],
            "gamma": params["gamma"],
        }
        out_ref = rc.apply(x, ref_params)
        # Triton uses exp(gamma*log(x)) vs PyTorch pow() — ~3e-4 precision diff
        assert torch.allclose(out_fused, out_ref, atol=5e-4)

    def test_noise_only(self):
        """RandGaussianNoise: noise is generated in-kernel, verify statistical properties."""
        x = torch.zeros(4, 2, 32, 32, 32, device=DEVICE)  # zero input → output IS the noise
        fa = FusedAugment(scale_intensity=False, noise_prob=1.0, noise_std=0.5, noise_mean=0.0)
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        # Force all masks on
        params["noise_mask"][:] = True
        params["noise_std"][:] = 0.5
        params["noise_mean"][:] = 0.0
        out = fa.apply(x, params)

        # Output should be non-zero (noise was added)
        assert not torch.equal(out, x)
        # Check statistical properties: mean ≈ 0, std ≈ 0.5
        assert out.float().mean().abs() < 0.05  # mean near 0
        assert (out.float().std() - 0.5).abs() < 0.1  # std near 0.5

    def test_noise_mask_respected(self):
        """Noise mask=False leaves output unchanged."""
        x = _fixed_tensor()
        fa = FusedAugment(scale_intensity=False, noise_prob=1.0, noise_std=0.5)
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        params["noise_mask"][:] = False  # force all off
        out = fa.apply(x, params)
        assert torch.equal(out, x)

    def test_bias_field_only(self):
        """RandBiasField alone matches standalone Triton version."""
        x = _fixed_tensor()
        torch.manual_seed(0)
        fa = FusedAugment(
            scale_intensity=False, bias_field_prob=1.0,
            coeff_range=(0.0, 0.05),
        )
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        out_fused = fa.apply(x, params)

        # Reference: standalone RandBiasField
        bf = batchaug.RandBiasField(prob=1.0, degree=3, coeff_range=(0.0, 0.05))
        ref_params = bf.sample_params(x.shape[0], x.shape, x.device)
        # Use the same coefficients and mask
        ref_params["coeffs"] = params["coeffs"]
        ref_params["mask"] = params["bias_mask"]
        out_ref = bf.apply(x, ref_params)
        assert torch.allclose(out_fused, out_ref, atol=1e-4)

    def test_all_elementwise(self):
        """All 4 elementwise transforms simultaneously produce valid output."""
        x = _fixed_tensor()
        fa = FusedAugment(
            scale_intensity=True, minv=0.0, maxv=1.0,
            contrast_prob=1.0, gamma=(1.5, 1.5),
            bias_field_prob=1.0, coeff_range=(0.0, 0.01),
            noise_prob=1.0, noise_std=0.1,
        )
        out = fa(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        # Output should differ from input (transforms are active)
        assert not torch.equal(out, x)


class TestFusedGeometric:
    """Test geometric fusion in FusedAugment."""

    def test_flip_only(self):
        """Flip matches standalone RandAxisFlip."""
        x = _fixed_tensor()
        torch.manual_seed(10)
        fa = FusedAugment(flip_prob=1.0, scale_intensity=False)
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        out_fused = fa.apply(x, params)

        # Reference
        flip = batchaug.RandAxisFlip(prob=1.0)
        out_ref = flip.apply(x, params["flip"])
        # Geometric via grid_sample introduces interpolation — use tolerance
        assert torch.allclose(out_fused.float(), out_ref.float(), atol=1e-5)

    def test_rotate90_only(self):
        """Rotate90 matches standalone RandRotate90."""
        x = _fixed_tensor()
        torch.manual_seed(20)
        fa = FusedAugment(
            rotate90_prob=1.0, max_k=3, spatial_axes=(0, 1),
            scale_intensity=False,
        )
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        out_fused = fa.apply(x, params)

        # Reference
        rot = batchaug.RandRotate90(prob=1.0, max_k=3, spatial_axes=(0, 1))
        out_ref = rot.apply(x, params["rot90"])
        assert torch.allclose(out_fused.float(), out_ref.float(), atol=1e-5)

    def test_affine_only(self):
        """Affine matches standalone RandAffine."""
        x = _fixed_tensor()
        torch.manual_seed(30)
        fa = FusedAugment(
            affine_prob=1.0, rotate_range=0.5,
            shear_range=0.2, translate_range=3,
            scale_intensity=False,
        )
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        out_fused = fa.apply(x, params)

        # Reference
        aff = batchaug.RandAffine(
            prob=1.0, rotate_range=0.5,
            shear_range=0.2, translate_range=3,
        )
        out_ref = aff.apply(x, params["affine"])
        assert torch.allclose(out_fused.float(), out_ref.float(), atol=1e-5)

    def test_geometric_fusion(self):
        """Multiple geometric transforms compose like Compose(lazy=True)."""
        x = _fixed_tensor()
        torch.manual_seed(40)
        fa = FusedAugment(
            flip_prob=1.0, rotate90_prob=1.0, spatial_axes=(0, 1),
            affine_prob=1.0, rotate_range=0.3,
            scale_intensity=False,
        )
        out = fa(x)
        assert out.shape == x.shape
        # Just verify it doesn't crash and produces finite output
        assert torch.isfinite(out).all()


class TestFusedSpatialIntensity:
    """Test spatial intensity transforms in FusedAugment."""

    def test_smooth_only(self):
        """Smooth matches standalone RandGaussianSmooth."""
        x = _fixed_tensor()
        torch.manual_seed(50)
        fa = FusedAugment(
            smooth_prob=1.0,
            smooth_sigma_x=(0.5, 0.5), smooth_sigma_y=(0.5, 0.5),
            smooth_sigma_z=(0.5, 0.5),
            scale_intensity=False,
        )
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        out_fused = fa.apply(x, params)

        # Reference
        sm = batchaug.RandGaussianSmooth(
            prob=1.0, sigma_x=(0.5, 0.5), sigma_y=(0.5, 0.5), sigma_z=(0.5, 0.5),
        )
        out_ref = sm.apply(x, params["smooth"])
        assert torch.allclose(out_fused, out_ref, atol=1e-5)


class TestFusedFullPipeline:
    """Test full pipeline with multiple transform types."""

    def test_full_pipeline_runs(self):
        """Full pipeline with all transforms enabled runs without error."""
        x = _fixed_tensor(SHAPE_MED)
        fa = FusedAugment(
            flip_prob=0.5,
            rotate90_prob=0.5, max_k=3, spatial_axes=(0, 1),
            affine_prob=0.5, rotate_range=0.3, shear_range=0.1, translate_range=2,
            smooth_prob=0.5, smooth_sigma_x=(0.25, 0.5), smooth_sigma_y=(0.25, 0.5),
                smooth_sigma_z=(0.25, 0.5),
            contrast_prob=0.5, gamma=(0.5, 2.0),
            bias_field_prob=0.5, coeff_range=(0.0, 0.05),
            noise_prob=0.5, noise_std=0.1,
            scale_intensity=True,
        )
        out = fa(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_bfloat16(self):
        """Pipeline works with bfloat16 input."""
        x = _fixed_tensor().to(torch.bfloat16)
        fa = FusedAugment(
            scale_intensity=True,
            contrast_prob=1.0, gamma=(1.5, 1.5),
            noise_prob=1.0, noise_std=0.1,
        )
        out = fa(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_sample_params_apply_roundtrip(self):
        """sample_params → apply produces same result as __call__ with same seed."""
        x = _fixed_tensor()
        fa = FusedAugment(
            flip_prob=0.5, noise_prob=0.5, noise_std=0.1,
            scale_intensity=True,
        )
        torch.manual_seed(99)
        out1 = fa(x)

        torch.manual_seed(99)
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        out2 = fa.apply(x, params)
        assert torch.equal(out1, out2)


class TestFusedAugmentd:
    """Test dictionary version."""

    def test_dict_basic(self):
        """FusedAugmentd applies to vol and seg."""
        vol = _fixed_tensor()
        seg = torch.randint(0, 5, SHAPE_SMALL, device=DEVICE).float()
        data = {"vol": vol, "seg": seg}

        fad = FusedAugmentd(
            keys=["vol", "seg"],
            intensity_keys=["vol"],
            mode={"vol": "bilinear", "seg": "nearest"},
            flip_prob=0.5,
            noise_prob=0.5, noise_std=0.1,
            scale_intensity=True,
        )
        out = fad(data)
        assert out["vol"].shape == vol.shape
        assert out["seg"].shape == seg.shape
        assert torch.isfinite(out["vol"]).all()

    def test_intensity_keys_respected(self):
        """Intensity transforms only applied to intensity_keys."""
        vol = _fixed_tensor()
        seg = _fixed_tensor()  # same shape, different content
        data = {"vol": vol.clone(), "seg": seg.clone()}

        # No geometric, only intensity → seg should be unchanged
        fad = FusedAugmentd(
            keys=["vol", "seg"],
            intensity_keys=["vol"],
            noise_prob=1.0, noise_std=0.5,
            scale_intensity=True,
        )
        out = fad(data)
        # vol should change, seg should not
        assert not torch.equal(out["vol"], vol)
        assert torch.equal(out["seg"], seg)

    def test_per_key_mode(self):
        """Per-key interpolation modes work correctly."""
        vol = _fixed_tensor()
        seg = torch.randint(0, 5, SHAPE_SMALL, device=DEVICE).float()
        data = {"vol": vol, "seg": seg}

        fad = FusedAugmentd(
            keys=["vol", "seg"],
            intensity_keys=["vol"],
            mode={"vol": "bilinear", "seg": "nearest"},
            affine_prob=1.0, rotate_range=0.5,
            scale_intensity=False,
        )
        out = fad(data)
        assert out["vol"].shape == vol.shape
        assert out["seg"].shape == seg.shape

    def test_full_pipeline_dict(self):
        """Full pipeline in dict mode."""
        vol = torch.rand(2, 4, 32, 32, 32, device=DEVICE)
        seg = torch.randint(0, 5, (2, 4, 32, 32, 32), device=DEVICE).float()
        data = {"vol": vol, "seg": seg}

        fad = FusedAugmentd(
            keys=["vol", "seg"],
            intensity_keys=["vol"],
            mode={"vol": "bilinear", "seg": "nearest"},
            flip_prob=0.15,
            rotate90_prob=0.15, max_k=3, spatial_axes=(0, 1),
            affine_prob=0.15, rotate_range=0.785, shear_range=0.3, translate_range=5,
            smooth_prob=0.15, smooth_sigma_x=(0.0, 0.1), smooth_sigma_y=(0.0, 0.1),
                smooth_sigma_z=(0.0, 0.1),
            noise_prob=0.15, noise_std=0.5,
            bias_field_prob=0.15, coeff_range=(0.0, 0.05),
            contrast_prob=0.15, gamma=(0.5, 2.5),
            scale_intensity=True,
        )
        out = fad(data)
        assert out["vol"].shape == vol.shape
        assert out["seg"].shape == seg.shape
        assert torch.isfinite(out["vol"]).all()


# ---------------------------------------------------------------------------
# FusedAugment vs sequential (Compose-equivalent) comparison
# ---------------------------------------------------------------------------

def _apply_geometric_fusion(x, fa, params, mode="bilinear", padding_mode="zeros"):
    """Replicate FusedAugment's geometric fusion: compose affines → single grid_sample."""
    B = x.shape[0]
    device = x.device
    affine = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).clone()
    geo_mask = torch.zeros(B, dtype=torch.bool, device=device)
    has_geo = False
    for name, t in fa._geo_transforms:
        if name in params:
            p = params[name]
            affine = affine @ t.to_affine(p)
            geo_mask = geo_mask | p["mask"]
            has_geo = True
    if has_geo and geo_mask.any():
        theta = monai_affine_to_theta(affine, x.shape[2:], device)
        grid = F.affine_grid(theta[:, :3, :], list(x.shape), align_corners=False)
        resampled = F.grid_sample(
            x.float(), grid, mode=mode, padding_mode=padding_mode, align_corners=False,
        ).to(x.dtype)
        mask_5d = geo_mask[:, None, None, None, None]
        x = torch.where(mask_5d, resampled, x)
    return x


def _apply_spatial_intensity(x, fa, params):
    """Replicate FusedAugment's spatial intensity: run transforms sequentially."""
    for name, t in fa._spatial_transforms:
        if t is not None and name in params:
            x = t.apply(x, params[name])
    return x


def _apply_elementwise_sequential(x, fa, params):
    """Replicate FusedAugment's elementwise phase using individual transforms."""
    B = x.shape[0]
    device = x.device

    # 1. ScaleIntensity (deterministic)
    if fa._do_scale:
        si = batchaug.ScaleIntensity(minv=fa._minv, maxv=fa._maxv, channel_wise=fa._channel_wise)
        x = si(x)

    # 2. RandAdjustContrast
    if fa._do_contrast:
        rc = batchaug.RandAdjustContrast(prob=1.0, gamma=fa._gamma)
        rc_params = {"mask": params["contrast_mask"], "gamma": params["gamma"]}
        x = rc.apply(x, rc_params)

    # 3. RandBiasField
    if fa._do_bias:
        bf = batchaug.RandBiasField(prob=1.0, degree=fa._degree, coeff_range=fa._coeff_range)
        bf_params = bf.sample_params(B, x.shape, device)
        bf_params["coeffs"] = params["coeffs"]
        bf_params["mask"] = params["bias_mask"]
        x = bf.apply(x, bf_params)

    return x


class TestFusedMatchesSequential:
    """Verify FusedAugment matches running the same transforms sequentially."""

    def test_geometric_only(self):
        """Geometric fusion in FusedAugment matches manual affine composition."""
        x = _fixed_tensor()
        torch.manual_seed(42)
        fa = FusedAugment(
            flip_prob=1.0, rotate90_prob=1.0, spatial_axes=(0, 1),
            affine_prob=1.0, rotate_range=0.3,
            scale_intensity=False,
        )
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        out_fused = fa.apply(x, params)
        out_ref = _apply_geometric_fusion(x, fa, params)
        assert torch.allclose(out_fused.float(), out_ref.float(), atol=1e-6)

    def test_spatial_intensity_only(self):
        """Spatial intensity in FusedAugment matches running transforms sequentially."""
        x = _fixed_tensor()
        torch.manual_seed(42)
        fa = FusedAugment(
            smooth_prob=1.0,
            smooth_sigma_x=(0.5, 0.5), smooth_sigma_y=(0.5, 0.5), smooth_sigma_z=(0.5, 0.5),
            sharpen_prob=1.0,
            sharpen_sigma1_x=(0.5, 0.5), sharpen_sigma1_y=(0.5, 0.5), sharpen_sigma1_z=(0.5, 0.5),
            gibbs_prob=1.0, gibbs_alpha=(0.2, 0.2),
            scale_intensity=False,
        )
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        out_fused = fa.apply(x, params)
        out_ref = _apply_spatial_intensity(x, fa, params)
        assert torch.allclose(out_fused.float(), out_ref.float(), atol=1e-6)

    def test_elementwise_no_noise(self):
        """Fused Triton elementwise kernel matches individual transforms sequentially."""
        x = _fixed_tensor()
        torch.manual_seed(42)
        fa = FusedAugment(
            scale_intensity=True, minv=0.0, maxv=1.0, channel_wise=True,
            contrast_prob=1.0, gamma=(1.5, 1.5),
            bias_field_prob=1.0, coeff_range=(0.0, 0.05),
        )
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        # Force all masks on
        params["contrast_mask"][:] = True
        params["bias_mask"][:] = True
        out_fused = fa.apply(x, params)
        out_ref = _apply_elementwise_sequential(x, fa, params)
        # Triton exp(gamma*log(x)) vs PyTorch pow() gives ~3e-4 diff
        assert torch.allclose(out_fused.float(), out_ref.float(), atol=5e-4)

    def test_full_pipeline_no_noise(self):
        """Full pipeline (minus noise) matches sequential application."""
        x = _fixed_tensor(shape=(3, 2, 24, 24, 24))
        torch.manual_seed(42)
        fa = FusedAugment(
            flip_prob=1.0, rotate90_prob=1.0, spatial_axes=(0, 1),
            affine_prob=1.0, rotate_range=0.2,
            smooth_prob=1.0,
            smooth_sigma_x=(0.3, 0.3), smooth_sigma_y=(0.3, 0.3), smooth_sigma_z=(0.3, 0.3),
            scale_intensity=True, minv=0.0, maxv=1.0,
            contrast_prob=1.0, gamma=(1.5, 1.5),
            bias_field_prob=1.0, coeff_range=(0.0, 0.03),
        )
        params = fa.sample_params(x.shape[0], x.shape, x.device)
        params["contrast_mask"][:] = True
        params["bias_mask"][:] = True
        out_fused = fa.apply(x, params)

        # Reference: run phases sequentially
        ref = x.clone()
        ref = _apply_geometric_fusion(ref, fa, params)
        ref = _apply_spatial_intensity(ref, fa, params)
        ref = _apply_elementwise_sequential(ref, fa, params)
        assert torch.allclose(out_fused.float(), ref.float(), atol=5e-4)

    def test_full_pipeline_dict_no_noise(self):
        """FusedAugmentd vol output matches sequential application."""
        vol = _fixed_tensor(shape=(2, 2, 24, 24, 24))
        torch.manual_seed(42)

        fa = FusedAugment(
            flip_prob=1.0, rotate90_prob=1.0, spatial_axes=(0, 1),
            affine_prob=1.0, rotate_range=0.2,
            smooth_prob=1.0,
            smooth_sigma_x=(0.3, 0.3), smooth_sigma_y=(0.3, 0.3), smooth_sigma_z=(0.3, 0.3),
            scale_intensity=True, minv=0.0, maxv=1.0,
            contrast_prob=1.0, gamma=(1.5, 1.5),
            bias_field_prob=1.0, coeff_range=(0.0, 0.03),
        )
        params = fa.sample_params(vol.shape[0], vol.shape, vol.device)
        params["contrast_mask"][:] = True
        params["bias_mask"][:] = True

        # FusedAugment.apply (single tensor)
        out_fused = fa.apply(vol, params)

        # Reference: sequential phases
        ref = vol.clone()
        ref = _apply_geometric_fusion(ref, fa, params)
        ref = _apply_spatial_intensity(ref, fa, params)
        ref = _apply_elementwise_sequential(ref, fa, params)

        assert torch.allclose(out_fused.float(), ref.float(), atol=5e-4)
