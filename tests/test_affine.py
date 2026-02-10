import torch
import monai.transforms

import batchaug
from batchaug.geometric.affine import (
    _build_rotation_matrices,
    _build_scale_matrices,
    _build_shear_matrices,
    _build_translation_matrices,
    monai_affine_to_theta,
)


class TestRandAffineMatchesMONAI:
    """B=1: batchaug affine matches MONAI's RandAffine."""

    def _compare_monai(self, vol, device, **kwargs):
        """Run both MONAI and batchaug with identical fixed params and compare."""
        input_4d = vol[0]  # (C, H, W, D)
        input_5d = vol[:1]  # (1, C, H, W, D)

        monai_t = monai.transforms.RandAffine(
            prob=1.0,
            padding_mode="zeros",
            mode="bilinear",
            **kwargs,
        )
        monai_t.randomize()

        # Extract sampled params from MONAI
        rand_grid = monai_t.rand_affine_grid
        rotate_params = rand_grid.rotate_params
        shear_params = rand_grid.shear_params
        translate_params = rand_grid.translate_params
        scale_params = rand_grid.scale_params

        monai_out = monai_t(input_4d, randomize=False)

        # Reconstruct the MONAI-space affine from sampled params
        affine = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        if rotate_params:
            angles = torch.tensor([rotate_params], device=device, dtype=torch.float32)
            affine = affine @ _build_rotation_matrices(angles)
        if shear_params:
            shear = torch.tensor([shear_params], device=device, dtype=torch.float32)
            affine = affine @ _build_shear_matrices(shear)
        if translate_params:
            shift = torch.tensor([translate_params], device=device, dtype=torch.float32)
            affine = affine @ _build_translation_matrices(shift)
        if scale_params:
            scale = torch.tensor([scale_params], device=device, dtype=torch.float32)
            affine = affine @ _build_scale_matrices(scale)

        spatial_shape = input_5d.shape[2:]

        ba_t = batchaug.RandAffine(prob=1.0, padding_mode="zeros", mode="bilinear")
        params = ba_t.sample_params(1, input_5d.shape, device)
        params["mask"] = torch.tensor([True], device=device)
        params["affine"] = affine
        params["theta"] = monai_affine_to_theta(affine, spatial_shape, device)

        ba_out = ba_t.apply(input_5d, params)
        return ba_out[0], monai_out

    def test_matches_monai_translate(self, vol, device):
        ba_out, monai_out = self._compare_monai(
            vol, device,
            translate_range=(2.0, 2.0, 2.0),
        )
        assert torch.allclose(ba_out, monai_out, atol=1e-4), (
            f"max diff: {(ba_out - monai_out).abs().max().item()}"
        )

    def test_matches_monai_rotate(self, vol, device):
        ba_out, monai_out = self._compare_monai(
            vol, device,
            rotate_range=(0.3, 0.3, 0.3),
        )
        assert torch.allclose(ba_out, monai_out, atol=1e-4), (
            f"max diff: {(ba_out - monai_out).abs().max().item()}"
        )

    def test_matches_monai_scale(self, vol, device):
        ba_out, monai_out = self._compare_monai(
            vol, device,
            scale_range=(0.2, 0.2, 0.2),
        )
        assert torch.allclose(ba_out, monai_out, atol=1e-4), (
            f"max diff: {(ba_out - monai_out).abs().max().item()}"
        )

    def test_no_change_when_mask_false(self, vol, device):
        ba_t = batchaug.RandAffine(
            prob=1.0, rotate_range=0.5, scale_range=0.2
        )
        params = ba_t.sample_params(1, vol[:1].shape, device)
        params["mask"] = torch.tensor([False], device=device)
        ba_out = ba_t.apply(vol[:1], params)
        assert torch.equal(ba_out, vol[:1])


class TestRandAffineBatchIndependence:
    """B>1: each element gets different affine."""

    def test_different_transform_per_element(self, vol, device):
        t = batchaug.RandAffine(prob=1.0, rotate_range=0.5, scale_range=0.2)
        result = t(vol)
        # Elements should differ from each other
        diffs = (result - vol).abs().flatten(1).mean(dim=1)
        assert diffs.max() - diffs.min() > 1e-4


class TestRandAffinedPerKeyModes:
    """Dict wrapper applies different interpolation per key."""

    def test_bilinear_vs_nearest(self, vol, seg, device):
        t = batchaug.RandAffined(
            keys=["vol", "seg"],
            prob=1.0,
            rotate_range=0.3,
            mode={"vol": "bilinear", "seg": "nearest"},
        )
        batch = {"vol": vol, "seg": seg}
        result = t(batch)

        # Both should be transformed (not equal to input)
        assert not torch.equal(result["vol"], vol)
        assert not torch.equal(result["seg"], seg)

        # Seg with nearest should have values from original label set
        # (0, 1, 2, 3, 4 or 0 from zero-padding)
        unique_vals = result["seg"].unique()
        expected = {0.0, 1.0, 2.0, 3.0, 4.0}
        assert all(v.item() in expected for v in unique_vals)


class TestRandAffineNonCubic:
    """Non-cubic spatial shapes (H != W != D) match MONAI."""

    def _compare_monai_nonsquare(self, device, **kwargs):
        """Run MONAI and batchaug on (1, 1, 12, 16, 20) and compare."""
        torch.manual_seed(0)
        input_4d = torch.rand(1, 12, 16, 20, device=device)
        input_5d = input_4d.unsqueeze(0)

        monai_t = monai.transforms.RandAffine(
            prob=1.0,
            padding_mode="zeros",
            mode="bilinear",
            **kwargs,
        )
        monai_t.randomize()

        rand_grid = monai_t.rand_affine_grid
        rotate_params = rand_grid.rotate_params
        shear_params = rand_grid.shear_params
        translate_params = rand_grid.translate_params
        scale_params = rand_grid.scale_params

        monai_out = monai_t(input_4d, randomize=False)

        affine = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        if rotate_params:
            angles = torch.tensor([rotate_params], device=device, dtype=torch.float32)
            affine = affine @ _build_rotation_matrices(angles)
        if shear_params:
            shear = torch.tensor([shear_params], device=device, dtype=torch.float32)
            affine = affine @ _build_shear_matrices(shear)
        if translate_params:
            shift = torch.tensor([translate_params], device=device, dtype=torch.float32)
            affine = affine @ _build_translation_matrices(shift)
        if scale_params:
            scale = torch.tensor([scale_params], device=device, dtype=torch.float32)
            affine = affine @ _build_scale_matrices(scale)

        spatial_shape = input_5d.shape[2:]

        ba_t = batchaug.RandAffine(prob=1.0, padding_mode="zeros", mode="bilinear")
        params = ba_t.sample_params(1, input_5d.shape, device)
        params["mask"] = torch.tensor([True], device=device)
        params["affine"] = affine
        params["theta"] = monai_affine_to_theta(affine, spatial_shape, device)

        ba_out = ba_t.apply(input_5d, params)
        return ba_out[0], monai_out

    def test_translate_nonsquare(self, device):
        ba_out, monai_out = self._compare_monai_nonsquare(
            device, translate_range=(2.0, 2.0, 2.0),
        )
        assert torch.allclose(ba_out, monai_out, atol=1e-4), (
            f"max diff: {(ba_out - monai_out).abs().max().item()}"
        )

    def test_rotate_nonsquare(self, device):
        ba_out, monai_out = self._compare_monai_nonsquare(
            device, rotate_range=(0.3, 0.3, 0.3),
        )
        assert torch.allclose(ba_out, monai_out, atol=1e-4), (
            f"max diff: {(ba_out - monai_out).abs().max().item()}"
        )

    def test_scale_nonsquare(self, device):
        ba_out, monai_out = self._compare_monai_nonsquare(
            device, scale_range=(0.2, 0.2, 0.2),
        )
        assert torch.allclose(ba_out, monai_out, atol=1e-4), (
            f"max diff: {(ba_out - monai_out).abs().max().item()}"
        )

    def test_combined_nonsquare(self, device):
        ba_out, monai_out = self._compare_monai_nonsquare(
            device,
            rotate_range=(0.2, 0.2, 0.2),
            scale_range=(0.1, 0.1, 0.1),
            translate_range=(1.0, 1.0, 1.0),
        )
        assert torch.allclose(ba_out, monai_out, atol=1e-4), (
            f"max diff: {(ba_out - monai_out).abs().max().item()}"
        )


class TestRandAffineBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandAffine(
            prob=1.0, rotate_range=0.3, scale_range=0.1
        )
        result = t(vol_bf16)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
