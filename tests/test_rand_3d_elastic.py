import torch
import torch.nn.functional as F
import numpy as np
import monai.transforms

import batchaug
from batchaug.pytorch.intensity.smooth import gaussian_1d_batch, separable_gaussian_conv3d


class TestRand3DElasticBasic:
    """Basic elastic deformation functionality."""

    def test_output_shape(self, vol, device):
        t = batchaug.Rand3DElastic(prob=1.0, sigma_range=(5.0, 5.0), magnitude_range=(0.1, 0.1))
        out = t(vol)
        assert out.shape == vol.shape

    def test_identity_zero_magnitude(self, vol, device):
        t = batchaug.Rand3DElastic(
            prob=1.0, sigma_range=(5.0, 5.0), magnitude_range=(0.0, 0.0)
        )
        out = t(vol)
        # With zero magnitude, displacement is zero → identity transform
        # grid_sample with align_corners=False introduces small interpolation errors
        assert torch.allclose(out, vol, atol=1e-3), (
            f"max diff: {(out - vol).abs().max().item()}"
        )

    def test_nonzero_magnitude_changes_output(self, vol, device):
        t = batchaug.Rand3DElastic(
            prob=1.0, sigma_range=(3.0, 3.0), magnitude_range=(1.0, 1.0)
        )
        out = t(vol)
        assert not torch.equal(out, vol)


class TestRand3DElasticMask:
    """Mask behavior."""

    def test_mask_preserves_input(self, vol, device):
        t = batchaug.Rand3DElastic(
            prob=1.0, sigma_range=(3.0, 3.0), magnitude_range=(1.0, 1.0)
        )
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        params["mask"] = torch.zeros(vol.shape[0], dtype=torch.bool, device=device)
        out = t.apply(vol, params)
        assert torch.equal(out, vol)

    def test_partial_mask(self, vol, device):
        t = batchaug.Rand3DElastic(
            prob=1.0, sigma_range=(3.0, 3.0), magnitude_range=(1.0, 1.0)
        )
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        params["mask"] = torch.tensor([True, False, True, False], device=device)
        out = t.apply(vol, params)
        # Masked-out elements should be unchanged
        assert torch.equal(out[1], vol[1])
        assert torch.equal(out[3], vol[3])
        # Masked-in elements should be changed
        assert not torch.equal(out[0], vol[0])
        assert not torch.equal(out[2], vol[2])


class TestRand3DElasticMatchesMONAI:
    """B=1: batchaug elastic matches MONAI's Rand3DElastic."""

    def test_match_monai(self, device):
        torch.manual_seed(42)
        input_4d = torch.rand(1, 16, 16, 16, device=device)
        input_5d = input_4d.unsqueeze(0)  # (1, 1, 16, 16, 16)
        H, W, D = 16, 16, 16

        # Create MONAI transform with fixed sigma and magnitude
        sigma_val = 5.0
        mag_val = 0.5
        monai_t = monai.transforms.Rand3DElastic(
            sigma_range=(sigma_val, sigma_val),
            magnitude_range=(mag_val, mag_val),
            prob=1.0,
            mode="bilinear",
            padding_mode="zeros",
        )
        monai_t.randomize(grid_size=(H, W, D))
        monai_out = monai_t(input_4d, randomize=False)

        # Extract MONAI's displacement
        rand_offset = torch.from_numpy(monai_t.rand_offset).to(device).float()  # (3, H, W, D)
        m_sigma = float(monai_t.sigma)
        m_magnitude = float(monai_t.magnitude)

        # Smooth displacement the same way MONAI does
        displacement = rand_offset.unsqueeze(0)  # (1, 3, H, W, D)
        sigma_tensor = torch.tensor([m_sigma], device=device)
        kernel = gaussian_1d_batch(sigma_tensor, truncated=3.0)
        for i in range(3):
            comp = displacement[:, i : i + 1, :, :, :]
            comp = separable_gaussian_conv3d(comp, kernel, kernel, kernel)
            displacement[:, i : i + 1, :, :, :] = comp
        displacement = displacement * m_magnitude

        # Build coordinate grid (same as MONAI's create_grid)
        coords = [
            torch.linspace(-(s - 1) / 2.0, (s - 1) / 2.0, s, device=device, dtype=torch.float32)
            for s in [H, W, D]
        ]
        grid_h, grid_w, grid_d = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([grid_h, grid_w, grid_d], dim=0).unsqueeze(0)  # (1, 3, H, W, D)
        grid = grid + displacement

        # Normalize to [-1, 1] (same as MONAI's Resample normalization)
        for i, s in enumerate([H, W, D]):
            grid[:, i] = grid[:, i] * 2.0 / (s - 1)

        # Convert to grid_sample format: (B, H, W, D, 3) with reversed axis order
        grid = grid.permute(0, 2, 3, 4, 1).flip(-1)

        ba_out = F.grid_sample(
            input_5d.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(ba_out[0], monai_out, atol=1e-4), (
            f"max diff: {(ba_out[0] - monai_out).abs().max().item()}"
        )

    def test_match_monai_via_class(self, device):
        """Test the full Rand3DElastic class matches MONAI by injecting same params."""
        torch.manual_seed(42)
        input_4d = torch.rand(1, 16, 16, 16, device=device)
        input_5d = input_4d.unsqueeze(0)
        H, W, D = 16, 16, 16

        sigma_val = 5.0
        mag_val = 0.5
        monai_t = monai.transforms.Rand3DElastic(
            sigma_range=(sigma_val, sigma_val),
            magnitude_range=(mag_val, mag_val),
            prob=1.0,
            mode="bilinear",
            padding_mode="zeros",
        )
        monai_t.randomize(grid_size=(H, W, D))
        monai_out = monai_t(input_4d, randomize=False)

        # Extract MONAI's displacement
        rand_offset = torch.from_numpy(monai_t.rand_offset).to(device).float()
        m_sigma = float(monai_t.sigma)
        m_magnitude = float(monai_t.magnitude)

        # Use batchaug class — reconstruct the grid the same way sample_params does
        ba_t = batchaug.Rand3DElastic(
            prob=1.0, sigma_range=(sigma_val, sigma_val),
            magnitude_range=(mag_val, mag_val),
            mode="bilinear", padding_mode="zeros",
        )

        # Build params dict manually with MONAI's displacement
        displacement = rand_offset.unsqueeze(0)  # (1, 3, H, W, D)
        sigma_tensor = torch.tensor([m_sigma], device=device)
        kernel = gaussian_1d_batch(sigma_tensor, truncated=3.0)
        for i in range(3):
            comp = displacement[:, i : i + 1, :, :, :]
            comp = separable_gaussian_conv3d(comp, kernel, kernel, kernel)
            displacement[:, i : i + 1, :, :, :] = comp
        displacement = displacement * m_magnitude

        coords = [
            torch.linspace(-(s - 1) / 2.0, (s - 1) / 2.0, s, device=device, dtype=torch.float32)
            for s in [H, W, D]
        ]
        grid_h, grid_w, grid_d = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([grid_h, grid_w, grid_d], dim=0).unsqueeze(0)
        grid = grid + displacement

        sizes = torch.tensor([H, W, D], device=device, dtype=torch.float32)
        half = (sizes - 1) / 2.0
        grid = grid / half.view(1, 3, 1, 1, 1)
        grid = grid.permute(0, 2, 3, 4, 1).flip(-1)

        params = {
            "mask": torch.tensor([True], device=device),
            "grid": grid,
        }
        ba_out = ba_t.apply(input_5d, params)

        assert torch.allclose(ba_out[0], monai_out, atol=1e-4), (
            f"max diff: {(ba_out[0] - monai_out).abs().max().item()}"
        )


class TestRand3DElasticd:
    """Dictionary wrapper tests."""

    def test_dict_paired(self, device):
        # Use larger volume with low sigma so displacement is significant
        torch.manual_seed(42)
        vol = torch.rand(2, 1, 32, 32, 32, device=device)
        seg = torch.randint(0, 5, (2, 1, 32, 32, 32), device=device).float()
        t = batchaug.Rand3DElasticd(
            keys=["vol", "seg"],
            prob=1.0,
            sigma_range=(0.5, 0.5),
            magnitude_range=(5.0, 5.0),
            mode={"vol": "bilinear", "seg": "nearest"},
        )
        data = {"vol": vol, "seg": seg}
        result = t(data)
        # Both should be transformed
        assert not torch.equal(result["vol"], vol)
        assert not torch.equal(result["seg"], seg)
        # Seg with nearest should preserve label values
        unique_vals = result["seg"].unique()
        expected = {0.0, 1.0, 2.0, 3.0, 4.0}
        assert all(v.item() in expected for v in unique_vals)


class TestRand3DElasticNonCubic:
    """Non-cubic spatial shapes."""

    def test_nonsquare(self, vol_nonsquare, device):
        t = batchaug.Rand3DElastic(
            prob=1.0, sigma_range=(3.0, 3.0), magnitude_range=(0.5, 0.5)
        )
        out = t(vol_nonsquare)
        assert out.shape == vol_nonsquare.shape
        assert not torch.equal(out, vol_nonsquare)


class TestRand3DElasticDtype:
    """dtype preservation."""

    def test_bfloat16(self, vol_bf16):
        t = batchaug.Rand3DElastic(
            prob=1.0, sigma_range=(3.0, 3.0), magnitude_range=(0.5, 0.5)
        )
        out = t(vol_bf16)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestRand3DElasticBatchIndependence:
    """Each batch element gets different deformation."""

    def test_different_per_element(self, vol, device):
        t = batchaug.Rand3DElastic(
            prob=1.0, sigma_range=(3.0, 3.0), magnitude_range=(1.0, 1.0)
        )
        out = t(vol)
        # Elements should differ from each other (different random displacements)
        diffs = []
        for i in range(vol.shape[0]):
            diffs.append((out[i] - vol[i]).abs().mean().item())
        # Not all diffs should be identical
        assert max(diffs) - min(diffs) > 1e-4

    def test_single_channel(self, device):
        vol = torch.rand(3, 1, 16, 16, 16, device=device)
        t = batchaug.Rand3DElastic(
            prob=1.0, sigma_range=(3.0, 3.0), magnitude_range=(0.5, 0.5)
        )
        out = t(vol)
        assert out.shape == vol.shape
        assert not torch.equal(out, vol)
