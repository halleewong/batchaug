import torch
import monai.transforms

import batchaug


class TestRandSimulateLowResolutionMatchesMONAI:
    """B=1: batchaug low-res simulation matches MONAI."""

    def test_matches_monai_zoom_0_7(self, vol, device):
        zoom = 0.7
        input_4d = vol[0]
        input_5d = vol[:1]

        monai_t = monai.transforms.RandSimulateLowResolution(
            prob=1.0, zoom_range=(zoom, zoom)
        )
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandSimulateLowResolution(
            prob=1.0, zoom_range=(zoom, zoom), downsample_mode="nearest"
        )  # downsample_mode="nearest" matches MONAI (BatchAug defaults to "nearest-exact")
        ba_out = ba_t(input_5d)

        assert torch.allclose(
            ba_out[0], torch.as_tensor(monai_out), atol=1e-5
        )

    def test_matches_monai_zoom_0_5(self, vol, device):
        zoom = 0.5
        input_4d = vol[0]
        input_5d = vol[:1]

        monai_t = monai.transforms.RandSimulateLowResolution(
            prob=1.0, zoom_range=(zoom, zoom)
        )
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandSimulateLowResolution(
            prob=1.0, zoom_range=(zoom, zoom), downsample_mode="nearest"
        )  # downsample_mode="nearest" matches MONAI (BatchAug defaults to "nearest-exact")
        ba_out = ba_t(input_5d)

        assert torch.allclose(
            ba_out[0], torch.as_tensor(monai_out), atol=1e-5
        )

    def test_no_change_when_mask_false(self, vol, device):
        ba_t = batchaug.RandSimulateLowResolution(prob=1.0, zoom_range=(0.5, 0.5))
        params = ba_t.sample_params(1, vol[:1].shape, device)
        params["mask"] = torch.tensor([False], device=device)
        ba_out = ba_t.apply(vol[:1], params)
        assert torch.equal(ba_out, vol[:1])


class TestRandSimulateLowResolutionBatchIndependence:
    """B>1: each batch element gets different zoom."""

    def test_different_zoom_per_element(self, vol, device):
        t = batchaug.RandSimulateLowResolution(prob=1.0, zoom_range=(0.3, 0.9))
        result = t(vol)
        diffs = [
            not torch.equal(result[0], result[i]) for i in range(1, vol.shape[0])
        ]
        assert any(diffs)


class TestRandSimulateLowResolutionChannelConsistency:
    """Same zoom is applied to all channels within a batch element."""

    def test_identical_channels_stay_identical(self, device):
        base = torch.rand(2, 1, 16, 16, 16, device=device)
        vol = base.repeat(1, 4, 1, 1, 1)

        t = batchaug.RandSimulateLowResolution(
            prob=1.0, zoom_range=(0.5, 0.5)
        )
        result = t(vol)

        for b in range(2):
            for c in range(1, 4):
                assert torch.allclose(result[b, 0], result[b, c], atol=1e-6)


class TestRandSimulateLowResolutionNonCubic:
    """Non-cubic spatial shapes (H != W != D)."""

    def test_matches_monai_nonsquare(self, device):
        """Low-res simulation matches MONAI on non-cubic (1, 12, 16, 20)."""
        zoom = 0.7
        torch.manual_seed(0)
        input_4d = torch.rand(1, 12, 16, 20, device=device)
        input_5d = input_4d.unsqueeze(0)

        monai_t = monai.transforms.RandSimulateLowResolution(
            prob=1.0, zoom_range=(zoom, zoom)
        )
        monai_out = monai_t(input_4d)

        ba_t = batchaug.RandSimulateLowResolution(
            prob=1.0, zoom_range=(zoom, zoom), downsample_mode="nearest"
        )  # downsample_mode="nearest" matches MONAI (BatchAug defaults to "nearest-exact")
        ba_out = ba_t(input_5d)

        assert ba_out.shape == input_5d.shape
        assert torch.allclose(
            ba_out[0], torch.as_tensor(monai_out), atol=1e-5
        )

    def test_nonsquare_preserves_shape(self, vol_nonsquare, device):
        t = batchaug.RandSimulateLowResolution(prob=1.0, zoom_range=(0.5, 0.9))
        result = t(vol_nonsquare)
        assert result.shape == vol_nonsquare.shape
        assert not torch.isnan(result).any()


class TestRandSimulateLowResolutionBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandSimulateLowResolution(prob=1.0, zoom_range=(0.7, 0.7))
        result = t(vol_bf16)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestRandSimulateLowResolutionDefaultMode:
    """BatchAug defaults to downsample_mode='nearest-exact' (not MONAI's 'nearest')."""

    def test_default_is_nearest_exact(self):
        t = batchaug.RandSimulateLowResolution(prob=1.0)
        assert t.downsample_mode == "nearest-exact"
        assert t.upsample_mode == "trilinear"

    def test_default_matches_explicit_nearest_exact(self, vol, device):
        ba_default = batchaug.RandSimulateLowResolution(
            prob=1.0, zoom_range=(0.7, 0.7)
        )
        ba_explicit = batchaug.RandSimulateLowResolution(
            prob=1.0, zoom_range=(0.7, 0.7), downsample_mode="nearest-exact"
        )
        assert torch.equal(ba_default(vol), ba_explicit(vol))

    def test_default_reduces_shift_vs_monai(self, device):
        """Default 'nearest-exact' keeps a centered blob's center of mass put;
        MONAI's 'nearest' shifts it toward +H/+W/+D."""
        N = 64
        zoom = 0.7
        coords = torch.arange(N, dtype=torch.float32, device=device)
        center = (N - 1) / 2.0
        g = torch.exp(-((coords - center) ** 2) / (2 * 5.0 ** 2))
        blob = g[:, None, None] * g[None, :, None] * g[None, None, :]
        vol = blob[None, None]  # (1, 1, N, N, N)

        def com(x):
            x = x[0, 0].clamp(min=0)
            total = x.sum()
            return torch.stack([
                (x.sum(dim=(1, 2)) * coords).sum() / total,
                (x.sum(dim=(0, 2)) * coords).sum() / total,
                (x.sum(dim=(0, 1)) * coords).sum() / total,
            ])

        com_in = com(vol)
        out_default = batchaug.RandSimulateLowResolution(
            prob=1.0, zoom_range=(zoom, zoom)
        )(vol)
        out_monai = batchaug.RandSimulateLowResolution(
            prob=1.0, zoom_range=(zoom, zoom), downsample_mode="nearest"
        )(vol)

        shift_default = (com(out_default) - com_in).abs().max().item()
        shift_monai = (com(out_monai) - com_in).abs().max().item()

        assert shift_default < 0.1
        assert shift_monai > 0.4
        assert shift_default < shift_monai


class TestLowResFallbackMatchesFastPath:
    """The per-element fallback (used when the gather kernel overflows CUDA's grid-dim cap
    on very large tensors) must produce results identical to the fast grouped gather path."""

    def test_looped_matches_indexed(self, device):
        torch.manual_seed(0)
        x = torch.rand(6, 2, 16, 16, 16, device=device)
        mask = torch.tensor([True, False, True, True, False, True], device=device)
        # shared (0.5, 0.5) and (0.7, 0.7) target sizes plus masked-out elements
        zoom = torch.tensor([0.5, 0.9, 0.5, 0.7, 0.9, 0.7], device=device)
        t = batchaug.RandSimulateLowResolution(prob=1.0, zoom_range=(0.5, 0.5))
        spatial_shape = list(x.shape[2:])
        targets = torch.stack(
            [(s * zoom).round().long().clamp(min=1) for s in spatial_shape], dim=1
        )
        active = mask.nonzero(as_tuple=True)[0]
        fast = t._lowres_indexed(x, active, targets, spatial_shape)
        slow = t._lowres_looped(x, active, targets, spatial_shape)
        assert torch.equal(fast, slow)
        assert torch.equal(fast[~mask], x[~mask])
