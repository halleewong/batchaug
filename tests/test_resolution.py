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
            prob=1.0, zoom_range=(zoom, zoom)
        )
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
            prob=1.0, zoom_range=(zoom, zoom)
        )
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


class TestRandSimulateLowResolutionBfloat16:
    """Transform works with bfloat16."""

    def test_preserves_dtype(self, vol_bf16):
        t = batchaug.RandSimulateLowResolution(prob=1.0, zoom_range=(0.7, 0.7))
        result = t(vol_bf16)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
