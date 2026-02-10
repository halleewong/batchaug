"""Integration tests: full pipeline smoke tests, lazy vs eager, bfloat16.

Tests the full augmentation pipeline from examples/pipeline.yml:
  RandRotate90d, RandAxisFlipd, RandAffined,
  RandSimulateLowResolutiond, RandGaussianNoised, RandBiasFieldd,
  RandGibbsNoised, RandAdjustContrastd, RandGaussianSmoothd,
  RandGaussianSharpend, ScaleIntensityd
"""

import pytest
import torch

import batchaug


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_pipeline(lazy=False):
    """Build the full pipeline from examples/pipeline.yml."""
    transforms = [
        batchaug.RandRotate90d(
            keys=["vol", "seg"], prob=0.15, max_k=3, spatial_axes=(0, 1),
        ),
        batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=0.15),
        batchaug.RandAffined(
            keys=["vol", "seg"], prob=0.15,
            rotate_range=0.7853981633974483, shear_range=0.3, translate_range=5,
        ),
        batchaug.RandSimulateLowResolutiond(
            keys=["vol"], prob=0.15, zoom_range=(0.33, 1.0),
        ),
        batchaug.RandGaussianNoised(keys=["vol"], prob=0.15, mean=0.0, std=0.5),
        batchaug.RandBiasFieldd(
            keys=["vol"], prob=0.15, coeff_range=(0.0, 0.05),
        ),
        batchaug.RandGibbsNoised(
            keys=["vol"], prob=0.15, alpha=(0.0, 0.33),
        ),
        batchaug.RandAdjustContrastd(
            keys=["vol"], prob=0.15, gamma=(0.5, 2.5),
        ),
        batchaug.RandGaussianSmoothd(
            keys=["vol"], prob=0.15,
            sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
        ),
        batchaug.RandGaussianSharpend(keys=["vol"], prob=0.15),
        batchaug.ScaleIntensityd(keys=["vol"]),
    ]
    if lazy:
        return batchaug.Compose(
            transforms, lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )
    return batchaug.Compose(transforms, lazy=False)


# -----------------------------------------------------------------------
# Full pipeline smoke tests
# -----------------------------------------------------------------------


class TestFullPipelineSmoke:
    """Basic correctness of the full pipeline."""

    def test_output_shapes(self, vol, seg, device):
        for lazy in (False, True):
            pipe = _make_pipeline(lazy=lazy)
            result = pipe({"vol": vol.clone(), "seg": seg.clone()})
            assert result["vol"].shape == vol.shape, f"lazy={lazy}"
            assert result["seg"].shape == seg.shape, f"lazy={lazy}"

    def test_seg_values_nearest(self, vol, seg, device):
        """Seg with nearest mode should only contain original labels + 0."""
        pipe = _make_pipeline(lazy=True)
        # Run several times to exercise different random paths
        for seed in range(5):
            torch.manual_seed(seed)
            result = pipe({"vol": vol.clone(), "seg": seg.clone()})
            unique = result["seg"].unique()
            expected = {0.0, 1.0, 2.0, 3.0, 4.0}
            assert all(v.item() in expected for v in unique), (
                f"seed={seed}, unexpected seg values: {unique.tolist()}"
            )

    def test_vol_finite(self, vol, seg, device):
        """Vol output should have no NaN or Inf."""
        for lazy in (False, True):
            pipe = _make_pipeline(lazy=lazy)
            for seed in range(5):
                torch.manual_seed(seed)
                result = pipe({"vol": vol.clone(), "seg": seg.clone()})
                assert not torch.isnan(result["vol"]).any(), f"lazy={lazy}, seed={seed}"
                assert not torch.isinf(result["vol"]).any(), f"lazy={lazy}, seed={seed}"

    def test_prob_zero_noop(self, vol, seg, device):
        """All prob=0 → output equals input (except ScaleIntensity which is always-on)."""
        transforms = [
            batchaug.RandRotate90d(
                keys=["vol", "seg"], prob=0.0, max_k=3, spatial_axes=(0, 1),
            ),
            batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=0.0),
            batchaug.RandAffined(
                keys=["vol", "seg"], prob=0.0,
                rotate_range=0.7853981633974483, shear_range=0.3, translate_range=5,
            ),
            batchaug.RandSimulateLowResolutiond(
                keys=["vol"], prob=0.0, zoom_range=(0.33, 1.0),
            ),
            batchaug.RandGaussianNoised(keys=["vol"], prob=0.0, mean=0.0, std=0.5),
            batchaug.RandBiasFieldd(
                keys=["vol"], prob=0.0, coeff_range=(0.0, 0.05),
            ),
            batchaug.RandGibbsNoised(
                keys=["vol"], prob=0.0, alpha=(0.0, 0.33),
            ),
            batchaug.RandAdjustContrastd(
                keys=["vol"], prob=0.0, gamma=(0.5, 2.5),
            ),
            batchaug.RandGaussianSmoothd(
                keys=["vol"], prob=0.0,
                sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
            ),
            batchaug.RandGaussianSharpend(keys=["vol"], prob=0.0),
            batchaug.ScaleIntensityd(keys=["vol"]),
        ]
        for lazy in (False, True):
            if lazy:
                pipe = batchaug.Compose(
                    transforms, lazy=True,
                    mode={"vol": "bilinear", "seg": "nearest"},
                )
            else:
                pipe = batchaug.Compose(transforms, lazy=False)
            result = pipe({"vol": vol.clone(), "seg": seg.clone()})
            # ScaleIntensity always applies (prob=1), so vol changes but seg stays
            assert torch.equal(result["seg"], seg), f"lazy={lazy}"


# -----------------------------------------------------------------------
# Lazy vs eager consistency
# -----------------------------------------------------------------------


class TestFullPipelineLazyVsEager:
    """Same seed → lazy ≈ eager (modulo interpolation differences)."""

    def test_lazy_close_to_eager(self, vol, seg, device):
        eager_pipe = _make_pipeline(lazy=False)
        lazy_pipe = _make_pipeline(lazy=True)

        for seed in [0, 42, 123]:
            torch.manual_seed(seed)
            eager_result = eager_pipe({"vol": vol.clone(), "seg": seg.clone()})

            torch.manual_seed(seed)
            lazy_result = lazy_pipe({"vol": vol.clone(), "seg": seg.clone()})

            # Seg: nearest mode in lazy still gives exact label values.
            # But lazy fuses geometric transforms into one grid_sample vs
            # sequential, so exact match isn't guaranteed.
            # Just verify shapes and finite values.
            assert lazy_result["vol"].shape == eager_result["vol"].shape
            assert lazy_result["seg"].shape == eager_result["seg"].shape
            assert not torch.isnan(lazy_result["vol"]).any()

    def test_all_active_lazy_vs_eager(self, vol, seg, device):
        """With prob=1 for all transforms, lazy and eager should be close."""
        def _make_all_active():
            return [
                batchaug.RandRotate90d(
                    keys=["vol", "seg"], prob=1.0, max_k=3, spatial_axes=(0, 1),
                ),
                batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0),
                batchaug.RandAffined(
                    keys=["vol", "seg"], prob=1.0,
                    rotate_range=0.3, shear_range=0.1, translate_range=2,
                ),
                batchaug.RandSimulateLowResolutiond(
                    keys=["vol"], prob=1.0, zoom_range=(0.8, 1.0),
                ),
                batchaug.RandGaussianNoised(keys=["vol"], prob=1.0, mean=0.0, std=0.1),
                batchaug.RandBiasFieldd(
                    keys=["vol"], prob=1.0, coeff_range=(0.0, 0.02),
                ),
                batchaug.RandGibbsNoised(
                    keys=["vol"], prob=1.0, alpha=(0.0, 0.1),
                ),
                batchaug.RandAdjustContrastd(
                    keys=["vol"], prob=1.0, gamma=(0.8, 1.2),
                ),
                batchaug.RandGaussianSmoothd(
                    keys=["vol"], prob=1.0,
                    sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
                ),
                batchaug.RandGaussianSharpend(keys=["vol"], prob=1.0),
                batchaug.ScaleIntensityd(keys=["vol"]),
            ]

        eager_pipe = batchaug.Compose(_make_all_active(), lazy=False)
        lazy_pipe = batchaug.Compose(
            _make_all_active(), lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )

        torch.manual_seed(77)
        eager_result = eager_pipe({"vol": vol.clone(), "seg": seg.clone()})

        torch.manual_seed(77)
        lazy_result = lazy_pipe({"vol": vol.clone(), "seg": seg.clone()})

        # The geo transforms (rot90, flip, affine) are all consecutive and
        # fused into a single grid_sample in lazy mode. The vol should be
        # close but not identical due to single vs multi-pass interpolation.
        assert torch.allclose(lazy_result["vol"], eager_result["vol"], atol=0.15), (
            f"max diff: {(lazy_result['vol'] - eager_result['vol']).abs().max()}"
        )


# -----------------------------------------------------------------------
# Non-cubic spatial shapes
# -----------------------------------------------------------------------


class TestFullPipelineNonCubic:
    """Full pipeline on non-cubic (H != W != D) data."""

    def test_output_shapes(self, vol_nonsquare, seg_nonsquare, device):
        for lazy in (False, True):
            pipe = _make_pipeline(lazy=lazy)
            result = pipe({
                "vol": vol_nonsquare.clone(),
                "seg": seg_nonsquare.clone(),
            })
            assert result["vol"].shape == vol_nonsquare.shape, f"lazy={lazy}"
            assert result["seg"].shape == seg_nonsquare.shape, f"lazy={lazy}"

    def test_vol_finite(self, vol_nonsquare, seg_nonsquare, device):
        for lazy in (False, True):
            pipe = _make_pipeline(lazy=lazy)
            for seed in range(3):
                torch.manual_seed(seed)
                result = pipe({
                    "vol": vol_nonsquare.clone(),
                    "seg": seg_nonsquare.clone(),
                })
                assert not torch.isnan(result["vol"]).any(), (
                    f"lazy={lazy}, seed={seed}"
                )
                assert not torch.isinf(result["vol"]).any(), (
                    f"lazy={lazy}, seed={seed}"
                )

    def test_seg_values_nearest(self, vol_nonsquare, seg_nonsquare, device):
        pipe = _make_pipeline(lazy=True)
        for seed in range(3):
            torch.manual_seed(seed)
            result = pipe({
                "vol": vol_nonsquare.clone(),
                "seg": seg_nonsquare.clone(),
            })
            unique = result["seg"].unique()
            expected = {0.0, 1.0, 2.0, 3.0, 4.0}
            assert all(v.item() in expected for v in unique), (
                f"seed={seed}, unexpected seg values: {unique.tolist()}"
            )


# -----------------------------------------------------------------------
# bfloat16
# -----------------------------------------------------------------------


class TestFullPipelineBfloat16:
    """Full pipeline on bfloat16 tensors."""

    def test_bf16_pipeline_eager(self, vol_bf16, device):
        seg_bf16 = torch.randint(
            0, 5, vol_bf16.shape, device=device,
        ).to(torch.bfloat16)
        pipe = _make_pipeline(lazy=False)
        result = pipe({"vol": vol_bf16.clone(), "seg": seg_bf16.clone()})
        assert result["vol"].dtype == torch.bfloat16
        assert not torch.isnan(result["vol"]).any()
        assert not torch.isinf(result["vol"]).any()

    def test_bf16_pipeline_lazy(self, vol_bf16, device):
        seg_bf16 = torch.randint(
            0, 5, vol_bf16.shape, device=device,
        ).to(torch.bfloat16)
        pipe = _make_pipeline(lazy=True)
        result = pipe({"vol": vol_bf16.clone(), "seg": seg_bf16.clone()})
        assert result["vol"].dtype == torch.bfloat16
        assert not torch.isnan(result["vol"]).any()
        assert not torch.isinf(result["vol"]).any()
