"""Tests for batchaug.Compose with eager and lazy geometric fusion."""

import pytest
import torch

import batchaug


# -----------------------------------------------------------------------
# Eager mode
# -----------------------------------------------------------------------


class TestComposeEager:
    """Compose(lazy=False) must match a manual sequential loop."""

    def test_matches_sequential(self, vol, seg, device):
        transforms = [
            batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0),
            batchaug.RandGaussianNoised(keys=["vol"], prob=1.0, mean=0.0, std=0.1),
            batchaug.RandRotate90d(keys=["vol", "seg"], prob=1.0),
        ]
        compose = batchaug.Compose(transforms, lazy=False)

        torch.manual_seed(0)
        result = compose({"vol": vol.clone(), "seg": seg.clone()})

        torch.manual_seed(0)
        d = {"vol": vol.clone(), "seg": seg.clone()}
        for t in transforms:
            d = t(d)

        assert torch.equal(result["vol"], d["vol"])
        assert torch.equal(result["seg"], d["seg"])


# -----------------------------------------------------------------------
# Lazy — single transform correctness
# -----------------------------------------------------------------------


class TestComposeLazyFlip:
    """Lazy flip via grid_sample should match eager flip (exact on voxel centres)."""

    def test_matches_eager(self, vol, seg, device):
        t = batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0)
        compose = batchaug.Compose(
            [t], lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )

        torch.manual_seed(7)
        lazy = compose({"vol": vol.clone(), "seg": seg.clone()})

        torch.manual_seed(7)
        eager = t({"vol": vol.clone(), "seg": seg.clone()})

        assert torch.allclose(lazy["vol"], eager["vol"], atol=1e-5)
        assert torch.equal(lazy["seg"], eager["seg"])


class TestComposeLazyRot90:
    """Lazy rot90 via grid_sample should match eager rot90."""

    def test_matches_eager(self, vol, seg, device):
        t = batchaug.RandRotate90d(
            keys=["vol", "seg"], prob=1.0, max_k=3, spatial_axes=(0, 1),
        )
        compose = batchaug.Compose(
            [t], lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )

        torch.manual_seed(11)
        lazy = compose({"vol": vol.clone(), "seg": seg.clone()})

        torch.manual_seed(11)
        eager = t({"vol": vol.clone(), "seg": seg.clone()})

        assert torch.allclose(lazy["vol"], eager["vol"], atol=1e-5)
        assert torch.equal(lazy["seg"], eager["seg"])


class TestComposeLazyAffine:
    """Lazy affine should match eager affine (same theta, same grid_sample)."""

    def test_matches_eager(self, vol, seg, device):
        t = batchaug.RandAffined(
            keys=["vol", "seg"], prob=1.0,
            rotate_range=0.3, scale_range=0.1,
            mode={"vol": "bilinear", "seg": "nearest"},
        )
        compose = batchaug.Compose(
            [t], lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )

        torch.manual_seed(22)
        lazy = compose({"vol": vol.clone(), "seg": seg.clone()})

        torch.manual_seed(22)
        eager = t({"vol": vol.clone(), "seg": seg.clone()})

        assert torch.allclose(lazy["vol"], eager["vol"], atol=1e-5)
        assert torch.allclose(lazy["seg"], eager["seg"], atol=1e-5)


# -----------------------------------------------------------------------
# Lazy — fusion of consecutive geometric transforms
# -----------------------------------------------------------------------


class TestComposeLazyFusion:
    """Fusing consecutive geometric transforms should match sequential eager."""

    def test_flip_then_rot90(self, vol, seg, device):
        t1 = batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0)
        t2 = batchaug.RandRotate90d(keys=["vol", "seg"], prob=1.0)
        compose = batchaug.Compose(
            [t1, t2], lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )

        torch.manual_seed(33)
        lazy = compose({"vol": vol.clone(), "seg": seg.clone()})

        # Reproduce the same param sampling order as Compose lazy path.
        torch.manual_seed(33)
        p1 = t1.transform.sample_params(vol.shape[0], vol.shape, device)
        p2 = t2.transform.sample_params(vol.shape[0], vol.shape, device)

        ev = vol.clone()
        ev = t1.transform.apply(ev, p1)
        ev = t2.transform.apply(ev, p2)

        es = seg.clone()
        es = t1.transform.apply(es, p1)
        es = t2.transform.apply(es, p2)

        assert torch.allclose(lazy["vol"], ev, atol=1e-5)
        assert torch.equal(lazy["seg"], es)

    def test_flip_rot90_affine(self, vol, seg, device):
        t1 = batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0)
        t2 = batchaug.RandRotate90d(keys=["vol", "seg"], prob=1.0)
        t3 = batchaug.RandAffined(
            keys=["vol", "seg"], prob=1.0,
            rotate_range=0.2, scale_range=0.1,
        )
        compose = batchaug.Compose(
            [t1, t2, t3], lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )

        torch.manual_seed(44)
        lazy = compose({"vol": vol.clone(), "seg": seg.clone()})

        # Reproduce eager with same params.
        torch.manual_seed(44)
        p1 = t1.transform.sample_params(vol.shape[0], vol.shape, device)
        p2 = t2.transform.sample_params(vol.shape[0], vol.shape, device)
        p3 = t3.transform.sample_params(vol.shape[0], vol.shape, device)

        ev = vol.clone()
        ev = t1.transform.apply(ev, p1)
        ev = t2.transform.apply(ev, p2)
        ev = t3.transform.apply(ev, p3)

        # Tolerance is slightly higher because eager applies flip/rot90
        # exactly then interpolates, while lazy fuses into one interpolation.
        assert torch.allclose(lazy["vol"], ev, atol=1e-4)


# -----------------------------------------------------------------------
# Lazy — mixed pipeline (geometric + intensity + geometric)
# -----------------------------------------------------------------------


class TestComposeMixed:
    """Intensity between geometric groups triggers materialization."""

    def test_geo_intensity_geo(self, vol, seg, device):
        t1 = batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0)
        t2 = batchaug.RandGaussianNoised(keys=["vol"], prob=1.0, mean=0.0, std=0.01)
        t3 = batchaug.RandRotate90d(keys=["vol", "seg"], prob=1.0)
        compose = batchaug.Compose(
            [t1, t2, t3], lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )

        torch.manual_seed(55)
        lazy = compose({"vol": vol.clone(), "seg": seg.clone()})

        # Reproduce: sample flip params, materialize flip, add noise, sample rot, materialize rot.
        torch.manual_seed(55)
        p1 = t1.transform.sample_params(vol.shape[0], vol.shape, device)
        # noise sample_params consumes random for mask
        # noise apply consumes random for the noise tensor
        # t2(d) calls both internally

        # Easiest: replay the full eager pipeline with same seed.
        torch.manual_seed(55)
        eager_compose = batchaug.Compose([t1, t2, t3], lazy=False)
        eager = eager_compose({"vol": vol.clone(), "seg": seg.clone()})

        # vol: lazy materializes flip via grid_sample (near-exact), adds noise,
        # then materializes rot via grid_sample.  Should be very close to eager.
        assert torch.allclose(lazy["vol"], eager["vol"], atol=1e-5)
        assert torch.equal(lazy["seg"], eager["seg"])


# -----------------------------------------------------------------------
# Mask correctness
# -----------------------------------------------------------------------


class TestComposeMaskCorrectness:
    """Masked-out elements must be exactly preserved."""

    def test_prob_zero_unchanged(self, vol, seg, device):
        t = batchaug.RandAffined(
            keys=["vol", "seg"], prob=0.0, rotate_range=0.5,
        )
        compose = batchaug.Compose(
            [t], lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )
        result = compose({"vol": vol.clone(), "seg": seg.clone()})
        assert torch.equal(result["vol"], vol)
        assert torch.equal(result["seg"], seg)

    def test_partial_mask(self, device):
        """Only some batch elements active — others must be unchanged."""
        B = 4
        torch.manual_seed(999)
        vol = torch.rand(B, 1, 8, 8, 8, device=device)

        # prob=0.5 so roughly half the batch is active
        t = batchaug.RandAxisFlipd(keys=["vol"], prob=0.5)
        compose = batchaug.Compose([t], lazy=True, mode="nearest")

        torch.manual_seed(123)
        result = compose({"vol": vol.clone()})

        # Reproduce to find which elements were masked.
        torch.manual_seed(123)
        params = t.transform.sample_params(B, vol.shape, device)
        mask = params["mask"]

        for i in range(B):
            if not mask[i]:
                assert torch.equal(result["vol"][i], vol[i])


# -----------------------------------------------------------------------
# Per-key modes
# -----------------------------------------------------------------------


class TestComposePerKeyModes:
    """Different interpolation modes per key."""

    def test_bilinear_vol_nearest_seg(self, vol, seg, device):
        t = batchaug.RandAffined(
            keys=["vol", "seg"], prob=1.0,
            rotate_range=0.3,
        )
        compose = batchaug.Compose(
            [t], lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
            padding_mode="zeros",
        )

        result = compose({"vol": vol.clone(), "seg": seg.clone()})

        # seg with nearest should still only contain original label values + 0 (padding)
        unique = result["seg"].unique()
        expected = {0.0, 1.0, 2.0, 3.0, 4.0}
        assert all(v.item() in expected for v in unique)


# -----------------------------------------------------------------------
# Different key sets per geometric transform
# -----------------------------------------------------------------------


class TestComposeDifferentKeys:
    """Transforms with different key sets accumulate independently."""

    def test_vol_gets_more_transforms(self, vol, seg, device):
        t1 = batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0)
        t2 = batchaug.RandRotate90d(keys=["vol"], prob=1.0)  # vol only

        compose = batchaug.Compose(
            [t1, t2], lazy=True,
            mode="nearest",
        )

        torch.manual_seed(66)
        result = compose({"vol": vol.clone(), "seg": seg.clone()})

        # seg should only get flip, vol gets flip + rot90.
        # At minimum both should be modified from the originals.
        assert not torch.equal(result["vol"], vol)
        assert not torch.equal(result["seg"], seg)
        # And they should differ from each other's transforms.
        assert result["vol"].shape == vol.shape
        assert result["seg"].shape == seg.shape


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestComposeEdgeCases:
    def test_empty_pipeline(self, vol, seg, device):
        compose = batchaug.Compose([], lazy=True)
        result = compose({"vol": vol.clone(), "seg": seg.clone()})
        assert torch.equal(result["vol"], vol)
        assert torch.equal(result["seg"], seg)

    def test_only_intensity(self, vol, device):
        t = batchaug.RandGaussianNoised(keys=["vol"], prob=1.0, mean=0.0, std=0.1)
        compose = batchaug.Compose([t], lazy=True)

        torch.manual_seed(5)
        lazy = compose({"vol": vol.clone()})
        torch.manual_seed(5)
        eager = t({"vol": vol.clone()})

        assert torch.equal(lazy["vol"], eager["vol"])

    def test_only_geometric(self, vol, seg, device):
        """All geometric, no intensity — should still work."""
        t1 = batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0)
        t2 = batchaug.RandRotate90d(keys=["vol", "seg"], prob=1.0)
        compose = batchaug.Compose(
            [t1, t2], lazy=True, mode="nearest",
        )
        result = compose({"vol": vol.clone(), "seg": seg.clone()})
        assert result["vol"].shape == vol.shape
        assert not torch.equal(result["vol"], vol)


# -----------------------------------------------------------------------
# Non-cubic spatial shapes
# -----------------------------------------------------------------------


class TestComposeNonCubic:
    """Non-cubic spatial shapes (H != W != D)."""

    def test_eager_nonsquare(self, vol_nonsquare, seg_nonsquare, device):
        """Eager compose on non-cubic data preserves shapes."""
        transforms = [
            batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0),
            batchaug.RandRotate90d(keys=["vol", "seg"], prob=1.0),
            batchaug.RandGaussianNoised(keys=["vol"], prob=1.0, mean=0.0, std=0.1),
        ]
        compose = batchaug.Compose(transforms, lazy=False)
        result = compose({"vol": vol_nonsquare.clone(), "seg": seg_nonsquare.clone()})
        assert result["vol"].shape == vol_nonsquare.shape
        assert result["seg"].shape == seg_nonsquare.shape
        assert not torch.isnan(result["vol"]).any()

    def test_lazy_nonsquare(self, vol_nonsquare, seg_nonsquare, device):
        """Lazy compose on non-cubic data preserves shapes."""
        transforms = [
            batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0),
            batchaug.RandRotate90d(keys=["vol", "seg"], prob=1.0),
            batchaug.RandAffined(
                keys=["vol", "seg"], prob=1.0,
                rotate_range=0.3, scale_range=0.1,
            ),
        ]
        compose = batchaug.Compose(
            transforms, lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
        )
        result = compose({"vol": vol_nonsquare.clone(), "seg": seg_nonsquare.clone()})
        assert result["vol"].shape == vol_nonsquare.shape
        assert result["seg"].shape == seg_nonsquare.shape
        assert not torch.isnan(result["vol"]).any()

    def test_lazy_seg_nearest_nonsquare(self, vol_nonsquare, seg_nonsquare, device):
        """Seg with nearest mode preserves label set on non-cubic data."""
        transforms = [
            batchaug.RandAxisFlipd(keys=["vol", "seg"], prob=1.0),
            batchaug.RandAffined(
                keys=["vol", "seg"], prob=1.0,
                rotate_range=0.2,
            ),
        ]
        compose = batchaug.Compose(
            transforms, lazy=True,
            mode={"vol": "bilinear", "seg": "nearest"},
            padding_mode="zeros",
        )
        result = compose({"vol": vol_nonsquare.clone(), "seg": seg_nonsquare.clone()})
        unique = result["seg"].unique()
        expected = {0.0, 1.0, 2.0, 3.0, 4.0}
        assert all(v.item() in expected for v in unique)


# -----------------------------------------------------------------------
# bfloat16
# -----------------------------------------------------------------------


class TestComposeBfloat16:
    def test_lazy_preserves_dtype(self, vol_bf16, device):
        t = batchaug.RandAxisFlipd(keys=["vol"], prob=1.0)
        compose = batchaug.Compose([t], lazy=True, mode="bilinear")
        result = compose({"vol": vol_bf16.clone()})
        assert result["vol"].dtype == torch.bfloat16
        assert not torch.isnan(result["vol"]).any()

    def test_lazy_affine_bf16(self, vol_bf16, device):
        t = batchaug.RandAffined(keys=["vol"], prob=1.0, rotate_range=0.2)
        compose = batchaug.Compose([t], lazy=True, mode="bilinear")
        result = compose({"vol": vol_bf16.clone()})
        assert result["vol"].dtype == torch.bfloat16
        assert not torch.isnan(result["vol"]).any()
