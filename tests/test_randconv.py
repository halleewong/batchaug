"""Tests for batchaug.RandConv and RandConvd (PyTorch backend)."""

import pytest
import torch

import batchaug


class TestRandConvShapeAndDtype:
    """Output shape and dtype are preserved."""

    def test_shape_preserved(self, vol):
        t = batchaug.RandConv(prob=1.0)
        result = t(vol)
        assert result.shape == vol.shape

    def test_shape_nonsquare(self, vol_nonsquare):
        t = batchaug.RandConv(prob=1.0)
        result = t(vol_nonsquare)
        assert result.shape == vol_nonsquare.shape

    def test_dtype_float32_preserved(self, vol):
        t = batchaug.RandConv(prob=1.0)
        result = t(vol)
        assert result.dtype == torch.float32

    def test_dtype_bfloat16_preserved(self, vol_bf16):
        t = batchaug.RandConv(prob=1.0)
        result = t(vol_bf16)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_no_nan(self, vol):
        t = batchaug.RandConv(prob=1.0)
        result = t(vol)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestRandConvMask:
    """Masked-out batch elements are returned unchanged."""

    def test_all_masked_out(self, vol):
        t = batchaug.RandConv(prob=1.0)
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        params["mask"] = torch.zeros(vol.shape[0], dtype=torch.bool, device=vol.device)
        result = t.apply(vol, params)
        assert torch.equal(result, vol)

    def test_all_masked_in(self, vol):
        t = batchaug.RandConv(prob=1.0)
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        params["mask"] = torch.ones(vol.shape[0], dtype=torch.bool, device=vol.device)
        result = t.apply(vol, params)
        # Should differ from input (random conv changes values)
        assert not torch.equal(result, vol)

    def test_partial_mask(self, vol):
        """Unmasked elements match original; masked elements are changed."""
        t = batchaug.RandConv(prob=1.0)
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        # Mask only first batch element
        mask = torch.zeros(vol.shape[0], dtype=torch.bool, device=vol.device)
        mask[0] = True
        params["mask"] = mask
        result = t.apply(vol, params)
        # Unmasked elements unchanged
        assert torch.equal(result[1:], vol[1:])
        # Masked element changed
        assert not torch.equal(result[0], vol[0])


class TestRandConvBatchIndependence:
    """Each batch element receives independent random conv weights."""

    def test_different_outputs_per_element(self, vol):
        t = batchaug.RandConv(prob=1.0)
        result = t(vol)
        # With different random weights, outputs should differ across batch
        for i in range(1, vol.shape[0]):
            assert not torch.equal(result[0], result[i])

    def test_single_element_batch(self, device):
        x = torch.rand(1, 3, 16, 16, 16, device=device)
        t = batchaug.RandConv(prob=1.0)
        result = t(x)
        assert result.shape == x.shape
        assert not torch.equal(result, x)


class TestRandConvChannelConsistency:
    """Same conv kernel is applied independently to every channel within a batch element."""

    def test_weights_shape(self, vol):
        """Weights have shape (B, 1, ks, ks, ks) — one depthwise kernel per batch element."""
        t = batchaug.RandConv(prob=1.0, kernel_sizes=3)
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        B, C, H, W, D = vol.shape
        ks = params["kernel_size"]
        assert params["weights"].shape == (B, 1, ks, ks, ks)

    def test_same_kernel_applied_to_all_channels(self, device):
        """Identical channels in → identical channels out.

        If all input channels are the same, and the same kernel is applied to
        each channel independently, all output channels must also be the same.
        """
        B, C = 2, 4
        # All channels identical within each batch element
        single = torch.rand(B, 1, 16, 16, 16, device=device)
        x = single.expand(B, C, 16, 16, 16).contiguous()

        t = batchaug.RandConv(prob=1.0)
        params = t.sample_params(B, x.shape, device)
        params["mask"] = torch.ones(B, dtype=torch.bool, device=device)
        result = t.apply(x, params)

        # Every channel in a batch element should be identical
        for b in range(B):
            for c in range(1, C):
                assert torch.allclose(result[b, 0], result[b, c], atol=1e-5), (
                    f"batch element {b}: channel 0 and channel {c} differ"
                )

    def test_different_kernels_across_batch_elements(self, device):
        """Different batch elements get different kernels (channels still match within each)."""
        B, C = 2, 4
        single = torch.rand(B, 1, 16, 16, 16, device=device)
        x = single.expand(B, C, 16, 16, 16).contiguous()

        t = batchaug.RandConv(prob=1.0)
        params = t.sample_params(B, x.shape, device)
        params["mask"] = torch.ones(B, dtype=torch.bool, device=device)
        result = t.apply(x, params)

        # Batch elements should differ (independent kernels)
        assert not torch.allclose(result[0, 0], result[1, 0], atol=1e-5)


class TestRandConvMixingMode:
    """RC_mix mode blends input and convolved output."""

    def test_mixing_output_is_blend(self, vol):
        t = batchaug.RandConv(prob=1.0, mixing=True)
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        # Force alpha=0: output should equal input
        params["alpha"] = torch.zeros(vol.shape[0], device=vol.device)
        result = t.apply(vol, params)
        assert torch.allclose(result, vol, atol=1e-5)

    def test_mixing_alpha_one_equals_no_mixing(self, vol):
        t_mix = batchaug.RandConv(prob=1.0, mixing=True)
        t_plain = batchaug.RandConv(prob=1.0, mixing=False)
        params = t_mix.sample_params(vol.shape[0], vol.shape, vol.device)
        params["alpha"] = torch.ones(vol.shape[0], device=vol.device)
        result_mix = t_mix.apply(vol, params)

        # Use the same weights for plain
        params_plain = {k: v for k, v in params.items() if k != "alpha"}
        result_plain = t_plain.apply(vol, params_plain)
        assert torch.allclose(result_mix, result_plain, atol=1e-5)

    def test_mixing_alpha_per_element(self, vol):
        """Different alpha per batch element produces per-element blends."""
        t = batchaug.RandConv(prob=1.0, mixing=True)
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        # alpha=0 for all → output equals input
        params["alpha"] = torch.zeros(vol.shape[0], device=vol.device)
        result = t.apply(vol, params)
        assert torch.allclose(result.float(), vol.float(), atol=1e-5)

    def test_mixing_shape_preserved(self, vol):
        """Mixing mode preserves the input shape."""
        t = batchaug.RandConv(prob=1.0, mixing=True)
        result = t(vol)
        assert result.shape == vol.shape


class TestRandConvKernelSizes:
    """Kernel size is sampled from the provided list."""

    def test_kernel_size_in_list(self, vol):
        kernel_sizes = [1, 3, 5]
        t = batchaug.RandConv(prob=1.0, kernel_sizes=kernel_sizes)
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        assert params["kernel_size"] in kernel_sizes

    def test_kernel_size_1(self, vol):
        t = batchaug.RandConv(prob=1.0, kernel_sizes=1)
        result = t(vol)
        assert result.shape == vol.shape

    def test_kernel_size_5(self, vol):
        t = batchaug.RandConv(prob=1.0, kernel_sizes=5)
        result = t(vol)
        assert result.shape == vol.shape

    def test_kernel_sizes_varied_over_calls(self, device):
        """With multiple kernel sizes, different sizes appear over many calls."""
        vol = torch.rand(2, 2, 8, 8, 8, device=device)
        t = batchaug.RandConv(prob=1.0, kernel_sizes=[1, 3, 5])
        seen = set()
        for _ in range(50):
            params = t.sample_params(vol.shape[0], vol.shape, vol.device)
            seen.add(params["kernel_size"])
        # Over 50 draws, expect to have seen at least 2 distinct sizes
        assert len(seen) >= 2


class TestRandConvDistributions:
    """All supported weight distributions initialise without error."""

    @pytest.mark.parametrize(
        "distribution",
        ["kaiming_normal", "kaiming_uniform", "xavier_normal"],
    )
    def test_distribution(self, vol, distribution):
        t = batchaug.RandConv(prob=1.0, distribution=distribution)
        result = t(vol)
        assert result.shape == vol.shape
        assert not torch.isnan(result).any()

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            batchaug.RandConv(distribution="bad_dist")


class TestRandConvBias:
    """rand_bias=True adds a random bias term."""

    def test_bias_in_params(self, vol):
        t = batchaug.RandConv(prob=1.0, rand_bias=True)
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        assert params["bias"] is not None
        # One bias per batch element (shared across channels)
        assert params["bias"].shape == (vol.shape[0],)

    def test_no_bias_in_params_by_default(self, vol):
        t = batchaug.RandConv(prob=1.0)
        params = t.sample_params(vol.shape[0], vol.shape, vol.device)
        assert params["bias"] is None

    def test_rand_bias_output_differs_from_no_bias(self, vol):
        torch.manual_seed(0)
        t_bias = batchaug.RandConv(prob=1.0, rand_bias=True)
        torch.manual_seed(0)
        t_no_bias = batchaug.RandConv(prob=1.0, rand_bias=False)
        result_bias = t_bias(vol)
        result_no_bias = t_no_bias(vol)
        assert not torch.equal(result_bias, result_no_bias)




class TestRandConvd:
    """Dictionary wrapper applies same params to all keys."""

    def test_same_transform_applied_to_all_keys(self, device):
        """When mask is True for all, both vol and seg are changed."""
        vol = torch.rand(2, 3, 16, 16, 16, device=device)
        seg = torch.rand(2, 3, 16, 16, 16, device=device)
        t = batchaug.RandConvd(keys=["vol", "seg"], prob=1.0)
        result = t({"vol": vol, "seg": seg})
        assert result["vol"].shape == vol.shape
        assert result["seg"].shape == seg.shape
        assert not torch.equal(result["vol"], vol)
        assert not torch.equal(result["seg"], seg)

    def test_same_weights_applied_to_all_keys(self, device):
        """Identical input tensors produce identical outputs under dict transform."""
        x = torch.rand(2, 3, 16, 16, 16, device=device)
        t = batchaug.RandConvd(keys=["a", "b"], prob=1.0)
        result = t({"a": x.clone(), "b": x.clone()})
        assert torch.allclose(result["a"], result["b"], atol=1e-6)

    def test_mask_preserves_key(self, device):
        vol = torch.rand(2, 3, 16, 16, 16, device=device)
        seg = torch.rand(2, 3, 16, 16, 16, device=device)
        t = batchaug.RandConvd(keys=["vol", "seg"], prob=0.0)
        result = t({"vol": vol, "seg": seg})
        assert torch.equal(result["vol"], vol)
        assert torch.equal(result["seg"], seg)

    def test_missing_key_ignored(self, device):
        vol = torch.rand(2, 3, 16, 16, 16, device=device)
        t = batchaug.RandConvd(keys=["vol"], prob=1.0)
        result = t({"vol": vol, "extra": 42})
        assert result["extra"] == 42
