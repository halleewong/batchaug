"""Tests verifying Triton contrast/scale kernels match PyTorch."""
import pytest
import torch

from batchaug.pytorch.intensity.contrast import (
    RandAdjustContrast as PT_RandAdjustContrast,
    ScaleIntensity as PT_ScaleIntensity,
)
from batchaug.triton.intensity.contrast import (
    RandAdjustContrast as TR_RandAdjustContrast,
    ScaleIntensity as TR_ScaleIntensity,
)


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def device():
    return torch.device("cuda")


# ── ScaleIntensity ────────────────────────────────────────────────────


class TestTritonScaleIntensity:
    """Triton ScaleIntensity matches PyTorch version."""

    def test_default_channel_wise(self, device):
        x = torch.rand(4, 3, 16, 16, 16, device=device)
        pt, tr = PT_ScaleIntensity(), TR_ScaleIntensity()
        params = pt.sample_params(4, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_channel_wise_false(self, device):
        x = torch.rand(4, 3, 16, 16, 16, device=device)
        pt = PT_ScaleIntensity(channel_wise=False)
        tr = TR_ScaleIntensity(channel_wise=False)
        params = pt.sample_params(4, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_custom_range(self, device):
        x = torch.rand(4, 3, 16, 16, 16, device=device)
        pt = PT_ScaleIntensity(minv=-1.0, maxv=1.0)
        tr = TR_ScaleIntensity(minv=-1.0, maxv=1.0)
        params = pt.sample_params(4, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_constant_input(self, device):
        """Edge case: min == max."""
        x = torch.full((2, 1, 8, 8, 8), 3.14, device=device)
        pt, tr = PT_ScaleIntensity(), TR_ScaleIntensity()
        params = pt.sample_params(2, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_factor(self, device):
        """Factor mode bypasses kernel — just multiply."""
        x = torch.rand(2, 1, 8, 8, 8, device=device)
        pt = PT_ScaleIntensity(factor=0.5)
        tr = TR_ScaleIntensity(factor=0.5)
        params = pt.sample_params(2, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-6)

    def test_single_channel(self, device):
        x = torch.rand(4, 1, 16, 16, 16, device=device)
        pt, tr = PT_ScaleIntensity(), TR_ScaleIntensity()
        params = pt.sample_params(4, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_nonsquare(self, device):
        x = torch.rand(2, 1, 12, 16, 20, device=device)
        pt, tr = PT_ScaleIntensity(), TR_ScaleIntensity()
        params = pt.sample_params(2, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_bfloat16(self, device):
        x = torch.rand(2, 3, 16, 16, 16, device=device, dtype=torch.bfloat16)
        pt, tr = PT_ScaleIntensity(), TR_ScaleIntensity()
        params = pt.sample_params(2, x.shape, device)
        pt_out = pt.apply(x, params)
        tr_out = tr.apply(x, params)
        assert tr_out.dtype == pt_out.dtype
        assert torch.allclose(pt_out.float(), tr_out.float(), atol=1e-2)


# ── RandAdjustContrast ────────────────────────────────────────────────


class TestTritonRandAdjustContrast:
    """Triton RandAdjustContrast matches PyTorch version."""

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5, 3.0])
    def test_fixed_gamma(self, device, gamma):
        x = torch.rand(4, 3, 16, 16, 16, device=device)
        pt = PT_RandAdjustContrast(prob=1.0, gamma=(gamma, gamma))
        tr = TR_RandAdjustContrast(prob=1.0, gamma=(gamma, gamma))
        torch.manual_seed(0)
        params = pt.sample_params(4, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_random_gamma_batch(self, device):
        x = torch.rand(8, 3, 16, 16, 16, device=device)
        pt = PT_RandAdjustContrast(prob=1.0, gamma=(0.5, 4.5))
        tr = TR_RandAdjustContrast(prob=1.0, gamma=(0.5, 4.5))
        torch.manual_seed(42)
        params = pt.sample_params(8, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_mask_preserves_unchanged(self, device):
        x = torch.rand(4, 3, 16, 16, 16, device=device)
        tr = TR_RandAdjustContrast(prob=1.0, gamma=(2.0, 2.0))
        torch.manual_seed(0)
        params = tr.sample_params(4, x.shape, device)
        params["mask"] = torch.tensor([True, False, True, False], device=device)
        result = tr.apply(x, params)
        assert torch.equal(result[1], x[1])
        assert torch.equal(result[3], x[3])

    def test_dtype_preserved(self, device):
        x = torch.rand(2, 1, 16, 16, 16, device=device)
        tr = TR_RandAdjustContrast(prob=1.0, gamma=(1.5, 1.5))
        torch.manual_seed(0)
        params = tr.sample_params(2, x.shape, device)
        result = tr.apply(x, params)
        assert result.dtype == x.dtype

    def test_bfloat16(self, device):
        x = torch.rand(2, 3, 16, 16, 16, device=device, dtype=torch.bfloat16)
        pt = PT_RandAdjustContrast(prob=1.0, gamma=(1.5, 1.5))
        tr = TR_RandAdjustContrast(prob=1.0, gamma=(1.5, 1.5))
        torch.manual_seed(0)
        params = pt.sample_params(2, x.shape, device)
        pt_out = pt.apply(x, params)
        tr_out = tr.apply(x, params)
        assert tr_out.dtype == pt_out.dtype
        assert torch.allclose(pt_out.float(), tr_out.float(), atol=1e-2)

    def test_nonsquare(self, device):
        x = torch.rand(2, 1, 12, 16, 20, device=device)
        pt = PT_RandAdjustContrast(prob=1.0, gamma=(2.0, 2.0))
        tr = TR_RandAdjustContrast(prob=1.0, gamma=(2.0, 2.0))
        torch.manual_seed(0)
        params = pt.sample_params(2, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)
