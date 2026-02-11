"""Tests verifying Triton smooth kernel matches PyTorch."""
import pytest
import torch

from batchaug.pytorch.intensity.smooth import RandGaussianSmooth as PT_Smooth
from batchaug.triton.intensity.smooth import RandGaussianSmooth as TR_Smooth


@pytest.fixture
def device():
    return torch.device("cuda")


class TestTritonRandGaussianSmooth:
    """Triton RandGaussianSmooth matches PyTorch version."""

    def test_uniform_sigma(self, device):
        x = torch.rand(4, 3, 16, 16, 16, device=device)
        pt = PT_Smooth(prob=1.0, sigma_x=(0.8, 0.8), sigma_y=(0.8, 0.8), sigma_z=(0.8, 0.8))
        tr = TR_Smooth(prob=1.0, sigma_x=(0.8, 0.8), sigma_y=(0.8, 0.8), sigma_z=(0.8, 0.8))
        torch.manual_seed(0)
        params = pt.sample_params(4, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_different_sigmas(self, device):
        x = torch.rand(4, 3, 16, 16, 16, device=device)
        pt = PT_Smooth(prob=1.0, sigma_x=(0.5, 1.5), sigma_y=(0.3, 1.0), sigma_z=(0.7, 1.2))
        tr = TR_Smooth(prob=1.0, sigma_x=(0.5, 1.5), sigma_y=(0.3, 1.0), sigma_z=(0.7, 1.2))
        torch.manual_seed(42)
        params = pt.sample_params(4, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_mask(self, device):
        x = torch.rand(4, 3, 16, 16, 16, device=device)
        tr = TR_Smooth(prob=1.0, sigma_x=(0.8, 0.8), sigma_y=(0.8, 0.8), sigma_z=(0.8, 0.8))
        torch.manual_seed(0)
        params = tr.sample_params(4, x.shape, device)
        params["mask"] = torch.tensor([True, False, True, False], device=device)
        result = tr.apply(x, params)
        assert torch.equal(result[1], x[1])
        assert torch.equal(result[3], x[3])

    def test_single_channel(self, device):
        x = torch.rand(2, 1, 16, 16, 16, device=device)
        pt = PT_Smooth(prob=1.0, sigma_x=(0.8, 0.8), sigma_y=(0.8, 0.8), sigma_z=(0.8, 0.8))
        tr = TR_Smooth(prob=1.0, sigma_x=(0.8, 0.8), sigma_y=(0.8, 0.8), sigma_z=(0.8, 0.8))
        torch.manual_seed(0)
        params = pt.sample_params(2, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_nonsquare(self, device):
        x = torch.rand(2, 1, 12, 16, 20, device=device)
        pt = PT_Smooth(prob=1.0, sigma_x=(0.8, 0.8), sigma_y=(0.8, 0.8), sigma_z=(0.8, 0.8))
        tr = TR_Smooth(prob=1.0, sigma_x=(0.8, 0.8), sigma_y=(0.8, 0.8), sigma_z=(0.8, 0.8))
        torch.manual_seed(0)
        params = pt.sample_params(2, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)

    def test_bfloat16(self, device):
        x = torch.rand(2, 3, 16, 16, 16, device=device, dtype=torch.bfloat16)
        pt = PT_Smooth(prob=1.0, sigma_x=(0.8, 0.8), sigma_y=(0.8, 0.8), sigma_z=(0.8, 0.8))
        tr = TR_Smooth(prob=1.0, sigma_x=(0.8, 0.8), sigma_y=(0.8, 0.8), sigma_z=(0.8, 0.8))
        torch.manual_seed(0)
        params = pt.sample_params(2, x.shape, device)
        pt_out = pt.apply(x, params)
        tr_out = tr.apply(x, params)
        assert tr_out.dtype == pt_out.dtype
        assert torch.allclose(pt_out.float(), tr_out.float(), atol=1e-2)

    def test_large_sigma(self, device):
        """Large sigma produces wider kernels."""
        x = torch.rand(2, 1, 16, 16, 16, device=device)
        pt = PT_Smooth(prob=1.0, sigma_x=(1.5, 1.5), sigma_y=(1.5, 1.5), sigma_z=(1.5, 1.5))
        tr = TR_Smooth(prob=1.0, sigma_x=(1.5, 1.5), sigma_y=(1.5, 1.5), sigma_z=(1.5, 1.5))
        torch.manual_seed(0)
        params = pt.sample_params(2, x.shape, device)
        assert torch.allclose(pt.apply(x, params), tr.apply(x, params), atol=1e-5)
