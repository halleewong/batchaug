import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cuda")


@pytest.fixture
def vol(device):
    """(B=4, C=3, H=16, W=16, D=16) float32 volume on CUDA."""
    torch.manual_seed(42)
    return torch.rand(4, 3, 16, 16, 16, device=device)


@pytest.fixture
def vol_bf16(device):
    """Same shape but bfloat16."""
    torch.manual_seed(42)
    return torch.rand(4, 3, 16, 16, 16, device=device, dtype=torch.bfloat16)


@pytest.fixture
def seg(device):
    """Integer segmentation labels cast to float."""
    torch.manual_seed(42)
    return torch.randint(0, 5, (4, 3, 16, 16, 16), device=device).float()
