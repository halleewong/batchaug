"""Triton-accelerated RandBiasField."""
from __future__ import annotations

import torch
import triton

from ...base import BatchDictTransform, BatchTransform
from ...pytorch.intensity.bias_field import RandBiasField as _PTRandBiasField
from ..kernels.bias_field import _bias_field_kernel_deg3

_MAX_GRID_DIM = 65535


class RandBiasField(_PTRandBiasField):
    """Triton-accelerated RandBiasField (same API as PyTorch version).

    Computes Legendre polynomial basis on-the-fly in the kernel,
    avoiding the large (n_coeff, H, W, D) basis tensor allocation.
    Only supports degree=3; falls back to PyTorch for other degrees.
    """

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        if self.degree != 3:
            return super().sample_params(batch_size, shape, device)

        # Get mask from BatchTransform (skip PyTorch's basis precomputation)
        params = BatchTransform.sample_params(self, batch_size, shape, device)

        # Sample coefficients: (B, 20) for degree=3
        n_coeff = 20
        low, high = self.coeff_range
        params["coeffs"] = (
            torch.rand(batch_size, n_coeff, device=device) * (high - low) + low
        )
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        if self.degree != 3:
            return super().apply(tensor, params)

        mask = params["mask"]
        coeffs = params["coeffs"]  # (B, 20)

        B, C, H, W, D = tensor.shape
        output = torch.empty_like(tensor)

        HWD = H * W * D
        BLOCK_SIZE = 1024
        while triton.cdiv(HWD, BLOCK_SIZE) > _MAX_GRID_DIM:
            BLOCK_SIZE *= 2
        grid = (B, triton.cdiv(HWD, BLOCK_SIZE))

        _bias_field_kernel_deg3[grid](
            tensor.contiguous(),
            output,
            coeffs.contiguous(),
            mask,
            B, C, H, W, D,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output


class RandBiasFieldd(BatchDictTransform):
    """Dictionary wrapper for Triton RandBiasField."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        degree: int = 3,
        coeff_range: tuple[float, float] = (0.0, 0.1),
    ):
        transform = RandBiasField(prob=prob, degree=degree, coeff_range=coeff_range)
        super().__init__(keys=keys, transform=transform)
