"""Triton-accelerated RandRicianNoise."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...base import BatchDictTransform
from ...pytorch.intensity.noise import RandRicianNoise as _PTRandRicianNoise

_MAX_GRID_DIM = 65535


def _safe_block_size(n_elements: int, min_block: int = 1024) -> int:
    bs = min_block
    while triton.cdiv(n_elements, bs) > _MAX_GRID_DIM:
        bs *= 2
    return bs


@triton.jit
def _rician_noise_kernel(
    input_ptr,
    output_ptr,
    noise_std_ptr,     # (B,) per-element noise std
    mask_ptr,          # (B,) bool
    mean: tl.constexpr,
    N_per_batch,       # C*H*W*D
    seed,              # base RNG seed
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Rician noise: output = sqrt((input + n1)^2 + n2^2).

    ``n1, n2 ~ N(mean, noise_std^2)`` generated in-kernel to avoid
    materialising large noise tensors.

    Grid: (B, cdiv(N_per_batch, BLOCK_SIZE)).
    """
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    m = tl.load(mask_ptr + batch_id)
    base = batch_id * N_per_batch
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N_per_batch

    vals = tl.load(input_ptr + base + offsets, mask=valid, other=0.0).to(tl.float32)

    if m == 0:
        tl.store(output_ptr + base + offsets, vals, mask=valid)
        return

    std = tl.load(noise_std_ptr + batch_id)

    # Two independent noise channels via different seed offsets.
    # Use global element index as the per-element offset for tl.randn.
    elem_offsets = base + offsets

    n1 = mean + std * tl.randn(seed, elem_offsets)
    n2 = mean + std * tl.randn(seed + 1, elem_offsets)

    result = tl.sqrt((vals + n1) * (vals + n1) + n2 * n2)
    tl.store(output_ptr + base + offsets, result, mask=valid)


class RandRicianNoise(_PTRandRicianNoise):
    """Triton-accelerated RandRicianNoise (same API as PyTorch version).

    Generates both noise channels inside the kernel using ``tl.randn``,
    avoiding materialisation of the two full-shape noise tensors.

    A ``triton_seed`` is stored in ``sample_params`` so that all keys in a
    dict transform receive the same noise draw.

    Falls back to the PyTorch version when ``relative=True`` (requires
    per-element signal std, which needs an extra reduction pass).
    """

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        # Fixed seed so dict keys get the same noise pattern.
        params["triton_seed"] = int(torch.randint(0, 2**31, ()).item())
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        if self.relative:
            # relative mode needs signal std reduction; use PyTorch path
            return super().apply(tensor, params)

        mask = params["mask"]
        noise_std = params["noise_std"]
        seed = params.get("triton_seed", int(torch.randint(0, 2**31, ()).item()))

        B = tensor.shape[0]
        N_per_batch = tensor[0].numel()

        output = torch.empty_like(tensor)
        BLOCK_SIZE = _safe_block_size(N_per_batch)
        grid = (B, triton.cdiv(N_per_batch, BLOCK_SIZE))
        _rician_noise_kernel[grid](
            tensor.contiguous().float(), output,
            noise_std.float(), mask,
            self.mean,
            N_per_batch,
            seed,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output.to(tensor.dtype)


class RandRicianNoised(BatchDictTransform):
    """Dictionary wrapper for Triton RandRicianNoise."""

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        mean: float = 0.0,
        std: float = 0.1,
        relative: bool = False,
        sample_std: bool = True,
    ):
        transform = RandRicianNoise(
            prob=prob, mean=mean, std=std,
            relative=relative, sample_std=sample_std,
        )
        super().__init__(keys=keys, transform=transform)
