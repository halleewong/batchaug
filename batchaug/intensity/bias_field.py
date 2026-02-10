from __future__ import annotations

import torch

from ..base import BatchDictTransform, BatchTransform


def _legendre_basis_3d(
    degree: int, shape: tuple[int, int, int], device: torch.device
) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
    """Precompute Legendre polynomial basis functions on a [-1,1] grid.

    Args:
        degree: Maximum polynomial degree (>= 1).
        shape: Spatial shape (H, W, D).
        device: Torch device.

    Returns:
        basis: (n_coeff, H, W, D) tensor of basis functions.
        triplets: List of (i, j, k) index triplets with i+j+k <= degree.
    """
    # Enumerate valid (i, j, k) with i+j+k <= degree, including (0,0,0).
    # MONAI includes (0,0,0) in its coefficient set (20 coefficients for degree=3).
    triplets: list[tuple[int, int, int]] = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            for k in range(degree + 1 - i - j):
                triplets.append((i, j, k))

    # Compute Legendre polynomials along each axis
    coords = [
        torch.linspace(-1.0, 1.0, s, dtype=torch.float32, device=device)
        for s in shape
    ]
    # Evaluate L_0..L_degree for each axis: (degree+1, S_i)
    polys = []
    for c in coords:
        p = [torch.ones_like(c), c]
        for n in range(2, degree + 1):
            # Recurrence: (n+1) L_{n+1}(x) = (2n+1) x L_n(x) - n L_{n-1}(x)
            p_next = ((2 * n - 1) * c * p[-1] - (n - 1) * p[-2]) / n
            p.append(p_next)
        polys.append(torch.stack(p[: degree + 1], dim=0))  # (degree+1, S_i)

    # Build outer products for each triplet: (n_coeff, H, W, D)
    basis_list = []
    for i, j, k in triplets:
        # L_i(x) * L_j(y) * L_k(z) via broadcasting
        b = polys[0][i, :, None, None] * polys[1][j, None, :, None] * polys[2][k, None, None, :]
        basis_list.append(b)

    basis = torch.stack(basis_list, dim=0)  # (n_coeff, H, W, D)
    return basis, triplets


class RandBiasField(BatchTransform):
    """Random polynomial bias field augmentation for MRI volumes.

    Generates a smooth multiplicative bias field as a linear combination
    of Legendre polynomial basis functions, then applies it as
    ``out = img * exp(bias_field)``.

    Coefficients are sampled independently per batch element.
    The same bias field is applied to all channels within a batch element.

    Input shape: (B, C, H, W, D).
    """

    def __init__(
        self,
        prob: float = 0.1,
        degree: int = 3,
        coeff_range: tuple[float, float] = (0.0, 0.1),
    ):
        super().__init__(prob=prob)
        if degree < 1:
            raise ValueError(f"degree should be no less than 1, got {degree}.")
        self.degree = degree
        self.coeff_range = coeff_range

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)
        spatial_shape = shape[2:]  # (H, W, D)

        # Precompute basis: (n_coeff, H, W, D)
        basis, triplets = _legendre_basis_3d(self.degree, spatial_shape, device)
        params["basis"] = basis

        # Sample coefficients per batch element: (B, n_coeff)
        n_coeff = len(triplets)
        low, high = self.coeff_range
        params["coeffs"] = (
            torch.rand(batch_size, n_coeff, device=device) * (high - low) + low
        )
        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]
        basis = params["basis"]  # (n_coeff, H, W, D)
        coeffs = params["coeffs"]  # (B, n_coeff)

        # bias_field: (B, H, W, D)
        bias_field = torch.einsum("bn,nhwd->bhwd", coeffs, basis)

        # Apply: out = img * exp(bias_field), broadcast over C
        multiplier = torch.exp(bias_field).unsqueeze(1)  # (B, 1, H, W, D)
        result = (tensor.float() * multiplier).to(tensor.dtype)

        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, result, tensor)


class RandBiasFieldd(BatchDictTransform):
    """Dictionary wrapper for RandBiasField.

    All keys receive the same bias field (same coefficients, same mask).
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.1,
        degree: int = 3,
        coeff_range: tuple[float, float] = (0.0, 0.1),
    ):
        transform = RandBiasField(prob=prob, degree=degree, coeff_range=coeff_range)
        super().__init__(keys=keys, transform=transform)
