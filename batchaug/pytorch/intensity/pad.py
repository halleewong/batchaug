from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class DivisiblePad:
    """Pad spatial dimensions to be divisible by k.

    Always applies (no probability). Output shape may differ from input.

    Input shape: (B, C, H, W, D).
    """

    def __init__(
        self,
        k: int | tuple[int, int, int],
        method: str = "symmetric",
        mode: str = "constant",
        **kwargs,
    ):
        if isinstance(k, int):
            self.k = (k, k, k)
        else:
            self.k = tuple(k)
        self.method = method
        self.mode = mode

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        H, W, D = tensor.shape[2:]

        # F.pad takes padding in reverse spatial order: (D_lo, D_hi, W_lo, W_hi, H_lo, H_hi)
        pad = []
        for dim_size, ki in reversed(list(zip([H, W, D], self.k))):
            if ki <= 0:
                pad.extend([0, 0])
                continue
            new_size = math.ceil(dim_size / ki) * ki
            total = new_size - dim_size
            if self.method == "symmetric":
                lo = total // 2
                hi = total - lo
            else:  # "end"
                lo = 0
                hi = total
            pad.extend([lo, hi])

        if all(p == 0 for p in pad):
            return tensor

        return F.pad(tensor, pad, mode=self.mode)


class DivisiblePadd:
    """Dictionary wrapper for DivisiblePad.

    Applies identical padding to all specified keys.
    """

    def __init__(
        self,
        keys: list[str],
        k: int | tuple[int, int, int],
        method: str = "symmetric",
        mode: str = "constant",
        **kwargs,
    ):
        self.keys = keys
        self.transform = DivisiblePad(k=k, method=method, mode=mode)

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = self.transform(d[key])
        return d
