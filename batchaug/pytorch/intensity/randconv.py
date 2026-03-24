from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import BatchDictTransform, BatchTransform


class RandConv(BatchTransform):
    """Batched random convolution augmentation.

    Applies a freshly randomized convolution to each batch element independently.
    Each channel is convolved with the same kernel (no cross-channel mixing) —
    equivalent to a depthwise convolution with shared weights across channels.
    Based on Xu et al. "Robust and Generalizable Visual Representation Learning
    via Random Convolutions" (ICLR 2021).

    A single kernel size is sampled per call (shared across the batch). Each
    batch element gets its own independent random weight tensor of shape
    (1, 1, ks, ks, ks), which is repeated across channels. The grouped
    convolution trick is used to process the whole batch in one CUDA call:
    input is reshaped to (1, B*C, H, W, D) and weights expanded to
    (B*C, 1, ks, ks, ks), then F.conv3d is called with groups=B*C.

    Output always has the same number of channels as the input.

    Args:
        prob:
            Probability of applying the transform to each batch element.
        kernel_sizes:
            One or more kernel sizes to sample from uniformly each call.
        mixing:
            If True (RC_mix mode), blends input and convolved output:
            ``alpha * randconv(x) + (1 - alpha) * x``, where alpha ~ U(0, 1)
            per batch element.
        distribution:
            Weight initialisation: "kaiming_normal" (default),
            "kaiming_uniform", or "xavier_normal".
        rand_bias:
            If True, also randomise the conv bias (one bias per batch element,
            shared across channels).
        padding_mode:
            How to pad the input before convolution. One of ``"zeros"``
            (default), ``"reflect"``, ``"replicate"``, or ``"circular"``.
            Use ``"reflect"`` or ``"replicate"`` to avoid edge artifacts.

    Input shape: ``(B, C, H, W, D)``.
    """

    def __init__(
        self,
        prob: float = 0.5,
        kernel_sizes: int | Sequence[int] = 3,
        mixing: bool = False,
        distribution: str = "kaiming_normal",
        rand_bias: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__(prob=prob)
        if isinstance(kernel_sizes, int):
            self.kernel_sizes = [kernel_sizes]
        else:
            self.kernel_sizes = list(kernel_sizes)
        self.mixing = mixing
        self.distribution = distribution
        self.rand_bias = rand_bias
        self.padding_mode = padding_mode

        if distribution not in ("kaiming_normal", "kaiming_uniform", "xavier_normal"):
            raise ValueError(
                f"Unknown distribution '{distribution}'. "
                "Choose from: kaiming_normal, kaiming_uniform, xavier_normal."
            )
        if padding_mode not in ("zeros", "reflect", "replicate", "circular"):
            raise ValueError(
                f"Unknown padding_mode '{padding_mode}'. "
                "Choose from: zeros, reflect, replicate, circular."
            )

    def _init_weights(self, w: torch.Tensor) -> None:
        with torch.no_grad():
            if self.distribution == "kaiming_normal":
                nn.init.kaiming_normal_(w, nonlinearity="conv2d")
            elif self.distribution == "kaiming_uniform":
                nn.init.kaiming_uniform_(w, nonlinearity="conv2d")
            elif self.distribution == "xavier_normal":
                nn.init.xavier_normal_(w)

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        params = super().sample_params(batch_size, shape, device)

        # Sample one kernel size for the whole batch
        idx = torch.randint(len(self.kernel_sizes), (1,)).item()
        ks = self.kernel_sizes[int(idx)]
        params["kernel_size"] = ks

        # One kernel per batch element; same weights applied to every channel
        # (depthwise-style, no cross-channel mixing): shape (B, 1, ks, ks, ks)
        w = torch.empty(batch_size, 1, ks, ks, ks, device=device)
        self._init_weights(w)
        params["weights"] = w

        if self.rand_bias:
            fan_in = ks * ks * ks
            bound = 1.0 / math.sqrt(fan_in)
            # One bias per batch element, shared across channels
            b = torch.empty(batch_size, device=device).uniform_(-bound, bound)
            params["bias"] = b
        else:
            params["bias"] = None

        if self.mixing:
            params["alpha"] = torch.rand(batch_size, device=device)

        return params

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        mask = params["mask"]  # (B,)
        ks = params["kernel_size"]
        weights = params["weights"]  # (B, 1, ks, ks, ks)
        bias = params["bias"]  # (B,) or None

        B, C, H, W, D = tensor.shape
        work = tensor.float()

        # Depthwise grouped conv: (B, C, H, W, D) → (1, B*C, H, W, D)
        x = work.reshape(1, B * C, H, W, D)
        pad = ks // 2
        # Expand each batch element's kernel across its C channels: (B*C, 1, ks, ks, ks)
        w_expanded = weights.repeat_interleave(C, dim=0).float()
        b_expanded = bias.repeat_interleave(C) if bias is not None else None
        if self.padding_mode == "zeros":
            out = F.conv3d(x, w_expanded, b_expanded, padding=pad, groups=B * C)
        else:
            # F.conv3d has no padding_mode arg — pre-pad manually then convolve with padding=0
            x = F.pad(x, (pad, pad, pad, pad, pad, pad), mode=self.padding_mode)
            out = F.conv3d(x, w_expanded, b_expanded, padding=0, groups=B * C)
        # out: (1, B*C, H, W, D) → (B, C, H, W, D)
        out = out.reshape(B, C, H, W, D)


        if self.mixing:
            alpha = params["alpha"].float()  # (B,)
            a = alpha[:, None, None, None, None]
            out = a * out + (1.0 - a) * work

        out = out.to(tensor.dtype)

        mask_5d = mask[:, None, None, None, None]
        return torch.where(mask_5d, out, tensor)


class RandConvd(BatchDictTransform):
    """Dictionary wrapper for RandConv.

    Params are sampled once from the first key's tensor, then the same random
    convolution (per batch element) is applied to all specified keys.

    See RandConv for full parameter documentation.
    """

    def __init__(
        self,
        keys: list[str],
        prob: float = 0.5,
        kernel_sizes: int | Sequence[int] = 3,
        mixing: bool = False,
        distribution: str = "kaiming_normal",
        rand_bias: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        transform = RandConv(
            prob=prob,
            kernel_sizes=kernel_sizes,
            mixing=mixing,
            distribution=distribution,
            rand_bias=rand_bias,
            padding_mode=padding_mode,
        )
        super().__init__(keys=keys, transform=transform)
