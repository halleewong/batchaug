from __future__ import annotations

import torch


class BatchTransform:
    """Base class for tensor-level batched transforms.

    All transforms expect input shape (B, C, H, W, D).
    Random parameters are sampled independently per batch element.
    The same parameters are applied to all channels within a batch element.

    Subclasses implement:
        - sample_params(batch_size, shape, device) -> dict with at least 'mask'
        - apply(tensor, params) -> transformed tensor
    """

    def __init__(self, prob: float = 1.0):
        self.prob = prob

    def sample_params(
        self, batch_size: int, shape: tuple[int, ...], device: torch.device
    ) -> dict:
        """Sample random parameters.

        Args:
            batch_size: Number of elements in the batch.
            shape: Full tensor shape (B, C, H, W, D).
            device: Device to create tensors on.

        Returns:
            Dict containing at least 'mask': (B,) bool tensor indicating
            which batch elements should be augmented.
        """
        mask = torch.rand(batch_size, device=device) < self.prob
        return {"mask": mask}

    def apply(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply the transform using pre-sampled params.

        Args:
            tensor: Input of shape (B, C, H, W, D).
            params: Dict from sample_params().

        Returns:
            Transformed tensor of same shape and dtype.
        """
        raise NotImplementedError

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        params = self.sample_params(tensor.shape[0], tensor.shape, tensor.device)
        return self.apply(tensor, params)


class BatchDictTransform:
    """Dict-level wrapper that routes keys to an underlying BatchTransform.

    Samples parameters ONCE from the first key's tensor, then applies
    to all specified keys with those same parameters. This ensures
    paired data (e.g. vol/seg) receives identical spatial transforms.
    """

    def __init__(self, keys: list[str], transform: BatchTransform):
        self.keys = keys
        self.transform = transform

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        d = dict(data)
        first_tensor = d[self.keys[0]]
        params = self.transform.sample_params(
            first_tensor.shape[0], first_tensor.shape, first_tensor.device
        )
        for key in self.keys:
            if key in d:
                d[key] = self.transform.apply(d[key], params)
        return d
