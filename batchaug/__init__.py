from .base import BatchDictTransform, BatchTransform
from .compose import Compose
from ._backend import get_backend, resolve_backend, set_backend

# Transform names that are dispatched to the active backend
_TRANSFORM_NAMES = [
    "DivisiblePad",
    "DivisiblePadd",
    "Rand3DElastic",
    "Rand3DElasticd",
    "RandAffine",
    "RandAffined",
    "RandAxisFlip",
    "RandAxisFlipd",
    "RandBiasField",
    "RandBiasFieldd",
    "RandGibbsNoise",
    "RandGibbsNoised",
    "ScaleIntensity",
    "ScaleIntensityd",
    "RandAdjustContrast",
    "RandAdjustContrastd",
    "RandRotate90",
    "RandRotate90d",
    "RandGaussianNoise",
    "RandGaussianNoised",
    "RandGaussianSmooth",
    "RandGaussianSmoothd",
    "RandGaussianSharpen",
    "RandGaussianSharpend",
    "RandSimulateLowResolution",
    "RandSimulateLowResolutiond",
]

# Names that only exist in the Triton backend (no PyTorch equivalent)
_TRITON_ONLY_NAMES = [
    "FusedAugment",
    "FusedAugmentd",
]

__all__ = [
    "BatchTransform",
    "BatchDictTransform",
    "Compose",
    "set_backend",
    "get_backend",
    "resolve_backend",
    *_TRANSFORM_NAMES,
    *_TRITON_ONLY_NAMES,
]


def __getattr__(name: str):
    if name in _TRANSFORM_NAMES:
        backend = resolve_backend()
        if backend == "triton":
            from . import triton as _mod
        else:
            from . import pytorch as _mod
        val = getattr(_mod, name)
        globals()[name] = val  # cache for subsequent access
        return val
    if name in _TRITON_ONLY_NAMES:
        from . import triton as _mod
        val = getattr(_mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'batchaug' has no attribute {name!r}")
