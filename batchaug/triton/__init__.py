# Triton backend for batchaug.
# Transforms with Triton kernels override apply(); others re-export from pytorch.
from ..pytorch import *  # noqa: F401, F403
from ..pytorch import __all__

# Override with Triton-accelerated versions (only those faster than cuDNN/PyTorch)
from .intensity.contrast import (  # noqa: F811
    RandAdjustContrast,
    RandAdjustContrastd,
    ScaleIntensity,
    ScaleIntensityd,
)
from .intensity.bias_field import RandBiasField, RandBiasFieldd  # noqa: F811

# NOTE: Triton smooth/sharpen kernels exist but are NOT exported by default
# because cuDNN's conv3d is faster. They are available for explicit use via:
#   from batchaug.triton.intensity.smooth import RandGaussianSmooth
#   from batchaug.triton.intensity.sharpen import RandGaussianSharpen
