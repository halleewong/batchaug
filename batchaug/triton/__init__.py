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
from .intensity.scale_shift import (  # noqa: F811
    RandScaleIntensity,
    RandScaleIntensityd,
    RandShiftIntensity,
    RandShiftIntensityd,
    RandStdShiftIntensity,
    RandStdShiftIntensityd,
    RandScaleIntensityFixedMean,
    RandScaleIntensityFixedMeand,
)
from .intensity.rician_noise import RandRicianNoise, RandRicianNoised  # noqa: F811

# NOTE: Triton smooth/sharpen kernels exist but are NOT exported by default
# because cuDNN's conv3d is faster. They are available for explicit use via:
#   from batchaug.triton.intensity.smooth import RandGaussianSmooth
#   from batchaug.triton.intensity.sharpen import RandGaussianSharpen
#
# NOTE: RandFlip, RandRotate, RandZoom use grid_sample / torch.flip and do
# not benefit from Triton kernels — re-exported from pytorch above.
