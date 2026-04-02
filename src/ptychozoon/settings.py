from dataclasses import dataclass
from enum import StrEnum, auto


class InterpolationTypes(StrEnum):
    FOURIER = auto()
    BARYCENTRIC = auto()


@dataclass
class DeconvolutionEnhancementSettings:

    use_gpu: bool = True

    interpolation: InterpolationTypes = InterpolationTypes.FOURIER