from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Optional


class InterpolationTypes(StrEnum):
    FOURIER = auto()  # only works on GPU
    BARYCENTRIC = auto()  # works on CPU and GPU, but slow


class SolverTypes(StrEnum):
    LSMR = auto()

@dataclass
class LSMRSettings:
    damping_factor: float = 0.0
    "Damping factor for regularized least-squares"

    max_iter: int = 10

    atol: float = 1e-6

    btol: float = 1e-6

    checkpoint_interval: Optional[int] = None
    "Yield the solution every this many iterations. If None, only yield the final result."


@dataclass
class GPUSettings:
    enabled: bool = True

    index: int = 0


@dataclass
class DeconvolutionEnhancementSettings:
    solver: SolverTypes = SolverTypes.LSMR

    lsmr: LSMRSettings = field(default_factory=LSMRSettings)

    interpolation: InterpolationTypes = InterpolationTypes.FOURIER

    gpu: GPUSettings = field(default_factory=GPUSettings)
