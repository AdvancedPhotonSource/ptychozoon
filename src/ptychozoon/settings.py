# Copyright © 2026 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com/AdvancedPhotonSource/ptychozoon/blob/main/LICENSE.TXT
"""Configuration dataclasses for the VSPI fluorescence enhancement algorithm."""

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Optional


class SaveFileExtensions(StrEnum):
    """Supported output file format extensions for saving VSPI results.

    Attributes
    ----------
    H5 : str
        HDF5 file format (``.h5``).
    TIFF : str
        TIFF image stack format (``.tiff``).
    """

    H5 = ".h5"
    TIFF = ".tiff"


class InterpolationTypes(StrEnum):
    """Interpolation strategies for mapping probe positions to object pixels.

    Attributes
    ----------
    FOURIER : str
        Fourier-shift interpolation.  Requires a GPU (CuPy) array module.
        Produces the most accurate sub-pixel results.
    BARYCENTRIC : str
        Bilinear (barycentric) interpolation.  Works on both CPU and GPU
        but is considerably slower than the Fourier method.
    """

    FOURIER = auto()  # only works on GPU
    BARYCENTRIC = auto()  # works on CPU and GPU, but slow


class SolverTypes(StrEnum):
    """Iterative linear solver to use for the VSPI deconvolution.

    Attributes
    ----------
    LSMR : str
        Least-Squares Minimum Residual solver
        (:func:`scipy.sparse.linalg.lsmr`).  Supports optional Tikhonov
        regularisation via a damping factor.
    """

    LSMR = auto()


@dataclass
class LSMRSettings:
    """Hyper-parameters for the LSMR iterative solver.

    Attributes
    ----------
    damping_factor : float
        Tikhonov regularisation parameter.  A value of ``0.0`` (default)
        means no regularisation is applied.
    max_iter : int
        Maximum number of LSMR iterations to run in total.
    atol : float
        Tolerance on the norm of the relative residual ``‖b − A x‖ / ‖b‖``.
        Iteration stops when the residual drops below this value.
    btol : float
        Tolerance on the norm of ``A^T (b − A x)``.  Controls convergence
        of the adjoint residual.
    checkpoint_interval : int or None
        If set, the algorithm yields an intermediate result every this many
        LSMR iterations (e.g. ``5`` → yields at iterations 5, 10, 15, …).
        If ``None``, only the final result is yielded.
    """

    damping_factor: float = 0.0
    "Damping factor for regularized least-squares"

    max_iter: int = 10

    atol: float = 1e-6

    btol: float = 1e-6

    checkpoint_interval: Optional[int] = None
    "Yield the solution every this many iterations. If None, only yield the final result."


@dataclass
class GPUSettings:
    """Settings for GPU (CuPy) accelerated computation.

    Attributes
    ----------
    enabled : bool
        Whether to use GPU acceleration.  Requires CuPy and a CUDA-capable
        device.  When ``False``, NumPy / SciPy are used instead.
    index : int
        Zero-based CUDA device index to use when ``enabled`` is ``True``.
    """

    enabled: bool = True

    index: int = 0


@dataclass
class DeconvolutionEnhancementSettings:
    """Top-level configuration for the VSPI fluorescence enhancement algorithm.

    Attributes
    ----------
    solver : SolverTypes
        Iterative linear solver to use for the deconvolution.
    lsmr : LSMRSettings
        Hyper-parameters for the LSMR solver.
    interpolation : InterpolationTypes
        Interpolation strategy for mapping probe positions to object pixels.
    gpu : GPUSettings
        GPU acceleration settings.
    """

    solver: SolverTypes = SolverTypes.LSMR

    lsmr: LSMRSettings = field(default_factory=LSMRSettings)

    gpu: GPUSettings = field(default_factory=GPUSettings)

    _interpolation: InterpolationTypes = InterpolationTypes.FOURIER
