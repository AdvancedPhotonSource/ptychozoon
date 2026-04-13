"""VSPI Fluorescence Enhancement Algorithm

Re-implementation of the Virtual Single Pixel Imaging algorithm for enhancing
fluorescence data using ptychography reconstructions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generator, Optional, Sequence
import logging
import time
import tqdm

import numpy as np
import cupy as cp

from chronos.timer_utils import timer, InlineTimer

from ptychozoon.settings import (
    DeconvolutionEnhancementSettings,
    InterpolationTypes,
    SolverTypes,
)
from .patches import extract_patches_fourier_shift, place_patches_fourier_shift

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElementMap:
    """2D spatial map of fluorescence signal for a single element in counts per second."""

    name: str
    counts_per_second: np.ndarray


@dataclass(frozen=True)
class FluorescenceDataset:
    """Collection of element maps with metadata."""

    element_maps: Sequence[ElementMap]


@dataclass(frozen=True)
class Product:
    """Ptychography reconstruction product.

    All arrays are stored as numpy arrays:
    - probe_positions: (N, 2) array of [y, x] coordinates in meters
    - probe: (n_opr, modes, height, width) complex array
    - object_array: (height, width) complex array
    - pixel_size_m: (y, x) pixel sizes in meters
    - object_center_m: (y, x) center coordinates in meters
    """

    probe_positions: np.ndarray  # (N, 2) float array [y, x] in meters
    probe: np.ndarray  # (n_opr, modes, height, width) complex array
    object_array: np.ndarray  # (height, width) complex array
    pixel_size_m: tuple[float, float]  # (pixel_height_m, pixel_width_m)
    object_center_m: tuple[float, float]  # (center_y_m, center_x_m)
    opr_mode_weights: Optional[np.ndarray] = None  # n_opr, N


class ArrayPatchInterpolator:
    """Bilinear interpolation for extracting and accumulating array patches."""

    def __init__(
        self,
        array: np.ndarray,
        center_y_px: float,
        center_x_px: float,
        shape: tuple[int, int],
    ) -> None:
        """Initialize interpolator for a patch centered at (center_y_px, center_x_px).

        Args:
            array: Full 2D array to extract patches from
            center_y_px: Y-coordinate of patch center in pixels
            center_x_px: X-coordinate of patch center in pixels
            shape: (height, width) of the patch to extract
        """
        # Top left corner of patch support
        xmin = center_x_px - shape[-1] / 2
        ymin = center_y_px - shape[-2] / 2

        # Whole components (pixel indexes)
        xmin_wh = int(xmin)
        ymin_wh = int(ymin)

        # Fractional (subpixel) components
        xmin_fr = xmin - xmin_wh
        ymin_fr = ymin - ymin_wh

        # Bottom right corner of patch support
        xmax_wh = xmin_wh + shape[-1] + 1
        ymax_wh = ymin_wh + shape[-2] + 1

        # Reused quantities
        xmin_fr_c = 1.0 - xmin_fr
        ymin_fr_c = 1.0 - ymin_fr

        # Barycentric interpolant weights
        self._weight00 = ymin_fr_c * xmin_fr_c
        self._weight01 = ymin_fr_c * xmin_fr
        self._weight10 = ymin_fr * xmin_fr_c
        self._weight11 = ymin_fr * xmin_fr

        # Extract patch support region from full object
        self._support = array[ymin_wh:ymax_wh, xmin_wh:xmax_wh]

    # @timer()
    def get_patch(self) -> np.ndarray:
        """Interpolate array support to extract patch."""
        patch = self._weight00 * self._support[:-1, :-1]
        patch += self._weight01 * self._support[:-1, 1:]
        patch += self._weight10 * self._support[1:, :-1]
        patch += self._weight11 * self._support[1:, 1:]
        return patch

    # @timer()
    def accumulate_patch(self, patch: np.ndarray) -> None:
        """Add patch update to array support."""
        self._support[:-1, :-1] += self._weight00 * patch
        self._support[:-1, 1:] += self._weight01 * patch
        self._support[1:, :-1] += self._weight10 * patch
        self._support[1:, 1:] += self._weight11 * patch


def _make_vspi_linear_operator(
    product: Product, xp, LinearOperator, settings: DeconvolutionEnhancementSettings
):
    """Factory that creates a VSPILinearOperator bound to the given array module and base class.

    Args:
        product: Ptychography reconstruction product (probe/object_array may be cupy or numpy)
        xp: Array module to use (numpy or cupy)
        LinearOperator: LinearOperator base class (scipy or cupyx)

    Returns:
        VSPILinearOperator instance
    """

    class VSPILinearOperator(LinearOperator):
        # """Linear operator A for VSPI: A[M,N] * X[N,P] = B[M,P]

        # Where:
        #     M: number of XRF positions (scan points)
        #     N: number of ptychography object pixels
        #     P: number of XRF channels
        # """
        """Linear operator A for VSPI: A[M,N] * X[N] = B[M]

        Where:
            M: number of XRF positions (scan points)
            N: number of ptychography object pixels
        """

        @timer()
        def __init__(self, interpolation_type: InterpolationTypes) -> None:
            M = len(product.probe_positions)  # Number of scan points
            N = (
                product.object_array.shape[0] * product.object_array.shape[1]
            )  # Total pixels
            super().__init__(float, (M, N))

            self.interpolation_type = interpolation_type

            # probe_positions stays as numpy for efficient Python-level iteration
            self._probe_positions = product.probe_positions
            self._probe = product.probe
            self._object_height_px = product.object_array.shape[0]
            self._object_width_px = product.object_array.shape[1]
            self._pixel_height_m = product.pixel_size_m[0]
            self._pixel_width_m = product.pixel_size_m[1]
            self._center_y_m = product.object_center_m[0]
            self._center_x_m = product.object_center_m[1]
            self._object_array = product.object_array * 0
            self._opr_mode_weights = product.opr_mode_weights
            self._call_count = 0

        # @timer()
        def _probe_to_object_coords(
            self, probe_y_m: float, probe_x_m: float
        ) -> tuple[float, float]:
            """Convert probe coordinates (meters) to object pixel coordinates.

            Args:
                probe_y_m: Probe Y position in meters
                probe_x_m: Probe X position in meters

            Returns:
                (y_px, x_px) in object pixel coordinates
            """
            ry_px = self._object_height_px / 2
            rx_px = self._object_width_px / 2

            y_px = (probe_y_m - self._center_y_m) / self._pixel_height_m + ry_px
            x_px = (probe_x_m - self._center_x_m) / self._pixel_width_m + rx_px

            return y_px, x_px

        @timer()
        def _matvec(self, v) -> np.ndarray:
            """Forward operator: A * v

            Args:
                v: Flattened object array (N,)

            Returns:
                Result vector (M,)
            """
            # gets the convolved image

            # input v is the upscaled XRF array after flattening
            object_array = v.reshape((self._object_height_px, self._object_width_px))
            result = xp.zeros(len(self._probe_positions))

            # Get probe intensity (sum over modes)
            if self._opr_mode_weights is not None:
                probe_intensity = _get_probe_intensity_at_each_position(
                    self._probe, self._opr_mode_weights
                )
                patch_size = probe_intensity.shape[1:]
                psf = probe_intensity / probe_intensity.sum((1, 2))[:, None, None]
            else:
                probe_intensity = xp.sum(xp.abs(self._probe) ** 2, axis=0)
                patch_size = probe_intensity.shape
                psf = probe_intensity / probe_intensity.sum()

            inline_timer = InlineTimer("Extract patches")
            inline_timer.start()
            if self.interpolation_type == InterpolationTypes.FOURIER:
                # convert probe positions to object coordinates
                positions_px = xp.array(
                    [
                        self._probe_to_object_coords(pos_m[0], pos_m[1])
                        for pos_m in self._probe_positions
                    ]
                )
                positions_px += -xp.array([1, 1]) * 0.5
                extracted_patches = extract_patches_fourier_shift(
                    object_array, positions_px, patch_size
                )
                # The extracted patches do not match the barycentric interpolation that was orignally here unless
                # `positions_px` is replaced `positions_px - xp.array([1, 1]) * 0.5`.
                result = (extracted_patches * psf).sum((1, 2))  # b_fit
                # Is it better to place patches and then multiply with the object array?
            elif self.interpolation_type == InterpolationTypes.BARYCENTRIC:
                result = xp.zeros(len(self._probe_positions))
                for index, position in enumerate(self._probe_positions):
                    # Convert probe position to object coordinates
                    probe_y_m, probe_x_m = float(position[0]), float(position[1])
                    obj_y_px, obj_x_px = self._probe_to_object_coords(
                        probe_y_m, probe_x_m
                    )

                    # Extract and accumulate patch
                    interpolator = ArrayPatchInterpolator(
                        object_array, obj_y_px, obj_x_px, psf.shape
                    )
                    result[index] = xp.sum(psf * interpolator.get_patch())
            inline_timer.end()

            return result

        @timer()
        def _rmatvec(self, u) -> np.ndarray:
            """Adjoint operator: A^T * u

            Args:
                v: Input vector (M,)

            Returns:
                Flattened object array (N,)
            """
            # gets the deconvolved image
            object_array = xp.zeros((self._object_height_px, self._object_width_px))

            # Get probe intensity (sum over modes)
            if self._opr_mode_weights is not None:
                probe_intensity = _get_probe_intensity_at_each_position(
                    self._probe, self._opr_mode_weights
                )
                psf = probe_intensity / probe_intensity.sum((1, 2))[:, None, None]
            else:
                probe_intensity = xp.sum(xp.abs(self._probe) ** 2, axis=0)
                psf = probe_intensity / probe_intensity.sum()

            inline_timer = InlineTimer("Accumulate patches")
            inline_timer.start()
            if self.interpolation_type == InterpolationTypes.FOURIER:
                positions_px = xp.array(
                    [
                        self._probe_to_object_coords(pos_m[0], pos_m[1])
                        for pos_m in self._probe_positions
                    ]
                )
                positions_px += -xp.array([1, 1]) * 0.5
                object_array = place_patches_fourier_shift(
                    object_array,
                    positions_px,
                    u[:, None, None] * psf,
                    "add",
                    adjoint_mode=False,
                )
            elif self.interpolation_type == InterpolationTypes.BARYCENTRIC:
                for index, position in enumerate(self._probe_positions):
                    # Convert probe position to object coordinates
                    probe_y_m, probe_x_m = float(position[0]), float(position[1])
                    obj_y_px, obj_x_px = self._probe_to_object_coords(
                        probe_y_m, probe_x_m
                    )

                    # Accumulate weighted patch
                    interpolator = ArrayPatchInterpolator(
                        object_array, obj_y_px, obj_x_px, psf.shape
                    )
                    interpolator.accumulate_patch(u[index] * psf)
            inline_timer.end()

            # import matplotlib.pyplot as plt; plt.imshow(object_array.get());plt.colorbar();plt.title("deconvolved?");plt.show()
            return object_array.flatten()

    return VSPILinearOperator(interpolation_type=settings.interpolation)


class VSPIFluorescenceEnhancingAlgorithm:
    """Virtual Single Pixel Imaging algorithm for fluorescence enhancement.

    This algorithm uses ptychography reconstruction data to enhance fluorescence
    measurements by solving a linear system that accounts for the finite size
    of the X-ray probe.
    """

    def enhance(
        self,
        dataset: FluorescenceDataset,
        product: Product,
        valid_pixel_index: Optional[list[int]] = None,
        select_maps: Optional[list[str]] = None,
        settings: Optional[DeconvolutionEnhancementSettings] = None,
    ) -> Generator[tuple[FluorescenceDataset, int]]:
        """Enhance fluorescence dataset using ptychography product.

        This is a generator that yields ``(FluorescenceDataset, convolve_fit, iteration)``
        tuples. If ``settings.lsmr.checkpoint_interval`` is set, it yields after every
        N iterations (e.g. 5, 10, 15, …); otherwise it yields once after all iterations.

        Args:
            dataset: Input fluorescence dataset
            product: Ptychography reconstruction product
            valid_pixel_index: Optional indices of valid scan positions
            select_maps: Optional list of element names to enhance (all if None)
            settings: Algorithm settings; uses defaults if None

        Yields:
            (FluorescenceDataset, iteration) at each checkpoint
        """
        if settings is None:
            settings = DeconvolutionEnhancementSettings()
        if settings.gpu.enabled:
            # use specified device
            cp.cuda.Device(settings.gpu.index).use()

            from cupyx.scipy.sparse.linalg import lsmr, LinearOperator

            xp = cp
            # Move probe and object_array to GPU; probe_positions stays on CPU
            # for efficient Python-level iteration over scan positions
            inline_timer = InlineTimer("Move data to GPU")
            inline_timer.start()
            if product.opr_mode_weights is not None:
                opr_mode_weights = cp.asarray(product.opr_mode_weights)
            else:
                opr_mode_weights = None
            gpu_product = Product(
                probe_positions=product.probe_positions,
                probe=cp.asarray(product.probe),
                object_array=product.object_array,
                pixel_size_m=product.pixel_size_m,
                object_center_m=product.object_center_m,
                opr_mode_weights=opr_mode_weights,
            )
            inline_timer.end()
        else:
            from scipy.sparse.linalg import lsmr, LinearOperator

            xp = np
            gpu_product = product

        inline_timer = InlineTimer("Make VSPI linear operator")
        inline_timer.start()
        A = _make_vspi_linear_operator(gpu_product, xp, LinearOperator, settings)
        inline_timer.end()

        if select_maps is not None:
            selected_element_maps = [
                emap for emap in dataset.element_maps if emap.name in select_maps
            ]
        else:
            selected_element_maps = dataset.element_maps

        e_cps_shape = (product.object_array.shape[0], product.object_array.shape[1])

        # Pre-flatten and move all element maps to GPU once
        m_cps_all = {}
        for emap in selected_element_maps:
            m_cps = emap.counts_per_second.flatten()
            if valid_pixel_index is not None:
                m_cps = m_cps[valid_pixel_index]
            if settings.gpu.enabled:
                m_cps = cp.asarray(m_cps)
            m_cps_all[emap.name] = m_cps

        # Warm-start solutions (None = start from zero)
        x0s: dict[str, Optional[np.ndarray]] = {
            emap.name: None for emap in selected_element_maps
        }

        # Build chunk schedule: [chunk_size, chunk_size, ..., remainder]
        max_iter = settings.lsmr.max_iter
        checkpoint_interval = settings.lsmr.checkpoint_interval
        if checkpoint_interval is None:
            chunks = [max_iter]
        else:
            chunks = [checkpoint_interval] * (max_iter // checkpoint_interval)
            if max_iter % checkpoint_interval:
                chunks.append(max_iter % checkpoint_interval)

        iterations_done = 0
        for chunk in tqdm.tqdm(chunks):
            enhanced_maps: list[ElementMap] = []
            e_cps = None

            for emap in selected_element_maps:
                logger.info(
                    f'Enhancing "{emap.name}" '
                    f"(iters {iterations_done + 1}–{iterations_done + chunk})..."
                )
                tic = time.perf_counter()

                # Solve the linear system A * e_cps = m_cps
                inline_timer = InlineTimer(settings.solver)
                inline_timer.start()
                if settings.solver == SolverTypes.LSMR:
                    result = lsmr(
                        A,
                        m_cps_all[emap.name],
                        damp=settings.lsmr.damping_factor,
                        maxiter=chunk,
                        atol=settings.lsmr.atol,
                        btol=settings.lsmr.btol,
                        x0=x0s[emap.name],
                    )
                inline_timer.end()

                logger.debug(f"Result: {result}")
                logger.debug(f"Number of iterations: {result[2]}")
                logger.debug(f"norm(b-Ax): {result[3]}")
                logger.debug(f"norm(A^H (b - Ax)): {result[4]}")
                logger.debug(f"norm(A)): {result[5]}")

                # Save solution as warm start for next chunk
                x0s[emap.name] = result[0]

                e_cps = result[0]
                if settings.gpu.enabled:
                    inline_timer = InlineTimer("Move upscaled counts GPU->CPU")
                    inline_timer.start()
                    e_cps = cp.asnumpy(e_cps)
                    inline_timer.end()
                e_cps = e_cps.reshape(e_cps_shape)

                toc = time.perf_counter()
                logger.info(f'Enhanced "{emap.name}" in {toc - tic:.4f} seconds.')

                enhanced_maps.append(ElementMap(emap.name, e_cps))

            iterations_done += chunk

            yield FluorescenceDataset(element_maps=enhanced_maps), iterations_done


def _get_probe_intensity_at_each_position(
    probe: np.ndarray, opr_mode_weights: np.ndarray
) -> np.ndarray:
    """Compute the weighted probe intensity at each scan position for OPR modes.

    Combines multiple Orthogonal Probe Relaxation (OPR) modes using the
    provided per-position mode weights, then adds the incoherent contribution
    from higher probe modes.

    Parameters
    ----------
    probe : ndarray
        Complex probe array of shape ``(n_opr, modes, height, width)``.
        May be a CuPy array for GPU computation.
    opr_mode_weights : ndarray
        Per-position OPR mixing coefficients of shape ``(n_opr, N)`` where
        *N* is the number of scan positions.  May be a CuPy array.

    Returns
    -------
    ndarray
        Real-valued probe intensity array of shape ``(N, height, width)``,
        giving the effective probe intensity at each scan position.
    """
    # - probe: (n_opr, modes, height, width) complex array
    # - opr_mode_weights: (n_opr, N)
    xp = cp.get_array_module(probe)
    p = xp.zeros((opr_mode_weights.shape[1], *probe.shape[2:]), dtype=xp.complex128)
    for i in range(len(opr_mode_weights)):
        p += probe[i, 0][None] * opr_mode_weights[i][:, None, None]
    p = np.abs(p) ** 2  # convert to intensity
    p += (np.abs(probe[0, 1:]) ** 2).sum(0)  # add intensity of other incoherent modes
    return p
