"""VSPI Fluorescence Enhancement Algorithm

Re-implementation of the Virtual Single Pixel Imaging algorithm for enhancing
fluorescence data using ptychography reconstructions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import logging
import time

import numpy as np
import cupy as cp

from chronos.timer_utils import timer, InlineTimer

from ptychozoon.settings import DeconvolutionEnhancementSettings, InterpolationTypes
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
    - probe: (modes, height, width) complex array
    - object_array: (height, width) complex array
    - pixel_size_m: (y, x) pixel sizes in meters
    - object_center_m: (y, x) center coordinates in meters
    """

    probe_positions: np.ndarray  # (N, 2) float array [y, x] in meters
    probe: np.ndarray  # (modes, height, width) complex array
    object_array: np.ndarray  # (height, width) complex array
    pixel_size_m: tuple[float, float]  # (pixel_height_m, pixel_width_m)
    object_center_m: tuple[float, float]  # (center_y_m, center_x_m)


class ArrayPatchInterpolator:
    """Bilinear interpolation for extracting and accumulating array patches."""

    def __init__(
        self,
        array: np.ndarray,
        center_y_px: float,
        center_x_px: float,
        shape: tuple[int, int]
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


def _make_vspi_linear_operator(product: Product, xp, LinearOperator, settings: DeconvolutionEnhancementSettings):
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
            N = product.object_array.shape[0] * product.object_array.shape[1]  # Total pixels
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
            self._object_array = product.object_array

        # @timer()
        def _probe_to_object_coords(self, probe_y_m: float, probe_x_m: float) -> tuple[float, float]:
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
            # input v is the upscaled XRF array after flattening
            object_array = v.reshape((self._object_height_px, self._object_width_px))
            result = xp.zeros(len(self._probe_positions))

            # Get probe intensity (sum over modes)
            probe_intensity = xp.sum(xp.abs(self._probe) ** 2, axis=0)
            psf = probe_intensity / probe_intensity.sum()

            inline_timer = InlineTimer("Extract patches")
            inline_timer.start()
            if self.interpolation_type == InterpolationTypes.FOURIER:
                # convert probe positions to object coordinates
                positions_px = xp.array([self._probe_to_object_coords(pos_m[0], pos_m[1]) for pos_m in self._probe_positions])
                positions_px += -xp.array([1, 1]) * 0.5
                extracted_patches = extract_patches_fourier_shift(object_array, positions_px, psf.shape)
                # The extracted patches do not match the barycentric interpolation that was orignally here unless
                # `positions_px` is replaced `positions_px - xp.array([1, 1]) * 0.5`. 
                result = (extracted_patches * psf).sum((1, 2))
            elif self.interpolation_type == InterpolationTypes.BARYCENTRIC:
                result = xp.zeros(len(self._probe_positions))
                for index, position in enumerate(self._probe_positions):
                    # Convert probe position to object coordinates
                    probe_y_m, probe_x_m = float(position[0]), float(position[1])
                    obj_y_px, obj_x_px = self._probe_to_object_coords(probe_y_m, probe_x_m)

                    # Extract and accumulate patch
                    interpolator = ArrayPatchInterpolator(object_array, obj_y_px, obj_x_px, psf.shape)
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
            object_array = xp.zeros((self._object_height_px, self._object_width_px))

            # Get probe intensity (sum over modes)
            probe_intensity = xp.sum(xp.abs(self._probe) ** 2, axis=0)
            psf = probe_intensity / probe_intensity.sum()

            inline_timer = InlineTimer("Accumulate patches")
            inline_timer.start()
            if self.interpolation_type == InterpolationTypes.FOURIER:
                positions_px = xp.array([self._probe_to_object_coords(pos_m[0], pos_m[1]) for pos_m in self._probe_positions])
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
                    obj_y_px, obj_x_px = self._probe_to_object_coords(probe_y_m, probe_x_m)

                    # Accumulate weighted patch
                    interpolator = ArrayPatchInterpolator(object_array, obj_y_px, obj_x_px, psf.shape)
                    interpolator.accumulate_patch(u[index] * psf)
            inline_timer.end()

            return object_array.flatten()

    return VSPILinearOperator(interpolation_type=settings.interpolation)


class VSPIFluorescenceEnhancingAlgorithm:
    """Virtual Single Pixel Imaging algorithm for fluorescence enhancement.

    This algorithm uses ptychography reconstruction data to enhance fluorescence
    measurements by solving a linear system that accounts for the finite size
    of the X-ray probe.
    """

    def __init__(self, damping_factor: float = 0.0, max_iterations: int = 100) -> None:
        """Initialize the VSPI algorithm.

        Args:
            damping_factor: Damping parameter for LSMR solver (default: 0.0)
            max_iterations: Maximum iterations for LSMR solver (default: 100)
        """
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations

    @timer()
    def enhance(
        self,
        dataset: FluorescenceDataset,
        product: Product,
        valid_pixel_index: Optional[list[int]] = None,
        select_maps: Optional[list[str]] = None,
        use_gpu: bool = False,
        settings: Optional[DeconvolutionEnhancementSettings] = None,
    ) -> FluorescenceDataset:
        """Enhance fluorescence dataset using ptychography product.

        Args:
            dataset: Input fluorescence dataset
            product: Ptychography reconstruction product
            valid_pixel_index: Optional indices of valid scan positions
            select_maps: Optional list of element names to enhance (all if None)
            use_gpu: If True, use GPU via cupyx for lsmr and array operations

        Returns:
            Enhanced fluorescence dataset with higher resolution
        """
        if settings is None:
            settings = DeconvolutionEnhancementSettings()
        if use_gpu:
            from cupyx.scipy.sparse.linalg import lsmr, LinearOperator
            xp = cp
            # Move probe and object_array to GPU; probe_positions stays on CPU
            # for efficient Python-level iteration over scan positions
            inline_timer = InlineTimer("Move data to GPU")
            inline_timer.start()
            gpu_product = Product(
                probe_positions=product.probe_positions,
                probe=cp.asarray(product.probe),
                object_array=cp.asarray(product.object_array),
                pixel_size_m=product.pixel_size_m,
                object_center_m=product.object_center_m,
            )
            inline_timer.end()
        else:
            from scipy.sparse.linalg import lsmr, LinearOperator
            xp = np
            gpu_product = product

        enhanced_maps: list[ElementMap] = []
        inline_timer = InlineTimer("Make VSPI linear operator")
        inline_timer.start()
        A = _make_vspi_linear_operator(gpu_product, xp, LinearOperator, settings)
        inline_timer.end()
        
        if select_maps is not None:
            selected_element_maps = [emap for emap in dataset.element_maps if emap.name in select_maps]
        else:
            selected_element_maps = dataset.element_maps

        for emap in selected_element_maps:
            # if select_maps is not None and emap.name not in select_maps:
            #     continue

            logger.info(f'Enhancing "{emap.name}"...')
            tic = time.perf_counter()

            # Flatten the measured counts per second
            m_cps = emap.counts_per_second.flatten()
            if valid_pixel_index is not None:
                m_cps = m_cps[valid_pixel_index]

            if use_gpu:
                m_cps = cp.asarray(m_cps)

            # Solve the linear system A * e_cps = m_cps
            inline_timer = InlineTimer("lsmr")
            inline_timer.start()
            result = lsmr(
                A, # size --> number of probe positions x number of pixels in ptycho object (30777 x 153908)
                m_cps, # size --> number of counts per second measurements (3077)
                damp=self.damping_factor,
                maxiter=self.max_iterations,
                # show=True,
            )
            inline_timer.end()
            # The way that this is defined implies that m_cps should be equal to the number of probe positions.
            # But how do you get the XRF data to be the correct size?

            logger.debug(f"LSMR result: {result}")

            # Reshape to object dimensions
            e_cps_shape = (product.object_array.shape[0], product.object_array.shape[1])
            e_cps = result[0]
            if use_gpu:
                inline_timer = InlineTimer("Move upscaled counts GPU->CPU")
                inline_timer.start()
                e_cps = cp.asnumpy(e_cps)
                inline_timer.end()
            inline_timer = InlineTimer("Reshape upscaled counts")
            inline_timer.start()
            e_cps = e_cps.reshape(e_cps_shape)
            inline_timer.end()

            # Create enhanced element map
            emap_enhanced = ElementMap(emap.name, e_cps)

            toc = time.perf_counter()
            logger.info(f'Enhanced "{emap.name}" in {toc - tic:.4f} seconds.')

            enhanced_maps.append(emap_enhanced)

        return FluorescenceDataset(
            element_maps=enhanced_maps,
            # counts_per_second_path=dataset.counts_per_second_path,
            # channel_names_path=dataset.channel_names_path,
        )
