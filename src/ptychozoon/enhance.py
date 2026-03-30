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
from scipy.sparse.linalg import lsmr, LinearOperator


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
    # counts_per_second_path: str
    # channel_names_path: str


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

    def get_patch(self) -> np.ndarray:
        """Interpolate array support to extract patch."""
        patch = self._weight00 * self._support[:-1, :-1]
        patch += self._weight01 * self._support[:-1, 1:]
        patch += self._weight10 * self._support[1:, :-1]
        patch += self._weight11 * self._support[1:, 1:]
        return patch

    def accumulate_patch(self, patch: np.ndarray) -> None:
        """Add patch update to array support."""
        self._support[:-1, :-1] += self._weight00 * patch
        self._support[:-1, 1:] += self._weight01 * patch
        self._support[1:, :-1] += self._weight10 * patch
        self._support[1:, 1:] += self._weight11 * patch


class VSPILinearOperator(LinearOperator):
    """Linear operator A for VSPI: A[M,N] * X[N,P] = B[M,P]

    Where:
        M: number of XRF positions (scan points)
        N: number of ptychography object pixels
        P: number of XRF channels
    """

    def __init__(self, product: Product) -> None:
        """Initialize the linear operator.

        Args:
            product: Ptychography reconstruction product
        """
        M = len(product.probe_positions)  # Number of scan points
        N = product.object_array.shape[0] * product.object_array.shape[1]  # Total pixels
        super().__init__(float, (M, N))
        self._product = product

        # Cache object dimensions
        self._object_height_px = product.object_array.shape[0]
        self._object_width_px = product.object_array.shape[1]

        # Cache pixel size and center
        self._pixel_height_m = product.pixel_size_m[0]
        self._pixel_width_m = product.pixel_size_m[1]
        self._center_y_m = product.object_center_m[0]
        self._center_x_m = product.object_center_m[1]

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

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Forward operator: A * x

        Args:
            x: Flattened object array (N,)

        Returns:
            Result vector (M,)
        """
        object_array = x.reshape((self._object_height_px, self._object_width_px))
        result = np.zeros(len(self._product.probe_positions))

        # Get probe intensity (sum over modes)
        probe_intensity = np.sum(np.abs(self._product.probe) ** 2, axis=0)
        psf = probe_intensity / probe_intensity.sum()

        for index, position in enumerate(self._product.probe_positions):
            # Convert probe position to object coordinates
            probe_y_m, probe_x_m = position
            obj_y_px, obj_x_px = self._probe_to_object_coords(probe_y_m, probe_x_m)

            # Extract and accumulate patch
            interpolator = ArrayPatchInterpolator(object_array, obj_y_px, obj_x_px, psf.shape)
            result[index] = np.sum(psf * interpolator.get_patch())

        return result

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Adjoint operator: A^T * x

        Args:
            x: Input vector (M,)

        Returns:
            Flattened object array (N,)
        """
        object_array = np.zeros((self._object_height_px, self._object_width_px))

        # Get probe intensity (sum over modes)
        probe_intensity = np.sum(np.abs(self._product.probe) ** 2, axis=0)
        psf = probe_intensity / probe_intensity.sum()

        for index, position in enumerate(self._product.probe_positions):
            # Convert probe position to object coordinates
            probe_y_m, probe_x_m = position
            obj_y_px, obj_x_px = self._probe_to_object_coords(probe_y_m, probe_x_m)

            # Accumulate weighted patch
            interpolator = ArrayPatchInterpolator(object_array, obj_y_px, obj_x_px, psf.shape)
            interpolator.accumulate_patch(x[index] * psf)

        return object_array.flatten()


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

    def enhance(
        self,
        dataset: FluorescenceDataset,
        product: Product,
        valid_pixel_index: Optional[list[int]] = None,
        select_maps: Optional[list[str]] = None,
    ) -> FluorescenceDataset:
        """Enhance fluorescence dataset using ptychography product.

        Args:
            dataset: Input fluorescence dataset
            product: Ptychography reconstruction product

        Returns:
            Enhanced fluorescence dataset with higher resolution
        """
        enhanced_maps: list[ElementMap] = []
        A = VSPILinearOperator(product)

        for emap in dataset.element_maps:
            if select_maps is not None and emap.name not in select_maps:
                continue

            logger.info(f'Enhancing "{emap.name}"...')
            tic = time.perf_counter()

            # Flatten the measured counts per second
            m_cps = emap.counts_per_second.flatten()
            if valid_pixel_index is not None:
                m_cps = m_cps[valid_pixel_index]

            # Solve the linear system A * e_cps = m_cps
            result = lsmr(
                A, # size --> number of probe positions x number of pixels in ptycho object
                m_cps, # size --> number of counts per second measurements (longer)
                damp=self.damping_factor,
                maxiter=self.max_iterations,
                show=True,
            )
            # The way that this is defined implies that m_cps should be equal to the number of probe positions.
            # But how do you get the XRF data to be the correct size?

            logger.debug(f"LSMR result: {result}")

            # Reshape to object dimensions
            e_cps_shape = (product.object_array.shape[0], product.object_array.shape[1])
            e_cps = result[0].reshape(e_cps_shape)

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
