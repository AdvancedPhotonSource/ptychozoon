# Copyright © 2026 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com/AdvancedPhotonSource/ptychozoon/blob/main/LICENSE.TXT
"""VSPI Fluorescence Enhancement Algorithm

Re-implementation of the Virtual Single Pixel Imaging algorithm for enhancing
fluorescence data using ptychography reconstructions.
"""

from __future__ import annotations
from typing import Generator, Optional
import logging
import time
import tqdm

import numpy as np
import cupy as cp

from chronos.timer_utils import timer, InlineTimer

from ptychozoon.data_structures import (
    ElementMap,
    FluorescenceDataset,
    PtychographyProduct,
)
from ptychozoon.patches import BilinearArrayPatchInterpolator
from ptychozoon.settings import (
    DeconvolutionEnhancementSettings,
    InterpolationTypes,
    SolverTypes,
)
from .patches import extract_patches_fourier_shift, place_patches_fourier_shift

logger = logging.getLogger(__name__)


def _make_vspi_linear_operator(
    product: PtychographyProduct,
    xp,
    LinearOperator,
    settings: DeconvolutionEnhancementSettings,
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
            self._object_array = product.object_array
            self._opr_mode_weights = product.opr_mode_weights
            self._call_count = 0

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
                probe_intensity = xp.sum(xp.abs(self._probe[0]) ** 2, axis=0)
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
                    interpolator = BilinearArrayPatchInterpolator(
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
                probe_intensity = xp.sum(xp.abs(self._probe[0]) ** 2, axis=0)
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
                    interpolator = BilinearArrayPatchInterpolator(
                        object_array, obj_y_px, obj_x_px, psf.shape
                    )
                    interpolator.accumulate_patch(u[index] * psf)
            inline_timer.end()

            # import matplotlib.pyplot as plt; plt.imshow(object_array.get());plt.colorbar();plt.title("deconvolved?");plt.show()
            return object_array.flatten()

    return VSPILinearOperator(interpolation_type=settings._interpolation)


def _make_gradient_regularizer(shape: tuple[int, int], lam: float, xp, LinearOperator):
    """Build a gradient-smoothness regularisation operator for a 2-D image.

    Constructs a linear operator R such that ``R @ x`` stacks the horizontal
    and vertical finite-difference images of ``x`` (reshaped to *shape*).
    Solving the augmented system ``[A; R] x = [b; 0]`` is equivalent to
    minimising ``‖Ax − b‖² + λ² ‖∇x‖²``.

    Args:
        shape: ``(H, W)`` shape of the 2-D image.
        lam: Regularisation weight λ.  Scales every output of R.
        xp: Array module (``numpy`` or ``cupy``).
        LinearOperator: Base class (``scipy`` or ``cupyx`` variant).

    Returns:
        LinearOperator of shape ``(H*(W-1) + (H-1)*W, H*W)``.
    """
    H, W = shape
    N = H * W
    M_reg = H * (W - 1) + (H - 1) * W

    def _matvec(x):
        f = x.reshape(H, W)
        dx = xp.diff(f, axis=1)  # (H, W-1): horizontal differences
        dy = xp.diff(f, axis=0)  # (H-1, W): vertical differences
        return lam * xp.concatenate([dx.ravel(), dy.ravel()])

    def _rmatvec(u):
        u_x = u[: H * (W - 1)].reshape(H, W - 1)
        u_y = u[H * (W - 1) :].reshape(H - 1, W)
        # Adjoint of horizontal forward-diff: maps (H, W-1) -> (H, W)
        result = xp.concatenate(
            [-u_x[:, :1], -xp.diff(u_x, axis=1), u_x[:, -1:]], axis=1
        )
        # Adjoint of vertical forward-diff: maps (H-1, W) -> (H, W), accumulated
        result = result + xp.concatenate(
            [-u_y[:1, :], -xp.diff(u_y, axis=0), u_y[-1:, :]], axis=0
        )
        return lam * result.ravel()

    return LinearOperator((M_reg, N), matvec=_matvec, rmatvec=_rmatvec, dtype=float)


def _make_augmented_operator(A, R, xp, LinearOperator):
    """Stack two linear operators vertically: ``[A; R]``.

    Args:
        A: Primary operator of shape ``(M_A, N)``.
        R: Regularisation operator of shape ``(M_R, N)``.
        xp: Array module (``numpy`` or ``cupy``).
        LinearOperator: Base class (``scipy`` or ``cupyx`` variant).

    Returns:
        LinearOperator of shape ``(M_A + M_R, N)``.
    """
    M_A, N = A.shape
    M_R, _ = R.shape

    def _matvec(x):
        return xp.concatenate([A @ x, R @ x])

    def _rmatvec(u):
        return A.T @ u[:M_A] + R.T @ u[M_A:]

    return LinearOperator((M_A + M_R, N), matvec=_matvec, rmatvec=_rmatvec, dtype=float)


class VSPIFluorescenceEnhancingAlgorithm:
    """Virtual Single Pixel Imaging algorithm for fluorescence enhancement.

    This algorithm uses ptychography reconstruction data to enhance fluorescence
    measurements by solving a linear system that accounts for the finite size
    of the X-ray probe.
    """

    def enhance(
        self,
        dataset: FluorescenceDataset,
        product: PtychographyProduct,
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
            gpu_product = PtychographyProduct(
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

        e_cps_shape = (product.object_array.shape[0], product.object_array.shape[1])

        if settings.lsmr.gradient_smoothness > 0:
            R = _make_gradient_regularizer(
                e_cps_shape, settings.lsmr.gradient_smoothness, xp, LinearOperator
            )
            A_solve = _make_augmented_operator(A, R, xp, LinearOperator)
            b_padding = xp.zeros(R.shape[0])
        else:
            A_solve = A
            b_padding = None

        if select_maps is not None:
            selected_element_maps = [
                emap for emap in dataset.element_maps if emap.name in select_maps
            ]
        else:
            selected_element_maps = dataset.element_maps

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
                    b = (
                        xp.concatenate([m_cps_all[emap.name], b_padding])
                        if b_padding is not None
                        else m_cps_all[emap.name]
                    )
                    result = lsmr(
                        A_solve,
                        b,
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
