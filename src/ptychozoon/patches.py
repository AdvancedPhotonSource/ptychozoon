# Copyright © 2026 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com/AdvancedPhotonSource/ptychozoon/blob/main/LICENSE.TXT
from typing import Literal, Optional, Tuple, TypeVar, Union
from types import ModuleType
from chronos.timer_utils import timer, InlineTimer
import cupy as cp
import numpy as np
import cupyx
import cupyx.scipy.fft
import scipy
import scipy.fft

ArrayType = Union[cp.ndarray, np.ndarray]


@timer()
def extract_patches_fourier_shift(
    image: ArrayType,
    positions: ArrayType,
    shape: Tuple[int, int],
    pad: Optional[int] = 1,
) -> ArrayType:
    """
    Extract patches from a 2D array. If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels.

    Parameters
    ----------
    image : ndarray
        The whole image.
    positions : ndarray
        An array of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    shape : tuple of int
        A tuple giving the patch shape in pixels.
    pad : Optional[int]
        If given, patches with larger size than the intended size by this amount are cropped
        out from the patches before shifting.

    Returns
    -------
    ndarray
        An array of shape (N, H, W) containing the extracted patches.
    """
    xp = cp.get_array_module(image)

    # Floating point ranges over which interpolations should be done
    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    sys = (xp.floor(sys_float) - pad).astype(int)
    eys = sys + shape[0] + 2 * pad
    sxs = (xp.floor(sxs_float) - pad).astype(int)
    exs = sxs + shape[1] + 2 * pad

    fractional_shifts = xp.stack([sys_float - sys - pad, sxs_float - sxs - pad], -1)

    pad_lengths = [
        max(-sxs.min(), 0),
        max(exs.max() - image.shape[1], 0),
        max(-sys.min(), 0),
        max(eys.max() - image.shape[0], 0),
    ]
    pad_lengths = [int(x) for x in pad_lengths]
    inline_timer = InlineTimer("pad image")
    inline_timer.start()
    image = xp.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]
    inline_timer.end()

    patches = batch_slice(
        image, sys, sxs, patch_size=[shape[i] + 2 * pad for i in range(2)]
    )

    # Apply Fourier shift to account for fractional shifts
    # if not torch.allclose(fractional_shifts, torch.zeros_like(fractional_shifts), atol=1e-7):
    if not xp.allclose(fractional_shifts, xp.zeros_like(fractional_shifts), atol=1e-7):
        patches = fourier_shift(patches, -fractional_shifts)
    patches = patches[:, pad : patches.shape[-2] - pad, pad : patches.shape[-1] - pad]

    return patches


@timer()
def fourier_shift(
    images: ArrayType, shifts: ArrayType, strictly_preserve_zeros: bool = False
) -> ArrayType:
    """
    Apply Fourier shift to a batch of images.

    Parameters
    ----------
    images : ndarray
        An array of shape (N, H, W) containing the images to shift.
    shifts : ndarray
        An array of shape (N, 2) giving the shifts in pixels.
    strictly_preserve_zeros : bool
        If True, a mask of strictly zero pixels will be generated and shifted
        by the same amount. Pixels that have a non-zero value in the shifted
        mask will be set to zero in the shifted image. This preserves the zero
        pixels in the original image, preventing FFT from introducing small
        non-zero values due to machine precision.

    Returns
    -------
    ndarray
        Shifted images of the same shape as *images*.
    """
    xp = cp.get_array_module(images)
    scipy_module = get_scipy_module(images)

    if strictly_preserve_zeros:
        zero_mask = images == 0
        zero_mask = zero_mask.float()
        zero_mask_shifted = fourier_shift(
            zero_mask, shifts, strictly_preserve_zeros=False
        )
    ft_images = scipy_module.fft.fft2(images)
    freq_y, freq_x = xp.meshgrid(
        scipy_module.fft.fftfreq(images.shape[-2]),
        scipy_module.fft.fftfreq(images.shape[-1]),
        indexing="ij",
    )
    mult = xp.exp(
        1j
        * -2
        * xp.pi
        * (freq_x * shifts[:, 1, None, None] + freq_y * shifts[:, 0, None, None])
    )

    ft_images = ft_images * mult
    shifted_images = scipy_module.fft.ifft2(ft_images)
    if not xp.iscomplexobj(images):
        shifted_images = shifted_images.real
    if strictly_preserve_zeros:
        shifted_images[zero_mask_shifted > 0] = 0
    return shifted_images


@timer()
def batch_slice(
    image: ArrayType, sy: ArrayType, sx: ArrayType, patch_size: Tuple[int, int]
) -> ArrayType:
    """
    Slice patches from an image at given window positions. The patch size is assumed
    to be the same for all patches.

    Parameters
    ----------
    image : ndarray
        A (H, W) array of the image.
    sy : ndarray
        A (N,) array of integers giving the starting y-coordinates of the patches.
    sx : ndarray
        A (N,) array of integers giving the starting x-coordinates of the patches.
    patch_size : tuple of int
        A tuple giving the patch shape in pixels.

    Returns
    -------
    ndarray
        An array of shape (N, h, w) containing the extracted patches.
    """
    xp = cp.get_array_module(image)

    h, w = image.shape[-2:]
    if (
        sy.min() < 0
        or sy.max() + patch_size[0] > image.shape[-2]
        or sx.min() < 0
        or sx.max() + patch_size[1] > image.shape[-1]
    ):
        raise ValueError("Patch indices are out of bounds.")

    x = xp.arange(patch_size[1])[None, :] + sx[:, None]  # (N, w)
    y = xp.arange(patch_size[0])[None, :] + sy[:, None]  # (N, h)
    inds = (y * w)[:, :, None] + x[:, None, :]  # (N, h, w)
    patches = image.reshape(-1)[inds.reshape(-1)]
    patches = patches.reshape(len(sy), patch_size[0], patch_size[1])
    return patches


@timer()
def place_patches_fourier_shift(
    image: ArrayType,
    positions: ArrayType,
    patches: ArrayType,
    op: Literal["add", "set"] = "add",
    adjoint_mode: bool = True,
    pad: Optional[int] = 1,
) -> ArrayType:
    """
    Place patches into a 2D array. If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels.

    Parameters
    ----------
    image : ndarray
        The whole image.
    positions : ndarray
        An array of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    patches : ndarray
        A (N, H, W) or (H, W) array of image patches.
    op : Literal["add", "set"]
        The operation to perform. ``"add"`` adds the patches to the image,
        ``"set"`` replaces the existing values with the patch values.
    adjoint_mode : bool
        If True, this function performs the exact adjoint operation of
        :func:`extract_patches_fourier_shift`. It runs the adjoint of every
        extraction step in reverse order: zero-pads the patches, shifts them
        back, and places them into the image. Use this when backpropagating
        gradients. Note that the zero-padding can introduce ripple artifacts
        around patch borders, so set this to False when placing non-gradient
        patches; in that case the patches are cropped after shifting to remove
        Fourier wrap-around.
    pad : Optional[int]
        If given, patches are padded (or cropped) by this amount before
        shifting. When ``adjoint_mode`` is True the patches are zero-padded;
        otherwise they are cropped after shifting to remove wrap-around.

    Returns
    -------
    ndarray
        An array with the same shape as *image* with patches added or set.
    """
    xp = cp.get_array_module(image)

    # If the input is a single patch, add the third dimension
    # and expand it to the correct number of patches
    if len(patches.shape) == 2:
        patches = xp.broadcast_to(patches[None], (len(positions),) + patches.shape)

    shape = patches.shape[-2:]

    if adjoint_mode:
        patch_padding = pad
        patches = xp.pad(
            patches,
            [(0, 0), (patch_padding, patch_padding), (patch_padding, patch_padding)],
        )
    else:
        patch_padding = -pad

    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    # Crop one more pixel each side for Fourier shift
    sys = (xp.floor(sys_float) - patch_padding).astype(int)
    eys = sys + shape[0] + 2 * patch_padding
    sxs = (xp.floor(sxs_float) - patch_padding).astype(int)
    exs = sxs + shape[1] + 2 * patch_padding

    fractional_shifts = xp.stack(
        [sys_float - sys - patch_padding, sxs_float - sxs - patch_padding], -1
    )

    pad_lengths = [
        max(-sxs.min(), 0),
        max(exs.max() - image.shape[1], 0),
        max(-sys.min(), 0),
        max(eys.max() - image.shape[0], 0),
    ]
    pad_lengths = [int(x) for x in pad_lengths]
    image = xp.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]

    if not xp.allclose(fractional_shifts, xp.zeros_like(fractional_shifts), atol=1e-7):
        patches = fourier_shift(patches, fractional_shifts)
    if not adjoint_mode:
        patches = patches[
            :,
            abs(patch_padding) : patches.shape[-2] - abs(patch_padding),
            abs(patch_padding) : patches.shape[-1] - abs(patch_padding),
        ]

    inline_timer = InlineTimer("add or set patches on image")
    inline_timer.start()
    image = batch_put(image, patches, sys, sxs, op=op)
    inline_timer.end()

    # Undo padding
    image = image[
        pad_lengths[2] : image.shape[0] - pad_lengths[3],
        pad_lengths[0] : image.shape[1] - pad_lengths[1],
    ]
    return image


@timer()
def batch_put(
    image: ArrayType,
    patches: ArrayType,
    sy: ArrayType,
    sx: ArrayType,
    op: Literal["add", "set"] = "add",
) -> ArrayType:
    """
    Place patches into an array at given window positions. The patch size is assumed
    to be the same for all patches.

    Parameters
    ----------
    image : ndarray
        A (H, W) array acting as the buffer to place the patches into.
    patches : ndarray
        A (N, h, w) array of the patches to place.
    sy : ndarray
        A (N,) array of integers giving the starting y-coordinates of the patches.
    sx : ndarray
        A (N,) array of integers giving the starting x-coordinates of the patches.
    op : Literal["add", "set"]
        The operation to perform. ``"add"`` adds the patches to the image,
        ``"set"`` replaces the existing values with the patch values.

    Returns
    -------
    ndarray
        An array of shape (H, W) containing the image with patches added or set.
    """
    xp = cp.get_array_module(image)
    h, w = image.shape[-2:]
    if (
        sy.min() < 0
        or sy.max() + patches.shape[-2] > image.shape[-2]
        or sx.min() < 0
        or sx.max() + patches.shape[-1] > image.shape[-1]
    ):
        raise ValueError("Patch indices are out of bounds.")

    patch_size = patches.shape[-2:]
    x = xp.arange(patch_size[1])[None, :] + sx[:, None]  # (N, w)
    y = xp.arange(patch_size[0])[None, :] + sy[:, None]  # (N, h)
    inds = (y * w)[:, :, None] + x[:, None, :]  # (N, h, w)

    image_flat = image.reshape(-1)
    inds_flat = inds.reshape(-1)
    patches_flat = patches.reshape(-1)

    if op == "add":
        if xp is cp:
            cupyx.scatter_add(image_flat, inds_flat, patches_flat)
        else:
            np.add.at(image_flat, inds_flat, patches_flat)
    else:
        image_flat[inds_flat] = patches_flat
    return image_flat.reshape(h, w)


def get_scipy_module(array: ArrayType):
    """Return the appropriate SciPy module for the given array type.

    Selects ``cupyx.scipy`` when *array* is a CuPy array, and ``scipy``
    for NumPy arrays, enabling array-module-agnostic code.

    Parameters
    ----------
    array : ndarray
        Array whose module determines the SciPy variant returned.

    Returns
    -------
    module
        ``cupyx.scipy`` for CuPy arrays, ``scipy`` for NumPy arrays.
    """
    if cp.get_array_module(array) == cp:
        module = cupyx.scipy
    else:
        module = scipy
    module: scipy
    return module


class BilinearArrayPatchInterpolator:
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

    # called very frequently; do not time
    def get_patch(self) -> np.ndarray:
        """Interpolate array support to extract patch."""
        patch = self._weight00 * self._support[:-1, :-1]
        patch += self._weight01 * self._support[:-1, 1:]
        patch += self._weight10 * self._support[1:, :-1]
        patch += self._weight11 * self._support[1:, 1:]
        return patch

    # called very frequently; do not time
    def accumulate_patch(self, patch: np.ndarray) -> None:
        """Add patch update to array support."""
        self._support[:-1, :-1] += self._weight00 * patch
        self._support[:-1, 1:] += self._weight01 * patch
        self._support[1:, :-1] += self._weight10 * patch
        self._support[1:, 1:] += self._weight11 * patch
