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
    Extract patches from 2D object. If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels.

    Parameters
    ----------
    image : Tensor
        The whole image.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    shape : tuple of int
        A tuple giving the patch shape in pixels.
    pad : Optional[int]
        If given, patches with larger size than the intended size by this amount are cropped
        out from the patches before shifting.

    Returns
    -------
    Tensor
        A tensor of shape (N, H, W) containing the extracted patches.
    """
    xp = cp.get_array_module(image)

    # Floating point ranges over which interpolations should be done
    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    # Crop one more pixel each side for Fourier shift
    # sys = sys_float.floor().int() - pad
    # eys = sys + shape[0] + 2 * pad
    # sxs = sxs_float.floor().int() - pad
    # exs = sxs + shape[1] + 2 * pad

    inline_timer = InlineTimer("array stuff")
    inline_timer.start()
    # sys = xp.int(xp.floor(sys_float)) - pad
    # sys = xp.array([int(xp.floor(y)) - pad for y in sys_float])
    sys = (xp.floor(sys_float) - pad).astype(int)
    eys = sys + shape[0] + 2 * pad
    # sxs = xp.int(xp.floor(sxs_float)) - pad
    # sxs = xp.array([int(xp.floor(x)) - pad for x in sxs_float])
    sxs = (xp.floor(sxs_float) - pad).astype(int)
    exs = sxs + shape[1] + 2 * pad
    inline_timer.end()

    # fractional_shifts = np.stack([sys_float - sys - pad, sxs_float - sxs - pad], -1)
    fractional_shifts = xp.stack([sys_float - sys - pad, sxs_float - sxs - pad], -1)

    pad_lengths = [
        max(-sxs.min(), 0),
        max(exs.max() - image.shape[1], 0),
        max(-sys.min(), 0),
        max(eys.max() - image.shape[0], 0),
    ]
    pad_lengths = [int(x) for x in pad_lengths]
    # image = torch.nn.functional.pad(image, pad_lengths)
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
    images : Tensor
        A [N, H, W] tensor of images.
    shifts : Tensor
        A [N, 2] tensor of shifts in pixels.
    strictly_preserve_zeros : bool
        If True, mask of strictly zero pixels will be generated and shifted
        by the same amount. Pixels that have a non-zero value in the shifted
        mask will be set to zero in the shifted image. This preserves the zero
        pixels in the original image, preventing FFT from introducing small
        non-zero values due to machine precision.

    Returns
    -------
    Tensor
        Shifted images.
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
    # freq_y, freq_x = torch.meshgrid(
    #     torch.fft.fftfreq(images.shape[-2]), torch.fft.fftfreq(images.shape[-1]), indexing="ij"
    # )
    freq_y, freq_x = xp.meshgrid(
        scipy_module.fft.fftfreq(images.shape[-2]),
        scipy_module.fft.fftfreq(images.shape[-1]),
        indexing="ij",
    )
    # freq_x = freq_x.to(ft_images.device)
    # freq_y = freq_y.to(ft_images.device)
    mult = xp.exp(
        1j
        * -2
        * xp.pi
        * (freq_x * shifts[:, 1, None, None] + freq_y * shifts[:, 0, None, None])
    )

    # mult = xp.exp(
    #     1j
    #     * -2
    #     * xp.pi
    #     * (freq_x * shifts[:, 1, None, None] + freq_y * shifts[:, 0, None, None])
    # )

    ft_images = ft_images * mult
    # shifted_images = pmath.ifft2_precise(ft_images)
    shifted_images = scipy_module.fft.ifft2(ft_images)
    # if not images.dtype.is_complex:
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
    Slice patches from an image at given window positions. The patch size is determined
    from the starting and ending coordinates in each direction, and is assumed to be
    the same for all patches.

    Parameters
    ----------
    image : Tensor
        A (H, W) tensor of the image.
    sy : Tensor
        A (N,) tensor of integers giving the starting y-coordinates of the patches.
    sx : Tensor
        A (N,) tensor of integers giving the starting x-coordinates of the patches.
    patch_size : tuple of int
        A tuple giving the patch shape in pixels.

    Returns
    -------
    Tensor
        A tensor of shape (N, h, w) containing the extracted patches.
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
    # x = np.arange(patch_size[1])[None, :] + sx[:, None]  # (N, w)
    # y = np.arange(patch_size[0])[None, :] + sy[:, None]  # (N, h)
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
    Place patches into a 2D object. If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels.

    Parameters
    ----------
    image : Tensor
        The whole image.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    patches : Tensor
        A (N, H, W) or (H, W) tensor of image patches.
    op : Literal["add", "set"]
        The operation to perform. "add" adds the patches to the image, 
        "set" sets the patches to the image replacing the existing values.
    adjoint_mode : bool
        If True, this function performs the exact adjoint operation of `extract_patches_fourier_shift`.
        This means it will run the adjoint operation of every step of the extraction 
        function in reverse order: it first zero-pads the patches, shifts them back, 
        and puts them back into the image. Turn it on if this function is used in
        backpropagating the gradient. Note that due to the zero-padding, ripple
        artifacts may appear around the borders of each patch, so it is not suitable
        for placing patches that are not gradients. In that case, set this to False,
        and it will skip the zero-padding and crop the patches before placing them
        to remove Fourier shift wrap-arounds.
    pad : Optional[int]
        If given, patches are padded (or cropped) by this amount before shifting. 
        The actual operations depend on `adjoint_mode`: when `adjoint_mode` is True, 
        the patches are padded with zeros; otherwise, pathces are cropped by this amount
        after shifting to remove the wrap-around portions.

    Returns
    -------
    Tensor
        A tensor with the same shape as the object with patches added onto it.
    """
    xp = cp.get_array_module(image)

    # If the input is a single patch, add the third dimension
    # and expand it to the correct number of patches
    if len(patches.shape) == 2:
        patches = xp.broadcast_to(patches[None], (len(positions),) + patches.shape)

    shape = patches.shape[-2:]

    if adjoint_mode:
        patch_padding = pad
        patches = xp.pad(patches, [(0, 0), (patch_padding, patch_padding), (patch_padding, patch_padding)])
    else:
        patch_padding = -pad

    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    # Crop one more pixel each side for Fourier shift
    sys = (xp.floor(sys_float) - patch_padding).astype(int)
    eys = sys + shape[0] + 2 * patch_padding
    sxs = (xp.floor(sxs_float) - patch_padding).astype(int)
    exs = sxs + shape[1] + 2 * patch_padding

    fractional_shifts = xp.stack([sys_float - sys - patch_padding, sxs_float - sxs - patch_padding], -1)

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
            abs(patch_padding) : patches.shape[-1] - abs(patch_padding)
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
    op: Literal["add", "set"] = "add"
) -> ArrayType:
    """
    Slice patches from an image at given window positions. The patch size is assumed 
    to be the same for all patches.
    
    Parameters
    ----------
    image : Tensor
        A (H, W) tensor of the buffer to place the patches into.
    patches : Tensor
        A (N, h, w) tensor of the patches.
    sy : Tensor
        A (N,) tensor of integers giving the starting y-coordinates of the patches.
    sx : Tensor
        A (N,) tensor of integers giving the starting x-coordinates of the patches.
    op : Literal["add", "set"]
        The operation to perform. "add" adds the patches to the image, 
        "set" sets the patches to the image replacing the existing values.
    
    Returns
    -------
    Tensor
        A tensor of shape (H, W) containing the image with patches added or set.
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
    if cp.get_array_module(array) == cp:
        module = cupyx.scipy
    else:
        module = scipy
    module: scipy
    return module

