"""Functions for saving VSPI fluorescence enhancement results."""

from __future__ import annotations

import os
from typing import List, Tuple

import h5py
import numpy as np
import tifffile

from ptychozoon.data_structures import FluorescenceDataset, ElementMap
from ptychozoon.settings import SaveFileExtensions


def save_vspi_results(
    folder: str,
    name: str,
    vspi_results: List[Tuple[FluorescenceDataset, int]],
    filetype: SaveFileExtensions,
    save_every_n_frames: int = 1,
) -> None:
    """Save VSPI results to disk.

    Each element's 2D maps across all checkpoint iterations are stacked into a
    3D array of shape (n_frames, height, width) before saving.  When saving to
    HDF5, an additional ``"epochs"`` dataset is written containing the
    iteration number corresponding to each frame.

    Args:
        folder: Parent output directory.
        name: Name suffix for this result set.
        vspi_results: List of ``(FluorescenceDataset, iteration_number)`` tuples
            as returned by ``VSPIFluorescenceEnhancingAlgorithm.enhance``.
        filetype: Output format — ``SaveFileExtensions.TIFF`` or
            ``SaveFileExtensions.H5``.
        save_every_n_frames: Stride for subsampling results before saving.
            For example, ``2`` saves every other result. Defaults to ``1``
            (save all results).
    """
    vspi_results = vspi_results[::save_every_n_frames]
    epochs = np.array([iteration for _, iteration in vspi_results])
    element_names = [em.name for em in vspi_results[0][0].element_maps]

    element_arrays: dict[str, np.ndarray] = {}
    for element_name in element_names:
        frames = []
        for dataset, _ in vspi_results:
            em = next(e for e in dataset.element_maps if e.name == element_name)
            frames.append(em.counts_per_second)
        element_arrays[element_name] = np.stack(frames, axis=0)

    if filetype == SaveFileExtensions.TIFF:
        _save_tiff(folder, name, element_arrays)
    elif filetype == SaveFileExtensions.H5:
        _save_h5(folder, name, element_arrays, epochs)
    else:
        raise ValueError(f"Unsupported filetype: {filetype!r}")


def _save_tiff(folder: str, name: str, element_arrays: dict[str, np.ndarray]) -> None:
    """Write per-element 3D image stacks to individual TIFF files.

    Parameters
    ----------
    folder : str
        Output directory (created if it does not exist).
    name : str
        Name prefix used in every output filename.
    element_arrays : dict[str, ndarray]
        Mapping from element name to a ``(n_frames, height, width)`` array.
    """
    if not os.path.exists(folder):
        os.mkdir(os.path.dirname(folder))
    for element_name, array_3d in element_arrays.items():
        tiff_path = os.path.join(folder, name + "_all_frames_" + element_name + SaveFileExtensions.TIFF)
        tifffile.imwrite(tiff_path, array_3d)
    print(f"Element arrays saved to {tiff_path}")


def _save_h5(
    folder: str,
    name: str,
    element_arrays: dict[str, np.ndarray],
    epochs: np.ndarray,
) -> None:
    """Write per-element 3D image stacks to a single HDF5 file.

    Each element is stored as a separate dataset at the root level of the
    file, with the dataset name equal to the element name. An ``epochs``
    dataset records the iteration number corresponding to each frame.

    Parameters
    ----------
    folder : str
        Output directory (created if it does not exist).
    name : str
        Name stem used in the output filename.
    element_arrays : dict[str, ndarray]
        Mapping from element name to a ``(n_frames, height, width)`` array.
    epochs : ndarray
        1-D array of iteration numbers, one per saved frame.
    """
    if not os.path.exists(folder):
        os.mkdir(os.path.dirname(folder))
    h5_path = os.path.join(folder, name + "_all_frames" + SaveFileExtensions.H5)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("epochs", data=epochs)
        for element_name, array_3d in element_arrays.items():
            f.create_dataset(element_name, data=array_3d)
    print(f"Element arrays saved to {h5_path}")


def load_vspi_results_h5(
    h5_path: str,
) -> List[Tuple[FluorescenceDataset, int]]:
    """Load VSPI results from an HDF5 file saved by :func:`save_vspi_results`.

    Parameters
    ----------
    h5_path : str
        Path to the ``.h5`` file.

    Returns
    -------
    List[Tuple[FluorescenceDataset, int]]
        List of ``(FluorescenceDataset, iteration_number)`` tuples, one per
        saved frame, in the same format as returned by
        ``VSPIFluorescenceEnhancingAlgorithm.enhance``.
    """
    with h5py.File(h5_path, "r") as f:
        epochs = f["epochs"][:]
        element_names = [key for key in f.keys() if key != "epochs"]
        arrays = {name: f[name][:] for name in element_names}

    vspi_results = []
    for i, epoch in enumerate(epochs):
        element_maps = [
            ElementMap(name=name, counts_per_second=arrays[name][i])
            for name in element_names
        ]
        vspi_results.append((FluorescenceDataset(element_maps=element_maps), int(epoch)))
    return vspi_results
