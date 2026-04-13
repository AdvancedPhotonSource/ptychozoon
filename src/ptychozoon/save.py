"""Functions for saving VSPI fluorescence enhancement results."""

from __future__ import annotations

import os
from typing import List, Tuple

import h5py
import numpy as np
import tifffile

from ptychozoon.enhance import FluorescenceDataset
from ptychozoon.settings import SaveFileExtensions


def save_vspi_results(
    folder: str,
    name: str,
    vspi_results: List[Tuple[FluorescenceDataset, int]],
    filetype: SaveFileExtensions,
) -> None:
    """Save VSPI results to disk.

    Each element's 2D maps across all checkpoint iterations are stacked into a
    3D array of shape (n_frames, height, width) before saving.

    Args:
        folder: Parent output directory.
        name: Name suffix for this result set.
        vspi_results: List of ``(FluorescenceDataset, iteration_number)`` tuples
            as returned by ``VSPIFluorescenceEnhancingAlgorithm.enhance``.
        filetype: Output format — ``SaveFileExtensions.TIFF`` or
            ``SaveFileExtensions.H5``.
    """
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
        _save_h5(folder, name, element_arrays)
    else:
        raise ValueError(f"Unsupported filetype: {filetype!r}")


def _save_tiff(folder: str, name: str, element_arrays: dict[str, np.ndarray]) -> None:
    if not os.path.exists(folder):
        os.mkdir(os.path.dirname(folder))
    for element_name, array_3d in element_arrays.items():
        tiff_path = os.path.join(folder, name + "_all_frames_" + element_name + SaveFileExtensions.TIFF)
        tifffile.imwrite(tiff_path, array_3d)
    print(f"Element arrays saved to {tiff_path}")


def _save_h5(folder: str, name: str, element_arrays: dict[str, np.ndarray]) -> None:
    if not os.path.exists(folder):
        os.mkdir(os.path.dirname(folder))
    h5_path = os.path.join(folder, name + "_all_frames" + SaveFileExtensions.H5)
    with h5py.File(h5_path, "w") as f:
        for element_name, array_3d in element_arrays.items():
            f.create_dataset(element_name, data=array_3d)
    print(f"Element arrays saved to {h5_path}")
