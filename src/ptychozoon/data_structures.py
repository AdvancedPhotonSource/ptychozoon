# Copyright © 2026 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com/AdvancedPhotonSource/ptychozoon/blob/main/LICENSE.TXT
from typing import Sequence, Optional
import numpy as np


from dataclasses import dataclass


@dataclass(frozen=True)
class PtychographyProduct:
    """Ptychography reconstruction product.

    Attributes:
        probe_positions: ``(N, 2)`` float array of ``[y, x]`` scan coordinates in meters.
        probe: ``(n_opr, modes, height, width)`` complex probe array.
        object_array: ``(height, width)`` complex object array.
        pixel_size_m: ``(pixel_height_m, pixel_width_m)`` pixel sizes in meters.
        object_center_m: ``(center_y_m, center_x_m)`` object center coordinates in meters.
        opr_mode_weights: ``(n_opr, N)`` OPR mode weights; ``None`` if not used.
    """

    probe_positions: np.ndarray  # (N, 2) float array [y, x] in meters
    probe: np.ndarray  # (n_opr, modes, height, width) complex array
    object_array: np.ndarray  # (height, width) complex array
    pixel_size_m: tuple[float, float]  # (pixel_height_m, pixel_width_m)
    object_center_m: tuple[float, float]  # (center_y_m, center_x_m)
    opr_mode_weights: Optional[np.ndarray] = None  # n_opr, N


@dataclass(frozen=True)
class ElementMap:
    """2D spatial map of fluorescence signal for a single element in counts per second."""

    name: str
    counts_per_second: np.ndarray


@dataclass(frozen=True)
class FluorescenceDataset:
    """Collection of element maps"""

    element_maps: Sequence[ElementMap]
