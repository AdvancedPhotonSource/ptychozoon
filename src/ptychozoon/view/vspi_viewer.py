# Copyright © 2026 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com/AdvancedPhotonSource/ptychozoon/blob/main/LICENSE.TXT
"""PyQt5-based viewer for VSPI fluorescence enhancement results."""

from __future__ import annotations

import os
from typing import List, Tuple

# Must be set before Qt is imported to avoid GLX errors in remote/headless environments
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")

try:
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    from PyQt5.QtCore import Qt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "PyQt5 and matplotlib are required for the viewer. "
        "Install with: pip install 'ptychozoon[qt]'"
    ) from e

import sys

from ptychozoon.data_structures import FluorescenceDataset


class VSPIResultsViewer(QMainWindow):
    """Window for browsing VSPI enhancement results across iterations and elements."""

    def __init__(self, vspi_results: List[Tuple[FluorescenceDataset, int]]):
        super().__init__()
        self._results = vspi_results
        self._element_names = [em.name for em in vspi_results[0][0].element_maps]

        self.setWindowTitle("VSPI Results Viewer")

        # --- controls ---
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(len(vspi_results) - 1)
        self._slider.setValue(0)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.setTickInterval(1)

        self._slider_label = QLabel("0")
        self._slider_label.setMinimumWidth(30)

        self._combo = QComboBox()
        self._combo.addItems(self._element_names)

        self._keep_clim_checkbox = QCheckBox("Keep colorscale constant")

        self._clim_min = QDoubleSpinBox()
        self._clim_min.setDecimals(4)
        self._clim_min.setRange(-1e12, 1e12)
        self._clim_min.setPrefix("min: ")

        self._clim_max = QDoubleSpinBox()
        self._clim_max.setDecimals(4)
        self._clim_max.setRange(-1e12, 1e12)
        self._clim_max.setPrefix("max: ")

        self._autoset_btn = QPushButton("Autoset color limit")

        # --- matplotlib canvas: create image and colorbar once ---
        first_em = self._results[0][0].element_maps[0]
        self._fig = Figure(figsize=(8, 6), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._im = self._ax.imshow(first_em.counts_per_second, cmap="bone")
        self._colorbar = self._fig.colorbar(self._im, ax=self._ax)
        self._canvas = FigureCanvasQTAgg(self._fig)

        # --- layout ---
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Iteration:"))
        controls.addWidget(self._slider)
        controls.addWidget(self._slider_label)
        controls.addSpacing(20)
        controls.addWidget(QLabel("Element:"))
        controls.addWidget(self._combo)

        clim_controls = QHBoxLayout()
        clim_controls.addWidget(self._keep_clim_checkbox)
        clim_controls.addSpacing(20)
        clim_controls.addWidget(self._clim_min)
        clim_controls.addWidget(self._clim_max)
        clim_controls.addSpacing(10)
        clim_controls.addWidget(self._autoset_btn)
        clim_controls.addStretch()

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addLayout(clim_controls)
        root.addWidget(self._canvas)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        # --- signals ---
        self._slider.valueChanged.connect(self._update)
        self._combo.currentTextChanged.connect(self._update)
        self._keep_clim_checkbox.stateChanged.connect(self._update)
        self._clim_min.valueChanged.connect(self._apply_manual_clim)
        self._clim_max.valueChanged.connect(self._apply_manual_clim)
        self._autoset_btn.clicked.connect(self._autoset_clim)

        self._update()

    def _update(self):
        """Refresh the displayed image based on the current slider and combo box values."""
        idx = self._slider.value()
        fluorescence_dataset, iteration_num = self._results[idx]
        self._slider_label.setText(str(iteration_num))

        element_name = self._combo.currentText()
        em = next(e for e in fluorescence_dataset.element_maps if e.name == element_name)

        data = em.counts_per_second
        self._im.set_data(data)

        if not self._keep_clim_checkbox.isChecked():
            vmin, vmax = float(data.min()), float(data.max())
            self._set_clim_spinboxes(vmin, vmax)
            self._im.set_clim(vmin=vmin, vmax=vmax)

        self._ax.set_title(f"{em.name} — iteration {iteration_num}")
        self._canvas.draw_idle()

    def _set_clim_spinboxes(self, vmin: float, vmax: float):
        """Update spinbox values without triggering _apply_manual_clim."""
        self._clim_min.blockSignals(True)
        self._clim_max.blockSignals(True)
        self._clim_min.setValue(vmin)
        self._clim_max.setValue(vmax)
        self._clim_min.blockSignals(False)
        self._clim_max.blockSignals(False)

    def _apply_manual_clim(self):
        """Apply the color limits currently shown in the min/max spin boxes."""
        self._im.set_clim(vmin=self._clim_min.value(), vmax=self._clim_max.value())
        self._canvas.draw_idle()

    def _autoset_clim(self):
        """Set color limits to the full data range of the currently displayed image."""
        data = self._im.get_array()
        vmin, vmax = float(data.min()), float(data.max())
        self._set_clim_spinboxes(vmin, vmax)
        self._im.set_clim(vmin=vmin, vmax=vmax)
        self._canvas.draw_idle()


def show_vspi_results(
    vspi_results: List[Tuple[FluorescenceDataset, int]],
    block: bool = True,
) -> VSPIResultsViewer:
    """Launch the VSPI results viewer window.

    Parameters
    ----------
    vspi_results:
        List of ``(FluorescenceDataset, iteration_number)`` tuples as returned
        by iterating over ``VSPIFluorescenceEnhancingAlgorithm.enhance``.
    block:
        If ``True`` (default), run the Qt event loop and block until the window
        is closed — suitable for scripts. Set to ``False`` when calling from a
        Jupyter notebook that already has ``%gui qt`` active.

    Returns
    -------
    VSPIResultsViewer
        The viewer window (keep a reference to prevent garbage collection when
        ``block=False``).
    """
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = VSPIResultsViewer(vspi_results)
    window.show()
    if block:
        app.exec_()
    return window
