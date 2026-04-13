User Guide
==========

Installation
------------

Install the package and its core dependencies with pip:

.. code-block:: bash

   pip install ptychozoon

To use the interactive :mod:`~ptychozoon.viewer` you also need PyQt5:

.. code-block:: bash

   pip install "ptychozoon[qt]"

For development (includes pytest):

.. code-block:: bash

   pip install "ptychozoon[dev]"

Quick Start
-----------

The typical workflow consists of three steps:

1. **Load data** — read a ptychography reconstruction and a fluorescence dataset.
2. **Run VSPI** — call :meth:`~ptychozoon.vspi_enhance.VSPIFluorescenceEnhancingAlgorithm.enhance`
   to deconvolve the fluorescence maps.
3. **Inspect / save results** — use the viewer or save to TIFF/HDF5.

.. code-block:: python

   import numpy as np
   from ptychozoon.vspi_enhance import (
       VSPIFluorescenceEnhancingAlgorithm,
       ElementMap,
       FluorescenceDataset,
       Product,
   )
   from ptychozoon.settings import DeconvolutionEnhancementSettings
   from ptychozoon.save import save_vspi_results, SaveFileExtensions

   # --- Build the ptychography product ---
   product = Product(
       probe_positions=np.load("positions.npy"),   # (N, 2) metres, [y, x]
       probe=np.load("probe.npy"),                 # (n_opr, modes, H, W) complex
       object_array=np.load("object.npy"),         # (H, W) complex
       pixel_size_m=(10e-9, 10e-9),                # 10 nm pixels
       object_center_m=(0.0, 0.0),
   )

   # --- Build the fluorescence dataset ---
   fe_map = ElementMap(name="Fe", counts_per_second=np.load("fe_map.npy"))
   dataset = FluorescenceDataset(element_maps=[fe_map])

   # --- Configure and run VSPI ---
   settings = DeconvolutionEnhancementSettings()
   settings.lsmr.max_iter = 20
   settings.lsmr.checkpoint_interval = 5   # yield every 5 iterations

   algorithm = VSPIFluorescenceEnhancingAlgorithm()
   vspi_results = list(algorithm.enhance(dataset, product, settings=settings))

   # --- Save results ---
   # Save every other checkpoint to an HDF5 file.
   # The file will contain one dataset per element plus an "epochs" dataset
   # recording the iteration number for each saved frame.
   save_vspi_results(
       folder="results/",
       name="scan_85",
       vspi_results=vspi_results,
       filetype=SaveFileExtensions.H5,
       save_every_n_frames=2,
   )

Viewing Results Interactively
------------------------------

Use :func:`~ptychozoon.viewer.show_vspi_results` to open a PyQt5 window that
lets you scrub through iterations and switch between element maps:

.. code-block:: python

   from ptychozoon.viewer import show_vspi_results

   show_vspi_results(vspi_results)   # blocks until the window is closed

In a Jupyter notebook with ``%gui qt`` active, pass ``block=False``:

.. code-block:: python

   %gui qt
   viewer = show_vspi_results(vspi_results, block=False)

Opening a Saved HDF5 File in the Viewer
-----------------------------------------

Results saved to HDF5 can be reloaded and viewed directly from the command
line using the ``view-vspi`` entry point (installed with the package):

.. code-block:: bash

   view-vspi results/scan_85_all_frames.h5

You can also load the file programmatically and pass the results to the viewer:

.. code-block:: python

   from ptychozoon.save import load_vspi_results_h5
   from ptychozoon.viewer import show_vspi_results

   vspi_results = load_vspi_results_h5("results/scan_85_all_frames.h5")
   show_vspi_results(vspi_results)

