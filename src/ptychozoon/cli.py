"""Command-line entry points for ptychozoon."""

from __future__ import annotations

import argparse
import sys


def view_vspi() -> None:
    """Open the VSPI results viewer from an HDF5 file.

    Usage::

        view-vspi path/to/result.h5
    """
    parser = argparse.ArgumentParser(
        prog="view-vspi",
        description="Open the VSPI results viewer from an HDF5 file.",
    )
    parser.add_argument("h5_file", help="Path to the VSPI result .h5 file")
    args = parser.parse_args()

    from ptychozoon.save import load_vspi_results_h5
    from ptychozoon.viewer import show_vspi_results

    vspi_results = load_vspi_results_h5(args.h5_file)
    show_vspi_results(vspi_results, block=True)
    sys.exit(0)
