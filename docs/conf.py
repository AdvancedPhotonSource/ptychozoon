# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the src/ directory so autodoc can import the package
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "ptychozoon"
copyright = "2024, Hanna Ruth"
author = "Hanna Ruth"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

# Napoleon settings — accept both NumPy and Google style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Mock heavy or platform-specific imports so docs can build without a GPU
autodoc_mock_imports = [
    "cupy",
    "cupyx",
    "chronos",
    "mda_xdrlib",
    "h5py",
    "tifffile",
    "PyQt5",
    "matplotlib",
    "tqdm",
]

# Cross-reference links to other projects' docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}
