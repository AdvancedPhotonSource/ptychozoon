# Ptychozoon

Ptychozoon: a GPU-accelerated python library for ptychography-enhanced x-ray fluorescence.

## Installation

Ptychozoon uses [CuPy](https://cupy.dev/) for GPU-accelerated computation. CuPy requires CUDA runtime libraries that are not bundled with the pip wheel, so it must be installed via conda-forge first:

```bash
conda install -c conda-forge cupy
```

Then install ptychozoon. It is recommended to install with qt dependencies, which will enable the use of GUIs for viewing the results of the analysis:

```bash
pip install ptychozoon[qt]
```

To install with no optional dependencies:

```bash
pip install ptychozoon
```

To install the package for development, clone the git repository and create an editable install:

```bash
pip install -e ".[dev,qt]"
```

## Usage

```python
import ptychozoon
```
