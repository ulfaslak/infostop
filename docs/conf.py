# Sphinx configuration for the Infostop documentation.
#
# Build locally with `make -C docs html`. Read the Docs builds it with the
# settings in `.readthedocs.yaml` at the repository root.

import os
import sys

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DOCS_DIR, ".."))

# The docs build neither compiles the C++ extension (`cpputils`) nor installs the
# runtime dependencies, so autodoc imports `infostop` against mocks of them.
autodoc_mock_imports = [
    "cpputils",
    "folium",
    "infomap",
    "numpy",
    "scipy",
    "sklearn",
    "tqdm",
]

metadata = {}
with open(os.path.join(DOCS_DIR, "..", "infostop", "metadata.py")) as f:
    exec(f.read(), metadata)

# -- Project information -------------------------------------------------------

project = "Infostop"
author = metadata["__author__"]
copyright = "2019, Ulf Aslak"
version = metadata["__version__"]
release = version

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build"]
autodoc_member_order = "bysource"
napoleon_google_docstring = False

# -- HTML output ---------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = True
