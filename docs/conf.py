# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Configuration file for the Sphinx documentation builder.

import sys
import shutil
from pathlib import Path
from sphinx.application import Sphinx

project = "SlangPy"
release = "0.18.2"
copyright = "2025, NVIDIA"
author = "Simon Kallweit, Chris Cummings, Benedikt Bitterli, Sai Bangaru, Yong He"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "nbsphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = ".rst"
master_doc = "index"
language = "en"

templates_path = ["_templates"]
exclude_patterns = ["CMakeLists.txt", "generated/*"]

# html configuration
html_theme = "furo"
html_title = "SlangPy"
html_static_path = ["_static"]
html_css_files = ["theme_overrides.css"]
html_theme_options = {
    "light_css_variables": {
        "color-api-background": "#f7f7f7",
    },
    "dark_css_variables": {
        "color-api-background": "#1e1e1e",
    },
}

# nbsphinx configuration
nbsphinx_execute = "never"


def initialize(app: Sphinx):
    # Copy tutorials to src directory.
    print("Copying tutorials to src directory...")
    CURRENT_DIR = Path(__file__).parent
    shutil.copytree(
        src=CURRENT_DIR / "../samples/tutorials",
        dst=CURRENT_DIR / "src/tutorials",
        dirs_exist_ok=True,
    )

    # Generate API documentation for slangpy module is available.
    try:
        print("Generating API documentation...")
        sys.path.append(str(Path(__file__).parent))
        from generate_api import generate_api

        sys.path.append(str(Path(__file__).parent.parent))
        import slangpy  # type: ignore

        generate_api()
    except ImportError:
        print("slangpy module not available, skipping API documentation generation.")


def setup(app: Sphinx):
    app.connect("builder-inited", initialize)
