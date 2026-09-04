# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Configuration file for the Sphinx documentation builder.

import re
from pathlib import Path

from pygments.lexers.graphics import HLSLShaderLexer
from pygments.lexers.python import PythonLexer
from sphinx.highlighting import lexers


def get_release() -> str:
    """Read the package version without importing the native extension."""
    header = (Path(__file__).parent.parent / "src" / "sgl" / "sgl.h").read_text(encoding="utf-8")
    values = {}
    for component in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"^#define SGL_VERSION_{component} (\d+)$", header, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Could not determine SGL_VERSION_{component} from src/sgl/sgl.h")
        values[component] = match.group(1)
    return ".".join(values[component] for component in ("MAJOR", "MINOR", "PATCH"))


project = "SlangPy"
release = get_release()
copyright = "2025-2026, NVIDIA"
author = "Simon Kallweit, Chris Cummings, Benedikt Bitterli, Sai Bangaru, Yong He"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_llm.txt",
    "sphinx_copybutton",
    "nbsphinx",
]

source_suffix = ".rst"
master_doc = "index"
language = "en"
nitpicky = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "CMakeLists.txt", "generated/api.rst"]
suppress_warnings = ["nbsphinx.localfile"]

# Agent-facing Markdown and llms.txt output. Do not enable ``sphinx_llm.docref``:
# that optional extension generates summaries through networked model calls and
# can rewrite source files.
llms_txt_description = (
    "SlangPy is a Python interface for GPU programming with Slang, including "
    "automatic data marshalling, differentiation, and low-level graphics APIs."
)
llms_txt_build_parallel = False
llms_txt_suffix_mode = "replace"
llms_txt_full_build = False

lexers["slang"] = HLSLShaderLexer()
lexers["ipython3"] = PythonLexer()

# html configuration
html_theme = "furo"
html_title = "SlangPy"
html_static_path = ["_static"]
html_extra_path = ["generated/site"]
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
