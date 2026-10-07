# Vendored pybind11_mkdoc

Source: https://github.com/pybind/pybind11_mkdoc

Revision: `4b43e39dd8169d5df96b8464b1a3df2dadafd76e` (2026-09-02), version 3.0.0.

The upstream `pybind11_mkdoc/` runtime package is copied here, with the upstream
MIT license in `LICENSE`. SlangPy invokes it through `tools/generate_pydoc.py`;
an installed `pybind11_mkdoc` package is not used. Python Clang bindings and a
compatible libclang installation are still required.

## Updating

Copy the runtime package and license from a reviewed upstream commit and update
this revision. Vendored files are excluded from repository formatting and license
rewriting to keep upstream changes visible.
