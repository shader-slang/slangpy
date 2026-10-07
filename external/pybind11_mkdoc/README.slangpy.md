# Vendored pybind11_mkdoc

Source: https://github.com/pybind/pybind11_mkdoc

Revision: `4b43e39dd8169d5df96b8464b1a3df2dadafd76e` (2026-09-02), version 3.0.0.

The upstream `pybind11_mkdoc/` runtime package is copied here, with the upstream
MIT license in `LICENSE`. SlangPy invokes it through `tools/generate_pydoc.py`;
an installed `pybind11_mkdoc` package is not used. Python Clang bindings and a
compatible libclang installation are still required.

## Local changes

`mkdoc_lib.extract()` ignores non-defining class, struct, class-template, and
enum declarations. They must not consume docstring names or numeric suffixes.
Function declarations and genuine overloads are unchanged. Comments are still
obtained from libclang, including comments it inherits from earlier declarations.
Forward-only opaque types do not get generated entries.

## Updating

Copy the runtime package and license from a reviewed upstream commit, reapply
the local changes, and update this revision. Run the pydoc target, build SlangPy,
and build the complete HTML documentation. Check that forward declarations do
not consume docstring names, definitions retain their descriptions, and genuine
function overloads retain their numbering. Review generated names and numeric
overload references as well as documentation text. Vendored files are excluded from
repository formatting and license rewriting to keep upstream changes visible.
