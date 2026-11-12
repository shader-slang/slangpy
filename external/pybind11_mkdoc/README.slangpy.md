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

`--include-docs <glob>` opts into extracting declarations from included files,
for example `--include-docs '*.slang'` for shared C++/Slang definitions. Patterns
are case-sensitive, match absolute paths with forward slashes, and may be repeated.
Files are parsed through the input headers, preserving their macros and namespaces.
Repeated declarations from the same source location are emitted once; distinct
overloads and namespace contexts are retained. Without this option, included-file
documentation remains excluded.

Resolved source paths and file identities are cached across workers for one
extraction run, retaining canonical path matching and hardlink equivalence without
repeated filesystem lookups for shared headers. Cache hits avoid taking a lock;
misses remain synchronized across workers.
With `--include-docs`,
the include graph limits traversal to selected files and their including ancestors,
preserving enclosing namespaces and classes while pruning unrelated dependencies.

Parsing uses libclang's `PARSE_SKIP_FUNCTION_BODIES`: extraction consumes
declarations and comments, not implementations. Declaration diagnostics are still
reported; errors solely inside skipped function bodies are left to the C++ build.

Extraction uses at most 16 workers by default, limited by the CPU count and number
of input headers. `-j <count>` / `--jobs <count>` overrides the default worker
limit; the Python entry points also accept `jobs`. Generated headers are only
written when their contents change, preserving timestamps on unchanged output
and avoiding unnecessary C++ binding recompilation.

## Updating

Copy the runtime package and license from a reviewed upstream commit, reapply
the local changes, and update this revision. Run the pydoc target, build SlangPy,
and build the complete HTML documentation. Check that forward declarations do
not consume docstring names, definitions retain their descriptions, and genuine
function overloads retain their numbering. Review generated names and numeric
overload references as well as documentation text. Vendored files are excluded from
repository formatting and license rewriting to keep upstream changes visible.
