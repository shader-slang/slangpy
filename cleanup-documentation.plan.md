# Replace SlangPy's documentation generator with a reproducible, agent-friendly pipeline

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root. When adopted, save it as `.agents/documentation-system-overhaul.md`.

## Purpose / Big Picture

After this work, SlangPy will have an explicitly defined public Python API, accurate API reference pages covering both Python and native nanobind objects, deterministic documentation builds, and machine-readable output suitable for coding agents.

A contributor will be able to build and validate all documentation with one documented command. Read the Docs will no longer silently publish stale API snapshots or machine-specific values. Maintainers will have a coverage report identifying undocumented public symbols, and agents will be able to request a bounded context package for one symbol or API family before improving its authoritative source documentation.

The existing Sphinx, Furo, Read the Docs, RST, and notebook infrastructure will remain in place. The API inventory and rendering pipeline will be replaced incrementally, with the old reference retained until the replacement has demonstrated sufficient coverage.

## Progress

- [x] (2026-09-04) Reviewed the existing documentation, native doc extraction, stub generation, Sphinx configuration, Read the Docs configuration, and CI integration.
- [x] (2026-09-04) Recorded the initial architecture and migration decisions in this plan.
- [x] (2026-09-04) Completed Phase 0: added explicit runtime/snapshot preparation, reproducibility guards, pinned dependencies, strict documentation entry points, and contributor instructions.
- [ ] Complete Phase 1: define the public API and generate a structured pilot inventory.
- [ ] Complete Phase 2: render the structured inventory as split Sphinx API pages.
- [ ] Complete Phase 3: add documentation CI, coverage measurement, and regression gates.
- [ ] Complete Phase 4: publish Markdown and structured outputs for agents.
- [ ] Complete Phase 5: establish and run an agent-assisted documentation campaign.
- [ ] Complete Phase 6: remove the legacy generator and consolidate documentation maintenance.
- [x] (2026-09-04) Ran the Windows release build, focused documentation tests, runtime and snapshot strict HTML builds, the CMake `doc` target, and pre-commit checks for Phase 0.
- [ ] Record final outcomes and remaining documentation debt.

## Surprises and Discoveries

- Observation: The generated API reference has 22,110 lines and 4,325 Python directives. Its “Miscellaneous” section starts at line 14,460 and occupies approximately 35 percent of the file.
  Evidence: `docs/generated/api.rst`.

- Observation: The generated reference includes 595 alias-only classes and 201 literal `N/A` descriptions.
  Evidence: Counts from `docs/generated/api.rst`.

- Observation: The generated reference captures machine-dependent values, including an absolute `SHADER_PATH` and local Git/build metadata.
  Evidence: `docs/generated/api.rst` includes a `C:\sw\slangpy` path and a branch description containing “local changes.”

- Observation: Important Python-facing classes such as `Tensor`, `Module`, and `Function` appear as aliases without their ordinary Python-defined methods.
  Evidence: `docs/generate_api.py` treats nanobind methods specially but does not render ordinary Python methods correctly.

- Observation: The checked-in generated API file is older than the current generated native documentation header.
  Evidence: At initial review, `docs/generated/api.rst` last changed on 2026-08-11 while `src/slangpy_ext/py_doc.h` changed on 2026-09-04.

- Observation: Read the Docs installs only `docs/requirements.txt`. It cannot import a built SlangPy extension, and `docs/conf.py` silently skips API regeneration after an `ImportError`.
  Evidence: `.readthedocs.yml` and `docs/conf.py`.

- Observation: Nanobind already generates detailed `.pyi` stubs containing structured overloads and docstrings. These files are ignored by Git and unused by the documentation generator.
  Evidence: `src/slangpy_ext/CMakeLists.txt`, `.gitignore`, and generated files under `slangpy/**/*.pyi`.

- Observation: The native documentation header contains 6,163 extracted constants, of which 4,296 are empty. This includes unbound and non-public C++ declarations, so it must not be treated as public Python documentation coverage.
  Evidence: `src/slangpy_ext/py_doc.h`.

- Observation: There are 344 explicit `D_NA(...)` call sites in the nanobind bindings.
  Evidence: `src/slangpy_ext/**/*.cpp` and `src/slangpy_ext/**/*.h`.

- Observation: The developer guide tells contributors to build a `pydoc` target, while CMake defines `slangpy_pydoc`.
  Evidence: `docs/src/developer_guide/compiling.rst` and `src/slangpy_ext/CMakeLists.txt`.

- Observation: The legacy generator renders ordinary Python methods as attributes whose values contain process-specific memory addresses.
  Evidence: Existing `docs/generated/api.rst` entries contain values such as `<function Module.load_from_file at 0x...>`.

- Observation: Native signatures can also contain object-valued defaults with process-specific memory addresses, so filtering only generated `:value:` fields is insufficient.
  Evidence: Consecutive Phase 0 generations initially differed in six signatures, including `Bitmap.resample`, `CommandEncoder.draw_indirect`, and `Profiler.start_capture`.

- Observation: The `pandoc` package previously listed in `docs/requirements.txt` is a Python library and does not provide the Pandoc executable required by nbconvert.
  Evidence: The first strict notebook build failed with `PandocMissing`; replacing it with `pypandoc-binary` made both notebooks render in a clean pip-installed documentation environment.

- Observation: Enabling Sphinx nitpicky mode against the legacy monolithic runtime reference produces hundreds of unresolved external and malformed native type references.
  Evidence: The Phase 0 trial reported 394 warnings, dominated by types such as `enum.Enum`, `collections.abc.Sequence`, and nanobind-specific ndarray spellings. The ordinary strict build now passes with zero warnings.

## Decision Log

- Decision: Retain Sphinx, Furo, Read the Docs, RST, and nbsphinx during this migration.
  Rationale: The existing narrative documentation and publishing infrastructure work. Replacing the site generator would add migration risk without fixing the inaccurate API inventory.
  Date/Author: 2026-09-04, Codex.

- Decision: Treat source docstrings and C++ documentation comments as authoritative. Generated RST, JSON, Markdown, `.pyi`, and `py_doc.h` files must never be edited to improve documentation content.
  Rationale: Documentation must remain adjacent to the implementation and available in IDE help and runtime `help()`, not only on the website.
  Date/Author: 2026-09-04, Codex.

- Decision: Use nanobind-generated `.pyi` files as the authoritative representation of native Python signatures and overloads.
  Rationale: Nanobind exposes structured signature information that is more reliable than parsing rendered runtime docstrings.
  Date/Author: 2026-09-04, Codex.

- Decision: Pilot Griffe as the API inventory model.
  Rationale: Griffe can parse Python source, merge adjacent `.pyi` stubs, resolve aliases, serialize the result to JSON, and compare public APIs. If the pilot cannot merge SlangPy's native and Python layers correctly, retain the same JSON schema and implement the missing `.pyi` merge with Python's standard `ast` module.
  Date/Author: 2026-09-04, Codex.

- Decision: Introduce an explicit `docs/public_api.toml` contract.
  Rationale: Reachability through imports is not a stable definition of public support. The current `api_order.json` should seed the initial contract, but unclassified objects must not automatically appear in the published reference.
  Date/Author: 2026-09-04, Codex.

- Decision: Check in a deterministic API JSON snapshot, but generate rendered API pages during the documentation build.
  Rationale: Read the Docs cannot currently compile and import SlangPy. A normalized snapshot allows a source-only documentation build while CI verifies that the snapshot matches a freshly built extension and its stubs.
  Date/Author: 2026-09-04, Codex.

- Decision: Produce one API page per conceptual section or public module rather than a single giant include.
  Rationale: Smaller pages improve navigation, search, review diffs, agent retrieval, and stable linking.
  Date/Author: 2026-09-04, Codex.

- Decision: Do not make networked language-model calls during documentation builds.
  Rationale: Documentation builds must be deterministic, auditable, and usable without credentials. Agents may author documentation in separate reviewed workflows.
  Date/Author: 2026-09-04, Codex.

- Decision: Ratchet documentation coverage from an initial baseline instead of requiring immediate total coverage.
  Rationale: A non-regression gate can be adopted immediately while high-value APIs are improved incrementally.
  Date/Author: 2026-09-04, Codex.

- Decision: Phase 0 provides explicit `runtime` and `snapshot` preparation modes. Developer and CI builds use `runtime`; Read the Docs uses `snapshot`.
  Rationale: A source-only hosted build is legitimate, but it must not be selected implicitly after a failed import.
  Date/Author: 2026-09-04, Codex.

- Decision: Pin Sphinx 7.4.7 during Phase 0 rather than the latest major release.
  Rationale: Sphinx 7.4.7 supports the project's Python 3.9 minimum while providing strict warning and nitpicky-reference modes. A later Python baseline can update Sphinx separately.
  Date/Author: 2026-09-04, Codex.

- Decision: Enforce `-W --keep-going` in Phase 0 and defer nitpicky-reference mode until the structured API renderer can normalize native annotations and provide a small reviewed ignore list.
  Rationale: Strict mode now catches all ordinary Sphinx warnings. Enabling nitpicky mode on the legacy snapshot adds hundreds of unresolved targets that cannot be addressed by a small maintainable exception list; the final pipeline still retains nitpicky mode as a Phase 2/3 requirement.
  Date/Author: 2026-09-04, Codex.

- Decision: Use `pypandoc-binary` for the documentation environment and add its bundled executable directory to the Sphinx subprocess when no system Pandoc is available.
  Rationale: This makes the documented pip installation self-contained across supported local and hosted platforms while continuing to prefer a system Pandoc when present.
  Date/Author: 2026-09-04, Codex.

## Outcomes and Retrospective

Phase 0 is complete. `tools/docs.py` is now the explicit preparation, validation, and strict-build entry point; `tools/ci.py docs`, the CMake `doc` target, and Read the Docs all use it with an explicit runtime or snapshot mode. Runtime import failures are fatal, while hosted source-only builds deliberately validate the checked-in snapshot.

The legacy snapshot no longer publishes local Git descriptions, build/package paths, callable representations, or object addresses. Consecutive runtime generations produced the same SHA-256 hash (`472CBC5933822DF8C9799174014BFD918EE4A3902713D61249C397A8190C49BF`) during validation. The generator also normalizes unstable object-valued defaults and trailing whitespace.

Validation completed with a full Windows release build, 11 focused tests, warning-free runtime and snapshot Sphinx HTML builds, a warning-free CMake `doc` target build, and a passing `pre-commit run --all-files`. The strict build also exposed and fixed malformed changelog markup, missing cross-reference labels, notebook tooling, and lexer registration.

Remaining debt is intentionally carried into later phases: 792 reachable objects still fall into the legacy `Miscellaneous` section, 80 Python properties lack native signature metadata, nitpicky-reference mode cannot yet be enabled with a small ignore list, and the generated reference remains a monolithic runtime-introspection artifact rather than the planned public structured inventory.

Update this section at the end of each phase with the observable improvements, remaining gaps, and any changes to the following milestones.

## Context and Orientation

SlangPy combines ordinary Python code under `slangpy/` with a native nanobind extension implemented under `src/slangpy_ext/`. Much of the native extension wraps C++ declarations under `src/sgl/`.

Native documentation begins as Doxygen-style comments in C++ headers. The `slangpy_pydoc` CMake target invokes `pybind11_mkdoc` and writes `src/slangpy_ext/py_doc.h`. Binding code passes strings from that header through the `D(...)` macro. Bindings without documentation often use `D_NA(...)`, which expands to the literal string `N/A`.

The native build also invokes nanobind's stub generator. It writes `.pyi` files next to the Python package. A `.pyi` file is a Python interface file containing classes, functions, overloads, types, and retained docstrings without executable implementations. These stubs already describe native signatures more reliably than `inspect.signature()`.

The current API renderer is `docs/generate_api.py`. It imports `slangpy`, recursively visits reachable modules and objects, parses docstrings, sorts selected names using `docs/api_order.json`, and writes `docs/generated/api.rst`. `docs/src/api_reference.rst` includes that large file.

`docs/conf.py` runs the generator from a Sphinx `builder-inited` callback and copies notebooks and supporting files from `samples/tutorials` into `docs/src/tutorials`. It catches `ImportError` when SlangPy is unavailable. `.readthedocs.yml` installs documentation dependencies but does not build SlangPy, so the hosted build normally uses the checked-in generated API file.

The replacement will add `tools/docs.py` as the explicit documentation pipeline. It will generate and validate a normalized API inventory, render temporary Sphinx source pages, calculate coverage, create agent context packages, and invoke Sphinx when requested. Tests for this tool will live under `slangpy/tests/docs/`, following the repository rule that new Python APIs have Python tests.

The term “public API contract” in this plan means the reviewed set of Python names that SlangPy intends users or extension authors to rely on. It is not every object reachable from `import slangpy`.

The term “API snapshot” means the deterministic `docs/api/api.json` file derived from Python source, generated stubs, and the public API contract. It is generated but checked in so Read the Docs can build without the native extension.

## Plan of Work

### Phase 0: Stabilize the existing process

This phase adds immediate safeguards without yet replacing the legacy reference. At its conclusion, documentation failures will no longer be silently ignored, machine-specific values will be rejected, dependency installation and contributor commands will be accurate, and the existing documentation can be built through one entry point.

Create `tools/docs.py` with an initial `build` and `check-generated` command. The initial `build` command may invoke the existing generator and Sphinx, but it must explicitly report whether the SlangPy module is available. In validation mode, failure to import required API inputs must be fatal.

Add a `docs` subcommand to `tools/ci.py`. It must call `tools/docs.py build` with strict Sphinx warnings enabled. Keep the project build as a separate required preceding step so the existing “always build before tests” convention remains visible.

Update `docs/src/developer_guide/compiling.rst` to use the actual `slangpy_pydoc` target. Document the distinction between regenerating C++ comments, building stubs, generating the API snapshot, rendering pages, and building HTML.

Remove the broad `try/except ImportError` behavior from validation builds. A source-only Read the Docs build may deliberately use the committed snapshot, but that mode must be explicit rather than inferred from an import failure.

Add a validation rule rejecting absolute Windows or POSIX paths, local Git descriptions, build directories, and unapproved runtime values in generated documentation. Only stable constants such as enum values and the package version may be included.

Fix the duplicate Sphinx configuration assignments, enable `sphinx.ext.intersphinx` if its mappings remain, derive `release` from the same source as packaging, and stop copying tutorials from a Sphinx event hook. Replace the copy with an explicit preparation step under `tools/docs.py`.

Pin documentation build dependencies to reviewed versions in `docs/requirements.txt`. Add `pybind11_mkdoc` there or in a clearly referenced native-documentation requirements file so the documented command is self-contained.

Acceptance for this phase is a strict local documentation build that contains no absolute source paths or local branch descriptions and fails with a clear error if expected generated input is missing.

### Phase 1: Define the public API and build a structured inventory pilot

Create `docs/public_api.toml`. Start from the named sections and patterns in `docs/api_order.json`, but do not carry the automatically generated “Miscellaneous” contents into the public contract. Give each section a stable identifier, title, audience, and ordered set of names or narrowly scoped patterns.

The initial contract should distinguish the primary user API from the extension-author API. Internal implementation types may remain importable without being published as supported public interfaces.

Extend `tools/docs.py` with these commands:

    python tools/docs.py inventory
    python tools/docs.py inventory --check
    python tools/docs.py report-unclassified

`inventory` will read ordinary Python source with Griffe, merge generated `.pyi` information, apply `docs/public_api.toml`, normalize aliases, and write `docs/api/api.json`. It must not import SlangPy merely to discover values. If limited native inspection is needed during the pilot, confine it to a separate extraction step and normalize its output before serialization.

The JSON document must contain a schema version, package version, sections, and symbols. Each symbol must contain its published name, canonical name, kind, signatures or overloads, aliases, documentation text, documentation status, section, audience, and repository-relative source references when available. It must not contain absolute paths, timestamps, build types, memory addresses, object representations, or local Git descriptions.

Use `Tensor`, `Module`, `Function`, and `Device` as the mandatory pilot. The resulting inventory must contain their user-facing constructors, properties, and methods rather than only alias records. Add tests using a small synthetic package containing a `.py` implementation, adjacent `.pyi` file, overloads, aliases, and `__all__`. Add an integration test against the built SlangPy package.

Run inventory generation twice and require byte-for-byte identical JSON. Run it from two different workspace paths when practical and require identical output.

`report-unclassified` must write or print reachable public-looking names that are not included in the public contract. This report is for review; unclassified names must not be published automatically.

Acceptance for this phase is deterministic JSON with complete method inventories for all four pilot classes and no environment-dependent data.

### Phase 2: Render split API pages and cut over safely

Extend `tools/docs.py` with a `render` command. It must consume only `docs/api/api.json` and `docs/public_api.toml`, allowing Read the Docs to render the reference without compiling SlangPy.

Generate temporary RST or MyST files under `docs/generated/api/`, with one page per conceptual section or public module. Keep this directory ignored by Git. Generated pages must use stable Sphinx Python-domain anchors based on the published canonical names.

Create a temporary `docs/src/api_reference_v2.rst` landing page and include the new generated pages in a hidden toctree. Preserve `docs/src/api_reference.rst` and the legacy `docs/generated/api.rst` during comparison.

Add rendering tests that verify overloads, parameters, properties, enums, aliases, cross-references, and empty documentation states. Missing documentation should display a neutral marker such as “Documentation pending” only when maintainers choose to expose it; the literal `N/A` must not be emitted.

Compare the legacy and structured inventories. Every name in the explicit public contract must appear in the new reference. Extra legacy names should be recorded in the unclassified report rather than automatically copied.

Once the pilot pages are reviewed, make `docs/src/api_reference.rst` the structured landing page. Preserve old useful anchors with aliases or redirects when practical. Remove the legacy include only after the new build passes all acceptance checks.

Acceptance for this phase is a navigable reference where `Tensor`, `Module`, `Function`, and `Device` have useful dedicated entries and the primary API no longer shares one 22,000-line page with internal implementation objects.

### Phase 3: Add CI, coverage, and regression gates

Add tests under `slangpy/tests/docs/` for the inventory schema, normalization, public API filtering, deterministic rendering, and coverage classification.

Implement a coverage command:

    python tools/docs.py coverage
    python tools/docs.py coverage --check

Classify each public symbol as `missing`, `placeholder`, `summary`, or `complete`. Treat an empty docstring and `N/A` as placeholders. The initial mechanical definition of `complete` should require a non-placeholder summary and documentation for parameters and returns when those sections are applicable. Examples should be tracked as a separate field because not every property or enum value needs one.

Write the reviewed starting counts to `docs/api/coverage-baseline.json`. The `--check` command must fail if total public coverage decreases, a completed symbol regresses, a new public symbol is undocumented, or a placeholder is added. Do not initially require all historical symbols to be complete.

Add `.github/workflows/docs.yml`. On an Ubuntu GitHub-hosted runner, install the same build prerequisites used by the existing Linux CI, configure a release build, build SlangPy and its stubs, install pinned documentation dependencies, regenerate the API snapshot, fail on a snapshot diff, run documentation tests, and build Sphinx with warnings treated as errors.

Configure the workflow to run when documentation, Python package source, native public headers, binding code, stub patterns, CMake documentation logic, or documentation tooling changes. Add a scheduled link-check job rather than making external network health block every pull request.

Examples in narrative documentation must be either executable tests, included from tested source files, or explicitly marked as illustrative. GPU-dependent notebooks may remain disabled in the ordinary Sphinx build, but they should be exercised through the existing sample tests on a supported GPU CI job when feasible.

Acceptance for this phase is demonstrated by tests that deliberately add an undocumented public symbol, an absolute path, a stale generated snapshot, and a broken internal reference and observe the appropriate CI failures.

### Phase 4: Publish outputs designed for agents

Add pinned `sphinx-llm` support with networked summary generation disabled. Generate Markdown for every Sphinx page and a curated `llms.txt`. Generate `llms-full.txt` only if its size remains useful after the internal API has been removed; otherwise disable it and rely on per-page Markdown.

Ensure the published API JSON is copied to a stable site path such as `/api/api.json`. Publish a normalized coverage report at `/api/coverage.json`. These files must describe their schema version.

Add these commands to `tools/docs.py`:

    python tools/docs.py context slangpy.Module
    python tools/docs.py tasks
    python tools/docs.py tasks --section core

`context` must generate a Markdown package for one symbol containing its signatures, existing docs, canonical source location, binding location when known, related public symbols, relevant tests, examples, and explicit missing documentation fields. It must use repository-relative links and must not modify source files.

`tasks` must generate an ordered list of missing or incomplete public documentation. Sort first by configured API priority, then by documentation status. The output should be available in JSON for automation and Markdown for humans.

The generated Markdown must preserve code blocks, signatures, cross-references, and source links without HTML navigation noise. Add tests for representative narrative, native API, Python API, and notebook-derived pages.

Acceptance for this phase is that an agent can fetch or generate the `slangpy.Module` context without crawling the entire repository and can find its signatures, implementation, tests, examples, and missing documentation fields in one bounded artifact.

### Phase 5: Run an agent-assisted documentation campaign

Begin with the highest-value user workflows rather than filling every blank mechanically. The first campaign should cover `create_device`, `Module`, `Function`, `Tensor`, `DiffPair`, buffer creation and transfer, texture creation and transfer, and shader/module loading.

Each agent task should cover one class or one tightly related function family. The agent must start from `tools/docs.py context`, inspect the implementation and tests, and edit only authoritative sources.

For native APIs, documentation belongs on the public C++ declaration under `src/sgl/` when such a declaration exists. Wrapper-only behavior may be documented adjacent to its binding under `src/slangpy_ext/`. Regenerate `src/slangpy_ext/py_doc.h` rather than editing it.

For Python APIs, documentation belongs in the implementation under `slangpy/`. Agents must not edit generated `.pyi`, generated API pages, `docs/api/api.json`, or coverage output to improve a score.

Every important entry point should gain a concise summary, parameter semantics, return behavior, relevant exceptions, ownership or lifetime constraints, synchronization behavior when relevant, backend restrictions, and a tested example. Agents must not infer unsupported guarantees. Claims should be confirmed by implementation, tests, or existing design documentation.

Each documentation pull request should build SlangPy first, run relevant behavioral tests, regenerate the native documentation header when applicable, regenerate the API snapshot, run documentation coverage and strict Sphinx builds, and run pre-commit.

Coverage targets should be raised in stages. First eliminate placeholders from the primary workflow APIs. Then require complete summaries and parameter documentation for the entire primary user API. Finally address the extension-author and low-level graphics APIs. Do not require agents to document internal objects merely because they are importable.

Human review remains required for semantic accuracy, public-support commitments, backend guarantees, and examples. Automated checks verify structure and regression, not truth.

Acceptance for this phase is that the primary workflow classes contain no placeholders, their examples run in tests, and the coverage dashboard shows a sustained increase without introducing unsupported public guarantees.

### Phase 6: Remove legacy machinery and consolidate maintenance

After the structured reference has been the default for at least one release cycle, remove `docs/generate_api.py`, `docs/api_order.json`, and `docs/generated/api.rst`. Remove the corresponding pre-commit exclusions and the Sphinx builder callback that generated or copied source files.

Retain `pybind11_mkdoc` only for transferring authoritative native C++ comments into runtime nanobind docstrings. Reassess it separately if it becomes unmaintained, but do not replace it with a C++ documentation renderer merely for novelty.

Consolidate contributor instructions into `docs/README.md` and link to it from the developer guide. Document the full source-to-output pipeline, how to add a public symbol, how to regenerate snapshots, how to interpret coverage, and how to prepare an agent task.

Permit MyST Markdown for new narrative pages if maintainers prefer Markdown authoring, but do not bulk-convert functioning RST pages. Reconsider a site-generator migration only if Sphinx itself becomes a measured maintenance problem after the API pipeline is fixed.

Acceptance for this phase is that no production documentation path imports `docs/generate_api.py`, no checked-in giant RST API snapshot remains, and a new contributor can reproduce the hosted reference using only the documented commands.

## Concrete Steps

All commands begin in the repository root:

    cd C:\projects\slangpy

Install the pinned documentation dependencies:

    python -m pip install -r docs/requirements.txt

When native C++ documentation comments change, regenerate their header using the actual target:

    cmake --preset windows-msvc
    cmake --build --preset windows-msvc-release --target slangpy_pydoc

Always build the project before running documentation tests:

    cmake --build --preset windows-msvc-release

Generate and validate the structured snapshot:

    python tools/docs.py inventory
    python tools/docs.py inventory --check
    python tools/docs.py coverage --check

Render and build the documentation:

    python tools/docs.py render --clean
    python tools/ci.py docs

Run documentation-specific Python tests after the build:

    pytest slangpy/tests/docs -v

Run relevant project and sample tests when documentation adds or changes executable examples:

    pytest slangpy/tests -v
    pytest samples/tests -vra

Generate a context package for an agent:

    python tools/docs.py context slangpy.Module
    python tools/docs.py tasks --section core

Run repository formatting and checks after completing a phase:

    pre-commit run --all-files

If pre-commit modifies files, inspect the changes and run it again until it exits successfully.

A successful inventory check should report results similar to:

    Loaded Python source and generated stubs.
    Public symbols: <count>
    Unclassified symbols: <count>
    Placeholders: <count>
    API snapshot is deterministic and current.
    No forbidden environment-dependent values found.

A successful documentation build should report:

    API pages rendered from docs/api/api.json.
    Sphinx build completed with 0 warnings.
    Markdown and API JSON outputs written to the HTML output directory.

Exact counts should be recorded after Phase 1 establishes the reviewed public contract.

## Validation and Acceptance

The full migration is accepted when all of the following behavior is observable.

A clean checkout can build SlangPy, generate stubs, validate the checked-in API snapshot, and build the documentation using the commands above. Running the same generation twice produces no Git diff.

Read the Docs can build the reference from the committed normalized snapshot without importing a native extension. A developer or CI build with the extension available verifies that the snapshot is current.

The published reference contains all names in `docs/public_api.toml` and no unclassified internal names. `Tensor`, `Module`, `Function`, and `Device` show their constructors, methods, properties, overloads, and documentation under stable pages.

Generated files contain no workspace paths, build directories, local branch descriptions, memory addresses, timestamps, or unstable runtime representations.

Strict Sphinx validation completes with no warnings. Invalid internal cross-references and stale snapshots fail CI. Scheduled link checking reports external failures separately.

Documentation coverage cannot decrease silently. Adding a new public symbol without documentation fails coverage validation. Existing historical debt remains visible and can only stay constant or improve.

Each published page has a Markdown equivalent. `llms.txt` links to the reviewed documentation structure. The public API and coverage JSON documents are downloadable and versioned by schema.

An agent can generate a context package for a public symbol and receive bounded implementation, signature, documentation, test, example, and missing-field information without parsing the entire generated reference.

## Idempotence and Recovery

Inventory generation and rendering must be idempotent. Commands should write files only when their contents change.

Keep the structured API reference parallel to the legacy reference through Phase 2. If the new renderer fails to represent an important API, fix the inventory or renderer while the legacy page remains available. Do not remove the legacy generator until the structured reference has passed CI and review.

`docs/api/api.json` is generated but checked in. If it becomes corrupted, rebuild SlangPy and rerun `python tools/docs.py inventory`; do not repair the JSON manually.

Files under `docs/generated/api/`, Sphinx build directories, agent context output, and task output are disposable. They may be deleted and regenerated. The committed public contract, source docstrings, API snapshot, and coverage baseline are not disposable.

If Griffe cannot correctly combine a particular Python implementation and native stub, capture the failing fixture in `slangpy/tests/docs/`. Add a narrow merge adapter using Python's `ast` module while retaining the documented JSON schema. Do not fall back to crawling the runtime import graph.

If Read the Docs cannot run a preparation command, generated pages may temporarily be checked in, but the snapshot-currentness check must remain. Prefer fixing the build configuration and returning pages to generated status.

## Artifacts and Notes

The intended new source-controlled files are:

    .agents/documentation-system-overhaul.md
    docs/README.md
    docs/public_api.toml
    docs/api/api.json
    docs/api/coverage-baseline.json
    tools/docs.py
    slangpy/tests/docs/test_api_inventory.py
    slangpy/tests/docs/test_api_rendering.py
    slangpy/tests/docs/test_api_coverage.py
    .github/workflows/docs.yml

The intended generated, non-source-controlled files are:

    docs/generated/api/index.rst
    docs/generated/api/<section>.rst
    docs/_build/html/
    docs/_build/agent-context/
    docs/_build/documentation-tasks.json
    docs/_build/documentation-tasks.md

The initial JSON schema should resemble this structure, although field names may be extended while the schema version remains explicit:

    {
      "schema_version": 1,
      "package_version": "0.43.0",
      "sections": [
        {
          "id": "functional-api",
          "title": "Functional API",
          "audience": "user"
        }
      ],
      "symbols": [
        {
          "name": "slangpy.Module.load_from_file",
          "canonical_name": "slangpy.core.module.Module.load_from_file",
          "kind": "staticmethod",
          "section": "functional-api",
          "audience": "user",
          "signatures": [
            "load_from_file(device: Device, path: str, ...) -> Module"
          ],
          "aliases": [],
          "documentation": {
            "summary": "Load a Slang module from a file.",
            "body": "",
            "parameters": {},
            "returns": "",
            "raises": [],
            "status": "summary"
          },
          "source": {
            "path": "slangpy/core/module.py",
            "line": 82
          }
        }
      ]
    }

Line numbers may be omitted when they cannot be generated deterministically. Repository-relative paths are mandatory whenever paths are included.

## Interfaces and Dependencies

`tools/docs.py` should use typed Python functions and expose a conventional `argparse` command-line interface consistent with `tools/ci.py`.

At minimum, it must provide these internal interfaces:

    def load_public_api(path: Path) -> PublicApiConfig
    def build_inventory(config: PublicApiConfig, package_root: Path) -> ApiInventory
    def normalize_inventory(inventory: ApiInventory, repository_root: Path) -> ApiInventory
    def write_inventory(inventory: ApiInventory, output: Path) -> None
    def render_sphinx(inventory: ApiInventory, output_dir: Path) -> None
    def calculate_coverage(inventory: ApiInventory) -> CoverageReport
    def validate_coverage(report: CoverageReport, baseline: Path) -> None
    def build_agent_context(inventory: ApiInventory, symbol: str) -> str

Use dataclasses or typed dictionaries for `PublicApiConfig`, `ApiInventory`, symbols, documentation records, and coverage reports. All function arguments must have type annotations.

Use Griffe for static Python and stub analysis during the pilot. Pin its version in `docs/requirements.txt`. Keep the public JSON schema independent of Griffe classes so the implementation can change without breaking consumers.

Use nanobind's existing CMake stub generation for native signatures. Do not introduce a second native signature parser.

Use Sphinx and its Python domain for rendering and cross-references. Keep Furo as the HTML theme. Enable strict warning and nitpicky-reference modes with a small reviewed ignore list.

Use `sphinx-llm` only for deterministic Markdown conversion and indexes. Keep its generated-summary feature disabled.

Do not introduce Doxygen or clang-doc as the main Python reference renderer. They describe C++ declarations and do not know SlangPy's Python names, aliases, wrapper behavior, or public support boundary.

Document any dependency or interface changes in the Decision Log. Update this ExecPlan after every stopping point so another contributor can continue using this file alone.

## Revision Note

This revision completes Phase 0. It records the final runtime/snapshot build contract, deterministic legacy generation, pinned and self-contained notebook tooling, strict warning policy, validation results, and the specific legacy-reference debt carried into Phase 1.
