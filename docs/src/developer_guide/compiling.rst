.. _sec-compiling:

Compiling
=========

In order to compile SlangPy from source, the following prerequisites are
required:

* A C++20 compliant compiler (tested with Visual Studio 2022, GCC 11 and Clang 14)

* Xcode >= 16, 16.4 recommended (on macOS)

* Python >= 3.9

* git

Optionally:

* CUDA Toolkit >= 11.8; 12.8 recommended (on Windows/Linux, for cuda
  acceleration)

* `PyTorch <https://pytorch.org/get-started/>`_ >= 2.7.1 (for optional
  integration)

.. tip::

    We strongly recommend using a Python virtual environment (anaconda on
    Windows, venv on Linux/macOS)


Cloning the repository
----------------------

The first step is to clone the repository. This can be done by running the
following command:

.. code-block:: bash

    git clone https://github.com/shader-slang/slangpy.git --recursive


Setup
-----

To make it easy to build ``slangpy`` reliably, an additional setup step is required:

.. code-block:: bash

    # Install Python build prerequisites
    pip install -r requirements-dev.txt

    # On Windows
    setup.bat

    # On Linux and macOS
    ./setup.sh


This will do the following:

* Make sure all git submodules are initialized and up-to-date.

* On the first run, setup a ``.vscode`` directory with initial settings for
  VS Code.

This script can be run anytime to ensure that both git submodules and host tools
are up-to-date.


Windows
-------

To build on Windows, make sure you have a recent version of
`Visual Studio 2022 <https://visualstudio.microsoft.com/vs/>`_
installed.

Open ``x64 Native Tools Command Prompt for VS 2022`` and use the following
commands to build the project:

.. code-block:: bash

    # Configure
    cmake --preset windows-msvc

    # Build "Debug" configuration
    cmake --build --preset windows-msvc-debug

    # Build "Release" configuration
    cmake --build --preset windows-msvc-release


The build artifacts are placed in ``build\windows-msvc\bin\Debug`` or
``build\windows-msvc\bin\Release``.

Alternatively you can use the ``windows-vs2022`` preset to configure the project
as a Visual Studio 2022 solution stored in ``build\windows-vs2022``.

**Tested on:**

* Windows 10 (build 19045)
* Visual Studio 2022 (Version 17.8.0)
* CMake 3.27.7
* Ninja 1.11.1


Linux
-----

To build on Linux, make sure you have the required build tools and dependencies
installed. The following commands can be used to install the required build
tools and dependencies:

.. code-block:: bash

    # Install build tools
    sudo apt install build-essential

    # Install required build dependencies
    sudo apt install libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config


Then use the following commands to build the project:

.. code-block:: bash

    # Configure
    cmake --preset linux-gcc

    # Build "Debug" configuration
    cmake --build --preset linux-gcc-debug

    # Build "Release" configuration
    cmake --build --preset linux-gcc-release


The build artifacts are placed in ``build\linux-gcc\bin\Debug`` or
``build\linux-gcc\bin\Release``.

Alternativaly you can also use the ``linux-clang`` preset to use the Clang
compiler.

**Tested on:**

* Ubuntu 22.04
* GCC 11.4.0
* CMake 3.27.7
* Ninja 1.11.1


macOS
-----

To build on macOS, make sure you have a recent version of XCode installed. You
also need to install the XCode command line tools by running the following
command:

.. code-block:: bash

    xcode-select --install

Some additional command line build tools are also required. An easy way to
install these is to install `brew <https://brew.sh>`_, and then use the
following commands:

.. code-block:: bash

    brew install cmake ninja pkg-config git-lfs
    git lfs install

If ``git-lfs`` wasn't installed before you cloned SlangPy, you will need to use
the following commands to retrieve and check out the files stored in LFS:

.. code-block:: bash

    git submodule foreach --recursive git lfs fetch
    git submodule foreach --recursive git lfs checkout

Then open a new shell and use the following commands to build the project:

.. code-block:: bash

    # Configure
    cmake --preset macos-arm64-clang

    # Build "Debug" configuration
    cmake --build --preset macos-arm64-clang-debug

    # Build "Release" configuration
    cmake --build --preset macos-arm64-clang-release

The build artifacts are placed in ``build\macos-arm64-clang\Debug`` or
``build\macos-arm64-clang\Release``.

To build for the x64 architecture, use the ``macos-x64-clang`` preset.

**Tested on:**

* macOS 15.5
* Xcode 16.4 (clang 17.0.0)
* CMake 4.0.3
* Ninja 1.13.1
* pkg-config 2.5.1


Configuration options
---------------------

SlangPy can be configured using the following CMake options. These options
can be specified on the command line when running CMake, for example:

.. code-block:: bash

    cmake --preset windows-msvc -DSGL_BUILD_DOC=ON -DSGL_BUILD_EXAMPLES=OFF -DSGL_BUILD_TESTS=OFF


The following table lists the available configuration options:

.. list-table::
    :widths: 35 10 35
    :header-rows: 1
    :align: left

    * - Option
      - Default
      - Description
    * - ``SGL_BUILD_PYTHON``
      - ``ON``
      - Build sgl Python extension
    * - ``SGL_BUILD_EXAMPLES``
      - ``ON``
      - Build sgl examples
    * - ``SGL_BUILD_TESTS``
      - ``ON``
      - Build sgl tests
    * - ``SGL_BUILD_DOC``
      - ``OFF``
      - Build sgl documentation
    * - ``SGL_USE_DYNAMIC_CUDA``
      - ``ON``
      - Load CUDA driver API dynamically
    * - ``SGL_DISABLE_ASSERTS``
      - ``OFF``
      - Disable asserts
    * - ``SGL_ENABLE_PCH``
      - ``OFF``
      - Enable precompiled headers
    * - ``SGL_ENABLE_ASAN``
      - ``OFF``
      - Enable AddressSanitizer (Clang only)
    * - ``SGL_ENABLE_UBSAN``
      - ``OFF``
      - Enable UndefinedBehaviorSanitizer (Clang only)
    * - ``SGL_ENABLE_HEADER_VALIDATION``
      - ``OFF``
      - Enable header validation


Sanitizer builds
----------------

SlangPy supports AddressSanitizer (ASan) and UndefinedBehaviorSanitizer
(UBSan) with Clang. Enabling these options instruments the native SlangPy
libraries, the Python extension, C++ tests, examples, and the embedded
``slang-rhi`` library. Prebuilt dependencies such as Slang, Python, and GPU
drivers are not instrumented.

Configure and build a sanitizer-enabled ``RelWithDebInfo`` build with the
platform's Clang preset. For example, on Linux use::

    cmake --preset linux-clang --fresh -DSGL_ENABLE_ASAN=ON -DSGL_ENABLE_UBSAN=ON
    cmake --build --preset linux-clang-relwithdebinfo

On Windows, use Clang 22 or newer and a ``RelWithDebInfo`` build. Clang 22
includes the fix for incorrect ASan instrumentation of C++ exception catch
parameters on Windows. Configure both sanitizers with::

    cmake --preset windows-clang --fresh -DSGL_ENABLE_ASAN=ON -DSGL_ENABLE_UBSAN=ON
    cmake --build --preset windows-clang-relwithdebinfo

``RelWithDebInfo`` is required on Windows because LLVM's UBSan C++ runtime uses
the release iterator ABI. It retains symbols while avoiding an ABI mismatch
with the Debug C++ runtime. The ``vptr`` check is disabled on Windows because
Clang ships its support library with a static-CRT ABI that is incompatible with
SlangPy's dynamic-CRT dependencies; SlangPy explicitly links the compatible
core runtime and the remaining UBSan checks are enabled.

On Apple Silicon, use ``macos-arm64-clang`` and
``macos-arm64-clang-relwithdebinfo`` with both sanitizer options.

On Linux, the native extension is loaded by an ordinary Python executable, so
test processes must preload the Clang sanitizer runtimes. The helper below
prints the required environment assignments for local use and writes them
directly to the GitHub Actions environment when ``GITHUB_ENV`` is present::

    python tools/setup-sanitizer-env.py \
        --os linux \
        --sanitizers address,undefined \
        --binary-dir build/linux-clang/RelWithDebInfo

On Linux, apply the printed environment assignments and run the C++ and Python
tests normally. Pass the generated leak reports through
``tools/filter-lsan-reports.py``. The filter fails for leaks originating in
SlangPy or ``slang-rhi`` and ignores reports whose allocation site is external
code such as Python or a GPU driver.

On Windows and macOS, loading the sanitizer runtimes only when the SlangPy
extension is imported is too late for CPython's existing allocations. ASan
builds on these platforms therefore also produce ``slangpy_sanitizer_python``
(``.exe`` on Windows). This small executable links the runtimes before it
initializes stock CPython; CPython itself does not need to be rebuilt. Use it in
place of the ordinary Python executable when running tests. On Windows, the
environment helper also copies the matching runtime DLL beside the build output
and adds its directory to ``PATH``. For example::

    build\windows-clang\RelWithDebInfo\slangpy_sanitizer_python.exe tools\ci.py unit-test-python --disable-torch

On macOS, use the corresponding host without the ``.exe`` suffix::

    build/macos-arm64-clang/RelWithDebInfo/slangpy_sanitizer_python \
        tools/ci.py unit-test-python --disable-torch

Only instrumented SlangPy and ``slang-rhi`` code receives complete ASan
coverage; prebuilt Python and GPU driver code does not. Sanitizer tests make
installed ``torch`` and ``slangpy_torch`` packages unavailable because they
load separately built native libraries into the instrumented process. Tests
that require them are skipped in the same way as in an environment where the
packages are not installed.




Updating the API Reference
--------------------------

SlangPy uses ``pybind11_mkdoc`` to extract documentation strings from the C++
source code. These comments are then used by ``nanobind`` to generate Python
documentation comments, which are in turn used when building the API Reference
document.

The documentation pipeline has these distinct outputs:

* ``src/slangpy_ext/py_doc.h`` contains docstrings extracted from C++ headers.
* ``slangpy/**/*.pyi`` contains Python signatures generated by nanobind during a
  normal build.
* ``docs/generated/api.rst`` is the legacy API snapshot generated from a built
  SlangPy package.
* ``docs/api/api.json`` is the structured, renderer-independent public API
  snapshot generated statically from Python sources and nanobind stubs.
* ``docs/generated/api/*.rst`` contains temporary section pages rendered only
  from the structured snapshot and public API contract.
* ``docs/api/coverage-baseline.json`` records the reviewed documentation
  coverage ratchet and per-symbol status.
* The Sphinx HTML directory contains the rendered documentation site.
* Every published Sphinx page has a Markdown counterpart at the same relative
  path with a ``.md`` suffix. These pages retain signatures, code blocks,
  cross-references, and source links without site navigation markup.
* ``llms.txt`` is a small reviewed index of the most useful Markdown pages.
  ``llms-full.txt`` is deliberately disabled because the restored low-level API
  makes a single combined artifact unnecessarily large.
* ``api/api.json`` and ``api/coverage.json`` are schema-versioned,
  machine-readable copies of the public inventory and current coverage report
  published with the site.

To update documentation extracted from C++ comments, run the
``slangpy_pydoc`` target before rebuilding SlangPy:

.. code-block:: bash

    # Install Python build prerequisites
    pip install -r requirements-dev.txt

    # Install Python documentation build prerequisites
    pip install -r docs/requirements.txt

    # Install native documentation extraction prerequisites
    pip install -r docs/requirements-native.txt

    # Configure
    cmake --preset windows-msvc

On Windows, make ``libclang.dll`` discoverable before running the extraction
target. For a default LLVM installation, use:

.. code-block:: powershell

    $env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin\libclang.dll"

The exact path may differ for a custom LLVM installation. If libclang is not
discoverable, ``pybind11_mkdoc`` can report failures from its worker threads
without updating ``py_doc.h``; always inspect the target output and the
resulting diff.

Then regenerate and rebuild:

.. code-block:: bash

    # Regenerate docstrings extracted from C++ headers
    cmake --build --preset windows-msvc-release --target slangpy_pydoc

    # Rebuild the extension and nanobind stubs
    cmake --build --preset windows-msvc-release

Generate and validate the structured API snapshot with:

.. code-block:: bash

    python tools/docs.py inventory
    python tools/docs.py inventory --check
    python tools/docs.py render
    python tools/docs.py coverage --check

The reviewed public surface is defined in ``docs/public_api.toml``. It includes
the named Core, Device, Math, UI, and other sections migrated from the legacy
API order, but deliberately excludes the legacy generator's automatically
populated ``Miscellaneous`` section. To list reachable public-looking names
that are not yet in the contract, run ``python tools/docs.py
report-unclassified``. Unclassified names are review candidates; they are not
added to the published API automatically. The generated section pages are
ignored by Git and are recreated by every documentation preparation or build.

Agent documentation tools
^^^^^^^^^^^^^^^^^^^^^^^^^

Generate a bounded Markdown context package for one reviewed symbol without
modifying source files:

.. code-block:: bash

    python tools/docs.py context slangpy.Module
    python tools/docs.py context slangpy.Module --output docs/_build/module-context.md

The package contains signatures, current documentation and gaps, the canonical
source location, native declaration and binding locations when applicable,
related public symbols, tests, and examples. All links are repository-relative.

List incomplete documentation in reviewed API priority order, then by coverage
status:

.. code-block:: bash

    python tools/docs.py tasks
    python tools/docs.py tasks --section core
    python tools/docs.py tasks --section core --format json

Markdown output is intended for maintainers; JSON output has a schema version
and is suitable for task automation. Documentation must still be changed in
authoritative Python docstrings or C++ comments, never in generated snapshots,
stubs, or rendered pages.

Regenerate the retained legacy snapshot and run a strict HTML build of the
structured reference with:

.. code-block:: bash

    python tools/ci.py --config Release docs

This command requires a built SlangPy package. Read the Docs deliberately uses
``python tools/docs.py prepare --api-mode snapshot`` instead, which validates
the checked-in snapshots and renders the structured pages without attempting
to import the native extension. The legacy ``docs/generated/api.rst`` remains
available for comparison during the migration, but it is no longer included in
the published API landing page.

The strict HTML build also runs the agent-output checks. It verifies that every
Sphinx source page has Markdown output, checks representative narrative, native
API, Python API, and notebook-derived pages for important structure, validates
the curated index and published JSON, and ensures ``llms-full.txt`` remains
absent.

Documentation coverage
^^^^^^^^^^^^^^^^^^^^^^

Coverage classifies each reviewed symbol as ``missing``, ``placeholder``,
``summary``, or ``complete``. A complete callable needs a real summary and
documentation for every public parameter and non-``None`` return. CI rejects a
lower total score, a regression of a completed symbol, a new undocumented
public symbol, or a newly introduced placeholder.

Only update the reviewed baseline after inspecting the source documentation and
the reported changes:

.. code-block:: bash

    python tools/docs.py coverage
    python tools/docs.py coverage --output docs/api/coverage-baseline.json
    python tools/docs.py coverage --check

Expanding the contract to include an existing API family requires a reviewed
migration baseline in the same change. This admits pre-existing documentation
debt for that family; after the baseline is checked in, the ordinary ratchet
again rejects undocumented additions and regressions.

Documentation examples
^^^^^^^^^^^^^^^^^^^^^^

Examples must have an explicit verification level. ``doctest`` snippets are
executable, ``literalinclude`` snippets must point to tested source, and
notebooks are covered by the corresponding sample tests on GPU-capable CI.
Ordinary ``code-block`` snippets are explicitly illustrative: they should stay
small, omit environment-specific output, and must not be described as tested or
complete programs. Prefer executable or included examples for user workflows
where correctness depends on more than the concept being illustrated.

**Tested on:**

* Windows 10 (build 19045)
* Visual Studio 2022 (Version 17.13.6)
* CMake 4.0.2
* Ninja 1.12.1


VS Code
-------

TBD
