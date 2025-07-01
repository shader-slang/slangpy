.. _sec-compiling:

Compiling
=========

In order to compile ``sgl`` from source, the following prerequisites are
required:

* A C++20 compliant compiler (tested with Visual Studio 2022, GCC 11 and Clang 14)
* Python ``>= 3.8``
* git


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

To build on macOS, make sure you have a recent version of XCode installed.
You also need to install the XCode command line tools by running the following
command:

.. code-block:: bash

    xcode-select --install


Then use the following commands to build the project:

.. code-block:: bash

    # Configure
    cmake --preset macos-arm64-clang

    # Build "Debug" configuration
    cmake --build --preset macos-arm64-clang-debug

    # Build "Release" configuration
    cmake --build --preset macos-arm64-clang-release

The build artifacts are placed in ``build\macos-arm64-clang\bin\Debug`` or
``build\macos-arm64-clang\bin\Release``.

To build for the x64 architecture, use the ``macos-x64-clang`` preset.

**Tested on:**

* macOS TBD
* clang TBD
* CMake 3.27.7
* Ninja 1.11.1


Configuration options
---------------------

SlangPy can be configured using the following CMake options. These options
can be specified on the command line when running CMake, for example:

.. code-block:: bash

    cmake --preset windows-msvc -DSGL_BUILD_DOCS=ON -DSGL_BUILD_EXAMPLES=OFF -DSGL_BUILD_TESTS=OFF


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
    * - ``SGL_BUILD_DOCS``
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
      - Enable address sanitizer
    * - ``SGL_ENABLE_HEADER_VALIDATION``
      - ``OFF``
      - Enable header validation


VS Code
-------

TBD
