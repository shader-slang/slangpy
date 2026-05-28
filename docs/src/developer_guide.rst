.. _sec-developer-guide:

Developer Guide
=================

Overview
--------

TBD

Project structure
-----------------

.. list-table::
    :widths: 2 5
    :header-rows: 1

    * - Directory
      - Description
    * - ``build``
      - Build output.
    * - ``cmake``
      - CMake modules.
    * - ``docs``
      - Documentation source code.
    * - ``data``
      - Binary data (git submodule).
    * - ``examples``
      - Example applications.
    * - ``external``
      - Third-party dependencies.
    * - ``resources``
      - Text resources.
    * - ``src``
      - Source code.
    * - ``tools``
      - Host tools and scripts (cmake, ninja, clang-tools, etc.).
    * - ``samples``
      - Tutorials, examples and experiments (git submodule).


Further reading
---------------

.. toctree::
    :maxdepth: 1

    developer_guide/compiling
    developer_guide/coding_style
    developer_guide/testing
