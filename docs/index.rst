Getting Started
===============

Why SlangPy?
------------

With a wide variety of Python libraries for GPU research already available, why create another? SlangPy aims to be the library of choice for real-time graphics and machine learning research by exposing GPUs in a highly efficient and developer-friendly way.

SlangPy achieves this through:

* **The Slang Shading Language:** At its core, SlangPy sits on top of the `Slang <https://shader-slang.org>`_ shading language. This allows you to write compute code using a modern, flexible language designed specifically for GPU computation.
* **Comprehensive Graphics API:** It provides a full-featured graphics API, giving you direct access to a lot of the hardware's underlying capabilities, which is crucial for advanced research.
* **Cross-Platform Support:** SlangPy is built for portability, with support for D3D12, Vulkan, Metal, and CUDA, ensuring your work can run across different platforms without significant changes.
* **A Functional, Boilerplate-Free API:** The library uses a functional API that dramatically reduces the boilerplate code typically involved in lower-level GPU programming, such as setting up compute kernels and managing binding logic.
* **Research-to-Production Workflow:** It creates a solid platform for combining Python and Slang, where the optimized Slang code from your research can be taken directly into production, streamlining your development process.

Getting Started
---------------

To begin using SlangPy, please install the library as instructed below. We then recommend moving to the basics section, starting with the tutorial at ``<src/basics/firstfunctions>``. This will guide you through using the higher-level functional API to write and call your first functions in Slang from Python.

Installation
------------

SlangPy is available as pre-compiled wheels via PyPi. Installing SlangPy is as simple as running:

.. code-block:: bash

    pip install slangpy

To enable PyTorch integration, simply ``pip install pytorch`` as usual and it will be detected automatically by SlangPy.

You can also compile SlangPy from source:

.. code-block:: bash

   git clone https://github.com/shader-slang/slangpy.git --recursive
   cd slangpy
   pip install -r requirements-dev.txt
   pip install .


See :ref:`developer guide <sec-compiling>` for more detailed information on how to compile from source.

Requirements
------------

* **python >= 3.9**

Optionally:

* CUDA Toolkit >= 11.8; 12.8 recommended (on Windows/Linux, for cuda
  acceleration)

* Xcode >= 16; 16.4 recommended (on macOS, the metal compiler is required for
  acceleration on a Metal 3.1+ capable device)

* `PyTorch <https://pytorch.org/get-started/>`_ >= 2.7.1 (for optional
  integration)

Citation
--------

If you use SlangPy in a research project leading to a publication, please cite the project. The BibTex entry is:

.. code-block:: bibtex

    @software{slangpy,
        title = {SlangPy},
        author = {Simon Kallweit and Chris Cummings and Benedikt Bitterli and Sai Bangaru and Yong He},
        note = {https://github.com/shader-slang/slangpy},
        version = {0.32.0},
        year = 2025
    }

.. toctree::
   :hidden:

   changelog
   self

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Basics

    src/basics/firstfunctions
    src/basics/returntype
    src/basics/buffers
    src/basics/textures
    src/basics/nested
    src/basics/typemethods
    src/basics/broadcasting
    src/basics/mapping
    src/basics/index_representation

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Auto-Diff

    src/autodiff/autodiff
    src/autodiff/pytorch

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Generators

    Overview <src/generators/generators>
    Ids <src/generators/generator_ids>
    Random Numbers <src/generators/generator_random>
    Grid <src/generators/generator_grid>

.. toctree::
    :maxdepth: 1
    :caption: Graphics Tutorials
    :hidden:

    src/tutorials/image_io_and_manipulation
    src/tutorials/compute_shader

.. toctree::
    :maxdepth: 1
    :caption: Guides
    :hidden:

    src/developer_guide

.. toctree::
    :maxdepth: 1
    :caption: Reference
    :hidden:

    src/api_reference
