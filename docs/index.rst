Getting Started
===============

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

- **python >= 3.9**

Citation
--------

If you use SlangPy in a research project leading to a publication, please cite the project. The BibTex entry is:

.. code-block:: bibtex

    @software{slangpy,
        title = {SlangPy},
        author = {Simon Kallweit and Chris Cummings and Benedikt Bitterli and Sai Bangaru and Yong He},
        note = {https://github.com/shader-slang/slangpy},
        version = {0.31.0},
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
    :caption: Tutorials
    :hidden:

    src/basic_tutorials

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
    src/api/slangpy
    src/api/reflection
    src/api/bindings
