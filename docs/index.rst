SlangPy
=======

SlangPy is a library designed to make calling GPU code written in Slang extremely simple and easy.
Its core objectives are to:

* Make it quick and simple to call Slang functions on the GPU from Python
* Eradicate the boilerplate and bugs associated with writing compute kernels
* Grant easy access to Slang's auto-diff features
* Provide optional PyTorch support out of the box

It is built upon 2 core technologies:

* `Slang shading language <https://shader-slang.com/>`_: A modern, platform agnostic shading language with full auto-diff support.
* `Slang graphics library (SGL) <https://github.com/shader-slang/sgl>`_: A powerful Python extension providing a thin wrapper around the graphics layer.

By bringing these 2 technologies together with a simple and flexible Python library, calling GPU code from Python is as simple and easy as calling a function.

Community 
---------

We actively encourage and maintain a community of users and developers, so if you have any questions, suggestions or want to participate, please 
`join our Discord server <https://khr.io/slangdiscord>`_.


Getting Started
---------------

SlangPy can be installed from pip using:

.. code-block:: bash

    pip install slangpy

Or for more information on building from source, see the `installation <installation.html>`_ section.

A full set of examples can be found in the * `examples directory <https://github.com/shader-slang/slangpy/tree/main/examples>`_ in the SlangPy repo. Each
example is also covered by an article in these docs, so we recommend downloading the lot and then working them, starting
with the intros in the `basics <firstfunctions.html>`_ section. 


.. toctree::
   :hidden:
   :maxdepth: 0
   
   changelog
   releasenotes

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Basics
   
   installation
   firstfunctions
   buffers
   textures
   nested
   typemethods
   broadcasting
   mapping

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Auto-Diff
   
   autodiff
   pytorch

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API
   
   api/slangpy
   api/reflection
   api/bindings


