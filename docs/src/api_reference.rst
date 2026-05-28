.. _sec-api-reference:

:tocdepth: 3

API reference
=============

Overview
--------

This API reference documentation was automatically generated from
Python docstrings. The docstrings are generated in the binding code
from the C++ API comments.

The main ``slangpy`` module contains all the basic types
required to load and call Slang functions from Python.

The ``slangpy.reflection`` module is a wrapper around the Slang reflection API
exposed by SlangPy. It is used extensively internally by SlangPy, but is a useful way
of introspecting the Slang code in general. The most common way to access reflection
data is by accessing the ``SlangProgramLayout`` of a module via the
``Module.layout`` attribute.

The ``slangpy.bindings`` module contains the tools required to extend SlangPy to
support new Python types. All slangpy's built-in types are implemented using these
classes (see ``slangpy/builtin``).

.. include:: ../generated/api.rst
