.. _sec-api-reference:

:tocdepth: 3

API reference
=============

This reference is generated from the reviewed public API contract and the
structured API inventory. It intentionally excludes importable implementation
details that have not been classified as supported interfaces.

Primary API
-----------

* :doc:`Functional API <../generated/api/functional-api>` -- tensors, modules,
  and callable Slang functions.
* :doc:`Additional SlangPy API <../generated/api/slangpy>` -- supporting
  high-level functions and types.

Low-level graphics API
----------------------

* :doc:`Core <../generated/api/core>` -- fundamental data and utility types.
* :doc:`Constants <../generated/api/constants>` -- public package constants.
* :doc:`Logging <../generated/api/logging>` -- logging levels, outputs, and
  helper functions.
* :doc:`Windowing <../generated/api/windowing>` -- windows and input events.
* :doc:`Platform <../generated/api/platform>` -- operating-system services.
* :doc:`Threading <../generated/api/threading>` -- thread utilities.
* :doc:`Device <../generated/api/device>` -- devices, resources, command
  encoding, pipelines, shader reflection, and ray tracing.
* :doc:`Application <../generated/api/application>` -- application and window
  helpers.
* :doc:`Math <../generated/api/math>` -- vectors, matrices, quaternions, and
  math functions.
* :doc:`UI <../generated/api/ui>` -- immediate-mode user-interface types.
* :doc:`Utilities <../generated/api/utilities>` -- texture loading and external
  tool integration.

Extension API
-------------

* :doc:`Extension Author API <../generated/api/extension-api>` -- interfaces
  used to add Python-to-Slang type bindings.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../generated/api/functional-api
   ../generated/api/slangpy
   ../generated/api/core
   ../generated/api/constants
   ../generated/api/logging
   ../generated/api/windowing
   ../generated/api/platform
   ../generated/api/threading
   ../generated/api/device
   ../generated/api/application
   ../generated/api/math
   ../generated/api/ui
   ../generated/api/utilities
   ../generated/api/extension-api
