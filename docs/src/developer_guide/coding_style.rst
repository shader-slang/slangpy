.. _sec-coding-style:

Coding Style
============

Because ``slangpy`` is both a C++ library and a Python module, the coding style
should be consistent across both languages. Therefore we derive our coding
conventions from Python's `PEP 8 <https://peps.python.org/pep-0008>`_
style guide.


Formatting
----------

We use ``clang-format`` to format C++ code, and ``black`` to format Python code.

To format all C++ files in the project, you can run:

.. code-block:: bash

    # Windows
    tools/format_code.bat

    # Linux and macOS
    ./tools/format_code.sh

To exclude files from formatting, add them to the ``.clang-format-ignore`` file.


Naming conventions
------------------

* Class and struct names are written in ``PascalCase``.
* Function and member function names are written in ``snake_case``.
* Constant names are written in ``UPPER_SNAKE_CASE``.
* Variable names are written in ``snake_case``.
* Enum type names are written in ``PascalCase``, and enum values are written in ``snake_case``.
* Private member variables are prefixed with ``m_``.

Properties
----------

In C++, we use getter and setter functions to access class properties.

* Getter functions are written as ``const T& property() const;``.
* Setter functions are written as ``void set_property(const T& property);``.

In Python, we expose these getters/setters as properties.

So while you do the following in C++:

.. code-block:: c++

    auto value = object.property();
    object.set_property(value);

You write the same in Python as:

.. code-block:: python

    value = object.property
    object.property = value

Predicate methods and boolean properties
---------------------------------------

Names beginning with ``is_`` or ``has_`` denote predicate methods that return a
boolean. Expose them as methods in Python, including when they take no arguments.
Use nanobind's ``def`` rather than ``def_prop_ro`` for these methods:

.. code-block:: python

    if window.is_visible():
        window.hide()

Boolean properties use names without ``is_`` or ``has_``, such as ``visible``,
``enabled``, or ``writable``. This convention applies to both native bindings and
Python-defined APIs.
