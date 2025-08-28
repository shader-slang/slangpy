.. _sec-testing:

Testing
=======

Because SlangPy aims to expose the entire API of its internal C++ library
(``sgl``) to Python, most unit tests should be written in Python. However,
some C++ unit tests are still needed for parts of ``sgl`` that are not exposed
to Python.

Python Unit Tests
-----------------

Python unit tests are written in the `pytest <https://docs.pytest.org/>`_
framework.

Unit tests are located next to where the code they test is located.
For example, the code for the ``Bitmap`` class is located in
``src/sgl/core/bitmap.cpp``,
and it's corresponding unit test is located in
``src/sgl/core/tests/test_bitmap.py``.

In order to run unit tests, you need to have the ``pytest`` package installed
in your active Python environment. You also need to have the ``sgl`` package
available. If you run a development build, use the ``setpath.bat`` or
``setpath.sh`` script to make the package available.

To run the tests, simply run the following command from the root directory:

.. code-block:: bash

    pytest src

To run all tests in a specific file, you can specify the file to run:

.. code-block:: bash

    pytest src/sgl/core/tests/test_bitmap.py

To run just one specific test case, you can additionally set the ``-k`` flag:

.. code-block:: bash

    pytest src/sgl/core/tests/test_bitmap.py -k test_jpg_io

Each test source file should also be executable as a standalone script. This is
most useful when debugging from an IDE. Allowing scripts to be run standalone
is accomplished by adding the following code at the end of each file:

.. code-block:: python

    if __name__ == "__main__":
        pytest.main([__file__, "-v"])



C++ Unit Tests
--------------

C++ unit tests are written in the `doctest <https://github.com/doctest/doctest>`_
framework.

The main application entry point is located in
``src/sgl/tests/sgl_tests.cpp``.
Similar to the Python unit tests, the C++ unit tests are located next to the
code they test in a ``tests`` directory.

To run the tests, simply run the following command from the binary output
directory:

.. code-block:: bash

    ./sgl_tests
