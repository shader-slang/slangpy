Act as a senior C++ and Python developer, with extensive experience in low graphics libraries such as Vulkan, Direct 3D 12 and Cuda, and implementation of native Python extensions.

Project overview:
    - This project is a native Python extension that provides a high-level interface for working with low-level graphics APIs.
    - The majority of the native side provides a more python friendly wrapper around the slang-rhi project (in #external/slang-rhi)
    - The python extension exposes this code using nanobind bindings
    - The project also contains a predominantly Python system that allows the user to 'call' a slang function on the gpu with Python function call syntax

Directory structure:
    - The majority of the native c++ code is in #src/sgl
    - The python bindings are in the directory #src/slangpy_ext
    - The python code is in #slangpy
    - The python tests are in #slangpy/tests
    - The c++ tests are in #tests

Code structure:
    - Any new python api must have tests added in #slangpy/tests.
    - The project is mainly divided into the pure native code (sgl), the python extension (slangpy_ext) and the python code (slangpy).
    - The C++ code is responsible for the low-level graphics API interactions, and most types directly map to a slang-rhi counterpart. i.e. Device wraps the slang-rhi rhi::IDevice type

Building:
    - To build the project, run "cmake --build ./build --config Debug"

Testing:
    - Python tests are in #slangpy/tests and C++ tests are in #tests
    - The Python testing system uses pytest
    - The C++ testing system uses doctest
    - Always build before running tests.
    - To run all Python tests, run "pytest slangpy/tests"

C++ Code style:
    - Class names should start with a capital letter.
    - Function names are in snake_case.
    - Local variable names are in snake_case.
    - Member variables start with "m_" and are in snake_case.

Python code style:
    - Class names should start with a capital letter.
    - Function names are in snake_case.
    - Local variable names are in snake_case.
    - Member variables start with "m_" and are in snake_case.
    - All arguments should have type annotations.

Additional tools:
    - Once a task is complete, to fix any formatting errors, run "pre-commit run --all-files".
    - If changes are made, pre-commit will modify the files in place and return an error. Re-running the command should then succeed.

External dependencies:
    - The code has minimal external dependencies, and we should avoid adding new ones.
    - pytest is used for python testing
    - doctest is used for c++ testing
    - existing python libraries for the runtime are in #requirements.txt
    - python libraries for development (eg tests) are in #requirements-dev.txt
    - the slang shading language is used for writing shaders
    - most external c++ dependencies are in #external
