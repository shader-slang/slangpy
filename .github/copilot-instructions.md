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
    - General utility scripts are in #tools
    - CI workflows are in #.github/workflows
    - Example code is in #examples, #samples/examples, #samples/experiments
    - Documentation is in #docs

Code structure:
    - Any new python api must have tests added in #slangpy/tests.
    - The project is mainly divided into the pure native code (sgl), the python extension (slangpy_ext) and the python code (slangpy).
    - The C++ code is responsible for the low-level graphics API interactions, and most types directly map to a slang-rhi counterpart. i.e. Device wraps the slang-rhi rhi::IDevice type

Building:
    - On windows, build using: 'cmake --build .\build\windows-msvc --config Debug'
    - On linux, build using: 'cmake --build ./build/linux-gcc --config Debug'
    - If the build environment gets corrupted, you can reconfigure with:
        - windows: cmake --preset windows-msvc --fresh
        - linux: cmake --preset linux-gcc --fresh

Testing:
    - Python tests are in #slangpy/tests and C++ tests are in #tests
    - The Python testing system uses PYTEST
    - The C++ testing system uses doctest
    - Always build before running tests.
    - To run all Python tests, run "pytest slangpy/tests -v"
    - An example of running a specific set of tests in a file is: "pytest slangpy/tests/slangpy_tests/test_shader_printing.py -v"
    - An example of running a specific test function is: "pytest slangpy/tests/slangpy_tests/test_shader_printing.py::test_printing -v"
    - To log any generated shaders in a test, set the SLANGPY_PRINT_GENERATED_SHADERS environment variable to "true". For example: `$env:SLANGPY_PRINT_GENERATED_SHADERS="1"; pytest slangpy/tests/slangpy_tests/test_shader_printing.py -v` (PowerShell syntax)

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

Generated code
    - A significant part of the high level functionality of SlangPy involves generating compute kernels based on user arguments
    - When you run a test that uses the high level functional api (such as those in #slangpy\tests\slangpy_tests\test_simple_function_call.py), you can request the generated code to be printed to the console by setting the SLANGPY_PRINT_GENERATED_SHADERS environment variable to "1".

CI
    - Our CI system uses github, and the main ci job is in #.github/workflows/ci.yml
    - It works by calling #tools/ci.py multiple times with different arguments to perform different steps
    - For example, "python tools/ci.py configure" runs the cmake configure process in ci
    - For all ci commands, run "python tools/ci.py --help"

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
