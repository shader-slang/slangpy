# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SlangPy is a cross-platform library that enables calling GPU code written in Slang from Python. It provides seamless integration with PyTorch for automatic differentiation and supports multiple GPU backends (Vulkan, D3D12, CUDA).

## Essential Commands

### Build Commands
```bash
# Install from source (recommended)
pip install .

# Configure and build with CMake
python tools/ci.py configure
python tools/ci.py build
```

### Testing Commands
```bash
# Run Python unit tests
pytest slangpy/tests -ra
# Or via CI tool
python tools/ci.py unit-test-python -p

# Run example tests
pytest samples/tests -vra

# Run C++ unit tests
python tools/ci.py unit-test-cpp

# Run a single test
pytest slangpy/tests/test_buffer.py::test_buffer_creation -v
```

### Code Quality Commands
```bash
# Type checking
pyright
# Or via CI tool
python tools/ci.py typing-check-python

# Format code (handled by pre-commit)
pre-commit run --all-files

# Black formatter for Python (line length: 100)
black . --line-length 100
```

## High-Level Architecture

### Three-Layer Architecture
1. **Python Layer** (`slangpy/`) - High-level API exposing Module, Function, Device classes
2. **C++ Binding Layer** (`src/slangpy_ext/`) - Nanobind-based Python-C++ interface
3. **Core SGL Layer** (`src/sgl/`) - Low-level GPU device management and shader compilation

### Key Components
- **Module**: Container for Slang shader code, loaded from `.slang` files
- **Function**: Callable GPU function with automatic Python↔GPU marshalling
- **Device**: GPU context managing resources and compute dispatch
- **CallData**: Cached execution plans for optimized repeated calls
- **Buffer/Texture**: GPU memory resources with Python array interface

### Call Flow
1. Python loads `.slang` file as Module
2. Functions extracted via reflection
3. Python calls trigger:
   - Argument type analysis and caching
   - Data marshalling to GPU buffers
   - Kernel compilation and dispatch
   - Results copied back to Python/NumPy/PyTorch

### Key Directories
- `slangpy/` - Python package implementation
- `src/slangpy_ext/` - C++ extension bindings
- `src/sgl/` - Slang Graphics Library (core GPU abstraction)
- `samples/` - Example Slang shaders and Python scripts
- `tests/` - Test Slang shaders used by unit tests
- `tools/` - Build and CI utilities

## Debugging Slang Compiler Issues

Many issues in SlangPy originate from the Slang compiler itself (shader-slang/slang repo). When you encounter such issues, follow these steps to build and debug Slang locally.

### Step 1: Clone the Slang Repository

```bash
# Clone into external/slang directory (with submodules)
git clone --recursive https://github.com/shader-slang/slang.git external/slang
```

### Step 2: Build Slang in Debug Mode

```bash
# Configure with CMake
cmake.exe --preset vs2022 -S external/slang

# Build debug configuration
cmake.exe --build external/slang/build --config Debug
```

The debug binaries will be generated in `external/slang/build/Debug/bin/`.

### Step 3: Configure SlangPy to Use Local Slang Build

Reconfigure SlangPy with the local Slang build:

```bash
# Reconfigure and rebuild slangpy
rm -fr build
SET CMAKE_ARGS="-DSGL_LOCAL_SLANG=ON -DSGL_LOCAL_SLANG_DIR=external/slang -DSGL_LOCAL_SLANG_BUILD_DIR=build/Debug"
pip.exe install .
```

### Key CMake Options for Local Slang

| Option | Default | Description |
|--------|---------|-------------|
| `SGL_LOCAL_SLANG` | OFF | Enable to use a local Slang build |
| `SGL_LOCAL_SLANG_DIR` | `../slang` | Path to the local Slang repository |
| `SGL_LOCAL_SLANG_BUILD_DIR` | `build/Debug` | Build directory within the Slang repo |

### Workflow for Debugging Slang Issues

1. Reproduce the issue in SlangPy
2. Clone and build Slang locally (steps above)
3. Read `external/slang/CLAUDE.md`
4. Make changes to Slang source code in `external/slang/`
5. Rebuild Slang: `cmake --build external/slang/build --config Debug`
6. Rebuild SlangPy to pick up changes
7. Test the fix

## Development Tips

- The project uses CMake with presets for different platforms (windows-msvc, windows-arm64-msvc, linux-gcc, macos-arm64-clang)
- PyTorch integration is automatic when PyTorch is installed
- Hot-reload is supported for shader development
- Use `python tools/ci.py` for most build/test tasks - it handles platform-specific configuration
- Pre-commit hooks enforce code formatting (Black for Python, clang-format for C++)
