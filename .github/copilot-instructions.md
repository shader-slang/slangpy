# Copilot Instructions for SlangPy

This file provides guidance to GitHub Copilot when working with code in this repository.

## Project Overview

SlangPy is a native Python extension that provides a high-level interface for working with low-level graphics APIs (Vulkan, Direct3D 12, CUDA). The native side wraps the slang-rhi project (`external/slang-rhi`) using nanobind bindings. The project also contains a "functional API" that allows users to call Slang functions on the GPU with Python function call syntax.

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `src/sgl/` | Native C++ code (core GPU abstraction layer) |
| `src/slangpy_ext/` | Python bindings (nanobind) |
| `slangpy/` | Python package implementation |
| `slangpy/tests/` | Python tests (pytest) |
| `tests/` | C++ tests (doctest) |
| `tools/` | General utility scripts |
| `.github/workflows/` | CI workflows |
| `examples/`, `samples/examples/`, `samples/experiments/` | Example code |
| `docs/` | Documentation |
| `external/` | External C++ dependencies |

## Key Rules

1. **New Python APIs must have tests** in `slangpy/tests/`
2. **Always build before running tests**
3. **Run pre-commit after completing tasks** (`pre-commit run --all-files`; re-run if it modifies files)
4. **Use type annotations** for all Python function arguments
5. **Minimize new dependencies** — the project has minimal external deps

## Terminal Guidelines

**NEVER use PowerShell piping** with build or test commands:
```bash
# BAD — do not do this:
cmake --build --preset windows-msvc-debug 2>&1 | Select-Object -Last 30
pytest slangpy/tests -v 2>&1 | Select-Object -Last 30

# GOOD:
cmake --build --preset windows-msvc-debug
pytest slangpy/tests -v
```

## Architecture

The project has three main layers:
1. **Python Layer** (`slangpy/`) — High-level API with Module, Function, Device classes
2. **C++ Binding Layer** (`src/slangpy_ext/`) — Nanobind-based Python-C++ interface
3. **Core SGL Layer** (`src/sgl/`) — Low-level GPU device management and shader compilation

C++ types typically map to slang-rhi counterparts (e.g., `Device` wraps `rhi::IDevice`).

### Key Components

- **Module**: Container for Slang shader code, loaded from `.slang` files
- **Function**: Callable GPU function with automatic Python↔GPU marshalling
- **Device**: GPU context managing resources and compute dispatch
- **CallData**: Cached execution plans for optimized repeated calls
- **Buffer/Texture**: GPU memory resources with Python array interface

### Package Structure

- `slangpy/__init__.py` re-exports the public API — check it to see what's user-facing vs internal.
- `slangpy/slangpy/` contains additional internal implementation modules.
- `slangpy/builtin/` contains built-in marshall implementations for standard types.
- Bindings in `src/slangpy_ext/` use nanobind with `ref<T>` pointers (ref-counted via `Object` base class). Use `make_ref<T>(...)` to create instances. Python ↔ C++ ownership is managed via nanobind type casters registered in the binding modules.

### Call Flow

1. Python loads `.slang` file as Module
2. Functions extracted via reflection
3. Python calls trigger: argument type analysis and caching → data marshalling to GPU buffers → kernel compilation and dispatch → results copied back to Python/NumPy/PyTorch

### Error Handling

- Python-layer errors raise standard Python exceptions (typically `ValueError`, `TypeError`, or `SlangPyError`).
- C++ errors are translated to Python exceptions via nanobind. Shader compile errors surface as exceptions containing the Slang compiler diagnostic text.
- GPU errors (device lost, out of memory) propagate as exceptions from the RHI layer.
- When debugging, set `SLANGPY_PRINT_GENERATED_SHADERS=1` to see the generated kernel code that gets compiled.

## Build Commands

Replace `<PLATFORM>` with `windows-msvc`, `linux-gcc`, or `macos-arm64-clang` as appropriate:

```bash
cmake --preset <PLATFORM>                    # Configure
cmake --build --preset <PLATFORM>-debug      # Build (debug)
cmake --preset <PLATFORM> --fresh            # Reconfigure from scratch
```

Available presets: `windows-msvc`, `windows-arm64-msvc`, `linux-gcc`, `macos-arm64-clang`.

## Testing

**Always build before running tests.**

```bash
pytest slangpy/tests -v                        # All Python tests
pytest samples/tests -vra                      # Example tests
python tools/ci.py unit-test-cpp               # C++ unit tests
pytest slangpy/tests/slangpy_tests/test_X.py -v          # Specific file
pytest slangpy/tests/slangpy_tests/test_X.py::test_fn -v # Specific function
```

Debug generated shaders (PowerShell):
```bash
$env:SLANGPY_PRINT_GENERATED_SHADERS="1"; pytest slangpy/tests/slangpy_tests/test_X.py -v
```

### Test Patterns

- GPU tests use a `device` fixture (typically `spy.Device()`) — look at existing tests for the pattern.
- Tests that need `.slang` shader files place them alongside the test file or in a `shaders/` subdirectory.
- Use `spy.Tensor.from_numpy(device, arr)` to create GPU tensors, and `.to_numpy()` to read back.
- For new marshalls or types, add tests that exercise both the kernel generation (first call) and cached dispatch (subsequent calls).

### Common Pitfalls

- Forgetting to rebuild C++ after changing `src/` files — Python will use stale bindings.
- Shader cache can mask issues; delete `.slangpy_cache/` if you suspect stale compilation.
- Platform-specific test failures: some tests skip on CPU-only or specific GPU vendors.
- PyTorch integration tests only run when PyTorch is installed.

## Code Style

### C++
- **Classes**: PascalCase | **Functions/variables**: snake_case | **Members**: `m_` prefix

### Python
- **Classes**: PascalCase | **Functions/variables**: snake_case | **Members**: `m_` prefix
- **All arguments must have type annotations**

## Documentation Style

### C++ (Doxygen)

```cpp
/// Description.
void do_something();

/// Pack two float values to 8-bit snorm.
/// @param v Float values in [-1,1].
/// @param options Packing options.
/// @return 8-bit snorm values in low bits, high bits all zero.
uint32_t pack_snorm2x8(float2 v, const PackOptions options = PackOptions::safe);
```

### Python (Sphinx)

```python
def myfunc(x: int, y: int) -> int:
    """
    Description.

    :param x: Some parameter.
    :param y: Some parameter.
    :return: Some return value.
    """
```

## Slang Language Basics

Slang is a shader language based on HLSL. Key patterns used in this project:
- `[shader("compute")]` attribute marks GPU entry points
- `StructuredBuffer<T>` / `RWStructuredBuffer<T>` for typed GPU arrays
- `uint3 tid : SV_DispatchThreadID` for thread indexing
- Generics via `<T>`, interfaces via `interface IFoo`, conformance via `struct Foo : IFoo`
- Differentiable functions: `[Differentiable] float foo(float x)` with `bwd_diff(foo)` for backprop
- See `.slang` files in `slangpy/tests/` for project-specific patterns

# The Functional API

The functional API allows calling Slang GPU functions from Python with automatic type marshalling, vectorization, and kernel generation.

## Overview

```slang
// myshader.slang
float add(float a, float b) { return a + b; }
```

```python
import slangpy as spy
import numpy as np

device = spy.Device()
module = spy.Module.load_from_file(device, "myshader.slang")
a = spy.Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
b = spy.Tensor.from_numpy(device, np.array([4, 5, 6], dtype=np.float32))
result = module.add(a, b)  # Returns Tensor([5, 7, 9])
```

## Architecture

```
Python call → Phase 1: Signature Lookup (C++) → Cache hit? → Phase 3: Dispatch (C++)
                                               ↓ Cache miss
                                        Phase 2: Kernel Generation (Python)
```

### Key Files

| File | Purpose |
|------|---------|
| `slangpy/core/function.py` | `FunctionNode` — Python entry point for function calls |
| `slangpy/core/calldata.py` | `CallData` — kernel generation and caching |
| `slangpy/core/callsignature.py` | Type resolution and binding helpers |
| `slangpy/bindings/boundvariable.py` | `BoundCall`/`BoundVariable` — tracks Python↔Slang bindings |
| `slangpy/bindings/marshall.py` | `Marshall` base class for type marshalling |
| `slangpy/bindings/typeregistry.py` | Maps Python types to their Marshall implementations |
| `src/slangpy_ext/utils/slangpyfunction.cpp` | `NativeFunctionNode::call` — native call entry |
| `src/slangpy_ext/utils/slangpy.cpp` | `NativeCallData::exec` — native dispatch |

### Key Classes

| Class | Layer | Purpose |
|-------|-------|---------|
| `FunctionNode` | Python | Represents a callable Slang function with modifiers |
| `CallData` | Python | Generated kernel data (bindings, compiled shader) |
| `BoundCall` | Python | Collection of `BoundVariable` for a single call |
| `BoundVariable` | Python | Pairs Python value with Slang parameter |
| `NativeMarshall` | Both | Type-specific marshalling (shape, data binding) |
| `NativeCallData` | C++ | Native call data with cached dispatch info |
| `NativeCallDataCache` | C++ | Signature → CallData cache |

## Phase 1: Signature Lookup (C++, every call)

**Location:** `src/slangpy_ext/utils/slangpyfunction.cpp`

Runs on every call — must be fast. Builds a unique signature string from the function node chain and argument types/properties, looks it up in `NativeCallDataCache`. Cache hit skips to Phase 3; cache miss triggers Phase 2.

## Phase 2: Kernel Generation (Python, once per signature)

**Location:** `slangpy/core/calldata.py` → `CallData.__init__()`

Runs once per unique call signature. The pipeline:

1. **Unpack arguments** — recursively resolve `IThis` wrappers via `get_this()`
2. **Build BoundCall** — create a `BoundVariable` per argument, each assigned a `NativeMarshall` based on Python type (`int/float` → `ScalarMarshall`, `Tensor` → `TensorMarshall`, `dict` → recursive children)
3. **Apply explicit vectorization** — user-specified `function.map()` dimension/type mappings
4. **Type resolution** (`slangpy/reflection/typeresolution.py`) — each marshall's `resolve_types()` determines compatible Slang types. Resolves overloaded functions by best match.
5. **Bind parameters** — pair each `BoundVariable` with its resolved Slang parameter
6. **Apply implicit vectorization** — calculate per-argument dimensionality
7. **Calculate call dimensionality** — max across all arguments
8. **Create return value binding** — auto-creates `ValueRef` (dim 0) or `Tensor` (dim > 0) for `_result`
9. **Finalize mappings** — resolve Python→kernel dimension mappings (default: right-aligned)
10. **Calculate differentiability** — determine gradient support per argument
11. **Generate code** — produce Slang compute kernel source
12. **Compile shader** — compile via Slang; cache the `CallData`

### Type Resolution Reference

| Python Value | Slang Parameter | Resolved Binding |
|--------------|-----------------|------------------|
| `Tensor[float, 2D]` | `float` | `float` (elementwise) |
| `Tensor[float, 2D]` | `Tensor<float,2>` | `Tensor<float,2>` (whole) |
| `Tensor[float, 2D]` | `float2` | `float2` (row as vector) |
| `Tensor[float, 2D]` | `vector<T,2>` | `vector<float,2>` (generic) |

### Vectorization Dimensionality Reference

| Python Value | Slang Parameter | Dimensionality |
|--------------|-----------------|----------------|
| `Tensor[float, 2D shape=(H,W)]` | `float` | 2 (one thread per element) |
| `float` | `float` | 0 (single thread) |
| `Tensor[float, 2D]` | `Tensor<float,2>` | 0 (whole tensor per thread) |
| `Tensor[float, 2D shape=(H,W)]` | `float2` | 1 (one thread per row) |

## Phase 3: Dispatch (C++, every call)

**Location:** `src/slangpy_ext/utils/slangpy.cpp`

Runs on every call, entirely in C++:
1. **Unpack arguments** (native `unpack_args`/`unpack_kwargs`)
2. **Calculate call shape** — each marshall's `get_shape()` returns concrete dimensions; call shape determines thread count
3. **Allocate return value** — create output Tensor/ValueRef if `_result` not provided
4. **Bind uniforms & dispatch** — marshalls write data to GPU via `write_shader_cursor_pre_dispatch()`, then `dispatch()`
5. **Read results** — post-dispatch readback via `read_calldata()` and `read_output()`

## Adding New Types

To support a new Python type in the functional API:

1. **Create a Marshall** in `slangpy/bindings/` or `slangpy/builtin/` — subclass `Marshall` and implement `resolve_types()`, `resolve_dimensionality()`, `gen_calldata()`. See existing marshalls (e.g., `TensorMarshall`) for the pattern.
2. **Register** in `slangpy/bindings/typeregistry.py` — add entry to `PYTHON_TYPES` dict.
3. **(Optional) Native signature** — for performance, add a type signature handler in `NativeCallDataCache` constructor (`src/slangpy_ext/utils/slangpyfunction.cpp`).

## CI System

The CI uses GitHub Actions (`.github/workflows/ci.yml`) and calls `tools/ci.py`:

```bash
python tools/ci.py configure  # CMake configure
python tools/ci.py --help     # All available commands
```

## Dependencies

- Python runtime: `requirements.txt` | Dev/tests: `requirements-dev.txt`
- C++ dependencies: `external/`
- Testing: pytest (Python), doctest (C++)
- Shading language: Slang
- Formatting: `pre-commit` hooks (Black for Python, clang-format for C++)

## Debugging Slang Compiler Issues

Many issues in SlangPy originate from the Slang compiler itself (shader-slang/slang repo).

### Local Slang Build

```bash
# 1. Clone
git clone --recursive https://github.com/shader-slang/slang.git external/slang

# 2. Build
cmake.exe --preset vs2022 -S external/slang
cmake.exe --build external/slang/build --config Debug

# 3. Rebuild SlangPy against local Slang
rm -fr build
SET CMAKE_ARGS="-DSGL_LOCAL_SLANG=ON -DSGL_LOCAL_SLANG_DIR=external/slang -DSGL_LOCAL_SLANG_BUILD_DIR=build/Debug"
pip.exe install .
```

| CMake Option | Default | Description |
|--------|---------|-------------|
| `SGL_LOCAL_SLANG` | OFF | Enable to use a local Slang build |
| `SGL_LOCAL_SLANG_DIR` | `../slang` | Path to the local Slang repository |
| `SGL_LOCAL_SLANG_BUILD_DIR` | `build/Debug` | Build directory within the Slang repo |

### Workflow

1. Reproduce the issue in SlangPy
2. Clone and build Slang locally (above)
3. Read `external/slang/CLAUDE.md`
4. Edit Slang source → rebuild Slang → rebuild SlangPy → test

## Development Tips

- Use `python tools/ci.py` for most build/test tasks — handles platform-specific config
- PyTorch integration is automatic when PyTorch is installed
- Hot-reload is supported for shader development
