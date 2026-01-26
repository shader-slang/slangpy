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

## Architecture

The project has three main layers:
1. **Python Layer** (`slangpy/`) - High-level API with Module, Function, Device classes
2. **C++ Binding Layer** (`src/slangpy_ext/`) - Nanobind-based Python-C++ interface
3. **Core SGL Layer** (`src/sgl/`) - Low-level GPU device management and shader compilation

C++ types typically map to slang-rhi counterparts (e.g., `Device` wraps `rhi::IDevice`).

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

## Build Commands

```bash
# Initial configure if needed
cmake --preset windows-msvc  # Windows
cmake --preset linux-gcc     # Linux

# Build (debug build for developement)
cmake --build --preset windows-msvc-debug
cmake --build --preset linux-gcc-debug

# Reconfigure if build is corrupted
cmake --preset windows-msvc --fresh  # Windows
cmake --preset linux-gcc --fresh     # Linux
```

AVOID powershell piping syntax when running builds. eg
```bash
- Bad: cmake --build --preset windows-msvc-debug 2>&1 | Select-Object -Last 30
- Good: cmake --build --preset windows-msvc-debug
```

## Testing Commands

**Always build before running tests.**

```bash
# Run all Python tests
pytest slangpy/tests -v

# Run example tests
pytest samples/tests -vra

# Run C++ unit tests
python tools/ci.py unit-test-cpp

# Run specific test file
pytest slangpy/tests/slangpy_tests/test_shader_printing.py -v

# Run specific test function
pytest slangpy/tests/slangpy_tests/test_shader_printing.py::test_printing -v

# Debug generated shaders (PowerShell)
$env:SLANGPY_PRINT_GENERATED_SHADERS="1"; pytest slangpy/tests/slangpy_tests/test_shader_printing.py -v
```

AVOID powershell piping syntax when running tests. eg
```bash
- Bad: pytest slangpy/tests -v 2>&1 | Select-Object -Last 30
- Good: pytest slangpy/tests -v
```

## Code Style

### C++

- **Class names**: PascalCase (start with capital letter)
- **Functions**: snake_case
- **Local variables**: snake_case
- **Member variables**: `m_` prefix with snake_case

### Python

- **Class names**: PascalCase (start with capital letter)
- **Functions**: snake_case
- **Local variables**: snake_case
- **Member variables**: `m_` prefix with snake_case
- **All arguments must have type annotations**

## Documentation Style

### C++ (Doxygen)

```cpp
// Simple getter/setter or void function
/// Description.
void do_something();

// Function with parameters/return values
/// Pack two float values to 8-bit snorm.
/// @param v Float values in [-1,1].
/// @param options Packing options.
/// @return 8-bit snorm values in low bits, high bits all zero.
uint32_t pack_snorm2x8(float2 v, const PackOptions options = PackOptions::safe);

// Enums and struct fields (comment before value)
enum class FunctionNodeType {
    /// Unknown or unspecified node type.
    unknown,
    /// Node representing uniform parameters.
    uniforms,
};
```

### Python (Sphinx)

Use Sphinx-style docstrings without type annotations in docstring (types are in function signature):

```python
def myfunc(x: int, y: int) -> int:
    """
    Description.

    :param x: Some parameter.
    :param y: Some parameter.
    :return: Some return value.
    """
```

# The Functional API

The functional API is the core feature of SlangPy that allows calling Slang GPU functions from Python with automatic type marshalling, vectorization, and kernel generation.

## Overview

```slang
// myshader.slang
float add(float a, float b) {
    return a + b;
}
```

```python
import slangpy as spy
import numpy as np

device = spy.Device()
module = spy.Module.load_from_file(device, "myshader.slang")

# Create tensors
a = spy.Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
b = spy.Tensor.from_numpy(device, np.array([4, 5, 6], dtype=np.float32))

# Call scalar 'add' on each element - SlangPy generates a kernel automatically
result = module.add(a, b)  # Returns Tensor([5, 7, 9])
```

**Debugging tip:** Set `SLANGPY_PRINT_GENERATED_SHADERS=1` to see generated kernels.

## Architecture Overview

The functional API operates in three phases:

```
Python call → Phase 1: Signature Lookup (C++) → Cache hit? → Phase 3: Dispatch (C++)
                                               ↓ Cache miss
                                        Phase 2: Kernel Generation (Python)
```

### Key Files

| File | Purpose |
|------|---------|
| `slangpy/core/function.py` | `FunctionNode` class - Python entry point for function calls |
| `slangpy/core/calldata.py` | `CallData` class - kernel generation and caching |
| `slangpy/core/callsignature.py` | Type resolution and binding helpers |
| `slangpy/bindings/boundvariable.py` | `BoundCall`/`BoundVariable` - tracks Python↔Slang bindings |
| `slangpy/bindings/marshall.py` | `Marshall` base class for type marshalling |
| `src/slangpy_ext/utils/slangpyfunction.cpp` | `NativeFunctionNode::call` - native call entry |
| `src/slangpy_ext/utils/slangpy.cpp` | `NativeCallData::exec` - native dispatch |

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

## Phase 1: Signature Lookup (Performance Critical)

**Entry:** `FunctionNode.call()` → `NativeFunctionNode::call()` (C++)

**Location:** `src/slangpy_ext/utils/slangpyfunction.cpp`

This phase runs on **every call** and must be extremely fast:

1. **Gather runtime options** - Collect `this` binding, CUDA stream, uniforms
2. **Build signature** - `NativeCallDataCache::get_args_signature()` creates a unique string from:
   - Function node chain (maps, constants, type conformances)
   - Argument types and properties (e.g., Tensor dtype, Buffer usage flags)
3. **Cache lookup** - Find existing `NativeCallData` for this signature
4. **Branch:**
   - **Cache hit** → Skip to Phase 3 (dispatch)
   - **Cache miss** → Call `generate_call_data()` (Phase 2)

```cpp
// Simplified from slangpyfunction.cpp
nb::object NativeFunctionNode::call(NativeCallDataCache* cache, nb::args args, nb::kwargs kwargs) {
    auto builder = make_ref<SignatureBuilder>();
    read_signature(builder);  // Add function chain info
    cache->get_args_signature(builder, args, kwargs);  // Add argument info

    std::string sig = builder->str();
    ref<NativeCallData> call_data = cache->find_call_data(sig);

    if (!call_data) {
        call_data = generate_call_data(args, kwargs);  // → Phase 2
        cache->add_call_data(sig, call_data);
    }
    return call_data->call(options, args, kwargs);  // → Phase 3
}
```

## Phase 2: Kernel Generation (One-time per signature)

**Entry:** `FunctionNode.generate_call_data()` → `CallData.__init__()`

**Location:** `slangpy/core/calldata.py`

This phase runs **once per unique call signature**. It analyzes argument types, resolves Slang function overloads, and generates GPU kernel code.

### Step 2.1: Unpack Arguments

```python
unpacked_args = unpack_args(*args)
unpacked_kwargs = unpack_kwargs(**kwargs)
```

Recursively converts Python wrapper classes (implementing `IThis` protocol with `get_this()`) to their underlying data (typically dicts).

### Step 2.2: Build BoundCall

```python
bindings = BoundCall(context, *unpacked_args, **unpacked_kwargs)
```

Creates a `BoundVariable` for each argument. Each `BoundVariable` gets a `NativeMarshall` based on the Python type:
- `int/float` → `ScalarMarshall`
- `Tensor` → `TensorMarshall`
- `dict` → recursive `BoundVariable` children

**Marshalls** define type-specific behavior for code generation and dispatch.

### Step 2.3: Apply Explicit Vectorization

```python
apply_explicit_vectorization(context, bindings, positional_mapping, keyword_mapping)
```

Applies user-specified mappings from `function.map()`:
```python
# Map arg1 dims to kernel dims (1,)(0,)
module.add.map((1,), (0,))(a, b)
# Cast arguments to specific types
module.process.map(module.Foo, module.Bar)(arg1, arg2)
```

### Step 2.4: Specialize (Type Resolution)

```python
resolve_result = specialize(context, bindings, function, diagnostics, this_type)
```

**Location:** `slangpy/reflection/typeresolution.py`

This is where SlangPy's "magic" happens. For each argument, the marshall's `resolve_types()` determines what Slang types it can bind to:

| Python Value | Slang Parameter | Resolved Binding |
|--------------|-----------------|------------------|
| `Tensor[float, 2D]` | `float` | `float` (elementwise) |
| `Tensor[float, 2D]` | `Tensor<float,2>` | `Tensor<float,2>` (whole) |
| `Tensor[float, 2D]` | `float2` | `float2` (row as vector) |
| `Tensor[float, 2D]` | `vector<T,2>` | `vector<float,2>` (generic) |

The system also resolves overloaded Slang functions by finding the single best match.

### Step 2.5: Bind Parameters

```python
bindings = bind(context, bindings, resolve_result.function, resolve_result.params)
```

Pairs each `BoundVariable` with its resolved Slang parameter. After binding:
- `BoundVariable.python` = `NativeMarshall` (Python type info)
- `BoundVariable.slang_type` = `SlangType` (Slang type info)

### Step 2.6: Apply Implicit Vectorization

```python
apply_implicit_vectorization(context, bindings)
```

Calculates **call dimensionality** - how many kernel dimensions needed:

| Python Value | Slang Parameter | Dimensionality |
|--------------|-----------------|----------------|
| `Tensor[float, 2D shape=(H,W)]` | `float` | 2 (one thread per element) |
| `float` | `float` | 0 (single thread) |
| `Tensor[float, 2D]` | `Tensor<float,2>` | 0 (whole tensor per thread) |
| `Tensor[float, 2D shape=(H,W)]` | `float2` | 1 (one thread per row) |

### Step 2.7: Calculate Call Dimensionality

```python
self.call_dimensionality = calculate_call_dimensionality(bindings)
```

The kernel dimensionality is the **maximum** across all arguments.

### Step 2.8: Create Return Value Binding

```python
create_return_value_binding(context, bindings, return_type)
```

If the function returns a value and user didn't provide `_result`, auto-creates:
- Dimensionality 0 → `ValueRef` (single value)
- Dimensionality > 0 → `Tensor` (array)

### Step 2.9: Finalize Mappings

```python
finalize_mappings(context, bindings)
```

Resolves dimension mappings - which Python dimensions map to which kernel dimensions. Default: right-align dimensions.

### Step 2.10: Calculate Differentiability

```python
calculate_differentiability(context, bindings)
```

Determines which arguments support gradients (for PyTorch autograd integration) based on:
- Marshall's `has_derivative` property
- Slang parameter's differentiability modifier

### Step 2.11: Generate Code

```python
generate_code(context, build_info, bindings, codegen)
```

Produces Slang kernel code. Example output:
```slang
// Auto-generated kernel
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID) {
    int idx = tid.x;  // Linear index for 1D call
    float a_val = call_data.a[idx];
    float b_val = call_data.b[idx];
    call_data._result[idx] = add(a_val, b_val);
}
```

### Step 2.12: Compile Shader

The generated code is compiled via Slang, and the `CallData` is cached for future calls with the same signature.

## Phase 3: Dispatch (Performance Critical)

**Entry:** `NativeCallData::call()` → `NativeCallData::exec()`

**Location:** `src/slangpy_ext/utils/slangpy.cpp`

This phase runs on **every call** and is implemented entirely in C++ for performance.

### Step 3.1: Unpack Arguments (Native)

```cpp
nb::list unpacked_args = unpack_args(args);
nb::dict unpacked_kwargs = unpack_kwargs(kwargs);
```

Same unpacking as Phase 2, but using native implementations.

### Step 3.2: Calculate Call Shape

```cpp
Shape call_shape = m_runtime->calculate_call_shape(
    m_call_dimensionality, unpacked_args, unpacked_kwargs, this);
```

Each `NativeMarshall::get_shape()` returns concrete dimensions. Example:
- Call dimensionality: 2
- `Tensor[float, shape=(100, 50)]` → contributes `(100, 50)`
- Final call shape: `(100, 50)` → dispatch 5000 threads

### Step 3.3: Allocate Return Value

```cpp
if (m_call_mode == CallMode::prim) {
    ref<NativeBoundVariableRuntime> rv_node = m_runtime->find_kwarg("_result");
    if (rv_node && !kwargs.contains("_result")) {
        nb::object output = rv_node->python_type()->create_output(context, rv_node.get());
        kwargs["_result"] = output;
    }
}
```

### Step 3.4: Bind Uniforms and Dispatch

```cpp
auto bind_call_data = [&](ShaderCursor cursor) {
    // Write call shape/stride metadata
    // ...

    // Write user arguments via marshalls
    m_runtime->write_shader_cursor_pre_dispatch(
        context, cursor, call_data_cursor,
        unpacked_args, unpacked_kwargs, read_back);
};

ref<ComputePassEncoder> pass_encoder = command_encoder->begin_compute_pass();
ShaderCursor cursor(pass_encoder->bind_pipeline(pipeline));
bind_call_data(cursor);
pass_encoder->dispatch(uint3(total_threads, 1, 1));
```

Each marshall's `write_shader_cursor_pre_dispatch()` writes its data to GPU uniforms.

### Step 3.5: Read Results

```cpp
// Post-dispatch: read back modified values
for (auto val : read_back) {
    bvr->python_type()->read_calldata(context, bvr.get(), rb_val, rb_data);
}

// Return result
if (m_call_mode == CallMode::prim) {
    auto rv_node = m_runtime->find_kwarg("_result");
    if (rv_node) {
        return rv_node->read_output(context, unpacked_kwargs["_result"]);
    }
}
```

## Adding New Types

To support a new Python type in the functional API:

1. **Create a Marshall** in `slangpy/bindings/` or `slangpy/builtin/`:
   ```python
   class MyTypeMarshall(Marshall):
       def resolve_types(self, context, bound_type) -> list[SlangType]:
           # Return compatible Slang types

       def resolve_dimensionality(self, context, binding, vector_target_type):
           # Calculate dimensions consumed by this type

       def gen_calldata(self, cgb, context, binding):
           # Generate uniform declarations
   ```

2. **Register in type registry** (`slangpy/bindings/typeregistry.py`):
   ```python
   PYTHON_TYPES[MyType] = lambda layout, value: MyTypeMarshall(layout, value)
   ```

3. **Implement native signature** (optional, for performance):
   ```cpp
   // In NativeCallDataCache constructor
   m_type_signature_table[typeid(MyType)] = [](builder, o) {
       // Write type-specific signature bytes
       return true;
   };
   ```

## CI System

The CI uses GitHub Actions (`.github/workflows/ci.yml`) and calls `tools/ci.py`:

```bash
python tools/ci.py configure  # CMake configure
python tools/ci.py --help     # All available commands
```

## Code Quality

```bash
# Fix formatting errors after completing a task
pre-commit run --all-files
```

If pre-commit modifies files, re-run the command to verify success.

## Dependencies

- **Minimize new dependencies** - the project has minimal external deps
- Python runtime: `requirements.txt`
- Python dev (tests): `requirements-dev.txt`
- C++ dependencies: `external/`
- Testing: pytest (Python), doctest (C++)
- Shading language: Slang

## Key Rules

1. **New Python APIs must have tests** in `slangpy/tests/`
2. **Always build before running tests**
3. **Run pre-commit after completing tasks**
4. **Use type annotations** for all Python function arguments

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
