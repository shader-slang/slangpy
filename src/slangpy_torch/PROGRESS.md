# SlangPy PyTorch Tensor Bridge - Progress Summary

## Overview
Created a native PyTorch extension (`slangpy-torch`) that provides fast C-callable functions to extract PyTorch tensor metadata from `PyObject*`, enabling ~28ns access from native code vs ~350ns for Python API calls.

## Architecture

### slangpy-torch (src/slangpy_torch/)
A standalone pip-installable PyTorch extension compiled against libtorch. Provides a C API that slangpy_ext can call via function pointers.

**Files:**
- `tensor_bridge_api.h` - C API header shared between slangpy_torch and slangpy_ext
- `torch_bridge_impl.cpp` - Implementation using libtorch internals (THPVariable_Check/Unpack)
- `setup.py` / `pyproject.toml` - Build configuration

**C API Functions (API Version 2):**
```cpp
typedef struct TensorBridgeAPI {
    int api_version;
    size_t info_struct_size;
    TensorBridge_ExtractFn extract;           // Full tensor metadata extraction (~28ns)
    TensorBridge_IsTensorFn is_tensor;        // Check if PyObject* is tensor (~10ns)
    TensorBridge_GetSignatureFn get_signature; // Fast signature string (~5ns)
    TensorBridge_GetErrorFn get_error;
    TensorBridge_GetCurrentCudaStreamFn get_current_cuda_stream;  // Get current CUDA stream
} TensorBridgeAPI;
```

**TensorBridgeInfo struct contains:**
- `data_ptr`, `shape[12]`, `strides[12]`, `ndim`
- `device_type`, `device_index`, `scalar_type`, `element_size`
- `numel`, `storage_offset`, `cuda_stream`
- `is_contiguous`, `is_cuda`, `requires_grad` flags

### slangpy_ext Integration (src/slangpy_ext/utils/)

**Files:**
- `torch_bridge.h` - TorchBridge singleton that lazy-loads slangpy_torch and caches API pointer
- `torch_bridge.cpp` - Python bindings for testing (`is_torch_bridge_available()`, `extract_torch_tensor_info()`, `is_torch_tensor()`, `extract_torch_tensor_signature()`)

**Key Features:**
- Automatic `import torch` before loading slangpy_torch (handles DLL dependencies)
- Only tries initialization once
- ABI version checking for compatibility
- Used in `slangpy.cpp` for fast tensor signature generation in hot paths

### Usage in slangpy.cpp
```cpp
// Fast path for PyTorch tensor signatures (~5ns vs ~500ns)
char buffer[64];
if (TorchBridge::instance().get_signature(obj, buffer, sizeof(buffer)) == 0) {
    *builder << buffer;
    return;
}
```

**TensorRef constructor** also uses the bridge for fast signature generation.

## Performance Results
- C API extraction: ~28ns per call
- Direct libtorch baseline: ~9.8ns
- Python API (nb::try_cast with ndarray): ~350ns
- **Speedup: ~12.6x over Python API**
- Fast signature (manual char building): ~5ns vs ~50ns (snprintf)

## Signature Format
`[torch,Dn,Sm]` where:
- `n` = number of dimensions
- `m` = scalar type code (0=uint8, 3=int32, 5=float16, 6=float32, etc.)

Uses fast manual character building (`fast_itoa`) instead of snprintf for ~10-20x speedup.

## Test Coverage
`slangpy/tests/utils/test_torch_bridge.py` - 16 tests covering:
- Bridge availability detection
- Tensor type checking
- CPU and CUDA tensor extraction
- CUDA stream extraction
- Different data types (float16/32/64, int8/16/32/64, uint8, bool)
- Non-contiguous tensors (transposed)
- Tensors with storage offset (slices)
- Tensors with gradients
- 0D, 1D, and high-dimensional tensors
- Error handling for non-tensors
- Data pointer validation
- Signature extraction

`slangpy/tests/slangpy_tests/test_torchintegration.py::test_torch_signature` - 6 tests for signature format.

## Build Commands
```powershell
# Build slangpy-torch (with release optimization + debug symbols for profiling)
cd src/slangpy_torch
pip install . --no-deps --no-build-isolation

# Build slangpy_ext
cd <repo_root>
cmake --preset windows-msvc --fresh  # if needed
cmake --build .\build\windows-msvc --config Debug --target slangpy_ext
```

## Key Implementation Details

### Why a Separate Extension?
- slangpy_ext cannot link against libtorch (would create hard dependency)
- slangpy_torch compiles against user's installed PyTorch (ABI match)
- Pure C API exported via function pointers (no C++ ABI concerns)
- Dynamic loading at runtime via Python import

### Fast Signature Generation
```cpp
// fast_itoa avoids snprintf overhead
static inline char* fast_itoa(char* p, int val) {
    if (val == 0) { *p++ = '0'; return p; }
    char tmp[12]; int i = 0;
    while (val > 0) { tmp[i++] = '0' + (val % 10); val /= 10; }
    while (i > 0) { *p++ = tmp[--i]; }
    return p;
}
```

### CUDA Stream Access
```cpp
// Get current CUDA stream for a device (added in API v2)
extern "C" void* tensor_bridge_get_current_cuda_stream(int device_index) {
    auto stream = c10::cuda::getCurrentCUDAStream(device_index);
    return stream.stream();
}
```

## Future Potential

### Native Autograd Integration
Since slangpy's core dispatch is already native, a fully native autograd path is possible:
- Implement `torch::autograd::Function<T>` in C++ within slangpy_torch
- Accept native slangpy function handles + tensor inputs
- Forward/backward call directly into slangpy's native dispatch
- No Python in the hot path

### Alternative Approaches Considered
1. **Dynamic loading (dlopen)** - Could load libtorch symbols directly, but `torch::Tensor` is C++ with complex ABI, making it fragile across PyTorch versions
2. **TensorImpl layout parsing** - Reading tensor internals directly is possible but extremely version-dependent
3. **Current approach** - Cleanest solution: separate extension compiled against user's libtorch, exports pure C API

## Files Modified/Created
```
src/slangpy_torch/           # NEW - standalone PyTorch extension
  tensor_bridge_api.h        # C API header (API v2)
  torch_bridge_impl.cpp      # Implementation with fast_itoa
  setup.py                   # Build with /Zi /DEBUG for profiling
  pyproject.toml
  README.md
  PROGRESS.md                # This file

src/slangpy_ext/
  utils/torch_bridge.h       # NEW - TorchBridge singleton
  utils/torch_bridge.cpp     # NEW - Python bindings
  utils/slangpy.h            # MODIFIED - TensorRef uses bridge, includes torch_bridge.h
  utils/slangpy.cpp          # MODIFIED - fast tensor signature path
  CMakeLists.txt             # MODIFIED - added torch_bridge files + include path
  slangpy_ext.cpp            # MODIFIED - init + export registration

slangpy/tests/
  utils/test_torch_bridge.py # NEW - comprehensive tests (16 tests)
```

## Files Modified/Created
```
src/slangpy_torch/           # NEW - standalone PyTorch extension
  tensor_bridge_api.h
  torch_bridge_impl.cpp
  setup.py
  pyproject.toml
  README.md

src/slangpy_ext/
  utils/torch_bridge.h       # NEW - TorchBridge singleton
  utils/torch_bridge.cpp     # NEW - Python bindings
  utils/slangpy.h            # MODIFIED - TensorRef uses bridge
  utils/slangpy.cpp          # MODIFIED - fast tensor signature path
  CMakeLists.txt             # MODIFIED - added torch_bridge files
  slangpy_ext.cpp            # MODIFIED - init + export registration

slangpy/tests/
  utils/test_torch_bridge.py # NEW - comprehensive tests
```
