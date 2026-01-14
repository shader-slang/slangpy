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

**C API Functions:**
```cpp
typedef struct TensorBridgeAPI {
    int api_version;
    size_t info_struct_size;
    TensorBridge_ExtractFn extract;      // Full tensor metadata extraction (~28ns)
    TensorBridge_IsTensorFn is_tensor;   // Check if PyObject* is tensor (~10ns)
    TensorBridge_GetSignatureFn get_signature;  // Fast signature string (~5ns)
    TensorBridge_GetErrorFn get_error;
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
- `torch_bridge.cpp` - Python bindings for testing (`is_torch_bridge_available()`, `extract_torch_tensor_info()`, `is_torch_tensor()`)

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

## Signature Format
`[torch,Dn,Sm]` where:
- `n` = number of dimensions
- `m` = scalar type code (0=uint8, 3=int32, 5=float16, 6=float32, etc.)

Uses fast manual character building instead of snprintf for ~10-20x speedup.

## Test Coverage
`slangpy/tests/utils/test_torch_bridge.py` - 15+ tests covering:
- Bridge availability detection
- Tensor type checking
- CPU and CUDA tensor extraction
- CUDA stream extraction
- Different data types
- Non-contiguous tensors, slices, gradients
- 0D, 1D, high-dimensional tensors
- Error handling

`slangpy/tests/slangpy_tests/test_torchintegration.py::test_torch_signature` - 6 tests for signature format.

## Build Commands
```powershell
# Build slangpy-torch (with release optimization + debug symbols)
cd src/slangpy_torch
pip install . --no-deps --no-build-isolation

# Build slangpy_ext
cd <repo_root>
cmake --preset windows-msvc --fresh  # if needed
cmake --build .\build\windows-msvc --config Debug --target slangpy_ext
```

## Future Potential
The same pattern could be extended for:
1. **Native autograd integration** - Implement `torch::autograd::Function<T>` in C++ to eliminate Python from forward/backward passes entirely (slangpy's core dispatch is already native)
2. **CUDA stream management** - Direct stream access already available via `TensorBridgeInfo.cuda_stream`

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
