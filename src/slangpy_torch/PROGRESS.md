# SlangPy PyTorch Tensor Bridge - Progress Summary

## Overview
Created a native PyTorch extension (`slangpy-torch`) that provides fast C-callable functions to extract PyTorch tensor metadata from `PyObject*`, enabling ~28ns access from native code vs ~350ns for Python API calls.

## Phase 1: Fast Tensor Metadata Extraction (Complete)

### slangpy-torch (src/slangpy_torch/)
A standalone pip-installable PyTorch extension compiled against libtorch. Provides a C API that slangpy_ext can call via function pointers.

**Files:**
- `tensor_bridge_api.h` - C API header shared between slangpy_torch and slangpy_ext
- `torch_bridge_impl.cpp` - Implementation using libtorch internals (THPVariable_Check/Unpack)
- `setup.py` / `pyproject.toml` - Build configuration

**C API Functions (API Version 3):**
```cpp
typedef struct TensorBridgeAPI {
    int api_version;
    size_t info_struct_size;
    TensorBridge_ExtractFn extract;           // Full tensor metadata extraction (~28ns)
    TensorBridge_IsTensorFn is_tensor;        // Check if PyObject* is tensor (~10ns)
    TensorBridge_GetSignatureFn get_signature; // Fast signature string (~5ns)
    TensorBridge_GetErrorFn get_error;
    TensorBridge_GetCurrentCudaStreamFn get_current_cuda_stream;  // Get current CUDA stream
    TensorBridge_CopyToBufferFn copy_to_buffer;    // Copy tensor data to CUDA pointer
    TensorBridge_CopyFromBufferFn copy_from_buffer; // Copy data from CUDA pointer to tensor
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
- `torch_bridge.cpp` - Python bindings for testing + tensor copy utilities

**Key Features:**
- Automatic `import torch` before loading slangpy_torch (handles DLL dependencies)
- Only tries initialization once
- ABI version checking for compatibility
- Used in `slangpy.cpp` for fast tensor signature generation in hot paths

## Phase 2: D3D12/Vulkan Support via Interop Buffers (Complete)

### Problem
PyTorch tensors on CUDA can be directly accessed via device pointers on CUDA backends, but D3D12/Vulkan cannot directly access CUDA memory.

### Solution
Use `sgl::cuda::InteropBuffer` for shared memory between CUDA and D3D12/Vulkan:
1. Create buffer with `BufferUsage::shared` flag
2. Use `Buffer::cuda_memory()` to get CUDA-accessible pointer
3. Copy tensor data to/from the interop buffer via PyTorch operations

### API v3 Additions
```cpp
// Copy tensor data to a CUDA pointer (for upload to GPU buffer)
int tensor_bridge_copy_to_buffer(PyObject* tensor, void* cuda_ptr, size_t size);

// Copy data from a CUDA pointer to a tensor (for readback from GPU buffer)
int tensor_bridge_copy_from_buffer(PyObject* tensor, void* cuda_ptr, size_t size);
```

Key implementation details:
- Uses `torch::NoGradGuard` to allow in-place operations on tensors with `requires_grad=True`
- Handles non-contiguous tensors correctly via `torch::from_blob` + `copy_`
- Works with PyTorch's current CUDA stream

### Buffer Binding Fix for D3D12/Vulkan
Fixed `NativeTorchTensorMarshall::write_torch_tensor_fields` to use `shader_object->set_buffer()` for D3D12/Vulkan backends instead of writing device address directly. This is required because D3D12/Vulkan need proper buffer resource binding.

## Phase 3: TensorRef Removal (Complete)

### Rationale
The `TensorRef` wrapper class was adding complexity without clear benefit:
- Required explicit wrapping of tensors
- Complicated the autograd integration
- Made the API less intuitive

### Changes Made

**Native code removed:**
- `TensorRef` class from `slangpy.h`
- `TensorRef` bindings from `slangpy.cpp`
- `write_torch_tensor_ref` from `slangpytensor.cpp`

**Python code removed:**
- `TensorRefMarshall` class from `torchtensormarshall.py`
- `TensorRef` handling from `calldata.py`
- Updated `autogradhook.py` to remove TensorRef dependencies

**Result:**
- Raw `torch.Tensor` objects now work directly with SlangPy
- 18/18 `test_add_tensors` tests pass across CUDA, D3D12, and Vulkan
- Simpler, more intuitive API

## Phase 4: Autograd Refactoring (In Progress)

### Current State
The autograd system is being refactored to work without `TensorRef`:

1. **`torch_autograd` flag**: Already tracked in `NativeCallData` based on whether any tensor has `requires_grad=True`

2. **`TorchAutoGradHook`**: Refactored to use `TrackedTensor` dataclass instead of `TensorRef`:
```python
@dataclass
class TrackedTensor:
    tensor: torch.Tensor
    access: AccessType  # read, write, readwrite
    arg_name: str       # for debugging
```

3. **Access pattern tracking**: Uses `NativeBoundVariableRuntime.access` property to determine tensor roles (input/output)

### Next Steps
- Implement tensor tracking during `write_shader_cursor_pre_dispatch`
- Complete `TorchAutoGradHook.backward()` implementation
- Wire up autograd hook in `_py_torch_autograd_call`

## Performance Results
- C API extraction: ~28ns per call
- Direct libtorch baseline: ~9.8ns
- Python API (nb::try_cast with ndarray): ~350ns
- **Speedup: ~12.6x over Python API**
- Fast signature (manual char building): ~5ns vs ~50ns (snprintf)

## Test Coverage
- `slangpy/tests/utils/test_torch_bridge.py` - 16 tests for bridge API
- `slangpy/tests/slangpy_tests/test_torchintegration.py::test_torch_signature` - 6 tests for signature format
- `slangpy/tests/slangpy_tests/test_torchintegration.py::test_add_tensors` - 18 tests across CUDA/D3D12/Vulkan
- `slangpy/tests/slangpy_tests/test_torchintegration.py::test_buffer_copy_*` - Round-trip buffer copy tests

## Build Commands
```powershell
# Build slangpy-torch
cd src/slangpy_torch
$env.DISTUTILS_USE_SDK=1 && pip install . --no-deps --no-build-isolation

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

### Interop Buffer Pattern (D3D12/Vulkan)
```cpp
// Create shared buffer accessible from both CUDA and D3D12/Vulkan
Buffer buffer = device.create_buffer(
    size=tensor.numel() * tensor.element_size(),
    usage=BufferUsage::shared | BufferUsage::unordered_access
);

// Get CUDA pointer for the buffer
void* cuda_ptr = buffer.cuda_memory();

// Copy tensor data to buffer (via slangpy_torch C API)
tensor_bridge_copy_to_buffer(tensor_pyobj, cuda_ptr, size);

// After dispatch, copy results back
tensor_bridge_copy_from_buffer(tensor_pyobj, cuda_ptr, size);
```

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

### Signature Format
`[torch,Dn,Sm]` where:
- `n` = number of dimensions
- `m` = scalar type code (0=uint8, 3=int32, 5=float16, 6=float32, etc.)

## Files Modified/Created

### New Files
```
src/slangpy_torch/           # Standalone PyTorch extension
  tensor_bridge_api.h        # C API header (API v3)
  torch_bridge_impl.cpp      # Implementation with copy functions
  setup.py                   # Build configuration
  pyproject.toml
  README.md
  PROGRESS.md                # This file

src/slangpy_ext/utils/
  torch_bridge.h             # TorchBridge singleton
  torch_bridge.cpp           # Python bindings + copy utilities
  slangpytorchtensor.h       # NativeTorchTensorMarshall
  slangpytorchtensor.cpp     # PyTorch tensor marshalling

slangpy/tests/utils/
  test_torch_bridge.py       # Bridge API tests (16 tests)
```

### Modified Files
```
src/slangpy_ext/utils/
  slangpy.h                  # Removed TensorRef class
  slangpy.cpp                # Removed TensorRef bindings, fast signature path
  slangpytensor.h            # Made CachedOffsets public for reuse
  slangpytensor.cpp          # Shared offset extraction methods
  CMakeLists.txt             # Added torch_bridge files

slangpy/
  __init__.py                # Export copy utilities
  core/calldata.py           # Removed TensorRef handling, simplified torch path
  torchintegration/
    torchtensormarshall.py   # Removed TensorRefMarshall
    autogradhook.py          # Refactored for TrackedTensor approach
    detection.py             # Tensor detection utilities
```

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Python Layer                                 │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │ torch.Tensor│──│TorchTensorMarshall│──│ TorchAutoGradHook     │ │
│  └─────────────┘  └──────────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Native Layer (slangpy_ext)                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  TorchBridge    │──│NativeTorchTensor │──│   CallContext     │  │
│  │  (C API caller) │  │    Marshall      │  │  (dispatch)       │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         slangpy_torch Extension                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  C API (tensor_bridge_api.h)                                  │  │
│  │  - extract()          - Fast tensor metadata (~28ns)          │  │
│  │  - get_signature()    - Fast signature (~5ns)                 │  │
│  │  - copy_to_buffer()   - Tensor → CUDA pointer                 │  │
│  │  - copy_from_buffer() - CUDA pointer → Tensor                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```
