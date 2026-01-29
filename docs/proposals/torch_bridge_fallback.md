# Proposal: Python Fallback for TorchBridge

## Summary

Create a fallback mechanism that allows the `TorchBridge` functionality to work via Python when the native `slangpy_torch` package is not installed. This ensures that PyTorch tensor integration works in all environments, with native performance when available.

## Background

Currently, the `TorchBridge` singleton (in `torch_bridge.h`) provides fast native access to PyTorch tensor metadata through the `slangpy_torch` package. This requires:
1. PyTorch to be installed
2. `slangpy_torch` to be built and installed separately (via `pip install src/slangpy_torch`)

If `slangpy_torch` is not available, many torch-related operations fail with errors like "slangpy_torch is not available."

## Goals

1. **Transparent Fallback**: When `slangpy_torch` is not present, fall back to Python-based implementations
2. **Same API**: The rest of the codebase should not need to change
3. **Testability**: Provide a way to toggle native mode on/off for testing
4. **Test Coverage**: Ensure torch integration tests run both with and without native support

## Proposed Design

### 1. Python Fallback Module

Create `slangpy/torchintegration/bridge_fallback.py` with Python implementations of all TorchBridge operations:

```python
"""
Python fallback implementations for TorchBridge operations.

These functions provide equivalent functionality to the native slangpy_torch
package using pure Python/PyTorch APIs. They are used when slangpy_torch
is not installed.
"""

import torch
from typing import Optional, Dict, Any

# PyTorch scalar type codes (from c10::ScalarType)
_SCALAR_TYPE_MAP = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.bool: 11,
    # ... more types as needed
}

def is_tensor(obj: Any) -> bool:
    """Check if object is a torch.Tensor."""
    return isinstance(obj, torch.Tensor)

def extract_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Extract tensor metadata as a dictionary.

    Equivalent to the native extract_torch_tensor_info() function.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Object is not a PyTorch tensor")

    return {
        "data_ptr": tensor.data_ptr(),
        "shape": tuple(tensor.shape),
        "strides": tuple(tensor.stride()),
        "ndim": tensor.ndim,
        "device_type": 1 if tensor.is_cuda else 0,
        "device_index": tensor.device.index if tensor.is_cuda else -1,
        "scalar_type": _SCALAR_TYPE_MAP.get(tensor.dtype, -1),
        "element_size": tensor.element_size(),
        "numel": tensor.numel(),
        "storage_offset": tensor.storage_offset(),
        "cuda_stream": _get_cuda_stream(tensor) if tensor.is_cuda else 0,
        "is_contiguous": tensor.is_contiguous(),
        "is_cuda": tensor.is_cuda,
        "requires_grad": tensor.requires_grad,
    }

def get_signature(tensor: torch.Tensor) -> str:
    """
    Get tensor signature string: "[Dn,Sm]" where n=ndim, m=scalar_type.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Object is not a PyTorch tensor")
    scalar_type = _SCALAR_TYPE_MAP.get(tensor.dtype, -1)
    return f"[D{tensor.ndim},S{scalar_type}]"

def get_current_cuda_stream(device_index: int) -> int:
    """Get the current CUDA stream pointer for a device."""
    if not torch.cuda.is_available():
        return 0
    stream = torch.cuda.current_stream(device_index)
    return stream.cuda_stream

def copy_to_buffer(tensor: torch.Tensor, dest_ptr: int, dest_size: int) -> bool:
    """
    Copy tensor data to a CUDA buffer.
    Handles non-contiguous tensors by making a contiguous copy first.
    """
    if not tensor.is_cuda:
        raise RuntimeError("Tensor must be on CUDA device")

    # Make contiguous if needed
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Use cudaMemcpy via PyTorch's internal APIs
    import ctypes
    src_ptr = tensor.data_ptr()
    byte_size = tensor.numel() * tensor.element_size()

    if byte_size > dest_size:
        raise RuntimeError(f"Destination buffer too small: {dest_size} < {byte_size}")

    # Use torch's CUDA copy mechanism
    # This requires creating a view tensor at the destination
    dest_tensor = torch.tensor([], dtype=tensor.dtype, device=tensor.device)
    dest_tensor.set_(
        torch.cuda.LongStorage.from_buffer(
            ctypes.c_void_p(dest_ptr), 'little', byte_size
        ),
        storage_offset=0,
        size=tensor.shape,
        stride=tensor.stride()
    )
    dest_tensor.copy_(tensor)
    return True

def copy_from_buffer(tensor: torch.Tensor, src_ptr: int, src_size: int) -> bool:
    """
    Copy data from a CUDA buffer to a tensor.
    Handles non-contiguous tensors.
    """
    if not tensor.is_cuda:
        raise RuntimeError("Tensor must be on CUDA device")

    byte_size = tensor.numel() * tensor.element_size()
    if byte_size > src_size:
        raise RuntimeError(f"Source buffer too small: {src_size} < {byte_size}")

    # Create source view and copy
    import ctypes
    src_tensor = torch.tensor([], dtype=tensor.dtype, device=tensor.device)
    # ... similar to copy_to_buffer
    tensor.copy_(src_tensor)
    return True

def _get_cuda_stream(tensor: torch.Tensor) -> int:
    """Get the CUDA stream pointer for the tensor's device."""
    if not tensor.is_cuda:
        return 0
    device_index = tensor.device.index or 0
    stream = torch.cuda.current_stream(device_index)
    return stream.cuda_stream
```

### 2. Modify TorchBridge C++ Class

Add a Python fallback path to `TorchBridge` in `torch_bridge.h`:

```cpp
class TorchBridge {
public:
    // ... existing methods ...

    /// Force use of Python fallback even if native is available (for testing)
    void set_force_python_fallback(bool force) {
        m_force_python_fallback = force;
        if (force) {
            // Initialize fallback functions if not already done
            if (!m_fallback_initialized) {
                init_fallback();
            }
            m_using_fallback = true;
        } else {
            // Re-try native initialization
            m_using_fallback = !m_api;
        }
    }

    /// Check if currently using Python fallback
    bool is_using_fallback() const { return m_using_fallback || m_force_python_fallback; }

    /// Extract tensor info - with fallback to Python
    bool extract(PyObject* tensor, TensorBridgeInfo& out) const {
        if (!m_force_python_fallback && m_api) {
            return m_api->extract(tensor, &out) == 0;
        }
        return python_extract(tensor, out);
    }

private:
    /// Initialize Python fallback - caches all function handles once
    void init_fallback() {
        if (m_fallback_initialized)
            return;

        try {
            // Import the fallback module once
            m_fallback_module = nb::module_::import_("slangpy.torchintegration.bridge_fallback");

            // Cache all function handles - these are looked up once and reused
            m_py_is_tensor = m_fallback_module.attr("is_tensor");
            m_py_extract_tensor_info = m_fallback_module.attr("extract_tensor_info");
            m_py_get_signature = m_fallback_module.attr("get_signature");
            m_py_get_current_cuda_stream = m_fallback_module.attr("get_current_cuda_stream");
            m_py_copy_to_buffer = m_fallback_module.attr("copy_to_buffer");
            m_py_copy_from_buffer = m_fallback_module.attr("copy_from_buffer");

            m_fallback_initialized = true;
        } catch (const std::exception& e) {
            // Log error but don't throw - fallback functions will check m_fallback_initialized
            m_fallback_initialized = false;
        }
    }

    // Python fallback implementations (use cached function handles)
    bool python_extract(PyObject* tensor, TensorBridgeInfo& out) const;
    int python_get_signature(PyObject* obj, char* buffer, size_t buffer_size) const;
    void* python_get_current_cuda_stream(int device_index) const;
    bool python_copy_to_buffer(PyObject* tensor, void* dest, size_t size) const;
    bool python_copy_from_buffer(PyObject* tensor, void* src, size_t size) const;


    // Native API state
    const TensorBridgeAPI* m_api = nullptr;
    bool m_initialized = false;

    // Fallback state
    bool m_force_python_fallback = false;
    bool m_using_fallback = false;
    bool m_fallback_initialized = false;

    // Cached Python objects (module and function handles)
    nb::object m_fallback_module;
    nb::object m_py_is_tensor;
    nb::object m_py_extract_tensor_info;
    nb::object m_py_get_signature;
    nb::object m_py_get_current_cuda_stream;
    nb::object m_py_copy_to_buffer;
    nb::object m_py_copy_from_buffer;
};
```

### 3. Implementation Strategy

The fallback implementations use cached Python function handles (obtained in `init_fallback()`):

```cpp
bool TorchBridge::python_extract(PyObject* tensor, TensorBridgeInfo& out) const {
    // Ensure fallback is initialized (function handles are cached)
    if (!m_fallback_initialized) {
        return false;
    }

    try {
        // Call cached function handle directly - no attribute lookup needed
        nb::dict info = m_py_extract_tensor_info(nb::handle(tensor));

        // Populate TensorBridgeInfo from dict
        out.data_ptr = reinterpret_cast<void*>(nb::cast<uintptr_t>(info["data_ptr"]));
        out.ndim = nb::cast<int>(info["ndim"]);
        out.numel = nb::cast<int64_t>(info["numel"]);
        out.element_size = nb::cast<int>(info["element_size"]);
        out.is_cuda = nb::cast<bool>(info["is_cuda"]);
        out.is_contiguous = nb::cast<bool>(info["is_contiguous"]);
        out.requires_grad = nb::cast<bool>(info["requires_grad"]);
        out.device_type = nb::cast<int>(info["device_type"]);
        out.device_index = nb::cast<int>(info["device_index"]);
        out.scalar_type = nb::cast<int>(info["scalar_type"]);
        out.storage_offset = nb::cast<int64_t>(info["storage_offset"]);
        out.cuda_stream = reinterpret_cast<void*>(nb::cast<uintptr_t>(info["cuda_stream"]));

        // Extract shape and strides tuples
        nb::tuple shape = nb::cast<nb::tuple>(info["shape"]);
        nb::tuple strides = nb::cast<nb::tuple>(info["strides"]);
        for (int i = 0; i < out.ndim && i < TENSOR_BRIDGE_MAX_DIMS; i++) {
            out.shape[i] = nb::cast<int64_t>(shape[i]);
            out.strides[i] = nb::cast<int64_t>(strides[i]);
        }

        return true;
    } catch (...) {
        return false;
    }
}

int TorchBridge::python_get_signature(PyObject* obj, char* buffer, size_t buffer_size) const {
    if (!m_fallback_initialized) {
        snprintf(buffer, buffer_size, "fallback not initialized");
        return -1;
    }

    try {
        // Call cached function handle
        std::string sig = nb::cast<std::string>(m_py_get_signature(nb::handle(obj)));
        snprintf(buffer, buffer_size, "%s", sig.c_str());
        return 0;
    } catch (...) {
        snprintf(buffer, buffer_size, "failed to get signature");
        return -1;
    }
}

bool TorchBridge::python_copy_to_buffer(PyObject* tensor, void* dest, size_t size) const {
    if (!m_fallback_initialized) {
        return false;
    }

    try {
        // Call cached function handle
        return nb::cast<bool>(m_py_copy_to_buffer(
            nb::handle(tensor),
            reinterpret_cast<uintptr_t>(dest),
            size
        ));
    } catch (...) {
        return false;
    }
}
```


**Performance Notes:**
- Function handles are looked up **once** during `init_fallback()` and cached as `nb::object` members
- Each call uses the cached handle directly, avoiding repeated `attr()` lookups
- This is significantly faster than looking up function names on every call

**Alternative: Direct Python C API calls**

For even more performance, one could use `PyObject_GetAttrString` to directly access tensor attributes without the Python function layer. However, this is more complex to maintain and the cached function approach is a good balance of simplicity and performance.

### 4. Expose Toggle Function to Python

In `torch_bridge.cpp`, add:

```cpp
SGL_PY_EXPORT(utils_torch_bridge)
{
    // ... existing exports ...

    m.def("set_torch_bridge_python_fallback", [](bool force) {
        TorchBridge::instance().set_force_python_fallback(force);
    }, nb::arg("force"),
    "Force use of Python fallback for torch bridge operations.\n"
    "This is primarily for testing to validate the fallback path works correctly.");

    m.def("is_torch_bridge_using_fallback", []() {
        return TorchBridge::instance().is_using_fallback();
    }, "Check if torch bridge is using Python fallback (either forced or due to slangpy_torch not being available).");
}
```

### 5. Test Integration with pytest

Create a pytest fixture that runs torch tests in both modes:

**Option A: Use pytest parametrize (simpler)**

Add to `slangpy/testing/plugin.py`:

```python
def pytest_configure(config: pytest.Config):
    # Register custom markers
    config.addinivalue_line(
        "markers",
        "torch_bridge_modes: mark test to run with both native and fallback torch bridge"
    )

def pytest_generate_tests(metafunc):
    """Generate test variants for torch_bridge_modes marker."""
    marker = metafunc.definition.get_closest_marker("torch_bridge_modes")
    if marker:
        # Parametrize with both modes
        metafunc.parametrize(
            "_torch_bridge_mode",
            ["native", "fallback"],
            indirect=True
        )

@pytest.fixture
def _torch_bridge_mode(request):
    """Fixture to set torch bridge mode for a test."""
    import slangpy

    mode = request.param
    original = slangpy.is_torch_bridge_using_fallback()

    if mode == "fallback":
        slangpy.set_torch_bridge_python_fallback(True)
    else:
        slangpy.set_torch_bridge_python_fallback(False)

    yield mode

    # Always restore original state
    slangpy.set_torch_bridge_python_fallback(original)
```

**Option B: Use a class-level fixture with autouse (more automatic)**

Create `slangpy/tests/conftest.py` or modify existing:

```python
import pytest

# Track if we need to run torch tests twice
_TORCH_BRIDGE_MODES = ["native", "fallback"]

@pytest.fixture(params=_TORCH_BRIDGE_MODES)
def torch_bridge_mode(request):
    """
    Fixture that runs the test in both native and fallback modes.
    Use this fixture in any torch-related test that should validate both paths.
    """
    import slangpy

    mode = request.param
    was_fallback = slangpy.is_torch_bridge_using_fallback()

    try:
        if mode == "fallback":
            slangpy.set_torch_bridge_python_fallback(True)
        else:
            slangpy.set_torch_bridge_python_fallback(False)

        yield mode
    finally:
        # Always restore
        slangpy.set_torch_bridge_python_fallback(was_fallback)
```

### 6. Marking Tests for Dual-Mode Execution

**Simple approach using fixture dependency:**

Tests that should run in both modes simply use the `torch_bridge_mode` fixture:

```python
# In test_torchintegration.py

class TestTorchIntegration:
    """Tests that should run with both native and fallback modes."""

    @pytest.fixture(autouse=True)
    def setup_torch_mode(self, torch_bridge_mode):
        """This makes all tests in this class run twice."""
        self.mode = torch_bridge_mode

    def test_polynomial(self, device_type: DeviceType):
        # Test runs twice: once with native, once with fallback
        module = load_test_module(device_type)
        # ... test code ...
```

**Alternative: Use a marker for selective tests:**

```python
@pytest.mark.torch_bridge_modes
def test_polynomial(device_type: DeviceType):
    # This test will be parameterized to run in both modes
    pass
```

### 7. Implementation Plan

#### Phase 1: Core Fallback Implementation
1. Create `slangpy/torchintegration/bridge_fallback.py` with Python implementations
2. Modify `TorchBridge` class to support fallback mode
3. Add `set_torch_bridge_python_fallback()` and `is_torch_bridge_using_fallback()` functions

#### Phase 2: Testing Infrastructure
1. Add `torch_bridge_mode` fixture to test plugin
2. Update `test_torch_bridge.py` to use dual-mode testing
3. Update `test_torchintegration.py` to use dual-mode testing

#### Phase 3: Validation
1. Run all torch tests with `slangpy_torch` installed (both modes should pass)
2. Uninstall `slangpy_torch` and verify fallback mode works
3. Ensure tests properly restore state after each run

## Files to Modify/Create

### New Files
- `slangpy/torchintegration/bridge_fallback.py` - Python fallback implementations

### Modified Files
- `src/slangpy_ext/utils/torch_bridge.h` - Add fallback support, toggle
- `src/slangpy_ext/utils/torch_bridge.cpp` - Add Python bindings for toggle
- `slangpy/testing/plugin.py` - Add torch_bridge_mode fixture
- `slangpy/tests/utils/test_torch_bridge.py` - Use dual-mode testing
- `slangpy/tests/slangpy_tests/test_torchintegration.py` - Use dual-mode testing

## Testing Strategy

The fixture ensures proper cleanup by:
1. Saving the original fallback state before the test
2. Setting the requested mode
3. Yielding to run the test
4. **Always** restoring the original state in a finally block

This ensures that even if a test fails or raises an exception, the state is restored.

## Performance Considerations

- The fallback path will be slower due to Python overhead
- This is acceptable because:
  - The fallback is meant for environments where `slangpy_torch` can't be built
  - Performance-critical code should use the native path
  - Tests run faster with native, fallback is just for validation

## Open Questions

1. Should the fallback work for CUDA memory copy operations? These are more complex and may require `cupy` or similar.
2. Should we emit a warning when falling back to Python mode in production use?
3. Should the `--device-types` pytest option interact with torch bridge mode selection?

## Appendix: Current TorchBridge Usage

The TorchBridge is used in these locations:
- `slangpytorchtensor.cpp`: Signature generation, shape extraction, tensor info, CUDA copies
- `slangpy.cpp`: Signature generation for cache keys
- `torch_bridge.cpp`: Python-exposed utility functions

All these need to work with the fallback path.
