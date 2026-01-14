// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// tensor_bridge_api.h
//
// This header defines the C API shared between:
// 1. slangpy_torch (compiled with libtorch) - implements the functions
// 2. slangpy_ext (no libtorch dependency) - calls the functions via function pointers
//
// USAGE IN SLANGPY_EXT:
// 1. Import slangpy_torch in Python
// 2. Get the function pointer via get_api_ptr()
// 3. Cast to TensorBridgeAPI* and call directly from C++ with PyObject*
//
// This allows your native code to extract PyTorch tensor data with ~28ns
// overhead instead of ~350ns for Python API calls.

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum dimensions we support
#define TENSOR_BRIDGE_MAX_DIMS 12

// Device type codes (matching c10::DeviceType)
#define TENSOR_BRIDGE_DEVICE_CPU 0
#define TENSOR_BRIDGE_DEVICE_CUDA 1

// The C-compatible struct containing all tensor metadata
// This struct is POD and can be safely memcpy'd
typedef struct TensorBridgeInfo {
    // Data pointer (GPU or CPU memory)
    void* data_ptr;

    // Shape and strides (in elements, not bytes)
    // Stored inline to avoid pointer chasing
    int64_t shape[TENSOR_BRIDGE_MAX_DIMS];
    int64_t strides[TENSOR_BRIDGE_MAX_DIMS];

    // Number of dimensions (0 for scalar)
    int32_t ndim;

    // Device info
    int32_t device_type;  // c10::DeviceType value
    int32_t device_index; // GPU index, or -1 for CPU

    // Data type (c10::ScalarType value)
    int32_t scalar_type;

    // Element size in bytes
    int32_t element_size;

    // Total number of elements
    int64_t numel;

    // Storage offset (for views)
    int64_t storage_offset;

    // Flags
    uint32_t is_contiguous : 1;
    uint32_t is_cuda : 1;
    uint32_t requires_grad : 1;
    uint32_t _padding : 29;

} TensorBridgeInfo;

// ============================================================================
// Function pointer types for the C API
// These are the functions that slangpy_torch exports
// ============================================================================

// Extract tensor info from a PyObject* (must be a torch.Tensor)
// Returns 0 on success, non-zero on error
// The 'out' struct is filled with the tensor metadata
typedef int (*TensorBridge_ExtractFn)(void* py_tensor_obj, TensorBridgeInfo* out);

// Check if a PyObject* is a torch.Tensor
// Returns 1 if true, 0 if false
typedef int (*TensorBridge_IsTensorFn)(void* py_tensor_obj);

// Get the last error message (if any function returned non-zero)
// Returns a pointer to a static thread-local buffer
typedef const char* (*TensorBridge_GetErrorFn)(void);

// ============================================================================
// Version info for ABI compatibility checking
// ============================================================================
#define TENSOR_BRIDGE_API_VERSION 1

typedef struct TensorBridgeAPI {
    int api_version;
    size_t info_struct_size;

    TensorBridge_ExtractFn extract;
    TensorBridge_IsTensorFn is_tensor;
    TensorBridge_GetErrorFn get_error;
} TensorBridgeAPI;

// Function to get the API struct (exported by the bridge module)
typedef const TensorBridgeAPI* (*TensorBridge_GetAPIFn)(void);

#ifdef __cplusplus
}
#endif
