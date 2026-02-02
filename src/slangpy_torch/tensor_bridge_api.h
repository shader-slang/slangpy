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

// Default inline storage size for common cases (covers 99%+ of tensors)
// Callers needing more dimensions can provide larger buffers
#define TENSOR_BRIDGE_DEFAULT_DIMS 16

// Device type codes (matching c10::DeviceType)
#define TENSOR_BRIDGE_DEVICE_CPU 0
#define TENSOR_BRIDGE_DEVICE_CUDA 1

// ============================================================================
// Result codes for tensor bridge API functions
// ============================================================================
typedef enum TensorBridgeResult {
    TENSOR_BRIDGE_SUCCESS = 0,                 // Operation completed successfully
    TENSOR_BRIDGE_ERROR_NULL_OBJECT = -1,      // PyObject pointer is null
    TENSOR_BRIDGE_ERROR_NULL_OUTPUT = -2,      // Output pointer is null
    TENSOR_BRIDGE_ERROR_NOT_TENSOR = -3,       // PyObject is not a torch.Tensor
    TENSOR_BRIDGE_ERROR_NOT_CUDA = -4,         // Tensor is not on CUDA device
    TENSOR_BRIDGE_ERROR_BUFFER_TOO_SMALL = -5, // Destination/source buffer too small
    TENSOR_BRIDGE_ERROR_EXCEPTION = -6,        // C++ exception occurred
    TENSOR_BRIDGE_ERROR_UNKNOWN = -7,          // Unknown error occurred
} TensorBridgeResult;

// The C-compatible struct containing all tensor metadata
// Shape and strides are stored via pointers to caller-provided buffers,
// allowing support for arbitrary dimension counts without allocation.
typedef struct TensorBridgeInfo {
    // Data pointer (GPU or CPU memory)
    void* data_ptr;

    // Shape and strides (in elements, not bytes)
    // These point to caller-provided buffers. Will be set to the provided
    // buffers if they have sufficient capacity, nullptr otherwise.
    // Always check ndim and buffer_capacity before accessing.
    int64_t* shape;
    int64_t* strides;

    // Number of dimensions (0 for scalar)
    // This is always set, even if shape/strides pointers are null
    int32_t ndim;

    // Capacity of the shape/strides buffers provided by caller
    // If ndim > buffer_capacity, shape/strides will be nullptr
    int32_t buffer_capacity;

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
// Caller provides buffers for shape and strides data.
// Parameters:
//   py_tensor_obj: PyObject* that must be a torch.Tensor
//   out: Output structure to populate. shape/strides pointers will be set
//        to shape_buffer/strides_buffer if buffer_capacity >= ndim
//   shape_buffer: Caller-provided buffer for shape data (may be NULL)
//   strides_buffer: Caller-provided buffer for strides data (may be NULL)
//   buffer_capacity: Number of elements the buffers can hold
// Returns: TENSOR_BRIDGE_SUCCESS (0) on success, or a negative TensorBridgeResult on error
// Note: out->ndim is always set. If ndim > buffer_capacity, shape/strides
//       will be nullptr and caller should retry with larger buffers.
typedef int (*TensorBridge_ExtractFn)(
    void* py_tensor_obj,
    TensorBridgeInfo* out,
    int64_t* shape_buffer,
    int64_t* strides_buffer,
    int32_t buffer_capacity
);

// Check if a PyObject* is a torch.Tensor
// Returns 1 if true, 0 if false
typedef int (*TensorBridge_IsTensorFn)(void* py_tensor_obj);

// Get a minimal signature for a PyObject* if it's a torch.Tensor
// Parameters:
//   py_tensor_obj: PyObject* that should be a torch.Tensor
//   buffer: Output buffer for signature string
//   buffer_size: Size of output buffer in bytes
// Returns: TENSOR_BRIDGE_SUCCESS (0) on success, or a negative TensorBridgeResult on error
// Format: "[Dn,Sm]" where n=ndim, m=scalar_type
// This is faster than full extraction when only signature is needed (~15ns)
typedef int (*TensorBridge_GetSignatureFn)(void* py_tensor_obj, char* buffer, size_t buffer_size);

// Get the current CUDA stream for a given device index
// Returns the cudaStream_t pointer, or nullptr if CUDA is not available
// device_index: the CUDA device index (0, 1, 2, etc.)
typedef void* (*TensorBridge_GetCurrentCudaStreamFn)(int device_index);

// Copy tensor data to a contiguous CUDA buffer
// This handles non-contiguous tensors by using PyTorch's copy mechanism.
// dest_cuda_ptr: destination CUDA pointer (e.g., from interop buffer mapped memory)
// dest_size: size in bytes of destination buffer
// Returns: TENSOR_BRIDGE_SUCCESS (0) on success, or a negative TensorBridgeResult on error
typedef int (*TensorBridge_CopyToBufferFn)(void* py_tensor_obj, void* dest_cuda_ptr, size_t dest_size);

// Copy data from a contiguous CUDA buffer back to a tensor
// This handles non-contiguous tensors by using PyTorch's copy mechanism.
// src_cuda_ptr: source CUDA pointer (e.g., from interop buffer mapped memory)
// src_size: size in bytes of source buffer
// Returns: TENSOR_BRIDGE_SUCCESS (0) on success, or a negative TensorBridgeResult on error
typedef int (*TensorBridge_CopyFromBufferFn)(void* py_tensor_obj, void* src_cuda_ptr, size_t src_size);

// ============================================================================
// Version info for ABI compatibility checking
// ============================================================================
#define TENSOR_BRIDGE_API_VERSION 5

typedef struct TensorBridgeAPI {
    int api_version;
    size_t info_struct_size;

    TensorBridge_ExtractFn extract;
    TensorBridge_IsTensorFn is_tensor;
    TensorBridge_GetSignatureFn get_signature;
    TensorBridge_GetCurrentCudaStreamFn get_current_cuda_stream;
    TensorBridge_CopyToBufferFn copy_to_buffer;
    TensorBridge_CopyFromBufferFn copy_from_buffer;
} TensorBridgeAPI;

// Function to get the API struct (exported by the bridge module)
typedef const TensorBridgeAPI* (*TensorBridge_GetAPIFn)(void);

#ifdef __cplusplus
}
#endif
