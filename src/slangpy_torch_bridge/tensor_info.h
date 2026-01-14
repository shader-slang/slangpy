// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// tensor_info.h
// This header defines the C-compatible struct that is shared between:
// 1. slangpy_torch_bridge (compiled with libtorch) - populates the struct
// 2. slangpy_ext (no libtorch dependency) - reads the struct
//
// The struct is allocated and owned by Python, and a raw pointer is passed
// to native code, allowing zero-overhead access to PyTorch tensor metadata.

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum dimensions we support (PyTorch tensors rarely exceed 8 dims)
#define TENSOR_INFO_MAX_DIMS 8

// Device type codes (matching torch device types)
typedef enum {
    TENSOR_DEVICE_CPU = 0,
    TENSOR_DEVICE_CUDA = 1,
    // Add others as needed
} TensorDeviceType;

// Data type codes (simplified, matching common use cases)
typedef enum {
    TENSOR_DTYPE_FLOAT16 = 0,
    TENSOR_DTYPE_FLOAT32 = 1,
    TENSOR_DTYPE_FLOAT64 = 2,
    TENSOR_DTYPE_INT8 = 3,
    TENSOR_DTYPE_INT16 = 4,
    TENSOR_DTYPE_INT32 = 5,
    TENSOR_DTYPE_INT64 = 6,
    TENSOR_DTYPE_UINT8 = 7,
    TENSOR_DTYPE_BOOL = 8,
    TENSOR_DTYPE_BFLOAT16 = 9,
    TENSOR_DTYPE_COMPLEX64 = 10,
    TENSOR_DTYPE_COMPLEX128 = 11,
    TENSOR_DTYPE_UNKNOWN = 255,
} TensorDType;

// The core struct - this is what gets passed to native code
// It's a plain C struct with no pointers to Python objects
typedef struct {
    // Data pointer (GPU or CPU memory)
    void* data_ptr;

    // Shape and strides (in elements, not bytes)
    int64_t shape[TENSOR_INFO_MAX_DIMS];
    int64_t strides[TENSOR_INFO_MAX_DIMS];

    // Number of dimensions
    int32_t ndim;

    // Device info
    int32_t device_type; // TensorDeviceType
    int32_t device_index;

    // Data type info
    int32_t dtype;        // TensorDType
    int32_t element_size; // Size in bytes of each element

    // Total number of elements
    int64_t numel;

    // Flags
    uint32_t is_contiguous : 1;
    uint32_t is_cuda : 1;
    uint32_t requires_grad : 1;
    uint32_t _reserved : 29;

    // Storage offset (for views)
    int64_t storage_offset;

} TensorInfo;

// Function pointer type for the extraction function
// This allows slangpy_ext to call the bridge without linking to it
typedef void (*ExtractTensorInfoFn)(void* tensor_capsule, TensorInfo* out_info);

#ifdef __cplusplus
}
#endif
