// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// slangpy_torch - Fast PyTorch tensor access for native code
//
// This extension provides C-callable functions to extract PyTorch tensor
// metadata from a raw PyObject*, enabling ~28ns access from native code
// vs ~350ns for Python API calls.

#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>
#include <c10/cuda/CUDAStream.h>
#include "tensor_bridge_api.h"

#include <chrono>
#include <cstring>

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" int tensor_bridge_is_tensor(void* py_obj)
{
    if (!py_obj)
        return 0;
    return THPVariable_Check(static_cast<PyObject*>(py_obj)) ? 1 : 0;
}

extern "C" int tensor_bridge_extract(
    void* py_obj,
    TensorBridgeInfo* out,
    int64_t* shape_buffer,
    int64_t* strides_buffer,
    int32_t buffer_capacity
)
{
    if (!py_obj)
        return TENSOR_BRIDGE_ERROR_NULL_OBJECT;
    if (!out)
        return TENSOR_BRIDGE_ERROR_NULL_OUTPUT;

    PyObject* obj = static_cast<PyObject*>(py_obj);
    if (!THPVariable_Check(obj))
        return TENSOR_BRIDGE_ERROR_NOT_TENSOR;

    const torch::Tensor& tensor = THPVariable_Unpack(obj);

    out->data_ptr = tensor.data_ptr();
    out->ndim = static_cast<int32_t>(tensor.dim());
    out->buffer_capacity = buffer_capacity;

    // Set shape/strides pointers based on buffer capacity
    if (buffer_capacity >= out->ndim && shape_buffer && strides_buffer) {
        out->shape = shape_buffer;
        out->strides = strides_buffer;

        auto sizes = tensor.sizes();
        auto strd = tensor.strides();
        for (int i = 0; i < out->ndim; ++i) {
            out->shape[i] = sizes[i];
            out->strides[i] = strd[i];
        }
    } else {
        // Buffer too small - caller needs to retry with larger buffers
        // We still populate all other fields so caller knows what size is needed
        out->shape = nullptr;
        out->strides = nullptr;
    }

    auto device = tensor.device();
    out->device_type = static_cast<int32_t>(device.type());
    out->device_index = device.has_index() ? static_cast<int32_t>(device.index()) : -1;
    out->is_cuda = device.is_cuda() ? 1 : 0;
    out->scalar_type = static_cast<int32_t>(tensor.scalar_type());
    out->element_size = static_cast<int32_t>(tensor.element_size());
    out->numel = tensor.numel();
    out->storage_offset = tensor.storage_offset();
    out->is_contiguous = tensor.is_contiguous() ? 1 : 0;
    out->requires_grad = tensor.requires_grad() ? 1 : 0;
    out->_padding = 0;

    // Get CUDA stream for CUDA tensors
    if (device.is_cuda()) {
        auto stream = c10::cuda::getCurrentCUDAStream(device.index());
        out->cuda_stream = stream.stream();
    } else {
        out->cuda_stream = nullptr;
    }

    return 0;
}

// Fast integer to string - returns pointer to end of written chars
static inline char* fast_itoa(char* p, int val)
{
    if (val == 0) {
        *p++ = '0';
        return p;
    }

    // Build digits in reverse, then flip
    char tmp[12];
    int i = 0;
    while (val > 0) {
        tmp[i++] = '0' + (val % 10);
        val /= 10;
    }
    while (i > 0) {
        *p++ = tmp[--i];
    }
    return p;
}

// Fast signature extraction - returns TENSOR_BRIDGE_SUCCESS on success
extern "C" int tensor_bridge_get_signature(void* py_obj, char* buffer, size_t buffer_size)
{
    if (!py_obj)
        return TENSOR_BRIDGE_ERROR_NULL_OBJECT;

    PyObject* obj = static_cast<PyObject*>(py_obj);
    if (!THPVariable_Check(obj))
        return TENSOR_BRIDGE_ERROR_NOT_TENSOR;

    const torch::Tensor& tensor = THPVariable_Unpack(obj);

    int ndim = static_cast<int>(tensor.dim());
    int scalar_type = static_cast<int>(tensor.scalar_type());

    // Format: "[Dn,Sm]" - compatible format, no snprintf
    char* p = buffer;
    *p++ = '[';
    *p++ = 'D';
    p = fast_itoa(p, ndim);
    *p++ = ',';
    *p++ = 'S';
    p = fast_itoa(p, scalar_type);
    *p++ = ']';
    *p = '\0';

    return TENSOR_BRIDGE_SUCCESS;
}

// Get the current CUDA stream for a given device index
extern "C" void* tensor_bridge_get_current_cuda_stream(int device_index)
{
    try {
        auto stream = c10::cuda::getCurrentCUDAStream(device_index);
        return stream.stream();
    } catch (...) {
        return nullptr;
    }
}

// Copy tensor data to a contiguous CUDA buffer
// This handles non-contiguous tensors by using PyTorch's copy mechanism.
extern "C" int tensor_bridge_copy_to_buffer(void* py_obj, void* dest_cuda_ptr, size_t dest_size)
{
    if (!py_obj)
        return TENSOR_BRIDGE_ERROR_NULL_OBJECT;
    if (!dest_cuda_ptr)
        return TENSOR_BRIDGE_ERROR_NULL_OUTPUT;

    PyObject* obj = static_cast<PyObject*>(py_obj);
    if (!THPVariable_Check(obj))
        return TENSOR_BRIDGE_ERROR_NOT_TENSOR;

    try {
        const torch::Tensor& src_tensor = THPVariable_Unpack(obj);

        // Verify source is a CUDA tensor
        if (!src_tensor.is_cuda())
            return TENSOR_BRIDGE_ERROR_NOT_CUDA;

        // Verify size matches
        size_t tensor_bytes = src_tensor.numel() * src_tensor.element_size();
        if (tensor_bytes > dest_size)
            return TENSOR_BRIDGE_ERROR_BUFFER_TOO_SMALL;

        // Create a tensor view over the destination buffer with same dtype
        // This creates a contiguous tensor backed by the destination memory
        auto options = torch::TensorOptions().dtype(src_tensor.dtype()).device(src_tensor.device());

        // Create a flat tensor view over dest memory
        torch::Tensor dest_tensor
            = torch::from_blob(dest_cuda_ptr, {static_cast<int64_t>(src_tensor.numel())}, options);

        // Reshape to match source shape for copy
        dest_tensor = dest_tensor.view(src_tensor.sizes());

        // Use PyTorch's copy_ which handles non-contiguous source tensors
        dest_tensor.copy_(src_tensor);

        return TENSOR_BRIDGE_SUCCESS;
    } catch (...) {
        return TENSOR_BRIDGE_ERROR_EXCEPTION;
    }
}

// Copy data from a contiguous CUDA buffer back to a tensor
// This handles non-contiguous destination tensors by using PyTorch's copy mechanism.
extern "C" int tensor_bridge_copy_from_buffer(void* py_obj, void* src_cuda_ptr, size_t src_size)
{
    if (!py_obj)
        return TENSOR_BRIDGE_ERROR_NULL_OBJECT;
    if (!src_cuda_ptr)
        return TENSOR_BRIDGE_ERROR_NULL_OUTPUT;

    PyObject* obj = static_cast<PyObject*>(py_obj);
    if (!THPVariable_Check(obj))
        return TENSOR_BRIDGE_ERROR_NOT_TENSOR;

    try {
        // THPVariable_Unpack returns const, but we need to modify via copy_
        // This is safe because copy_ modifies the underlying storage, not the tensor object
        const torch::Tensor& dest_tensor = THPVariable_Unpack(obj);

        // Verify destination is a CUDA tensor
        if (!dest_tensor.is_cuda())
            return TENSOR_BRIDGE_ERROR_NOT_CUDA;

        // Verify size matches
        size_t tensor_bytes = dest_tensor.numel() * dest_tensor.element_size();
        if (tensor_bytes > src_size)
            return TENSOR_BRIDGE_ERROR_BUFFER_TOO_SMALL;

        // Create a tensor view over the source buffer with same dtype
        auto options = torch::TensorOptions().dtype(dest_tensor.dtype()).device(dest_tensor.device());

        // Create a flat tensor view over src memory
        torch::Tensor src_tensor = torch::from_blob(src_cuda_ptr, {static_cast<int64_t>(dest_tensor.numel())}, options);

        // Reshape to match destination shape for copy
        src_tensor = src_tensor.view(dest_tensor.sizes());

        // Use PyTorch's copy_ which handles non-contiguous destination tensors
        // We need to const_cast here because THPVariable_Unpack returns const,
        // but copy_ is safe as it only modifies the underlying storage
        // Use torch::NoGradGuard to allow in-place operations on tensors with requires_grad=True
        {
            torch::NoGradGuard no_grad;
            const_cast<torch::Tensor&>(dest_tensor).copy_(src_tensor);
        }

        return TENSOR_BRIDGE_SUCCESS;
    } catch (...) {
        return TENSOR_BRIDGE_ERROR_EXCEPTION;
    }
}

static const TensorBridgeAPI g_api
    = {TENSOR_BRIDGE_API_VERSION,
       sizeof(TensorBridgeInfo),
       tensor_bridge_extract,
       tensor_bridge_is_tensor,
       tensor_bridge_get_signature,
       tensor_bridge_get_current_cuda_stream,
       tensor_bridge_copy_to_buffer,
       tensor_bridge_copy_from_buffer};

extern "C" const TensorBridgeAPI* tensor_bridge_get_api(void)
{
    return &g_api;
}

// ============================================================================
// Python Module
// ============================================================================

PYBIND11_MODULE(slangpy_torch, m)
{
    m.doc() = "Fast PyTorch tensor access for slangpy";

    // Core API - function pointers for native code
    m.def(
        "get_api_ptr",
        []() -> uintptr_t
        {
            return reinterpret_cast<uintptr_t>(tensor_bridge_get_api());
        },
        "Get pointer to the TensorBridgeAPI struct"
    );

    m.attr("API_VERSION") = TENSOR_BRIDGE_API_VERSION;
    m.attr("INFO_STRUCT_SIZE") = sizeof(TensorBridgeInfo);

    // Benchmark function for performance validation
    m.def(
        "benchmark_c_api",
        [](py::object tensor_obj, int iterations) -> py::dict
        {
            PyObject* raw_ptr = tensor_obj.ptr();
            TensorBridgeInfo info;
            int64_t shape_buffer[TENSOR_BRIDGE_DEFAULT_DIMS];
            int64_t strides_buffer[TENSOR_BRIDGE_DEFAULT_DIMS];

            for (int i = 0; i < 1000; ++i)
                tensor_bridge_extract(raw_ptr, &info, shape_buffer, strides_buffer, TENSOR_BRIDGE_DEFAULT_DIMS);

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i)
                tensor_bridge_extract(raw_ptr, &info, shape_buffer, strides_buffer, TENSOR_BRIDGE_DEFAULT_DIMS);
            auto end = std::chrono::high_resolution_clock::now();

            double total_ns = std::chrono::duration<double, std::nano>(end - start).count();

            py::dict result;
            result["per_call_ns"] = total_ns / iterations;
            result["data_ptr"] = reinterpret_cast<uintptr_t>(info.data_ptr);
            result["ndim"] = info.ndim;
            result["numel"] = info.numel;
            return result;
        },
        py::arg("tensor"),
        py::arg("iterations") = 1000000
    );

    m.def(
        "benchmark_libtorch",
        [](const torch::Tensor& tensor, int iterations) -> py::dict
        {
            void* ptr;
            int64_t numel;

            for (int i = 0; i < 1000; ++i) {
                ptr = tensor.data_ptr();
                (void)tensor.sizes();
                (void)tensor.strides();
                numel = tensor.numel();
            }

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                ptr = tensor.data_ptr();
                (void)tensor.sizes();
                (void)tensor.strides();
                numel = tensor.numel();
            }
            auto end = std::chrono::high_resolution_clock::now();

            double total_ns = std::chrono::duration<double, std::nano>(end - start).count();

            py::dict result;
            result["per_call_ns"] = total_ns / iterations;
            result["data_ptr"] = reinterpret_cast<uintptr_t>(ptr);
            result["numel"] = numel;
            return result;
        },
        py::arg("tensor"),
        py::arg("iterations") = 1000000
    );
}
