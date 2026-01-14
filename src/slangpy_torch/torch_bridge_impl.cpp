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

static thread_local char g_error_buffer[256] = {0};

static void set_error(const char* msg)
{
    strncpy(g_error_buffer, msg, sizeof(g_error_buffer) - 1);
    g_error_buffer[sizeof(g_error_buffer) - 1] = '\0';
}

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" int tensor_bridge_is_tensor(void* py_obj)
{
    if (!py_obj)
        return 0;
    return THPVariable_Check(static_cast<PyObject*>(py_obj)) ? 1 : 0;
}

extern "C" int tensor_bridge_extract(void* py_obj, TensorBridgeInfo* out)
{
    if (!py_obj) {
        set_error("null PyObject pointer");
        return -1;
    }
    if (!out) {
        set_error("null output pointer");
        return -2;
    }

    PyObject* obj = static_cast<PyObject*>(py_obj);
    if (!THPVariable_Check(obj)) {
        set_error("PyObject is not a torch.Tensor");
        return -3;
    }

    const torch::Tensor& tensor = THPVariable_Unpack(obj);

    out->data_ptr = tensor.data_ptr();
    out->ndim = static_cast<int32_t>(tensor.dim());

    if (out->ndim > TENSOR_BRIDGE_MAX_DIMS) {
        set_error("tensor has too many dimensions");
        return -4;
    }

    auto sizes = tensor.sizes();
    auto strd = tensor.strides();
    for (int i = 0; i < out->ndim; ++i) {
        out->shape[i] = sizes[i];
        out->strides[i] = strd[i];
    }
    for (int i = out->ndim; i < TENSOR_BRIDGE_MAX_DIMS; ++i) {
        out->shape[i] = 0;
        out->strides[i] = 0;
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

extern "C" const char* tensor_bridge_get_error(void)
{
    return g_error_buffer;
}

// Fast signature extraction - returns NULL if not a tensor
extern "C" int tensor_bridge_get_signature(void* py_obj, char* buffer, size_t buffer_size)
{
    if (!py_obj)
        return -1;

    PyObject* obj = static_cast<PyObject*>(py_obj);
    if (!THPVariable_Check(obj))
        return -2;

    const torch::Tensor& tensor = THPVariable_Unpack(obj);

    // Format: "[torch,Dn,Sm]" where n=ndim, m=scalar_type
    int ndim = static_cast<int>(tensor.dim());
    int scalar_type = static_cast<int>(tensor.scalar_type());

    snprintf(buffer, buffer_size, "[torch,D%d,S%d]", ndim, scalar_type);

    return 0;
}

static const TensorBridgeAPI g_api
    = {TENSOR_BRIDGE_API_VERSION,
       sizeof(TensorBridgeInfo),
       tensor_bridge_extract,
       tensor_bridge_is_tensor,
       tensor_bridge_get_signature,
       tensor_bridge_get_error};

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

            for (int i = 0; i < 1000; ++i)
                tensor_bridge_extract(raw_ptr, &info);

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i)
                tensor_bridge_extract(raw_ptr, &info);
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
