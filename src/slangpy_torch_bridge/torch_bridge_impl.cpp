// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// PyTorch Native Extension Bridge - Implementation
//
// This extension is compiled against libtorch and provides C-callable
// functions that can extract PyTorch tensor metadata from a raw PyObject*.
//
// The key design:
// 1. This module exports a C API (function pointers, no C++ ABI issues)
// 2. The calling code (slangpy_ext) obtains these function pointers at init time
// 3. In the hot path, slangpy_ext calls the C function directly with PyObject*
// 4. No Python interpreter calls needed - just pure C++ libtorch access

#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h> // For THPVariable_Check/Unpack
#include "tensor_bridge_api.h"

#include <cstring>
#include <chrono>

// Thread-local error message buffer
static thread_local char g_error_buffer[256] = {0};

static void set_error(const char* msg)
{
    strncpy(g_error_buffer, msg, sizeof(g_error_buffer) - 1);
    g_error_buffer[sizeof(g_error_buffer) - 1] = '\0';
}

// ============================================================================
// C API Implementation
// ============================================================================

// Check if PyObject* is a torch.Tensor
extern "C" int tensor_bridge_is_tensor(void* py_obj)
{
    if (!py_obj)
        return 0;
    PyObject* obj = static_cast<PyObject*>(py_obj);
    return THPVariable_Check(obj) ? 1 : 0;
}

// Extract tensor info from PyObject* - THE HOT PATH FUNCTION
// This function does NO Python API calls - it goes directly to libtorch
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

    // Check if it's a tensor - this is a fast type check
    if (!THPVariable_Check(obj)) {
        set_error("PyObject is not a torch.Tensor");
        return -3;
    }

    // Unpack the tensor - this gives us direct access to the C++ Tensor object
    // NO Python API calls happen here!
    const torch::Tensor& tensor = THPVariable_Unpack(obj);

    // Extract all metadata directly from the C++ tensor object
    out->data_ptr = tensor.data_ptr();
    out->ndim = static_cast<int32_t>(tensor.dim());

    // Copy shape and strides (sizes()/strides() return IntArrayRef - just pointer+size)
    auto sizes = tensor.sizes();
    auto strd = tensor.strides();

    int ndim = out->ndim;
    if (ndim > TENSOR_BRIDGE_MAX_DIMS) {
        set_error("tensor has too many dimensions");
        return -4;
    }

    for (int i = 0; i < ndim; ++i) {
        out->shape[i] = sizes[i];
        out->strides[i] = strd[i];
    }
    // Zero out remaining dims
    for (int i = ndim; i < TENSOR_BRIDGE_MAX_DIMS; ++i) {
        out->shape[i] = 0;
        out->strides[i] = 0;
    }

    // Device info
    auto device = tensor.device();
    out->device_type = static_cast<int32_t>(device.type());
    out->device_index = device.has_index() ? static_cast<int32_t>(device.index()) : -1;
    out->is_cuda = device.is_cuda() ? 1 : 0;

    // Dtype info
    out->scalar_type = static_cast<int32_t>(tensor.scalar_type());
    out->element_size = static_cast<int32_t>(tensor.element_size());

    // Other metadata
    out->numel = tensor.numel();
    out->storage_offset = tensor.storage_offset();
    out->is_contiguous = tensor.is_contiguous() ? 1 : 0;
    out->requires_grad = tensor.requires_grad() ? 1 : 0;
    out->_padding = 0;

    return 0;
}

// Get last error message
extern "C" const char* tensor_bridge_get_error(void)
{
    return g_error_buffer;
}

// ============================================================================
// API Struct - single point of access for all functions
// ============================================================================

static const TensorBridgeAPI g_api
    = {TENSOR_BRIDGE_API_VERSION,
       sizeof(TensorBridgeInfo),
       tensor_bridge_extract,
       tensor_bridge_is_tensor,
       tensor_bridge_get_error};

extern "C" const TensorBridgeAPI* tensor_bridge_get_api(void)
{
    return &g_api;
}

// ============================================================================
// Python Module - exposes function pointers to Python
// ============================================================================

PYBIND11_MODULE(slangpy_torch_bridge, m)
{
    m.doc() = "Fast PyTorch tensor access bridge for slangpy";

    // Export function pointers as integers (for ctypes or direct C++ use)
    m.def(
        "get_api_ptr",
        []() -> uintptr_t
        {
            return reinterpret_cast<uintptr_t>(tensor_bridge_get_api());
        },
        "Get pointer to the TensorBridgeAPI struct"
    );

    m.def(
        "get_extract_fn_ptr",
        []() -> uintptr_t
        {
            return reinterpret_cast<uintptr_t>(tensor_bridge_extract);
        },
        "Get pointer to the extract function (for direct C++ calls)"
    );

    m.def(
        "get_is_tensor_fn_ptr",
        []() -> uintptr_t
        {
            return reinterpret_cast<uintptr_t>(tensor_bridge_is_tensor);
        },
        "Get pointer to the is_tensor check function"
    );

    // Version/compatibility info
    m.attr("API_VERSION") = TENSOR_BRIDGE_API_VERSION;
    m.attr("INFO_STRUCT_SIZE") = sizeof(TensorBridgeInfo);

    // ========================================================================
    // Python-friendly wrappers (for testing/debugging, not hot path)
    // ========================================================================

    m.def(
        "extract_info",
        [](const torch::Tensor& tensor) -> py::dict
        {
            TensorBridgeInfo info;

            // Use PyTorch's Python wrapper to get the PyObject*
            // Then call our C function to verify it works
            info.data_ptr = tensor.data_ptr();
            info.ndim = static_cast<int32_t>(tensor.dim());

            auto sizes = tensor.sizes();
            auto strd = tensor.strides();
            for (int i = 0; i < info.ndim; ++i) {
                info.shape[i] = sizes[i];
                info.strides[i] = strd[i];
            }

            auto device = tensor.device();
            info.device_type = static_cast<int32_t>(device.type());
            info.device_index = device.has_index() ? static_cast<int32_t>(device.index()) : -1;
            info.scalar_type = static_cast<int32_t>(tensor.scalar_type());
            info.element_size = static_cast<int32_t>(tensor.element_size());
            info.numel = tensor.numel();
            info.storage_offset = tensor.storage_offset();
            info.is_contiguous = tensor.is_contiguous();
            info.is_cuda = device.is_cuda();
            info.requires_grad = tensor.requires_grad();

            py::dict result;
            result["data_ptr"] = reinterpret_cast<uintptr_t>(info.data_ptr);

            std::vector<int64_t> shape_vec(info.shape, info.shape + info.ndim);
            std::vector<int64_t> stride_vec(info.strides, info.strides + info.ndim);
            result["shape"] = py::tuple(py::cast(shape_vec));
            result["strides"] = py::tuple(py::cast(stride_vec));

            result["ndim"] = info.ndim;
            result["device_type"] = info.device_type;
            result["device_index"] = info.device_index;
            result["scalar_type"] = info.scalar_type;
            result["element_size"] = info.element_size;
            result["numel"] = info.numel;
            result["storage_offset"] = info.storage_offset;
            result["is_contiguous"] = info.is_contiguous ? true : false;
            result["is_cuda"] = info.is_cuda ? true : false;
            result["requires_grad"] = info.requires_grad ? true : false;

            return result;
        },
        "Extract tensor info as a Python dict (for debugging)",
        py::arg("tensor")
    );

    // Test function that exercises the full C API path
    m.def(
        "test_c_api",
        [](py::object tensor_obj) -> py::dict
        {
            // Get raw PyObject* from py::object
            PyObject* raw_ptr = tensor_obj.ptr();

            // Call through the C API
            TensorBridgeInfo info;
            int result = tensor_bridge_extract(raw_ptr, &info);

            py::dict out;
            out["success"] = (result == 0);
            out["error_code"] = result;
            if (result != 0) {
                out["error_msg"] = tensor_bridge_get_error();
            } else {
                out["data_ptr"] = reinterpret_cast<uintptr_t>(info.data_ptr);
                out["ndim"] = info.ndim;
                out["numel"] = info.numel;
                out["is_cuda"] = bool(info.is_cuda);

                std::vector<int64_t> shape_vec(info.shape, info.shape + info.ndim);
                out["shape"] = py::tuple(py::cast(shape_vec));
            }
            return out;
        },
        "Test the C API extraction path",
        py::arg("tensor")
    );

    // ========================================================================
    // Native benchmark function - measures ACTUAL C function call overhead
    // ========================================================================

    m.def(
        "benchmark_c_api",
        [](py::object tensor_obj, int iterations) -> py::dict
        {
            // Get raw PyObject* - this simulates what slangpy_ext has
            PyObject* raw_ptr = tensor_obj.ptr();
            TensorBridgeInfo info;

            // Warm up
            for (int i = 0; i < 1000; ++i) {
                tensor_bridge_extract(raw_ptr, &info);
            }

            // Benchmark the actual C function call
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                tensor_bridge_extract(raw_ptr, &info);
            }
            auto end = std::chrono::high_resolution_clock::now();

            double total_ns = std::chrono::duration<double, std::nano>(end - start).count();
            double per_call_ns = total_ns / iterations;

            py::dict result;
            result["iterations"] = iterations;
            result["total_ns"] = total_ns;
            result["per_call_ns"] = per_call_ns;
            result["data_ptr"] = reinterpret_cast<uintptr_t>(info.data_ptr);
            result["ndim"] = info.ndim;
            result["numel"] = info.numel;

            return result;
        },
        "Benchmark actual C function call overhead",
        py::arg("tensor"),
        py::arg("iterations") = 1000000
    );

    // Benchmark direct libtorch access for comparison
    m.def(
        "benchmark_libtorch",
        [](const torch::Tensor& tensor, int iterations) -> py::dict
        {
            void* ptr;
            int64_t numel;

            // Warm up
            for (int i = 0; i < 1000; ++i) {
                ptr = tensor.data_ptr();
                auto sizes = tensor.sizes();
                auto strides = tensor.strides();
                numel = tensor.numel();
            }

            // Benchmark direct libtorch access
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                ptr = tensor.data_ptr();
                auto sizes = tensor.sizes();
                auto strides = tensor.strides();
                numel = tensor.numel();
            }
            auto end = std::chrono::high_resolution_clock::now();

            double total_ns = std::chrono::duration<double, std::nano>(end - start).count();
            double per_call_ns = total_ns / iterations;

            py::dict result;
            result["iterations"] = iterations;
            result["total_ns"] = total_ns;
            result["per_call_ns"] = per_call_ns;
            result["data_ptr"] = reinterpret_cast<uintptr_t>(ptr);
            result["numel"] = numel;

            return result;
        },
        "Benchmark direct libtorch access (baseline)",
        py::arg("tensor"),
        py::arg("iterations") = 1000000
    );
}
