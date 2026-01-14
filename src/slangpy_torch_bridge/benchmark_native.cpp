// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// benchmark_native.cpp
//
// This file creates a native extension that benchmarks the actual
// C function call overhead without Python/ctypes involvement.

#include <torch/extension.h>
#include "../tensor_bridge_api.h"
#include <chrono>

// External declaration of the function we want to call
extern "C" int tensor_bridge_extract(void* py_obj, TensorBridgeInfo* out);
extern "C" const TensorBridgeAPI* tensor_bridge_get_api();

// Benchmark function that measures actual C call overhead
py::dict benchmark_native_extraction(const torch::Tensor& tensor, int iterations)
{
    // Get the PyObject* directly from the tensor
    // In pybind11/torch, we can get this via the Python object
    py::object py_tensor = py::cast(tensor);
    PyObject* raw_ptr = py_tensor.ptr();

    TensorBridgeInfo info;

    // Warm up
    for (int i = 0; i < 1000; ++i) {
        tensor_bridge_extract(raw_ptr, &info);
    }

    // Benchmark the C function call
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
}

// Benchmark just getting data_ptr via libtorch (baseline)
py::dict benchmark_libtorch_baseline(const torch::Tensor& tensor, int iterations)
{
    // Warm up
    void* ptr;
    for (int i = 0; i < 1000; ++i) {
        ptr = tensor.data_ptr();
    }

    // Benchmark direct libtorch access
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ptr = tensor.data_ptr();
        auto sizes = tensor.sizes();
        auto strides = tensor.strides();
        // Force use to prevent optimization
        if (ptr == nullptr)
            break;
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ns = std::chrono::duration<double, std::nano>(end - start).count();
    double per_call_ns = total_ns / iterations;

    py::dict result;
    result["iterations"] = iterations;
    result["total_ns"] = total_ns;
    result["per_call_ns"] = per_call_ns;

    return result;
}

PYBIND11_MODULE(bridge_benchmark, m)
{
    m.doc() = "Native benchmark for tensor bridge";

    m.def(
        "benchmark_native_extraction",
        &benchmark_native_extraction,
        "Benchmark the C function call overhead",
        py::arg("tensor"),
        py::arg("iterations") = 1000000
    );

    m.def(
        "benchmark_libtorch_baseline",
        &benchmark_libtorch_baseline,
        "Benchmark direct libtorch access (baseline)",
        py::arg("tensor"),
        py::arg("iterations") = 1000000
    );
}
