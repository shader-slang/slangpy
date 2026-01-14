// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// PyTorch Native Extension Bridge
//
// This extension is compiled against libtorch and provides fast access
// to PyTorch tensor internals without Python API overhead.
//
// The key insight is that this module is compiled against the user's
// installed PyTorch, ensuring ABI compatibility, while the main slangpy
// library can consume the extracted data without linking to libtorch.

#include <torch/extension.h>
#include <cstdint>

// C-compatible struct that can be consumed by code not linked to libtorch
// This struct is designed to be stable across PyTorch versions
struct TensorInfo {
    void* data_ptr;         // Device pointer to tensor data
    int64_t* shape;         // Pointer to shape array (borrowed, valid while tensor alive)
    int64_t* strides;       // Pointer to strides array (borrowed, valid while tensor alive)
    int32_t ndim;           // Number of dimensions
    int32_t device_type;    // 0=CPU, 1=CUDA, etc (maps to c10::DeviceType)
    int32_t device_index;   // Device index (e.g., GPU 0, 1, etc.)
    int8_t dtype_code;      // Scalar type code (maps to c10::ScalarType)
    int8_t dtype_bits;      // Bits per element
    int64_t numel;          // Total number of elements
    int64_t storage_offset; // Offset in storage (in elements)
    bool is_contiguous;     // Whether tensor is contiguous in memory
};

// Extract tensor info with zero Python API overhead
// This is the critical hot-path function
TensorInfo extract_tensor_info(const torch::Tensor& tensor)
{
    TensorInfo info;

    // Direct access to tensor internals - no Python calls
    info.data_ptr = tensor.data_ptr();

    // sizes() and strides() return IntArrayRef which is just a pointer+size
    // No allocation happens here
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();

    info.shape = const_cast<int64_t*>(sizes.data());
    info.strides = const_cast<int64_t*>(strides.data());
    info.ndim = static_cast<int32_t>(tensor.dim());

    // Device info
    info.device_type = static_cast<int32_t>(tensor.device().type());
    info.device_index = static_cast<int32_t>(tensor.device().index());

    // Dtype info
    info.dtype_code = static_cast<int8_t>(tensor.scalar_type());
    info.dtype_bits = static_cast<int8_t>(tensor.element_size() * 8);

    info.numel = tensor.numel();
    info.storage_offset = tensor.storage_offset();
    info.is_contiguous = tensor.is_contiguous();

    return info;
}

// Version that returns the raw address of a TensorInfo for external consumption
// The caller MUST ensure the tensor remains alive while using this info
uintptr_t get_tensor_info_ptr(const torch::Tensor& tensor)
{
    // We use thread_local storage to avoid allocation on each call
    // This is safe because we only need the data to survive until the
    // caller copies it out
    thread_local TensorInfo cached_info;
    cached_info = extract_tensor_info(tensor);
    return reinterpret_cast<uintptr_t>(&cached_info);
}

// Batch version for extracting info from multiple tensors at once
// This amortizes any remaining overhead across multiple tensors
std::vector<TensorInfo> extract_tensor_infos(const std::vector<torch::Tensor>& tensors)
{
    std::vector<TensorInfo> infos;
    infos.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        infos.push_back(extract_tensor_info(tensor));
    }
    return infos;
}

// Python-accessible version that returns info as a dict (for debugging/validation)
py::dict get_tensor_info_dict(const torch::Tensor& tensor)
{
    TensorInfo info = extract_tensor_info(tensor);

    py::dict result;
    result["data_ptr"] = reinterpret_cast<uintptr_t>(info.data_ptr);
    result["shape"] = py::tuple(py::cast(std::vector<int64_t>(info.shape, info.shape + info.ndim)));
    result["strides"] = py::tuple(py::cast(std::vector<int64_t>(info.strides, info.strides + info.ndim)));
    result["ndim"] = info.ndim;
    result["device_type"] = info.device_type;
    result["device_index"] = info.device_index;
    result["dtype_code"] = info.dtype_code;
    result["dtype_bits"] = info.dtype_bits;
    result["numel"] = info.numel;
    result["storage_offset"] = info.storage_offset;
    result["is_contiguous"] = info.is_contiguous;
    return result;
}

// Pybind11 module definition
PYBIND11_MODULE(slangpy_torch_bridge, m)
{
    m.doc() = "Fast PyTorch tensor access bridge for slangpy";

    // Expose TensorInfo struct for direct consumption
    py::class_<TensorInfo>(m, "TensorInfo")
        .def_property_readonly(
            "data_ptr",
            [](const TensorInfo& self)
            {
                return reinterpret_cast<uintptr_t>(self.data_ptr);
            }
        )
        .def_readonly("ndim", &TensorInfo::ndim)
        .def_readonly("device_type", &TensorInfo::device_type)
        .def_readonly("device_index", &TensorInfo::device_index)
        .def_readonly("dtype_code", &TensorInfo::dtype_code)
        .def_readonly("dtype_bits", &TensorInfo::dtype_bits)
        .def_readonly("numel", &TensorInfo::numel)
        .def_readonly("storage_offset", &TensorInfo::storage_offset)
        .def_readonly("is_contiguous", &TensorInfo::is_contiguous)
        .def_property_readonly(
            "shape_tuple",
            [](const TensorInfo& self)
            {
                return py::tuple(py::cast(std::vector<int64_t>(self.shape, self.shape + self.ndim)));
            }
        )
        .def_property_readonly(
            "strides_tuple",
            [](const TensorInfo& self)
            {
                return py::tuple(py::cast(std::vector<int64_t>(self.strides, self.strides + self.ndim)));
            }
        );

    // Fast extraction function - returns TensorInfo directly
    m.def("extract_info", &extract_tensor_info, "Extract tensor info with minimal overhead", py::arg("tensor"));

    // Ultra-fast version that returns pointer to cached info
    m.def(
        "get_info_ptr",
        &get_tensor_info_ptr,
        "Get pointer to cached TensorInfo (caller must copy immediately)",
        py::arg("tensor")
    );

    // Batch extraction
    m.def("extract_infos", &extract_tensor_infos, "Extract info from multiple tensors", py::arg("tensors"));

    // Debug/validation version
    m.def(
        "get_info_dict",
        &get_tensor_info_dict,
        "Get tensor info as Python dict (slower, for debugging)",
        py::arg("tensor")
    );

    // Export struct size for ctypes interop
    m.attr("TENSOR_INFO_SIZE") = sizeof(TensorInfo);

    // Export device type constants
    m.attr("DEVICE_TYPE_CPU") = static_cast<int32_t>(c10::DeviceType::CPU);
    m.attr("DEVICE_TYPE_CUDA") = static_cast<int32_t>(c10::DeviceType::CUDA);
}
