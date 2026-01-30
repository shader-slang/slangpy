// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torch_bridge.h"

#include "sgl/device/device.h"
#include "sgl/device/cuda_utils.h"

#include <vector>

namespace sgl {

/// Convert TensorBridgeInfo to a Python dictionary.
nb::dict tensor_info_to_dict(const TensorBridgeInfo& info)
{
    nb::dict result;

    result["data_ptr"] = reinterpret_cast<uintptr_t>(info.data_ptr);

    // Convert shape and strides to Python lists (if available)
    nb::list shape_list, strides_list;
    if (info.shape && info.strides) {
        for (int i = 0; i < info.ndim; ++i) {
            shape_list.append(info.shape[i]);
            strides_list.append(info.strides[i]);
        }
    }
    result["shape"] = nb::tuple(shape_list);
    result["strides"] = nb::tuple(strides_list);

    result["ndim"] = info.ndim;
    result["device_type"] = info.device_type;
    result["device_index"] = info.device_index;
    result["scalar_type"] = info.scalar_type;
    result["element_size"] = info.element_size;
    result["numel"] = info.numel;
    result["storage_offset"] = info.storage_offset;
    result["is_contiguous"] = static_cast<bool>(info.is_contiguous);
    result["is_cuda"] = static_cast<bool>(info.is_cuda);
    result["requires_grad"] = static_cast<bool>(info.requires_grad);

    return result;
}

/// Extract PyTorch tensor metadata as a dictionary.
/// @param tensor PyTorch tensor to extract info from.
/// @return Dictionary containing tensor metadata.
/// @throws std::runtime_error if torch bridge is not available.
/// @throws std::invalid_argument if object is not a PyTorch tensor.
nb::object extract_torch_tensor_info(nb::handle tensor)
{
    auto& bridge = TorchBridge::instance();

    // Ensure we attempt to initialize (lazy loading)
    bridge.try_init();

    if (!bridge.is_available()) {
        throw std::runtime_error("Torch bridge is not available. Make sure torch is imported before slangpy.");
    }

    if (!bridge.is_tensor(tensor)) {
        throw std::invalid_argument("Object is not a PyTorch tensor");
    }

    // Use default buffer size for shape/strides extraction
    int64_t shape_buffer[TENSOR_BRIDGE_DEFAULT_DIMS];
    int64_t strides_buffer[TENSOR_BRIDGE_DEFAULT_DIMS];

    TensorBridgeInfo info;
    // extract() now throws on error, so we don't need to check the return value
    bridge.extract(tensor, info, shape_buffer, strides_buffer, TENSOR_BRIDGE_DEFAULT_DIMS);

    return tensor_info_to_dict(info);
}

/// Extract PyTorch tensor signature string.
/// @param tensor PyTorch tensor to get signature from.
/// @return Signature string in format "[Dn,Sm]" where n=ndim, m=scalar_type.
/// @throws std::runtime_error if torch bridge is not available.
/// @throws std::invalid_argument if object is not a PyTorch tensor.
std::string extract_torch_tensor_signature(nb::handle tensor)
{
    auto& bridge = TorchBridge::instance();
    bridge.try_init();

    if (!bridge.is_available()) {
        throw std::runtime_error("Torch bridge is not available");
    }

    if (!bridge.is_tensor(tensor)) {
        throw std::invalid_argument("Object is not a PyTorch tensor");
    }

    char buffer[64];
    int result = bridge.get_signature(tensor, buffer, sizeof(buffer));
    if (result != 0) {
        throw std::runtime_error(std::string("get_signature failed: ") + tensor_bridge_result_to_string(result));
    }

    return std::string(buffer);
}

/// Check if the torch bridge is available (native or Python fallback).
bool is_torch_bridge_available()
{
    auto& bridge = TorchBridge::instance();
    bridge.try_init(); // Ensure we attempt to initialize
    return bridge.is_available();
}

/// Check if an object is a PyTorch tensor.
/// @param obj Object to check.
/// @return True if obj is a PyTorch tensor.
bool is_torch_tensor(nb::handle obj)
{
    auto& bridge = TorchBridge::instance();
    bridge.try_init(); // Ensure we attempt to initialize
    if (!bridge.is_available())
        return false;
    return bridge.is_tensor(obj);
}

/// Check if the torch bridge is using Python fallback mode.
/// @return True if using Python fallback.
bool is_torch_bridge_using_fallback()
{
    return TorchBridge::instance().is_using_fallback();
}

/// Force use of Python fallback for torch bridge operations.
/// @param force If true, force Python fallback mode.
void set_torch_bridge_python_fallback(bool force)
{
    TorchBridge::instance().set_force_python_fallback(force);
}

/// Copy a PyTorch CUDA tensor to a buffer's CUDA memory.
/// @param tensor PyTorch CUDA tensor to copy from.
/// @param buffer Buffer created with BufferUsage::shared.
/// @return True on success.
/// @throws std::runtime_error if copy fails or buffer not compatible.
bool copy_torch_tensor_to_buffer(nb::handle tensor, ref<Buffer> buffer)
{
    auto& bridge = TorchBridge::instance();
    bridge.try_init();

    if (!bridge.is_available()) {
        throw std::runtime_error("Torch bridge is not available");
    }

    // Extract tensor info (don't need shape/strides for copy, just numel and element_size)
    // extract() now throws on error
    TensorBridgeInfo info;
    bridge.extract(tensor, info, nullptr, nullptr, 0);

    if (!info.is_cuda) {
        throw std::runtime_error("Tensor must be on CUDA device");
    }

    // Calculate expected buffer size
    size_t tensor_size = static_cast<size_t>(info.numel) * static_cast<size_t>(info.element_size);
    if (buffer->size() < tensor_size) {
        throw std::runtime_error("Buffer too small for tensor data");
    }

    // Get CUDA memory pointer from buffer
    void* cuda_ptr = buffer->cuda_memory();
    if (!cuda_ptr) {
        throw std::runtime_error(
            "Buffer cuda_memory() returned nullptr - ensure buffer was created with BufferUsage::shared"
        );
    }

    // Copy tensor to buffer - copy_to_buffer() now throws on error
    bridge.copy_to_buffer(tensor, cuda_ptr, tensor_size);

    return true;
}

/// Copy from a buffer's CUDA memory to a PyTorch CUDA tensor.
/// @param buffer Buffer created with BufferUsage::shared.
/// @param tensor PyTorch CUDA tensor to copy to.
/// @return True on success.
/// @throws std::runtime_error if copy fails or buffer not compatible.
bool copy_buffer_to_torch_tensor(ref<Buffer> buffer, nb::handle tensor)
{
    auto& bridge = TorchBridge::instance();
    bridge.try_init();

    if (!bridge.is_available()) {
        throw std::runtime_error("Torch bridge is not available");
    }

    // Extract tensor info (don't need shape/strides for copy, just numel and element_size)
    // extract() now throws on error
    TensorBridgeInfo info;
    bridge.extract(tensor, info, nullptr, nullptr, 0);

    if (!info.is_cuda) {
        throw std::runtime_error("Tensor must be on CUDA device");
    }

    // Calculate expected tensor size
    size_t tensor_size = static_cast<size_t>(info.numel) * static_cast<size_t>(info.element_size);
    if (buffer->size() < tensor_size) {
        throw std::runtime_error("Buffer too small for tensor data");
    }

    // Get CUDA memory pointer from buffer
    void* cuda_ptr = buffer->cuda_memory();
    if (!cuda_ptr) {
        throw std::runtime_error(
            "Buffer cuda_memory() returned nullptr - ensure buffer was created with BufferUsage::shared"
        );
    }

    // Copy buffer to tensor - copy_from_buffer() now throws on error
    bridge.copy_from_buffer(tensor, cuda_ptr, tensor_size);

    return true;
}

} // namespace sgl

SGL_PY_EXPORT(utils_torch_bridge)
{
    using namespace sgl;

    m.def("is_torch_bridge_available", &is_torch_bridge_available, D_NA(is_torch_bridge_available));

    m.def("is_torch_bridge_using_fallback", &is_torch_bridge_using_fallback, D_NA(is_torch_bridge_using_fallback));

    m.def(
        "set_torch_bridge_python_fallback",
        &set_torch_bridge_python_fallback,
        "force"_a,
        D_NA(set_torch_bridge_python_fallback)
    );

    m.def("is_torch_tensor", &is_torch_tensor, "obj"_a, D_NA(is_torch_tensor));

    m.def("extract_torch_tensor_info", &extract_torch_tensor_info, "tensor"_a, D_NA(extract_torch_tensor_info));

    m.def(
        "extract_torch_tensor_signature",
        &extract_torch_tensor_signature,
        "tensor"_a,
        D_NA(extract_torch_tensor_signature)
    );

    m.def(
        "copy_torch_tensor_to_buffer",
        &copy_torch_tensor_to_buffer,
        "tensor"_a,
        "buffer"_a,
        D_NA(copy_torch_tensor_to_buffer)
    );

    m.def(
        "copy_buffer_to_torch_tensor",
        &copy_buffer_to_torch_tensor,
        "buffer"_a,
        "tensor"_a,
        D_NA(copy_buffer_to_torch_tensor)
    );
}
