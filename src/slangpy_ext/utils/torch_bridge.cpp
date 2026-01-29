// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torch_bridge.h"
#include "../nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/cuda_utils.h"

#include <vector>

namespace sgl {

/// Convert TensorBridgeInfo to a Python dictionary for testing/debugging.
nb::dict tensor_info_to_dict(const TensorBridgeInfo& info)
{
    nb::dict result;

    result["data_ptr"] = reinterpret_cast<uintptr_t>(info.data_ptr);

    // Convert shape and strides to Python lists
    nb::list shape_list, strides_list;
    for (int i = 0; i < info.ndim; ++i) {
        shape_list.append(info.shape[i]);
        strides_list.append(info.strides[i]);
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
    result["cuda_stream"] = reinterpret_cast<uintptr_t>(info.cuda_stream);
    result["is_contiguous"] = static_cast<bool>(info.is_contiguous);
    result["is_cuda"] = static_cast<bool>(info.is_cuda);
    result["requires_grad"] = static_cast<bool>(info.requires_grad);

    return result;
}

/// Python-exposed function to extract PyTorch tensor info via the bridge.
/// Returns a dict with all tensor metadata, or throws if bridge unavailable.
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

    TensorBridgeInfo info;
    if (!bridge.extract(tensor, info)) {
        throw std::runtime_error(bridge.get_error());
    }

    return tensor_info_to_dict(info);
}

/// Extract PyTorch tensor signature via the bridge
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
    if (bridge.get_signature(tensor, buffer, sizeof(buffer)) != 0) {
        throw std::runtime_error(bridge.get_error());
    }

    return std::string(buffer);
}

/// Check if the torch bridge is available
bool is_torch_bridge_available()
{
    auto& bridge = TorchBridge::instance();
    bridge.try_init(); // Ensure we attempt to initialize
    return bridge.is_available();
}

/// Check if an object is a PyTorch tensor (via the bridge)
bool is_torch_tensor(nb::handle obj)
{
    auto& bridge = TorchBridge::instance();
    bridge.try_init(); // Ensure we attempt to initialize
    if (!bridge.is_available())
        return false;
    return bridge.is_tensor(obj);
}

/// Copy a PyTorch tensor to a buffer's CUDA memory.
/// The buffer must have been created with BufferUsage::shared.
/// Returns true on success.
bool copy_torch_tensor_to_buffer(nb::handle tensor, ref<Buffer> buffer)
{
    auto& bridge = TorchBridge::instance();
    bridge.try_init();

    if (!bridge.is_available()) {
        throw std::runtime_error("Torch bridge is not available");
    }

    // Extract tensor info
    TensorBridgeInfo info;
    if (!bridge.extract(tensor, info)) {
        throw std::runtime_error(std::string("Failed to extract tensor info: ") + bridge.get_error());
    }

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

    // Copy tensor to buffer
    if (!bridge.copy_to_buffer(tensor, cuda_ptr, tensor_size)) {
        throw std::runtime_error(std::string("copy_to_buffer failed: ") + bridge.get_error());
    }

    return true;
}

/// Copy from a buffer's CUDA memory to a PyTorch tensor.
/// The buffer must have been created with BufferUsage::shared.
/// Returns true on success.
bool copy_buffer_to_torch_tensor(ref<Buffer> buffer, nb::handle tensor)
{
    auto& bridge = TorchBridge::instance();
    bridge.try_init();

    if (!bridge.is_available()) {
        throw std::runtime_error("Torch bridge is not available");
    }

    // Extract tensor info
    TensorBridgeInfo info;
    if (!bridge.extract(tensor, info)) {
        throw std::runtime_error(std::string("Failed to extract tensor info: ") + bridge.get_error());
    }

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

    // Copy buffer to tensor
    if (!bridge.copy_from_buffer(tensor, cuda_ptr, tensor_size)) {
        throw std::runtime_error(std::string("copy_from_buffer failed: ") + bridge.get_error());
    }

    return true;
}

} // namespace sgl

SGL_PY_EXPORT(utils_torch_bridge)
{
    using namespace sgl;

    m.def(
        "is_torch_bridge_available",
        &is_torch_bridge_available,
        "Check if torch bridge is available (native or Python fallback)"
    );

    m.def(
        "is_torch_bridge_using_fallback",
        []()
        {
            return TorchBridge::instance().is_using_fallback();
        },
        "Check if the torch bridge is using Python fallback mode.\n\n"
        "Returns True if using Python fallback (either because slangpy_torch is not\n"
        "installed, or because fallback mode was forced for testing)."
    );

    m.def(
        "set_torch_bridge_python_fallback",
        [](bool force)
        {
            TorchBridge::instance().set_force_python_fallback(force);
        },
        nb::arg("force"),
        "Force use of Python fallback for torch bridge operations.\n\n"
        "This is primarily for testing to validate the fallback path works correctly.\n"
        "When force=True, all torch bridge operations will use the Python fallback\n"
        "implementations even if the native slangpy_torch is available.\n\n"
        "Args:\n"
        "  force: If True, force Python fallback mode. If False, use native if available."
    );

    m.def("is_torch_tensor", &is_torch_tensor, nb::arg("obj"), "Check if an object is a PyTorch tensor");

    m.def(
        "extract_torch_tensor_info",
        &extract_torch_tensor_info,
        nb::arg("tensor"),
        "Extract PyTorch tensor metadata as a dictionary.\n\n"
        "Returns a dict containing:\n"
        "  - data_ptr: GPU/CPU memory address\n"
        "  - shape: tuple of dimensions\n"
        "  - strides: tuple of strides (in elements)\n"
        "  - ndim: number of dimensions\n"
        "  - device_type: 0=CPU, 1=CUDA\n"
        "  - device_index: GPU index or -1 for CPU\n"
        "  - scalar_type: PyTorch scalar type code\n"
        "  - element_size: bytes per element\n"
        "  - numel: total number of elements\n"
        "  - storage_offset: offset in storage\n"
        "  - cuda_stream: CUDA stream pointer (0 for CPU)\n"
        "  - is_contiguous: whether tensor is contiguous\n"
        "  - is_cuda: whether tensor is on CUDA\n"
        "  - requires_grad: whether tensor requires gradients\n\n"
        "Raises:\n"
        "  RuntimeError: if torch bridge is not available\n"
        "  ValueError: if object is not a PyTorch tensor"
    );

    m.def(
        "extract_torch_tensor_signature",
        &extract_torch_tensor_signature,
        nb::arg("tensor"),
        "Extract PyTorch tensor signature as a string.\n\n"
        "Returns a string containing the tensor signature in the format:\n"
        "  [Dn,Sm] where n=ndim, m=scalar_type\n"
        "\n"
        "Raises:\n"
        "  RuntimeError: if torch bridge is not available\n"
        "  ValueError: if object is not a PyTorch tensor"
    );

    m.def(
        "copy_torch_tensor_to_buffer",
        &copy_torch_tensor_to_buffer,
        nb::arg("tensor"),
        nb::arg("buffer"),
        "Copy a PyTorch CUDA tensor to a buffer's CUDA memory.\n\n"
        "The buffer must have been created with BufferUsage.shared.\n"
        "Handles non-contiguous tensors correctly.\n\n"
        "Args:\n"
        "  tensor: A PyTorch CUDA tensor\n"
        "  buffer: A Buffer created with BufferUsage.shared\n\n"
        "Returns:\n"
        "  True on success\n\n"
        "Raises:\n"
        "  RuntimeError: if copy fails or buffer not compatible"
    );

    m.def(
        "copy_buffer_to_torch_tensor",
        &copy_buffer_to_torch_tensor,
        nb::arg("buffer"),
        nb::arg("tensor"),
        "Copy from a buffer's CUDA memory to a PyTorch CUDA tensor.\n\n"
        "The buffer must have been created with BufferUsage.shared.\n"
        "Handles non-contiguous tensors correctly.\n\n"
        "Args:\n"
        "  buffer: A Buffer created with BufferUsage.shared\n"
        "  tensor: A PyTorch CUDA tensor\n\n"
        "Returns:\n"
        "  True on success\n\n"
        "Raises:\n"
        "  RuntimeError: if copy fails or buffer not compatible"
    );
}
