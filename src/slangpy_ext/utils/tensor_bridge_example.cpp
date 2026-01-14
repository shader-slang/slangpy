// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Example: How to use the tensor bridge in slangpy_ext
//
// This file shows the integration pattern for your native code.
// The critical point is that once you have the function pointer,
// calling it is just a normal C function call with ~10-50ns overhead.

#include <nanobind/nanobind.h>
#include "../slangpy_torch_bridge/tensor_bridge_api.h"

namespace nb = nanobind;

namespace sgl {

// Singleton to hold the bridge API
class TensorBridge {
public:
    static TensorBridge& instance()
    {
        static TensorBridge inst;
        return inst;
    }

    // Initialize by importing the Python module and getting the API pointer
    bool initialize()
    {
        if (m_api)
            return true; // Already initialized

        try {
            // Import the bridge module (this only happens once)
            nb::module_ bridge = nb::module_::import_("slangpy_torch_bridge");

            // Get the API struct pointer
            nb::object api_ptr_obj = bridge.attr("get_api_ptr")();
            uintptr_t api_ptr = nb::cast<uintptr_t>(api_ptr_obj);

            m_api = reinterpret_cast<const TensorBridgeAPI*>(api_ptr);

            // Verify compatibility
            if (m_api->api_version != TENSOR_BRIDGE_API_VERSION
                || m_api->info_struct_size != sizeof(TensorBridgeInfo)) {
                m_api = nullptr;
                return false;
            }

            return true;
        } catch (...) {
            m_api = nullptr;
            return false;
        }
    }

    bool is_available() const { return m_api != nullptr; }

    // Check if a handle is a torch.Tensor - FAST (~10ns)
    bool is_tensor(nb::handle h) const
    {
        if (!m_api)
            return false;
        return m_api->is_tensor(h.ptr()) != 0;
    }

    // Extract tensor info - THE HOT PATH FUNCTION
    // This is what you call in your performance-critical code
    // Overhead: ~20-50ns (just a C function call)
    bool extract(nb::handle tensor, TensorBridgeInfo& out) const
    {
        if (!m_api)
            return false;
        return m_api->extract(tensor.ptr(), &out) == 0;
    }

    // Convenience: extract with error throwing
    TensorBridgeInfo extract_or_throw(nb::handle tensor) const
    {
        TensorBridgeInfo info;
        if (!extract(tensor, info)) {
            throw std::runtime_error(m_api ? m_api->get_error() : "tensor bridge not initialized");
        }
        return info;
    }

private:
    TensorBridge() = default;
    const TensorBridgeAPI* m_api = nullptr;
};

// Example usage in your dispatch code:
//
// void dispatch_kernel(nb::handle tensor_arg) {
//     auto& bridge = TensorBridge::instance();
//
//     // Fast check if it's a tensor
//     if (bridge.is_tensor(tensor_arg)) {
//         TensorBridgeInfo info;
//
//         // Extract tensor info - THIS IS FAST (~20-50ns)
//         if (bridge.extract(tensor_arg, info)) {
//             // Now you have:
//             // - info.data_ptr  (GPU memory pointer)
//             // - info.shape[0..ndim-1]
//             // - info.strides[0..ndim-1]
//             // - info.ndim
//             // - info.numel
//             // - info.device_type, device_index
//             // - info.scalar_type, element_size
//             // - info.is_contiguous, is_cuda
//
//             // Use this data to set up your GPU kernel dispatch
//             void* gpu_ptr = info.data_ptr;
//             // ... dispatch kernel ...
//         }
//     }
// }

} // namespace sgl
