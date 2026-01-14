// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "tensor_bridge_api.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace sgl {

/// Singleton providing fast access to PyTorch tensor metadata.
///
/// Usage:
///   // At module init or on first PyTorch tensor encounter:
///   TorchBridge::instance().try_init();
///
///   // In hot path (~28ns):
///   if (TorchBridge::instance().is_available()) {
///       TensorBridgeInfo info;
///       if (TorchBridge::instance().extract(handle, info)) {
///           // Use info.data_ptr, info.shape, etc.
///       }
///   }
class TorchBridge {
public:
    static TorchBridge& instance()
    {
        static TorchBridge inst;
        return inst;
    }

    /// Attempt to load torch and slangpy_torch. Returns true if available.
    /// Safe to call multiple times - will only try once.
    /// Automatically imports torch first if needed (slangpy_torch links against it).
    bool try_init()
    {
        // Only try once
        if (m_initialized)
            return m_api != nullptr;

        m_initialized = true;

        try {
            // First, try to import torch - slangpy_torch links against libtorch
            // and will fail to load if torch DLLs aren't available
            nb::module_::import_("torch");

            // Now import slangpy_torch
            nb::module_ bridge = nb::module_::import_("slangpy_torch");
            nb::object api_ptr_obj = bridge.attr("get_api_ptr")();
            uintptr_t api_ptr = nb::cast<uintptr_t>(api_ptr_obj);

            m_api = reinterpret_cast<const TensorBridgeAPI*>(api_ptr);

            // Verify compatibility
            if (m_api->api_version != TENSOR_BRIDGE_API_VERSION
                || m_api->info_struct_size != sizeof(TensorBridgeInfo)) {
                m_api = nullptr;
            }
        } catch (...) {
            m_api = nullptr;
        }

        return m_api != nullptr;
    }

    /// Check if the bridge is available (slangpy_torch is installed)
    bool is_available() const { return m_api != nullptr; }

    /// Check if a PyObject is a torch.Tensor (~10ns)
    bool is_tensor(PyObject* obj) const
    {
        if (!m_api)
            return false;
        return m_api->is_tensor(obj) != 0;
    }

    bool is_tensor(nb::handle h) const { return is_tensor(h.ptr()); }

    /// Extract tensor info (~28ns). Returns true on success.
    bool extract(PyObject* tensor, TensorBridgeInfo& out) const
    {
        if (!m_api)
            return false;
        return m_api->extract(tensor, &out) == 0;
    }

    bool extract(nb::handle h, TensorBridgeInfo& out) const { return extract(h.ptr(), out); }

    /// Get a minimal signature string for a tensor (~15ns)
    /// Returns nullptr if not a tensor or bridge unavailable
    /// Format: "[torch,Dn,Sm]" where n=ndim, m=scalar_type
    int get_signature(PyObject* obj, char* buffer, size_t buffer_size) const
    {
        if (!m_api) {
            snprintf(buffer, buffer_size, "slangpy_torch not available");
            return -1;
        }
        if (m_api->get_signature(obj, buffer, buffer_size) != 0) {
            snprintf(buffer, buffer_size, "Failed to get signature");
        }
        return 0;
    }

    int get_signature(nb::handle h, char* buffer, size_t buffer_size) const
    {
        return get_signature(h.ptr(), buffer, buffer_size);
    }

    /// Get last error message
    const char* get_error() const
    {
        if (!m_api)
            return "slangpy_torch not available";
        return m_api->get_error();
    }

private:
    TorchBridge() = default;
    TorchBridge(const TorchBridge&) = delete;
    TorchBridge& operator=(const TorchBridge&) = delete;

    const TensorBridgeAPI* m_api = nullptr;
    bool m_initialized = false;
};

} // namespace sgl
