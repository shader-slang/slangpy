// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

#include <nanobind/nanobind.h>

// Include the tensor bridge API header from slangpy_torch
// This header is shared between slangpy_torch and slangpy_ext
#include "../../slangpy_torch/tensor_bridge_api.h"

namespace nb = nanobind;

namespace sgl {

/// Singleton providing fast access to PyTorch tensor metadata.
///
/// Usage:
///   // At module init or on first PyTorch tensor encounter:
///   TorchBridge::instance().try_init();
///
///   // In hot path (~28ns with native, slower with Python fallback):
///   if (TorchBridge::instance().is_available()) {
///       TensorBridgeInfo info;
///       if (TorchBridge::instance().extract(handle, info)) {
///           // Use info.data_ptr, info.shape, etc.
///       }
///   }
///
/// The bridge supports two modes:
/// 1. Native mode (fast): Uses slangpy_torch C API for ~28ns tensor metadata extraction
/// 2. Python fallback mode: Uses Python/PyTorch APIs when slangpy_torch is unavailable
///
/// For testing, you can force Python fallback mode via set_force_python_fallback(true).
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
    /// If slangpy_torch is not available, initializes Python fallback.
    bool try_init()
    {
        // Only try once
        if (m_initialized)
            return is_available();

        m_initialized = true;

        try {
            // First, try to import torch - slangpy_torch links against libtorch
            // and will fail to load if torch DLLs aren't available
            nb::module_::import_("torch");

            // Now try to import slangpy_torch for native support
            try {
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
                // slangpy_torch not available, will use Python fallback
                m_api = nullptr;
            }

            // If native API not available, initialize Python fallback
            if (!m_api) {
                init_python_fallback();
            }
        } catch (...) {
            // torch not available
            m_api = nullptr;
        }

        return is_available();
    }

    /// Check if the bridge is available (either native or Python fallback)
    bool is_available() const
    {
        if (m_force_python_fallback) {
            return m_fallback_initialized;
        }
        return m_api != nullptr || m_fallback_initialized;
    }

    /// Check if using Python fallback mode
    bool is_using_fallback() const { return m_force_python_fallback || (m_api == nullptr && m_fallback_initialized); }

    /// Force use of Python fallback even if native is available (for testing)
    void set_force_python_fallback(bool force)
    {
        m_force_python_fallback = force;
        if (force && !m_fallback_initialized) {
            init_python_fallback();
        }
    }

    /// Check if a PyObject is a torch.Tensor (~10ns native, slower with fallback)
    bool is_tensor(PyObject* obj) const
    {
        if (!m_force_python_fallback && m_api) {
            return m_api->is_tensor(obj) != 0;
        }
        return python_is_tensor(obj);
    }

    bool is_tensor(nb::handle h) const { return is_tensor(h.ptr()); }

    /// Extract tensor info (~28ns native, slower with fallback). Returns true on success.
    bool extract(PyObject* tensor, TensorBridgeInfo& out) const
    {
        if (!m_force_python_fallback && m_api) {
            return m_api->extract(tensor, &out) == 0;
        }
        return python_extract(tensor, out);
    }

    bool extract(nb::handle h, TensorBridgeInfo& out) const { return extract(h.ptr(), out); }

    /// Get a minimal signature string for a tensor (~15ns native)
    /// Returns nullptr if not a tensor or bridge unavailable
    /// Format: "[Dn,Sm]" where n=ndim, m=scalar_type
    int get_signature(PyObject* obj, char* buffer, size_t buffer_size) const
    {
        if (!m_force_python_fallback && m_api) {
            if (m_api->get_signature(obj, buffer, buffer_size) != 0) {
                snprintf(buffer, buffer_size, "Failed to get signature");
                return -1;
            }
            return 0;
        }
        return python_get_signature(obj, buffer, buffer_size);
    }

    int get_signature(nb::handle h, char* buffer, size_t buffer_size) const
    {
        return get_signature(h.ptr(), buffer, buffer_size);
    }

    /// Get last error message
    /// Note: When using Python fallback, errors propagate as exceptions instead
    const char* get_error() const
    {
        if (!m_force_python_fallback && m_api)
            return m_api->get_error();
        return ""; // Python fallback uses exceptions
    }

    /// Get the current CUDA stream for a device
    void* get_current_cuda_stream(int device_index) const
    {
        if (!m_force_python_fallback && m_api) {
            return m_api->get_current_cuda_stream(device_index);
        }
        return python_get_current_cuda_stream(device_index);
    }

    /// Copy tensor data to a contiguous CUDA buffer.
    /// Handles non-contiguous tensors via PyTorch's copy mechanism.
    /// Returns true on success.
    bool copy_to_buffer(PyObject* tensor, void* dest_cuda_ptr, size_t dest_size) const
    {
        if (!m_force_python_fallback && m_api) {
            return m_api->copy_to_buffer(tensor, dest_cuda_ptr, dest_size) == 0;
        }
        return python_copy_to_buffer(tensor, dest_cuda_ptr, dest_size);
    }

    bool copy_to_buffer(nb::handle h, void* dest_cuda_ptr, size_t dest_size) const
    {
        return copy_to_buffer(h.ptr(), dest_cuda_ptr, dest_size);
    }

    /// Copy data from a contiguous CUDA buffer back to a tensor.
    /// Handles non-contiguous tensors via PyTorch's copy mechanism.
    /// Returns true on success.
    bool copy_from_buffer(PyObject* tensor, void* src_cuda_ptr, size_t src_size) const
    {
        if (!m_force_python_fallback && m_api) {
            return m_api->copy_from_buffer(tensor, src_cuda_ptr, src_size) == 0;
        }
        return python_copy_from_buffer(tensor, src_cuda_ptr, src_size);
    }

    bool copy_from_buffer(nb::handle h, void* src_cuda_ptr, size_t src_size) const
    {
        return copy_from_buffer(h.ptr(), src_cuda_ptr, src_size);
    }

private:
    TorchBridge() = default;
    TorchBridge(const TorchBridge&) = delete;
    TorchBridge& operator=(const TorchBridge&) = delete;

    /// Initialize Python fallback - caches all function handles once
    void init_python_fallback()
    {
        if (m_fallback_initialized)
            return;

        // Import the fallback module once
        m_fallback_module = nb::module_::import_("slangpy.torchintegration.bridge_fallback");

        // Cache all function handles - these are looked up once and reused
        m_py_is_tensor = m_fallback_module.attr("is_tensor");
        m_py_extract_tensor_info = m_fallback_module.attr("extract_tensor_info");
        m_py_get_signature = m_fallback_module.attr("get_signature");
        m_py_get_current_cuda_stream = m_fallback_module.attr("get_current_cuda_stream");
        m_py_copy_to_buffer = m_fallback_module.attr("copy_to_buffer");
        m_py_copy_from_buffer = m_fallback_module.attr("copy_from_buffer");

        m_fallback_initialized = true;
    }

    // Python fallback implementations (use cached function handles)
    // Note: These are only called when m_fallback_initialized is true
    bool python_is_tensor(PyObject* obj) const { return nb::cast<bool>(m_py_is_tensor(nb::handle(obj))); }

    bool python_extract(PyObject* tensor, TensorBridgeInfo& out) const
    {
        // Call cached function handle directly - no attribute lookup needed
        nb::object result = m_py_extract_tensor_info(nb::handle(tensor));
        nb::dict info = nb::cast<nb::dict>(result);

        // Populate TensorBridgeInfo from dict
        out.data_ptr = reinterpret_cast<void*>(nb::cast<uintptr_t>(info["data_ptr"]));
        out.ndim = nb::cast<int32_t>(info["ndim"]);
        out.numel = nb::cast<int64_t>(info["numel"]);
        out.element_size = nb::cast<int32_t>(info["element_size"]);
        out.is_cuda = nb::cast<bool>(info["is_cuda"]) ? 1 : 0;
        out.is_contiguous = nb::cast<bool>(info["is_contiguous"]) ? 1 : 0;
        out.requires_grad = nb::cast<bool>(info["requires_grad"]) ? 1 : 0;
        out.device_type = nb::cast<int32_t>(info["device_type"]);
        out.device_index = nb::cast<int32_t>(info["device_index"]);
        out.scalar_type = nb::cast<int32_t>(info["scalar_type"]);
        out.storage_offset = nb::cast<int64_t>(info["storage_offset"]);
        out.cuda_stream = reinterpret_cast<void*>(nb::cast<uintptr_t>(info["cuda_stream"]));

        // Extract shape and strides tuples
        nb::tuple shape = nb::cast<nb::tuple>(info["shape"]);
        nb::tuple strides = nb::cast<nb::tuple>(info["strides"]);
        for (int i = 0; i < out.ndim && i < TENSOR_BRIDGE_MAX_DIMS; i++) {
            out.shape[i] = nb::cast<int64_t>(shape[i]);
            out.strides[i] = nb::cast<int64_t>(strides[i]);
        }

        return true;
    }

    int python_get_signature(PyObject* obj, char* buffer, size_t buffer_size) const
    {
        std::string sig = nb::cast<std::string>(m_py_get_signature(nb::handle(obj)));
        snprintf(buffer, buffer_size, "%s", sig.c_str());
        return 0;
    }

    void* python_get_current_cuda_stream(int device_index) const
    {
        uintptr_t stream = nb::cast<uintptr_t>(m_py_get_current_cuda_stream(device_index));
        return reinterpret_cast<void*>(stream);
    }

    bool python_copy_to_buffer(PyObject* tensor, void* dest, size_t size) const
    {
        return nb::cast<bool>(m_py_copy_to_buffer(nb::handle(tensor), reinterpret_cast<uintptr_t>(dest), size));
    }

    bool python_copy_from_buffer(PyObject* tensor, void* src, size_t size) const
    {
        return nb::cast<bool>(m_py_copy_from_buffer(nb::handle(tensor), reinterpret_cast<uintptr_t>(src), size));
    }

    // Native API state
    const TensorBridgeAPI* m_api = nullptr;
    bool m_initialized = false;

    // Fallback state
    bool m_force_python_fallback = false;
    bool m_fallback_initialized = false;

    // Cached Python objects (module and function handles)
    nb::object m_fallback_module;
    nb::object m_py_is_tensor;
    nb::object m_py_extract_tensor_info;
    nb::object m_py_get_signature;
    nb::object m_py_get_current_cuda_stream;
    nb::object m_py_copy_to_buffer;
    nb::object m_py_copy_from_buffer;
};

} // namespace sgl
