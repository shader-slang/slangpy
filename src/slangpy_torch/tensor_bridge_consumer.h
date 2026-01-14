// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// tensor_bridge_consumer.h
//
// This header shows how slangpy_ext can consume the tensor bridge API.
// Include this in your slangpy_ext code.
//
// USAGE:
// 1. At module initialization, call init_tensor_bridge()
// 2. In your hot path, call extract_tensor_info() with the PyObject*
// 3. The function call overhead is ~20-50ns (just a function pointer call)

#pragma once

#include "tensor_bridge_api.h"
#include <nanobind/nanobind.h>
#include <stdexcept>

namespace sgl {

// Global API pointer - initialized once at module load time
inline const TensorBridgeAPI* g_tensor_bridge_api = nullptr;

// Initialize the tensor bridge by importing the Python module
// and extracting the function pointer. Call this once at startup.
inline bool init_tensor_bridge()
{
    if (g_tensor_bridge_api != nullptr) {
        return true; // Already initialized
    }

    try {
        // Import the bridge module
        nanobind::module_ bridge = nanobind::module_::import_("slangpy_torch");

        // Get the API pointer
        nanobind::object api_ptr_obj = bridge.attr("get_api_ptr")();
        uintptr_t api_ptr = nanobind::cast<uintptr_t>(api_ptr_obj);

        g_tensor_bridge_api = reinterpret_cast<const TensorBridgeAPI*>(api_ptr);

        // Verify API version compatibility
        if (g_tensor_bridge_api->api_version != TENSOR_BRIDGE_API_VERSION) {
            g_tensor_bridge_api = nullptr;
            return false;
        }

        // Verify struct size matches
        if (g_tensor_bridge_api->info_struct_size != sizeof(TensorBridgeInfo)) {
            g_tensor_bridge_api = nullptr;
            return false;
        }

        return true;
    } catch (...) {
        g_tensor_bridge_api = nullptr;
        return false;
    }
}

// Check if the bridge is available
inline bool is_tensor_bridge_available()
{
    return g_tensor_bridge_api != nullptr;
}

// Check if a PyObject is a torch.Tensor
// This is a fast type check (~10-20ns)
inline bool is_torch_tensor(PyObject* obj)
{
    if (!g_tensor_bridge_api)
        return false;
    return g_tensor_bridge_api->is_tensor(obj) != 0;
}

// Extract tensor info from a PyObject*
// THIS IS THE HOT PATH FUNCTION - ~20-50ns overhead
// The PyObject* must be a torch.Tensor
inline bool extract_tensor_info(PyObject* tensor_obj, TensorBridgeInfo* out)
{
    if (!g_tensor_bridge_api)
        return false;
    return g_tensor_bridge_api->extract(tensor_obj, out) == 0;
}

// Convenience function that works with nanobind::handle
inline bool extract_tensor_info(nanobind::handle tensor_handle, TensorBridgeInfo* out)
{
    return extract_tensor_info(tensor_handle.ptr(), out);
}

// Get the last error message
inline const char* get_tensor_bridge_error()
{
    if (!g_tensor_bridge_api)
        return "tensor bridge not initialized";
    return g_tensor_bridge_api->get_error();
}

} // namespace sgl
