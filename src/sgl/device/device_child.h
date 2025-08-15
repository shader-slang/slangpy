// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"

#include "sgl/core/object.h"

namespace sgl {

class SGL_API DeviceChild : public Object {
    SGL_OBJECT(DeviceChild)
public:
    DeviceChild(ref<Device> device);
    virtual ~DeviceChild();

    Device* device() const { return m_device; }

    /// Release all underlying slang-rhi resources.
    /// This is used as a workaround during shutdown, to ensure all resources are released
    /// when slangpy fails to clean up properly due to reference cycles introduced in Python.
    virtual void _release_rhi_resources() = 0;

    struct MemoryUsage {
        /// The amount of memory in bytes used on the device.
        size_t device{0};
        /// The amount of memory in bytes used on the host.
        size_t host{0};
    };

    /// The memory usage by this resource.
    virtual MemoryUsage memory_usage() const;

protected:
    ref<Device> m_device;
};

} // namespace sgl
