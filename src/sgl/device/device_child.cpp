// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "device_child.h"
#include "device.h"

namespace sgl {

DeviceChild::DeviceChild(ref<Device> device)
    : m_device(std::move(device))
{
    m_device->_register_device_child(this);
}

DeviceChild::~DeviceChild()
{
    m_device->_unregister_device_child(this);
}

DeviceChild::MemoryUsage DeviceChild::memory_usage() const
{
    return {};
}

} // namespace sgl
