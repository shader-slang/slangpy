// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/device_child.h"
#include "sgl/device/device.h"

SGL_PY_EXPORT(device_device_child)
{
    using namespace sgl;

    nb::class_<DeviceChild, Object> device_child(m, "DeviceChild", D(DeviceChild));

    nb::class_<DeviceChild::MemoryUsage>(device_child, "MemoryUsage", D(DeviceChild, MemoryUsage))
        .def_ro("device", &DeviceChild::MemoryUsage::device, D(DeviceChild, MemoryUsage, device))
        .def_ro("host", &DeviceChild::MemoryUsage::host, D(DeviceChild, MemoryUsage, host));

    device_child //
        .def_prop_ro("device", &DeviceChild::device, D(DeviceChild, device))
        .def_prop_ro("memory_usage", &DeviceChild::memory_usage, D(DeviceChild, memory_usage));
}
