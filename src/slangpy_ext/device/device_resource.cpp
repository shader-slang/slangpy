// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/device_resource.h"
#include "sgl/device/device.h"

SGL_PY_EXPORT(device_device_resource)
{
    using namespace sgl;

    nb::class_<DeviceResource, Object> device_resource(m, "DeviceResource", D(DeviceResource));

    nb::class_<DeviceResource::MemoryUsage>(device_resource, "MemoryUsage", D(DeviceResource, MemoryUsage))
        .def_ro("device", &DeviceResource::MemoryUsage::device, D(DeviceResource, MemoryUsage, device))
        .def_ro("host", &DeviceResource::MemoryUsage::host, D(DeviceResource, MemoryUsage, host));

    device_resource //
        .def_prop_ro("device", &DeviceResource::device, D(DeviceResource, device))
        .def_prop_ro("memory_usage", &DeviceResource::memory_usage, D(DeviceResource, memory_usage));
}
