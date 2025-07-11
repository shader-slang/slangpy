// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/native_handle.h"

SGL_PY_EXPORT(device_native_handle)
{
    using namespace sgl;

    nb::sgl_enum<NativeHandleType>(m, "NativeHandleType", D(NativeHandleType));

    nb::class_<NativeHandle>(m, "NativeHandle", D(NativeHandle))
        .def_prop_ro("type", &NativeHandle::type, D(NativeHandle, type))
        .def_prop_ro("value", &NativeHandle::value, D(NativeHandle, value))
        .def("__bool__", &NativeHandle::is_valid)
        .def("__repr__", &NativeHandle::to_string);
}
