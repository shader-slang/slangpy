// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/native_handle_traits.h"
#include "sgl/device/native_handle.h"

SGL_PY_EXPORT(device_native_handle)
{
    using namespace sgl;

    nb::sgl_enum<NativeHandleType>(m, "NativeHandleType", D(NativeHandleType));

    nb::class_<NativeHandle>(m, "NativeHandle", D_NA(NativeHandle))
        .def(nb::init<>())
        .def(nb::init<NativeHandleType, uint64_t>())
        .def_prop_ro("type", &NativeHandle::type, D_NA(NativeHandle, type))
        .def_prop_ro("value", &NativeHandle::value, D_NA(NativeHandle, value))
        .def("__bool__", &NativeHandle::is_valid)
        .def("__repr__", &NativeHandle::to_string)
        .def(
            "__hash__",
            [](const NativeHandle& nh)
            { return std::hash<uint64_t>()(static_cast<uint64_t>(nh.type())) ^ std::hash<uint64_t>()(nh.value()); }
        )
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def_static(
            "from_cuda_stream",
            [](uint64_t stream) { return NativeHandle(reinterpret_cast<CUstream>(stream)); },
            "stream"_a,
            D_NA(NativeHandle, from_cuda_stream)
        );
}
