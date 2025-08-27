// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/fence.h"

namespace sgl {
SGL_DICT_TO_DESC_BEGIN(FenceDesc)
SGL_DICT_TO_DESC_FIELD(initial_value, uint64_t)
SGL_DICT_TO_DESC_FIELD(shared, bool)
SGL_DICT_TO_DESC_END()
} // namespace sgl

SGL_PY_EXPORT(device_fence)
{
    using namespace sgl;

    nb::class_<FenceDesc>(m, "FenceDesc", D(FenceDesc))
        .def(nb::init<>())
        .def("__init__", [](FenceDesc* self, nb::dict dict) { new (self) FenceDesc(dict_to_FenceDesc(dict)); })
        .def_rw("initial_value", &FenceDesc::initial_value, D(FenceDesc, initial_value))
        .def_rw("shared", &FenceDesc::shared, D(FenceDesc, shared));
    nb::implicitly_convertible<nb::dict, FenceDesc>();

    nb::class_<Fence, DeviceChild>(m, "Fence", D(Fence))
        .def_prop_ro("desc", &Fence::desc, D(Fence, desc))
        .def("signal", &Fence::signal, "value"_a = Fence::AUTO, D(Fence, signal))
        .def("wait", &Fence::wait, "value"_a = Fence::AUTO, "timeout_ns"_a = Fence::TIMEOUT_INFINITE, D(Fence, wait))
        .def_prop_ro("current_value", &Fence::current_value, D(Fence, current_value))
        .def_prop_ro("signaled_value", &Fence::signaled_value, D(Fence, signaled_value))
        .def_prop_ro("shared_handle", &Fence::shared_handle, D(Fence, shared_handle))
        .def_prop_ro("native_handle", &Fence::native_handle, D(Fence, native_handle));

    m.attr("Fence").attr("AUTO") = Fence::AUTO;
    m.attr("Fence").attr("TIMEOUT_INFINITE") = Fence::TIMEOUT_INFINITE;
}
