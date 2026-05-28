// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/utils/renderdoc.h"
#include "sgl/core/window.h"
#include "sgl/device/device.h"

SGL_PY_EXPORT(utils_renderdoc)
{
    using namespace sgl;

    nb::module_ renderdoc = nb::module_::import_("slangpy.renderdoc");

    renderdoc.def("is_available", &renderdoc::is_available, D(renderdoc, is_available));
    renderdoc.def(
        "start_frame_capture",
        &renderdoc::start_frame_capture,
        "device"_a,
        "window"_a.none() = nb::none(),
        D(renderdoc, start_frame_capture)
    );
    renderdoc.def("end_frame_capture", &renderdoc::end_frame_capture, D(renderdoc, end_frame_capture));
    renderdoc.def("is_frame_capturing", &renderdoc::is_frame_capturing, D(renderdoc, is_frame_capturing));
}
