// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/ui/ui.h"
#include "sgl/ui/widgets.h"

#include "sgl/core/input.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"

#undef D
#define D(...) DOC(sgl, ui, __VA_ARGS__)

namespace sgl {

template<>
struct GcHelper<ui::Context> {
    void traverse(ui::Context*, GcVisitor& visitor) { visitor("screen"); }
    void clear(ui::Context*) { }
};

} // namespace sgl

SGL_PY_EXPORT(ui)
{
    using namespace sgl;

    nb::module_ ui = nb::module_::import_("slangpy.ui");

    nb::class_<ui::Context, Object>(ui, "Context", gc_helper_type_slots<ui::Context>(), D(Context))
        .def(nb::init<ref<Device>>(), "device"_a)
        .def("begin_frame", &ui::Context::begin_frame, "width"_a, "height"_a, D(Context, begin_frame))
        .def(
            "end_frame",
            nb::overload_cast<TextureView*, CommandEncoder*>(&ui::Context::end_frame),
            "texture_view"_a,
            "command_encoder"_a,
            D(Context, end_frame)
        )
        .def(
            "end_frame",
            nb::overload_cast<Texture*, CommandEncoder*>(&ui::Context::end_frame),
            "texture"_a,
            "command_encoder"_a,
            D(Context, end_frame, 2)
        )
        .def("handle_keyboard_event", &ui::Context::handle_keyboard_event, "event"_a, D(Context, handle_keyboard_event))
        .def("handle_mouse_event", &ui::Context::handle_mouse_event, "event"_a, D(Context, handle_mouse_event))
        .def_prop_ro("screen", &ui::Context::screen, D(Context, screen));
}
