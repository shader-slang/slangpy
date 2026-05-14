// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/refl/layout.h"

SGL_PY_EXPORT(native_refl)
{
    namespace refl = sgl::refl;

    nb::module_ native_refl = nb::module_::import_("slangpy.native_refl");

    nb::class_<refl::Layout, sgl::Object>(native_refl, "Layout")
        .def(
            "__init__",
            [](refl::Layout* self, nb::object low_level_layout)
            {
                new (self) refl::Layout(
                    sgl::ref<const sgl::ProgramLayout>(nb::cast<const sgl::ProgramLayout*>(low_level_layout))
                );
            },
            "low_level_layout"_a
        )
        .def_prop_ro("generation", &refl::Layout::generation)
        .def_prop_ro("is_valid", &refl::Layout::is_valid)
        .def(
            "on_hot_reload",
            [](refl::Layout& self, nb::object low_level_layout)
            {
                self.on_hot_reload(
                    sgl::ref<const sgl::ProgramLayout>(nb::cast<const sgl::ProgramLayout*>(low_level_layout))
                );
            },
            "low_level_layout"_a
        )
        .def("__repr__", &refl::Layout::to_string);
}
