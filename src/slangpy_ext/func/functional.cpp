// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/func/base_module.h"
#include "sgl/func/base_struct.h"

#include <utility>

SGL_PY_EXPORT(native_func)
{
    namespace func = sgl::func;
    namespace refl = sgl::refl;

    nb::module_ native_func = nb::module_::import_("slangpy.native_func");

    nb::class_<func::BaseModule, sgl::Object>(native_func, "BaseModule", D_NA(BaseModule))
        .def(
            "__init__",
            [](func::BaseModule* self, nb::object module, nb::object layout)
            {
                new (self) func::BaseModule(
                    sgl::ref<sgl::SlangModule>(nb::cast<sgl::SlangModule*>(module)),
                    nb::cast<sgl::ref<refl::Layout>>(layout)
                );
            },
            "module"_a,
            "layout"_a,
            D_NA(BaseModule, BaseModule)
        )
        .def(
            "on_hot_reload",
            [](func::BaseModule& self, nb::object module, nb::object low_level_layout)
            {
                self.on_hot_reload(
                    sgl::ref<sgl::SlangModule>(nb::cast<sgl::SlangModule*>(module)),
                    sgl::ref<const sgl::ProgramLayout>(nb::cast<const sgl::ProgramLayout*>(low_level_layout))
                );
            },
            "module"_a,
            "low_level_layout"_a,
            D_NA(BaseModule, on_hot_reload)
        )
        .def("__repr__", &func::BaseModule::to_string, D_NA(BaseModule, to_string));

    nb::class_<func::BaseStruct, sgl::Object>(native_func, "BaseStruct", D_NA(BaseStruct))
        .def(
            "__init__",
            [](func::BaseStruct* self, nb::object module, nb::object type)
            {
                sgl::ref<func::BaseModule> base_module(nb::cast<func::BaseModule*>(module));
                new (self) func::BaseStruct(std::move(base_module), nb::cast<sgl::ref<refl::Type>>(type));
            },
            "module"_a,
            "type"_a,
            D_NA(BaseStruct, BaseStruct)
        )
        .def(
            "on_hot_reload",
            [](func::BaseStruct& self, nb::object type)
            {
                sgl::ref<refl::Type> semantic_type = nb::cast<sgl::ref<refl::Type>>(type);
                self.on_hot_reload(sgl::ref<const sgl::TypeReflection>(semantic_type->reflection()));
            },
            "type"_a,
            D_NA(BaseStruct, on_hot_reload)
        )
        .def("__repr__", &func::BaseStruct::to_string, D_NA(BaseStruct, to_string));
}
