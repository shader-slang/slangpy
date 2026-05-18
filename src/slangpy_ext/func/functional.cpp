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
        .def_prop_ro(
            "device_module",
            [](func::BaseModule& self)
            {
                return sgl::ref<sgl::SlangModule>(self.module());
            },
            D_NA(BaseModule, device_module)
        )
        .def_prop_ro(
            "layout",
            [](func::BaseModule& self)
            {
                return sgl::ref<refl::Layout>(self.layout());
            },
            D_NA(BaseModule, layout)
        )
        .def_prop_ro(
            "session",
            [](func::BaseModule& self)
            {
                return self.module()->session();
            },
            D_NA(BaseModule, session)
        )
        .def_prop_ro("device", &func::BaseModule::device, D_NA(BaseModule, device))
        .def_prop_ro("name", &func::BaseModule::name, D_NA(BaseModule, name))
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
                sgl::ref<refl::Type> reflection_type = nb::cast<sgl::ref<refl::Type>>(type);
                self.on_hot_reload(sgl::ref<const sgl::TypeReflection>(reflection_type->reflection()));
            },
            "type"_a,
            D_NA(BaseStruct, on_hot_reload)
        )
        .def_prop_ro(
            "module",
            [](func::BaseStruct& self)
            {
                return sgl::ref<func::BaseModule>(self.module());
            },
            D_NA(BaseStruct, module)
        )
        .def_prop_ro(
            "layout",
            [](func::BaseStruct& self)
            {
                return sgl::ref<refl::Layout>(self.layout());
            },
            D_NA(BaseStruct, layout)
        )
        .def_prop_ro(
            "program",
            [](func::BaseStruct& self)
            {
                return sgl::ref<refl::Layout>(self.layout());
            },
            D_NA(BaseStruct, program)
        )
        .def_prop_ro("type", &func::BaseStruct::type, D_NA(BaseStruct, type))
        .def_prop_ro("struct", &func::BaseStruct::type, D_NA(BaseStruct, struct))
        .def_prop_ro(
            "type_reflection",
            [](func::BaseStruct& self)
            {
                return sgl::ref<const sgl::TypeReflection>(self.reflection());
            },
            D_NA(BaseStruct, type_reflection)
        )
        .def_prop_ro("name", &func::BaseStruct::name, D_NA(BaseStruct, name))
        .def_prop_ro("full_name", &func::BaseStruct::full_name, D_NA(BaseStruct, full_name))
        .def_prop_ro("shape", &func::BaseStruct::shape, D_NA(BaseStruct, shape))
        .def("__repr__", &func::BaseStruct::to_string, D_NA(BaseStruct, to_string));
}
