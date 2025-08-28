// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"

#include "utils/slangpypackedarg.h"

#include <fmt/format.h>

namespace sgl::slangpy {

std::string NativePackedArg::to_string() const
{
    std::string python_object_type_name = "None";
    if (!m_python_object.is_none()) {
        python_object_type_name = nb::cast<std::string>(m_python_object.type().attr("__name__"));
    }

    return fmt::format(
        "NativePackedArg(\n"
        "  python_type = \"{}\",\n"
        "  shader_object = {},\n"
        "  python_object_type = \"{}\"\n"
        ")",
        m_python ? "present" : "None",
        m_shader_object ? "present" : "None",
        python_object_type_name
    );
}

} // namespace sgl::slangpy

SGL_PY_EXPORT(utils_slangpy_packedarg)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = m.attr("slangpy");

    nb::class_<NativePackedArg, NativeObject>(slangpy, "NativePackedArg")
        .def(
            "__init__",
            [](NativePackedArg& self,
               ref<NativeMarshall> python,
               ref<ShaderObject> shader_object,
               nb::object python_object)
            { new (&self) NativePackedArg(std::move(python), std::move(shader_object), python_object); },
            "python"_a,
            "shader_object"_a,
            "python_object"_a,
            D_NA(NativePackedArg, NativePackedArg)
        )
        .def_prop_ro("python", &NativePackedArg::python, D_NA(NativePackedArg, python))
        .def_prop_ro("shader_object", &NativePackedArg::shader_object, D_NA(NativePackedArg, shader_object))
        .def_prop_ro("python_object", &NativePackedArg::python_object, D_NA(NativePackedArg, python_object))
        .def("__repr__", &NativePackedArg::to_string);
}
