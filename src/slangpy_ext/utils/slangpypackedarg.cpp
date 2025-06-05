// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"

#include "utils/slangpypackedarg.h"

SGL_PY_EXPORT(utils_slangpy_packedarg)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = m.attr("slangpy");

    nb::class_<NativePackedArg, NativeObject>(slangpy, "NativePackedArg")
        .def(
            "__init__",
            [](NativePackedArg& self, ref<NativeMarshall> python, ref<ShaderObject> shader_object)
            { new (&self) NativePackedArg(std::move(python), std::move(shader_object)); },
            "python"_a,
            "shader_object"_a,
            D_NA(NativePackedArg, NativePackedArg)
        )
        .def_prop_ro("python", &NativePackedArg::python, D_NA(NativePackedArg, python))
        .def_prop_ro("shader_object", &NativePackedArg::shader_object, D_NA(NativePackedArg, shader_object));
}
