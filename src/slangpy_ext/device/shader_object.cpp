// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/shader_object.h"

SGL_PY_EXPORT(device_shader_object)
{
    using namespace sgl;

    nb::class_<ShaderObject, Object>(m, "ShaderObject", D(ShaderObject));
}
