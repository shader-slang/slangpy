// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "nanobind.h"

#include "sgl/core/macros.h"
#include "sgl/core/fwd.h"

#include "sgl/device/fwd.h"

#include "utils/slangpy.h"

namespace sgl::slangpy {

class NativeMarshall;

class NativePackedArg : public NativeObject {
public:
    NativePackedArg(ref<NativeMarshall> python, ref<ShaderObject> shader_object)
        : m_python(std::move(python))
        , m_shader_object(std::move(shader_object))
    {
    }

    virtual ~NativePackedArg() = default;

    /// Get the Python marshall.
    ref<NativeMarshall> python() const { return m_python; }

    /// Get the shader object.
    ref<ShaderObject> shader_object() const { return m_shader_object; }

private:
    ref<NativeMarshall> m_python;
    ref<ShaderObject> m_shader_object;
};

} // namespace sgl::slangpy
