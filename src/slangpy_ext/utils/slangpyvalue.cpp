// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "utils/slangpyvalue.h"

namespace sgl {
extern void write_shader_cursor(ShaderCursor& cursor, nb::object value);
extern std::function<void(ShaderCursor&, nb::object)>
get_shader_cursor_writer(slang::TypeLayoutReflection* type_layout);
} // namespace sgl

namespace sgl::slangpy {

void NativeValueMarshall::ensure_cached(ShaderCursor cursor, NativeBoundVariableRuntime* binding) const
{
    if (m_cached.is_valid)
        return;
    ShaderCursor field = cursor[binding->variable_name()]["value"];
    m_cached.value_offset = field.offset();
    m_cached.value_type_layout = field.slang_type_layout();
    m_cached.writer = get_shader_cursor_writer(m_cached.value_type_layout);
    m_cached.is_valid = true;
}

void NativeValueMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(read_back);
    AccessType primal_access = binding->access().first;
    if (!value.is_none() && (primal_access == AccessType::read || primal_access == AccessType::readwrite)) {
        ensure_cached(cursor, binding);
        ShaderCursor value_cursor(cursor.shader_object(), m_cached.value_type_layout, m_cached.value_offset);
        if (m_cached.writer) {
            m_cached.writer(value_cursor, value);
        } else {
            write_shader_cursor(value_cursor, value);
        }
    }
}

} // namespace sgl::slangpy


SGL_PY_EXPORT(utils_slangpy_value)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = m.attr("slangpy");

    nb::class_<NativeValueMarshall, NativeMarshall>(slangpy, "NativeValueMarshall") //
        .def(
            "__init__",
            [](NativeValueMarshall& self)
            {
                new (&self) NativeValueMarshall();
            },
            D_NA(NativeValueMarshall, NativeValueMarshall)
        );
}
