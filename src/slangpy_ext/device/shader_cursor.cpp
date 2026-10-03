// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "device/shader_cursor.h"

#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/cuda_interop.h"

#include "device/cursor_utils.h"

namespace sgl {
namespace detail {

    // Only the Python representation owns references. Native ShaderCursor values
    // remain trivial, non-owning locations used by the cached dispatch path.
    class PythonShaderCursor : public ShaderCursor {
    public:
        explicit PythonShaderCursor(nb::pointer_and_handle<ShaderObject> object)
            : ShaderCursor(object.p)
            , m_owner(nb::borrow<nb::object>(object.h))
        {
            object.p->retain_rhi_shader_object();
        }

        explicit PythonShaderCursor(const ShaderCursor& cursor)
            : ShaderCursor(cursor)
            , m_owner(cursor.shader_object() ? nb::cast(ref<ShaderObject>(cursor.shader_object())) : nb::object())
        {
            if (cursor.shader_object())
                cursor.shader_object()->retain_rhi_shader_object();
        }

        PythonShaderCursor operator[](std::string_view name) const { return child(ShaderCursor::operator[](name)); }
        PythonShaderCursor operator[](uint32_t index) const { return child(ShaderCursor::operator[](index)); }
        PythonShaderCursor find_field(std::string_view name) const { return child(ShaderCursor::find_field(name)); }
        PythonShaderCursor find_element(uint32_t index) const { return child(ShaderCursor::find_element(index)); }
        PythonShaderCursor get_field_by_index(int32_t index) const
        {
            return child(ShaderCursor::get_field_by_index(index));
        }
        PythonShaderCursor find_entry_point(uint32_t index) const
        {
            return child(ShaderCursor::find_entry_point(index));
        }
        PythonShaderCursor dereference() const { return child(ShaderCursor::dereference()); }

        PythonShaderCursor reinterpret(const ref<TypeLayoutReflection>& layout) const
        {
            auto result = child(ShaderCursor::reinterpret(layout->get_slang_type_layout()));
            // Only reinterpretation needs two owners. Ordinary traversal copies
            // one reference, including when that reference owns this pair.
            nb::object object_owner = nb::isinstance<nb::tuple>(m_owner) ? m_owner[0] : m_owner;
            result.m_owner = nb::make_tuple(object_owner, layout);
            return result;
        }

    private:
        PythonShaderCursor child(const ShaderCursor& cursor) const { return PythonShaderCursor(cursor, m_owner); }

        PythonShaderCursor(const ShaderCursor& cursor, nb::object owner)
            : ShaderCursor(cursor)
            , m_owner(std::move(owner))
        {
        }

        nb::object m_owner;
    };

    class ShaderCursorWriteConverterTable : public WriteConverterTable<ShaderCursor> {
    public:
        bool write_custom_value(ShaderCursor& self, nb::object nbval) override
        {
            nb::ndarray<nb::device::cuda> cudaarray;
            if (nb::try_cast(nbval, cudaarray)) {
                self.set_cuda_tensor_view(ndarray_to_cuda_tensor_view(cudaarray));
                return true;
            }

            return false;
        }
    };

    static ShaderCursorWriteConverterTable _writeconv;
} // namespace detail
} // namespace sgl

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

bool type_caster<sgl::ShaderCursor>::from_python(handle source, uint8_t flags, cleanup_list* cleanup) noexcept
{
    type_caster_base<sgl::detail::PythonShaderCursor> caster;
    if (!caster.from_python(source, flags, cleanup))
        return false;
    value = caster.operator sgl::detail::PythonShaderCursor*();
    return true;
}

handle type_caster<sgl::ShaderCursor>::from_cpp(const Type& cursor, rv_policy, cleanup_list*)
{
    // Callbacks receiving native cursors get the same ownership guarantees as
    // explicitly constructed Python cursors.
    return nb::cast(sgl::detail::PythonShaderCursor(cursor)).release();
}

handle type_caster<sgl::ShaderCursor>::from_cpp(const Type* cursor, rv_policy policy, cleanup_list* cleanup)
{
    return cursor ? from_cpp(*cursor, policy, cleanup) : none().release();
}

type_caster<sgl::ShaderCursor>::operator Type&()
{
    raise_next_overload_if_null(value);
    return *value;
}

type_caster<sgl::ShaderCursor>::operator Type&&()
{
    raise_next_overload_if_null(value);
    return static_cast<Type&&>(*value);
}

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

namespace sgl {

void write_shader_cursor(ShaderCursor& cursor, nb::object value)
{
    detail::_writeconv.write(cursor, value);
}

std::function<void(ShaderCursor&, nb::object)> get_shader_cursor_writer(slang::TypeLayoutReflection* type_layout)
{
    return detail::_writeconv.get_writer(type_layout);
}

} // namespace sgl

SGL_PY_EXPORT(device_shader_cursor)
{
    using namespace sgl;

    nb::class_<ShaderOffset>(m, "ShaderOffset", D(ShaderOffset))
        .def_ro("uniform_offset", &ShaderOffset::uniform_offset, D(ShaderOffset, uniform_offset))
        .def_ro("binding_range_index", &ShaderOffset::binding_range_index, D(ShaderOffset, binding_range_index))
        .def_ro("binding_array_index", &ShaderOffset::binding_array_index, D(ShaderOffset, binding_array_index))
        .def("is_valid", &ShaderOffset::is_valid, D(ShaderOffset, is_valid));

    using PythonShaderCursor = detail::PythonShaderCursor;
    nb::class_<PythonShaderCursor> shader_cursor(m, "ShaderCursor", D(ShaderCursor));

    shader_cursor //
        .def(nb::init<nb::pointer_and_handle<ShaderObject>>(), "shader_object"_a, D(ShaderCursor, ShaderCursor))
        .def_prop_ro("_offset", &ShaderCursor::offset, D(ShaderCursor, offset))
        .def("reinterpret", &PythonShaderCursor::reinterpret, "new_layout"_a, D(ShaderCursor, reinterpret))
        .def("dereference", &PythonShaderCursor::dereference, D(ShaderCursor, dereference))
        .def("find_entry_point", &PythonShaderCursor::find_entry_point, "index"_a, D(ShaderCursor, find_entry_point))
        .def(
            "get_field_by_index",
            &PythonShaderCursor::get_field_by_index,
            "field_index"_a,
            D_NA(ShaderCursor, get_field_by_index)
        )
        .def("find_field_index", &ShaderCursor::find_field_index, "name"_a, D_NA(ShaderCursor, find_field_index));

    bind_traversable_cursor(shader_cursor);

    bind_writable_cursor(detail::_writeconv, shader_cursor);
}
