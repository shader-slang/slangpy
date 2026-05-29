// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/cuda_interop.h"

#include "device/cursor_utils.h"

namespace sgl {
namespace detail {

    class ShaderCursorWriteConverterTable : public WriteConverterTable<ShaderCursor> {
    public:
        bool write_value(ShaderCursor& self, nb::object nbval) override
        {
            if (WriteConverterTable<ShaderCursor>::write_value(self, nbval))
                return true;

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

    nb::class_<ShaderCursor> shader_cursor(m, "ShaderCursor", D(ShaderCursor));

    shader_cursor //
        .def(nb::init<ShaderObject*>(), "shader_object"_a, D(ShaderCursor, ShaderCursor))
        .def_prop_ro("_offset", &ShaderCursor::offset, D(ShaderCursor, offset))
        .def(
            "reinterpret",
            [](ShaderCursor& self, ref<TypeLayoutReflection> new_layout)
            {
                return self.reinterpret(new_layout->get_slang_type_layout());
            },
            "new_layout"_a,
            D(ShaderCursor, reinterpret)
        )
        .def("dereference", &ShaderCursor::dereference, D(ShaderCursor, dereference))
        .def("find_entry_point", &ShaderCursor::find_entry_point, "index"_a, D(ShaderCursor, find_entry_point))
        .def(
            "get_field_by_index",
            &ShaderCursor::get_field_by_index,
            "field_index"_a,
            D_NA(ShaderCursor, get_field_by_index)
        )
        .def("find_field_index", &ShaderCursor::find_field_index, "name"_a, D_NA(ShaderCursor, find_field_index));

    bind_traversable_cursor(shader_cursor);

    bind_writable_cursor(detail::_writeconv, shader_cursor);
}
