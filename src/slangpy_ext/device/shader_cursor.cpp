// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/resource.h"
#include "sgl/device/sampler.h"
#include "sgl/device/raytracing.h"
#include "sgl/device/cuda_interop.h"

#include "device/cursor_utils.h"

#include <typeindex>
#include <unordered_map>

namespace sgl {
namespace detail {

    class ShaderCursorWriteConverterTable : public WriteConverterTable<ShaderCursor> {
    public:
#define add_type(c_type_name, set_func_name)                                                                           \
    m_write_table[typeid(c_type_name)] = [](ShaderCursor& self, nb::object nbval)                                      \
    {                                                                                                                  \
        self.set_func_name(nb::cast<ref<c_type_name>>(nbval));                                                         \
        return true;                                                                                                   \
    };

        ShaderCursorWriteConverterTable()
            : WriteConverterTable<ShaderCursor>()
        {
            add_type(ShaderObject, set_object);
            add_type(Buffer, set_buffer);
            add_type(BufferView, set_buffer_view);
            add_type(Texture, set_texture);
            add_type(TextureView, set_texture_view);
            add_type(Sampler, set_sampler);
            add_type(AccelerationStructure, set_acceleration_structure);
            m_write_table[typeid(DescriptorHandle)] = [](ShaderCursor& self, nb::object nbval)
            {
                self.set_descriptor_handle(nb::cast<DescriptorHandle>(nbval));
                return true;
            };
        }

        bool write_value(ShaderCursor& self, nb::object nbval) override
        {
            if (WriteConverterTable<ShaderCursor>::write_value(self, nbval))
                return true;

            nb::handle type = nbval.type();
            if (nb::type_check(type)) {
                const std::type_info& ti = nb::type_info(type);
                auto it = m_write_table.find(ti);
                if (it != m_write_table.end()) {
                    return it->second(self, nbval);
                }
            }

            nb::ndarray<nb::device::cuda> cudaarray;
            if (nb::try_cast(nbval, cudaarray)) {
                self.set_cuda_tensor_view(ndarray_to_cuda_tensor_view(cudaarray));
                return true;
            }

            return false;
        }

    private:
        std::unordered_map<std::type_index, std::function<bool(ShaderCursor&, nb::object)>> m_write_table;
    };

    static ShaderCursorWriteConverterTable _writeconv;
} // namespace detail

void write_shader_cursor(ShaderCursor& cursor, nb::object value)
{
    detail::_writeconv.write(cursor, value);
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
        .def("dereference", &ShaderCursor::dereference, D(ShaderCursor, dereference))
        .def("find_entry_point", &ShaderCursor::find_entry_point, "index"_a, D(ShaderCursor, find_entry_point));

    bind_traversable_cursor(shader_cursor);

    bind_writable_cursor(detail::_writeconv, shader_cursor);
}
