// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/string.h"
#include "sgl/device/resource.h"
#include "sgl/device/buffer_cursor.h"

#include "device/cursor_utils.h"


namespace sgl {
namespace detail {
    static ReadConverterTable<BufferElementCursor> _readconv;
    static WriteConverterTable<BufferElementCursor> _writeconv;
} // namespace detail
} // namespace sgl

SGL_PY_EXPORT(device_buffer_cursor)
{
    using namespace sgl;

    nb::class_<BufferElementCursor> buffer_element_cursor(m, "BufferElementCursor", D(BufferElementCursor));

    buffer_element_cursor //
        .def_prop_ro("_offset", &BufferElementCursor::offset, D(BufferElementCursor, offset))
        .def(
            "set_data",
            [](BufferElementCursor& self, nb::ndarray<nb::device::cpu> data)
            {
                SGL_CHECK(is_ndarray_contiguous(data), "data is not contiguous");
                self.set_data(data.data(), data.nbytes());
            },
            "data"_a,
            D(BufferElementCursor, set_data)
        )
        .def(
            "__dir__",
            [](const BufferElementCursor& self)
            {
                // Overridden dir function contains the fields we want to expose,
                // plus the fields of the struct (if it is one).
                SGL_UNUSED(self);
                std::vector<std::string> attributes;
                attributes.push_back("_offset");
                attributes.push_back("_type");
                attributes.push_back("_type_layout");
                if (self.type()->kind() == TypeReflection::Kind::struct_) {
                    for (uint32_t i = 0; i < self.type()->field_count(); i++) {
                        auto field = self.type()->get_field_by_index(i);
                        attributes.push_back(field->name());
                    }
                }
                return attributes;
            }
        )
        .def(
            "__repr__",
            [](const BufferElementCursor& self)
            {
                // __repr__ function basically reads the value into a nanobind object and
                // calls its __repr__ function.
                auto val = nb::cast<std::string>(detail::_readconv.read(self).attr("__repr__")());
                if (self.is_valid())
                    val += fmt::format(" [{}]", self.type()->full_name());
                return val;
            }
        );

    // Bind traversal functions.
    bind_traversable_cursor(buffer_element_cursor);

    // Bind read and write functions
    bind_readable_cursor(detail::_readconv, buffer_element_cursor);
    bind_writable_cursor(detail::_writeconv, buffer_element_cursor);

    // Interface to simpler root cursor object that maps to the larger buffer.
    nb::class_<BufferCursor, Object>(m, "BufferCursor", D(BufferCursor)) //
        .def(nb::init<ref<TypeLayoutReflection>, size_t>(), "element_layout"_a, "size"_a, D(BufferCursor, BufferCursor))
        .def(
            nb::init<ref<TypeLayoutReflection>, ref<Buffer>, bool>(),
            "element_layout"_a,
            "buffer_resource"_a,
            "load_before_write"_a = true,
            D(BufferCursor, BufferCursor)
        )
        .def(
            nb::init<ref<TypeLayoutReflection>, ref<Buffer>, size_t, size_t, bool>(),
            "element_layout"_a,
            "buffer_resource"_a,
            "size"_a,
            "offset"_a,
            "load_before_write"_a = true,
            D(BufferCursor, BufferCursor)
        )
        .def_prop_ro("element_type_layout", &BufferCursor::element_type_layout, D(BufferCursor, element_type_layout))
        .def_prop_ro("element_type", &BufferCursor::element_type, D(BufferCursor, element_type))
        .def("find_element", &BufferCursor::find_element, "index"_a, D(BufferCursor, find_element))
        .def_prop_ro("element_count", &BufferCursor::element_count, D(BufferCursor, element_count))
        .def_prop_ro("element_size", &BufferCursor::element_size, D(BufferCursor, element_size))
        .def_prop_ro("element_stride", &BufferCursor::element_stride, D(BufferCursor, element_stride))
        .def_prop_ro("size", &BufferCursor::size, D(BufferCursor, size))
        .def_prop_ro("is_loaded", &BufferCursor::is_loaded, D(BufferCursor, is_loaded))
        .def("load", &BufferCursor::load, D(BufferCursor, load))
        .def("apply", &BufferCursor::apply, D(BufferCursor, apply))
        .def_prop_ro("resource", &BufferCursor::resource, D(BufferCursor, resource))
        .def("__getitem__", [](BufferCursor& self, int index) { return self[index]; })
        .def("__len__", [](BufferCursor& self) { return self.element_count(); })
        .def(
            "write_from_numpy",
            [](BufferCursor& self, nb::object nbval) { detail::_writeconv.write_from_numpy(self, nbval); },
            "data"_a
        )
        .def(
            "to_numpy",
            [](BufferCursor& self)
            {
                size_t data_size = self.size();
                void* data = new uint8_t[data_size];
                self.read_data(0, data, data_size);
                nb::capsule owner(data, [](void* p) noexcept { delete[] reinterpret_cast<uint8_t*>(p); });
                size_t shape[1] = {data_size};
                return nb::ndarray<
                    nb::numpy>(data, 1, shape, owner, nullptr, nb::dtype<uint8_t>(), nb::device::cpu::value);
            }
        )
        .def(
            "copy_from_numpy",
            [](BufferCursor& self, nb::ndarray<nb::numpy> data)
            {
                SGL_CHECK(is_ndarray_contiguous(data), "numpy array is not contiguous");

                size_t buffer_size = self.size();
                size_t data_size = data.nbytes();
                SGL_CHECK(
                    data_size <= buffer_size,
                    "numpy array is larger than the buffer ({} > {})",
                    data_size,
                    buffer_size
                );

                self.write_data(0, data.data(), data_size);
            },
            "data"_a
        )
        .def(
            "__dir__",
            [](BufferCursor& self)
            {
                // Overridden dir function contains the fields we want to expose,
                // plus a set of [x:y] fields so it shows up in the VS watch
                // window elegantly (only up to field 1000).
                size_t first = 0;
                std::vector<std::string> attributes;
                attributes.push_back("element_type_layout");
                attributes.push_back("element_type");
                attributes.push_back("size");
                attributes.push_back("element_count");
                attributes.push_back("element_size");
                attributes.push_back("is_loaded");
                attributes.push_back("resource");
                while (first < self.element_count() && first < 1000) {
                    size_t last = std::min(first + 100, self.element_count());
                    attributes.push_back(fmt::format("[{}:{}]", first, last - 1));
                    first += 100;
                }
                return attributes;
            }
        )
        .def(
            "__getattr__",
            [](BufferCursor& self, std::string_view name)
            {
                // Overridden getattr by name function checks if one
                // of the [x:y] accessors is being used and returns the
                // corresponding array of elements if so.
                if (name[0] == '[' && name[name.length() - 1] == ']') {
                    uint32_t first = 0;
                    uint32_t last = 0;
                    auto parts = string::split(name, ":");
                    if (parts.size() == 2) {
                        first = std::stoul(parts[0].substr(1));
                        last = std::stoul(parts[1].substr(0, parts[1].length() - 1));
                    }
                    std::vector<BufferElementCursor> res;
                    for (uint32_t i = first; i <= last; i++) {
                        res.push_back(self[i]);
                    }
                    return nb::cast(res);
                } else {
                    return nb::none();
                }
            }
        );
}
