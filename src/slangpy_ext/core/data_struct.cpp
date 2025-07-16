// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/data_struct.h"

SGL_PY_EXPORT(core_data_struct)
{
    using namespace sgl;

    nb::class_<DataStruct, Object> struct_(m, "DataStruct", D(DataStruct));

    nb::sgl_enum<DataStruct::Type>(struct_, "Type", D(DataStruct, Type));
    nb::sgl_enum_flags<DataStruct::Flags>(struct_, "Flags", D(DataStruct, Flags));
    nb::sgl_enum<DataStruct::ByteOrder>(struct_, "ByteOrder", D(DataStruct, ByteOrder));

    nb::class_<DataStruct::Field>(struct_, "Field", D(DataStruct, Field))
        .def_rw("name", &DataStruct::Field::name, D(DataStruct, Field, name))
        .def_rw("type", &DataStruct::Field::type, D(DataStruct, Field, type))
        .def_rw("flags", &DataStruct::Field::flags, D(DataStruct, Field, flags))
        .def_rw("size", &DataStruct::Field::size, D(DataStruct, Field, size))
        .def_rw("offset", &DataStruct::Field::offset, D(DataStruct, Field, offset))
        .def_rw("default_value", &DataStruct::Field::default_value, D(DataStruct, Field, default_value))
        .def("is_integer", &DataStruct::Field::is_integer, D(DataStruct, Field, is_integer))
        .def("is_unsigned", &DataStruct::Field::is_unsigned, D(DataStruct, Field, is_unsigned))
        .def("is_signed", &DataStruct::Field::is_signed, D(DataStruct, Field, is_signed))
        .def("is_float", &DataStruct::Field::is_float, D(DataStruct, Field, is_float))
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("__repr__", &DataStruct::Field::to_string);

    struct_ //
        .def(
            nb::init<bool, DataStruct::ByteOrder>(),
            "pack"_a = false,
            "byte_order"_a = DataStruct::ByteOrder::host,
            D(DataStruct, DataStruct)
        )
        .def(
            "append",
            nb::overload_cast<DataStruct::Field>(&DataStruct::append),
            "field"_a,
            nb::rv_policy::reference,
            D(DataStruct, append)
        )
        .def(
            "append",
            nb::overload_cast<
                std::string_view,
                DataStruct::Type,
                DataStruct::Flags,
                double,
                const DataStruct::Field::BlendList&>(&DataStruct::append),
            "name"_a,
            "type"_a,
            "flags"_a = DataStruct::Flags::none,
            "default_value"_a = 0.0,
            "blend"_a = DataStruct::Field::BlendList(),
            nb::rv_policy::reference,
            D(DataStruct, append, 2)
        )
        .def("has_field", &DataStruct::has_field, "name"_a, D(DataStruct, has_field))
        .def(
            "field",
            nb::overload_cast<std::string_view>(&DataStruct::field),
            "name"_a,
            nb::rv_policy::reference,
            D(DataStruct, field)
        )
        .def(
            "__getitem__",
            [](DataStruct& self, Py_ssize_t i) -> DataStruct::Field&
            {
                i = detail::sanitize_getitem_index(i, self.field_count());
                return self[i];
            },
            nb::rv_policy::reference_internal
        )
        .def("__len__", &DataStruct::field_count)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def_prop_ro("size", &DataStruct::size, D(DataStruct, size))
        .def_prop_ro("alignment", &DataStruct::alignment, D(DataStruct, alignment))
        .def_prop_ro("byte_order", &DataStruct::byte_order, D(DataStruct, byte_order))
        .def_static("type_size", &DataStruct::type_size, D(DataStruct, type_size))
        .def_static("type_range", &DataStruct::type_range, D(DataStruct, type_range))
        .def_static("is_integer", &DataStruct::is_integer, D(DataStruct, is_integer))
        .def_static("is_unsigned", &DataStruct::is_unsigned, D(DataStruct, is_unsigned))
        .def_static("is_signed", &DataStruct::is_signed, D(DataStruct, is_signed))
        .def_static("is_float", &DataStruct::is_float, D(DataStruct, is_float));

    nb::class_<DataStructConverter, Object>(m, "DataStructConverter", D(DataStructConverter))
        .def(
            "__init__",
            [](DataStructConverter* self, const DataStruct* src, const DataStruct* dst)
            { new (self) DataStructConverter(ref<const DataStruct>(src), ref<const DataStruct>(dst)); },
            "src"_a,
            "dst"_a,
            D(DataStructConverter, DataStructConverter)
        )
        .def_prop_ro("src", &DataStructConverter::src, D(DataStructConverter, src))
        .def_prop_ro("dst", &DataStructConverter::dst, D(DataStructConverter, dst))
        .def(
            "convert",
            [](DataStructConverter* self, nb::bytes input) -> nb::bytes
            {
                size_t count = input.size() / self->src()->size();
                std::string output(self->dst()->size() * count, '\0');
                self->convert(input.c_str(), output.data(), count);
                return nb::bytes(output.data(), output.size());
            },
            "input"_a
        );
}
