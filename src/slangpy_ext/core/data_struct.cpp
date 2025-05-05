// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/data_struct.h"

SGL_PY_EXPORT(core_data_struct)
{
    using namespace sgl;

    // TODO(slangpy-merge): Struct is also defined in slangpy python part.
    // we should probably rename the c++ Struct type to DataStruct too.
    nb::class_<Struct, Object> struct_(m, "DataStruct", D(Struct));

    nb::sgl_enum<Struct::Type>(struct_, "Type", D(Struct, Type));
    nb::sgl_enum_flags<Struct::Flags>(struct_, "Flags", D(Struct, Flags));
    nb::sgl_enum<Struct::ByteOrder>(struct_, "ByteOrder", D(Struct, ByteOrder));

    nb::class_<Struct::Field>(struct_, "Field", D(Struct, Field))
        .def_rw("name", &Struct::Field::name, D(Struct, Field, name))
        .def_rw("type", &Struct::Field::type, D(Struct, Field, type))
        .def_rw("flags", &Struct::Field::flags, D(Struct, Field, flags))
        .def_rw("size", &Struct::Field::size, D(Struct, Field, size))
        .def_rw("offset", &Struct::Field::offset, D(Struct, Field, offset))
        .def_rw("default_value", &Struct::Field::default_value, D(Struct, Field, default_value))
        .def("is_integer", &Struct::Field::is_integer, D(Struct, Field, is_integer))
        .def("is_unsigned", &Struct::Field::is_unsigned, D(Struct, Field, is_unsigned))
        .def("is_signed", &Struct::Field::is_signed, D(Struct, Field, is_signed))
        .def("is_float", &Struct::Field::is_float, D(Struct, Field, is_float))
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("__repr__", &Struct::Field::to_string);

    struct_ //
        .def(
            nb::init<bool, Struct::ByteOrder>(),
            "pack"_a = false,
            "byte_order"_a = Struct::ByteOrder::host,
            D(Struct, Struct)
        )
        .def(
            "append",
            nb::overload_cast<Struct::Field>(&Struct::append),
            "field"_a,
            nb::rv_policy::reference,
            D(Struct, append)
        )
        .def(
            "append",
            nb::overload_cast<std::string_view, Struct::Type, Struct::Flags, double, const Struct::Field::BlendList&>(
                &Struct::append
            ),
            "name"_a,
            "type"_a,
            "flags"_a = Struct::Flags::none,
            "default_value"_a = 0.0,
            "blend"_a = Struct::Field::BlendList(),
            nb::rv_policy::reference,
            D(Struct, append, 2)
        )
        .def("has_field", &Struct::has_field, "name"_a, D(Struct, has_field))
        .def(
            "field",
            nb::overload_cast<std::string_view>(&Struct::field),
            "name"_a,
            nb::rv_policy::reference,
            D(Struct, field)
        )
        .def(
            "__getitem__",
            [](Struct& self, size_t i) -> Struct::Field&
            {
                if (i >= self.field_count())
                    throw nb::index_error();
                return self[i];
            },
            nb::rv_policy::reference_internal
        )
        .def("__len__", &Struct::field_count)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def_prop_ro("size", &Struct::size, D(Struct, size))
        .def_prop_ro("alignment", &Struct::alignment, D(Struct, alignment))
        .def_prop_ro("byte_order", &Struct::byte_order, D(Struct, byte_order))
        .def_static("type_size", &Struct::type_size, D(Struct, type_size))
        .def_static("type_range", &Struct::type_range, D(Struct, type_range))
        .def_static("is_integer", &Struct::is_integer, D(Struct, is_integer))
        .def_static("is_unsigned", &Struct::is_unsigned, D(Struct, is_unsigned))
        .def_static("is_signed", &Struct::is_signed, D(Struct, is_signed))
        .def_static("is_float", &Struct::is_float, D(Struct, is_float));

    nb::class_<StructConverter, Object>(m, "StructConverter", D(StructConverter))
        .def(
            "__init__",
            [](StructConverter* self, const Struct* src, const Struct* dst)
            { new (self) StructConverter(ref<const Struct>(src), ref<const Struct>(dst)); },
            "src"_a,
            "dst"_a,
            D(StructConverter, StructConverter)
        )
        .def_prop_ro("src", &StructConverter::src, D(StructConverter, src))
        .def_prop_ro("dst", &StructConverter::dst, D(StructConverter, dst))
        .def(
            "convert",
            [](StructConverter* self, nb::bytes input) -> nb::bytes
            {
                size_t count = input.size() / self->src()->size();
                std::string output(self->dst()->size() * count, '\0');
                self->convert(input.c_str(), output.data(), count);
                return nb::bytes(output.data(), output.size());
            },
            "input"_a
        );
}
