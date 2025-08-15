// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/input_layout.h"

namespace sgl {
SGL_DICT_TO_DESC_BEGIN(InputElementDesc)
SGL_DICT_TO_DESC_FIELD(semantic_name, std::string)
SGL_DICT_TO_DESC_FIELD(semantic_index, uint32_t)
SGL_DICT_TO_DESC_FIELD(format, Format)
SGL_DICT_TO_DESC_FIELD(offset, size_t)
SGL_DICT_TO_DESC_FIELD(buffer_slot_index, uint32_t)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(VertexStreamDesc)
SGL_DICT_TO_DESC_FIELD(stride, size_t)
SGL_DICT_TO_DESC_FIELD(slot_class, InputSlotClass)
SGL_DICT_TO_DESC_FIELD(instance_data_step_rate, uint32_t)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(InputLayoutDesc)
SGL_DICT_TO_DESC_FIELD_LIST(input_elements, InputElementDesc)
SGL_DICT_TO_DESC_FIELD_LIST(vertex_streams, VertexStreamDesc)
SGL_DICT_TO_DESC_END()

} // namespace sgl

SGL_PY_EXPORT(device_input_layout)
{
    using namespace sgl;

    nb::sgl_enum<InputSlotClass>(m, "InputSlotClass");

    nb::class_<InputElementDesc>(m, "InputElementDesc", D(InputElementDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](InputElementDesc* self, nb::dict dict) { new (self) InputElementDesc(dict_to_InputElementDesc(dict)); }
        )
        .def_rw("semantic_name", &InputElementDesc::semantic_name, D(InputElementDesc, semantic_name))
        .def_rw("semantic_index", &InputElementDesc::semantic_index, D(InputElementDesc, semantic_index))
        .def_rw("format", &InputElementDesc::format, D(InputElementDesc, format))
        .def_rw("offset", &InputElementDesc::offset, D(InputElementDesc, offset))
        .def_rw("buffer_slot_index", &InputElementDesc::buffer_slot_index, D(InputElementDesc, buffer_slot_index));
    nb::implicitly_convertible<nb::dict, InputElementDesc>();

    nb::class_<VertexStreamDesc>(m, "VertexStreamDesc", D(VertexStreamDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](VertexStreamDesc* self, nb::dict dict) { new (self) VertexStreamDesc(dict_to_VertexStreamDesc(dict)); }
        )
        .def_rw("stride", &VertexStreamDesc::stride, D(VertexStreamDesc, stride))
        .def_rw("slot_class", &VertexStreamDesc::slot_class, D(VertexStreamDesc, slot_class))
        .def_rw(
            "instance_data_step_rate",
            &VertexStreamDesc::instance_data_step_rate,
            D(VertexStreamDesc, instance_data_step_rate)
        );
    nb::implicitly_convertible<nb::dict, VertexStreamDesc>();

    nb::class_<InputLayoutDesc>(m, "InputLayoutDesc", D(InputLayoutDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](InputLayoutDesc* self, nb::dict dict) { new (self) InputLayoutDesc(dict_to_InputLayoutDesc(dict)); }
        )
        .def_rw("input_elements", &InputLayoutDesc::input_elements, D(InputLayoutDesc, input_elements))
        .def_rw("vertex_streams", &InputLayoutDesc::vertex_streams, D(InputLayoutDesc, vertex_streams));
    nb::implicitly_convertible<nb::dict, InputLayoutDesc>();

    nb::class_<InputLayout, DeviceChild>(m, "InputLayout", D(InputLayout))
        .def_prop_ro("desc", &InputLayout::desc, D(InputLayout, desc));
}
