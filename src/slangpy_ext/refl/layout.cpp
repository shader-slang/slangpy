// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/refl/layout.h"
#include "sgl/refl/type.h"

SGL_PY_EXPORT(native_refl)
{
    namespace refl = sgl::refl;

    nb::module_ native_refl = nb::module_::import_("slangpy.native_refl");

    nb::class_<refl::TypeLayout, sgl::Object>(native_refl, "TypeLayout")
        .def_prop_ro("size", &refl::TypeLayout::size)
        .def_prop_ro("alignment", &refl::TypeLayout::alignment)
        .def_prop_ro("stride", &refl::TypeLayout::stride)
        .def("__repr__", &refl::TypeLayout::to_string);

    nb::class_<refl::Type, sgl::Object> type(native_refl, "Type");
    type.def_prop_ro("name", &refl::Type::name)
        .def_prop_ro("full_name", &refl::Type::full_name)
        .def_prop_ro("element_type", &refl::Type::element_type)
        .def_prop_ro("shape", &refl::Type::shape)
        .def_prop_ro("num_dims", &refl::Type::num_dims)
        .def_prop_ro("is_generic", &refl::Type::is_generic)
        .def_prop_ro("vector_type_name", &refl::Type::vector_type_name)
        .def_prop_ro("uniform_layout", &refl::Type::uniform_layout)
        .def_prop_ro("buffer_layout", &refl::Type::buffer_layout)
        .def_prop_ro("derivative", &refl::Type::derivative)
        .def("__repr__", &refl::Type::to_string);

    nb::class_<refl::UnknownType, refl::Type>(native_refl, "UnknownType");
    nb::class_<refl::VoidType, refl::Type>(native_refl, "VoidType");
    nb::class_<refl::PointerType, refl::Type>(native_refl, "PointerType")
        .def_prop_ro("target_type", &refl::PointerType::target_type)
        .def_prop_ro("slang_scalar_type", &refl::PointerType::slang_scalar_type);
    nb::class_<refl::ScalarType, refl::Type>(native_refl, "ScalarType")
        .def_prop_ro("slang_scalar_type", &refl::ScalarType::slang_scalar_type);
    nb::class_<refl::VectorType, refl::Type>(native_refl, "VectorType")
        .def_prop_ro("num_elements", &refl::VectorType::num_elements)
        .def_prop_ro("scalar_type", &refl::VectorType::scalar_type)
        .def_prop_ro("slang_scalar_type", &refl::VectorType::slang_scalar_type);
    nb::class_<refl::MatrixType, refl::Type>(native_refl, "MatrixType")
        .def_prop_ro("rows", &refl::MatrixType::rows)
        .def_prop_ro("cols", &refl::MatrixType::cols)
        .def_prop_ro("scalar_type", &refl::MatrixType::scalar_type)
        .def_prop_ro("slang_scalar_type", &refl::MatrixType::slang_scalar_type)
        .def_prop_ro("inner_element_type", &refl::MatrixType::inner_element_type);
    nb::class_<refl::ArrayType, refl::Type>(native_refl, "ArrayType")
        .def_prop_ro("num_elements", &refl::ArrayType::num_elements)
        .def_prop_ro("array_shape", &refl::ArrayType::array_shape)
        .def_prop_ro("any_generic_dims", &refl::ArrayType::any_generic_dims)
        .def_prop_ro("inner_element_type", &refl::ArrayType::inner_element_type)
        .def_prop_ro("array_dims", &refl::ArrayType::array_dims);
    nb::class_<refl::StructType, refl::Type>(native_refl, "StructType");
    nb::class_<refl::InterfaceType, refl::Type>(native_refl, "InterfaceType");
    nb::class_<refl::ResourceType, refl::Type>(native_refl, "ResourceType")
        .def_prop_ro("resource_shape", &refl::ResourceType::resource_shape)
        .def_prop_ro("resource_access", &refl::ResourceType::resource_access)
        .def_prop_ro("writable", &refl::ResourceType::writable);
    nb::class_<refl::TextureType, refl::ResourceType>(native_refl, "TextureType")
        .def_prop_ro("texture_dims", &refl::TextureType::texture_dims);
    nb::class_<refl::StructuredBufferType, refl::ResourceType>(native_refl, "StructuredBufferType");
    nb::class_<refl::ByteAddressBufferType, refl::ResourceType>(native_refl, "ByteAddressBufferType");
    nb::class_<refl::DifferentialPairType, refl::Type>(native_refl, "DifferentialPairType")
        .def_prop_ro("primal", &refl::DifferentialPairType::primal);
    nb::class_<refl::RaytracingAccelerationStructureType, refl::Type>(
        native_refl,
        "RaytracingAccelerationStructureType"
    );
    nb::class_<refl::SamplerStateType, refl::Type>(native_refl, "SamplerStateType");

    nb::class_<refl::TensorType, refl::Type> tensor_type(native_refl, "TensorType");
    nb::sgl_enum<refl::TensorType::Kind>(tensor_type, "Kind");
    nb::sgl_enum<refl::TensorType::Access>(tensor_type, "Access");
    tensor_type.def_prop_ro("tensor_kind", &refl::TensorType::tensor_kind)
        .def_prop_ro("access", &refl::TensorType::access)
        .def_prop_ro("readable", &refl::TensorType::readable)
        .def_prop_ro("writable", &refl::TensorType::writable)
        .def_prop_ro("diff_tensor", &refl::TensorType::diff_tensor)
        .def_prop_ro("dims", &refl::TensorType::dims)
        .def_prop_ro("dtype", &refl::TensorType::dtype)
        .def_prop_ro("has_grad_in", &refl::TensorType::has_grad_in)
        .def_prop_ro("has_grad_out", &refl::TensorType::has_grad_out);

    nb::class_<refl::TensorViewType, refl::Type>(native_refl, "TensorViewType")
        .def_prop_ro("dtype", &refl::TensorViewType::dtype);
    nb::class_<refl::DiffTensorViewType, refl::Type>(native_refl, "DiffTensorViewType")
        .def_prop_ro("dtype", &refl::DiffTensorViewType::dtype)
        .def_prop_ro("wrapper_type", &refl::DiffTensorViewType::wrapper_type);
    nb::class_<refl::UnhandledType, refl::Type>(native_refl, "UnhandledType");

    nb::class_<refl::Layout, sgl::Object>(native_refl, "Layout")
        .def(
            "__init__",
            [](refl::Layout* self, nb::object low_level_layout)
            {
                new (self) refl::Layout(
                    sgl::ref<const sgl::ProgramLayout>(nb::cast<const sgl::ProgramLayout*>(low_level_layout))
                );
            },
            "low_level_layout"_a
        )
        .def_prop_ro("generation", &refl::Layout::generation)
        .def_prop_ro("is_valid", &refl::Layout::is_valid)
        .def(
            "find_type",
            [](refl::Layout& self, nb::object type_reflection)
            {
                return self.find_type(
                    sgl::ref<const sgl::TypeReflection>(nb::cast<const sgl::TypeReflection*>(type_reflection))
                );
            },
            "type_reflection"_a
        )
        .def("find_type_by_name", &refl::Layout::find_type_by_name, "name"_a)
        .def("require_type_by_name", &refl::Layout::require_type_by_name, "name"_a)
        .def("scalar_type", &refl::Layout::scalar_type, "scalar_type"_a)
        .def("vector_type", &refl::Layout::vector_type, "scalar_type"_a, "size"_a)
        .def("matrix_type", &refl::Layout::matrix_type, "scalar_type"_a, "rows"_a, "cols"_a)
        .def("array_type", &refl::Layout::array_type, "element_type"_a, "count"_a)
        .def(
            "tensor_type",
            &refl::Layout::tensor_type,
            "element_type"_a,
            "dims"_a,
            "access"_a = refl::TensorType::Access::read_write,
            "tensor_kind"_a = refl::TensorType::Kind::tensor
        )
        .def("tensorview_type", &refl::Layout::tensorview_type, "element_type"_a)
        .def("difftensorview_type", &refl::Layout::difftensorview_type, "element_type"_a)
        .def(
            "on_hot_reload",
            [](refl::Layout& self, nb::object low_level_layout)
            {
                self.on_hot_reload(
                    sgl::ref<const sgl::ProgramLayout>(nb::cast<const sgl::ProgramLayout*>(low_level_layout))
                );
            },
            "low_level_layout"_a
        )
        .def("__repr__", &refl::Layout::to_string);
}
