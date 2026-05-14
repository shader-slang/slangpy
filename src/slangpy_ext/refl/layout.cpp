// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/refl/layout.h"
#include "sgl/refl/type.h"

SGL_PY_EXPORT(native_refl)
{
    namespace refl = sgl::refl;

    nb::module_ native_refl = nb::module_::import_("slangpy.native_refl");

    nb::class_<refl::TypeLayout, sgl::Object>(native_refl, "TypeLayout", D_NA(TypeLayout))
        .def_prop_ro("size", &refl::TypeLayout::size, D_NA(TypeLayout, size))
        .def_prop_ro("alignment", &refl::TypeLayout::alignment, D_NA(TypeLayout, alignment))
        .def_prop_ro("stride", &refl::TypeLayout::stride, D_NA(TypeLayout, stride))
        .def("__repr__", &refl::TypeLayout::to_string, D_NA(TypeLayout, to_string));

    nb::class_<refl::Type, sgl::Object> type(native_refl, "Type", D_NA(Type));
    type.def_prop_ro("name", &refl::Type::name, D_NA(Type, name))
        .def_prop_ro("full_name", &refl::Type::full_name, D_NA(Type, full_name))
        .def_prop_ro("element_type", &refl::Type::element_type, D_NA(Type, element_type))
        .def_prop_ro("shape", &refl::Type::shape, D_NA(Type, shape))
        .def_prop_ro("num_dims", &refl::Type::num_dims, D_NA(Type, num_dims))
        .def_prop_ro("is_generic", &refl::Type::is_generic, D_NA(Type, is_generic))
        .def_prop_ro("vector_type_name", &refl::Type::vector_type_name, D_NA(Type, vector_type_name))
        .def_prop_ro("uniform_layout", &refl::Type::uniform_layout, D_NA(Type, uniform_layout))
        .def_prop_ro("buffer_layout", &refl::Type::buffer_layout, D_NA(Type, buffer_layout))
        .def_prop_ro("derivative", &refl::Type::derivative, D_NA(Type, derivative))
        .def("__repr__", &refl::Type::to_string, D_NA(Type, to_string));

    nb::class_<refl::UnknownType, refl::Type>(native_refl, "UnknownType", D_NA(UnknownType));

    nb::class_<refl::VoidType, refl::Type>(native_refl, "VoidType", D_NA(VoidType));

    nb::class_<refl::PointerType, refl::Type>(native_refl, "PointerType", D_NA(PointerType))
        .def_prop_ro("target_type", &refl::PointerType::target_type, D_NA(PointerType, target_type))
        .def_prop_ro("slang_scalar_type", &refl::PointerType::slang_scalar_type, D_NA(PointerType, slang_scalar_type));

    nb::class_<refl::ScalarType, refl::Type>(native_refl, "ScalarType", D_NA(ScalarType))
        .def_prop_ro("slang_scalar_type", &refl::ScalarType::slang_scalar_type, D_NA(ScalarType, slang_scalar_type));

    nb::class_<refl::VectorType, refl::Type>(native_refl, "VectorType", D_NA(VectorType))
        .def_prop_ro("num_elements", &refl::VectorType::num_elements, D_NA(VectorType, num_elements))
        .def_prop_ro("scalar_type", &refl::VectorType::scalar_type, D_NA(VectorType, scalar_type))
        .def_prop_ro("slang_scalar_type", &refl::VectorType::slang_scalar_type, D_NA(VectorType, slang_scalar_type));

    nb::class_<refl::MatrixType, refl::Type>(native_refl, "MatrixType", D_NA(MatrixType))
        .def_prop_ro("rows", &refl::MatrixType::rows, D_NA(MatrixType, rows))
        .def_prop_ro("cols", &refl::MatrixType::cols, D_NA(MatrixType, cols))
        .def_prop_ro("scalar_type", &refl::MatrixType::scalar_type, D_NA(MatrixType, scalar_type))
        .def_prop_ro("slang_scalar_type", &refl::MatrixType::slang_scalar_type, D_NA(MatrixType, slang_scalar_type))
        .def_prop_ro("inner_element_type", &refl::MatrixType::inner_element_type, D_NA(MatrixType, inner_element_type));

    nb::class_<refl::ArrayType, refl::Type>(native_refl, "ArrayType", D_NA(ArrayType))
        .def_prop_ro("num_elements", &refl::ArrayType::num_elements, D_NA(ArrayType, num_elements))
        .def_prop_ro("array_shape", &refl::ArrayType::array_shape, D_NA(ArrayType, array_shape))
        .def_prop_ro("any_generic_dims", &refl::ArrayType::any_generic_dims, D_NA(ArrayType, any_generic_dims))
        .def_prop_ro("inner_element_type", &refl::ArrayType::inner_element_type, D_NA(ArrayType, inner_element_type))
        .def_prop_ro("array_dims", &refl::ArrayType::array_dims, D_NA(ArrayType, array_dims));

    nb::class_<refl::StructType, refl::Type>(native_refl, "StructType", D_NA(StructType));

    nb::class_<refl::InterfaceType, refl::Type>(native_refl, "InterfaceType", D_NA(InterfaceType));

    nb::class_<refl::ResourceType, refl::Type>(native_refl, "ResourceType", D_NA(ResourceType))
        .def_prop_ro("resource_shape", &refl::ResourceType::resource_shape, D_NA(ResourceType, resource_shape))
        .def_prop_ro("resource_access", &refl::ResourceType::resource_access, D_NA(ResourceType, resource_access))
        .def_prop_ro("writable", &refl::ResourceType::writable, D_NA(ResourceType, writable));

    nb::class_<refl::TextureType, refl::ResourceType>(native_refl, "TextureType", D_NA(TextureType))
        .def_prop_ro("texture_dims", &refl::TextureType::texture_dims, D_NA(TextureType, texture_dims));

    nb::class_<refl::StructuredBufferType, refl::ResourceType>(
        native_refl,
        "StructuredBufferType",
        D_NA(StructuredBufferType)
    );

    nb::class_<refl::ByteAddressBufferType, refl::ResourceType>(
        native_refl,
        "ByteAddressBufferType",
        D_NA(ByteAddressBufferType)
    );

    nb::class_<refl::DifferentialPairType, refl::Type>(native_refl, "DifferentialPairType", D_NA(DifferentialPairType))
        .def_prop_ro("primal", &refl::DifferentialPairType::primal, D_NA(DifferentialPairType, primal));

    nb::class_<refl::RaytracingAccelerationStructureType, refl::Type>(
        native_refl,
        "RaytracingAccelerationStructureType",
        D_NA(RaytracingAccelerationStructureType)
    );

    nb::class_<refl::SamplerStateType, refl::Type>(native_refl, "SamplerStateType", D_NA(SamplerStateType));

    nb::class_<refl::TensorType, refl::Type> tensor_type(native_refl, "TensorType", D_NA(TensorType));
    nb::sgl_enum<refl::TensorType::Kind>(tensor_type, "Kind", D_NA(TensorType, Kind));
    nb::sgl_enum<refl::TensorType::Access>(tensor_type, "Access", D_NA(TensorType, Access));
    tensor_type.def_prop_ro("tensor_kind", &refl::TensorType::tensor_kind, D_NA(TensorType, tensor_kind))
        .def_prop_ro("access", &refl::TensorType::access, D_NA(TensorType, access))
        .def_prop_ro("readable", &refl::TensorType::readable, D_NA(TensorType, readable))
        .def_prop_ro("writable", &refl::TensorType::writable, D_NA(TensorType, writable))
        .def_prop_ro("diff_tensor", &refl::TensorType::diff_tensor, D_NA(TensorType, diff_tensor))
        .def_prop_ro("dims", &refl::TensorType::dims, D_NA(TensorType, dims))
        .def_prop_ro("dtype", &refl::TensorType::dtype, D_NA(TensorType, dtype))
        .def_prop_ro("has_grad_in", &refl::TensorType::has_grad_in, D_NA(TensorType, has_grad_in))
        .def_prop_ro("has_grad_out", &refl::TensorType::has_grad_out, D_NA(TensorType, has_grad_out));

    nb::class_<refl::TensorViewType, refl::Type>(native_refl, "TensorViewType", D_NA(TensorViewType))
        .def_prop_ro("dtype", &refl::TensorViewType::dtype, D_NA(TensorViewType, dtype));

    nb::class_<refl::DiffTensorViewType, refl::Type>(native_refl, "DiffTensorViewType", D_NA(DiffTensorViewType))
        .def_prop_ro("dtype", &refl::DiffTensorViewType::dtype, D_NA(DiffTensorViewType, dtype))
        .def_prop_ro("wrapper_type", &refl::DiffTensorViewType::wrapper_type, D_NA(DiffTensorViewType, wrapper_type));

    nb::class_<refl::UnhandledType, refl::Type>(native_refl, "UnhandledType", D_NA(UnhandledType));

    nb::class_<refl::Layout, sgl::Object>(native_refl, "Layout", D_NA(Layout))
        .def(
            "__init__",
            [](refl::Layout* self, nb::object low_level_layout)
            {
                new (self) refl::Layout(
                    sgl::ref<const sgl::ProgramLayout>(nb::cast<const sgl::ProgramLayout*>(low_level_layout))
                );
            },
            "low_level_layout"_a,
            D_NA(Layout, Layout)
        )
        .def_prop_ro("generation", &refl::Layout::generation, D_NA(Layout, generation))
        .def_prop_ro("is_valid", &refl::Layout::is_valid, D_NA(Layout, is_valid))
        .def(
            "find_type",
            [](refl::Layout& self, nb::object type_reflection)
            {
                return self.find_type(
                    sgl::ref<const sgl::TypeReflection>(nb::cast<const sgl::TypeReflection*>(type_reflection))
                );
            },
            "type_reflection"_a,
            D_NA(Layout, find_type)
        )
        .def("find_type_by_name", &refl::Layout::find_type_by_name, "name"_a, D_NA(Layout, find_type_by_name))
        .def("require_type_by_name", &refl::Layout::require_type_by_name, "name"_a, D_NA(Layout, require_type_by_name))
        .def("scalar_type", &refl::Layout::scalar_type, "scalar_type"_a, D_NA(Layout, scalar_type))
        .def("vector_type", &refl::Layout::vector_type, "scalar_type"_a, "size"_a, D_NA(Layout, vector_type))
        .def("matrix_type", &refl::Layout::matrix_type, "scalar_type"_a, "rows"_a, "cols"_a, D_NA(Layout, matrix_type))
        .def("array_type", &refl::Layout::array_type, "element_type"_a, "count"_a, D_NA(Layout, array_type))
        .def(
            "tensor_type",
            &refl::Layout::tensor_type,
            "element_type"_a,
            "dims"_a,
            "access"_a = refl::TensorType::Access::read_write,
            "tensor_kind"_a = refl::TensorType::Kind::tensor,
            D_NA(Layout, tensor_type)
        )
        .def("tensorview_type", &refl::Layout::tensorview_type, "element_type"_a, D_NA(Layout, tensorview_type))
        .def(
            "difftensorview_type",
            &refl::Layout::difftensorview_type,
            "element_type"_a,
            D_NA(Layout, difftensorview_type)
        )
        .def(
            "on_hot_reload",
            [](refl::Layout& self, nb::object low_level_layout)
            {
                self.on_hot_reload(
                    sgl::ref<const sgl::ProgramLayout>(nb::cast<const sgl::ProgramLayout*>(low_level_layout))
                );
            },
            "low_level_layout"_a,
            D_NA(Layout, on_hot_reload)
        )
        .def("__repr__", &refl::Layout::to_string, D_NA(Layout, to_string));
}
