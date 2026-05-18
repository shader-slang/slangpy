// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/func/base_struct.h"
#include "sgl/refl/function.h"
#include "sgl/refl/layout.h"
#include "sgl/refl/lookup.h"
#include "sgl/refl/type.h"

#include <utility>

SGL_PY_EXPORT(native_refl)
{
    namespace refl = sgl::refl;
    namespace func = sgl::func;

    nb::module_ native_refl = nb::module_::import_("slangpy.native_refl");

    native_refl.def("get_builtin_layout", &refl::get_builtin_layout, "device"_a, D_NA(get_builtin_layout));
    native_refl.def("name_for_scalar_type", &refl::name_for_scalar_type, "scalar_type"_a, D_NA(name_for_scalar_type));
    nb::sgl_enum<refl::IOType>(native_refl, "IOType", D_NA(IOType));
    native_refl.def(
        "resolve_layout",
        [](sgl::Device* device, nb::object element_type, nb::object explicit_layout)
        {
            refl::Layout* layout = nullptr;
            if (!explicit_layout.is_none())
                layout = nb::cast<refl::Layout*>(explicit_layout);

            refl::Type* type = nullptr;
            if (!element_type.is_none() && nb::try_cast<refl::Type*>(element_type, type))
                return refl::resolve_layout(device, type, layout);

            func::BaseStruct* struct_type = nullptr;
            if (!element_type.is_none() && nb::try_cast<func::BaseStruct*>(element_type, struct_type))
                return refl::resolve_layout(device, struct_type, layout);

            return refl::resolve_layout(device, static_cast<const refl::Type*>(nullptr), layout);
        },
        "device"_a,
        "element_type"_a.none() = nb::none(),
        "layout"_a.none() = nb::none(),
        D_NA(resolve_layout)
    );
    native_refl.def(
        "resolve_element_type",
        [](refl::Layout& layout, nb::object element_type) -> sgl::ref<refl::Type>
        {
            refl::Type* type = nullptr;
            if (nb::try_cast<refl::Type*>(element_type, type))
                return refl::resolve_element_type(&layout, type);

            const sgl::TypeReflection* type_reflection = nullptr;
            if (nb::try_cast<const sgl::TypeReflection*>(element_type, type_reflection))
                return refl::resolve_element_type(&layout, type_reflection);

            const sgl::TypeLayoutReflection* type_layout_reflection = nullptr;
            if (nb::try_cast<const sgl::TypeLayoutReflection*>(element_type, type_layout_reflection))
                return refl::resolve_element_type(&layout, type_layout_reflection);

            func::BaseStruct* struct_type = nullptr;
            if (nb::try_cast<func::BaseStruct*>(element_type, struct_type))
                return refl::resolve_element_type(&layout, struct_type);

            nb::str name;
            if (nb::try_cast<nb::str>(element_type, name))
                return refl::resolve_element_type(&layout, nb::cast<std::string>(name));

            return nullptr;
        },
        "layout"_a,
        "element_type"_a,
        D_NA(resolve_element_type)
    );

    nb::class_<refl::TypeLayout, sgl::Object>(native_refl, "TypeLayout", D_NA(TypeLayout))
        .def_prop_ro(
            "reflection",
            [](refl::TypeLayout& self)
            {
                return sgl::ref<const sgl::TypeLayoutReflection>(self.reflection());
            },
            D_NA(TypeLayout, reflection)
        )
        .def_prop_ro("size", &refl::TypeLayout::size, D_NA(TypeLayout, size))
        .def_prop_ro("alignment", &refl::TypeLayout::alignment, D_NA(TypeLayout, alignment))
        .def_prop_ro("stride", &refl::TypeLayout::stride, D_NA(TypeLayout, stride))
        .def("__repr__", &refl::TypeLayout::to_string, D_NA(TypeLayout, to_string));

    nb::class_<refl::Type, sgl::Object> type(native_refl, "Type", D_NA(Type));
    type.def_prop_ro(
            "layout",
            [](refl::Type& self)
            {
                return sgl::ref<refl::Layout>(self.layout());
            },
            D_NA(Type, layout)
    )
        .def_prop_ro(
            "program",
            [](refl::Type& self)
            {
                return sgl::ref<refl::Layout>(self.layout());
            },
            D_NA(Type, program)
        )
        .def_prop_ro(
            "type_reflection",
            [](refl::Type& self)
            {
                return sgl::ref<const sgl::TypeReflection>(self.reflection());
            },
            D_NA(Type, type_reflection)
        )
        .def_prop_ro("name", &refl::Type::name, D_NA(Type, name))
        .def_prop_ro("full_name", &refl::Type::full_name, D_NA(Type, full_name))
        .def_prop_ro("element_type", &refl::Type::element_type, D_NA(Type, element_type))
        .def_prop_ro("shape", &refl::Type::shape, D_NA(Type, shape))
        .def_prop_ro("num_dims", &refl::Type::num_dims, D_NA(Type, num_dims))
        .def_prop_ro("is_generic", &refl::Type::is_generic, D_NA(Type, is_generic))
        .def_prop_ro("vector_type_name", &refl::Type::vector_type_name, D_NA(Type, vector_type_name))
        .def_prop_ro("uniform_layout", &refl::Type::uniform_layout, D_NA(Type, uniform_layout))
        .def_prop_ro("buffer_layout", &refl::Type::buffer_layout, D_NA(Type, buffer_layout))
        .def_prop_ro("derivative", &refl::Type::derivative, D_NA(Type, derivative))
        .def_prop_ro(
            "differentiable",
            [](refl::Type& self)
            {
                return self.derivative() != nullptr;
            },
            D_NA(Type, differentiable)
        )
        .def_prop_ro(
            "fields",
            [](refl::Type& self)
            {
                nb::dict result;
                for (const auto& [name, field] : self.fields())
                    result[nb::str(name.c_str())] = nb::cast(field);
                return result;
            },
            D_NA(Type, fields)
        )
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
        .def_prop_ro("texture_dims", &refl::TextureType::texture_dims, D_NA(TextureType, texture_dims))
        .def_prop_ro("usage", &refl::TextureType::usage, D_NA(TextureType, usage));

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
        .def_prop_ro("tensor_type", &refl::TensorType::tensor_kind, D_NA(TensorType, tensor_kind))
        .def_prop_ro("access", &refl::TensorType::access, D_NA(TensorType, access))
        .def_prop_ro("readable", &refl::TensorType::readable, D_NA(TensorType, readable))
        .def_prop_ro("writable", &refl::TensorType::writable, D_NA(TensorType, writable))
        .def_prop_ro("diff_tensor", &refl::TensorType::diff_tensor, D_NA(TensorType, diff_tensor))
        .def_prop_ro("difftensor", &refl::TensorType::diff_tensor, D_NA(TensorType, diff_tensor))
        .def_prop_ro("dims", &refl::TensorType::dims, D_NA(TensorType, dims))
        .def_prop_ro("dtype", &refl::TensorType::dtype, D_NA(TensorType, dtype))
        .def_prop_ro("has_grad_in", &refl::TensorType::has_grad_in, D_NA(TensorType, has_grad_in))
        .def_prop_ro("has_grad_out", &refl::TensorType::has_grad_out, D_NA(TensorType, has_grad_out))
        .def_static(
            "build_tensor_name",
            [](refl::Type& element_type, int dims, refl::TensorType::Access access, refl::TensorType::Kind tensor_kind)
            {
                return refl::TensorType::build_tensor_name(element_type, dims, access, tensor_kind);
            },
            "element_type"_a,
            "dims"_a,
            "access"_a = refl::TensorType::Access::read_write,
            "tensor_kind"_a = refl::TensorType::Kind::tensor,
            D_NA(TensorType, build_tensor_name)
        );

    nb::class_<refl::TensorViewType, refl::Type>(native_refl, "TensorViewType", D_NA(TensorViewType))
        .def_prop_ro("dtype", &refl::TensorViewType::dtype, D_NA(TensorViewType, dtype))
        .def_static(
            "build_tensorview_name",
            &refl::TensorViewType::build_tensorview_name,
            "element_type"_a,
            D_NA(TensorViewType, build_tensorview_name)
        );

    nb::class_<refl::DiffTensorViewType, refl::Type>(native_refl, "DiffTensorViewType", D_NA(DiffTensorViewType))
        .def_prop_ro("dtype", &refl::DiffTensorViewType::dtype, D_NA(DiffTensorViewType, dtype))
        .def_prop_ro("wrapper_type", &refl::DiffTensorViewType::wrapper_type, D_NA(DiffTensorViewType, wrapper_type))
        .def_static(
            "build_difftensorview_name",
            &refl::DiffTensorViewType::build_difftensorview_name,
            "element_type"_a,
            D_NA(DiffTensorViewType, build_difftensorview_name)
        );

    nb::class_<refl::UnhandledType, refl::Type>(native_refl, "UnhandledType", D_NA(UnhandledType));

    nb::class_<refl::Variable, sgl::Object>(native_refl, "Variable", D_NA(Variable))
        .def_prop_ro(
            "layout",
            [](refl::Variable& self)
            {
                return sgl::ref<refl::Layout>(self.layout());
            },
            D_NA(Variable, layout)
        )
        .def_prop_ro(
            "program",
            [](refl::Variable& self)
            {
                return sgl::ref<refl::Layout>(self.layout());
            },
            D_NA(Variable, program)
        )
        .def_prop_ro(
            "reflection",
            [](refl::Variable& self)
            {
                return sgl::ref<const sgl::VariableReflection>(self.reflection());
            },
            D_NA(Variable, reflection)
        )
        .def_prop_ro("type", &refl::Variable::type, D_NA(Variable, type))
        .def_prop_ro("name", &refl::Variable::name, D_NA(Variable, name))
        .def_prop_ro("modifiers", &refl::Variable::modifiers, D_NA(Variable, modifiers))
        .def_prop_ro("declaration", &refl::Variable::declaration, D_NA(Variable, declaration))
        .def_prop_ro("io_type", &refl::Variable::io_type, D_NA(Variable, io_type))
        .def_prop_ro("no_diff", &refl::Variable::no_diff, D_NA(Variable, no_diff))
        .def_prop_ro("differentiable", &refl::Variable::differentiable, D_NA(Variable, differentiable))
        .def_prop_ro("derivative", &refl::Variable::derivative, D_NA(Variable, derivative))
        .def("has_modifier", &refl::Variable::has_modifier, "modifier"_a, D_NA(Variable, has_modifier))
        .def("__repr__", &refl::Variable::to_string, D_NA(Variable, to_string));

    nb::class_<refl::Field, refl::Variable>(native_refl, "Field", D_NA(Field));

    nb::class_<refl::Parameter, refl::Variable>(native_refl, "Parameter", D_NA(Parameter))
        .def_prop_ro("index", &refl::Parameter::index, D_NA(Parameter, index))
        .def_prop_ro("has_default", &refl::Parameter::has_default, D_NA(Parameter, has_default));

    nb::class_<refl::Function, sgl::Object>(native_refl, "Function", D_NA(Function))
        .def_prop_ro(
            "layout",
            [](refl::Function& self)
            {
                return sgl::ref<refl::Layout>(self.layout());
            },
            D_NA(Function, layout)
        )
        .def_prop_ro(
            "program",
            [](refl::Function& self)
            {
                return sgl::ref<refl::Layout>(self.layout());
            },
            D_NA(Function, program)
        )
        .def_prop_ro(
            "reflection",
            [](refl::Function& self)
            {
                return sgl::ref<const sgl::FunctionReflection>(self.reflection());
            },
            D_NA(Function, reflection)
        )
        .def_prop_ro("name", &refl::Function::name, D_NA(Function, name))
        .def_prop_ro("full_name", &refl::Function::full_name, D_NA(Function, full_name))
        .def_prop_ro("this_type", &refl::Function::this_type, D_NA(Function, this_type))
        .def_prop_ro("this", &refl::Function::this_type, D_NA(Function, this_type))
        .def_prop_ro("return_type", &refl::Function::return_type, D_NA(Function, return_type))
        .def_prop_ro("parameters", &refl::Function::parameters, D_NA(Function, parameters))
        .def_prop_ro("have_return_value", &refl::Function::have_return_value, D_NA(Function, have_return_value))
        .def_prop_ro("differentiable", &refl::Function::differentiable, D_NA(Function, differentiable))
        .def_prop_ro("mutating", &refl::Function::mutating, D_NA(Function, mutating))
        .def_prop_ro("static", &refl::Function::static_, D_NA(Function, static_))
        .def_prop_ro("is_overloaded", &refl::Function::is_overloaded, D_NA(Function, is_overloaded))
        .def_prop_ro("overloads", &refl::Function::overloads, D_NA(Function, overloads))
        .def_prop_ro("is_constructor", &refl::Function::is_constructor, D_NA(Function, is_constructor))
        .def(
            "specialize_with_arg_types",
            &refl::Function::specialize_with_arg_types,
            "types"_a,
            D_NA(Function, specialize_with_arg_types)
        )
        .def("__repr__", &refl::Function::to_string, D_NA(Function, to_string));

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
        .def_prop_ro(
            "low_level_layout",
            [](refl::Layout& self)
            {
                return sgl::ref<const sgl::ProgramLayout>(self.low_level_layout());
            },
            D_NA(Layout, low_level_layout)
        )
        .def_prop_ro(
            "program_layout",
            [](refl::Layout& self)
            {
                return sgl::ref<const sgl::ProgramLayout>(self.low_level_layout());
            },
            D_NA(Layout, program_layout)
        )
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
        .def(
            "find_function",
            [](refl::Layout& self, nb::object function_reflection, nb::object this_type)
            {
                sgl::ref<refl::Type> reflected_this;
                if (!this_type.is_none())
                    reflected_this = nb::cast<sgl::ref<refl::Type>>(this_type);

                return self.find_function(
                    sgl::ref<const sgl::FunctionReflection>(
                        nb::cast<const sgl::FunctionReflection*>(function_reflection)
                    ),
                    std::move(reflected_this)
                );
            },
            "function_reflection"_a,
            "this_type"_a.none() = nb::none(),
            D_NA(Layout, find_function)
        )
        .def(
            "find_function_by_name",
            &refl::Layout::find_function_by_name,
            "name"_a,
            D_NA(Layout, find_function_by_name)
        )
        .def(
            "require_function_by_name",
            &refl::Layout::require_function_by_name,
            "name"_a,
            D_NA(Layout, require_function_by_name)
        )
        .def(
            "find_function_by_name_in_type",
            &refl::Layout::find_function_by_name_in_type,
            "type"_a,
            "name"_a,
            D_NA(Layout, find_function_by_name_in_type)
        )
        .def(
            "require_function_by_name_in_type",
            &refl::Layout::require_function_by_name_in_type,
            "type"_a,
            "name"_a,
            D_NA(Layout, require_function_by_name_in_type)
        )
        .def(
            "scalar_type",
            [](refl::Layout& self, sgl::TypeReflection::ScalarType scalar_type) -> sgl::ref<refl::Type>
            {
                return self.require_type_by_name(refl::name_for_scalar_type(scalar_type));
            },
            "scalar_type"_a,
            D_NA(Layout, scalar_type)
        )
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
            "get_resolved_generic_args",
            [](refl::Layout& self, nb::object type_reflection) -> nb::object
            {
                const sgl::TypeReflection* type = nullptr;
                refl::Type* semantic_type = nullptr;
                if (nb::try_cast<refl::Type*>(type_reflection, semantic_type))
                    type = semantic_type->reflection();
                else
                    type = nb::cast<const sgl::TypeReflection*>(type_reflection);

                std::optional<refl::GenericArgs> args = self.get_resolved_generic_args(type);
                if (!args)
                    return nb::none();

                nb::list result;
                for (size_t i = 0; i < args->size(); ++i) {
                    const refl::GenericArg& arg = (*args)[i];
                    result.append(arg.is_integer() ? nb::cast(arg.integer()) : nb::cast(arg.type()));
                }
                return result;
            },
            "type_reflection"_a,
            D_NA(Layout, get_resolved_generic_args)
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
