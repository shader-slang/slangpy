// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/refl/layout.h"

#include <fmt/format.h>

#include <utility>

namespace sgl::refl {

Layout::Layout(ref<const sgl::ProgramLayout> low_level_layout)
    : m_low_level_layout(std::move(low_level_layout))
{
    SGL_CHECK(m_low_level_layout, "Layout requires a low-level layout");
}

ref<Type> Layout::find_type(ref<const TypeReflection> reflection)
{
    if (!reflection)
        return nullptr;

    if (auto it = m_types_by_reflection.find(reflection.get()); it != m_types_by_reflection.end())
        return it->second;

    ref<Type> type = create_builtin_type(*this, reflection);
    if (!type)
        type = make_ref<UnhandledType>(ref(this), reflection);

    m_types_by_reflection[reflection.get()] = type;
    m_types_by_name[type->full_name()] = type;
    return type;
}

ref<Type> Layout::find_type_by_name(std::string_view name)
{
    if (auto it = m_types_by_name.find(std::string(name)); it != m_types_by_name.end())
        return it->second;

    std::string type_name(name);
    ref<const TypeReflection> reflection = m_low_level_layout->find_type_by_name(type_name.c_str());
    if (!reflection)
        return nullptr;

    ref<Type> type = find_type(std::move(reflection));
    m_types_by_name[type_name] = type;
    return type;
}

ref<Type> Layout::require_type_by_name(std::string_view name)
{
    ref<Type> type = find_type_by_name(name);
    SGL_CHECK(type, "Type '{}' not found", name);
    return type;
}

ref<ScalarType> Layout::scalar_type(TypeReflection::ScalarType scalar_type)
{
    return dynamic_ref_cast<ScalarType>(require_type_by_name(name_for_scalar_type(scalar_type)));
}

ref<VectorType> Layout::vector_type(TypeReflection::ScalarType scalar_type, int size)
{
    return dynamic_ref_cast<VectorType>(
        require_type_by_name(fmt::format("vector<{},{}>", name_for_scalar_type(scalar_type), size))
    );
}

ref<MatrixType> Layout::matrix_type(TypeReflection::ScalarType scalar_type, int rows, int cols)
{
    return dynamic_ref_cast<MatrixType>(
        require_type_by_name(fmt::format("matrix<{},{},{}>", name_for_scalar_type(scalar_type), rows, cols))
    );
}

ref<ArrayType> Layout::array_type(ref<Type> element_type, int count)
{
    SGL_CHECK(element_type, "Array type requires an element type");
    if (count > 0)
        return dynamic_ref_cast<ArrayType>(
            require_type_by_name(fmt::format("{}[{}]", element_type->full_name(), count))
        );
    return dynamic_ref_cast<ArrayType>(require_type_by_name(fmt::format("{}[]", element_type->full_name())));
}

ref<TensorType>
Layout::tensor_type(ref<Type> element_type, int dims, TensorType::Access access, TensorType::Kind tensor_kind)
{
    SGL_CHECK(element_type, "Tensor type requires an element type");
    return dynamic_ref_cast<TensorType>(
        require_type_by_name(TensorType::build_tensor_name(*element_type, dims, access, tensor_kind))
    );
}

ref<TensorViewType> Layout::tensorview_type(ref<Type> element_type)
{
    SGL_CHECK(element_type, "TensorView type requires an element type");
    return dynamic_ref_cast<TensorViewType>(find_type_by_name(TensorViewType::build_tensorview_name(*element_type)));
}

ref<DiffTensorViewType> Layout::difftensorview_type(ref<Type> element_type)
{
    SGL_CHECK(element_type, "DiffTensorView type requires an element type");
    return dynamic_ref_cast<DiffTensorViewType>(
        find_type_by_name(DiffTensorViewType::build_difftensorview_name(*element_type))
    );
}

std::optional<GenericArgs> Layout::get_resolved_generic_args(const TypeReflection* type)
{
    SGL_CHECK(type, "Generic argument lookup requires a type");
    return parse_generic_args(*this, type);
}

void Layout::on_hot_reload(ref<const sgl::ProgramLayout> low_level_layout)
{
    SGL_CHECK(low_level_layout, "Layout hot reload requires a low-level layout");
    m_low_level_layout = std::move(low_level_layout);
    m_types_by_reflection.clear();
    m_types_by_name.clear();
    ++m_generation;
}

std::string Layout::to_string() const
{
    return fmt::format("refl::Layout(generation={}, valid={})", m_generation, is_valid());
}

} // namespace sgl::refl
