// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/reflection.h"
#include "sgl/refl/type.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace sgl::refl {

/// Native semantic reflection code must remain Python-free.
/// Bindings and Python-specific adaptation belong in src/slangpy_ext.

/// Transitional native owner for a low-level Slang program layout.
class SGL_API Layout : public Object {
    SGL_OBJECT(Layout)
public:
    explicit Layout(ref<const sgl::ProgramLayout> low_level_layout);

    const sgl::ProgramLayout* low_level_layout() const { return m_low_level_layout.get(); }
    ref<const sgl::ProgramLayout> low_level_layout_ref() const { return m_low_level_layout; }

    uint64_t generation() const { return m_generation; }
    bool is_valid() const { return m_low_level_layout && m_low_level_layout->is_valid(); }

    ref<Type> find_type(ref<const TypeReflection> reflection);
    ref<Type> find_type_by_name(std::string_view name);
    ref<Type> require_type_by_name(std::string_view name);

    ref<ScalarType> scalar_type(TypeReflection::ScalarType scalar_type);
    ref<VectorType> vector_type(TypeReflection::ScalarType scalar_type, int size);
    ref<MatrixType> matrix_type(TypeReflection::ScalarType scalar_type, int rows, int cols);
    ref<ArrayType> array_type(ref<Type> element_type, int count);
    ref<TensorType> tensor_type(
        ref<Type> element_type,
        int dims,
        TensorType::Access access = TensorType::Access::read_write,
        TensorType::Kind tensor_kind = TensorType::Kind::tensor
    );
    ref<TensorViewType> tensorview_type(ref<Type> element_type);
    ref<DiffTensorViewType> difftensorview_type(ref<Type> element_type);

    std::optional<GenericArgs> get_resolved_generic_args(const TypeReflection* type);

    void on_hot_reload(ref<const sgl::ProgramLayout> low_level_layout);

    std::string to_string() const override;

private:
    ref<const sgl::ProgramLayout> m_low_level_layout;
    uint64_t m_generation = 0;
    std::unordered_map<const TypeReflection*, ref<Type>> m_types_by_reflection;
    std::unordered_map<std::string, ref<Type>> m_types_by_name;
};

} // namespace sgl::refl
