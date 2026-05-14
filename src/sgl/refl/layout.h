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
    /// Create a semantic layout over an existing low-level Slang program layout.
    explicit Layout(ref<const sgl::ProgramLayout> low_level_layout);

    /// Return the low-level SGL program layout.
    const sgl::ProgramLayout* low_level_layout() const { return m_low_level_layout.get(); }
    /// Return the low-level SGL program layout with reference ownership.
    ref<const sgl::ProgramLayout> low_level_layout_ref() const { return m_low_level_layout; }

    /// Return the layout generation, incremented whenever hot reload replaces the low-level layout.
    uint64_t generation() const { return m_generation; }
    /// Return true if the wrapped low-level layout is still valid.
    bool is_valid() const { return m_low_level_layout && m_low_level_layout->is_valid(); }

    /// Find or create a semantic type for a low-level type reflection.
    ref<Type> find_type(ref<const TypeReflection> reflection);
    /// Find or create a semantic type by reflected type name.
    ref<Type> find_type_by_name(std::string_view name);
    /// Find or create a semantic type by reflected type name, throwing if absent.
    ref<Type> require_type_by_name(std::string_view name);

    /// Return the semantic scalar type for a Slang scalar id.
    ref<ScalarType> scalar_type(TypeReflection::ScalarType scalar_type);
    /// Return the semantic vector type for a scalar id and lane count.
    ref<VectorType> vector_type(TypeReflection::ScalarType scalar_type, int size);
    /// Return the semantic matrix type for a scalar id and shape.
    ref<MatrixType> matrix_type(TypeReflection::ScalarType scalar_type, int rows, int cols);
    /// Return the semantic array type for an element type and element count.
    ref<ArrayType> array_type(ref<Type> element_type, int count);
    /// Return the semantic tensor type for an element type, rank, access mode, and tensor family.
    ref<TensorType> tensor_type(
        ref<Type> element_type,
        int dims,
        TensorType::Access access = TensorType::Access::read_write,
        TensorType::Kind tensor_kind = TensorType::Kind::tensor
    );
    /// Return the semantic TensorView type for an element type.
    ref<TensorViewType> tensorview_type(ref<Type> element_type);
    /// Return the semantic DiffTensorView type for an element type.
    ref<DiffTensorViewType> difftensorview_type(ref<Type> element_type);

    /// Parse and resolve generic arguments from a low-level reflected type.
    std::optional<GenericArgs> get_resolved_generic_args(const TypeReflection* type);

    /// Replace the wrapped low-level layout after hot reload and clear semantic caches.
    void on_hot_reload(ref<const sgl::ProgramLayout> low_level_layout);

    /// Return a debug string for this semantic layout.
    std::string to_string() const override;

private:
    ref<const sgl::ProgramLayout> m_low_level_layout;
    uint64_t m_generation = 0;
    std::unordered_map<const TypeReflection*, ref<Type>> m_types_by_reflection;
    std::unordered_map<std::string, ref<Type>> m_types_by_name;
};

} // namespace sgl::refl
