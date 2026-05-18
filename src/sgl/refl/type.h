// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/reflection.h"
#include "sgl/device/resource.h"
#include "sgl/utils/slangpy.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sgl::refl {

class Field;
class Layout;

/// Native semantic reflection code must remain Python-free.
/// Bindings and Python-specific adaptation belong in src/slangpy_ext.

/// Semantic size/alignment/stride information for a reflected type in a specific layout context.
class SGL_API TypeLayout : public Object {
    SGL_OBJECT(TypeLayout)
public:
    /// Create a semantic layout wrapper from low-level Slang type layout reflection.
    explicit TypeLayout(ref<const TypeLayoutReflection> reflection);

    /// Return the low-level SGL type layout reflection.
    const TypeLayoutReflection* reflection() const { return m_reflection.get(); }

    /// Return the byte size of the type under uniform layout rules.
    size_t size() const { return m_reflection->size(); }
    /// Return the byte alignment of the type under uniform layout rules.
    size_t alignment() const { return m_reflection->alignment(); }
    /// Return the byte stride of the type under uniform layout rules.
    size_t stride() const { return m_reflection->stride(); }

    /// Return a debug string for this semantic layout wrapper.
    std::string to_string() const override;

private:
    ref<const TypeLayoutReflection> m_reflection;
};

/// Base semantic reflection type used by native SlangPy runtime code.
class SGL_API Type : public Object {
    SGL_OBJECT(Type)
public:
    /// Create a semantic type from low-level Slang type reflection.
    Type(ref<Layout> layout, ref<const TypeReflection> reflection, ref<Type> element_type, slangpy::Shape local_shape);

    /// Return the semantic layout that owns this type.
    Layout* layout() const { return m_layout.get(); }

    /// Return the low-level SGL type reflection.
    const TypeReflection* reflection() const { return m_reflection.get(); }

    /// Return the short reflected type name.
    std::string name() const;
    /// Return the fully specialized reflected type name.
    std::string full_name() const;

    /// Return the element type for container-like types, or null for opaque types.
    ref<Type> element_type() const { return m_element_type; }
    /// Return the local shape combined with any element type shape.
    const slangpy::Shape& shape() const { return m_shape; }
    /// Return the number of dimensions represented by shape().
    size_t num_dims() const { return m_shape.valid() ? m_shape.size() : 0; }

    /// Return true if this type contains unresolved generic type or value parameters.
    virtual bool is_generic() const;
    /// Return the type spelling used for vectorized generic specialization.
    virtual std::string vector_type_name() const;
    /// Return the derivative type, if one exists.
    virtual ref<Type> derivative();
    /// Return true when derivative() resolves to a valid type.
    bool has_derivative() { return derivative() != nullptr; }
    /// Return the reflected fields for aggregate-like types.
    const std::unordered_map<std::string, ref<Field>>& fields();

    /// Return the type layout when used as uniform data.
    ref<TypeLayout> uniform_layout();
    /// Return the low-level uniform type layout reflection.
    ref<TypeLayoutReflection> uniform_type_layout()
    {
        return ref<TypeLayoutReflection>(const_cast<TypeLayoutReflection*>(uniform_layout()->reflection()));
    }
    /// Return the type layout when used as an element of a structured buffer.
    ref<TypeLayout> buffer_layout();
    /// Return the low-level structured-buffer element type layout reflection.
    ref<TypeLayoutReflection> buffer_type_layout()
    {
        return ref<TypeLayoutReflection>(const_cast<TypeLayoutReflection*>(buffer_layout()->reflection()));
    }

    /// Refresh low-level reflection after hot reload and clear derived caches.
    virtual void on_hot_reload(ref<const TypeReflection> reflection);

    /// Return a debug string for this semantic type.
    std::string to_string() const override;

protected:
    void set_element_type(ref<Type> element_type);
    void set_local_shape(slangpy::Shape local_shape);
    void update_shape();
    ref<Type> find_type_by_name(std::string_view name) const;
    /// Build fields for this semantic type. Override in types that expose field-like members.
    virtual std::unordered_map<std::string, ref<Field>> build_fields();

    ref<Layout> m_layout;
    ref<const TypeReflection> m_reflection;
    ref<Type> m_element_type;
    slangpy::Shape m_local_shape;
    slangpy::Shape m_shape;

private:
    ref<TypeLayout> m_uniform_layout;
    ref<TypeLayout> m_buffer_layout;
    ref<Type> m_derivative;
    std::optional<std::unordered_map<std::string, ref<Field>>> m_fields;
    mutable std::optional<std::string> m_vector_type_name;
};

/// Semantic type for Slang's Unknown placeholder type.
class SGL_API UnknownType final : public Type {
    SGL_OBJECT(UnknownType)
public:
    UnknownType(ref<Layout> layout, ref<const TypeReflection> reflection);

    std::string vector_type_name() const override { return "Unknown"; }
};

/// Return true when type is the semantic Unknown placeholder.
SGL_API bool is_unknown(const Type* type);
/// Return true when type is non-null and not the semantic Unknown placeholder.
SGL_API bool is_known(const Type* type);
/// Return true when type is null or not the semantic Unknown placeholder.
SGL_API bool is_known_or_none(const Type* type);

/// Semantic type for void.
class SGL_API VoidType final : public Type {
    SGL_OBJECT(VoidType)
public:
    VoidType(ref<Layout> layout, ref<const TypeReflection> reflection);
};

/// Semantic type for pointer-like reflected types.
class SGL_API PointerType final : public Type {
    SGL_OBJECT(PointerType)
public:
    PointerType(ref<Layout> layout, ref<const TypeReflection> reflection);

    /// Return the pointer target type when it could be resolved from generic arguments.
    ref<Type> target_type() const { return m_target_type; }
    /// Return the scalar representation used for pointer values in native marshalling.
    TypeReflection::ScalarType slang_scalar_type() const { return TypeReflection::ScalarType::uint64; }
    bool is_generic() const override;
    std::string vector_type_name() const override;

private:
    ref<Type> m_target_type;
};

/// Semantic type for scalar values.
class SGL_API ScalarType final : public Type {
    SGL_OBJECT(ScalarType)
public:
    ScalarType(ref<Layout> layout, ref<const TypeReflection> reflection);

    /// Return the low-level Slang scalar type id.
    TypeReflection::ScalarType slang_scalar_type() const { return m_reflection->scalar_type(); }
};

/// Semantic type for vector values.
class SGL_API VectorType final : public Type {
    SGL_OBJECT(VectorType)
public:
    VectorType(ref<Layout> layout, ref<const TypeReflection> reflection);

    bool is_generic() const override { return num_elements() == 0; }
    /// Return the number of vector lanes.
    int num_elements() const { return m_num_elements; }
    /// Return the scalar element type.
    ref<ScalarType> scalar_type() const;
    /// Return the low-level Slang scalar type id for each lane.
    TypeReflection::ScalarType slang_scalar_type() const { return m_reflection->scalar_type(); }
    std::string vector_type_name() const override;

protected:
    std::unordered_map<std::string, ref<Field>> build_fields() override;

private:
    int m_num_elements = 0;
};

/// Semantic type for matrix values.
class SGL_API MatrixType final : public Type {
    SGL_OBJECT(MatrixType)
public:
    MatrixType(ref<Layout> layout, ref<const TypeReflection> reflection);

    bool is_generic() const override { return rows() == 0 || cols() == 0; }
    /// Return the reflected row count.
    int rows() const { return m_rows; }
    /// Return the reflected column count.
    int cols() const { return m_cols; }
    /// Return the scalar element type.
    ref<ScalarType> scalar_type() const;
    /// Return the low-level Slang scalar type id.
    TypeReflection::ScalarType slang_scalar_type() const { return m_reflection->scalar_type(); }
    /// Return the innermost scalar element type.
    ref<Type> inner_element_type() const;
    std::string vector_type_name() const override;

private:
    int m_rows = 0;
    int m_cols = 0;
};

/// Semantic type for sized or unsized arrays.
class SGL_API ArrayType final : public Type {
    SGL_OBJECT(ArrayType)
public:
    ArrayType(ref<Layout> layout, ref<const TypeReflection> reflection);

    bool is_generic() const override { return num_elements() == 0; }
    /// Return the element count for this array dimension, or 0 for unresolved generic dimensions.
    int num_elements() const { return m_num_elements; }
    /// Return all nested array dimensions from outermost to innermost.
    slangpy::Shape array_shape() const;
    /// Return true when this array or a nested array has unresolved dimensions.
    bool any_generic_dims() const;
    /// Return the non-array element type after unwrapping nested arrays.
    ref<Type> inner_element_type() const;
    /// Return the number of nested array dimensions.
    size_t array_dims() const { return array_shape().size(); }
    std::string vector_type_name() const override;

private:
    int m_num_elements = 0;
};

/// Semantic type for concrete and generic Slang structs.
class SGL_API StructType : public Type {
    SGL_OBJECT(StructType)
public:
    StructType(ref<Layout> layout, ref<const TypeReflection> reflection);

    bool is_generic() const override;
    std::string vector_type_name() const override;

protected:
    std::unordered_map<std::string, ref<Field>> build_fields() override;

    bool m_is_generic = false;
};

/// Semantic type for Slang interfaces.
class SGL_API InterfaceType final : public Type {
    SGL_OBJECT(InterfaceType)
public:
    InterfaceType(ref<Layout> layout, ref<const TypeReflection> reflection);

    bool is_generic() const override { return m_is_generic; }
    std::string vector_type_name() const override { return "Unknown"; }

private:
    bool m_is_generic = false;
};

/// Base semantic type for resource-like Slang types.
class SGL_API ResourceType : public Type {
    SGL_OBJECT(ResourceType)
public:
    ResourceType(
        ref<Layout> layout,
        ref<const TypeReflection> reflection,
        ref<Type> element_type = nullptr,
        slangpy::Shape local_shape = slangpy::Shape(std::vector<int>{})
    );

    /// Return the reflected resource shape.
    TypeReflection::ResourceShape resource_shape() const { return m_reflection->resource_shape(); }
    /// Return the reflected resource access mode.
    TypeReflection::ResourceAccess resource_access() const { return m_reflection->resource_access(); }
    /// Return true if the resource can be written by shader code.
    bool writable() const;
};

/// Semantic type for texture resources.
class SGL_API TextureType final : public ResourceType {
    SGL_OBJECT(TextureType)
public:
    TextureType(ref<Layout> layout, ref<const TypeReflection> reflection);

    /// Return the dimensionality of the texture resource shape.
    int texture_dims() const { return m_texture_dims; }
    /// Return the texture usage needed to bind this reflected texture type.
    TextureUsage usage() const;

private:
    int m_texture_dims = 0;
};

/// Semantic type for structured buffer resources.
class SGL_API StructuredBufferType final : public ResourceType {
    SGL_OBJECT(StructuredBufferType)
public:
    StructuredBufferType(ref<Layout> layout, ref<const TypeReflection> reflection);

    std::string vector_type_name() const override;
};

/// Semantic type for byte-address buffer resources.
class SGL_API ByteAddressBufferType final : public ResourceType {
    SGL_OBJECT(ByteAddressBufferType)
public:
    ByteAddressBufferType(ref<Layout> layout, ref<const TypeReflection> reflection);
};

/// Semantic type for Slang differential pairs.
class SGL_API DifferentialPairType final : public Type {
    SGL_OBJECT(DifferentialPairType)
public:
    DifferentialPairType(ref<Layout> layout, ref<const TypeReflection> reflection);

    /// Return the primal type stored in the differential pair.
    ref<Type> primal() const { return m_primal; }
    ref<Type> derivative() override;

private:
    ref<Type> m_primal;
};

/// Semantic type for raytracing acceleration structures.
class SGL_API RaytracingAccelerationStructureType final : public Type {
    SGL_OBJECT(RaytracingAccelerationStructureType)
public:
    RaytracingAccelerationStructureType(ref<Layout> layout, ref<const TypeReflection> reflection);
};

/// Semantic type for sampler states.
class SGL_API SamplerStateType final : public Type {
    SGL_OBJECT(SamplerStateType)
public:
    SamplerStateType(ref<Layout> layout, ref<const TypeReflection> reflection);
};

/// Semantic type for Tensor/ITensor/DiffTensor families.
class SGL_API TensorType final : public Type {
    SGL_OBJECT(TensorType)
public:
    /// Tensor storage/interface family.
    enum class Kind { tensor, itensor, diff_tensor, idiff_tensor, primal_tensor, atomic };
    /// Tensor read/write capability.
    enum class Access { read, write, read_write };

    SGL_ENUM_INFO(
        Kind,
        {
            {Kind::tensor, "tensor"},
            {Kind::itensor, "itensor"},
            {Kind::diff_tensor, "diff_tensor"},
            {Kind::idiff_tensor, "idiff_tensor"},
            {Kind::primal_tensor, "primal_tensor"},
            {Kind::atomic, "atomic"},
        }
    );

    SGL_ENUM_INFO(
        Access,
        {
            {Access::read, "read"},
            {Access::write, "write"},
            {Access::read_write, "read_write"},
        }
    );

    TensorType(ref<Layout> layout, ref<const TypeReflection> reflection);

    /// Return the tensor family.
    Kind tensor_kind() const { return m_kind; }
    /// Return the tensor read/write capability.
    Access access() const { return m_access; }
    /// Return true if shader code can read this tensor.
    bool readable() const { return m_access == Access::read || m_access == Access::read_write; }
    /// Return true if shader code can write this tensor.
    bool writable() const { return m_access == Access::write || m_access == Access::read_write; }
    /// Return true if this is a differentiable tensor family.
    bool diff_tensor() const { return m_kind == Kind::diff_tensor || m_kind == Kind::idiff_tensor; }
    /// Return the tensor rank.
    int dims() const { return m_dims; }
    /// Return the tensor element type.
    ref<Type> dtype() const { return m_element_type; }
    /// Return true if backward dispatch needs a gradient input buffer.
    bool has_grad_in() const { return writable() && diff_tensor(); }
    /// Return true if backward dispatch needs a gradient output buffer.
    bool has_grad_out() const { return readable() && diff_tensor(); }
    bool is_generic() const override;

    /// Build the canonical Slang type name for a Tensor family specialization.
    static std::string build_tensor_name(
        const Type& element_type,
        int dims,
        Access access = Access::read_write,
        Kind tensor_kind = Kind::tensor
    );

private:
    Kind m_kind = Kind::tensor;
    Access m_access = Access::read;
    int m_dims = 0;
};

/// Semantic type for CUDA tensor view interop values.
class SGL_API TensorViewType final : public Type {
    SGL_OBJECT(TensorViewType)
public:
    TensorViewType(ref<Layout> layout, ref<const TypeReflection> reflection);

    /// Return the view element type.
    ref<Type> dtype() const { return m_element_type; }
    bool is_generic() const override;

    /// Build the canonical Slang type name for a TensorView specialization.
    static std::string build_tensorview_name(const Type& element_type);
};

/// Semantic type for differentiable CUDA tensor view interop values.
class SGL_API DiffTensorViewType final : public Type {
    SGL_OBJECT(DiffTensorViewType)
public:
    DiffTensorViewType(ref<Layout> layout, ref<const TypeReflection> reflection);

    /// Return the primal view element type.
    ref<Type> dtype() const { return m_element_type; }
    /// Return the optional wrapper type argument when present.
    ref<Type> wrapper_type() const { return m_wrapper_type; }
    bool is_generic() const override;

    /// Build the canonical Slang type name for a DiffTensorView specialization.
    static std::string build_difftensorview_name(const Type& element_type);

private:
    ref<Type> m_wrapper_type;
};

/// Semantic fallback for reflected types that do not yet have a native specialization.
class SGL_API UnhandledType final : public Type {
    SGL_OBJECT(UnhandledType)
public:
    UnhandledType(ref<Layout> layout, ref<const TypeReflection> reflection);
};

/// Resolved generic argument value used by native semantic reflection.
class SGL_API GenericArg {
public:
    /// Generic argument kind.
    enum class Kind { integer, type };

    /// Create an integer generic argument.
    static GenericArg integer(int value);
    /// Create a type generic argument.
    static GenericArg type(ref<Type> value);

    /// Return the argument kind.
    Kind kind() const { return m_kind; }
    /// Return true if this argument stores an integer value.
    bool is_integer() const { return m_kind == Kind::integer; }
    /// Return true if this argument stores a semantic type.
    bool is_type() const { return m_kind == Kind::type; }
    /// Return the integer value. Only valid when is_integer() is true.
    int integer() const { return m_integer; }
    /// Return the semantic type value. Only valid when is_type() is true.
    ref<Type> type() const { return m_type; }

private:
    Kind m_kind = Kind::integer;
    int m_integer = 0;
    ref<Type> m_type;
};

/// List of resolved generic arguments.
using GenericArgs = std::vector<GenericArg>;

/// Create the built-in native semantic type for a low-level reflection type.
SGL_API ref<Type> create_builtin_type(Layout& layout, ref<const TypeReflection> reflection);
/// Parse and resolve generic arguments from a reflected specialized type name.
SGL_API std::optional<GenericArgs> parse_generic_args(Layout& layout, const TypeReflection* reflection);
/// Return the canonical SlangPy scalar type spelling for a low-level scalar id.
SGL_API std::string name_for_scalar_type(TypeReflection::ScalarType scalar_type);

SGL_ENUM_REGISTER(TensorType::Kind);
SGL_ENUM_REGISTER(TensorType::Access);

} // namespace sgl::refl
