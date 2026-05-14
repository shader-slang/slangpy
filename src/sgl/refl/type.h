// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/reflection.h"
#include "sgl/utils/slangpy.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sgl::refl {

class Layout;

/// Native semantic reflection code must remain Python-free.
/// Bindings and Python-specific adaptation belong in src/slangpy_ext.

class SGL_API TypeLayout : public Object {
    SGL_OBJECT(TypeLayout)
public:
    explicit TypeLayout(ref<const TypeLayoutReflection> reflection);

    const TypeLayoutReflection* reflection() const { return m_reflection.get(); }
    ref<const TypeLayoutReflection> reflection_ref() const { return m_reflection; }

    size_t size() const { return m_reflection->size(); }
    size_t alignment() const { return m_reflection->alignment(); }
    size_t stride() const { return m_reflection->stride(); }

    std::string to_string() const override;

private:
    ref<const TypeLayoutReflection> m_reflection;
};

class SGL_API Type : public Object {
    SGL_OBJECT(Type)
public:
    Type(ref<Layout> layout, ref<const TypeReflection> reflection, ref<Type> element_type, slangpy::Shape local_shape);

    Layout* layout() const { return m_layout.get(); }
    ref<Layout> layout_ref() const { return m_layout; }

    const TypeReflection* reflection() const { return m_reflection.get(); }
    ref<const TypeReflection> reflection_ref() const { return m_reflection; }

    std::string name() const;
    std::string full_name() const;

    ref<Type> element_type() const { return m_element_type; }
    const slangpy::Shape& shape() const { return m_shape; }
    size_t num_dims() const { return m_shape.valid() ? m_shape.size() : 0; }

    virtual bool is_generic() const;
    virtual std::string vector_type_name() const;
    virtual ref<Type> derivative();

    ref<TypeLayout> uniform_layout();
    ref<TypeLayout> buffer_layout();

    virtual void on_hot_reload(ref<const TypeReflection> reflection);

    std::string to_string() const override;

protected:
    void set_element_type(ref<Type> element_type);
    void set_local_shape(slangpy::Shape local_shape);
    void update_shape();
    ref<Type> find_type_by_name(std::string_view name) const;

    ref<Layout> m_layout;
    ref<const TypeReflection> m_reflection;
    ref<Type> m_element_type;
    slangpy::Shape m_local_shape;
    slangpy::Shape m_shape;

private:
    ref<TypeLayout> m_uniform_layout;
    ref<TypeLayout> m_buffer_layout;
    ref<Type> m_derivative;
    mutable std::optional<std::string> m_vector_type_name;
};

class SGL_API UnknownType final : public Type {
    SGL_OBJECT(UnknownType)
public:
    UnknownType(ref<Layout> layout, ref<const TypeReflection> reflection);

    std::string vector_type_name() const override { return "Unknown"; }
};

class SGL_API VoidType final : public Type {
    SGL_OBJECT(VoidType)
public:
    VoidType(ref<Layout> layout, ref<const TypeReflection> reflection);
};

class SGL_API PointerType final : public Type {
    SGL_OBJECT(PointerType)
public:
    PointerType(ref<Layout> layout, ref<const TypeReflection> reflection);

    ref<Type> target_type() const { return m_target_type; }
    TypeReflection::ScalarType slang_scalar_type() const { return TypeReflection::ScalarType::uint64; }
    bool is_generic() const override;
    std::string vector_type_name() const override;

private:
    ref<Type> m_target_type;
};

class SGL_API ScalarType final : public Type {
    SGL_OBJECT(ScalarType)
public:
    ScalarType(ref<Layout> layout, ref<const TypeReflection> reflection);

    TypeReflection::ScalarType slang_scalar_type() const { return m_reflection->scalar_type(); }
};

class SGL_API VectorType final : public Type {
    SGL_OBJECT(VectorType)
public:
    VectorType(ref<Layout> layout, ref<const TypeReflection> reflection);

    bool is_generic() const override { return num_elements() == 0; }
    int num_elements() const { return m_num_elements; }
    ref<ScalarType> scalar_type() const;
    TypeReflection::ScalarType slang_scalar_type() const { return m_reflection->scalar_type(); }
    std::string vector_type_name() const override;

private:
    int m_num_elements = 0;
};

class SGL_API MatrixType final : public Type {
    SGL_OBJECT(MatrixType)
public:
    MatrixType(ref<Layout> layout, ref<const TypeReflection> reflection);

    bool is_generic() const override { return rows() == 0 || cols() == 0; }
    int rows() const { return m_rows; }
    int cols() const { return m_cols; }
    ref<ScalarType> scalar_type() const;
    TypeReflection::ScalarType slang_scalar_type() const { return m_reflection->scalar_type(); }
    ref<Type> inner_element_type() const;
    std::string vector_type_name() const override;

private:
    int m_rows = 0;
    int m_cols = 0;
};

class SGL_API ArrayType final : public Type {
    SGL_OBJECT(ArrayType)
public:
    ArrayType(ref<Layout> layout, ref<const TypeReflection> reflection);

    bool is_generic() const override { return num_elements() == 0; }
    int num_elements() const { return m_num_elements; }
    slangpy::Shape array_shape() const;
    bool any_generic_dims() const;
    ref<Type> inner_element_type() const;
    size_t array_dims() const { return array_shape().size(); }
    std::string vector_type_name() const override;

private:
    int m_num_elements = 0;
};

class SGL_API StructType : public Type {
    SGL_OBJECT(StructType)
public:
    StructType(ref<Layout> layout, ref<const TypeReflection> reflection);

    bool is_generic() const override;
    std::string vector_type_name() const override;

protected:
    bool m_is_generic = false;
};

class SGL_API InterfaceType final : public Type {
    SGL_OBJECT(InterfaceType)
public:
    InterfaceType(ref<Layout> layout, ref<const TypeReflection> reflection);

    bool is_generic() const override { return m_is_generic; }
    std::string vector_type_name() const override { return "Unknown"; }

private:
    bool m_is_generic = false;
};

class SGL_API ResourceType : public Type {
    SGL_OBJECT(ResourceType)
public:
    ResourceType(
        ref<Layout> layout,
        ref<const TypeReflection> reflection,
        ref<Type> element_type = nullptr,
        slangpy::Shape local_shape = slangpy::Shape(std::vector<int>{})
    );

    TypeReflection::ResourceShape resource_shape() const { return m_reflection->resource_shape(); }
    TypeReflection::ResourceAccess resource_access() const { return m_reflection->resource_access(); }
    bool writable() const;
};

class SGL_API TextureType final : public ResourceType {
    SGL_OBJECT(TextureType)
public:
    TextureType(ref<Layout> layout, ref<const TypeReflection> reflection);

    int texture_dims() const { return m_texture_dims; }

private:
    int m_texture_dims = 0;
};

class SGL_API StructuredBufferType final : public ResourceType {
    SGL_OBJECT(StructuredBufferType)
public:
    StructuredBufferType(ref<Layout> layout, ref<const TypeReflection> reflection);

    std::string vector_type_name() const override;
};

class SGL_API ByteAddressBufferType final : public ResourceType {
    SGL_OBJECT(ByteAddressBufferType)
public:
    ByteAddressBufferType(ref<Layout> layout, ref<const TypeReflection> reflection);
};

class SGL_API DifferentialPairType final : public Type {
    SGL_OBJECT(DifferentialPairType)
public:
    DifferentialPairType(ref<Layout> layout, ref<const TypeReflection> reflection);

    ref<Type> primal() const { return m_primal; }
    ref<Type> derivative() override;

private:
    ref<Type> m_primal;
};

class SGL_API RaytracingAccelerationStructureType final : public Type {
    SGL_OBJECT(RaytracingAccelerationStructureType)
public:
    RaytracingAccelerationStructureType(ref<Layout> layout, ref<const TypeReflection> reflection);
};

class SGL_API SamplerStateType final : public Type {
    SGL_OBJECT(SamplerStateType)
public:
    SamplerStateType(ref<Layout> layout, ref<const TypeReflection> reflection);
};

class SGL_API TensorType final : public Type {
    SGL_OBJECT(TensorType)
public:
    enum class Kind { tensor, itensor, diff_tensor, idiff_tensor, primal_tensor, atomic };
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

    Kind tensor_kind() const { return m_kind; }
    Access access() const { return m_access; }
    bool readable() const { return m_access == Access::read || m_access == Access::read_write; }
    bool writable() const { return m_access == Access::write || m_access == Access::read_write; }
    bool diff_tensor() const { return m_kind == Kind::diff_tensor || m_kind == Kind::idiff_tensor; }
    int dims() const { return m_dims; }
    ref<Type> dtype() const { return m_element_type; }
    bool has_grad_in() const { return writable() && diff_tensor(); }
    bool has_grad_out() const { return readable() && diff_tensor(); }
    bool is_generic() const override;

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

class SGL_API TensorViewType final : public Type {
    SGL_OBJECT(TensorViewType)
public:
    TensorViewType(ref<Layout> layout, ref<const TypeReflection> reflection);

    ref<Type> dtype() const { return m_element_type; }
    bool is_generic() const override;

    static std::string build_tensorview_name(const Type& element_type);
};

class SGL_API DiffTensorViewType final : public Type {
    SGL_OBJECT(DiffTensorViewType)
public:
    DiffTensorViewType(ref<Layout> layout, ref<const TypeReflection> reflection);

    ref<Type> dtype() const { return m_element_type; }
    ref<Type> wrapper_type() const { return m_wrapper_type; }
    bool is_generic() const override;

    static std::string build_difftensorview_name(const Type& element_type);

private:
    ref<Type> m_wrapper_type;
};

class SGL_API UnhandledType final : public Type {
    SGL_OBJECT(UnhandledType)
public:
    UnhandledType(ref<Layout> layout, ref<const TypeReflection> reflection);
};

class SGL_API GenericArg {
public:
    enum class Kind { integer, type };

    static GenericArg integer(int value);
    static GenericArg type(ref<Type> value);

    Kind kind() const { return m_kind; }
    bool is_integer() const { return m_kind == Kind::integer; }
    bool is_type() const { return m_kind == Kind::type; }
    int integer() const { return m_integer; }
    ref<Type> type() const { return m_type; }

private:
    Kind m_kind = Kind::integer;
    int m_integer = 0;
    ref<Type> m_type;
};

using GenericArgs = std::vector<GenericArg>;

SGL_API ref<Type> create_builtin_type(Layout& layout, ref<const TypeReflection> reflection);
SGL_API std::optional<GenericArgs> parse_generic_args(Layout& layout, const TypeReflection* reflection);
SGL_API std::string name_for_scalar_type(TypeReflection::ScalarType scalar_type);

SGL_ENUM_REGISTER(TensorType::Kind);
SGL_ENUM_REGISTER(TensorType::Access);

} // namespace sgl::refl
