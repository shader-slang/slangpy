// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/buffer_cursor.h"
#include "sgl/device/shader_cursor.h"

#include "sgl/core/config.h"

#include <type_traits>
#include <utility>
#include <variant>

namespace sgl {

/// Type-erased writable cursor over ShaderCursor and BufferElementCursor.
///
/// AnyCursor is intentionally passed by value in polymorphic write_to_cursor
/// implementations so concrete cursor temporaries, such as cursor["field"], can
/// be used directly.
class SGL_API AnyCursor {
public:
    AnyCursor(ShaderCursor cursor);
    AnyCursor(BufferElementCursor cursor);

    bool is_shader_cursor() const;
    bool is_buffer_element_cursor() const;

    ShaderCursor* as_shader_cursor();
    const ShaderCursor* as_shader_cursor() const;
    BufferElementCursor* as_buffer_element_cursor();
    const BufferElementCursor* as_buffer_element_cursor() const;

    bool is_valid() const;
    std::string to_string() const;
    slang::TypeLayoutReflection* slang_type_layout() const;

    AnyCursor operator[](std::string_view name) const;
    AnyCursor operator[](uint32_t index) const;

    AnyCursor find_field(std::string_view name) const;
    AnyCursor find_element(uint32_t index) const;

    bool has_field(std::string_view name) const;
    bool has_element(uint32_t index) const;

    void set_object(const ref<ShaderObject>& object);

    void set_buffer(const ref<Buffer>& buffer);
    void set_buffer_view(const ref<BufferView>& buffer_view);
    void set_texture(const ref<Texture>& texture);
    void set_texture_view(const ref<TextureView>& texture_view);
    void set_sampler(const ref<Sampler>& sampler);
    void set_acceleration_structure(const ref<AccelerationStructure>& acceleration_structure);

    void set_descriptor_handle(const DescriptorHandle& handle);

    void set_data(const void* data, size_t size);
    void set_cuda_tensor_view(const cuda::TensorView& tensor_view);
    void set_pointer(uint64_t pointer_value);

    template<typename T>
    AnyCursor& operator=(const T& value)
    {
        set(value);
        return *this;
    }

    template<typename T>
    void set(const T& value)
    {
        using Value = std::decay_t<T>;
        if constexpr (std::is_same_v<Value, ref<ShaderObject>>) {
            set_object(value);
        } else if constexpr (std::is_same_v<Value, ref<Buffer>>) {
            set_buffer(value);
        } else if constexpr (std::is_same_v<Value, ref<BufferView>>) {
            set_buffer_view(value);
        } else if constexpr (std::is_same_v<Value, ref<Texture>>) {
            set_texture(value);
        } else if constexpr (std::is_same_v<Value, ref<TextureView>>) {
            set_texture_view(value);
        } else if constexpr (std::is_same_v<Value, ref<Sampler>>) {
            set_sampler(value);
        } else if constexpr (std::is_same_v<Value, ref<AccelerationStructure>>) {
            set_acceleration_structure(value);
        } else if constexpr (std::is_same_v<Value, DescriptorHandle>) {
            set_descriptor_handle(value);
        } else if constexpr (std::is_same_v<Value, cuda::TensorView>) {
            set_cuda_tensor_view(value);
        } else if constexpr (HasWriteToCursor<T, AnyCursor>) {
            value.write_to_cursor(*this);
        } else {
            std::visit(
                [&](auto& cursor)
                {
                    cursor.set(value);
                },
                m_cursor
            );
        }
    }

    void _set_array(const void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, size_t element_count);
    void _set_scalar(const void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type);
    void _set_vector(const void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, int dimension);
    void
    _set_matrix(const void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, int rows, int cols);

private:
    using CursorVariant = std::variant<ShaderCursor, BufferElementCursor>;

    CursorVariant m_cursor;
};

/// Interface for values that can write themselves to any supported cursor type.
class SGL_API CursorWritable {
public:
    virtual ~CursorWritable() = default;
    virtual void write_to_cursor(AnyCursor cursor) const = 0;
};

} // namespace sgl
