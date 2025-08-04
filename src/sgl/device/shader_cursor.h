// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/shader_offset.h"
#include "sgl/device/reflection.h"
#include "sgl/device/cursor_utils.h"

#include "sgl/core/config.h"
#include "sgl/core/macros.h"

#include "sgl/device/cursor_access_wrappers.h"

#include <string_view>

namespace sgl {

/// Cursor used for parsing and setting shader object fields. This class does *NOT* use
/// the SGL reflection wrappers for accessing due to the performance implications of
/// allocating/freeing them repeatedly. This is far faster, however does introduce
/// a risk of mem access problems if the shader cursor is kept alive longer than
/// the shader object it was created from.
class SGL_API ShaderCursor : public CursorWriteWrappers<ShaderCursor, ShaderOffset> {
public:
    ShaderCursor() = default;

    ShaderCursor(ShaderObject* shader_object);
    ShaderCursor(ShaderObject* shader_object, bool need_dereference, slang::TypeLayoutReflection* parent_type_layout);

    ShaderOffset offset() const { return m_offset; }

    bool is_valid() const { return m_offset.is_valid(); }

    std::string to_string() const;

    bool is_reference() const;

    ShaderCursor dereference() const;

    slang::TypeLayoutReflection* slang_type_layout() const { return m_type_layout; }

    //
    // Navigation
    //

    ShaderCursor operator[](std::string_view name) const;
    ShaderCursor operator[](uint32_t index) const;

    ShaderCursor find_field(std::string_view name) const;
    ShaderCursor find_element(uint32_t index) const;

    ShaderCursor find_entry_point(uint32_t index) const;

    bool has_field(std::string_view name) const { return find_field(name).is_valid(); }
    bool has_element(uint32_t index) const { return find_element(index).is_valid(); }

    //
    // Resource binding
    //

    void set_object(const ref<ShaderObject>& object) const;

    void set_buffer(const ref<Buffer>& buffer) const;
    void set_buffer_view(const ref<BufferView>& buffer_view) const;
    void set_texture(const ref<Texture>& texture) const;
    void set_texture_view(const ref<TextureView>& texture_view) const;
    void set_sampler(const ref<Sampler>& sampler) const;
    void set_acceleration_structure(const ref<AccelerationStructure>& acceleration_structure) const;

    void set_descriptor_handle(const DescriptorHandle& handle) const;

    void set_data(const void* data, size_t size) const;

    void set_cuda_tensor_view(const cuda::TensorView& tensor_view) const;

    void set_pointer(uint64_t pointer_value) const;

    template<typename T>
    const ShaderCursor& operator=(const T& value) const
    {
        set(value);
        return *this;
    }

    template<typename T>
    void set(const T& value) const;

    void
    _set_array_unsafe(const void* data, size_t size, size_t element_count, TypeReflection::ScalarType cpu_scalar_type)
        const;

    /// CursorWriteWrappers, CursorReadWrappers
    void _set_data(ShaderOffset offset, const void* data, size_t size) const;
    ShaderOffset _get_offset() const { return m_offset; }
    static ShaderOffset _increment_offset(ShaderOffset offset, size_t diff)
    {
        offset.uniform_offset += narrow_cast<uint32_t>(diff);
        return offset;
    }
    DeviceType _get_device_type() const;

private:
    slang::TypeLayoutReflection* m_type_layout;
    ShaderObject* m_shader_object{nullptr};
    ShaderOffset m_offset;
};

} // namespace sgl
