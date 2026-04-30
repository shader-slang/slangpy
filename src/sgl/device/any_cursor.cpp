// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "any_cursor.h"

#include "sgl/core/error.h"

#include "sgl/device/cuda_interop.h"
#include "sgl/device/resource.h"

namespace sgl {
namespace {

    [[noreturn]] void throw_unsupported(std::string_view operation)
    {
        SGL_THROW("{} is not supported by BufferElementCursor-backed AnyCursor.", operation);
    }

    bool is_pointer_cursor(const BufferElementCursor& cursor)
    {
        slang::TypeLayoutReflection* type_layout = cursor.slang_type_layout();
        slang::TypeReflection* type = type_layout ? type_layout->getType() : nullptr;
        return type && type->getKind() == slang::TypeReflection::Kind::Pointer;
    }

    void require_pointer_cursor(const BufferElementCursor& cursor, std::string_view operation)
    {
        SGL_CHECK(
            is_pointer_cursor(cursor),
            "{} requires a pointer field when used with BufferElementCursor-backed AnyCursor.",
            operation
        );
    }

} // namespace

AnyCursor::AnyCursor(ShaderCursor cursor)
    : m_cursor(cursor)
{
}

AnyCursor::AnyCursor(BufferElementCursor cursor)
    : m_cursor(std::move(cursor))
{
}

bool AnyCursor::is_shader_cursor() const
{
    return std::holds_alternative<ShaderCursor>(m_cursor);
}

bool AnyCursor::is_buffer_element_cursor() const
{
    return std::holds_alternative<BufferElementCursor>(m_cursor);
}

ShaderCursor* AnyCursor::as_shader_cursor()
{
    return std::get_if<ShaderCursor>(&m_cursor);
}

const ShaderCursor* AnyCursor::as_shader_cursor() const
{
    return std::get_if<ShaderCursor>(&m_cursor);
}

BufferElementCursor* AnyCursor::as_buffer_element_cursor()
{
    return std::get_if<BufferElementCursor>(&m_cursor);
}

const BufferElementCursor* AnyCursor::as_buffer_element_cursor() const
{
    return std::get_if<BufferElementCursor>(&m_cursor);
}

bool AnyCursor::is_valid() const
{
    return std::visit(
        [](const auto& cursor)
        {
            return cursor.is_valid();
        },
        m_cursor
    );
}

std::string AnyCursor::to_string() const
{
    return std::visit(
        [](const auto& cursor)
        {
            return cursor.to_string();
        },
        m_cursor
    );
}

slang::TypeLayoutReflection* AnyCursor::slang_type_layout() const
{
    return std::visit(
        [](const auto& cursor)
        {
            return cursor.slang_type_layout();
        },
        m_cursor
    );
}

AnyCursor AnyCursor::operator[](std::string_view name) const
{
    return std::visit(
        [&](const auto& cursor) -> AnyCursor
        {
            return cursor[name];
        },
        m_cursor
    );
}

AnyCursor AnyCursor::operator[](uint32_t index) const
{
    return std::visit(
        [&](const auto& cursor) -> AnyCursor
        {
            return cursor[index];
        },
        m_cursor
    );
}

AnyCursor AnyCursor::find_field(std::string_view name) const
{
    return std::visit(
        [&](const auto& cursor) -> AnyCursor
        {
            return cursor.find_field(name);
        },
        m_cursor
    );
}

AnyCursor AnyCursor::find_element(uint32_t index) const
{
    return std::visit(
        [&](const auto& cursor) -> AnyCursor
        {
            return cursor.find_element(index);
        },
        m_cursor
    );
}

bool AnyCursor::has_field(std::string_view name) const
{
    return std::visit(
        [&](const auto& cursor)
        {
            return cursor.has_field(name);
        },
        m_cursor
    );
}

bool AnyCursor::has_element(uint32_t index) const
{
    return std::visit(
        [&](const auto& cursor)
        {
            return cursor.has_element(index);
        },
        m_cursor
    );
}

void AnyCursor::set_object(const ref<ShaderObject>& object)
{
    std::visit(
        [&](auto& cursor)
        {
            using Cursor = std::decay_t<decltype(cursor)>;
            if constexpr (std::is_same_v<Cursor, ShaderCursor>) {
                cursor.set_object(object);
            } else {
                throw_unsupported("set_object");
            }
        },
        m_cursor
    );
}

void AnyCursor::set_buffer(const ref<Buffer>& buffer)
{
    std::visit(
        [&](auto& cursor)
        {
            using Cursor = std::decay_t<decltype(cursor)>;
            if constexpr (std::is_same_v<Cursor, ShaderCursor>) {
                cursor.set_buffer(buffer);
            } else {
                require_pointer_cursor(cursor, "set_buffer");
                SGL_CHECK(buffer, "set_buffer requires a non-null buffer for pointer fields.");
                cursor.set_pointer(buffer->device_address());
            }
        },
        m_cursor
    );
}

void AnyCursor::set_buffer_view(const ref<BufferView>& buffer_view)
{
    std::visit(
        [&](auto& cursor)
        {
            using Cursor = std::decay_t<decltype(cursor)>;
            if constexpr (std::is_same_v<Cursor, ShaderCursor>) {
                cursor.set_buffer_view(buffer_view);
            } else {
                require_pointer_cursor(cursor, "set_buffer_view");
                SGL_CHECK(buffer_view && buffer_view->buffer(), "set_buffer_view requires a non-null buffer view.");
                cursor.set_pointer(buffer_view->buffer()->device_address() + buffer_view->range().offset);
            }
        },
        m_cursor
    );
}

void AnyCursor::set_texture(const ref<Texture>& texture)
{
    std::visit(
        [&](auto& cursor)
        {
            using Cursor = std::decay_t<decltype(cursor)>;
            if constexpr (std::is_same_v<Cursor, ShaderCursor>) {
                cursor.set_texture(texture);
            } else {
                throw_unsupported("set_texture");
            }
        },
        m_cursor
    );
}

void AnyCursor::set_texture_view(const ref<TextureView>& texture_view)
{
    std::visit(
        [&](auto& cursor)
        {
            using Cursor = std::decay_t<decltype(cursor)>;
            if constexpr (std::is_same_v<Cursor, ShaderCursor>) {
                cursor.set_texture_view(texture_view);
            } else {
                throw_unsupported("set_texture_view");
            }
        },
        m_cursor
    );
}

void AnyCursor::set_sampler(const ref<Sampler>& sampler)
{
    std::visit(
        [&](auto& cursor)
        {
            using Cursor = std::decay_t<decltype(cursor)>;
            if constexpr (std::is_same_v<Cursor, ShaderCursor>) {
                cursor.set_sampler(sampler);
            } else {
                throw_unsupported("set_sampler");
            }
        },
        m_cursor
    );
}

void AnyCursor::set_acceleration_structure(const ref<AccelerationStructure>& acceleration_structure)
{
    std::visit(
        [&](auto& cursor)
        {
            using Cursor = std::decay_t<decltype(cursor)>;
            if constexpr (std::is_same_v<Cursor, ShaderCursor>) {
                cursor.set_acceleration_structure(acceleration_structure);
            } else {
                throw_unsupported("set_acceleration_structure");
            }
        },
        m_cursor
    );
}

void AnyCursor::set_descriptor_handle(const DescriptorHandle& handle)
{
    std::visit(
        [&](auto& cursor)
        {
            using Cursor = std::decay_t<decltype(cursor)>;
            if constexpr (std::is_same_v<Cursor, ShaderCursor>) {
                cursor.set_descriptor_handle(handle);
            } else {
                cursor.set(handle);
            }
        },
        m_cursor
    );
}

void AnyCursor::set_data(const void* data, size_t size)
{
    std::visit(
        [&](auto& cursor)
        {
            cursor.set_data(data, size);
        },
        m_cursor
    );
}

void AnyCursor::set_cuda_tensor_view(const cuda::TensorView& tensor_view)
{
    std::visit(
        [&](auto& cursor)
        {
            using Cursor = std::decay_t<decltype(cursor)>;
            if constexpr (std::is_same_v<Cursor, ShaderCursor>) {
                cursor.set_cuda_tensor_view(tensor_view);
            } else {
                SGL_UNUSED(tensor_view);
                throw_unsupported("set_cuda_tensor_view");
            }
        },
        m_cursor
    );
}

void AnyCursor::set_pointer(uint64_t pointer_value)
{
    std::visit(
        [&](auto& cursor)
        {
            cursor.set_pointer(pointer_value);
        },
        m_cursor
    );
}

void AnyCursor::_set_array(
    const void* data,
    size_t size,
    TypeReflection::ScalarType cpu_scalar_type,
    size_t element_count
)
{
    std::visit(
        [&](auto& cursor)
        {
            cursor._set_array(data, size, cpu_scalar_type, element_count);
        },
        m_cursor
    );
}

void AnyCursor::_set_scalar(const void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type)
{
    std::visit(
        [&](auto& cursor)
        {
            cursor._set_scalar(data, size, cpu_scalar_type);
        },
        m_cursor
    );
}

void AnyCursor::_set_vector(const void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, int dimension)
{
    std::visit(
        [&](auto& cursor)
        {
            cursor._set_vector(data, size, cpu_scalar_type, dimension);
        },
        m_cursor
    );
}

void AnyCursor::_set_matrix(
    const void* data,
    size_t size,
    TypeReflection::ScalarType cpu_scalar_type,
    int rows,
    int cols
)
{
    std::visit(
        [&](auto& cursor)
        {
            cursor._set_matrix(data, size, cpu_scalar_type, rows, cols);
        },
        m_cursor
    );
}

} // namespace sgl
