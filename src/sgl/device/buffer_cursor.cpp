// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "buffer_cursor.h"

#include "sgl/device/shader_object.h"
#include "sgl/device/resource.h"
#include "sgl/device/cuda_interop.h"
#include "sgl/device/cursor_utils.h"

#include "sgl/core/error.h"

#include "sgl/math/vector_types.h"
#include "sgl/math/matrix_types.h"

namespace sgl {

BufferElementCursor::BufferElementCursor(ref<TypeLayoutReflection> layout, ref<BufferCursor> owner)
    : m_type_layout(std::move(layout))
    , m_buffer(std::move(owner))
    , m_offset(0)
{
}

std::string BufferElementCursor::to_string() const
{
    return "BufferElementCursor()";
}

BufferElementCursor BufferElementCursor::operator[](std::string_view name) const
{
    SGL_CHECK(is_valid(), "Invalid cursor");
    BufferElementCursor result = find_field(name);
    SGL_CHECK(result.is_valid(), "Field \"{}\" not found.", name);
    return result;
}

BufferElementCursor BufferElementCursor::operator[](uint32_t index) const
{
    SGL_CHECK(is_valid(), "Invalid cursor");
    BufferElementCursor result = find_element(index);
    SGL_CHECK(result.is_valid(), "Element {} not found.", index);
    return result;
}

BufferElementCursor BufferElementCursor::find_field(std::string_view name) const
{
    if (!is_valid())
        return *this;

    switch (m_type_layout->kind()) {
    case TypeReflection::Kind::struct_: {

        int32_t field_index = m_type_layout->find_field_index_by_name(name.data(), name.data() + name.size());
        if (field_index < 0)
            break;

        ref<const VariableLayoutReflection> field_layout = m_type_layout->get_field_by_index(field_index);
        BufferElementCursor field_cursor;

        field_cursor.m_buffer = m_buffer;
        field_cursor.m_type_layout = field_layout->type_layout();
        field_cursor.m_offset = m_offset + field_layout->offset();

        return field_cursor;
    }

    default:
        break;
    }

    return {};
}

BufferElementCursor BufferElementCursor::find_element(uint32_t index) const
{
    if (!is_valid())
        return *this;

    switch (m_type_layout->kind()) {
    case TypeReflection::Kind::array: {
        BufferElementCursor element_cursor;
        element_cursor.m_buffer = m_buffer;
        element_cursor.m_type_layout = m_type_layout->element_type_layout();
        element_cursor.m_offset = m_offset + index * m_type_layout->element_stride();
        return element_cursor;
    } break;

    case TypeReflection::Kind::vector:
    case TypeReflection::Kind::matrix: {
        BufferElementCursor field_cursor;
        field_cursor.m_buffer = m_buffer;
        field_cursor.m_type_layout = m_type_layout->element_type_layout();
        field_cursor.m_offset = m_offset + m_type_layout->element_stride() * index;
        return field_cursor;
    } break;

    default:
        break;
    }

    return {};
}

void BufferElementCursor::set_pointer(uint64_t pointer_value)
{
    size_t pointer_size = m_type_layout->size();
    SGL_CHECK(pointer_size == 8, "Expected a pointer type with size 8, got {}", pointer_size);
    set_data(&pointer_value, 8);
}

void BufferElementCursor::set_data(const void* data, size_t size)
{
    if (m_type_layout->parameter_category() != TypeReflection::ParameterCategory::uniform)
        SGL_THROW("\"{}\" cannot bind data", m_type_layout->name());
    write_data(m_offset, data, size);
}

void BufferElementCursor::_set_array(
    const void* data,
    size_t size,
    TypeReflection::ScalarType scalar_type,
    size_t element_count
)
{
    ref<const TypeReflection> element_type = m_type_layout->unwrap_array()->type();
    size_t element_size = cursor_utils::get_scalar_type_size(element_type->scalar_type());

    cursor_utils::check_array(m_type_layout->slang_target(), size, scalar_type, element_count);

    size_t stride = m_type_layout->element_stride();
    if (element_size == stride) {
        write_data(m_offset, data, size);
    } else {
        size_t offset = m_offset;
        for (size_t i = 0; i < element_count; ++i) {
            write_data(offset, reinterpret_cast<const uint8_t*>(data) + i * element_size, element_size);
            offset += stride;
        }
    }
}

void BufferElementCursor::_get_array(
    void* data,
    size_t size,
    TypeReflection::ScalarType scalar_type,
    size_t element_count
) const
{
    ref<const TypeReflection> element_type = m_type_layout->unwrap_array()->type();
    size_t element_size = cursor_utils::get_scalar_type_size(element_type->scalar_type());

    cursor_utils::check_array(m_type_layout->slang_target(), size, scalar_type, element_count);

    size_t stride = m_type_layout->element_stride();
    if (element_size == stride) {
        read_data(m_offset, data, size);
    } else {
        size_t offset = m_offset;
        for (size_t i = 0; i < element_count; ++i) {
            read_data(offset, reinterpret_cast<uint8_t*>(data) + i * element_size, element_size);
            offset += stride;
        }
    }
}
void BufferElementCursor::_set_scalar(const void* data, size_t size, TypeReflection::ScalarType scalar_type)
{
    cursor_utils::check_scalar(m_type_layout->slang_target(), size, scalar_type);
    write_data(m_offset, data, size);
}

void BufferElementCursor::_get_scalar(void* data, size_t size, TypeReflection::ScalarType scalar_type) const
{
    cursor_utils::check_scalar(m_type_layout->slang_target(), size, scalar_type);
    read_data(m_offset, data, size);
}

void BufferElementCursor::_set_vector(
    const void* data,
    size_t size,
    TypeReflection::ScalarType scalar_type,
    int dimension
)
{
    cursor_utils::check_vector(m_type_layout->slang_target(), size, scalar_type, dimension);
    write_data(m_offset, data, size);
}

void BufferElementCursor::_get_vector(void* data, size_t size, TypeReflection::ScalarType scalar_type, int dimension)
    const
{
    cursor_utils::check_vector(m_type_layout->slang_target(), size, scalar_type, dimension);
    read_data(m_offset, data, size);
}

void BufferElementCursor::_set_matrix(
    const void* data,
    size_t size,
    TypeReflection::ScalarType scalar_type,
    int rows,
    int cols
)
{
    cursor_utils::check_matrix(m_type_layout->slang_target(), size, scalar_type, rows, cols);
    size_t stride = slang_type_layout()->getStride();
    if (stride != size) {
        size_t row_stride = stride / rows;
        size_t row_size = size / rows;
        for (int i = 0; i < rows; ++i) {
            write_data(m_offset + i * row_stride, reinterpret_cast<const uint8_t*>(data) + i * row_size, row_size);
        }
    } else {
        write_data(m_offset, data, size);
    }
}

void BufferElementCursor::_get_matrix(
    void* data,
    size_t size,
    TypeReflection::ScalarType scalar_type,
    int rows,
    int cols
) const
{
    cursor_utils::check_matrix(m_type_layout->slang_target(), size, scalar_type, rows, cols);
    size_t stride = slang_type_layout()->getStride();
    if (stride != size) {
        size_t row_stride = stride / rows;
        size_t row_size = size / rows;
        for (int i = 0; i < rows; ++i) {
            read_data(m_offset + i * row_stride, reinterpret_cast<uint8_t*>(data) + i * row_size, row_size);
        }
    } else {
        read_data(m_offset, data, size);
    }
}


//
// Setter specializations
//

#define GETSET_SCALAR(type, scalar_type)                                                                               \
    template<>                                                                                                         \
    SGL_API void BufferElementCursor::set(const type& value)                                                           \
    {                                                                                                                  \
        _set_scalar(&value, sizeof(value), TypeReflection::ScalarType::scalar_type);                                   \
    }                                                                                                                  \
    template<>                                                                                                         \
    SGL_API void BufferElementCursor::get(type& value) const                                                           \
    {                                                                                                                  \
        _get_scalar(&value, sizeof(value), TypeReflection::ScalarType::scalar_type);                                   \
    }

#define GETSET_VECTOR(type, scalar_type)                                                                               \
    template<>                                                                                                         \
    SGL_API void BufferElementCursor::set(const type& value)                                                           \
    {                                                                                                                  \
        _set_vector(&value, sizeof(value), TypeReflection::ScalarType::scalar_type, type::dimension);                  \
    }                                                                                                                  \
    template<>                                                                                                         \
    SGL_API void BufferElementCursor::get(type& value) const                                                           \
    {                                                                                                                  \
        _get_vector(&value, sizeof(value), TypeReflection::ScalarType::scalar_type, type::dimension);                  \
    }

#define GETSET_MATRIX(type, scalar_type)                                                                               \
    template<>                                                                                                         \
    SGL_API void BufferElementCursor::set(const type& value)                                                           \
    {                                                                                                                  \
        _set_matrix(&value, sizeof(value), TypeReflection::ScalarType::scalar_type, type::rows, type::cols);           \
    }                                                                                                                  \
    template<>                                                                                                         \
    SGL_API void BufferElementCursor::get(type& value) const                                                           \
    {                                                                                                                  \
        _get_matrix(&value, sizeof(value), TypeReflection::ScalarType::scalar_type, type::rows, type::cols);           \
    }

GETSET_SCALAR(int8_t, int8);
GETSET_SCALAR(uint8_t, uint8);
GETSET_SCALAR(int16_t, int16);
GETSET_SCALAR(uint16_t, uint16);

GETSET_SCALAR(int, int32);
GETSET_VECTOR(int1, int32);
GETSET_VECTOR(int2, int32);
GETSET_VECTOR(int3, int32);
GETSET_VECTOR(int4, int32);

GETSET_SCALAR(uint, uint32);
GETSET_VECTOR(uint1, uint32);
GETSET_VECTOR(uint2, uint32);
GETSET_VECTOR(uint3, uint32);
GETSET_VECTOR(uint4, uint32);

// MACOS treats these as separate types to int/uint, so they need to be
// provided as extra overloads for linking to succeed.
#if SGL_MACOS
GETSET_SCALAR(long, int32);
GETSET_SCALAR(unsigned long, uint32);
#endif

GETSET_SCALAR(int64_t, int64);
GETSET_SCALAR(uint64_t, uint64);

GETSET_SCALAR(float16_t, float16);
GETSET_VECTOR(float16_t1, float16);
GETSET_VECTOR(float16_t2, float16);
GETSET_VECTOR(float16_t3, float16);
GETSET_VECTOR(float16_t4, float16);

GETSET_SCALAR(float, float32);
GETSET_VECTOR(float1, float32);
GETSET_VECTOR(float2, float32);
GETSET_VECTOR(float3, float32);
GETSET_VECTOR(float4, float32);

GETSET_MATRIX(float2x2, float32);
GETSET_MATRIX(float2x3, float32);
GETSET_MATRIX(float2x4, float32);

GETSET_MATRIX(float3x2, float32);
GETSET_MATRIX(float3x3, float32);
GETSET_MATRIX(float3x4, float32);

GETSET_MATRIX(float4x2, float32);
GETSET_MATRIX(float4x3, float32);
GETSET_MATRIX(float4x4, float32);

GETSET_SCALAR(double, float64);

#undef GETSET_SCALAR
#undef GETSET_VECTOR
#undef GETSET_MATRIX

// Template specialization to allow setting booleans on a parameter block.
// On the host side a bool is 1B and the device 4B. We cast bools to 32-bit integers here.
// Note that this applies to our boolN vectors as well, which are currently 1B per element.

template<>
SGL_API void BufferElementCursor::set(const bool& value)
{
    uint v = value ? 1 : 0;
    _set_scalar(&v, sizeof(v), TypeReflection::ScalarType::bool_);
}
template<>
SGL_API void BufferElementCursor::get(bool& value) const
{
    uint v;
    _get_scalar(&v, sizeof(v), TypeReflection::ScalarType::bool_);
    value = v != 0;
}

template<>
SGL_API void BufferElementCursor::set(const bool1& value)
{
#if SGL_MACOS
    bool1 v = value;
#else
    uint1 v(value.x ? 1 : 0);
#endif
    _set_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, 1);
}
template<>
SGL_API void BufferElementCursor::get(bool1& value) const
{
#if SGL_MACOS
    bool1 v;
#else
    uint1 v;
#endif
    _get_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, 1);
    value = bool1(v.x != 0);
}

template<>
SGL_API void BufferElementCursor::set(const bool2& value)
{
#if SGL_MACOS
    bool2 v = value;
#else
    uint2 v = {value.x ? 1 : 0, value.y ? 1 : 0};
#endif
    _set_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, 2);
}

template<>
SGL_API void BufferElementCursor::get(bool2& value) const
{
#if SGL_MACOS
    bool2 v;
#else
    uint2 v;
#endif
    _get_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, 2);
    value = {v.x != 0, v.y != 0};
}

template<>
SGL_API void BufferElementCursor::set(const bool3& value)
{
#if SGL_MACOS
    bool3 v = value;
#else
    uint3 v = {value.x ? 1 : 0, value.y ? 1 : 0, value.z ? 1 : 0};
#endif
    _set_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, 3);
}

template<>
SGL_API void BufferElementCursor::get(bool3& value) const
{
#if SGL_MACOS
    bool3 v;
#else
    uint3 v;
#endif
    _get_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, 3);
    value = {v.x != 0, v.y != 0, v.z != 0};
}

template<>
SGL_API void BufferElementCursor::set(const bool4& value)
{
#if SGL_MACOS
    bool4 v = value;
#else
    uint4 v = {value.x ? 1 : 0, value.y ? 1 : 0, value.z ? 1 : 0, value.w ? 1 : 0};
#endif
    _set_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, 4);
}

template<>
SGL_API void BufferElementCursor::get(bool4& value) const
{
#if SGL_MACOS
    bool4 v;
#else
    uint4 v;
#endif
    _get_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, 4);
    value = {v.x != 0, v.y != 0, v.z != 0, v.w != 0};
}

template<>
SGL_API void BufferElementCursor::set(const DescriptorHandle& value)
{
    write_data(m_offset, &value.value, sizeof(value.value));
}

template<>
SGL_API void BufferElementCursor::get(DescriptorHandle& value) const
{
    read_data(m_offset, &value.value, sizeof(value.value));
}

void BufferElementCursor::write_data(size_t offset, const void* data, size_t size)
{
    m_buffer->write_data(offset, data, size);
}

void BufferElementCursor::read_data(size_t offset, void* data, size_t size) const
{
    m_buffer->read_data(offset, data, size);
}

BufferCursor::BufferCursor(ref<TypeLayoutReflection> element_layout, void* data, size_t size)
    : m_element_type_layout(std::move(element_layout))
    , m_buffer((uint8_t*)data)
    , m_size(size)
    , m_owner(false)
{
}

BufferCursor::BufferCursor(ref<TypeLayoutReflection> element_layout, size_t element_count)
    : m_element_type_layout(std::move(element_layout))
{
    m_size = element_count * m_element_type_layout->stride();
    m_buffer = new uint8_t[m_size];
    m_owner = true;
}

BufferCursor::BufferCursor(ref<TypeLayoutReflection> element_layout, ref<Buffer> resource, bool load_before_write)
    : m_element_type_layout(std::move(element_layout))
{
    m_resource = std::move(resource);
    m_size = m_resource->size();
    m_offset = 0;
    m_buffer = nullptr;
    m_owner = true;
    m_load_before_write = load_before_write;
}

BufferCursor::BufferCursor(
    ref<TypeLayoutReflection> element_layout,
    ref<Buffer> resource,
    size_t size,
    size_t offset,
    bool load_before_write
)
    : m_element_type_layout(std::move(element_layout))
{
    m_resource = std::move(resource);
    m_size = size;
    m_offset = offset;
    m_buffer = nullptr;
    m_owner = true;
    m_load_before_write = load_before_write;
}

BufferCursor::~BufferCursor()
{
    if (m_owner)
        delete[] m_buffer;
    m_buffer = nullptr;
}

BufferElementCursor BufferCursor::find_element(uint32_t index)
{
    SGL_CHECK(index < element_count(), "Index {} out of range in buffer with element count {}", index, element_count());
    BufferElementCursor element_cursor;
    element_cursor.m_buffer = ref(this);
    element_cursor.m_type_layout = m_element_type_layout;
    element_cursor.m_offset = index * m_element_type_layout->stride();
    return element_cursor;
}

void BufferCursor::write_data(size_t offset, const void* data, size_t size)
{
    if (!m_buffer) {
        // Load data on demand if haven't done so yet.
        SGL_CHECK(m_resource, "Buffer resource not set");
        if (m_load_before_write)
            load();
        else
            m_buffer = new uint8_t[m_size];
    }

    SGL_CHECK(offset + size <= m_size, "Buffer overflow");
    memcpy(m_buffer + offset, data, size);
}

void BufferCursor::read_data(size_t offset, void* data, size_t size) const
{
    if (!m_buffer) {
        // Load data on demand if haven't done so yet.
        // (note: treating local buffer as cache, so allowing call
        // to load_from_device to it even from const function).
        SGL_CHECK(m_resource, "Buffer resource not set");
        const_cast<BufferCursor*>(this)->load();
    }

    SGL_CHECK(offset + size <= m_size, "Buffer overflow");
    memcpy(data, m_buffer + offset, size);
}

void BufferCursor::load()
{
    if (m_resource) {
        if (!m_buffer) {
            m_buffer = new uint8_t[m_size];
        }
        if (m_resource->memory_type() != MemoryType::upload) {
            m_resource->get_data(m_buffer, m_size, m_offset);
        }
    }
}

void BufferCursor::apply()
{
    if (m_resource && m_buffer) {
        if (m_resource->memory_type() != MemoryType::read_back) {
            m_resource->set_data(m_buffer, m_size, m_offset);
        }
    }
}


} // namespace sgl
