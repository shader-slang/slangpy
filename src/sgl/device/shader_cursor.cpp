// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shader_cursor.h"

#include "sgl/device/shader_object.h"
#include "sgl/device/device.h"
#include "sgl/device/resource.h"
#include "sgl/device/cuda_interop.h"
#include "sgl/device/cursor_utils.h"

#include "sgl/core/error.h"

#include "sgl/math/vector_types.h"
#include "sgl/math/matrix_types.h"

// TODO: Decide if we want to disable / optimize type checks
// currently can represent 50% of the cost of writes in
// certain situations.
#define SGL_ENABLE_CURSOR_TYPE_CHECKS

namespace sgl {

ShaderCursor::ShaderCursor(ShaderObject* shader_object)
    : m_type_layout(shader_object->slang_element_type_layout())
    , m_shader_object(shader_object)
    , m_offset(ShaderOffset::zero())
{
    SGL_ASSERT(m_type_layout);
    SGL_ASSERT(m_shader_object);
    SGL_ASSERT(m_offset.is_valid());
}

ShaderCursor::ShaderCursor(
    ShaderObject* shader_object,
    bool need_dereference,
    slang::TypeLayoutReflection* parent_type_layout
)
    : m_type_layout(shader_object->slang_element_type_layout())
    , m_shader_object(shader_object)
    , m_offset(ShaderOffset::zero())
{
    SGL_ASSERT(m_type_layout);
    SGL_ASSERT(m_shader_object);
    SGL_ASSERT(m_offset.is_valid());

    if (need_dereference) {
        Device* device = m_shader_object->device();

        if (device->type() == DeviceType::metal) {
            slang::ISession* session = device->slang_session()->get_slang_session();
            m_type_layout = session->getTypeLayout(
                parent_type_layout->getElementTypeLayout()->getType(),
                0,
                slang::LayoutRules::MetalArgumentBufferTier2
            );
        }
    }
}

std::string ShaderCursor::to_string() const
{
    return "ShaderCursor()";
}

bool ShaderCursor::is_reference() const
{
    switch ((TypeReflection::Kind)m_type_layout->getKind()) {
    case TypeReflection::Kind::constant_buffer:
    case TypeReflection::Kind::parameter_block:
        return true;
    default:
        return false;
    }
}

ShaderCursor ShaderCursor::dereference() const
{
    SGL_CHECK(is_valid(), "Invalid cursor");
    switch ((TypeReflection::Kind)m_type_layout->getKind()) {
    case TypeReflection::Kind::constant_buffer:
    case TypeReflection::Kind::parameter_block: {
        ShaderCursor d = ShaderCursor(m_shader_object->get_object(m_offset), true, m_type_layout);
        return d;
    }
    default:
        return {};
    }
}

//
// Navigation
//

ShaderCursor ShaderCursor::operator[](std::string_view name) const
{
    SGL_CHECK(is_valid(), "Invalid cursor");
    ShaderCursor result = find_field(name);
    SGL_CHECK(result.is_valid(), "Field \"{}\" not found.", name);
    return result;
}

ShaderCursor ShaderCursor::operator[](uint32_t index) const
{
    SGL_CHECK(is_valid(), "Invalid cursor");
    ShaderCursor result = find_element(index);
    SGL_CHECK(result.is_valid(), "Element {} not found.", index);
    return result;
}

ShaderCursor ShaderCursor::find_field(std::string_view name) const
{
    if (!is_valid())
        return *this;

    // If the cursor is valid, we want to consider the type of data
    // it is referencing.
    //
    switch ((TypeReflection::Kind)m_type_layout->getKind()) {
        // The easy/expected case is when the value has a structure type.
        //
    case TypeReflection::Kind::struct_: {
        // We start by looking up the index of a field matching `name`.
        //
        // If there is no such field, we have an error.
        //
        int32_t field_index = (int32_t)m_type_layout->findFieldIndexByName(name.data(), name.data() + name.size());
        if (field_index < 0)
            break;

        // Once we know the index of the field being referenced,
        // we create a cursor to point at the field, based on
        // the offset information already in this cursor, plus
        // offsets derived from the field's layout.
        //
        slang::VariableLayoutReflection* field_layout = m_type_layout->getFieldByIndex(field_index);
        ShaderCursor field_cursor;

        // The field cursor will point into the same parent object.
        //
        field_cursor.m_shader_object = m_shader_object;

        // The type being pointed to is the type of the field.
        //
        field_cursor.m_type_layout = field_layout->getTypeLayout();

        // The byte offset is the current offset plus the relative offset of the field.
        // The offset in binding ranges is computed similarly.
        //
        field_cursor.m_offset.uniform_offset
            = m_offset.uniform_offset + narrow_cast<uint32_t>(field_layout->getOffset());
        field_cursor.m_offset.binding_range_index = m_offset.binding_range_index
            + narrow_cast<int32_t>(m_type_layout->getFieldBindingRangeOffset(field_index));

        // The index of the field within any binding ranges will be the same
        // as the index computed for the parent structure.
        //
        // Note: this case would arise for an array of structures with texture-type
        // fields. Suppose we have:
        //
        //      struct S { Texture2D t; Texture2D u; }
        //      S g[4];
        //
        // In this scenario, `g` holds two binding ranges:
        //
        // * Range #0 comprises 4 textures, representing `g[...].t`
        // * Range #1 comprises 4 textures, representing `g[...].u`
        //
        // A cursor for `g[2]` would have a `binding_range_index` of zero but
        // a `binding_array_index` of 2, iindicating that we could end up
        // referencing either range, but no matter what we know the index
        // is 2. Thus when we form a cursor for `g[2].u` we want to
        // apply the binding range offset to get a `binding_range_index` of
        // 1, while the `binding_array_index` is unmodified.
        //
        // The result is that `g[2].u` is stored in range #1 at array index 2.
        //
        field_cursor.m_offset.binding_array_index = m_offset.binding_array_index;

        return field_cursor;
    }

    // In some cases the user might be trying to acess a field by name
    // from a cursor that references a constant buffer or parameter block,
    // and in these cases we want the access to Just Work.
    //
    case TypeReflection::Kind::constant_buffer:
    case TypeReflection::Kind::parameter_block: {
        // We basically need to "dereference" the current cursor
        // to go from a pointer to a constant buffer to a pointer
        // to the *contents* of the constant buffer.
        //
        ShaderCursor d = dereference();
        return d.find_field(name);
    }

    default:
        break;
    }

#if 0
    // If a cursor is pointing at a root shader object (created for a
    // program), then we will also iterate over the entry point shader
    // objects attached to it and look for a matching parameter name
    // on them.
    //
    // This is a bit of "do what I mean" logic and could potentially
    // lead to problems if there could be multiple entry points with
    // the same parameter name.
    //
    // TODO: figure out whether we should support this long-term.
    //
    auto entryPointCount = (GfxIndex)m_shader_object->getEntryPointCount();
    for (GfxIndex e = 0; e < entryPointCount; ++e) {
        ComPtr<IShaderObject> entryPoint;
        m_shader_object->getEntryPoint(e, entryPoint.writeRef());

        ShaderCursor entryPointCursor(entryPoint);

        auto result = entryPointCursor.getField(name, nameEnd, outCursor);
        if (SLANG_SUCCEEDED(result))
            return result;
    }
#endif
    return {};
}

ShaderCursor ShaderCursor::find_element(uint32_t index) const
{
    if (!is_valid())
        return *this;

#if 0
    if (m_containerType != ShaderObjectContainerType::None) {
        ShaderCursor element_cursor;
        element_cursor.m_shader_object = m_shader_object;
        element_cursor.m_type_layout = m_type_layout->getElementTypeLayout();
        element_cursor.m_containerType = m_containerType;
        element_cursor.m_offset.uniform_offset = index * m_type_layout->getStride();
        element_cursor.m_offset.binding_range_index = 0;
        element_cursor.m_offset.binding_array_index = index;
        return element_cursor;
    }
#endif

    switch ((TypeReflection::Kind)m_type_layout->getKind()) {
    case TypeReflection::Kind::array: {
        ShaderCursor element_cursor;
        element_cursor.m_shader_object = m_shader_object;
        element_cursor.m_type_layout = m_type_layout->getElementTypeLayout();
        element_cursor.m_offset.uniform_offset = m_offset.uniform_offset
            + index
                * narrow_cast<uint32_t>(
                    m_type_layout->getElementStride(SlangParameterCategory::SLANG_PARAMETER_CATEGORY_UNIFORM)
                );
        element_cursor.m_offset.binding_range_index = m_offset.binding_range_index;
        element_cursor.m_offset.binding_array_index
            = m_offset.binding_array_index * narrow_cast<uint32_t>(m_type_layout->getElementCount()) + index;
        return element_cursor;
    } break;

    case TypeReflection::Kind::vector:
    case TypeReflection::Kind::matrix: {
        ShaderCursor field_cursor;
        field_cursor.m_shader_object = m_shader_object;
        field_cursor.m_type_layout = m_type_layout->getElementTypeLayout();
        field_cursor.m_offset.uniform_offset = m_offset.uniform_offset
            + narrow_cast<uint32_t>(m_type_layout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM)) * index;
        field_cursor.m_offset.binding_range_index = m_offset.binding_range_index;
        field_cursor.m_offset.binding_array_index = m_offset.binding_array_index;
        return field_cursor;
    } break;

    default:
        break;
    }

    return {};
}

ShaderCursor ShaderCursor::find_entry_point(uint32_t index) const
{
    if (!is_valid())
        return *this;

    // TODO check index
    // uint32_t count = m_shader_object->get_entry_point_count();
    return ShaderCursor(m_shader_object->get_entry_point(index));
}

//
// Resource binding
//

inline bool is_parameter_block(slang::TypeReflection* type)
{
    return (TypeReflection::Kind)type->getKind() == TypeReflection::Kind::parameter_block;
}

inline bool is_resource_type(slang::TypeReflection* type)
{
    switch ((TypeReflection::Kind)type->getKind()) {
    case TypeReflection::Kind::constant_buffer:
    case TypeReflection::Kind::resource:
    case TypeReflection::Kind::sampler_state:
    case TypeReflection::Kind::texture_buffer:
    case TypeReflection::Kind::shader_storage_buffer:
    case TypeReflection::Kind::parameter_block:
        return true;
    default:
        return false;
    }
}

inline bool is_buffer_resource_type(slang::TypeReflection* type)
{
    switch ((TypeReflection::Kind)type->getKind()) {
    case TypeReflection::Kind::constant_buffer:
    case TypeReflection::Kind::resource: {
        auto shape = (TypeReflection::ResourceShape)(
            type->getResourceShape() & SlangResourceShape::SLANG_RESOURCE_BASE_SHAPE_MASK
        );
        switch (shape) {
        case TypeReflection::ResourceShape::texture_buffer:
        case TypeReflection::ResourceShape::structured_buffer:
        case TypeReflection::ResourceShape::byte_address_buffer:
            return true;
        default:
            return false;
        }
    }
    case TypeReflection::Kind::texture_buffer:
    case TypeReflection::Kind::shader_storage_buffer:
    case TypeReflection::Kind::parameter_block:
        return true;
    default:
        return false;
    }
}

inline bool is_texture_resource_type(slang::TypeReflection* type)
{
    switch ((TypeReflection::Kind)type->getKind()) {
    case TypeReflection::Kind::resource: {
        auto shape = (TypeReflection::ResourceShape)(
            type->getResourceShape() & SlangResourceShape::SLANG_RESOURCE_BASE_SHAPE_MASK
        );
        switch (shape) {
        case TypeReflection::ResourceShape::texture_1d:
        case TypeReflection::ResourceShape::texture_2d:
        case TypeReflection::ResourceShape::texture_3d:
        case TypeReflection::ResourceShape::texture_cube:
            return true;
        default:
            return false;
        }
        break;
    }
    default:
        return false;
    }
}

inline bool is_sampler_type(slang::TypeReflection* type)
{
    return (TypeReflection::Kind)type->getKind() == TypeReflection::Kind::sampler_state;
}

inline bool is_shader_resource_type(slang::TypeReflection* type)
{
    return (TypeReflection::ResourceAccess)type->getResourceAccess() == TypeReflection::ResourceAccess::read;
}

inline bool is_unordered_access_type(slang::TypeReflection* type)
{
    return (TypeReflection::ResourceAccess)type->getResourceAccess() == TypeReflection::ResourceAccess::read_write;
}

inline bool is_acceleration_structure_resource_type(slang::TypeReflection* type)
{
    return (TypeReflection::Kind)type->getKind() == TypeReflection::Kind::resource
        && (TypeReflection::ResourceShape)type->getResourceShape()
        == TypeReflection::ResourceShape::acceleration_structure;
}

void ShaderCursor::set_buffer(const ref<Buffer>& buffer) const
{
    slang::TypeReflection* type = cursor_utils::unwrap_array(m_type_layout)->getType();

    if (type->getKind() == slang::TypeReflection::Kind::Pointer) {
        set_pointer(buffer->device_address());
    } else {
        SGL_CHECK(is_buffer_resource_type(type), "\"{}\" cannot bind a buffer", m_type_layout->getName());
        m_shader_object->set_buffer(m_offset, buffer);
    }
}

void ShaderCursor::set_buffer_view(const ref<BufferView>& buffer_view) const
{
    slang::TypeReflection* type = cursor_utils::unwrap_array(m_type_layout)->getType();

    if (type->getKind() == slang::TypeReflection::Kind::Pointer) {
        set_pointer(buffer_view->buffer()->device_address() + buffer_view->range().offset);
    } else {
        SGL_CHECK(is_buffer_resource_type(type), "\"{}\" cannot bind a buffer view", m_type_layout->getName());
        m_shader_object->set_buffer_view(m_offset, buffer_view);
    }
}

void ShaderCursor::set_texture(const ref<Texture>& texture) const
{
    slang::TypeReflection* type = cursor_utils::unwrap_array(m_type_layout)->getType();

    SGL_CHECK(is_texture_resource_type(type), "\"{}\" cannot bind a texture", m_type_layout->getName());

    m_shader_object->set_texture(m_offset, texture);
}

void ShaderCursor::set_texture_view(const ref<TextureView>& texture_view) const
{
    slang::TypeReflection* type = cursor_utils::unwrap_array(m_type_layout)->getType();

    SGL_CHECK(is_texture_resource_type(type), "\"{}\" cannot bind a texture view", m_type_layout->getName());

    m_shader_object->set_texture_view(m_offset, texture_view);
}

void ShaderCursor::set_sampler(const ref<Sampler>& sampler) const
{
    slang::TypeReflection* type = cursor_utils::unwrap_array(m_type_layout)->getType();

    SGL_CHECK(is_sampler_type(type), "\"{}\" cannot bind a sampler", m_type_layout->getName());

    m_shader_object->set_sampler(m_offset, sampler);
}

void ShaderCursor::set_acceleration_structure(const ref<AccelerationStructure>& acceleration_structure) const
{
    slang::TypeReflection* type = m_type_layout->getType();

    SGL_CHECK(
        is_acceleration_structure_resource_type(type),
        "\"{}\" cannot bind an acceleration structure",
        m_type_layout->getName()
    );

    m_shader_object->set_acceleration_structure(m_offset, acceleration_structure);
}

void ShaderCursor::set_descriptor_handle(const DescriptorHandle& handle) const
{
    // TODO: check type
    // currently not possible because they are reported differently depending on the backend
    m_shader_object->set_descriptor_handle(m_offset, handle);
}

void ShaderCursor::set_data(const void* data, size_t size) const
{
    if ((TypeReflection::ParameterCategory)m_type_layout->getParameterCategory()
        != TypeReflection::ParameterCategory::uniform)
        SGL_THROW("\"{}\" cannot bind data", m_type_layout->getName());
    m_shader_object->set_data(m_offset, data, size);
}

void ShaderCursor::set_object(const ref<ShaderObject>& object) const
{
    slang::TypeReflection* type = m_type_layout->getType();

    SGL_CHECK(is_parameter_block(type), "\"{}\" cannot bind an object", m_type_layout->getName());

    m_shader_object->set_object(m_offset, object);
}

void ShaderCursor::set_cuda_tensor_view(const cuda::TensorView& tensor_view) const
{
    slang::TypeReflection* type = cursor_utils::unwrap_array(m_type_layout)->getType();

    SGL_CHECK(
        is_buffer_resource_type(type) || type->getKind() == slang::TypeReflection::Kind::Pointer,
        "\"{}\" cannot bind a CUDA tensor view",
        m_type_layout->getName()
    );

    if (type->getKind() == slang::TypeReflection::Kind::Pointer) {
        m_shader_object->set_cuda_tensor_view_pointer(m_offset, tensor_view);
    } else if (is_shader_resource_type(type)) {
        m_shader_object->set_cuda_tensor_view_buffer(m_offset, tensor_view, false);
    } else if (is_unordered_access_type(type)) {
        m_shader_object->set_cuda_tensor_view_buffer(m_offset, tensor_view, true);
    } else {
        SGL_THROW("\"{}\" expects a valid buffer", m_type_layout->getName());
    }
}

void ShaderCursor::set_pointer(uint64_t pointer_value) const
{
    size_t pointer_size = m_type_layout->getSize();
    SGL_CHECK(pointer_size == 8, "Expected a pointer type with size 8, got {}", pointer_size);
    set_data(&pointer_value, 8);
}

void ShaderCursor::_set_array_unsafe(
    const void* data,
    size_t size,
    size_t element_count,
    TypeReflection::ScalarType cpu_scalar_type
) const
{
    slang::TypeReflection* element_type = cursor_utils::unwrap_array(m_type_layout)->getType();
    size_t cpu_element_size = cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type);

    size_t element_stride = m_type_layout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
    if (element_stride == cpu_element_size) {
        m_shader_object->set_data(m_offset, data, size);
    } else {
        ShaderOffset offset = m_offset;
        for (size_t i = 0; i < element_count; ++i) {
            m_shader_object
                ->set_data(offset, reinterpret_cast<const uint8_t*>(data) + i * cpu_element_size, cpu_element_size);
            offset.uniform_offset += narrow_cast<uint32_t>(element_stride);
        }
    }
}

void ShaderCursor::_set_data(ShaderOffset offset, const void* data, size_t size) const
{
    m_shader_object->set_data(offset, data, size);
}

DeviceType ShaderCursor::_get_device_type() const
{
    return m_shader_object->device()->type();
}

// Explicit instantiation of the methods
template void
CursorWriteWrappers<ShaderCursor, ShaderOffset>::_set_array(const void*, size_t, TypeReflection::ScalarType, size_t)
    const;

template void
CursorWriteWrappers<ShaderCursor, ShaderOffset>::_set_vector(const void*, size_t, TypeReflection::ScalarType, int)
    const;

//
// Setter specializations
//

template<>
SGL_API void ShaderCursor::set(const ref<ShaderObject>& value) const
{
    set_object(value);
}

template<>
SGL_API void ShaderCursor::set(const ref<Buffer>& value) const
{
    set_buffer(value);
}

template<>
SGL_API void ShaderCursor::set(const ref<BufferView>& value) const
{
    set_buffer_view(value);
}

template<>
SGL_API void ShaderCursor::set(const ref<Texture>& value) const
{
    set_texture(value);
}

template<>
SGL_API void ShaderCursor::set(const ref<TextureView>& value) const
{
    set_texture_view(value);
}

template<>
SGL_API void ShaderCursor::set(const ref<Sampler>& value) const
{
    set_sampler(value);
}

template<>
SGL_API void ShaderCursor::set(const ref<AccelerationStructure>& value) const
{
    set_acceleration_structure(value);
}

template<>
SGL_API void ShaderCursor::set(const DescriptorHandle& value) const
{
    set_descriptor_handle(value);
}

#define SET_SCALAR(type, scalar_type)                                                                                  \
    template<>                                                                                                         \
    SGL_API void ShaderCursor::set(const type& value) const                                                            \
    {                                                                                                                  \
        _set_scalar(&value, sizeof(value), TypeReflection::ScalarType::scalar_type);                                   \
    }

#define SET_VECTOR(type, scalar_type)                                                                                  \
    template<>                                                                                                         \
    SGL_API void ShaderCursor::set(const type& value) const                                                            \
    {                                                                                                                  \
        _set_vector(&value, sizeof(value), TypeReflection::ScalarType::scalar_type, type::dimension);                  \
    }

#define SET_MATRIX(type, scalar_type)                                                                                  \
    template<>                                                                                                         \
    SGL_API void ShaderCursor::set(const type& value) const                                                            \
    {                                                                                                                  \
        _set_matrix(&value, sizeof(value), TypeReflection::ScalarType::scalar_type, type::rows, type::cols);           \
    }

SET_SCALAR(bool, bool_);
// bool1 case specifically cannot be handled due to:
// https://github.com/shader-slang/slang/issues/7441
// SET_VECTOR(bool1, bool_);
SET_VECTOR(bool2, bool_);
SET_VECTOR(bool3, bool_);
SET_VECTOR(bool4, bool_);

SET_SCALAR(int8_t, int8);
SET_SCALAR(uint8_t, uint8);
SET_SCALAR(int16_t, int16);
SET_SCALAR(uint16_t, uint16);

SET_SCALAR(int, int32);
SET_VECTOR(int1, int32);
SET_VECTOR(int2, int32);
SET_VECTOR(int3, int32);
SET_VECTOR(int4, int32);

SET_SCALAR(uint, uint32);
SET_VECTOR(uint1, uint32);
SET_VECTOR(uint2, uint32);
SET_VECTOR(uint3, uint32);
SET_VECTOR(uint4, uint32);

// MACOS treats these as separate types to int/uint, so they need to be
// provided as extra overloads for linking to succeed.
#if SGL_MACOS
SET_SCALAR(long, int32);
SET_SCALAR(unsigned long, uint32);
#endif

SET_SCALAR(int64_t, int64);
SET_SCALAR(uint64_t, uint64);

SET_SCALAR(float16_t, float16);
SET_VECTOR(float16_t1, float16);
SET_VECTOR(float16_t2, float16);
SET_VECTOR(float16_t3, float16);
SET_VECTOR(float16_t4, float16);

SET_SCALAR(float, float32);
SET_VECTOR(float1, float32);
SET_VECTOR(float2, float32);
SET_VECTOR(float3, float32);
SET_VECTOR(float4, float32);

SET_MATRIX(float2x2, float32);
SET_MATRIX(float2x3, float32);
SET_MATRIX(float2x4, float32);

SET_MATRIX(float3x2, float32);
SET_MATRIX(float3x3, float32);
SET_MATRIX(float3x4, float32);

SET_MATRIX(float4x2, float32);
SET_MATRIX(float4x3, float32);
SET_MATRIX(float4x4, float32);

SET_SCALAR(double, float64);

#undef SET_SCALAR
#undef SET_VECTOR
#undef SET_MATRIX

template<>
SGL_API void ShaderCursor::set(const bool1& v) const
{
    SGL_CHECK(_get_device_type() != DeviceType::cuda, "bool1 currently not supported due to CUDA backend issues.");
    _set_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, 1);
}

} // namespace sgl
