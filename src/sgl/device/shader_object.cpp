// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shader_object.h"

#include "sgl/device/reflection.h"
#include "sgl/device/resource.h"
#include "sgl/device/sampler.h"
#include "sgl/device/helpers.h"
#include "sgl/device/command.h"
#include "sgl/device/shader.h"
#include "sgl/device/device.h"
#include "sgl/device/cuda_interop.h"

namespace sgl {

inline rhi::ShaderOffset rhi_shader_offset(const ShaderOffset& offset)
{
    return {
        .uniformOffset = offset.uniform_offset,
        .bindingRangeIndex = offset.binding_range_index,
        .bindingArrayIndex = offset.binding_array_index,
    };
}

inline rhi::DescriptorHandle rhi_descriptor_handle(const DescriptorHandle& handle)
{
    return {
        .type = static_cast<rhi::DescriptorHandleType>(handle.type),
        .value = handle.value,
    };
}

//
// ShaderObject
//

ShaderObject::ShaderObject(ref<Device> device, rhi::IShaderObject* shader_object, bool retain)
    : m_device(std::move(device))
    , m_shader_object(shader_object)
    , m_retain(retain)

{
    if (m_retain)
        m_shader_object->addRef();
}

ShaderObject::~ShaderObject()
{
    if (m_retain)
        m_shader_object->release();
}

ref<const TypeLayoutReflection> ShaderObject::element_type_layout() const
{
    return TypeLayoutReflection::from_slang(ref(this), slang_element_type_layout());
}

slang::TypeLayoutReflection* ShaderObject::slang_element_type_layout() const
{
    return m_shader_object->getElementTypeLayout();
}

uint32_t ShaderObject::get_entry_point_count() const
{
    return m_shader_object->getEntryPointCount();
}

ref<ShaderObject> ShaderObject::get_entry_point(uint32_t index)
{
    ref<ShaderObject> shader_object = make_ref<ShaderObject>(m_device, m_shader_object->getEntryPoint(index));
    // TODO(slang-rhi) this is required to keep shader object's alive (shader cursor uses weak references)
    m_objects.insert(shader_object);
    return shader_object;
}

ref<ShaderObject> ShaderObject::get_object(const ShaderOffset& offset)
{
    ref<ShaderObject> shader_object
        = make_ref<ShaderObject>(m_device, m_shader_object->getObject(rhi_shader_offset(offset)));
    // TODO(slang-rhi) this is required to keep shader object's alive (shader cursor uses weak references)
    m_objects.insert(shader_object);
    return shader_object;
}

void ShaderObject::set_object(const ShaderOffset& offset, const ref<ShaderObject>& object)
{
    SLANG_RHI_CALL(m_shader_object->setObject(rhi_shader_offset(offset), object ? object->rhi_shader_object() : nullptr)
    );
}

void ShaderObject::set_buffer(const ShaderOffset& offset, const ref<Buffer>& buffer)
{
    SLANG_RHI_CALL(
        m_shader_object->setBinding(rhi_shader_offset(offset), rhi::Binding(buffer ? buffer->rhi_buffer() : nullptr))
    );
}

void ShaderObject::set_buffer_view(const ShaderOffset& offset, const ref<BufferView>& buffer_view)
{
    SLANG_RHI_CALL(m_shader_object->setBinding(
        rhi_shader_offset(offset),
        rhi::Binding(
            buffer_view->buffer()->rhi_buffer(),
            rhi::BufferRange{buffer_view->range().offset, buffer_view->range().size}
        )
    ));
}

void ShaderObject::set_texture(const ShaderOffset& offset, const ref<Texture>& texture)
{
    SLANG_RHI_CALL(
        m_shader_object->setBinding(rhi_shader_offset(offset), rhi::Binding(texture ? texture->rhi_texture() : nullptr))
    );
}

void ShaderObject::set_texture_view(const ShaderOffset& offset, const ref<TextureView>& texture_view)
{
    SLANG_RHI_CALL(m_shader_object->setBinding(
        rhi_shader_offset(offset),
        rhi::Binding(texture_view ? texture_view->rhi_texture_view() : nullptr)
    ));
}

void ShaderObject::set_sampler(const ShaderOffset& offset, const ref<Sampler>& sampler)
{
    SLANG_RHI_CALL(
        m_shader_object->setBinding(rhi_shader_offset(offset), rhi::Binding(sampler ? sampler->rhi_sampler() : nullptr))
    );
}

void ShaderObject::set_acceleration_structure(
    const ShaderOffset& offset,
    const ref<AccelerationStructure>& acceleration_structure
)
{
    SLANG_RHI_CALL(m_shader_object->setBinding(
        rhi_shader_offset(offset),
        rhi::Binding(acceleration_structure ? acceleration_structure->rhi_acceleration_structure() : nullptr)
    ));
}

void ShaderObject::set_descriptor_handle(const ShaderOffset& offset, const DescriptorHandle& handle)
{
    SLANG_RHI_CALL(m_shader_object->setDescriptorHandle(rhi_shader_offset(offset), rhi_descriptor_handle(handle)));
}

void ShaderObject::set_data(const ShaderOffset& offset, const void* data, size_t size)
{
    SLANG_RHI_CALL(m_shader_object->setData(rhi_shader_offset(offset), data, size));
}

void ShaderObject::set_cuda_tensor_view(const ShaderOffset& offset, const cuda::TensorView& tensor_view, bool is_uav)
{
    SGL_CHECK(m_device->supports_cuda_interop(), "Device does not support CUDA interop");
    ref<cuda::InteropBuffer> cuda_interop_buffer = make_ref<cuda::InteropBuffer>(m_device, tensor_view, is_uav);
    set_buffer(offset, cuda_interop_buffer->buffer());
    m_cuda_interop_buffers.push_back(cuda_interop_buffer);
}

void ShaderObject::get_cuda_interop_buffers(std::vector<ref<cuda::InteropBuffer>>& cuda_interop_buffers) const
{
    cuda_interop_buffers
        .insert(cuda_interop_buffers.end(), m_cuda_interop_buffers.begin(), m_cuda_interop_buffers.end());
}

} // namespace sgl
