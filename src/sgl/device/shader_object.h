// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/config.h"
#include "sgl/core/object.h"
#include "sgl/core/type_utils.h"

#include "sgl/device/fwd.h"
#include "sgl/device/shader_offset.h"
#include "sgl/device/resource.h"

#include <slang-rhi.h>

#include <string_view>
#include <vector>
#include <set>

namespace sgl {

class SGL_API ShaderObject : public Object {
    SGL_OBJECT(ShaderObject)
public:
    ShaderObject(ref<Device> device, rhi::IShaderObject* shader_object, bool retain = true);
    virtual ~ShaderObject();

    Device* device() const { return m_device.get(); }

    virtual ref<const TypeLayoutReflection> element_type_layout() const;

    virtual slang::TypeLayoutReflection* slang_element_type_layout() const;

    virtual uint32_t get_entry_point_count() const;
    virtual ref<ShaderObject> get_entry_point(uint32_t index);

    virtual ref<ShaderObject> get_object(const ShaderOffset& offset);
    virtual void set_object(const ShaderOffset& offset, const ref<ShaderObject>& object);

    virtual void set_buffer(const ShaderOffset& offset, const ref<Buffer>& buffer);
    virtual void set_buffer_view(const ShaderOffset& offset, const ref<BufferView>& buffer_view);
    virtual void set_texture(const ShaderOffset& offset, const ref<Texture>& texture);
    virtual void set_texture_view(const ShaderOffset& offset, const ref<TextureView>& texture_view);
    virtual void set_sampler(const ShaderOffset& offset, const ref<Sampler>& sampler);
    virtual void
    set_acceleration_structure(const ShaderOffset& offset, const ref<AccelerationStructure>& acceleration_structure);
    virtual void set_descriptor_handle(const ShaderOffset& offset, const DescriptorHandle& handle);
    virtual void set_data(const ShaderOffset& offset, const void* data, size_t size);

    virtual void
    set_cuda_tensor_view_buffer(const ShaderOffset& offset, const cuda::TensorView& tensor_view, bool is_uav);
    virtual void set_cuda_tensor_view_pointer(const ShaderOffset& offset, const cuda::TensorView& tensor_view);
    virtual void get_cuda_interop_buffers(std::vector<ref<cuda::InteropBuffer>>& cuda_interop_buffers) const;

    rhi::IShaderObject* rhi_shader_object() const { return m_shader_object; }

protected:
    ref<Device> m_device;
    rhi::IShaderObject* m_shader_object;
    bool m_retain;
    std::vector<ref<cuda::InteropBuffer>> m_cuda_interop_buffers;
    std::set<ref<ShaderObject>> m_objects;
};

} // namespace sgl
