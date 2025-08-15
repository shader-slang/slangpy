// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "raytracing.h"

#include "sgl/device/device.h"
#include "sgl/device/helpers.h"
#include "sgl/device/shader.h"

#include "sgl/core/type_utils.h"
#include "sgl/core/short_vector.h"

#include <slang-rhi/acceleration-structure-utils.h>

namespace sgl {

namespace detail {
    // TODO(slang-rhi) this is also in command.cpp and should be shared
    rhi::BufferOffsetPair to_rhi(const BufferOffsetPair& buffer_with_offset);
} // namespace detail

AccelerationStructureBuildDescConverter::AccelerationStructureBuildDescConverter(
    const AccelerationStructureBuildDesc& desc
)
{
    for (const auto& input : desc.inputs) {
        if (auto* instances = std::get_if<AccelerationStructureBuildInputInstances>(&input)) {
            rhi::AccelerationStructureBuildInput rhi_build_input{
                .type = rhi::AccelerationStructureBuildInputType::Instances,
                .instances{
                    .instanceBuffer = detail::to_rhi(instances->instance_buffer),
                    .instanceStride = instances->instance_stride,
                    .instanceCount = instances->instance_count,
                },
            };
            rhi_build_inputs.push_back(rhi_build_input);
        } else if (auto* triangles = std::get_if<AccelerationStructureBuildInputTriangles>(&input)) {
            rhi::AccelerationStructureBuildInput rhi_build_input{
                .type = rhi::AccelerationStructureBuildInputType::Triangles,
                .triangles{
                    .vertexBufferCount = narrow_cast<uint32_t>(triangles->vertex_buffers.size()),
                    .vertexFormat = static_cast<rhi::Format>(triangles->vertex_format),
                    .vertexCount = triangles->vertex_count,
                    .vertexStride = triangles->vertex_stride,
                    .indexBuffer = detail::to_rhi(triangles->index_buffer),
                    .indexFormat = static_cast<rhi::IndexFormat>(triangles->index_format),
                    .indexCount = triangles->index_count,
                    .preTransformBuffer = detail::to_rhi(triangles->pre_transform_buffer),
                    .flags = static_cast<rhi::AccelerationStructureGeometryFlags>(triangles->flags),
                },
            };
            for (size_t i = 0; i < triangles->vertex_buffers.size(); ++i)
                rhi_build_input.triangles.vertexBuffers[i] = detail::to_rhi(triangles->vertex_buffers[i]);
            rhi_build_inputs.push_back(rhi_build_input);
        } else if (auto* proceduralPrimitives = std::get_if<AccelerationStructureBuildInputProceduralPrimitives>(&input)) {
            rhi::AccelerationStructureBuildInput rhi_build_input{
                .type = rhi::AccelerationStructureBuildInputType::ProceduralPrimitives,
                .proceduralPrimitives{
                    .aabbBufferCount = narrow_cast<uint32_t>(proceduralPrimitives->aabb_buffers.size()),
                    .aabbStride = proceduralPrimitives->aabb_stride,
                    .primitiveCount = proceduralPrimitives->primitive_count,
                    .flags = static_cast<rhi::AccelerationStructureGeometryFlags>(proceduralPrimitives->flags),
                },
            };
            for (size_t i = 0; i < proceduralPrimitives->aabb_buffers.size(); ++i)
                rhi_build_input.proceduralPrimitives.aabbBuffers[i]
                    = detail::to_rhi(proceduralPrimitives->aabb_buffers[i]);
            rhi_build_inputs.push_back(rhi_build_input);
        }
    }

    rhi_desc.inputs = rhi_build_inputs.data();
    rhi_desc.inputCount = narrow_cast<uint32_t>(rhi_build_inputs.size());

    rhi_desc.motionOptions.keyCount = desc.motion_options.key_count;
    rhi_desc.motionOptions.timeStart = desc.motion_options.time_start;
    rhi_desc.motionOptions.timeEnd = desc.motion_options.time_end;

    rhi_desc.mode = static_cast<rhi::AccelerationStructureBuildMode>(desc.mode);
    rhi_desc.flags = static_cast<rhi::AccelerationStructureBuildFlags>(desc.flags);
}

AccelerationStructure::AccelerationStructure(ref<Device> device, AccelerationStructureDesc desc)
    : DeviceChild(std::move(device))
    , m_desc(std::move(desc))
{
    rhi::AccelerationStructureDesc rhi_desc{
        .size = m_desc.size,
        .label = m_desc.label.c_str(),
    };
    SLANG_RHI_CALL(
        m_device->rhi_device()->createAccelerationStructure(rhi_desc, m_rhi_acceleration_structure.writeRef())
    );
}

AccelerationStructure::~AccelerationStructure() { }

AccelerationStructureHandle AccelerationStructure::handle() const
{
    return m_rhi_acceleration_structure->getHandle();
}

std::string AccelerationStructure::to_string() const
{
    return fmt::format(
        "AccelerationStructure(\n"
        "  device = {},\n"
        "  size = {},\n"
        "  label = {}\n",
        ")",
        m_device,
        m_desc.size,
        m_desc.label
    );
}

AccelerationStructureInstanceList::AccelerationStructureInstanceList(ref<Device> device, size_t size)
    : DeviceChild(std::move(device))
{
    m_instance_type = rhi::getAccelerationStructureInstanceDescType(static_cast<rhi::DeviceType>(m_device->type()));
    m_instance_stride = rhi::getAccelerationStructureInstanceDescSize(m_instance_type);
    resize(size);
}

AccelerationStructureInstanceList::~AccelerationStructureInstanceList() { }

void AccelerationStructureInstanceList::resize(size_t size)
{
    m_instances.resize(size);
    m_dirty = true;
}

void AccelerationStructureInstanceList::write(size_t index, const AccelerationStructureInstanceDesc& instance)
{
    m_instances[index] = instance;
    m_dirty = true;
}

void AccelerationStructureInstanceList::write(size_t index, std::span<AccelerationStructureInstanceDesc> instances)
{
    std::copy(instances.begin(), instances.end(), m_instances.begin() + index);
    m_dirty = true;
}

ref<Buffer> AccelerationStructureInstanceList::buffer() const
{
    if (m_dirty) {
        size_t native_size = m_instances.size() * m_instance_stride;

        std::unique_ptr<uint8_t[]> native_descs(new uint8_t[native_size]);

        rhi::convertAccelerationStructureInstanceDescs(
            m_instances.size(),
            m_instance_type,
            native_descs.get(),
            m_instance_stride,
            reinterpret_cast<const rhi::AccelerationStructureInstanceDescGeneric*>(m_instances.data()),
            sizeof(rhi::AccelerationStructureInstanceDescGeneric)
        );

        m_buffer = m_device->create_buffer({
            .usage = BufferUsage::acceleration_structure_build_input,
            .data = native_descs.get(),
            .data_size = native_size,
        });

        m_dirty = false;
    }

    return m_buffer;
}

AccelerationStructureBuildInputInstances AccelerationStructureInstanceList::build_input_instances() const
{
    return AccelerationStructureBuildInputInstances{
        .instance_buffer = BufferOffsetPair{buffer(), 0},
        .instance_stride = narrow_cast<uint32_t>(m_instance_stride),
        .instance_count = narrow_cast<uint32_t>(m_instances.size()),
    };
}

std::string AccelerationStructureInstanceList::to_string() const
{
    return fmt::format(
        "AccelerationStructureInstanceList(\n"
        "  device = {}\n"
        "  size = {}\n"
        ")",
        m_device,
        m_instances.size()
    );
}

ShaderTable::ShaderTable(ref<Device> device, ShaderTableDesc desc)
    : DeviceChild(std::move(device))
{
    short_vector<const char*, 16> rhi_ray_gen_entry_points;
    rhi_ray_gen_entry_points.reserve(desc.ray_gen_entry_points.size());
    for (const auto& name : desc.ray_gen_entry_points)
        rhi_ray_gen_entry_points.push_back(name.c_str());

    short_vector<const char*, 16> rhi_miss_entry_points;
    rhi_miss_entry_points.reserve(desc.miss_entry_points.size());
    for (const auto& name : desc.miss_entry_points)
        rhi_miss_entry_points.push_back(name.c_str());

    short_vector<const char*, 16> rhi_hit_group_names;
    rhi_hit_group_names.reserve(desc.hit_group_names.size());
    for (const auto& name : desc.hit_group_names)
        rhi_hit_group_names.push_back(name.c_str());

    short_vector<const char*, 16> rhi_callable_names;
    rhi_callable_names.reserve(desc.callable_entry_points.size());
    for (const auto& name : desc.callable_entry_points)
        rhi_callable_names.push_back(name.c_str());

    rhi::ShaderTableDesc rhi_desc{
        .rayGenShaderCount = narrow_cast<uint32_t>(rhi_ray_gen_entry_points.size()),
        .rayGenShaderEntryPointNames = rhi_ray_gen_entry_points.data(),
        .rayGenShaderRecordOverwrites = nullptr,
        .missShaderCount = narrow_cast<uint32_t>(rhi_miss_entry_points.size()),
        .missShaderEntryPointNames = rhi_miss_entry_points.data(),
        .missShaderRecordOverwrites = nullptr,
        .hitGroupCount = narrow_cast<uint32_t>(rhi_hit_group_names.size()),
        .hitGroupNames = rhi_hit_group_names.data(),
        .hitGroupRecordOverwrites = nullptr,
        .callableShaderCount = narrow_cast<uint32_t>(rhi_callable_names.size()),
        .callableShaderEntryPointNames = rhi_callable_names.data(),
        .callableShaderRecordOverwrites = nullptr,
        .program = desc.program->rhi_shader_program(),
    };

    SLANG_RHI_CALL(m_device->rhi_device()->createShaderTable(rhi_desc, m_rhi_shader_table.writeRef()));
}

ShaderTable::~ShaderTable() { }

std::string ShaderTable::to_string() const
{
    return fmt::format(
        "ShaderTable(\n"
        "  device = {}\n"
        ")",
        m_device
    );
}

} // namespace sgl
