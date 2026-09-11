// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "raytracing.h"

#include "sgl/device/device.h"
#include "sgl/device/helpers.h"
#include "sgl/device/shader.h"
#include "sgl/device/shader_cursor.h"

#include "sgl/core/error.h"
#include "sgl/core/type_utils.h"
#include "sgl/core/short_vector.h"

#include <slang-rhi/acceleration-structure-utils.h>

namespace sgl {

MicromapBuildDescConverter::MicromapBuildDescConverter(const MicromapBuildDesc& desc)
{
    rhi_histogram.reserve(desc.histogram.size());
    for (const auto& usage : desc.histogram) {
        rhi_histogram.push_back({
            .count = usage.count,
            .subdivisionLevel = usage.subdivision_level,
            .format = static_cast<uint32_t>(usage.format),
        });
    }

    rhi_desc = {
        .type = static_cast<rhi::MicromapType>(desc.type),
        .flags = static_cast<rhi::MicromapBuildFlags>(desc.flags),
        .dataBuffer = detail::to_rhi(desc.data_buffer),
        .descriptorBuffer = detail::to_rhi(desc.descriptor_buffer),
        .descriptorStride = desc.descriptor_stride,
        .histogram = rhi_histogram.data(),
        .histogramCount = narrow_cast<uint32_t>(rhi_histogram.size()),
    };
}

rhi::ClusterOperationParams detail::to_rhi(const ClusterOperationParams& params)
{
    return {
        .maxArgCount = params.max_arg_count,
        .type = static_cast<rhi::ClusterOperationType>(params.type),
        .mode = static_cast<rhi::ClusterOperationMode>(params.mode),
        .flags = static_cast<rhi::ClusterOperationFlags>(params.flags),
        .move{
            .type = static_cast<rhi::ClusterOperationMoveType>(params.move.type),
            .maxSize = params.move.max_size,
        },
        .clas{
            .vertexFormat = static_cast<rhi::Format>(params.clas.vertex_format),
            .maxGeometryIndex = params.clas.max_geometry_index,
            .maxUniqueGeometryCount = params.clas.max_unique_geometry_count,
            .maxTriangleCount = params.clas.max_triangle_count,
            .maxVertexCount = params.clas.max_vertex_count,
            .maxTotalTriangleCount = params.clas.max_total_triangle_count,
            .maxTotalVertexCount = params.clas.max_total_vertex_count,
            .minPositionTruncateBitCount = params.clas.min_position_truncate_bit_count,
        },
        .blas{
            .maxClasCount = params.blas.max_clas_count,
            .maxTotalClasCount = params.blas.max_total_clas_count,
        },
    };
}

AccelerationStructureBuildDescConverter::AccelerationStructureBuildDescConverter(
    const AccelerationStructureBuildDesc& desc
)
{
    rhi_build_inputs.reserve(desc.inputs.size());
    rhi_opacity_micromap_descs.resize(desc.inputs.size());
    rhi_opacity_micromap_usage_counts.resize(desc.inputs.size());

    for (size_t input_index = 0; input_index < desc.inputs.size(); ++input_index) {
        const auto& input = desc.inputs[input_index];
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

            if (triangles->opacity_micromap) {
                const auto& opacity_micromap = *triangles->opacity_micromap;
                auto& rhi_usage_counts = rhi_opacity_micromap_usage_counts[input_index];
                rhi_usage_counts.reserve(opacity_micromap.usage_counts.size());
                for (const auto& usage : opacity_micromap.usage_counts) {
                    rhi_usage_counts.push_back({
                        .count = usage.count,
                        .subdivisionLevel = usage.subdivision_level,
                        .format = static_cast<uint32_t>(usage.format),
                    });
                }

                auto& rhi_opacity_micromap = rhi_opacity_micromap_descs[input_index];
                rhi_opacity_micromap.link = {
                    .micromap = opacity_micromap.micromap ? opacity_micromap.micromap->rhi_micromap() : nullptr,
                    .indexingMode = static_cast<rhi::MicromapIndexingMode>(opacity_micromap.indexing_mode),
                    .indexBuffer = detail::to_rhi(opacity_micromap.index_buffer),
                    .indexFormat = static_cast<rhi::MicromapIndexFormat>(opacity_micromap.index_format),
                    .indexStride = opacity_micromap.index_stride,
                    .baseMicromapIndex = opacity_micromap.base_micromap_index,
                    .usageCounts = rhi_usage_counts.data(),
                    .usageCount = narrow_cast<uint32_t>(rhi_usage_counts.size()),
                };
                rhi_build_input.triangles.next = &rhi_opacity_micromap;
            }
            rhi_build_inputs.push_back(rhi_build_input);
        } else if (auto* procedural_primitives
                   = std::get_if<AccelerationStructureBuildInputProceduralPrimitives>(&input)) {
            rhi::AccelerationStructureBuildInput rhi_build_input{
                .type = rhi::AccelerationStructureBuildInputType::ProceduralPrimitives,
                .proceduralPrimitives{
                    .aabbBufferCount = narrow_cast<uint32_t>(procedural_primitives->aabb_buffers.size()),
                    .aabbStride = procedural_primitives->aabb_stride,
                    .primitiveCount = procedural_primitives->primitive_count,
                    .flags = static_cast<rhi::AccelerationStructureGeometryFlags>(procedural_primitives->flags),
                },
            };
            for (size_t i = 0; i < procedural_primitives->aabb_buffers.size(); ++i)
                rhi_build_input.proceduralPrimitives.aabbBuffers[i]
                    = detail::to_rhi(procedural_primitives->aabb_buffers[i]);
            rhi_build_inputs.push_back(rhi_build_input);
        } else if (auto* spheres = std::get_if<AccelerationStructureBuildInputSpheres>(&input)) {
            rhi::AccelerationStructureBuildInput rhi_build_input{
                .type = rhi::AccelerationStructureBuildInputType::Spheres,
                .spheres{
                    .vertexBufferCount = narrow_cast<uint32_t>(spheres->vertex_position_buffers.size()),
                    .vertexCount = spheres->vertex_count,
                    .vertexPositionFormat = static_cast<rhi::Format>(spheres->vertex_position_format),
                    .vertexPositionStride = spheres->vertex_position_stride,
                    .vertexRadiusFormat = static_cast<rhi::Format>(spheres->vertex_radius_format),
                    .vertexRadiusStride = spheres->vertex_radius_stride,
                    .indexBuffer = detail::to_rhi(spheres->index_buffer),
                    .indexFormat = static_cast<rhi::IndexFormat>(spheres->index_format),
                    .indexCount = spheres->index_count,
                    .flags = static_cast<rhi::AccelerationStructureGeometryFlags>(spheres->flags),
                },
            };
            for (size_t i = 0; i < spheres->vertex_position_buffers.size(); ++i)
                rhi_build_input.spheres.vertexPositionBuffers[i] = detail::to_rhi(spheres->vertex_position_buffers[i]);
            for (size_t i = 0; i < spheres->vertex_radius_buffers.size(); ++i)
                rhi_build_input.spheres.vertexRadiusBuffers[i] = detail::to_rhi(spheres->vertex_radius_buffers[i]);
            rhi_build_inputs.push_back(rhi_build_input);
        } else if (auto* linear_swept_spheres
                   = std::get_if<AccelerationStructureBuildInputLinearSweptSpheres>(&input)) {
            rhi::AccelerationStructureBuildInput rhi_build_input{
                .type = rhi::AccelerationStructureBuildInputType::LinearSweptSpheres,
                .linearSweptSpheres{
                    .vertexBufferCount = narrow_cast<uint32_t>(linear_swept_spheres->vertex_position_buffers.size()),
                    .vertexCount = linear_swept_spheres->vertex_count,
                    .primitiveCount = linear_swept_spheres->primitive_count,
                    .vertexPositionFormat = static_cast<rhi::Format>(linear_swept_spheres->vertex_position_format),
                    .vertexPositionStride = linear_swept_spheres->vertex_position_stride,
                    .vertexRadiusFormat = static_cast<rhi::Format>(linear_swept_spheres->vertex_radius_format),
                    .vertexRadiusStride = linear_swept_spheres->vertex_radius_stride,
                    .indexBuffer = detail::to_rhi(linear_swept_spheres->index_buffer),
                    .indexFormat = static_cast<rhi::IndexFormat>(linear_swept_spheres->index_format),
                    .indexCount = linear_swept_spheres->index_count,
                    .indexingMode
                    = static_cast<rhi::LinearSweptSpheresIndexingMode>(linear_swept_spheres->indexing_mode),
                    .endCapsMode = static_cast<rhi::LinearSweptSpheresEndCapsMode>(linear_swept_spheres->end_caps_mode),
                    .flags = static_cast<rhi::AccelerationStructureGeometryFlags>(linear_swept_spheres->flags),
                },
            };
            for (size_t i = 0; i < linear_swept_spheres->vertex_position_buffers.size(); ++i)
                rhi_build_input.linearSweptSpheres.vertexPositionBuffers[i]
                    = detail::to_rhi(linear_swept_spheres->vertex_position_buffers[i]);
            for (size_t i = 0; i < linear_swept_spheres->vertex_radius_buffers.size(); ++i)
                rhi_build_input.linearSweptSpheres.vertexRadiusBuffers[i]
                    = detail::to_rhi(linear_swept_spheres->vertex_radius_buffers[i]);
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

Micromap::Micromap(ref<Device> device, MicromapDesc desc)
    : Resource(std::move(device))
    , m_desc(std::move(desc))
{
    rhi::MicromapDesc rhi_desc{
        .type = static_cast<rhi::MicromapType>(m_desc.type),
        .size = m_desc.size,
        .flags = static_cast<rhi::MicromapBuildFlags>(m_desc.flags),
        .label = m_desc.label.c_str(),
    };
    SLANG_RHI_CALL(m_device->rhi_device()->createMicromap(rhi_desc, m_rhi_micromap.writeRef()), m_device);
}

Micromap::~Micromap() { }

std::string Micromap::to_string() const
{
    return fmt::format(
        "Micromap(\n"
        "  device = {},\n"
        "  size = {},\n"
        "  label = {}\n"
        ")",
        m_device,
        m_desc.size,
        m_desc.label
    );
}

AccelerationStructure::AccelerationStructure(ref<Device> device, AccelerationStructureDesc desc)
    : DeviceChild(std::move(device))
    , m_desc(std::move(desc))
{
    rhi::AccelerationStructureDesc rhi_desc{
        .kind = static_cast<rhi::AccelerationStructureKind>(desc.kind),
        .size = m_desc.size,
        .label = m_desc.label.c_str(),
    };
    SLANG_RHI_CALL(
        m_device->rhi_device()->createAccelerationStructure(rhi_desc, m_rhi_acceleration_structure.writeRef()),
        m_device
    );
}

AccelerationStructure::~AccelerationStructure() { }

AccelerationStructureHandle AccelerationStructure::handle() const
{
    return m_rhi_acceleration_structure->getHandle();
}

void AccelerationStructure::set_micromap_dependencies(const AccelerationStructureBuildDesc& desc)
{
    m_micromap_dependencies.clear();
    for (const auto& input : desc.inputs) {
        if (const auto* triangles = std::get_if<AccelerationStructureBuildInputTriangles>(&input)) {
            if (triangles->opacity_micromap && triangles->opacity_micromap->micromap)
                m_micromap_dependencies.push_back(triangles->opacity_micromap->micromap);
        }
    }
}

void AccelerationStructure::copy_micromap_dependencies(const AccelerationStructure& src)
{
    m_micromap_dependencies = src.m_micromap_dependencies;
}

void AccelerationStructure::write_to_cursor(const ShaderCursor& cursor, const AccelerationStructure* value)
{
    cursor.set_acceleration_structure(ref<const AccelerationStructure>(value));
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

    SLANG_RHI_CALL(m_device->rhi_device()->createShaderTable(rhi_desc, m_rhi_shader_table.writeRef()), m_device);
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
