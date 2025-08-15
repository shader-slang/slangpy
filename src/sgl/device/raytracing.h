// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/types.h"
#include "sgl/device/device_child.h"
#include "sgl/device/resource.h"

#include "sgl/math/vector_types.h"
#include "sgl/math/matrix_types.h"

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/core/enum.h"
#include "sgl/core/static_vector.h"
#include "sgl/core/short_vector.h"

#include <slang-rhi.h>

#include <variant>

namespace sgl {

using AccelerationStructureHandle = rhi::AccelerationStructureHandle;

enum class AccelerationStructureGeometryFlags : uint32_t {
    none = static_cast<uint32_t>(rhi::AccelerationStructureGeometryFlags::None),
    opaque = static_cast<uint32_t>(rhi::AccelerationStructureGeometryFlags::Opaque),
    no_duplicate_any_hit_invocation
    = static_cast<uint32_t>(rhi::AccelerationStructureGeometryFlags::NoDuplicateAnyHitInvocation),
};

SGL_ENUM_CLASS_OPERATORS(AccelerationStructureGeometryFlags);
SGL_ENUM_INFO(
    AccelerationStructureGeometryFlags,
    {
        {AccelerationStructureGeometryFlags::none, "none"},
        {AccelerationStructureGeometryFlags::opaque, "opaque"},
        {AccelerationStructureGeometryFlags::no_duplicate_any_hit_invocation, "no_duplicate_any_hit_invocation"},
    }
);
SGL_ENUM_REGISTER(AccelerationStructureGeometryFlags);

enum class AccelerationStructureInstanceFlags : uint32_t {
    none = static_cast<uint32_t>(rhi::AccelerationStructureInstanceFlags::None),
    triangle_facing_cull_disable
    = static_cast<uint32_t>(rhi::AccelerationStructureInstanceFlags::TriangleFacingCullDisable),
    triangle_front_counter_clockwise
    = static_cast<uint32_t>(rhi::AccelerationStructureInstanceFlags::TriangleFrontCounterClockwise),
    force_opaque = static_cast<uint32_t>(rhi::AccelerationStructureInstanceFlags::ForceOpaque),
    no_opaque = static_cast<uint32_t>(rhi::AccelerationStructureInstanceFlags::NoOpaque),
};

SGL_ENUM_CLASS_OPERATORS(AccelerationStructureInstanceFlags);
SGL_ENUM_INFO(
    AccelerationStructureInstanceFlags,
    {
        {AccelerationStructureInstanceFlags::none, "none"},
        {AccelerationStructureInstanceFlags::triangle_facing_cull_disable, "triangle_facing_cull_disable"},
        {AccelerationStructureInstanceFlags::triangle_front_counter_clockwise, "triangle_front_counter_clockwise"},
        {AccelerationStructureInstanceFlags::force_opaque, "force_opaque"},
        {AccelerationStructureInstanceFlags::no_opaque, "no_opaque"},
    }
);
SGL_ENUM_REGISTER(AccelerationStructureInstanceFlags);

struct AccelerationStructureInstanceDesc {
    float3x4 transform;
    uint32_t instance_id : 24;
    uint32_t instance_mask : 8;
    uint32_t instance_contribution_to_hit_group_index : 24;
    AccelerationStructureInstanceFlags flags : 8;
    AccelerationStructureHandle acceleration_structure;
};
static_assert(sizeof(AccelerationStructureInstanceDesc) == sizeof(rhi::AccelerationStructureInstanceDescGeneric));

struct AccelerationStructureBuildInputInstances {
    BufferOffsetPair instance_buffer;
    uint32_t instance_stride{0};
    uint32_t instance_count{0};
};

static constexpr size_t MAX_ACCELERATION_STRUCTURE_MOTION_KEY_COUNT = 2;

struct AccelerationStructureBuildInputTriangles {
    /// List of vertex buffers, one for each motion step.
    static_vector<BufferOffsetPair, MAX_ACCELERATION_STRUCTURE_MOTION_KEY_COUNT> vertex_buffers;
    Format vertex_format{Format::undefined};
    uint32_t vertex_count{0};
    uint32_t vertex_stride{0};

    BufferOffsetPair index_buffer;
    IndexFormat index_format{IndexFormat::uint32};
    uint32_t index_count{0};

    /// Optional buffer containing 3x4 transform matrix applied to each vertex.
    BufferOffsetPair pre_transform_buffer;

    AccelerationStructureGeometryFlags flags{AccelerationStructureGeometryFlags::none};
};

struct AccelerationStructureBuildInputProceduralPrimitives {
    /// List of AABB buffers, one for each motion step.
    static_vector<BufferOffsetPair, MAX_ACCELERATION_STRUCTURE_MOTION_KEY_COUNT> aabb_buffers;
    uint32_t aabb_stride{0};
    uint32_t primitive_count{0};

    AccelerationStructureGeometryFlags flags{AccelerationStructureGeometryFlags::none};
};

using AccelerationStructureBuildInput = std::variant<
    AccelerationStructureBuildInputInstances,
    AccelerationStructureBuildInputTriangles,
    AccelerationStructureBuildInputProceduralPrimitives>;

struct AccelerationStructureBuildInputMotionOptions {
    uint32_t key_count{1};
    float time_start{0.f};
    float time_end{1.f};
};

enum class AccelerationStructureBuildMode : uint32_t {
    build = static_cast<uint32_t>(rhi::AccelerationStructureBuildMode::Build),
    update = static_cast<uint32_t>(rhi::AccelerationStructureBuildMode::Update),
};

SGL_ENUM_INFO(
    AccelerationStructureBuildMode,
    {
        {AccelerationStructureBuildMode::build, "build"},
        {AccelerationStructureBuildMode::update, "update"},
    }
);
SGL_ENUM_REGISTER(AccelerationStructureBuildMode);

enum class AccelerationStructureBuildFlags : uint32_t {
    none = static_cast<uint32_t>(rhi::AccelerationStructureBuildFlags::None),
    allow_update = static_cast<uint32_t>(rhi::AccelerationStructureBuildFlags::AllowUpdate),
    allow_compaction = static_cast<uint32_t>(rhi::AccelerationStructureBuildFlags::AllowCompaction),
    prefer_fast_trace = static_cast<uint32_t>(rhi::AccelerationStructureBuildFlags::PreferFastTrace),
    prefer_fast_build = static_cast<uint32_t>(rhi::AccelerationStructureBuildFlags::PreferFastBuild),
    minimize_memory = static_cast<uint32_t>(rhi::AccelerationStructureBuildFlags::MinimizeMemory),
};

SGL_ENUM_CLASS_OPERATORS(AccelerationStructureBuildFlags);
SGL_ENUM_INFO(
    AccelerationStructureBuildFlags,
    {
        {AccelerationStructureBuildFlags::none, "none"},
        {AccelerationStructureBuildFlags::allow_update, "allow_update"},
        {AccelerationStructureBuildFlags::allow_compaction, "allow_compaction"},
        {AccelerationStructureBuildFlags::prefer_fast_trace, "prefer_fast_trace"},
        {AccelerationStructureBuildFlags::prefer_fast_build, "prefer_fast_build"},
        {AccelerationStructureBuildFlags::minimize_memory, "minimize_memory"},
    }
);
SGL_ENUM_REGISTER(AccelerationStructureBuildFlags);

struct AccelerationStructureBuildDesc {
    /// List of build inputs. All inputs must be of the same type.
    std::vector<AccelerationStructureBuildInput> inputs;

    AccelerationStructureBuildInputMotionOptions motion_options;

    AccelerationStructureBuildMode mode{AccelerationStructureBuildMode::build};
    AccelerationStructureBuildFlags flags{AccelerationStructureBuildFlags::none};
};

struct AccelerationStructureBuildDescConverter {
    rhi::AccelerationStructureBuildDesc rhi_desc;
    // TODO(slang-rhi) probably use short_vector instead, but short_vector needs some more work
    std::vector<rhi::AccelerationStructureBuildInput> rhi_build_inputs;
    AccelerationStructureBuildDescConverter(const AccelerationStructureBuildDesc& desc);
};

enum class AccelerationStructureCopyMode : uint32_t {
    clone = static_cast<uint32_t>(rhi::AccelerationStructureCopyMode::Clone),
    compact = static_cast<uint32_t>(rhi::AccelerationStructureCopyMode::Compact),
};

SGL_ENUM_INFO(
    AccelerationStructureCopyMode,
    {
        {AccelerationStructureCopyMode::clone, "clone"},
        {AccelerationStructureCopyMode::compact, "compact"},
    }
);
SGL_ENUM_REGISTER(AccelerationStructureCopyMode);

struct AccelerationStructureSizes {
    DeviceSize acceleration_structure_size{0};
    DeviceSize scratch_size{0};
    DeviceSize update_scratch_size{0};
};

struct AccelerationStructureQueryDesc {
    QueryType query_type;
    QueryPool* query_pool;
    uint32_t first_query_index;
};

struct AccelerationStructureDesc {
    DeviceSize size{0};
    std::string label;
};

class SGL_API AccelerationStructure : public DeviceChild {
    SGL_OBJECT(AccelerationStructure)
public:
    AccelerationStructure(ref<Device> device, AccelerationStructureDesc desc);
    ~AccelerationStructure();

    virtual void _release_rhi_resources() override { m_rhi_acceleration_structure.setNull(); }

    const AccelerationStructureDesc& desc() const { return m_desc; }

    AccelerationStructureHandle handle() const;

    rhi::IAccelerationStructure* rhi_acceleration_structure() const { return m_rhi_acceleration_structure; }

    std::string to_string() const override;

private:
    AccelerationStructureDesc m_desc;
    Slang::ComPtr<rhi::IAccelerationStructure> m_rhi_acceleration_structure;
};

class SGL_API AccelerationStructureInstanceList : public DeviceChild {
    SGL_OBJECT(AccelerationStructureInstanceList)
public:
    AccelerationStructureInstanceList(ref<Device> device, size_t size = 0);
    ~AccelerationStructureInstanceList();

    virtual void _release_rhi_resources() override { }

    size_t size() const { return m_instances.size(); }

    size_t instance_stride() const { return m_instance_stride; }

    void resize(size_t size);

    void write(size_t index, const AccelerationStructureInstanceDesc& instance);
    void write(size_t index, std::span<AccelerationStructureInstanceDesc> instances);

    ref<Buffer> buffer() const;

    AccelerationStructureBuildInputInstances build_input_instances() const;

    std::string to_string() const override;

private:
    std::vector<AccelerationStructureInstanceDesc> m_instances;
    rhi::AccelerationStructureInstanceDescType m_instance_type;
    size_t m_instance_stride;
    mutable bool m_dirty{true};
    mutable ref<Buffer> m_buffer;
};

struct ShaderTableDesc {
    ref<ShaderProgram> program;
    std::vector<std::string> ray_gen_entry_points;
    std::vector<std::string> miss_entry_points;
    std::vector<std::string> hit_group_names;
    std::vector<std::string> callable_entry_points;
};

class SGL_API ShaderTable : public DeviceChild {
    SGL_OBJECT(ShaderTable)
public:
    ShaderTable(ref<Device> device, ShaderTableDesc desc);
    ~ShaderTable();

    virtual void _release_rhi_resources() override { m_rhi_shader_table.setNull(); }

    rhi::IShaderTable* rhi_shader_table() const { return m_rhi_shader_table; }

    std::string to_string() const override;

private:
    Slang::ComPtr<rhi::IShaderTable> m_rhi_shader_table;
};

} // namespace sgl
