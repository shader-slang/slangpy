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

#include <optional>
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
SGL_ENUM_FLAGS_INFO(
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
    force_opacity_micromap_2_state
    = static_cast<uint32_t>(rhi::AccelerationStructureInstanceFlags::ForceOpacityMicromap2State),
    disable_opacity_micromaps = static_cast<uint32_t>(rhi::AccelerationStructureInstanceFlags::DisableOpacityMicromaps),
};

SGL_ENUM_CLASS_OPERATORS(AccelerationStructureInstanceFlags);
SGL_ENUM_FLAGS_INFO(
    AccelerationStructureInstanceFlags,
    {
        {AccelerationStructureInstanceFlags::none, "none"},
        {AccelerationStructureInstanceFlags::triangle_facing_cull_disable, "triangle_facing_cull_disable"},
        {AccelerationStructureInstanceFlags::triangle_front_counter_clockwise, "triangle_front_counter_clockwise"},
        {AccelerationStructureInstanceFlags::force_opaque, "force_opaque"},
        {AccelerationStructureInstanceFlags::no_opaque, "no_opaque"},
        {AccelerationStructureInstanceFlags::force_opacity_micromap_2_state, "force_opacity_micromap_2_state"},
        {AccelerationStructureInstanceFlags::disable_opacity_micromaps, "disable_opacity_micromaps"},
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

enum class MicromapType : uint32_t {
    opacity = static_cast<uint32_t>(rhi::MicromapType::Opacity),
};
SGL_ENUM_INFO(MicromapType, {{MicromapType::opacity, "opacity"}});
SGL_ENUM_REGISTER(MicromapType);

enum class OpacityMicromapFormat : uint16_t {
    two_state = static_cast<uint16_t>(rhi::OpacityMicromapFormat::TwoState),
    four_state = static_cast<uint16_t>(rhi::OpacityMicromapFormat::FourState),
};
SGL_ENUM_INFO(
    OpacityMicromapFormat,
    {
        {OpacityMicromapFormat::two_state, "two_state"},
        {OpacityMicromapFormat::four_state, "four_state"},
    }
);
SGL_ENUM_REGISTER(OpacityMicromapFormat);

enum class OpacityMicromapSpecialIndex : int32_t {
    fully_transparent = static_cast<int32_t>(rhi::OpacityMicromapSpecialIndex::FullyTransparent),
    fully_opaque = static_cast<int32_t>(rhi::OpacityMicromapSpecialIndex::FullyOpaque),
    fully_unknown_transparent = static_cast<int32_t>(rhi::OpacityMicromapSpecialIndex::FullyUnknownTransparent),
    fully_unknown_opaque = static_cast<int32_t>(rhi::OpacityMicromapSpecialIndex::FullyUnknownOpaque),
};
SGL_ENUM_INFO(
    OpacityMicromapSpecialIndex,
    {
        {OpacityMicromapSpecialIndex::fully_transparent, "fully_transparent"},
        {OpacityMicromapSpecialIndex::fully_opaque, "fully_opaque"},
        {OpacityMicromapSpecialIndex::fully_unknown_transparent, "fully_unknown_transparent"},
        {OpacityMicromapSpecialIndex::fully_unknown_opaque, "fully_unknown_opaque"},
    }
);
SGL_ENUM_REGISTER(OpacityMicromapSpecialIndex);

struct MicromapUsageCount {
    uint32_t count{0};
    uint32_t subdivision_level{0};
    OpacityMicromapFormat format{OpacityMicromapFormat::two_state};
};

enum class MicromapIndexingMode : uint32_t {
    linear = static_cast<uint32_t>(rhi::MicromapIndexingMode::Linear),
    indexed = static_cast<uint32_t>(rhi::MicromapIndexingMode::Indexed),
};
SGL_ENUM_INFO(
    MicromapIndexingMode,
    {
        {MicromapIndexingMode::linear, "linear"},
        {MicromapIndexingMode::indexed, "indexed"},
    }
);
SGL_ENUM_REGISTER(MicromapIndexingMode);

enum class MicromapIndexFormat : uint32_t {
    none = static_cast<uint32_t>(rhi::MicromapIndexFormat::None),
    uint16 = static_cast<uint32_t>(rhi::MicromapIndexFormat::Uint16),
    uint32 = static_cast<uint32_t>(rhi::MicromapIndexFormat::Uint32),
};
SGL_ENUM_INFO(
    MicromapIndexFormat,
    {
        {MicromapIndexFormat::none, "none"},
        {MicromapIndexFormat::uint16, "uint16"},
        {MicromapIndexFormat::uint32, "uint32"},
    }
);
SGL_ENUM_REGISTER(MicromapIndexFormat);

enum class MicromapBuildFlags : uint32_t {
    none = static_cast<uint32_t>(rhi::MicromapBuildFlags::None),
    prefer_fast_trace = static_cast<uint32_t>(rhi::MicromapBuildFlags::PreferFastTrace),
    prefer_fast_build = static_cast<uint32_t>(rhi::MicromapBuildFlags::PreferFastBuild),
    allow_compaction = static_cast<uint32_t>(rhi::MicromapBuildFlags::AllowCompaction),
};
SGL_ENUM_CLASS_OPERATORS(MicromapBuildFlags);
SGL_ENUM_FLAGS_INFO(
    MicromapBuildFlags,
    {
        {MicromapBuildFlags::none, "none"},
        {MicromapBuildFlags::prefer_fast_trace, "prefer_fast_trace"},
        {MicromapBuildFlags::prefer_fast_build, "prefer_fast_build"},
        {MicromapBuildFlags::allow_compaction, "allow_compaction"},
    }
);
SGL_ENUM_REGISTER(MicromapBuildFlags);

struct MicromapBuildDesc {
    MicromapType type{MicromapType::opacity};
    MicromapBuildFlags flags{MicromapBuildFlags::none};
    BufferOffsetPair data_buffer;
    BufferOffsetPair descriptor_buffer;
    uint32_t descriptor_stride{sizeof(rhi::MicromapTriangleDesc)};
    std::vector<MicromapUsageCount> histogram;
};

struct MicromapBuildDescConverter {
    rhi::MicromapBuildDesc rhi_desc;
    std::vector<rhi::MicromapUsageCount> rhi_histogram;
    MicromapBuildDescConverter(const MicromapBuildDesc& desc);
};

struct MicromapSizes {
    DeviceSize micromap_size{0};
    DeviceSize scratch_size{0};
};

struct MicromapDesc {
    MicromapType type{MicromapType::opacity};
    DeviceSize size{0};
    MicromapBuildFlags flags{MicromapBuildFlags::none};
    std::string label;
};

class SGL_API Micromap : public Resource {
    SGL_OBJECT(Micromap)
public:
    Micromap(ref<Device> device, MicromapDesc desc);
    ~Micromap();

    virtual void _release_rhi_resources() override { m_rhi_micromap.setNull(); }

    const MicromapDesc& desc() const { return m_desc; }
    DeviceAddress device_address() const { return m_rhi_micromap->getDeviceAddress(); }

    rhi::IMicromap* rhi_micromap() const { return m_rhi_micromap; }
    virtual rhi::IResource* rhi_resource() const override { return m_rhi_micromap; }

    std::string to_string() const override;

private:
    MicromapDesc m_desc;
    Slang::ComPtr<rhi::IMicromap> m_rhi_micromap;
};

struct AccelerationStructureOpacityMicromapDesc {
    ref<Micromap> micromap;
    MicromapIndexingMode indexing_mode{MicromapIndexingMode::linear};
    BufferOffsetPair index_buffer;
    MicromapIndexFormat index_format{MicromapIndexFormat::none};
    uint32_t index_stride{0};
    uint32_t base_micromap_index{0};
    std::vector<MicromapUsageCount> usage_counts;
};

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

    /// Optional opacity micromap attachment.
    std::optional<AccelerationStructureOpacityMicromapDesc> opacity_micromap;
};

struct AccelerationStructureBuildInputProceduralPrimitives {
    /// List of AABB buffers, one for each motion step.
    static_vector<BufferOffsetPair, MAX_ACCELERATION_STRUCTURE_MOTION_KEY_COUNT> aabb_buffers;
    uint32_t aabb_stride{0};
    uint32_t primitive_count{0};

    AccelerationStructureGeometryFlags flags{AccelerationStructureGeometryFlags::none};
};

struct AccelerationStructureBuildInputSpheres {
    uint32_t vertex_count{0};

    static_vector<BufferOffsetPair, MAX_ACCELERATION_STRUCTURE_MOTION_KEY_COUNT> vertex_position_buffers;
    Format vertex_position_format{Format::undefined};
    uint32_t vertex_position_stride{0};

    static_vector<BufferOffsetPair, MAX_ACCELERATION_STRUCTURE_MOTION_KEY_COUNT> vertex_radius_buffers;
    Format vertex_radius_format{Format::undefined};
    uint32_t vertex_radius_stride{0};

    BufferOffsetPair index_buffer;
    IndexFormat index_format{IndexFormat::uint32};
    uint32_t index_count{0};

    AccelerationStructureGeometryFlags flags{AccelerationStructureGeometryFlags::none};
};

enum class LinearSweptSpheresIndexingMode {
    list = static_cast<uint32_t>(rhi::LinearSweptSpheresIndexingMode::List),
    successive = static_cast<uint32_t>(rhi::LinearSweptSpheresIndexingMode::Successive),
};
SGL_ENUM_INFO(
    LinearSweptSpheresIndexingMode,
    {
        {LinearSweptSpheresIndexingMode::list, "list"},
        {LinearSweptSpheresIndexingMode::successive, "successive"},
    }
);
SGL_ENUM_REGISTER(LinearSweptSpheresIndexingMode);

enum class LinearSweptSpheresEndCapsMode {
    none = static_cast<uint32_t>(rhi::LinearSweptSpheresEndCapsMode::None),
    chained = static_cast<uint32_t>(rhi::LinearSweptSpheresEndCapsMode::Chained),
};
SGL_ENUM_INFO(
    LinearSweptSpheresEndCapsMode,
    {
        {LinearSweptSpheresEndCapsMode::none, "none"},
        {LinearSweptSpheresEndCapsMode::chained, "chained"},
    }
);
SGL_ENUM_REGISTER(LinearSweptSpheresEndCapsMode);

struct AccelerationStructureBuildInputLinearSweptSpheres {
    uint32_t vertex_count{0};
    uint32_t primitive_count{0};

    static_vector<BufferOffsetPair, MAX_ACCELERATION_STRUCTURE_MOTION_KEY_COUNT> vertex_position_buffers;
    Format vertex_position_format{Format::undefined};
    uint32_t vertex_position_stride{0};

    static_vector<BufferOffsetPair, MAX_ACCELERATION_STRUCTURE_MOTION_KEY_COUNT> vertex_radius_buffers;
    Format vertex_radius_format{Format::undefined};
    uint32_t vertex_radius_stride{0};

    BufferOffsetPair index_buffer;
    IndexFormat index_format{IndexFormat::uint32};
    uint32_t index_count{0};

    LinearSweptSpheresIndexingMode indexing_mode{LinearSweptSpheresIndexingMode::list};
    LinearSweptSpheresEndCapsMode end_caps_mode{LinearSweptSpheresEndCapsMode::none};

    AccelerationStructureGeometryFlags flags{AccelerationStructureGeometryFlags::none};
};

using AccelerationStructureBuildInput = std::variant<
    AccelerationStructureBuildInputInstances,
    AccelerationStructureBuildInputTriangles,
    AccelerationStructureBuildInputProceduralPrimitives,
    AccelerationStructureBuildInputSpheres,
    AccelerationStructureBuildInputLinearSweptSpheres>;

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
    allow_opacity_micromap_update
    = static_cast<uint32_t>(rhi::AccelerationStructureBuildFlags::AllowOpacityMicromapUpdate),
    allow_disable_opacity_micromaps
    = static_cast<uint32_t>(rhi::AccelerationStructureBuildFlags::AllowDisableOpacityMicromaps),
};

SGL_ENUM_CLASS_OPERATORS(AccelerationStructureBuildFlags);
SGL_ENUM_FLAGS_INFO(
    AccelerationStructureBuildFlags,
    {
        {AccelerationStructureBuildFlags::none, "none"},
        {AccelerationStructureBuildFlags::allow_update, "allow_update"},
        {AccelerationStructureBuildFlags::allow_compaction, "allow_compaction"},
        {AccelerationStructureBuildFlags::prefer_fast_trace, "prefer_fast_trace"},
        {AccelerationStructureBuildFlags::prefer_fast_build, "prefer_fast_build"},
        {AccelerationStructureBuildFlags::minimize_memory, "minimize_memory"},
        {AccelerationStructureBuildFlags::allow_opacity_micromap_update, "allow_opacity_micromap_update"},
        {AccelerationStructureBuildFlags::allow_disable_opacity_micromaps, "allow_disable_opacity_micromaps"},
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
    std::vector<rhi::AccelerationStructureOpacityMicromapDesc> rhi_opacity_micromap_descs;
    std::vector<std::vector<rhi::MicromapUsageCount>> rhi_opacity_micromap_usage_counts;
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

enum class AccelerationStructureKind : uint32_t {
    unknown = static_cast<uint32_t>(rhi::AccelerationStructureKind::Unknown),
    bottom_level = static_cast<uint32_t>(rhi::AccelerationStructureKind::BottomLevel),
    top_level = static_cast<uint32_t>(rhi::AccelerationStructureKind::TopLevel),
};

SGL_ENUM_INFO(
    AccelerationStructureKind,
    {
        {AccelerationStructureKind::unknown, "unknown"},
        {AccelerationStructureKind::bottom_level, "bottom_level"},
        {AccelerationStructureKind::top_level, "top_level"},
    }
);
SGL_ENUM_REGISTER(AccelerationStructureKind);

struct AccelerationStructureDesc {
    AccelerationStructureKind kind{AccelerationStructureKind::unknown};
    DeviceSize size{0};
    std::string label;
};

class SGL_API AccelerationStructure : public DeviceChild {
    SGL_OBJECT(AccelerationStructure)
public:
    AccelerationStructure(ref<Device> device, AccelerationStructureDesc desc);
    ~AccelerationStructure();

    virtual void _release_rhi_resources() override
    {
        m_rhi_acceleration_structure.setNull();
        m_micromap_dependencies.clear();
    }

    const AccelerationStructureDesc& desc() const { return m_desc; }

    AccelerationStructureHandle handle() const;

    rhi::IAccelerationStructure* rhi_acceleration_structure() const { return m_rhi_acceleration_structure; }

    void set_micromap_dependencies(const AccelerationStructureBuildDesc& desc);
    void copy_micromap_dependencies(const AccelerationStructure& src);

    /// Bind a nullable acceleration structure value to a shader cursor.
    static void write_to_cursor(const ShaderCursor& cursor, const AccelerationStructure* value);

    std::string to_string() const override;

private:
    AccelerationStructureDesc m_desc;
    Slang::ComPtr<rhi::IAccelerationStructure> m_rhi_acceleration_structure;
    std::vector<ref<Micromap>> m_micromap_dependencies;
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

// ----------------------------------------------------------------------------
// Cluster acceleration structures
// ----------------------------------------------------------------------------

enum class ClusterOperationType : uint32_t {
    move_objects = static_cast<uint32_t>(rhi::ClusterOperationType::MoveObjects),
    clas_from_triangles = static_cast<uint32_t>(rhi::ClusterOperationType::CLASFromTriangles),
    blas_from_clas = static_cast<uint32_t>(rhi::ClusterOperationType::BLASFromCLAS),
    templates_from_triangles = static_cast<uint32_t>(rhi::ClusterOperationType::TemplatesFromTriangles),
    clas_from_templates = static_cast<uint32_t>(rhi::ClusterOperationType::CLASFromTemplates),
};
SGL_ENUM_INFO(
    ClusterOperationType,
    {
        {ClusterOperationType::move_objects, "move_objects"},
        {ClusterOperationType::clas_from_triangles, "clas_from_triangles"},
        {ClusterOperationType::blas_from_clas, "blas_from_clas"},
        {ClusterOperationType::templates_from_triangles, "templates_from_triangles"},
        {ClusterOperationType::clas_from_templates, "clas_from_templates"},
    }
);
SGL_ENUM_REGISTER(ClusterOperationType);

enum class ClusterOperationMode : uint32_t {
    implicit_destinations = static_cast<uint32_t>(rhi::ClusterOperationMode::ImplicitDestinations),
    explicit_destinations = static_cast<uint32_t>(rhi::ClusterOperationMode::ExplicitDestinations),
    get_sizes = static_cast<uint32_t>(rhi::ClusterOperationMode::GetSizes),
};
SGL_ENUM_INFO(
    ClusterOperationMode,
    {
        {ClusterOperationMode::implicit_destinations, "implicit_destinations"},
        {ClusterOperationMode::explicit_destinations, "explicit_destinations"},
        {ClusterOperationMode::get_sizes, "get_sizes"},
    }
);
SGL_ENUM_REGISTER(ClusterOperationMode);

enum class ClusterOperationFlags : uint32_t {
    none = static_cast<uint32_t>(rhi::ClusterOperationFlags::None),
    fast_trace = static_cast<uint32_t>(rhi::ClusterOperationFlags::FastTrace),
    fast_build = static_cast<uint32_t>(rhi::ClusterOperationFlags::FastBuild),
    no_overlap = static_cast<uint32_t>(rhi::ClusterOperationFlags::NoOverlap),
    allow_omm = static_cast<uint32_t>(rhi::ClusterOperationFlags::AllowOMM),
};
SGL_ENUM_CLASS_OPERATORS(ClusterOperationFlags);
SGL_ENUM_FLAGS_INFO(
    ClusterOperationFlags,
    {
        {ClusterOperationFlags::none, "none"},
        {ClusterOperationFlags::fast_trace, "fast_trace"},
        {ClusterOperationFlags::fast_build, "fast_build"},
        {ClusterOperationFlags::no_overlap, "no_overlap"},
        {ClusterOperationFlags::allow_omm, "allow_omm"},
    }
);
SGL_ENUM_REGISTER(ClusterOperationFlags);

enum class ClusterOperationMoveType : uint32_t {
    bottom_level = static_cast<uint32_t>(rhi::ClusterOperationMoveType::BottomLevel),
    cluster_level = static_cast<uint32_t>(rhi::ClusterOperationMoveType::ClusterLevel),
    template_ = static_cast<uint32_t>(rhi::ClusterOperationMoveType::Template),
};
SGL_ENUM_INFO(
    ClusterOperationMoveType,
    {
        {ClusterOperationMoveType::bottom_level, "bottom_level"},
        {ClusterOperationMoveType::cluster_level, "cluster_level"},
        {ClusterOperationMoveType::template_, "template"},
    }
);
SGL_ENUM_REGISTER(ClusterOperationMoveType);

struct ClusterOperationMoveParams {
    ClusterOperationMoveType type{ClusterOperationMoveType::bottom_level};
    uint32_t max_size{0};
};

struct ClusterOperationClasBuildParams {
    Format vertex_format{Format::rgb32_float};
    uint32_t max_geometry_index{0};
    uint32_t max_unique_geometry_count{1};
    uint32_t max_triangle_count{0};
    uint32_t max_vertex_count{0};
    uint32_t max_total_triangle_count{0};
    uint32_t max_total_vertex_count{0};
    uint32_t min_position_truncate_bit_count{0};
};

struct ClusterOperationBlasBuildParams {
    uint32_t max_clas_count{0};
    uint32_t max_total_clas_count{0};
};

struct ClusterOperationParams {
    uint32_t max_arg_count{0};
    ClusterOperationType type{ClusterOperationType::clas_from_triangles};
    ClusterOperationMode mode{ClusterOperationMode::implicit_destinations};
    ClusterOperationFlags flags{ClusterOperationFlags::none};
    ClusterOperationMoveParams move;
    ClusterOperationClasBuildParams clas;
    ClusterOperationBlasBuildParams blas;
};

namespace detail {
    SGL_API rhi::ClusterOperationParams to_rhi(const ClusterOperationParams& params);
}

struct ClusterOperationDesc {
    ClusterOperationParams params;
    BufferOffsetPair arg_count_buffer;
    BufferOffsetPair args_buffer;
    uint64_t args_buffer_stride{0};
    BufferOffsetPair scratch_buffer;
    BufferOffsetPair addresses_buffer;
    size_t addresses_buffer_stride{rhi::kClusterDefaultHandleStride};
    BufferOffsetPair result_buffer;
    BufferOffsetPair sizes_buffer;
    size_t sizes_buffer_stride{sizeof(uint32_t)};
};

struct ClusterOperationSizes {
    DeviceSize result_size{0};
    DeviceSize scratch_size{0};
};

static constexpr uint32_t CLUSTER_MAX_TRIANGLE_COUNT = rhi::kClusterMaxTriangleCount;
static constexpr uint32_t CLUSTER_MAX_VERTEX_COUNT = rhi::kClusterMaxVertexCount;
static constexpr uint32_t CLUSTER_MAX_GEOMETRY_INDEX = rhi::kClusterMaxGeometryIndex;
static constexpr uint32_t CLUSTER_DEFAULT_HANDLE_STRIDE = rhi::kClusterDefaultHandleStride;
static constexpr uint32_t CLUSTER_OUTPUT_ALIGNMENT = rhi::kClusterOutputAlignment;

// ----------------------------------------------------------------------------
// ShaderTable
// ----------------------------------------------------------------------------

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
