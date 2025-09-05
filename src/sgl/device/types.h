// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/formats.h"

#include "sgl/core/macros.h"
#include "sgl/core/enum.h"

#include "sgl/math/vector_types.h"
#include "sgl/math/matrix_types.h"

#include <slang-rhi.h>

namespace sgl {

/// Represents an address in device memory.
using DeviceAddress = uint64_t;
/// Represents an offset in device memory (in bytes).
using DeviceOffset = uint64_t;
/// Represents a size in device memory (in bytes).
using DeviceSize = uint64_t;
/// Adapter LUID (locally unique identifier).
using AdapterLUID = std::array<uint8_t, 16>;

enum CommandQueueType : uint32_t {
    graphics = static_cast<uint32_t>(rhi::QueueType::Graphics),
};

SGL_ENUM_INFO(
    CommandQueueType,
    {
        {CommandQueueType::graphics, "graphics"},
    }
);
SGL_ENUM_REGISTER(CommandQueueType);

enum class Feature : uint32_t {
    hardware_device = static_cast<uint32_t>(rhi::Feature::HardwareDevice),
    software_device = static_cast<uint32_t>(rhi::Feature::SoftwareDevice),
    parameter_block = static_cast<uint32_t>(rhi::Feature::ParameterBlock),
    bindless = static_cast<uint32_t>(rhi::Feature::Bindless),
    surface = static_cast<uint32_t>(rhi::Feature::Surface),
    pipeline_cache = static_cast<uint32_t>(rhi::Feature::PipelineCache),
    // Rasterization features
    rasterization = static_cast<uint32_t>(rhi::Feature::Rasterization),
    barycentrics = static_cast<uint32_t>(rhi::Feature::Barycentrics),
    multi_view = static_cast<uint32_t>(rhi::Feature::MultiView),
    rasterizer_ordered_views = static_cast<uint32_t>(rhi::Feature::RasterizerOrderedViews),
    conservative_rasterization = static_cast<uint32_t>(rhi::Feature::ConservativeRasterization),
    custom_border_color = static_cast<uint32_t>(rhi::Feature::CustomBorderColor),
    fragment_shading_rate = static_cast<uint32_t>(rhi::Feature::FragmentShadingRate),
    sampler_feedback = static_cast<uint32_t>(rhi::Feature::SamplerFeedback),
    // Ray tracing features
    acceleration_structure = static_cast<uint32_t>(rhi::Feature::AccelerationStructure),
    acceleration_structure_spheres = static_cast<uint32_t>(rhi::Feature::AccelerationStructureSpheres),
    acceleration_structure_linear_swept_spheres
    = static_cast<uint32_t>(rhi::Feature::AccelerationStructureLinearSweptSpheres),
    ray_tracing = static_cast<uint32_t>(rhi::Feature::RayTracing),
    ray_query = static_cast<uint32_t>(rhi::Feature::RayQuery),
    shader_execution_reordering = static_cast<uint32_t>(rhi::Feature::ShaderExecutionReordering),
    ray_tracing_validation = static_cast<uint32_t>(rhi::Feature::RayTracingValidation),
    // Other features
    timestamp_query = static_cast<uint32_t>(rhi::Feature::TimestampQuery),
    realtime_clock = static_cast<uint32_t>(rhi::Feature::RealtimeClock),
    cooperative_vector = static_cast<uint32_t>(rhi::Feature::CooperativeVector),
    cooperative_matrix = static_cast<uint32_t>(rhi::Feature::CooperativeMatrix),
    sm_5_1 = static_cast<uint32_t>(rhi::Feature::SM_5_1),
    sm_6_0 = static_cast<uint32_t>(rhi::Feature::SM_6_0),
    sm_6_1 = static_cast<uint32_t>(rhi::Feature::SM_6_1),
    sm_6_2 = static_cast<uint32_t>(rhi::Feature::SM_6_2),
    sm_6_3 = static_cast<uint32_t>(rhi::Feature::SM_6_3),
    sm_6_4 = static_cast<uint32_t>(rhi::Feature::SM_6_4),
    sm_6_5 = static_cast<uint32_t>(rhi::Feature::SM_6_5),
    sm_6_6 = static_cast<uint32_t>(rhi::Feature::SM_6_6),
    sm_6_7 = static_cast<uint32_t>(rhi::Feature::SM_6_7),
    sm_6_8 = static_cast<uint32_t>(rhi::Feature::SM_6_8),
    sm_6_9 = static_cast<uint32_t>(rhi::Feature::SM_6_9),
    half = static_cast<uint32_t>(rhi::Feature::Half),
    double_ = static_cast<uint32_t>(rhi::Feature::Double),
    int16 = static_cast<uint32_t>(rhi::Feature::Int16),
    int64 = static_cast<uint32_t>(rhi::Feature::Int64),
    atomic_float = static_cast<uint32_t>(rhi::Feature::AtomicFloat),
    atomic_half = static_cast<uint32_t>(rhi::Feature::AtomicHalf),
    atomic_int64 = static_cast<uint32_t>(rhi::Feature::AtomicInt64),
    wave_ops = static_cast<uint32_t>(rhi::Feature::WaveOps),
    mesh_shader = static_cast<uint32_t>(rhi::Feature::MeshShader),
    pointer = static_cast<uint32_t>(rhi::Feature::Pointer),
    // D3D12 specific features
    conservative_rasterization1 = static_cast<uint32_t>(rhi::Feature::ConservativeRasterization1),
    conservative_rasterization2 = static_cast<uint32_t>(rhi::Feature::ConservativeRasterization2),
    conservative_rasterization3 = static_cast<uint32_t>(rhi::Feature::ConservativeRasterization3),
    programmable_sample_positions1 = static_cast<uint32_t>(rhi::Feature::ProgrammableSamplePositions1),
    programmable_sample_positions2 = static_cast<uint32_t>(rhi::Feature::ProgrammableSamplePositions2),
    // Vulkan specific features
    shader_resource_min_lod = static_cast<uint32_t>(rhi::Feature::ShaderResourceMinLod),
    // Metal specific features
    argument_buffer_tier2 = static_cast<uint32_t>(rhi::Feature::ArgumentBufferTier2),

    count,
};

static_assert(
    static_cast<uint32_t>(Feature::count) == static_cast<uint32_t>(rhi::Feature::_Count),
    "Feature table size mismatch"
);

SGL_ENUM_INFO(
    Feature,
    {
        {Feature::hardware_device, "hardware_device"},
        {Feature::software_device, "software_device"},
        {Feature::parameter_block, "parameter_block"},
        {Feature::bindless, "bindless"},
        {Feature::surface, "surface"},
        {Feature::pipeline_cache, "pipeline_cache"},
        {Feature::rasterization, "rasterization"},
        {Feature::barycentrics, "barycentrics"},
        {Feature::multi_view, "multi_view"},
        {Feature::rasterizer_ordered_views, "rasterizer_ordered_views"},
        {Feature::conservative_rasterization, "conservative_rasterization"},
        {Feature::custom_border_color, "custom_border_color"},
        {Feature::fragment_shading_rate, "fragment_shading_rate"},
        {Feature::sampler_feedback, "sampler_feedback"},
        {Feature::acceleration_structure, "acceleration_structure"},
        {Feature::acceleration_structure_spheres, "acceleration_structure_spheres"},
        {Feature::acceleration_structure_linear_swept_spheres, "acceleration_structure_linear_swept_spheres"},
        {Feature::ray_tracing, "ray_tracing"},
        {Feature::ray_query, "ray_query"},
        {Feature::shader_execution_reordering, "shader_execution_reordering"},
        {Feature::ray_tracing_validation, "ray_tracing_validation"},
        {Feature::timestamp_query, "timestamp_query"},
        {Feature::realtime_clock, "realtime_clock"},
        {Feature::cooperative_vector, "cooperative_vector"},
        {Feature::cooperative_matrix, "cooperative_matrix"},
        {Feature::sm_5_1, "sm_5_1"},
        {Feature::sm_6_0, "sm_6_0"},
        {Feature::sm_6_1, "sm_6_1"},
        {Feature::sm_6_2, "sm_6_2"},
        {Feature::sm_6_3, "sm_6_3"},
        {Feature::sm_6_4, "sm_6_4"},
        {Feature::sm_6_5, "sm_6_5"},
        {Feature::sm_6_6, "sm_6_6"},
        {Feature::sm_6_7, "sm_6_7"},
        {Feature::sm_6_8, "sm_6_8"},
        {Feature::sm_6_9, "sm_6_9"},
        {Feature::half, "half"},
        {Feature::double_, "double_"},
        {Feature::int16, "int16"},
        {Feature::int64, "int64"},
        {Feature::atomic_float, "atomic_float"},
        {Feature::atomic_half, "atomic_half"},
        {Feature::atomic_int64, "atomic_int64"},
        {Feature::wave_ops, "wave_ops"},
        {Feature::mesh_shader, "mesh_shader"},
        {Feature::pointer, "pointer"},
        {Feature::conservative_rasterization1, "conservative_rasterization1"},
        {Feature::conservative_rasterization2, "conservative_rasterization2"},
        {Feature::conservative_rasterization3, "conservative_rasterization3"},
        {Feature::programmable_sample_positions1, "programmable_sample_positions1"},
        {Feature::programmable_sample_positions2, "programmable_sample_positions2"},
        {Feature::shader_resource_min_lod, "shader_resource_min_lod"},
        {Feature::argument_buffer_tier2, "argument_buffer_tier2"},
    }
);
SGL_ENUM_REGISTER(Feature);
static_assert(EnumInfo<Feature>::items.size() == static_cast<size_t>(Feature::count));

enum class DescriptorHandleType {
    undefined = static_cast<uint32_t>(rhi::DescriptorHandleType::Undefined),
    buffer = static_cast<uint32_t>(rhi::DescriptorHandleType::Buffer),
    rw_buffer = static_cast<uint32_t>(rhi::DescriptorHandleType::RWBuffer),
    texture = static_cast<uint32_t>(rhi::DescriptorHandleType::Texture),
    rw_texture = static_cast<uint32_t>(rhi::DescriptorHandleType::RWTexture),
    sampler = static_cast<uint32_t>(rhi::DescriptorHandleType::Sampler),
    acceleration_structure = static_cast<uint32_t>(rhi::DescriptorHandleType::AccelerationStructure),
};

SGL_ENUM_INFO(
    DescriptorHandleType,
    {
        {DescriptorHandleType::undefined, "undefined"},
        {DescriptorHandleType::buffer, "buffer"},
        {DescriptorHandleType::rw_buffer, "rw_buffer"},
        {DescriptorHandleType::texture, "texture"},
        {DescriptorHandleType::rw_texture, "rw_texture"},
        {DescriptorHandleType::sampler, "sampler"},
        {DescriptorHandleType::acceleration_structure, "acceleration_structure"},
    }
);
SGL_ENUM_REGISTER(DescriptorHandleType);

struct SGL_API DescriptorHandle {
    DescriptorHandleType type{DescriptorHandleType::undefined};
    uint64_t value{0};

    explicit DescriptorHandle(const rhi::DescriptorHandle& handle)
    {
        type = static_cast<DescriptorHandleType>(handle.type);
        value = handle.value;
    }

    bool is_valid() const { return type != DescriptorHandleType::undefined; }

    explicit operator bool() const { return is_valid(); }

    std::string to_string() const;
};

enum class ShaderModel : uint32_t {
    unknown = 0,
    sm_6_0 = 60,
    sm_6_1 = 61,
    sm_6_2 = 62,
    sm_6_3 = 63,
    sm_6_4 = 64,
    sm_6_5 = 65,
    sm_6_6 = 66,
    sm_6_7 = 67,
};

SGL_ENUM_INFO(
    ShaderModel,
    {
        {ShaderModel::unknown, "unknown"},
        {ShaderModel::sm_6_0, "sm_6_0"},
        {ShaderModel::sm_6_1, "sm_6_1"},
        {ShaderModel::sm_6_2, "sm_6_2"},
        {ShaderModel::sm_6_3, "sm_6_3"},
        {ShaderModel::sm_6_4, "sm_6_4"},
        {ShaderModel::sm_6_5, "sm_6_5"},
        {ShaderModel::sm_6_6, "sm_6_6"},
        {ShaderModel::sm_6_7, "sm_6_7"},
    }
);
SGL_ENUM_REGISTER(ShaderModel);

inline uint32_t get_shader_model_major_version(ShaderModel sm)
{
    return static_cast<uint32_t>(sm) / 10;
}

inline uint32_t get_shader_model_minor_version(ShaderModel sm)
{
    return static_cast<uint32_t>(sm) % 10;
}

enum class ShaderStage : uint32_t {
    none = SLANG_STAGE_NONE,
    vertex = SLANG_STAGE_VERTEX,
    hull = SLANG_STAGE_HULL,
    domain = SLANG_STAGE_DOMAIN,
    geometry = SLANG_STAGE_GEOMETRY,
    fragment = SLANG_STAGE_FRAGMENT,
    compute = SLANG_STAGE_COMPUTE,
    ray_generation = SLANG_STAGE_RAY_GENERATION,
    intersection = SLANG_STAGE_INTERSECTION,
    any_hit = SLANG_STAGE_ANY_HIT,
    closest_hit = SLANG_STAGE_CLOSEST_HIT,
    miss = SLANG_STAGE_MISS,
    callable = SLANG_STAGE_CALLABLE,
    mesh = SLANG_STAGE_MESH,
    amplification = SLANG_STAGE_AMPLIFICATION,
};

SGL_ENUM_INFO(
    ShaderStage,
    {
        {ShaderStage::none, "none"},
        {ShaderStage::vertex, "vertex"},
        {ShaderStage::hull, "hull"},
        {ShaderStage::domain, "domain"},
        {ShaderStage::geometry, "geometry"},
        {ShaderStage::fragment, "fragment"},
        {ShaderStage::compute, "compute"},
        {ShaderStage::ray_generation, "ray_generation"},
        {ShaderStage::intersection, "intersection"},
        {ShaderStage::any_hit, "any_hit"},
        {ShaderStage::closest_hit, "closest_hit"},
        {ShaderStage::miss, "miss"},
        {ShaderStage::callable, "callable"},
        {ShaderStage::mesh, "mesh"},
        {ShaderStage::amplification, "amplification"},
    }
);
SGL_ENUM_REGISTER(ShaderStage);

enum class ComparisonFunc : uint32_t {
    never = static_cast<uint32_t>(rhi::ComparisonFunc::Never),
    less = static_cast<uint32_t>(rhi::ComparisonFunc::Less),
    equal = static_cast<uint32_t>(rhi::ComparisonFunc::Equal),
    less_equal = static_cast<uint32_t>(rhi::ComparisonFunc::LessEqual),
    greater = static_cast<uint32_t>(rhi::ComparisonFunc::Greater),
    not_equal = static_cast<uint32_t>(rhi::ComparisonFunc::NotEqual),
    greater_equal = static_cast<uint32_t>(rhi::ComparisonFunc::GreaterEqual),
    always = static_cast<uint32_t>(rhi::ComparisonFunc::Always),
};

SGL_ENUM_INFO(
    ComparisonFunc,
    {
        {ComparisonFunc::never, "never"},
        {ComparisonFunc::less, "less"},
        {ComparisonFunc::equal, "equal"},
        {ComparisonFunc::less_equal, "less_equal"},
        {ComparisonFunc::greater, "greater"},
        {ComparisonFunc::not_equal, "not_equal"},
        {ComparisonFunc::greater_equal, "greater_equal"},
        {ComparisonFunc::always, "always"},
    }
);
SGL_ENUM_REGISTER(ComparisonFunc);

// ----------------------------------------------------------------------------
// Sampler
// ----------------------------------------------------------------------------

enum class TextureFilteringMode : uint32_t {
    point = static_cast<uint32_t>(rhi::TextureFilteringMode::Point),
    linear = static_cast<uint32_t>(rhi::TextureFilteringMode::Linear),
};

SGL_ENUM_INFO(
    TextureFilteringMode,
    {
        {TextureFilteringMode::point, "point"},
        {TextureFilteringMode::linear, "linear"},
    }
);
SGL_ENUM_REGISTER(TextureFilteringMode);

enum class TextureAddressingMode : uint32_t {
    wrap = static_cast<uint32_t>(rhi::TextureAddressingMode::Wrap),
    clamp_to_edge = static_cast<uint32_t>(rhi::TextureAddressingMode::ClampToEdge),
    clamp_to_border = static_cast<uint32_t>(rhi::TextureAddressingMode::ClampToBorder),
    mirror_repeat = static_cast<uint32_t>(rhi::TextureAddressingMode::MirrorRepeat),
    mirror_once = static_cast<uint32_t>(rhi::TextureAddressingMode::MirrorOnce),
};

SGL_ENUM_INFO(
    TextureAddressingMode,
    {
        {TextureAddressingMode::wrap, "wrap"},
        {TextureAddressingMode::clamp_to_edge, "clamp_to_edge"},
        {TextureAddressingMode::clamp_to_border, "clamp_to_border"},
        {TextureAddressingMode::mirror_repeat, "mirror_repeat"},
        {TextureAddressingMode::mirror_once, "mirror_once"},
    }
);
SGL_ENUM_REGISTER(TextureAddressingMode);

enum class TextureReductionOp : uint32_t {
    average = static_cast<uint32_t>(rhi::TextureReductionOp::Average),
    comparison = static_cast<uint32_t>(rhi::TextureReductionOp::Comparison),
    minimum = static_cast<uint32_t>(rhi::TextureReductionOp::Minimum),
    maximum = static_cast<uint32_t>(rhi::TextureReductionOp::Maximum),
};

SGL_ENUM_INFO(
    TextureReductionOp,
    {
        {TextureReductionOp::average, "average"},
        {TextureReductionOp::comparison, "comparison"},
        {TextureReductionOp::minimum, "minimum"},
        {TextureReductionOp::maximum, "maximum"},
    }
);
SGL_ENUM_REGISTER(TextureReductionOp);

// ----------------------------------------------------------------------------
// Compute
// ----------------------------------------------------------------------------

struct IndirectDispatchArguments {
    uint32_t thread_group_count_x;
    uint32_t thread_group_count_y;
    uint32_t thread_group_count_z;
};

// ----------------------------------------------------------------------------
// Graphics
// ----------------------------------------------------------------------------

struct DrawArguments {
    uint32_t vertex_count{0};
    uint32_t instance_count{1};
    uint32_t start_vertex_location{0};
    uint32_t start_instance_location{0};
    uint32_t start_index_location{0};
};

struct IndirectDrawArguments {
    uint32_t vertex_count_per_instance{0};
    uint32_t instance_count{1};
    uint32_t start_vertex_location{0};
    uint32_t start_instance_location{0};
};

struct IndirectDrawIndexedArguments {
    uint32_t index_count_per_instance{0};
    uint32_t instance_count{1};
    uint32_t start_index_location{0};
    int32_t base_vertex_location{0};
    uint32_t start_instance_location{0};
};

struct SGL_API Viewport {
    float x{0.f};
    float y{0.f};
    float width{0.f};
    float height{0.f};
    float min_depth{0.f};
    float max_depth{1.f};

    static Viewport from_size(float width, float height) { return {0.f, 0.f, width, height, 0.f, 1.f}; }

    std::string to_string() const;
};

static_assert(
    (sizeof(Viewport) == sizeof(rhi::Viewport)) && (offsetof(Viewport, x) == offsetof(rhi::Viewport, originX))
        && (offsetof(Viewport, y) == offsetof(rhi::Viewport, originY))
        && (offsetof(Viewport, width) == offsetof(rhi::Viewport, extentX))
        && (offsetof(Viewport, height) == offsetof(rhi::Viewport, extentY))
        && (offsetof(Viewport, min_depth) == offsetof(rhi::Viewport, minZ))
        && (offsetof(Viewport, max_depth) == offsetof(rhi::Viewport, maxZ)),
    "Viewport struct mismatch"
);

struct SGL_API ScissorRect {
    uint32_t min_x{0};
    uint32_t min_y{0};
    uint32_t max_x{0};
    uint32_t max_y{0};

    static ScissorRect from_size(uint32_t width, uint32_t height) { return {0, 0, width, height}; }

    std::string to_string() const;
};

static_assert(
    (sizeof(ScissorRect) == sizeof(rhi::ScissorRect))
        && (offsetof(ScissorRect, min_x) == offsetof(rhi::ScissorRect, minX))
        && (offsetof(ScissorRect, min_y) == offsetof(rhi::ScissorRect, minY))
        && (offsetof(ScissorRect, max_x) == offsetof(rhi::ScissorRect, maxX))
        && (offsetof(ScissorRect, max_y) == offsetof(rhi::ScissorRect, maxY)),
    "ScissorRect struct mismatch"
);

enum class IndexFormat : uint8_t {
    uint16 = static_cast<uint8_t>(rhi::IndexFormat::Uint16),
    uint32 = static_cast<uint8_t>(rhi::IndexFormat::Uint32),
};

SGL_ENUM_INFO(
    IndexFormat,
    {
        {IndexFormat::uint16, "uint16"},
        {IndexFormat::uint32, "uint32"},
    }
);
SGL_ENUM_REGISTER(IndexFormat);

enum class PrimitiveTopology : uint8_t {
    point_list = static_cast<uint8_t>(rhi::PrimitiveTopology::PointList),
    line_list = static_cast<uint8_t>(rhi::PrimitiveTopology::LineList),
    line_strip = static_cast<uint8_t>(rhi::PrimitiveTopology::LineStrip),
    triangle_list = static_cast<uint8_t>(rhi::PrimitiveTopology::TriangleList),
    triangle_strip = static_cast<uint8_t>(rhi::PrimitiveTopology::TriangleStrip),
    patch_list = static_cast<uint8_t>(rhi::PrimitiveTopology::PatchList),
};

SGL_ENUM_INFO(
    PrimitiveTopology,
    {
        {PrimitiveTopology::point_list, "point_list"},
        {PrimitiveTopology::line_list, "line_list"},
        {PrimitiveTopology::line_strip, "line_strip"},
        {PrimitiveTopology::triangle_list, "triangle_list"},
        {PrimitiveTopology::triangle_strip, "triangle_strip"},
        {PrimitiveTopology::patch_list, "patch_list"},
    }
);
SGL_ENUM_REGISTER(PrimitiveTopology);

enum class LoadOp : uint8_t {
    load = static_cast<uint8_t>(rhi::LoadOp::Load),
    clear = static_cast<uint8_t>(rhi::LoadOp::Clear),
    dont_care = static_cast<uint8_t>(rhi::LoadOp::DontCare),
};

SGL_ENUM_INFO(
    LoadOp,
    {
        {LoadOp::load, "load"},
        {LoadOp::clear, "clear"},
        {LoadOp::dont_care, "dont_care"},
    }
);
SGL_ENUM_REGISTER(LoadOp);

enum class StoreOp : uint8_t {
    store = static_cast<uint8_t>(rhi::StoreOp::Store),
    dont_care = static_cast<uint8_t>(rhi::StoreOp::DontCare),
};

SGL_ENUM_INFO(
    StoreOp,
    {
        {StoreOp::store, "store"},
        {StoreOp::dont_care, "dont_care"},
    }
);
SGL_ENUM_REGISTER(StoreOp);

enum class StencilOp : uint8_t {
    keep = static_cast<uint8_t>(rhi::StencilOp::Keep),
    zero = static_cast<uint8_t>(rhi::StencilOp::Keep),
    replace = static_cast<uint8_t>(rhi::StencilOp::Keep),
    increment_saturate = static_cast<uint8_t>(rhi::StencilOp::Keep),
    decrement_saturate = static_cast<uint8_t>(rhi::StencilOp::Keep),
    invert = static_cast<uint8_t>(rhi::StencilOp::Keep),
    increment_wrap = static_cast<uint8_t>(rhi::StencilOp::Keep),
    decrement_wrap = static_cast<uint8_t>(rhi::StencilOp::Keep),
};

SGL_ENUM_INFO(
    StencilOp,
    {
        {StencilOp::keep, "keep"},
        {StencilOp::zero, "zero"},
        {StencilOp::replace, "replace"},
        {StencilOp::increment_saturate, "increment_saturate"},
        {StencilOp::decrement_saturate, "decrement_saturate"},
        {StencilOp::invert, "invert"},
        {StencilOp::increment_wrap, "increment_wrap"},
        {StencilOp::decrement_wrap, "decrement_wrap"},
    }
);
SGL_ENUM_REGISTER(StencilOp);

enum class FillMode : uint8_t {
    solid = static_cast<uint8_t>(rhi::FillMode::Solid),
    wireframe = static_cast<uint8_t>(rhi::FillMode::Wireframe),
};

SGL_ENUM_INFO(
    FillMode,
    {
        {FillMode::solid, "solid"},
        {FillMode::wireframe, "wireframe"},
    }
);
SGL_ENUM_REGISTER(FillMode);

enum class CullMode : uint8_t {
    none = static_cast<uint8_t>(rhi::CullMode::None),
    front = static_cast<uint8_t>(rhi::CullMode::Front),
    back = static_cast<uint8_t>(rhi::CullMode::Back),
};

SGL_ENUM_INFO(
    CullMode,
    {
        {CullMode::none, "none"},
        {CullMode::front, "front"},
        {CullMode::back, "back"},
    }
);
SGL_ENUM_REGISTER(CullMode);

enum class FrontFaceMode : uint8_t {
    counter_clockwise = static_cast<uint8_t>(rhi::FrontFaceMode::CounterClockwise),
    clockwise = static_cast<uint8_t>(rhi::FrontFaceMode::Clockwise),
};

SGL_ENUM_INFO(
    FrontFaceMode,
    {
        {FrontFaceMode::counter_clockwise, "counter_clockwise"},
        {FrontFaceMode::clockwise, "clockwise"},
    }
);
SGL_ENUM_REGISTER(FrontFaceMode);

enum class LogicOp : uint8_t {
    no_op = static_cast<uint8_t>(rhi::LogicOp::NoOp),
};

SGL_ENUM_INFO(
    LogicOp,
    {
        {LogicOp::no_op, "no_op"},
    }
);
SGL_ENUM_REGISTER(LogicOp);

enum class BlendOp : uint8_t {
    add = static_cast<uint8_t>(rhi::BlendOp::Add),
    subtract = static_cast<uint8_t>(rhi::BlendOp::Subtract),
    reverse_subtract = static_cast<uint8_t>(rhi::BlendOp::ReverseSubtract),
    min = static_cast<uint8_t>(rhi::BlendOp::Min),
    max = static_cast<uint8_t>(rhi::BlendOp::Max),
};

SGL_ENUM_INFO(
    BlendOp,
    {
        {BlendOp::add, "add"},
        {BlendOp::subtract, "subtract"},
        {BlendOp::reverse_subtract, "reverse_subtract"},
        {BlendOp::min, "min"},
        {BlendOp::max, "max"},
    }
);
SGL_ENUM_REGISTER(BlendOp);

enum class BlendFactor : uint8_t {
    zero = static_cast<uint8_t>(rhi::BlendFactor::Zero),
    one = static_cast<uint8_t>(rhi::BlendFactor::One),
    src_color = static_cast<uint8_t>(rhi::BlendFactor::SrcColor),
    inv_src_color = static_cast<uint8_t>(rhi::BlendFactor::InvSrcColor),
    src_alpha = static_cast<uint8_t>(rhi::BlendFactor::SrcAlpha),
    inv_src_alpha = static_cast<uint8_t>(rhi::BlendFactor::InvSrcAlpha),
    dest_alpha = static_cast<uint8_t>(rhi::BlendFactor::DestAlpha),
    inv_dest_alpha = static_cast<uint8_t>(rhi::BlendFactor::InvDestAlpha),
    dest_color = static_cast<uint8_t>(rhi::BlendFactor::DestColor),
    inv_dest_color = static_cast<uint8_t>(rhi::BlendFactor::InvDestColor),
    src_alpha_saturate = static_cast<uint8_t>(rhi::BlendFactor::SrcAlphaSaturate),
    blend_color = static_cast<uint8_t>(rhi::BlendFactor::BlendColor),
    inv_blend_color = static_cast<uint8_t>(rhi::BlendFactor::InvBlendColor),
    secondary_src_color = static_cast<uint8_t>(rhi::BlendFactor::SecondarySrcColor),
    inv_secondary_src_color = static_cast<uint8_t>(rhi::BlendFactor::InvSecondarySrcColor),
    secondary_src_alpha = static_cast<uint8_t>(rhi::BlendFactor::SecondarySrcAlpha),
    inv_secondary_src_alpha = static_cast<uint8_t>(rhi::BlendFactor::InvSecondarySrcAlpha),
};

SGL_ENUM_INFO(
    BlendFactor,
    {
        {BlendFactor::zero, "zero"},
        {BlendFactor::one, "one"},
        {BlendFactor::src_color, "src_color"},
        {BlendFactor::inv_src_color, "inv_src_color"},
        {BlendFactor::src_alpha, "src_alpha"},
        {BlendFactor::inv_src_alpha, "inv_src_alpha"},
        {BlendFactor::dest_alpha, "dest_alpha"},
        {BlendFactor::inv_dest_alpha, "inv_dest_alpha"},
        {BlendFactor::dest_color, "dest_color"},
        {BlendFactor::inv_dest_color, "inv_dest_color"},
        {BlendFactor::src_alpha_saturate, "src_alpha_saturate"},
        {BlendFactor::blend_color, "blend_color"},
        {BlendFactor::inv_blend_color, "inv_blend_color"},
        {BlendFactor::secondary_src_color, "secondary_src_color"},
        {BlendFactor::inv_secondary_src_color, "inv_secondary_src_color"},
        {BlendFactor::secondary_src_alpha, "secondary_src_alpha"},
        {BlendFactor::inv_secondary_src_alpha, "inv_secondary_src_alpha"},
    }
);
SGL_ENUM_REGISTER(BlendFactor);

enum class RenderTargetWriteMask : uint8_t {
    none = static_cast<uint8_t>(rhi::RenderTargetWriteMask::None),
    red = static_cast<uint8_t>(rhi::RenderTargetWriteMask::Red),
    green = static_cast<uint8_t>(rhi::RenderTargetWriteMask::Green),
    blue = static_cast<uint8_t>(rhi::RenderTargetWriteMask::Blue),
    alpha = static_cast<uint8_t>(rhi::RenderTargetWriteMask::Alpha),
    all = static_cast<uint8_t>(rhi::RenderTargetWriteMask::All),
};

SGL_ENUM_CLASS_OPERATORS(RenderTargetWriteMask);
SGL_ENUM_INFO(
    RenderTargetWriteMask,
    {
        {RenderTargetWriteMask::none, "none"},
        {RenderTargetWriteMask::red, "red"},
        {RenderTargetWriteMask::green, "green"},
        {RenderTargetWriteMask::blue, "blue"},
        {RenderTargetWriteMask::alpha, "alpha"},
        {RenderTargetWriteMask::all, "all"},
    }
);
SGL_ENUM_REGISTER(RenderTargetWriteMask);

struct AspectBlendDesc {
    BlendFactor src_factor{BlendFactor::one};
    BlendFactor dst_factor{BlendFactor::zero};
    BlendOp op{BlendOp::add};
};

struct ColorTargetDesc {
    Format format{Format::undefined};
    AspectBlendDesc color;
    AspectBlendDesc alpha;
    bool enable_blend{false};
    LogicOp logic_op{LogicOp::no_op};
    RenderTargetWriteMask write_mask{RenderTargetWriteMask::all};
};

struct MultisampleDesc {
    uint32_t sample_count{1};
    uint32_t sample_mask{0xffffffff};
    bool alpha_to_coverage_enable{false};
    bool alpha_to_one_enable{false};
};

struct DepthStencilOpDesc {
    StencilOp stencil_fail_op{StencilOp::keep};
    StencilOp stencil_depth_fail_op{StencilOp::keep};
    StencilOp stencil_pass_op{StencilOp::keep};
    ComparisonFunc stencil_func{ComparisonFunc::always};
};

struct DepthStencilDesc {
    Format format{Format::undefined};

    bool depth_test_enable{false};
    bool depth_write_enable{true};
    ComparisonFunc depth_func{ComparisonFunc::less};

    bool stencil_enable{false};
    uint32_t stencil_read_mask{0xffffffff};
    uint32_t stencil_write_mask{0xffffffff};
    DepthStencilOpDesc front_face;
    DepthStencilOpDesc back_face;
};

struct RasterizerDesc {
    FillMode fill_mode{FillMode::solid};
    CullMode cull_mode{CullMode::none};
    FrontFaceMode front_face{FrontFaceMode::counter_clockwise};
    int32_t depth_bias{0};
    float depth_bias_clamp{0.0f};
    float slope_scaled_depth_bias{0.0f};
    bool depth_clip_enable{true};
    bool scissor_enable{false};
    bool multisample_enable{false};
    bool antialiased_line_enable{false};
    bool enable_conservative_rasterization{false};
    uint32_t forced_sample_count{0};
};

// ----------------------------------------------------------------------------
// Queries
// ----------------------------------------------------------------------------

enum class QueryType : uint32_t {
    timestamp = static_cast<uint32_t>(rhi::QueryType::Timestamp),
    acceleration_structure_compacted_size = static_cast<uint32_t>(rhi::QueryType::AccelerationStructureCompactedSize),
    acceleration_structure_serialized_size = static_cast<uint32_t>(rhi::QueryType::AccelerationStructureSerializedSize),
    acceleration_structure_current_size = static_cast<uint32_t>(rhi::QueryType::AccelerationStructureCurrentSize),
};

SGL_ENUM_INFO(
    QueryType,
    {
        {QueryType::timestamp, "timestamp"},
        {QueryType::acceleration_structure_compacted_size, "acceleration_structure_compacted_size"},
        {QueryType::acceleration_structure_serialized_size, "acceleration_structure_serialized_size"},
        {QueryType::acceleration_structure_current_size, "acceleration_structure_current_size"},
    }
);
SGL_ENUM_REGISTER(QueryType);

// ----------------------------------------------------------------------------
// RayTracing
// ----------------------------------------------------------------------------

enum class RayTracingPipelineFlags : uint8_t {
    none = static_cast<uint8_t>(rhi::RayTracingPipelineFlags::None),
    skip_triangles = static_cast<uint8_t>(rhi::RayTracingPipelineFlags::SkipTriangles),
    skip_procedurals = static_cast<uint8_t>(rhi::RayTracingPipelineFlags::SkipProcedurals),
};

SGL_ENUM_CLASS_OPERATORS(RayTracingPipelineFlags);
SGL_ENUM_INFO(
    RayTracingPipelineFlags,
    {
        {RayTracingPipelineFlags::none, "none"},
        {RayTracingPipelineFlags::skip_triangles, "skip_triangles"},
        {RayTracingPipelineFlags::skip_procedurals, "skip_procedurals"},
    }
);
SGL_ENUM_REGISTER(RayTracingPipelineFlags);

/// Report information for a memory heap.
struct HeapReport {
    /// Name of the heap.
    std::string label;
    /// Number of pages in the heap.
    uint32_t num_pages{0};
    /// Total allocated memory in bytes.
    uint64_t total_allocated{0};
    /// Total memory usage in bytes.
    uint64_t total_mem_usage{0};
    /// Number of allocations.
    uint64_t num_allocations{0};

    std::string to_string() const
    {
        return fmt::format(
            "HeapReport(\n"
            "  label = \"{}\",\n"
            "  num_pages = {},\n"
            "  total_allocated = {},\n"
            "  total_mem_usage = {},\n"
            "  num_allocations = {}\n"
            ")",
            label,
            num_pages,
            total_allocated,
            total_mem_usage,
            num_allocations
        );
    }
};

} // namespace sgl
