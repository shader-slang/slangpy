// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pipeline.h"

#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/input_layout.h"
#include "sgl/device/helpers.h"

#include "sgl/core/config.h"
#include "sgl/core/short_vector.h"
#include "sgl/core/type_utils.h"

#include "sgl/math/vector.h"


namespace sgl {

// ----------------------------------------------------------------------------
// Pipeline
// ----------------------------------------------------------------------------

Pipeline::Pipeline(ref<Device> device, ref<ShaderProgram> program)
    : DeviceResource(std::move(device))
    , m_program(std::move(program))
{
    m_program->_register_pipeline(this);
}

Pipeline::~Pipeline()
{
    m_program->_unregister_pipeline(this);
}

void Pipeline::notify_program_reloaded()
{
    recreate();
}

// ----------------------------------------------------------------------------
// ComputePipeline
// ----------------------------------------------------------------------------

ComputePipeline::ComputePipeline(ref<Device> device, ComputePipelineDesc desc)
    : Pipeline(std::move(device), desc.program)
    , m_desc(std::move(desc))
{
    recreate();
}

void ComputePipeline::recreate()
{
    rhi::ComputePipelineDesc rhi_desc{
        .program = m_desc.program->rhi_shader_program(),
        .label = m_desc.label.empty() ? nullptr : m_desc.label.c_str(),
    };
    SLANG_RHI_CALL(
        m_device->rhi_device()->createComputePipeline(rhi_desc, (rhi::IComputePipeline**)m_rhi_pipeline.writeRef())
    );
    m_thread_group_size = m_desc.program->layout()->get_entry_point_by_index(0)->compute_thread_group_size();
}

NativeHandle ComputePipeline::native_handle() const
{
    rhi::NativeHandle rhi_handle = {};
    m_rhi_pipeline->getNativeHandle(&rhi_handle);
    return NativeHandle(rhi_handle);
}

std::string ComputePipeline::to_string() const
{
    return fmt::format(
        "ComputePipeline(\n"
        "  device = {},\n"
        "  program = {},\n"
        "  label = {},\n"
        "  thread_group_size = {}\n"
        ")",
        m_device,
        m_desc.program,
        m_desc.label,
        m_thread_group_size
    );
}

// ----------------------------------------------------------------------------
// RenderPipeline
// ----------------------------------------------------------------------------

RenderPipeline::RenderPipeline(ref<Device> device, RenderPipelineDesc desc)
    : Pipeline(std::move(device), desc.program)
    , m_desc(std::move(desc))
{
    m_stored_input_layout = ref<const InputLayout>(m_desc.input_layout);
    recreate();
}

void RenderPipeline::recreate()
{
    const RenderPipelineDesc& desc = m_desc;

    short_vector<rhi::ColorTargetDesc, 8> rhi_targets;
    for (size_t i = 0; i < desc.targets.size(); ++i) {
        const auto& target = desc.targets[i];
        rhi_targets.push_back({
            .format = static_cast<rhi::Format>(target.format),
            .color = {
                .srcFactor = static_cast<rhi::BlendFactor>(target.color.src_factor),
                .dstFactor = static_cast<rhi::BlendFactor>(target.color.dst_factor),
                .op = static_cast<rhi::BlendOp>(target.color.op),
            },
            .alpha = {
                .srcFactor = static_cast<rhi::BlendFactor>(target.alpha.src_factor),
                .dstFactor = static_cast<rhi::BlendFactor>(target.alpha.dst_factor),
                .op = static_cast<rhi::BlendOp>(target.alpha.op),
            },
            .enableBlend = target.enable_blend,
            .logicOp = static_cast<rhi::LogicOp>(target.logic_op),
            .writeMask = static_cast<rhi::RenderTargetWriteMaskT>(target.write_mask),
        });
    }

    rhi::RenderPipelineDesc rhi_desc{
        .program =  m_desc.program->rhi_shader_program(),
        .inputLayout = desc.input_layout ? desc.input_layout->rhi_input_layout() : nullptr,
        .primitiveTopology = static_cast<rhi::PrimitiveTopology>(desc.primitive_topology),
        .targets = rhi_targets.data(),
        .targetCount = narrow_cast<uint32_t>(rhi_targets.size()),
        .depthStencil = {
            .format = static_cast<rhi::Format>(desc.depth_stencil.format),
            .depthTestEnable = desc.depth_stencil.depth_test_enable,
            .depthWriteEnable = desc.depth_stencil.depth_write_enable,
            .depthFunc = static_cast<rhi::ComparisonFunc>(desc.depth_stencil.depth_func),
            .stencilEnable = desc.depth_stencil.stencil_enable,
            .stencilReadMask = desc.depth_stencil.stencil_read_mask,
            .stencilWriteMask = desc.depth_stencil.stencil_write_mask,
            .frontFace = {
                .stencilFailOp = static_cast<rhi::StencilOp>(desc.depth_stencil.front_face.stencil_fail_op),
                .stencilDepthFailOp = static_cast<rhi::StencilOp>(desc.depth_stencil.front_face.stencil_depth_fail_op),
                .stencilPassOp = static_cast<rhi::StencilOp>(desc.depth_stencil.front_face.stencil_pass_op),
                .stencilFunc = static_cast<rhi::ComparisonFunc>(desc.depth_stencil.front_face.stencil_func),
            },
            .backFace = {
                .stencilFailOp = static_cast<rhi::StencilOp>(desc.depth_stencil.back_face.stencil_fail_op),
                .stencilDepthFailOp = static_cast<rhi::StencilOp>(desc.depth_stencil.back_face.stencil_depth_fail_op),
                .stencilPassOp = static_cast<rhi::StencilOp>(desc.depth_stencil.back_face.stencil_pass_op),
                .stencilFunc = static_cast<rhi::ComparisonFunc>(desc.depth_stencil.back_face.stencil_func),
            },
        },
        .rasterizer = {
            .fillMode = static_cast<rhi::FillMode>(desc.rasterizer.fill_mode),
            .cullMode = static_cast<rhi::CullMode>(desc.rasterizer.cull_mode),
            .frontFace = static_cast<rhi::FrontFaceMode>(desc.rasterizer.front_face),
            .depthBias = desc.rasterizer.depth_bias,
            .depthBiasClamp = desc.rasterizer.depth_bias_clamp,
            .slopeScaledDepthBias = desc.rasterizer.slope_scaled_depth_bias,
            .depthClipEnable = desc.rasterizer.depth_clip_enable,
            .scissorEnable = desc.rasterizer.scissor_enable,
            .multisampleEnable = desc.rasterizer.multisample_enable,
            .antialiasedLineEnable = desc.rasterizer.antialiased_line_enable,
            .enableConservativeRasterization = desc.rasterizer.enable_conservative_rasterization,
            .forcedSampleCount = desc.rasterizer.forced_sample_count,
        },
        .multisample = {
            .sampleCount = desc.multisample.sample_count,
            .sampleMask = desc.multisample.sample_mask,
            .alphaToCoverageEnable = desc.multisample.alpha_to_coverage_enable,
            .alphaToOneEnable = desc.multisample.alpha_to_one_enable,
        },
        .label = m_desc.label.empty() ? nullptr : m_desc.label.c_str(),
    };

    SLANG_RHI_CALL(
        m_device->rhi_device()->createRenderPipeline(rhi_desc, (rhi::IRenderPipeline**)m_rhi_pipeline.writeRef())
    );
}

NativeHandle RenderPipeline::native_handle() const
{
    rhi::NativeHandle rhi_handle = {};
    m_rhi_pipeline->getNativeHandle(&rhi_handle);
    return NativeHandle(rhi_handle);
}

std::string RenderPipeline::to_string() const
{
    return fmt::format(
        "RenderPipeline(\n"
        "  device = {},\n"
        "  program = {},\n"
        "  label = {}\n"
        ")",
        m_device,
        m_desc.program,
        m_desc.label
    );
}

// ----------------------------------------------------------------------------
// RayTracingPipeline
// ----------------------------------------------------------------------------

RayTracingPipeline::RayTracingPipeline(ref<Device> device, RayTracingPipelineDesc desc)
    : Pipeline(std::move(device), desc.program)
    , m_desc(std::move(desc))
{
    recreate();
}

void RayTracingPipeline::recreate()
{
    const RayTracingPipelineDesc& desc = m_desc;

    short_vector<rhi::HitGroupDesc, 16> rhi_hit_groups;
    rhi_hit_groups.reserve(desc.hit_groups.size());
    for (const auto& hit_group : desc.hit_groups) {
        rhi_hit_groups.push_back({
            .hitGroupName = hit_group.hit_group_name.c_str(),
            .closestHitEntryPoint
            = hit_group.closest_hit_entry_point.empty() ? nullptr : hit_group.closest_hit_entry_point.c_str(),
            .anyHitEntryPoint = hit_group.any_hit_entry_point.empty() ? nullptr : hit_group.any_hit_entry_point.c_str(),
            .intersectionEntryPoint
            = hit_group.intersection_entry_point.empty() ? nullptr : hit_group.intersection_entry_point.c_str(),
        });
    }

    rhi::RayTracingPipelineDesc rhi_desc{
        .program = m_desc.program->rhi_shader_program(),
        .hitGroupCount = narrow_cast<uint32_t>(rhi_hit_groups.size()),
        .hitGroups = rhi_hit_groups.data(),
        .maxRecursion = desc.max_recursion,
        .maxRayPayloadSize = desc.max_ray_payload_size,
        .maxAttributeSizeInBytes = desc.max_attribute_size,
        .flags = static_cast<rhi::RayTracingPipelineFlags>(desc.flags),
        .label = m_desc.label.empty() ? nullptr : m_desc.label.c_str(),
    };
    SLANG_RHI_CALL(m_device->rhi_device()
                       ->createRayTracingPipeline(rhi_desc, (rhi::IRayTracingPipeline**)m_rhi_pipeline.writeRef()));
}

NativeHandle RayTracingPipeline::native_handle() const
{
    rhi::NativeHandle rhi_handle = {};
    m_rhi_pipeline->getNativeHandle(&rhi_handle);
    return NativeHandle(rhi_handle);
}

std::string RayTracingPipeline::to_string() const
{
    return fmt::format(
        "RayTracingPipeline(\n"
        "  device = {},\n"
        "  program = {},\n"
        "  label = {}\n"
        ")",
        m_device,
        m_desc.program,
        m_desc.label
    );
}

} // namespace sgl
