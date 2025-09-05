// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/device_child.h"
#include "sgl/device/types.h"
#include "sgl/device/native_handle.h"

#include "sgl/core/macros.h"
#include "sgl/core/enum.h"
#include "sgl/core/object.h"

#include "sgl/math/vector_types.h"

#include <slang-rhi.h>

#include <map>
#include <set>

namespace sgl {


/// Pipeline base class.
class SGL_API Pipeline : public DeviceChild {
    SGL_OBJECT(Pipeline)
public:
    Pipeline(ref<Device> device, ref<ShaderProgram> program);
    virtual ~Pipeline();

    void notify_program_reloaded();

protected:
    virtual void recreate() = 0;

private:
    /// Pipelines store program (and thus maintain the ref count)
    /// in their descriptor - this is just so we can register/unregister
    /// the with program. However due to order of destruction
    /// this still needs to hold a strong reference.
    ref<ShaderProgram> m_program;
};

struct ComputePipelineDesc {
    ref<ShaderProgram> program;
    bool defer_target_compilation{false};
    std::string label;
};

/// Compute pipeline.
class SGL_API ComputePipeline : public Pipeline {
public:
    ComputePipeline(ref<Device> device, ComputePipelineDesc desc);

    virtual void _release_rhi_resources() override { m_rhi_pipeline.setNull(); }

    const ComputePipelineDesc& desc() const { return m_desc; }

    /// Thread group size.
    /// Used to determine the number of thread groups to dispatch.
    uint3 thread_group_size() const { return m_thread_group_size; }

    /// Get the native pipeline handle.
    NativeHandle native_handle() const;

    rhi::IComputePipeline* rhi_pipeline() const { return m_rhi_pipeline; }

    std::string to_string() const override;

protected:
    virtual void recreate() override;

private:
    ComputePipelineDesc m_desc;
    uint3 m_thread_group_size;
    Slang::ComPtr<rhi::IComputePipeline> m_rhi_pipeline;
};

struct RenderPipelineDesc {
    ref<ShaderProgram> program;
    ref<InputLayout> input_layout;
    PrimitiveTopology primitive_topology{PrimitiveTopology::triangle_list};
    std::vector<ColorTargetDesc> targets;
    DepthStencilDesc depth_stencil;
    RasterizerDesc rasterizer;
    MultisampleDesc multisample;
    bool defer_target_compilation{false};
    std::string label;
};

/// Render pipeline.
class SGL_API RenderPipeline : public Pipeline {
public:
    RenderPipeline(ref<Device> device, RenderPipelineDesc desc);

    virtual void _release_rhi_resources() override { m_rhi_pipeline.setNull(); }

    const RenderPipelineDesc& desc() const { return m_desc; }

    /// Get the native pipeline handle.
    NativeHandle native_handle() const;

    rhi::IRenderPipeline* rhi_pipeline() const { return m_rhi_pipeline; }

    std::string to_string() const override;

protected:
    virtual void recreate() override;

private:
    RenderPipelineDesc m_desc;

    // These are stored to ensure the layouts aren't freed when pipeline
    // relies on them if it needs to be recreated for hot reload.
    ref<const InputLayout> m_stored_input_layout;

    Slang::ComPtr<rhi::IRenderPipeline> m_rhi_pipeline;
};

struct HitGroupDesc {
    std::string hit_group_name;
    std::string closest_hit_entry_point;
    std::string any_hit_entry_point;
    std::string intersection_entry_point;
};

struct RayTracingPipelineDesc {
    ref<ShaderProgram> program;
    std::vector<HitGroupDesc> hit_groups;
    uint32_t max_recursion{0};
    uint32_t max_ray_payload_size{0};
    uint32_t max_attribute_size{8};
    RayTracingPipelineFlags flags{RayTracingPipelineFlags::none};
    bool defer_target_compilation{false};
    std::string label;
};

/// Ray tracing pipeline.
class SGL_API RayTracingPipeline : public Pipeline {
public:
    RayTracingPipeline(ref<Device> device, RayTracingPipelineDesc desc);

    virtual void _release_rhi_resources() override { m_rhi_pipeline.setNull(); }

    const RayTracingPipelineDesc& desc() const { return m_desc; }

    /// Get the native pipeline handle.
    NativeHandle native_handle() const;

    rhi::IRayTracingPipeline* rhi_pipeline() const { return m_rhi_pipeline; }

    std::string to_string() const override;

protected:
    virtual void recreate() override;

private:
    RayTracingPipelineDesc m_desc;
    Slang::ComPtr<rhi::IRayTracingPipeline> m_rhi_pipeline;
};

} // namespace sgl
