// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/device_resource.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/native_handle.h"

#include "sgl/core/macros.h"
#include "sgl/core/object.h"

#include <functional>

namespace sgl {

class SGL_API Kernel : public DeviceResource {
    SGL_OBJECT(Kernel)
public:
    using BindVarsCallback = std::function<void(ShaderCursor)>;

    virtual ~Kernel() = default;

    ShaderProgram* program() const { return m_program; }
    ReflectionCursor reflection() const { return ReflectionCursor(program()); }

protected:
    Kernel(ref<Device> device, ref<ShaderProgram> program);

    ref<ShaderProgram> m_program;
};

struct ComputeKernelDesc {
    ref<ShaderProgram> program;
};

class SGL_API ComputeKernel : public Kernel {
    SGL_OBJECT(ComputeKernel)
public:
    ComputeKernel(ref<Device> device, ComputeKernelDesc desc);
    ~ComputeKernel() = default;

    ComputePipeline* pipeline() const;

    void dispatch(uint3 thread_count, BindVarsCallback bind_vars, CommandEncoder* command_encoder);

    void dispatch(
        uint3 thread_count,
        BindVarsCallback bind_vars,
        CommandQueueType queue = CommandQueueType::graphics,
        NativeHandle cuda_stream = {}
    );

private:
    uint3 m_thread_group_size;
    mutable ref<ComputePipeline> m_pipeline;
};

class SGL_API RayTracingKernel : public Kernel {
    SGL_OBJECT(RayTracingKernel)
public:
    RayTracingKernel(ref<Device> device, ref<ShaderProgram> program);
    ~RayTracingKernel() = default;

    RayTracingPipeline* pipeline() const;

    void dispatch(uint3 thread_count, BindVarsCallback bind_vars, CommandEncoder* command_encoder = nullptr);

private:
    mutable ref<RayTracingPipeline> m_pipeline;
};

} // namespace sgl
