// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kernel.h"

#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/command.h"
#include "sgl/device/shader_cursor.h"

#include "sgl/core/maths.h"

namespace sgl {

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------

Kernel::Kernel(ref<Device> device, ref<ShaderProgram> program)
    : DeviceChild(std::move(device))
    , m_program(std::move(program))
{
}

// ----------------------------------------------------------------------------
// ComputeKernel
// ----------------------------------------------------------------------------

ComputeKernel::ComputeKernel(ref<Device> device, ComputeKernelDesc desc)
    : Kernel(std::move(device), std::move(desc.program))
{
    m_thread_group_size = m_program->layout()->get_entry_point_by_index(0)->compute_thread_group_size();
}

ComputePipeline* ComputeKernel::pipeline() const
{
    if (!m_pipeline)
        m_pipeline = m_device->create_compute_pipeline({.program = m_program});
    return m_pipeline;
}

void ComputeKernel::dispatch(uint3 thread_count, BindVarsCallback bind_vars, CommandEncoder* command_encoder)
{
    {
        auto pass_encoder = command_encoder->begin_compute_pass();
        ShaderObject* shader_object = pass_encoder->bind_pipeline(pipeline());
        if (bind_vars)
            bind_vars(ShaderCursor(shader_object));
        pass_encoder->dispatch(thread_count);
        pass_encoder->end();
    }
}

void ComputeKernel::dispatch(
    uint3 thread_count,
    BindVarsCallback bind_vars,
    CommandQueueType queue,
    NativeHandle cuda_stream
)
{
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();

    dispatch(thread_count, bind_vars, command_encoder);

    m_device->submit_command_buffer(command_encoder->finish(), queue, cuda_stream);
}

// ----------------------------------------------------------------------------
// RayTracingKernel
// ----------------------------------------------------------------------------

RayTracingKernel::RayTracingKernel(ref<Device> device, ref<ShaderProgram> program)
    : Kernel(std::move(device), std::move(program))
{
}

RayTracingPipeline* RayTracingKernel::pipeline() const
{
    if (!m_pipeline)
        m_pipeline = m_device->create_ray_tracing_pipeline({.program = m_program});
    return m_pipeline;
}

// void RayTracingKernel::dispatch(uint3 thread_count, BindVarsCallback bind_vars, CommandEncoder* command_encoder) { }

} // namespace sgl
