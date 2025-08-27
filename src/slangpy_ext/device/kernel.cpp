// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/kernel.h"
#include "sgl/device/command.h"
#include "sgl/device/resource.h"
#include "sgl/device/sampler.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/shader.h"

namespace sgl {

extern void write_shader_cursor(ShaderCursor& cursor, nb::object value);

inline void bind_python_var(ShaderCursor cursor, nb::handle var)
{
    write_shader_cursor(cursor, nb::cast<nb::object>(var));
}

} // namespace sgl

SGL_PY_EXPORT(device_kernel)
{
    using namespace sgl;

    nb::class_<Kernel, DeviceChild>(m, "Kernel", D(Kernel)) //
        .def_prop_ro("program", &Kernel::program, D(Kernel, program))
        .def_prop_ro("reflection", &Kernel::reflection, D(Kernel, reflection));

    nb::class_<ComputeKernelDesc>(m, "ComputeKernelDesc", D(ComputeKernelDesc))
        .def(nb::init<>())
        .def_rw("program", &ComputeKernelDesc::program, D(ComputeKernelDesc, program));

    nb::class_<ComputeKernel, Kernel>(m, "ComputeKernel", D(ComputeKernel))
        .def_prop_ro("pipeline", &ComputeKernel::pipeline, D(ComputeKernel, pipeline))
        .def(
            "dispatch",
            [](ComputeKernel* self,
               uint3 thread_count,
               nb::dict vars,
               CommandEncoder* command_encoder,
               CommandQueueType queue,
               NativeHandle cuda_stream,
               nb::kwargs kwargs)
            {
                auto bind_vars = [&](ShaderCursor cursor)
                {
                    // bind locals
                    if (kwargs.size() > 0)
                        bind_python_var(cursor.find_entry_point(0), kwargs);
                    // bind globals
                    bind_python_var(cursor, vars);
                };
                if (command_encoder) {
                    SGL_CHECK(
                        !cuda_stream.is_valid(),
                        "Can not specify CUDA stream if appending to a command encoder."
                    );
                    self->dispatch(thread_count, bind_vars, command_encoder);
                } else {
                    self->dispatch(thread_count, bind_vars, queue, cuda_stream);
                }
            },
            "thread_count"_a,
            "vars"_a = nb::dict(),
            "command_encoder"_a = nullptr,
            "queue"_a = CommandQueueType::graphics,
            "cuda_stream"_a = NativeHandle(),
            "kwargs"_a,
            D(ComputeKernel, dispatch)
        );
}
