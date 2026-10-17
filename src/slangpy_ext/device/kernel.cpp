// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/kernel.h"
#include "sgl/device/command.h"
#include "sgl/device/resource.h"
#include "sgl/device/sampler.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/query.h"
#include "sgl/device/shader.h"

namespace sgl {

extern void write_shader_cursor(ShaderCursor& cursor, nb::object value);

inline void bind_python_var(ShaderCursor cursor, nb::handle var)
{
    write_shader_cursor(cursor, nb::cast<nb::object>(var));
}

/// Writing a dict to a cursor skips names the shader does not declare, so check them first:
/// a global passed as a keyword argument (or a typo) would otherwise leave the resource unbound.
inline void check_dispatch_names(const ComputeKernel* kernel, nb::dict vars, nb::kwargs kwargs)
{
    ref<const ProgramLayout> layout = kernel->program()->layout();
    ReflectionCursor globals(layout->globals_type_layout().get());
    ReflectionCursor entry_point(layout->get_entry_point_by_index(0));
    for (auto [key, value] : kwargs) {
        std::string name = nb::cast<std::string>(key);
        if (entry_point.has_field(name))
            continue;
        if (globals.has_field(name))
            SGL_THROW("\"{}\" is a global shader parameter, pass it in vars instead of as a keyword argument.", name);
        SGL_THROW("Entry point has no parameter named \"{}\".", name);
    }
    for (auto [key, value] : vars) {
        std::string name = nb::cast<std::string>(key);
        SGL_CHECK(globals.has_field(name), "Program has no global shader parameter named \"{}\".", name);
    }
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
               QueryPool* query_pool,
               uint32_t query_index_before,
               uint32_t query_index_after,
               nb::kwargs kwargs)
            {
                check_dispatch_names(self, vars, kwargs);
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
                    self->dispatch(
                        thread_count,
                        bind_vars,
                        command_encoder,
                        query_pool,
                        query_index_before,
                        query_index_after
                    );
                } else {
                    self->dispatch(
                        thread_count,
                        bind_vars,
                        queue,
                        cuda_stream,
                        query_pool,
                        query_index_before,
                        query_index_after
                    );
                }
            },
            "thread_count"_a,
            "vars"_a = nb::dict(),
            "command_encoder"_a = nullptr,
            "queue"_a = CommandQueueType::graphics,
            "cuda_stream"_a = NativeHandle(),
            "query_pool"_a.none() = nb::none(),
            "query_index_before"_a = 0,
            "query_index_after"_a = 0,
            "kwargs"_a,
            D(ComputeKernel, dispatch)
        );
}
