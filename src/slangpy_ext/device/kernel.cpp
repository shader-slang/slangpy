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

/// Bind each (name, value) of a dict to the field of that name under `cursor`.
/// Writing a dict to a cursor walks the shader's fields and skips names the shader does not declare,
/// so a global passed as a keyword argument (or a typo) would leave the resource unbound. Walking the
/// dict instead makes an unknown name a lookup miss binding already does, so reporting it costs nothing.
template<typename OnMiss>
inline void bind_named_vars(const ShaderCursor& cursor, nb::dict vars, OnMiss&& on_miss)
{
    for (auto [key, value] : vars) {
        Py_ssize_t size;
        const char* data = PyUnicode_AsUTF8AndSize(key.ptr(), &size);
        if (!data)
            throw nb::python_error();
        std::string_view name(data, size);
        ShaderCursor field = cursor.find_field(name);
        if (!field.is_valid())
            on_miss(name);
        try {
            bind_python_var(field, value);
        } catch (const std::exception& err) {
            // Keep the "name.path: message" form the dict writer produces.
            std::string_view msg = err.what();
            SGL_THROW("{}{}{}", name, msg.starts_with(':') ? "" : ".", msg);
        }
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
                auto bind_vars = [&](ShaderCursor cursor)
                {
                    // bind locals
                    if (kwargs.size() > 0) {
                        bind_named_vars(
                            cursor.find_entry_point(0),
                            kwargs,
                            [&](std::string_view name)
                            {
                                if (cursor.find_field(name).is_valid())
                                    SGL_THROW(
                                        "\"{}\" is a global shader parameter, pass it in vars instead of as a "
                                        "keyword argument.",
                                        name
                                    );
                                SGL_THROW("Entry point has no parameter named \"{}\".", name);
                            }
                        );
                    }
                    // bind globals
                    bind_named_vars(
                        cursor,
                        vars,
                        [](std::string_view name)
                        {
                            SGL_THROW("Program has no global shader parameter named \"{}\".", name);
                        }
                    );
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
