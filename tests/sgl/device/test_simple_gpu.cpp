// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/kernel.h"

using namespace sgl;

TEST_SUITE_BEGIN("simple_gpu");

TEST_CASE_GPU("add_one")
{
    // Simple compute shader that adds 1 to each element in a buffer
    const char* shader_source = R"(
RWStructuredBuffer<uint> buffer;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < 1024)
        buffer[tid.x] = buffer[tid.x] + 1;
}
)";

    // Create and load the shader program
    ref<ShaderProgram> program = ctx.device->load_program_from_source("simple_gpu", shader_source, {"main"});
    CHECK(program);

    // Create a compute kernel
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(kernel);

    // Create input data (0, 1, 2, ..., 1023)
    std::vector<uint32_t> input_data(1024);
    for (uint32_t i = 0; i < 1024; ++i) {
        input_data[i] = i;
    }

    // Create buffer with input data
    ref<Buffer> buffer = ctx.device->create_buffer({
        .element_count = 1024,
        .struct_size = sizeof(uint32_t),
        .usage = BufferUsage::shader_resource | BufferUsage::unordered_access,
        .data = input_data.data(),
        .data_size = input_data.size() * sizeof(uint32_t),
    });
    CHECK(buffer);

    // Dispatch the compute kernel
    kernel->dispatch(
        uint3(16, 1, 1), // 16 * 64 = 1024 threads
        [&buffer](ShaderCursor cursor)
        {
            cursor["buffer"] = buffer;
        }
    );

    // Read back results
    std::vector<uint32_t> output_data(1024);
    buffer->get_data(output_data.data(), output_data.size() * sizeof(uint32_t));

    // Verify results: each element should be input + 1
    bool all_correct = true;
    for (uint32_t i = 0; i < 1024; ++i) {
        if (output_data[i] != i + 1) {
            all_correct = false;
            break;
        }
    }
    CHECK(all_correct);
}

TEST_SUITE_END();
