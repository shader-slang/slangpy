// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/kernel.h"

using namespace sgl;

TEST_SUITE_BEGIN("simple_gpu");

/// Simple GPU test example that demonstrates basic compute shader functionality.
/// This test:
/// 1. Creates a compute shader from source
/// 2. Creates a buffer with test data
/// 3. Dispatches the shader to modify the data on the GPU
/// 4. Reads back the results and verifies correctness
TEST_CASE_GPU("add_one")
{
    // Define a simple compute shader that adds 1 to each element in a buffer.
    // The shader uses Slang syntax and runs with 64 threads per workgroup.
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

    // Create and load the shader program from source code.
    // The program is identified by name "simple_gpu" and has one entry point "main".
    ref<ShaderProgram> program = ctx.device->load_program_from_source("simple_gpu", shader_source, {"main"});
    CHECK(program);

    // Create a compute kernel from the shader program.
    // The kernel is used to dispatch the shader on the GPU.
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(kernel);

    // Create input data: an array containing values from 0 to 1023.
    std::vector<uint32_t> input_data(1024);
    for (uint32_t i = 0; i < 1024; ++i) {
        input_data[i] = i;
    }

    // Create a GPU buffer and initialize it with the input data.
    // The buffer is accessible for both reading and writing from shaders.
    ref<Buffer> buffer = ctx.device->create_buffer({
        .element_count = 1024,
        .struct_size = sizeof(uint32_t),
        .usage = BufferUsage::shader_resource | BufferUsage::unordered_access,
        .data = input_data.data(),
        .data_size = input_data.size() * sizeof(uint32_t),
    });
    CHECK(buffer);

    // Dispatch the compute kernel to execute on the GPU.
    // We use 16 workgroups of 64 threads each (16 * 64 = 1024 total threads).
    kernel->dispatch(
        uint3(16, 1, 1),
        [&buffer](ShaderCursor cursor)
        {
            // Bind the buffer to the shader's "buffer" parameter.
            cursor["buffer"] = buffer;
        }
    );

    // Read back the results from the GPU buffer to CPU memory.
    std::vector<uint32_t> output_data(1024);
    buffer->get_data(output_data.data(), output_data.size() * sizeof(uint32_t));

    // Verify that each element was incremented by 1 as expected.
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
