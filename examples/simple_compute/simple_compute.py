# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import numpy as np
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent

device = spy.Device(
    enable_debug_layers=True,
    compiler_options={"include_paths": [EXAMPLE_DIR]},
)

program = device.load_program("simple_compute.slang", ["compute_main"])
kernel = device.create_compute_kernel(program)

N = 1024

buffer_a = device.create_buffer(
    element_count=N,
    resource_type_layout=kernel.reflection.processor.a,
    usage=spy.BufferUsage.shader_resource,
    data=np.linspace(0, N - 1, N, dtype=np.uint32),
)

buffer_b = device.create_buffer(
    element_count=N,
    resource_type_layout=kernel.reflection.processor.b,
    usage=spy.BufferUsage.shader_resource,
    data=np.linspace(N, 1, N, dtype=np.uint32),
)

buffer_c = device.create_buffer(
    element_count=N,
    resource_type_layout=kernel.reflection.processor.c,
    usage=spy.BufferUsage.unordered_access,
)

if True:
    # Method 1: Manual command encoding
    command_encoder = device.create_command_encoder()
    with command_encoder.begin_compute_pass() as pass_encoder:
        shader_object = pass_encoder.bind_pipeline(kernel.pipeline)
        processor = spy.ShaderCursor(shader_object)["processor"]
        processor["a"] = buffer_a
        processor["b"] = buffer_b
        processor["c"] = buffer_c
        pass_encoder.dispatch([N, 1, 1])
    device.submit_command_buffer(command_encoder.finish())

    result = buffer_c.to_numpy().view(np.uint32)
    print(result)

if True:
    # Method 2: Use compute kernel dispatch
    kernel.dispatch(
        thread_count=[N, 1, 1],
        vars={"processor": {"a": buffer_a, "b": buffer_b, "c": buffer_c}},
    )

    result = buffer_c.to_numpy().view(np.uint32)
    print(result)

if True:
    # Method 3: Use shader object
    processor_object = device.create_shader_object(kernel.reflection["processor"])
    processor = spy.ShaderCursor(processor_object)
    processor.a = buffer_a
    processor.b = buffer_b
    processor.c = buffer_c

    kernel.dispatch(
        thread_count=[N, 1, 1],
        vars={"processor": processor_object},
    )

    result = buffer_c.to_numpy().view(np.uint32)
    print(result)
