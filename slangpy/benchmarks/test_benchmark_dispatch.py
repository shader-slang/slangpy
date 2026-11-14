# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CPU cost of binding names in ComputeKernel.dispatch: a trivial kernel recorded into a command
# encoder, so the time is binding + recording. "declared" globals in the shader, "bound" of them
# passed in vars, plus one entry-point uniform passed as a keyword argument.

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.benchmark import BenchmarkPythonFunction

ITERATIONS = 10
SUB_ITERATIONS = 2000
WARMUPS = 10

# (declared, bound)
CASES = [(1, 1), (4, 4), (16, 16), (64, 64), (64, 4)]


def make_source(declared: int):
    lines = [
        "struct Inner { float a; float3 b; };",
        "struct Outer { Inner inner; float c; };",
        "RWStructuredBuffer<float> out_buf;",
    ]
    names = ["out_buf"]
    expr = ["0.0"]
    if declared >= 4:
        lines.append("uniform Outer s;")
        names.append("s")
        expr.append("s.inner.a + s.inner.b.y + s.c")
    for i in range(declared - len(names)):
        lines.append(f"uniform float g{i};")
        names.append(f"g{i}")
        expr.append(f"g{i}")
    lines += [
        '[shader("compute")]',
        "[numthreads(1, 1, 1)]",
        "void compute_main(uint3 tid: SV_DispatchThreadID, uniform float scale)",
        "{",
        f"    out_buf[tid.x] = ({' + '.join(expr)}) * scale;",
        "}",
    ]
    return "\n".join(lines), names


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("declared,bound", CASES)
def test_dispatch_bind_names(
    device_type: spy.DeviceType,
    declared: int,
    bound: int,
    benchmark_python_function: BenchmarkPythonFunction,
):
    device = helpers.get_device(type=device_type)
    source, names = make_source(declared)
    module = device.load_module_from_source(f"bench_dispatch_{declared}", source)
    kernel = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("compute_main")])
    )
    buffer = device.create_buffer(
        element_count=4,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(4, dtype=np.float32),
    )
    vars = {}
    for name in names[:bound]:
        if name == "out_buf":
            vars[name] = buffer
        elif name == "s":
            vars[name] = {"inner": {"a": 1.0, "b": spy.float3(1, 2, 3)}, "c": 2.0}
        else:
            vars[name] = 1.0
    thread_count = spy.uint3(1, 1, 1)
    encoder = device.create_command_encoder()

    def dispatch():
        kernel.dispatch(thread_count, vars=vars, command_encoder=encoder, scale=2.0)

    benchmark_python_function(
        device,
        dispatch,
        iterations=ITERATIONS,
        warmup_iterations=WARMUPS,
        sub_iterations=SUB_ITERATIONS,
    )
    device.submit_command_buffer(encoder.finish())
    device.wait()
