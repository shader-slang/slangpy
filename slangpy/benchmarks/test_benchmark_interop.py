# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from time import time
import pytest

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.benchmark import (
    BenchmarkPythonFunction,
    BenchmarkSlangFunction,
    BenchmarkComputeKernel,
    ReportFixture,
)

ADD_FLOATS = """
float add_floats(float a, float b) {
    return a + b;
}
"""

ADD_COMPUTE = """
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, StructuredBuffer<float> a, StructuredBuffer<float> b, RWStructuredBuffer<float> res) {
    res[tid.x] = a[tid.x] + b[tid.x];
}
"""

ADD_COMPUTE_4X = """
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, StructuredBuffer<float> a, StructuredBuffer<float> b, RWStructuredBuffer<float> res) {
    uint idx = tid.x*4;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
}
"""

ADD_COMPUTE_16X = """
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, StructuredBuffer<float> a, StructuredBuffer<float> b, RWStructuredBuffer<float> res) {
    uint idx = tid.x*16;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
    idx++;
    res[idx] = a[idx] + b[idx];
}
"""

ADD_COMPUTE_16X_LOOP = """
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, StructuredBuffer<float> a, StructuredBuffer<float> b, RWStructuredBuffer<float> res) {
    uint base = tid.x*16;
    for(uint i = 0; i < 16; i++) {
        uint idx = base + i;
        res[idx] = a[idx] + b[idx];
    }
}
"""

ADD_COMPUTE_16X_UNROLL = """
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, StructuredBuffer<float> a, StructuredBuffer<float> b, RWStructuredBuffer<float> res) {
    uint base = tid.x*16;
    [ForceUnroll]
    for(uint i = 0; i < 16; i++) {
        uint idx = base + i;
        res[idx] = a[idx] + b[idx];
    }
}
"""


@pytest.mark.skip(reason="Pytorch integration is affecting benchmarks")
def test_pytorch_tensor_addition_cpu(benchmark_python_function: BenchmarkPythonFunction):
    device = helpers.get_device(spy.DeviceType.cuda)

    BUFFER_SIZE = 10
    import torch

    buffer0 = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")
    buffer1 = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")

    def tensor_addition(a: torch.Tensor, b: torch.Tensor):
        res = a + b

    benchmark_python_function(device, tensor_addition, a=buffer0, b=buffer1)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compute_addition_noalloc_cpu(
    device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction
):
    device = helpers.get_device(device_type)

    BUFFER_SIZE = 10
    buffer0 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    buffer1 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    resbuffer = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)

    module = device.load_module_from_source("test_compute_addition_noalloc_cpu", ADD_COMPUTE)
    program = device.link_program([module], [module.entry_point("compute_main")])
    kernel = device.create_compute_kernel(program)

    def tensor_addition(a: spy.NDBuffer, b: spy.NDBuffer, res: spy.NDBuffer):
        kernel.dispatch(spy.uint3(BUFFER_SIZE, 1, 1), a=a.storage, b=b.storage, res=res.storage)

    benchmark_python_function(device, tensor_addition, a=buffer0, b=buffer1, res=resbuffer)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slangpy_tensor_addition_noalloc_cpu(
    device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction
):
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_floats", ADD_FLOATS)

    BUFFER_SIZE = 10
    buffer0 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    buffer1 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    resbuffer = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)

    def tensor_addition(a: spy.NDBuffer, b: spy.NDBuffer, res: spy.NDBuffer):
        func(a, b, _result=res)

    benchmark_python_function(device, tensor_addition, a=buffer0, b=buffer1, res=resbuffer)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slangpy_tensor_addition_alloc_cpu(
    device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction
):
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_floats", ADD_FLOATS)

    BUFFER_SIZE = 10
    buffer0 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    buffer1 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)

    def tensor_addition(a: spy.NDBuffer, b: spy.NDBuffer):
        func(a, b)

    benchmark_python_function(device, tensor_addition, a=buffer0, b=buffer1)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slangpy_tensor_addition_appendonly_cpu(
    device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction
):
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_floats", ADD_FLOATS)

    BUFFER_SIZE = 10
    buffer0 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    buffer1 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    resbuffer = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    ce = device.create_command_encoder()

    def tensor_addition(a: spy.NDBuffer, b: spy.NDBuffer, res: spy.NDBuffer):
        func(a, b, _result=res, _append_to=ce)

    benchmark_python_function(device, tensor_addition, a=buffer0, b=buffer1, res=resbuffer)


@pytest.mark.skip(reason="Pytorch integration is affecting benchmarks")
def test_pytorch_tensor_addition_gpu_est(report: ReportFixture):
    device = helpers.get_device(spy.DeviceType.cuda)

    BUFFER_SIZE = 65535 * 32
    import torch

    buffer0 = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")
    buffer1 = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")

    # Warmup + wait for cuda
    res = buffer0 + buffer1
    torch.cuda.synchronize()

    deltas = []

    start = time()
    for _ in range(100):
        istart = time()
        for _ in range(1000):
            res = buffer0 + buffer1
        torch.cuda.synchronize()
        iend = time()
        deltas.append((iend - istart) / 1000)

    end = time()

    report(device, [d * 1000 for d in deltas], end - start)  # convert to ms


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compute_addition_noalloc_gpu(
    device_type: spy.DeviceType, benchmark_compute_kernel: BenchmarkComputeKernel
):
    device = helpers.get_device(device_type)

    BUFFER_SIZE = 65535 * 32
    buffer0 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    buffer1 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    resbuffer = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)

    module = device.load_module_from_source("test_compute_addition_noalloc_gpu", ADD_COMPUTE)
    program = device.link_program([module], [module.entry_point("compute_main")])
    kernel = device.create_compute_kernel(program)

    benchmark_compute_kernel(
        device,
        kernel,
        spy.uint3(BUFFER_SIZE, 1, 1),
        a=buffer0.storage,
        b=buffer1.storage,
        res=resbuffer.storage,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compute_addition_4x_noalloc_gpu(
    device_type: spy.DeviceType, benchmark_compute_kernel: BenchmarkComputeKernel
):
    device = helpers.get_device(device_type)

    BUFFER_SIZE = 65535 * 32
    buffer0 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    buffer1 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    resbuffer = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)

    module = device.load_module_from_source("test_compute_addition_noalloc_gpu_4x", ADD_COMPUTE_4X)
    program = device.link_program([module], [module.entry_point("compute_main")])
    kernel = device.create_compute_kernel(program)

    benchmark_compute_kernel(
        device,
        kernel,
        spy.uint3(int(BUFFER_SIZE / 4), 1, 1),
        a=buffer0.storage,
        b=buffer1.storage,
        res=resbuffer.storage,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compute_addition_16x_noalloc_gpu(
    device_type: spy.DeviceType, benchmark_compute_kernel: BenchmarkComputeKernel
):
    device = helpers.get_device(device_type)

    BUFFER_SIZE = 65535 * 32
    buffer0 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    buffer1 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    resbuffer = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)

    module = device.load_module_from_source(
        "test_compute_addition_noalloc_gpu_16x", ADD_COMPUTE_16X
    )
    program = device.link_program([module], [module.entry_point("compute_main")])
    kernel = device.create_compute_kernel(program)

    benchmark_compute_kernel(
        device,
        kernel,
        spy.uint3(int(BUFFER_SIZE / 16), 1, 1),
        a=buffer0.storage,
        b=buffer1.storage,
        res=resbuffer.storage,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compute_addition_16xloop_noalloc_gpu(
    device_type: spy.DeviceType, benchmark_compute_kernel: BenchmarkComputeKernel
):
    device = helpers.get_device(device_type)

    BUFFER_SIZE = 65535 * 32
    buffer0 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    buffer1 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    resbuffer = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)

    module = device.load_module_from_source(
        "test_compute_addition_16xloop_noalloc_gpu", ADD_COMPUTE_16X_LOOP
    )
    program = device.link_program([module], [module.entry_point("compute_main")])
    kernel = device.create_compute_kernel(program)

    benchmark_compute_kernel(
        device,
        kernel,
        spy.uint3(int(BUFFER_SIZE / 16), 1, 1),
        a=buffer0.storage,
        b=buffer1.storage,
        res=resbuffer.storage,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compute_addition_16xunroll_noalloc_gpu(
    device_type: spy.DeviceType, benchmark_compute_kernel: BenchmarkComputeKernel
):
    device = helpers.get_device(device_type)

    BUFFER_SIZE = 65535 * 32
    buffer0 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    buffer1 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    resbuffer = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)

    module = device.load_module_from_source(
        "test_compute_addition_16xunroll_noalloc_gpu", ADD_COMPUTE_16X_UNROLL
    )
    program = device.link_program([module], [module.entry_point("compute_main")])
    kernel = device.create_compute_kernel(program)

    benchmark_compute_kernel(
        device,
        kernel,
        spy.uint3(int(BUFFER_SIZE / 16), 1, 1),
        a=buffer0.storage,
        b=buffer1.storage,
        res=resbuffer.storage,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slangpy_tensor_addition_noalloc_gpu(
    device_type: spy.DeviceType, benchmark_slang_function: BenchmarkSlangFunction
):
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_floats", ADD_FLOATS)

    BUFFER_SIZE = 65535 * 32
    buffer0 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    buffer1 = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)
    resbuffer = spy.NDBuffer.empty(device, shape=(BUFFER_SIZE,), dtype=float)

    benchmark_slang_function(device, func, a=buffer0, b=buffer1, _result=resbuffer)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.skip(reason="Memory leak")
def test_slangpy_addition_noalloc_interop_cpu(
    device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction
):
    device = helpers.get_torch_device(device_type)
    func = helpers.create_function_from_module(device, "add_floats", ADD_FLOATS)

    BUFFER_SIZE = 10
    import torch

    buffer0 = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")
    buffer1 = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")
    resbuffer = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")

    def tensor_addition(a: spy.NDBuffer, b: spy.NDBuffer, res: spy.NDBuffer):
        func(a, b, _result=res)

    benchmark_python_function(device, tensor_addition, a=buffer0, b=buffer1, res=resbuffer)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.skip(reason="Memory leak")
def test_slangpy_addition_alloc_interop_cpu(
    device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction
):
    device = helpers.get_torch_device(device_type)
    func = helpers.create_function_from_module(device, "add_floats", ADD_FLOATS)

    BUFFER_SIZE = 10
    import torch

    buffer0 = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")
    buffer1 = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")

    def tensor_addition(a: spy.NDBuffer, b: spy.NDBuffer):
        func(a, b)

    benchmark_python_function(device, tensor_addition, a=buffer0, b=buffer1)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.skip(reason="Memory leak")
def test_slangpy_addition_noalloc_interop_gpu_est(
    device_type: spy.DeviceType, report: ReportFixture
):
    device = helpers.get_torch_device(device_type)
    func = helpers.create_function_from_module(device, "add_floats", ADD_FLOATS)

    BUFFER_SIZE = 65535 * 32
    import torch

    a = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")
    b = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")
    res = torch.randn((BUFFER_SIZE,), dtype=torch.float32, device="cuda")

    # Warmup + wait for cuda
    func(a, b, _result=res)
    device.wait_for_idle()

    deltas = []

    start = time()
    for _ in range(10):
        istart = time()
        for _ in range(100):
            func(a, b, _result=res)
        device.wait_for_idle()
        iend = time()
        deltas.append((iend - istart) / 100)

    end = time()

    report(device, [d * 1000 for d in deltas], end - start)  # convert to ms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
