# SPDX-License-Identifier: Apache-2.0
import math
from pathlib import Path
from typing import Any
import sgl
import slangpy as spy
import numpy as np
from timeit import timeit
from time import sleep, time
import torch

SHADER_DIR = Path(__file__).parent


def raw_compute_test(compute_kernel: sgl.ComputeKernel, a: spy.NDBuffer, b: spy.NDBuffer, res: spy.NDBuffer, threads: int):
    compute_kernel.dispatch(sgl.uint3(threads, 1, 1), vars={
        'addKernelData': {
            "a": a.storage,
            "b": b.storage,
            "res": res.storage,
            "count": threads
        }
    })


def raw_queue_kernel_test(compute_kernel: sgl.ComputeKernel, a: spy.NDBuffer, b: spy.NDBuffer, res: spy.NDBuffer, threads: int, cb: sgl.CommandBuffer):
    compute_kernel.dispatch(sgl.uint3(threads, 1, 1), vars={
        'addKernelData': {
            "a": a.storage,
            "b": b.storage,
            "res": res.storage,
            "count": threads
        }
    }, command_buffer=cb)


def perf_test(name: str, device: sgl.Device, func: Any):
    device.wait_for_idle()
    res = timeit(func, number=1000)
    device.wait_for_idle()
    av = 1000 * res/1000
    print(f"{name}: {av}ms")
    return av


def run():
    device = spy.create_device(sgl.DeviceType.d3d12, include_paths=[SHADER_DIR])

    sgl_module = device.load_module("performance.slang")
    sgl_program = device.link_program([sgl_module], [sgl_module.entry_point("addkernel")])

    spy_module = spy.Module.load_from_module(device, sgl_module)

    compute_kernel = device.create_compute_kernel(sgl_program)

    a = spy.NDBuffer(device, spy_module.float, 10000000)
    b = spy.NDBuffer(device, spy_module.float, 10000000)
    res = spy.NDBuffer(device, spy_module.float, 10000000)

    a_data = np.random.rand(10000000).astype(np.float32)
    b_data = np.random.rand(10000000).astype(np.float32)
    a.from_numpy(a_data)
    b.from_numpy(b_data)

    raw_compute_test(compute_kernel, a, b, res, 1000)
    res_data = res.to_numpy().view(dtype=np.float32)[0:1000]
    expected = (a_data + b_data)[0:1000]
    assert np.allclose(res_data, expected)

    a_small = spy.NDBuffer(device, spy_module.float, 1000)
    b_small = spy.NDBuffer(device, spy_module.float, 1000)
    res_small = spy.NDBuffer(device, spy_module.float, 1000)
    a.from_numpy(a_data[0:1000])
    b.from_numpy(b_data[0:1000])

    # Ensure compilation of spy module
    spy_module.add(a=a, b=b, _result=res)
    spy_module.add(a=a_small, b=b_small, _result=res_small)

    sql_dispatch = perf_test("Small direct dispatch", device,
                             lambda: raw_compute_test(compute_kernel, a, b, res, 1000))

    perf_test("Large direct dispatch", device, lambda: raw_compute_test(
        compute_kernel, a, b, res, 10000000))

    command_buffer = device.create_command_buffer()
    sgl_queue = perf_test("Queue dispatch", device, lambda: raw_queue_kernel_test(
        compute_kernel, a, b, res, 10000000, command_buffer))
    command_buffer.submit()
    device.wait_for_idle()

    perf_test("Spy small invoke", device, lambda: spy_module.add(
        a=a_small, b=b_small, _result=res_small))

    perf_test("Spy large invoke", device, lambda: spy_module.add(a=a, b=b, _result=res))

    command_buffer = device.create_command_buffer()
    spy_queue = perf_test("Spy append", device, lambda: spy_module.add.append_to(
        command_buffer, a=a, b=b, _result=res))
    command_buffer.submit()
    device.wait_for_idle()

    perf_test("Spy large dispatch", device, lambda: spy_module.addkernel.dispatch(sgl.uint3(10000000, 1, 1), vars={
        'addKernelData': {
            "a": a.storage,
            "b": b.storage,
            "res": res.storage,
            "count": 10000000
        }
    }))

    # read a character to wait for quit
    input("Press Enter to quit")


def run_for_profiling():
    device = spy.create_device(sgl.DeviceType.d3d12, include_paths=[SHADER_DIR])

    sgl_module = device.load_module("performance.slang")
    sgl_add_program = device.link_program([sgl_module], [sgl_module.entry_point("addkernel")])
    sgl_add_with_shapes_program = device.link_program(
        [sgl_module], [sgl_module.entry_point("addkernelWithShapes")])

    spy_module = spy.Module.load_from_module(device, sgl_module)

    add_kernel = device.create_compute_kernel(sgl_add_program)
    add_with_shapes_kernel = device.create_compute_kernel(sgl_add_with_shapes_program)

    size = 1024

    a_data = np.random.rand(size).astype(np.float32)
    b_data = np.random.rand(size).astype(np.float32)
    a_small = spy.NDBuffer(device, spy_module.float, size)
    a_small.from_numpy(a_data)
    b_small = spy.NDBuffer(device, spy_module.float, size)
    b_small.from_numpy(b_data)
    res_small = spy.NDBuffer(device, spy_module.float, size)

    a_texture = device.create_texture(format=sgl.Format.r32_float,
                                      width=size, usage=sgl.ResourceUsage.shader_resource)
    a_texture.from_numpy(a_data)
    b_texture = device.create_texture(format=sgl.Format.r32_float,
                                      width=size, usage=sgl.ResourceUsage.shader_resource)
    b_texture.from_numpy(b_data)
    res_texture = device.create_texture(format=sgl.Format.r32_float, width=size,
                                        usage=sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access)

    # Ensure compilation of spy module
    spy_module.add(a=a_small, b=b_small, _result=res_small)

    # Pre-configure version for texture that explicitly passes in float1
    float_type = spy_module.float1
    spy_module.add.map(a=float_type, b=float_type, _result=float_type)(
        a=a_texture, b=b_texture, _result=res_texture)

    input("Press Enter to start")

    direct_dispatch = 0
    direct_dispatch_2 = 0
    spy_append = 0
    spy_complex_append = 0
    spy_tex_append = 0
    iterations = 100000
    interval = 0.1

    device.wait_for_idle()

    # Bare bones append
    if False:
        command_buffer = device.create_command_buffer()

        def add_command():
            add_kernel.dispatch(sgl.uint3(32, 1, 1), vars={
                'addKernelData': {
                    "a": a_small.storage,
                    "b": b_small.storage,
                    "res": res_small.storage,
                    "count": 32
                }
            }, command_buffer=command_buffer)

        start = time()
        for i in range(0, iterations):
            add_command()
        direct_dispatch = time() - start

        command_buffer.submit()
        device.wait_for_idle()

        sleep(interval)

    # SGL ND buffer append
    if False:
        command_buffer = device.create_command_buffer()

        def add_shapes_command():
            add_with_shapes_kernel.dispatch(sgl.uint3(32, 1, 1), vars={
                'addKernelWithShapesData': {
                    "a": a_small.uniforms(),
                    "b": b_small.uniforms(),
                    "res": res_small.uniforms(),
                    "count": 32
                }
            }, command_buffer=command_buffer)

        start = time()
        for i in range(0, iterations):
            add_shapes_command()
        direct_dispatch_2 = time() - start

        command_buffer.submit()
        device.wait_for_idle()

        sleep(interval)

    # SlangPy append
    if False:
        command_buffer = device.create_command_buffer()

        def sp_command():
            spy_module.add.append_to(command_buffer, a=a_small, b=b_small, _result=res_small)

        start = time()
        for i in range(0, iterations):
            sp_command()
        spy_append = time() - start

        command_buffer.submit()
        device.wait_for_idle()

        sleep(interval)

    # SlangPy complex append
    if False:
        command_buffer = device.create_command_buffer()

        def comp_command():
            spy_module.add \
                .map(a=(0,), b=(0,), _result=(0,)) \
                .set({}) \
                .constants({'x': 10}) \
                .append_to(command_buffer, a=a_small, b=b_small, _result=res_small)

        start = time()
        for i in range(0, iterations):
            comp_command()
        spy_complex_append = time() - start

        command_buffer.submit()
        device.wait_for_idle()

        sleep(interval)

    # SlangPy texture append
    if True:
        command_buffer = device.create_command_buffer()

        def tex_command():
            spy_module.add.map(a=float_type, b=float_type, _result=float_type).append_to(
                command_buffer, a=a_texture, b=b_texture, _result=res_texture)

        start = time()
        for i in range(0, iterations):
            tex_command()
        spy_tex_append = time() - start

        command_buffer.submit()
        device.wait_for_idle()

        sleep(interval)

    print(f"types=NDBuffer[float,1] func=add, its={iterations}:")
    print(f"  Bare bones:       {direct_dispatch}")
    print(f"  SGL:              {direct_dispatch_2}")
    print(f"  SlangPy:          {spy_append}")
    print(f"  SlangPy Complex:  {spy_complex_append}")
    print(f"  SlangPy Texture:  {spy_tex_append}")

    sleep(0.25)


def run_torch_comparison():
    device = spy.create_device(sgl.DeviceType.d3d12, include_paths=[SHADER_DIR])
    sgl_module = device.load_module("performance.slang")
    sgl_inc_with_shapes_program = device.link_program(
        [sgl_module], [sgl_module.entry_point("incrementkernelWithShapes")])

    add_with_shapes_kernel = device.create_compute_kernel(sgl_inc_with_shapes_program)

    spy_module = spy.Module.load_from_module(device, sgl_module)
    spy_torch_module = spy.TorchModule.load_from_module(device, sgl_module)

    buffer_size = 1000000
    iterations = 10000

    val = spy.NDBuffer(device, spy_module.float, buffer_size)
    total = spy.NDBuffer(device, spy_module.float, buffer_size)

    val_data = np.random.rand(buffer_size).astype(np.float32)
    total_data = np.zeros_like(val_data)

    val_tensor = torch.tensor(val_data, device='cuda')
    total_tensor = torch.tensor(total_data, device='cuda')

    input("Press Enter to start")

    start = time()
    if False:
        for i in range(0, iterations):
            add_with_shapes_kernel.dispatch(sgl.uint3(buffer_size, 1, 1), vars={
                'incrementKernelWithShapesData': {
                    "val": val.uniforms(),
                    "total": total.uniforms(),
                    "count": buffer_size
                }
            })
    direct_dispatch = time() - start
    device.wait_for_idle()

    sleep(1)

    start = time()
    if True:
        for i in range(0, iterations):
            spy_module.increment(val=val, total=total)
    spy_append = time() - start

    sleep(1)

    start = time()
    for i in range(0, iterations):
        spy_torch_module.increment(val=val_tensor, total=total_tensor)
    spy_torch_append = time() - start

    sleep(1)
    start = time()
    if False:
        for i in range(0, iterations):
            for j in range(0, 64):
                total_tensor += val_tensor
            # cpu = total_tensor.cpu()
    torch_add = time() - start

    sleep(1)

    print(f"Direct dispatch: {direct_dispatch}")
    print(f"Spy dispatch:    {spy_append}")
    print(f"Spy torch:       {spy_torch_append}")
    print(f"Torch add:       {torch_add}")


def run_for_sig_test():
    device = spy.create_device(sgl.DeviceType.d3d12, include_paths=[SHADER_DIR])

    sgl_module = device.load_module("performance.slang")

    spy_module = spy.Module.load_from_module(device, sgl_module)

    a_data = np.random.rand(1000).astype(np.float32)
    b_data = np.random.rand(1000).astype(np.float32)

    a_small = spy.NDBuffer(device, spy_module.float, 1000)
    a_small.from_numpy(a_data)
    b_small = spy.NDBuffer(device, spy_module.float, 1000)
    b_small.from_numpy(b_data)
    res_small = spy.NDBuffer(device, spy_module.float, 1000)

    iterations = 5

    # Ensure compilation of spy module
    spy_module.add(a=a_small, b=10.0, _result=res_small)

    # Run perf test
    command_buffer = device.create_command_buffer()
    start = time()
    for i in range(0, iterations):
        spy_module.add.map().append_to(command_buffer, a=a_small, b=b_small, _result=res_small)
    spy_append = time() - start
    command_buffer.submit()
    device.wait_for_idle()

    print(f"SlangPy:  {spy_append}")


if __name__ == "__main__":

    run_for_profiling()
