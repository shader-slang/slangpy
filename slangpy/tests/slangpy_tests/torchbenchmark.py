# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import numpy as np
from pathlib import Path
import torch
from time import time
from typing import cast


SHADER_DIR = Path(__file__).parent


def run_benchmark(mode: int, buffer_size: int, iterations: int):
    device = spy.create_device(spy.DeviceType.d3d12, include_paths=[SHADER_DIR])
    program = device.load_program("performance.slang", ["addkernel"])
    kernel = device.create_compute_kernel(program)

    buffer_a = device.create_buffer(
        element_count=1024,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.shared,
    )
    buffer_b = device.create_buffer(
        element_count=1024,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.shared,
    )
    buffer_res = device.create_buffer(
        element_count=1024,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access
        | spy.BufferUsage.shared,
    )

    buffer_a.copy_from_numpy(np.random.rand(1024).astype(np.float32))
    buffer_b.copy_from_numpy(np.random.rand(1024).astype(np.float32))
    buffer_res.copy_from_numpy(np.zeros(1024).astype(np.float32))

    tensor_a = torch.zeros(buffer_size, dtype=torch.float32, device="cuda")
    tensor_b = torch.zeros(buffer_size, dtype=torch.float32, device="cuda")
    tensor_res = torch.zeros(buffer_size, dtype=torch.float32, device="cuda")

    temp_tensor_a = cast(
        torch.Tensor, buffer_a.to_torch(type=spy.DataType.float32, shape=(buffer_size,))
    )
    temp_tensor_b = cast(
        torch.Tensor, buffer_b.to_torch(type=spy.DataType.float32, shape=(buffer_size,))
    )
    temp_tensor_res = cast(
        torch.Tensor,
        buffer_res.to_torch(type=spy.DataType.float32, shape=(buffer_size,)),
    )

    def kernel_only_func():
        kernel.dispatch(
            spy.uint3(1024, 1, 1),
            vars={
                "a": buffer_a,
                "b": buffer_b,
                "res": buffer_res,
                "count": buffer_size,
            },
        )

    def kernel_and_sync_func():
        device.sync_to_cuda()
        kernel_only_func()
        device.sync_to_device()

    def kernel_and_copysync_func():
        temp_tensor_a.copy_(tensor_a)
        temp_tensor_b.copy_(tensor_b)
        kernel_and_sync_func()
        tensor_res.copy_(temp_tensor_res)

    total = 0
    if mode == 0:
        print(f"Running kernel only sync with buffersize={buffer_size}, iterations={iterations}")
        start = time()
        for i in range(iterations):
            kernel_only_func()
        total = time() - start
    elif mode == 1:
        print(f"Running kernel and sync with buffersize={buffer_size}, iterations={iterations}")
        start = time()
        for i in range(iterations):
            kernel_and_sync_func()
        total = time() - start
    elif mode == 2:
        print(f"Running kernel and copysync with buffersize={buffer_size}, iterations={iterations}")
        start = time()
        for i in range(iterations):
            kernel_and_copysync_func()
        total = time() - start

    print(f"Total: {total}s")
    print(f"Average: {1000.0*total/iterations}ms per iteration")


if __name__ == "__main__":

    # read params from commandline
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark for kernel execution")
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="0: kernel only, 1: kernel and sync, 2: kernel and copysync",
    )
    parser.add_argument("--buffer_size", type=int, default=1024, help="Size of buffer")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of iterations")
    parser.add_argument("--wait", action="store_true", help="Wait for user input before exit")

    args = parser.parse_args()

    if args.wait:
        input("Press Enter to continue...")

    run_benchmark(args.mode, args.buffer_size, args.iterations)
