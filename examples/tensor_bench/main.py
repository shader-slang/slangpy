# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from typing import Tuple
import slangpy as spy
import numpy as np
from pathlib import Path
import json

DIR = Path(__file__).parent.resolve()

BACKENDS = {
    "d3d12": spy.DeviceType.d3d12,
    "vulkan": spy.DeviceType.vulkan,
    "cuda": spy.DeviceType.cuda,
}


class Tensor:
    def __init__(self, device: spy.Device, shape: Tuple[int, int]):
        super().__init__()
        self.shape = shape
        self.offset = 0
        self.strides = (shape[1], 1)
        self.buffer = device.create_buffer(
            shape[1] * shape[0] * 4,
            usage=spy.BufferUsage.shader_resource,
        )

    def uniforms(self):
        return {
            "buffer": self.buffer,
            "layout": {
                "offset": self.offset,
                "strides": self.strides,
            },
            "shape": self.shape,
            # "offset": self.offset,
            # "strides": self.strides,
        }


class RWTensor:
    def __init__(self, device: spy.Device, shape: Tuple[int, int]):
        super().__init__()
        self.shape = shape
        self.offset = 0
        self.strides = (shape[1], 1)
        self.buffer = device.create_buffer(
            shape[1] * shape[0] * 4,
            usage=spy.BufferUsage.unordered_access,
        )

    def uniforms(self):
        return {
            "buffer": self.buffer,
            "layout": {
                "offset": self.offset,
                "strides": self.strides,
            },
            "shape": self.shape,
            # "offset": self.offset,
            # "strides": self.strides,
        }


def run(
    device_type: spy.DeviceType,
    input_tensor_count: int,
    index_mode: str,
    read_pattern: str,
    read_mode: str,
    shape: Tuple[int, int] = (1024, 1024),
):
    device = spy.Device(
        type=device_type,
        enable_debug_layers=True,
        enable_compilation_reports=True,
        compiler_options={
            "include_paths": [DIR],
            "defines": {
                "INPUT_TENSOR_COUNT": str(input_tensor_count),
                "INDEX_MODE": index_mode,
                "READ_PATTERN": read_pattern,
                "READ_MODE": read_mode,
            },
            "dump_intermediates": True,
        },
    )

    ITERATIONS = 100

    query_pool = device.create_query_pool(type=spy.QueryType.timestamp, count=2)

    program = device.load_program("test.slang", entry_point_names=["compute_main"])
    kernel = device.create_compute_kernel(program)

    input_tensors = [Tensor(device, shape) for _ in range(input_tensor_count)]
    output_tensor = RWTensor(device, shape)
    index_count = 100
    indices = np.linspace(0, index_count - 1, index_count, dtype=np.uint32) % input_tensor_count
    index_buffer = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=indices)

    deltas = np.zeros(ITERATIONS, dtype=np.float32)
    for i in range(ITERATIONS):
        command_encoder = device.create_command_encoder()
        command_encoder.write_timestamp(query_pool, 0)
        kernel.dispatch(
            thread_count=(shape[0], shape[1], 1),
            vars={"data":{
                "input": [tensor.uniforms() for tensor in input_tensors],
                "output": output_tensor.uniforms(),
                "input_tensor_count": input_tensor_count,
                "index_buffer": index_buffer,
                "index_count": index_count,
            }},
            command_encoder=command_encoder,
        )
        command_encoder.write_timestamp(query_pool, 1)
        device.submit_command_buffer(command_encoder.finish())
        device.wait()
        queries = query_pool.get_results(0, 2)
        frequency = float(device.info.timestamp_frequency)
        deltas[i] = (queries[1] - queries[0]) / frequency * 1000.0
    # print(deltas)
    # remove 50% of the highest and lowest values
    deltas.sort()
    deltas = deltas[int(ITERATIONS * 0.25) : int(ITERATIONS * 0.75)]
    print(f"Min time: {np.min(deltas):.3f} ms")
    print(f"Max time: {np.max(deltas):.3f} ms")
    avg_time = np.mean(deltas)
    print(f"Average time: {avg_time:.3f} ms")

    device.close()

    return float(avg_time)


if False:
    run(
        spy.DeviceType.vulkan,
        10,
        "INDEX_MODE_VECTOR",
        "READ_PATTERN_SINGLE",
        "READ_MODE_INDIRECT",
        (1024, 1024),
    )
    sys.exit(0)


results = {}

for name, device_type in BACKENDS.items():
    print(f"Running on {name} backend")
    results[name] = {}
    for input_tensor_count in [1, 2, 4, 8, 16, 32]:
        time = run(
            device_type,
            input_tensor_count,
            "INDEX_MODE_VECTOR",
            "READ_PATTERN_SINGLE",
            "READ_MODE_INDIRECT",
            (1024, 1024),
        )
        print(f"N={input_tensor_count}, Time: {time:.3f} ms")
        results[name][input_tensor_count] = time

print(json.dumps(results, indent=4))
