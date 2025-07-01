from typing import Tuple
import slangpy as spy
import numpy as np
from pathlib import Path

DIR = Path(__file__).parent.resolve()

BACKENDS = {
    "d3d12": spy.DeviceType.d3d12,
    "vulkan": spy.DeviceType.vulkan,
    "cuda": spy.DeviceType.cuda,
}


class Tensor:
    def __init__(self, device: spy.Device, shape: Tuple[int, int, int]):
        super().__init__()
        self.shape = shape
        self.offset = 0
        self.strides = (shape[1] * shape[2], shape[2], 1)
        self.buffer = device.create_buffer(
            size=shape[2] * shape[1] * shape[0] * 4,
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
    def __init__(self, device: spy.Device, shape: Tuple[int, int, int]):
        super().__init__()
        self.shape = shape
        self.offset = 0
        self.strides = (shape[1] * shape[2], shape[2], 1)
        self.buffer = device.create_buffer(
            size=shape[2] * shape[1] * shape[0] * 4,
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
    access_mode: str,
    shape: Tuple[int, int, int] = (1024, 1024, 3),
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
                "ACCESS_MODE": access_mode,
            },
        },
    )

    ITERATIONS = 100

    query_pool = device.create_query_pool(type=spy.QueryType.timestamp, count=2)

    program = device.load_program("test.slang", entry_point_names=["compute_main"])
    kernel = device.create_compute_kernel(program)

    input_tensors = [Tensor(device, shape) for _ in range(input_tensor_count)]
    output_tensor = RWTensor(device, shape)

    deltas = np.zeros(ITERATIONS, dtype=np.float32)
    for i in range(ITERATIONS):
        command_encoder = device.create_command_encoder()
        command_encoder.write_timestamp(query_pool, 0)
        kernel.dispatch(
            thread_count=(shape[0], shape[1], 1),
            data={
                "input": [tensor.uniforms() for tensor in input_tensors],
                "output": output_tensor.uniforms(),
            },
            command_encoder=command_encoder,
        )
        command_encoder.write_timestamp(query_pool, 1)
        device.submit_command_buffer(command_encoder.finish())
        device.wait()
        queries = query_pool.get_results(0, 2)
        frequency = float(device.info.timestamp_frequency)
        deltas[i] = (queries[1] - queries[0]) / frequency * 1000.0
    # print(deltas)
    avg_time = np.mean(deltas)
    print(f"Average time: {avg_time:.3f} ms")

    device.wait()
    device.close()


run(spy.DeviceType.vulkan, 5, "INDEX_MODE_ARRAY", "ACCESS_MODE_REGION", (1024, 1024, 3))

# for name, device_type in BACKENDS.items():
#     for input_tensor_count in [1, 2, 4, 8]:
#         print(f"Running on {name} backend")
#         run(device_type)
#         print(f"Finished running on {name} backend\n")
