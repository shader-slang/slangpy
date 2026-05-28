# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import numpy as np
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent

device = spy.Device(
    enable_debug_layers=True,
    compiler_options={"include_paths": [EXAMPLE_DIR]},
)

vertices = np.array([-1, -1, 0, 1, -1, 0, 0, 1, 0], dtype=np.float32)
indices = np.array([0, 1, 2], dtype=np.uint32)

vertex_buffer = device.create_buffer(
    usage=spy.BufferUsage.shader_resource,
    label="vertex_buffer",
    data=vertices,
)

index_buffer = device.create_buffer(
    usage=spy.BufferUsage.shader_resource,
    label="index_buffer",
    data=indices,
)

blas_input_triangles = spy.AccelerationStructureBuildInputTriangles(
    {
        "vertex_buffers": [vertex_buffer],
        "vertex_format": spy.Format.rgb32_float,
        "vertex_count": vertices.size // 3,
        "vertex_stride": vertices.itemsize * 3,
        "index_buffer": index_buffer,
        "index_format": spy.IndexFormat.uint32,
        "index_count": indices.size,
        "flags": spy.AccelerationStructureGeometryFlags.opaque,
    }
)

blas_build_desc = spy.AccelerationStructureBuildDesc(
    {
        "inputs": [blas_input_triangles],
    }
)

blas_sizes = device.get_acceleration_structure_sizes(blas_build_desc)

blas_scratch_buffer = device.create_buffer(
    size=blas_sizes.scratch_size,
    usage=spy.BufferUsage.unordered_access,
    label="blas_scratch_buffer",
)

blas = device.create_acceleration_structure(
    kind=spy.AccelerationStructureKind.bottom_level,
    size=blas_sizes.acceleration_structure_size,
    label="blas",
)

command_encoder = device.create_command_encoder()
command_encoder.build_acceleration_structure(
    desc=blas_build_desc, dst=blas, src=None, scratch_buffer=blas_scratch_buffer
)
device.submit_command_buffer(command_encoder.finish())

instance_list = device.create_acceleration_structure_instance_list(1)
instance_list.write(
    0,
    {
        "transform": spy.float3x4.identity(),
        "instance_id": 0,
        "instance_mask": 0xFF,
        "instance_contribution_to_hit_group_index": 0,
        "flags": spy.AccelerationStructureInstanceFlags.none,
        "acceleration_structure": blas.handle,
    },
)

tlas_build_desc = spy.AccelerationStructureBuildDesc(
    {
        "inputs": [instance_list.build_input_instances()],
    }
)

tlas_sizes = device.get_acceleration_structure_sizes(tlas_build_desc)

tlas_scratch_buffer = device.create_buffer(
    size=tlas_sizes.scratch_size,
    usage=spy.BufferUsage.unordered_access,
    label="tlas_scratch_buffer",
)

tlas = device.create_acceleration_structure(
    kind=spy.AccelerationStructureKind.top_level,
    size=tlas_sizes.acceleration_structure_size,
    label="tlas",
)

command_encoder = device.create_command_encoder()
command_encoder.build_acceleration_structure(
    desc=tlas_build_desc, dst=tlas, src=None, scratch_buffer=tlas_scratch_buffer
)
device.submit_command_buffer(command_encoder.finish())

render_texture = device.create_texture(
    format=spy.Format.rgba32_float,
    width=1024,
    height=1024,
    usage=spy.TextureUsage.unordered_access,
    label="render_texture",
)

program = device.load_program("raytracing.slang", ["main"])
kernel = device.create_compute_kernel(program)

kernel.dispatch(
    thread_count=[1024, 1024, 1],
    vars={
        "tlas": tlas,
        "render_texture": render_texture,
    },
)

spy.tev.show(render_texture, "raytracing")
