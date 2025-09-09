# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy import DeviceType, Module
from slangpy.types.tensor import Tensor
from slangpy.testing import helpers


def build_blas(
    device: spy.Device, vertices: np.ndarray, indices: np.ndarray
) -> spy.AccelerationStructure:
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
        size=blas_sizes.acceleration_structure_size,
        label="blas",
    )

    command_encoder = device.create_command_encoder()
    command_encoder.build_acceleration_structure(
        desc=blas_build_desc, dst=blas, src=None, scratch_buffer=blas_scratch_buffer
    )
    device.submit_command_buffer(command_encoder.finish())

    return blas


def build_tlas(
    device: spy.Device, instance_list: spy.AccelerationStructureInstanceList
) -> spy.AccelerationStructure:
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
        size=tlas_sizes.acceleration_structure_size,
        label="tlas",
    )

    command_encoder = device.create_command_encoder()
    command_encoder.build_acceleration_structure(
        desc=tlas_build_desc, dst=tlas, src=None, scratch_buffer=tlas_scratch_buffer
    )
    device.submit_command_buffer(command_encoder.finish())

    return tlas


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_raytracing(device_type: DeviceType):
    device = helpers.get_device(device_type)

    if not device.has_feature(spy.Feature.acceleration_structure):
        pytest.skip("Acceleration structures not supported on this device")
    if not device.has_feature(spy.Feature.ray_tracing):
        pytest.skip("Ray tracing not supported on this device")

    vertices = np.array([-1, -1, 0, 1, -1, 0, -1, 1, 0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    blas = build_blas(device, vertices, indices)

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
    tlas = build_tlas(device, instance_list)

    tensor = Tensor.zeros(device, (64, 64, 3), dtype=float)
    module = Module(device.load_module("test_raytracing.slang"))

    module.trace.ray_tracing(
        hit_groups=[{"hit_group_name": "hit_group", "closest_hit_entry_point": "closest_hit"}],
        miss_entry_points=["miss"],
        max_recursion=1,
        max_ray_payload_size=12,
    )(tid=spy.call_id(), tlas=tlas, _result=tensor)

    data = tensor.to_numpy()

    # spy.tev.show(spy.Bitmap(data))

    assert np.allclose(data[0, 0, :], [0, 0, 0], atol=0.01)
    assert np.allclose(data[0, 63, :], [1, 0, 0], atol=0.01)
    assert np.allclose(data[63, 0, :], [0, 1, 0], atol=0.01)
    assert np.allclose(data[63, 63, :], [1, 0, 1], atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
