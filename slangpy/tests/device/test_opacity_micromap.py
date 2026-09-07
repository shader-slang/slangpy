# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc

import numpy as np
import pytest

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_opacity_micromap_trace(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(type=device_type)
    if not device.has_feature(spy.Feature.opacity_micromap):
        pytest.skip("Opacity micromaps are not supported on this device")

    opacity_data = np.zeros(256, dtype=np.uint8)
    opacity_data[1] = 1
    opacity_data[2] = 0b1010
    opacity_data_buffer = device.create_buffer(
        data=opacity_data,
        usage=spy.BufferUsage.micromap_build_input,
        default_state=spy.ResourceState.micromap_build_input,
        label="opacity_data",
    )

    triangle_descs = np.zeros(32, dtype=spy.gpu_structs.micromap_triangle_desc_dtype)
    triangle_descs[0] = (0, 0, spy.OpacityMicromapFormat.two_state.value)
    triangle_descs[1] = (1, 0, spy.OpacityMicromapFormat.two_state.value)
    triangle_descs[2] = (2, 1, spy.OpacityMicromapFormat.two_state.value)
    triangle_desc_buffer = device.create_buffer(
        data=triangle_descs.view(np.uint8),
        usage=spy.BufferUsage.micromap_build_input,
        default_state=spy.ResourceState.micromap_build_input,
        label="micromap_triangle_descs",
    )

    usage_counts = [
        {
            "count": 2,
            "subdivision_level": 0,
            "format": spy.OpacityMicromapFormat.two_state,
        },
        {
            "count": 1,
            "subdivision_level": 1,
            "format": spy.OpacityMicromapFormat.two_state,
        },
    ]
    micromap_build_desc = spy.MicromapBuildDesc(
        {
            "data_buffer": opacity_data_buffer,
            "descriptor_buffer": triangle_desc_buffer,
            "histogram": usage_counts,
        }
    )
    micromap_sizes = device.get_micromap_sizes(micromap_build_desc)
    assert micromap_sizes.micromap_size > 0
    assert micromap_sizes.scratch_size > 0

    micromap = device.create_micromap(
        size=micromap_sizes.micromap_size,
        label="opacity_micromap",
    )
    micromap_scratch = device.create_buffer(
        size=micromap_sizes.scratch_size,
        usage=spy.BufferUsage.unordered_access,
        default_state=spy.ResourceState.unordered_access,
        label="micromap_scratch",
    )

    command_encoder = device.create_command_encoder()
    command_encoder.build_micromap(
        micromap_build_desc,
        micromap,
        micromap_scratch,
    )
    device.submit_command_buffer(command_encoder.finish())
    device.wait_for_idle()

    vertices = np.array(
        [
            [-0.9, -0.5, 1.0],
            [-0.1, -0.5, 1.0],
            [-0.5, 0.5, 1.0],
            [0.1, -0.5, 1.0],
            [0.9, -0.5, 1.0],
            [0.5, 0.5, 1.0],
            [1.0, -0.5, 1.0],
            [2.0, -0.5, 1.0],
            [1.5, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    vertex_buffer = device.create_buffer(
        data=vertices,
        usage=spy.BufferUsage.acceleration_structure_build_input,
        default_state=spy.ResourceState.acceleration_structure_build_output,
        label="vertices",
    )

    opacity_attachment = spy.AccelerationStructureOpacityMicromapDesc(
        {
            "micromap": micromap,
            "usage_counts": usage_counts,
        }
    )
    triangle_input = spy.AccelerationStructureBuildInputTriangles(
        {
            "vertex_buffers": [vertex_buffer],
            "vertex_format": spy.Format.rgb32_float,
            "vertex_count": vertices.shape[0],
            "vertex_stride": vertices.strides[0],
            "flags": spy.AccelerationStructureGeometryFlags.none,
            "opacity_micromap": opacity_attachment,
        }
    )
    blas_build_desc = spy.AccelerationStructureBuildDesc({"inputs": [triangle_input]})
    blas_sizes = device.get_acceleration_structure_sizes(blas_build_desc)
    blas = device.create_acceleration_structure(
        kind=spy.AccelerationStructureKind.bottom_level,
        size=blas_sizes.acceleration_structure_size,
        label="micromap_blas",
    )
    blas_scratch = device.create_buffer(
        size=blas_sizes.scratch_size,
        usage=spy.BufferUsage.unordered_access,
        label="blas_scratch",
    )
    command_encoder = device.create_command_encoder()
    command_encoder.build_acceleration_structure(
        blas_build_desc,
        blas,
        None,
        blas_scratch,
    )
    device.submit_command_buffer(command_encoder.finish())
    device.wait_for_idle()

    triangle_input.opacity_micromap = None
    opacity_attachment = None
    blas_build_desc = None
    micromap = None
    gc.collect()

    instance_list = device.create_acceleration_structure_instance_list(1)
    instance_list.write(
        0,
        {
            "transform": spy.float3x4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
            "instance_id": 0,
            "instance_mask": 0xFF,
            "instance_contribution_to_hit_group_index": 0,
            "flags": spy.AccelerationStructureInstanceFlags.none,
            "acceleration_structure": blas.handle,
        },
    )
    tlas_build_desc = spy.AccelerationStructureBuildDesc(
        {"inputs": [instance_list.build_input_instances()]}
    )
    tlas_sizes = device.get_acceleration_structure_sizes(tlas_build_desc)
    tlas = device.create_acceleration_structure(
        kind=spy.AccelerationStructureKind.top_level,
        size=tlas_sizes.acceleration_structure_size,
        label="micromap_tlas",
    )
    tlas_scratch = device.create_buffer(
        size=tlas_sizes.scratch_size,
        usage=spy.BufferUsage.unordered_access,
        label="tlas_scratch",
    )
    command_encoder = device.create_command_encoder()
    command_encoder.build_acceleration_structure(
        tlas_build_desc,
        tlas,
        None,
        tlas_scratch,
    )
    device.submit_command_buffer(command_encoder.finish())
    device.wait_for_idle()

    program = device.load_program(
        "test_opacity_micromap.slang",
        ["ray_gen", "miss", "closest_hit"],
    )
    pipeline = device.create_ray_tracing_pipeline(
        program=program,
        hit_groups=[
            spy.HitGroupDesc(
                hit_group_name="hit_group",
                closest_hit_entry_point="closest_hit",
            )
        ],
        max_recursion=1,
        max_ray_payload_size=4,
        flags=spy.RayTracingPipelineFlags.enable_opacity_micromaps,
    )
    shader_table = device.create_shader_table(
        program=program,
        ray_gen_entry_points=["ray_gen"],
        miss_entry_points=["miss"],
        hit_group_names=["hit_group"],
    )
    result_buffer = device.create_buffer(
        data=np.zeros(6, dtype=np.uint32),
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source,
        label="results",
    )

    command_encoder = device.create_command_encoder()
    with command_encoder.begin_ray_tracing_pass() as pass_encoder:
        shader_object = pass_encoder.bind_pipeline(pipeline, shader_table)
        cursor = spy.ShaderCursor(shader_object)
        cursor.result_buffer = result_buffer
        cursor.scene_bvh = tlas
        pass_encoder.dispatch_rays(0, [6, 1, 1])
    device.submit_command_buffer(command_encoder.finish())
    device.wait_for_idle()

    np.testing.assert_array_equal(
        result_buffer.to_numpy().view(np.uint32),
        np.array([1, 2, 1, 2, 1, 2], dtype=np.uint32),
    )
