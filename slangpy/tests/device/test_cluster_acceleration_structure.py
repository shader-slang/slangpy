# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

import slangpy as spy
from slangpy.testing import helpers


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _create_accel_input_buffer(device: spy.Device, data: np.ndarray) -> spy.Buffer:
    return device.create_buffer(
        data=data.view(np.uint8),
        usage=(
            spy.BufferUsage.acceleration_structure_build_input
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        ),
        default_state=spy.ResourceState.acceleration_structure_build_output,
    )


def _create_uav_buffer(device: spy.Device, size: int) -> spy.Buffer:
    return device.create_buffer(
        size=size,
        usage=(
            spy.BufferUsage.unordered_access
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        ),
        default_state=spy.ResourceState.unordered_access,
    )


def _create_handles_buffer(device: spy.Device, count: int) -> spy.Buffer:
    size = _align_up(
        count * spy.CLUSTER_DEFAULT_HANDLE_STRIDE,
        spy.CLUSTER_OUTPUT_ALIGNMENT,
    )
    return device.create_buffer(
        size=size,
        usage=(
            spy.BufferUsage.unordered_access
            | spy.BufferUsage.acceleration_structure
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        ),
        default_state=spy.ResourceState.unordered_access,
    )


def _execute_implicit_cluster_operation(
    device: spy.Device,
    desc: spy.ClusterOperationDesc,
    result_size: int,
    scratch_size: int,
    handle_count: int,
) -> tuple[spy.Buffer, spy.Buffer]:
    arg_count_buffer = _create_accel_input_buffer(
        device,
        np.array([handle_count], dtype=np.uint32),
    )
    scratch_buffer = _create_uav_buffer(device, scratch_size)
    addresses_buffer = _create_handles_buffer(device, handle_count)
    result_buffer = device.create_buffer(
        size=result_size,
        usage=spy.BufferUsage.acceleration_structure,
    )

    desc.arg_count_buffer = arg_count_buffer
    desc.scratch_buffer = scratch_buffer
    desc.addresses_buffer = addresses_buffer
    desc.result_buffer = result_buffer

    command_encoder = device.create_command_encoder()
    command_encoder.execute_cluster_operation(desc)
    device.submit_command_buffer(command_encoder.finish())
    device.wait_for_idle()
    return result_buffer, addresses_buffer


def test_cluster_argument_records() -> None:
    triangle_args = np.zeros(1, dtype=spy.gpu_structs.triangle_cluster_args_dtype)
    triangle_args["cluster_id"] = 7
    triangle_args["packed_counts_and_formats"] = spy.gpu_structs.pack_triangle_cluster_args_fields(
        triangle_count=1,
        vertex_count=3,
        index_format=spy.gpu_structs.CLUSTER_INDEX_FORMAT_UINT32,
    )
    assert triangle_args["cluster_id"][0] == 7
    assert triangle_args.nbytes == 72

    template_args = np.zeros(1, dtype=spy.gpu_structs.instantiate_template_args_dtype)
    assert template_args.nbytes == 32

    cluster_args = np.zeros(1, dtype=spy.gpu_structs.cluster_args_dtype)
    cluster_args["cluster_handles_stride"] = spy.CLUSTER_DEFAULT_HANDLE_STRIDE
    assert cluster_args["cluster_handles_stride"][0] == spy.CLUSTER_DEFAULT_HANDLE_STRIDE
    assert cluster_args.nbytes == 16

    handle = spy.AccelerationStructureHandle(123)
    assert handle.value == 123


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_cluster_explicit_destination(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(type=device_type)
    if not device.has_feature(spy.Feature.cluster_acceleration_structure):
        pytest.skip("Cluster acceleration structures are not supported on this device")

    vertices = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2], dtype=np.uint32)
    vertex_buffer = _create_accel_input_buffer(device, vertices)
    index_buffer = _create_accel_input_buffer(device, indices)
    triangle_args_data = np.zeros(1, dtype=spy.gpu_structs.triangle_cluster_args_dtype)
    triangle_args_data["packed_counts_and_formats"] = (
        spy.gpu_structs.pack_triangle_cluster_args_fields(
            triangle_count=1,
            vertex_count=3,
            index_format=spy.gpu_structs.CLUSTER_INDEX_FORMAT_UINT32,
        )
    )
    triangle_args_data["vertex_buffer_stride"] = vertices.strides[0]
    triangle_args_data["index_buffer"] = index_buffer.device_address
    triangle_args_data["vertex_buffer"] = vertex_buffer.device_address
    args_buffer = _create_accel_input_buffer(device, triangle_args_data)
    arg_count_buffer = _create_accel_input_buffer(
        device,
        np.array([1], dtype=np.uint32),
    )

    params_values = {
        "type": spy.ClusterOperationType.clas_from_triangles,
        "max_arg_count": 1,
        "clas": {
            "max_unique_geometry_count": 1,
            "max_triangle_count": 1,
            "max_vertex_count": 3,
            "max_total_triangle_count": 1,
            "max_total_vertex_count": 3,
        },
    }
    operation_sizes = device.get_cluster_operation_sizes(spy.ClusterOperationParams(params_values))
    scratch_buffer = _create_uav_buffer(device, operation_sizes.scratch_size)
    per_clas_sizes = _create_uav_buffer(device, np.dtype(np.uint32).itemsize)

    get_sizes_desc = spy.ClusterOperationDesc(
        {
            "params": {
                **params_values,
                "mode": spy.ClusterOperationMode.get_sizes,
            },
            "arg_count_buffer": arg_count_buffer,
            "args_buffer": args_buffer,
            "args_buffer_stride": triangle_args_data.nbytes,
            "scratch_buffer": scratch_buffer,
            "sizes_buffer": per_clas_sizes,
        }
    )
    command_encoder = device.create_command_encoder()
    command_encoder.execute_cluster_operation(get_sizes_desc)
    device.submit_command_buffer(command_encoder.finish())
    device.wait_for_idle()

    clas_size = int(per_clas_sizes.to_numpy().view(np.uint32)[0])
    assert clas_size > 0
    arena = device.create_buffer(
        size=_align_up(clas_size, spy.CLUSTER_OUTPUT_ALIGNMENT),
        usage=spy.BufferUsage.acceleration_structure,
    )
    destination = np.array([arena.device_address], dtype=np.uint64)
    destination_buffer = device.create_buffer(
        data=destination,
        usage=(
            spy.BufferUsage.unordered_access
            | spy.BufferUsage.acceleration_structure
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        ),
        default_state=spy.ResourceState.unordered_access,
    )
    explicit_desc = spy.ClusterOperationDesc(
        {
            "params": {
                **params_values,
                "mode": spy.ClusterOperationMode.explicit_destinations,
            },
            "arg_count_buffer": arg_count_buffer,
            "args_buffer": args_buffer,
            "args_buffer_stride": triangle_args_data.nbytes,
            "scratch_buffer": scratch_buffer,
            "addresses_buffer": destination_buffer,
        }
    )
    command_encoder = device.create_command_encoder()
    command_encoder.execute_cluster_operation(explicit_desc)
    device.submit_command_buffer(command_encoder.finish())
    device.wait_for_idle()

    cluster_handle = int(destination_buffer.to_numpy().view(np.uint64)[0])
    assert cluster_handle == arena.device_address


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_cluster_acceleration_structure_trace(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(type=device_type)
    if not device.has_feature(spy.Feature.cluster_acceleration_structure):
        pytest.skip("Cluster acceleration structures are not supported on this device")

    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2], dtype=np.uint32)
    vertex_buffer = _create_accel_input_buffer(device, vertices)
    index_buffer = _create_accel_input_buffer(device, indices)

    triangle_args_stride = spy.gpu_structs.triangle_cluster_args_dtype.itemsize
    triangle_args_buffer = device.create_buffer(
        size=triangle_args_stride,
        struct_size=triangle_args_stride,
        usage=(
            spy.BufferUsage.unordered_access
            | spy.BufferUsage.acceleration_structure_build_input
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        ),
        default_state=spy.ResourceState.unordered_access,
    )
    write_args_program = device.load_program(
        "test_cluster_acceleration_structure.slang",
        ["write_triangle_args"],
    )
    write_args_kernel = device.create_compute_kernel(write_args_program)
    write_args_kernel.dispatch(
        thread_count=[1, 1, 1],
        vars={
            "triangle_args": triangle_args_buffer,
            "index_buffer_address": index_buffer.device_address,
            "vertex_buffer_address": vertex_buffer.device_address,
            "vertex_buffer_stride": vertices.strides[0],
        },
    )
    device.wait_for_idle()

    clas_params = spy.ClusterOperationParams(
        {
            "type": spy.ClusterOperationType.clas_from_triangles,
            "max_arg_count": 1,
            "clas": {
                "max_unique_geometry_count": 1,
                "max_triangle_count": 1,
                "max_vertex_count": 3,
                "max_total_triangle_count": 1,
                "max_total_vertex_count": 3,
            },
        }
    )
    clas_sizes = device.get_cluster_operation_sizes(clas_params)
    assert clas_sizes.result_size > 0
    assert clas_sizes.scratch_size > 0

    clas_desc = spy.ClusterOperationDesc(
        {
            "params": clas_params,
            "args_buffer": triangle_args_buffer,
            "args_buffer_stride": triangle_args_stride,
        }
    )
    clas_result_buffer, clas_handles_buffer = _execute_implicit_cluster_operation(
        device,
        clas_desc,
        clas_sizes.result_size,
        clas_sizes.scratch_size,
        1,
    )
    clas_handle = int(clas_handles_buffer.to_numpy().view(np.uint64)[0])
    assert clas_handle != 0

    cluster_args_data = np.zeros(1, dtype=spy.gpu_structs.cluster_args_dtype)
    cluster_args_data["cluster_handles_count"] = 1
    cluster_args_data["cluster_handles_stride"] = spy.CLUSTER_DEFAULT_HANDLE_STRIDE
    cluster_args_data["cluster_handles_buffer"] = clas_handles_buffer.device_address
    cluster_args_buffer = _create_accel_input_buffer(device, cluster_args_data)

    blas_params = spy.ClusterOperationParams(
        {
            "type": spy.ClusterOperationType.blas_from_clas,
            "max_arg_count": 1,
            "blas": {
                "max_clas_count": 1,
                "max_total_clas_count": 1,
            },
        }
    )
    blas_sizes = device.get_cluster_operation_sizes(blas_params)
    assert blas_sizes.result_size > 0
    assert blas_sizes.scratch_size > 0

    blas_desc = spy.ClusterOperationDesc(
        {
            "params": blas_params,
            "args_buffer": cluster_args_buffer,
            "args_buffer_stride": cluster_args_data.nbytes,
        }
    )
    blas_result_buffer, blas_handles_buffer = _execute_implicit_cluster_operation(
        device,
        blas_desc,
        blas_sizes.result_size,
        blas_sizes.scratch_size,
        1,
    )
    blas_handle = int(blas_handles_buffer.to_numpy().view(np.uint64)[0])
    assert blas_handle != 0

    instance_list = device.create_acceleration_structure_instance_list(1)
    instance_list.write(
        0,
        {
            "transform": spy.float3x4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
            "instance_id": 0,
            "instance_mask": 0xFF,
            "instance_contribution_to_hit_group_index": 0,
            "flags": spy.AccelerationStructureInstanceFlags.none,
            "acceleration_structure": spy.AccelerationStructureHandle(blas_handle),
        },
    )
    tlas_build_desc = spy.AccelerationStructureBuildDesc(
        {"inputs": [instance_list.build_input_instances()]}
    )
    tlas_sizes = device.get_acceleration_structure_sizes(tlas_build_desc)
    tlas = device.create_acceleration_structure(
        kind=spy.AccelerationStructureKind.top_level,
        size=tlas_sizes.acceleration_structure_size,
        label="cluster_tlas",
    )
    tlas_scratch = _create_uav_buffer(device, tlas_sizes.scratch_size)
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
        "test_cluster_acceleration_structure.slang",
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
        flags=spy.RayTracingPipelineFlags.enable_clusters,
    )
    shader_table = device.create_shader_table(
        program=program,
        ray_gen_entry_points=["ray_gen"],
        miss_entry_points=["miss"],
        hit_group_names=["hit_group"],
    )
    trace_result = device.create_buffer(
        data=np.zeros(1, dtype=np.uint32),
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source,
    )

    command_encoder = device.create_command_encoder()
    with command_encoder.begin_ray_tracing_pass() as pass_encoder:
        shader_object = pass_encoder.bind_pipeline(pipeline, shader_table)
        cursor = spy.ShaderCursor(shader_object)
        cursor.result_buffer = trace_result
        cursor.scene_bvh = tlas
        pass_encoder.dispatch_rays(0, [1, 1, 1])
    device.submit_command_buffer(command_encoder.finish())
    device.wait_for_idle()

    assert trace_result.to_numpy().view(np.uint32)[0] == 2

    # Keep buffers that back cluster handles alive through the ray tracing dispatch.
    assert clas_result_buffer.size > 0
    assert blas_result_buffer.size > 0
