# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers


def get_coop_vec_device(device_type: spy.DeviceType) -> spy.Device:
    device = helpers.get_device(device_type)
    if not device.has_feature(spy.Feature.cooperative_vector):
        pytest.skip("Device does not support cooperative vector")
    return device


SIZE_CHECKS = [
    (4, 4, spy.DataType.float32, 64),
    (4, 8, spy.DataType.float32, 128),
    (8, 4, spy.DataType.float32, 128),
    (4, 4, spy.DataType.float16, 64),  # rounded up to 64 from 32
    (4, 8, spy.DataType.float16, 64),
    (8, 4, spy.DataType.float16, 64),
    (4, 4, spy.DataType.float8_e4m3, 64),  # rounded up to 64 from 16
    (8, 16, spy.DataType.float8_e4m3, 128),
    (4, 4, spy.DataType.float8_e5m2, 64),  # rounded up to 64 from 16
    (16, 8, spy.DataType.float8_e5m2, 128),
]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("rows, cols, dtype, expected_size", SIZE_CHECKS)
def test_matrix_size(
    device_type: spy.DeviceType,
    rows: int,
    cols: int,
    dtype: spy.DataType,
    expected_size: int,
):
    device = get_coop_vec_device(device_type)

    size = device.get_coop_vec_matrix_size(rows, cols, spy.CoopVecMatrixLayout.row_major, dtype)
    assert size == expected_size

    size = device.get_coop_vec_matrix_size(rows, cols, spy.CoopVecMatrixLayout.column_major, dtype)
    assert size == expected_size


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("rows", [4, 8])
@pytest.mark.parametrize("cols", [4, 8])
@pytest.mark.parametrize("dtype", [spy.DataType.float32, spy.DataType.float16])
@pytest.mark.parametrize(
    "layout",
    [
        spy.CoopVecMatrixLayout.row_major,
        spy.CoopVecMatrixLayout.column_major,
        spy.CoopVecMatrixLayout.inferencing_optimal,
        spy.CoopVecMatrixLayout.training_optimal,
    ],
)
def test_matrix_desc(
    device_type: spy.DeviceType,
    rows: int,
    cols: int,
    dtype: spy.DataType,
    layout: spy.CoopVecMatrixLayout,
):
    device = get_coop_vec_device(device_type)
    if (
        device.info.type == spy.DeviceType.cuda
        and dtype == spy.DataType.float32
        and layout
        in [spy.CoopVecMatrixLayout.inferencing_optimal, spy.CoopVecMatrixLayout.training_optimal]
    ):
        pytest.skip("float32 not supported for inferencing/training optimal layout on CUDA")
    desc = device.create_coop_vec_matrix_desc(rows, cols, layout, dtype)
    size = device.get_coop_vec_matrix_size(rows, cols, layout, dtype)
    assert desc.rows == rows
    assert desc.cols == cols
    assert desc.layout == layout
    assert desc.size == size
    assert desc.element_type == dtype


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_matrix_convert_matrix_host(device_type: spy.DeviceType):
    device = get_coop_vec_device(device_type)

    src = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    dst = np.zeros_like(src)

    device.convert_coop_vec_matrix(
        dst, src, spy.CoopVecMatrixLayout.column_major, spy.CoopVecMatrixLayout.row_major
    )

    # Conversion should have changed memory layout, but not shape
    src_t = np.transpose(src)
    dst = dst.reshape(src_t.shape)
    assert np.allclose(src_t, dst)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_matrix_convert_huge_matrix_host(device_type: spy.DeviceType):
    device = get_coop_vec_device(device_type)

    src = np.random.random((128, 128)).astype(np.float32)
    dst = np.zeros_like(src)

    device.convert_coop_vec_matrix(
        dst,
        src,
        spy.CoopVecMatrixLayout.row_major,
        spy.CoopVecMatrixLayout.column_major,
    )

    # Conversion should have changed memory layout, but not shape
    src_t = np.transpose(src)
    dst = dst.reshape(src_t.shape)
    assert np.allclose(src_t, dst)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_matrix_convert_matrix_device(device_type: spy.DeviceType):
    device = get_coop_vec_device(device_type)

    data = np.random.random((128, 128)).astype(np.float32)

    src_desc = device.create_coop_vec_matrix_desc(
        rows=data.shape[0],
        cols=data.shape[1],
        layout=spy.CoopVecMatrixLayout.row_major,
        element_type=spy.DataType.float32,
    )
    dst_desc = device.create_coop_vec_matrix_desc(
        rows=data.shape[0],
        cols=data.shape[1],
        layout=spy.CoopVecMatrixLayout.column_major,
        element_type=spy.DataType.float32,
    )

    src_buf = device.create_buffer(data=data, usage=spy.BufferUsage.shader_resource)
    dst_buf = device.create_buffer(
        src_buf.size,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    encoder = device.create_command_encoder()
    encoder.convert_coop_vec_matrix(dst_buf, dst_desc, src_buf, src_desc)
    device.submit_command_buffer(encoder.finish())

    result = dst_buf.to_numpy().view(np.float32).reshape(data.shape)

    data_t = np.transpose(data)
    result = result.reshape(
        data_t.shape
    )  # probably shouldn't have to do this but conversion doesn't change shape
    assert np.allclose(data_t, result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
