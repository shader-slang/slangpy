# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

from slangpy import DeviceType, float3, Logger, LogLevel, float2x3, Module
from slangpy.types import NDBuffer, Tensor
from slangpy.testing import helpers

from typing import Any


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    m = Module(device.load_module("test_transforms.slang"))
    m.logger = Logger(LogLevel.debug)
    return m


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_values_basic_input_transform(device_type: DeviceType):
    # Really simple test that just copies values from one buffer to another
    # with a transform involved

    m = load_test_module(device_type)

    # Create input+output buffers
    a = NDBuffer(device=m.device, shape=(2, 2), dtype=float)
    b = NDBuffer(device=m.device, shape=(2, 2), dtype=float)

    # Populate input
    a_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    helpers.write_ndbuffer_from_numpy(a, a_data.flatten(), 1)

    # Call function, which should copy to output with dimensions flipped
    func = m.copy_values.as_func()
    func = func.map((1, 0))
    func(a, b)

    # Get and verify output
    b_data = helpers.read_ndbuffer_from_numpy(b).reshape(-1, 2)
    for i in range(2):
        for j in range(2):
            a = a_data[j, i]
            b = b_data[i, j]
            assert a == b


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("data_type", [[float3, [3]], [float2x3, [2, 3]]])
def test_add_vectors_matrix_basic_input_transform(device_type: DeviceType, data_type: Any):
    m = load_test_module(device_type)

    # Slightly more complex test involving 2 inputs of float3s/float2x3s,
    # outputing to a result buffer
    type = data_type[0]
    shape = data_type[1]

    a = NDBuffer(device=m.device, shape=(2, 3), dtype=type)
    b = NDBuffer(device=m.device, shape=(3, 2), dtype=type)

    a_data = np.random.rand(2, 3, *shape).astype(np.float32)
    b_data = np.random.rand(3, 2, *shape).astype(np.float32)

    helpers.write_ndbuffer_from_numpy(a, a_data.flatten())
    helpers.write_ndbuffer_from_numpy(b, b_data.flatten())

    if type == float3:
        func = m.add_vectors.map((1, 0))
    else:
        func = m.add_matrix.map((1, 0))

    res: NDBuffer = func(a, b)

    assert res.shape == (3, 2)

    res_data = helpers.read_ndbuffer_from_numpy(res).reshape(3, 2, *shape)

    for i in range(3):
        for j in range(2):
            a = a_data[j, i]
            b = b_data[i, j]
            expected = a + b
            r = res_data[i, j]
            assert np.allclose(r, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("data_type", [[float3, [3]], [float2x3, [2, 3]]])
def test_add_vectors_matrix_vecindex_inputcontainer_input_transform(
    device_type: DeviceType, data_type: Any
):
    m = load_test_module(device_type)

    # Test 1
    # Test remapping when one of the inputs is an 3D buffer of floats
    # instead of 2D buffer of float3s. In this case, the remapping
    # involves only the lower 2 dimensions (i.e. those of the container)

    # Test 2
    # Test remapping when one of the inputs is an 4D buffer of floats
    # instead of 2D buffer of float2x3s. In this case, the remapping
    # involves only the lower 2 dimensions (i.e. those of the container)

    type = data_type[0]
    shape = data_type[1]

    a = NDBuffer(device=m.device, shape=(2, 3, *shape), dtype=float)
    b = NDBuffer(device=m.device, shape=(3, 2), dtype=type)
    a_data = np.random.rand(2, 3, *shape).astype(np.float32)
    b_data = np.random.rand(3, 2, *shape).astype(np.float32)

    helpers.write_ndbuffer_from_numpy(a, a_data.flatten())
    helpers.write_ndbuffer_from_numpy(b, b_data.flatten())

    if type == float3:
        func = m.add_vectors.map((1, 0))
    else:
        func = m.add_matrix.map((1, 0))

    res: NDBuffer = func(a, b)
    assert res.shape == (3, 2)

    res_data = helpers.read_ndbuffer_from_numpy(res).reshape(3, 2, *shape)

    for i in range(3):
        for j in range(2):
            a = a_data[j, i]
            b = b_data[i, j]
            expected = a + b
            r = res_data[i, j]
            assert np.allclose(r, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("data_type", [[float3, [3]], [float2x3, [2, 3]]])
def test_copy_data_vecindex_inputcontainer_input_transform(device_type: DeviceType, data_type: Any):
    m = load_test_module(device_type)

    # Test 1
    # Test remapping when one of the inputs is an 3D buffer of floats
    # instead of 2D buffer of float3. In this case, the remapping
    # involves only the lower 2 dimensions (i.e. those of the container)

    # Test 2
    # Test remapping when one of the inputs is an 4D buffer of floats
    # instead of 2D buffer of float2x3. In this case, the remapping
    # involves only the lower 2 dimensions (i.e. those of the container)

    type = data_type[0]
    shape = data_type[1]

    # inn is a 4D buffer of floats to represent a 2x3 float2x3 matrix

    inn = NDBuffer(device=m.device, shape=(2, 3, *shape), dtype=float)
    out = NDBuffer(device=m.device, shape=(3, 2), dtype=type)
    inn_data = np.random.rand(2, 3, *shape).astype(np.float32)

    helpers.write_ndbuffer_from_numpy(inn, inn_data.flatten())

    if type == float3:
        func = m.copy_vectors.map((1, 0))
    else:
        func = m.copy_matrix.map((1, 0))

    func(inn, out)

    out_data = helpers.read_ndbuffer_from_numpy(out).reshape(3, 2, *shape)

    for i in range(3):
        for j in range(2):
            inn = inn_data[j, i]
            out = out_data[i, j]
            assert np.allclose(inn, out)


@pytest.mark.skip(reason="Can't get working on build machine atm")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_vectors_vecindex_outputcontainer_input_transform(device_type: DeviceType):
    m = load_test_module(device_type)

    # Test remapping when one of the inputs is an 3D buffer of floats
    # instead of 2D buffer of float3s. In this case, the remapping
    # involves only the lower 2 dimensions (i.e. those of the container)

    inn = NDBuffer(device=m.device, shape=(2, 3, 3), dtype=float)
    out = NDBuffer(device=m.device, shape=(3, 2), dtype=float3)

    inn_data = np.random.rand(2, 3, 3).astype(np.float32)
    helpers.write_ndbuffer_from_numpy(inn, inn_data.flatten(), 1)

    func = m.copy_vectors.map(None, (1, 0))

    func(inn, out)

    out_data = helpers.read_ndbuffer_from_numpy(out).reshape(3, 2, 3)

    for i in range(2):
        for j in range(3):
            inn = inn_data[i, j]
            out = out_data[j, i]
            assert np.allclose(inn, out)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("data_type", [[float3, [3]], [float2x3, [2, 3]]])
def test_add_vectors_matrix_basic_output_transform(device_type: DeviceType, data_type: Any):
    m = load_test_module(device_type)

    # Test the output transform, where we take 2 1D buffers with different
    # sizes and braodcast each to a different dimension.

    type = data_type[0]
    shape = data_type[1]

    a = NDBuffer(device=m.device, shape=(5,), dtype=type)
    b = NDBuffer(device=m.device, shape=(10,), dtype=type)

    a_data = np.random.rand(a.shape[0], *shape).astype(np.float32)
    b_data = np.random.rand(b.shape[0], *shape).astype(np.float32)

    helpers.write_ndbuffer_from_numpy(a, a_data.flatten())
    helpers.write_ndbuffer_from_numpy(b, b_data.flatten())

    if type == float3:
        func = m.add_vectors.map((0,), (1,))
    else:
        func = m.add_matrix.map((0,), (1,))

    res: NDBuffer = func(a, b)

    assert res.shape == (a.shape[0], b.shape[0])

    res_data = helpers.read_ndbuffer_from_numpy(res).reshape(a.shape[0], b.shape[0], *shape)

    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            a = a_data[i]
            b = b_data[j]
            expected = a + b
            r = res_data[i, j]
            assert np.allclose(r, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("data_type", [[float3, [3]], [float2x3, [2, 3]]])
def test_add_vectors_matrix_broadcast_from_buffer(device_type: DeviceType, data_type: Any):
    m = load_test_module(device_type)

    # Test the output transform, where we take 2 1D buffers with different
    # sizes and braodcast each to a different dimension.

    type = data_type[0]
    shape = data_type[1]

    a = NDBuffer(device=m.device, shape=(1,), dtype=type)
    b = NDBuffer(device=m.device, shape=(10,), dtype=type)

    a_data = np.random.rand(a.shape[0], *shape).astype(np.float32)
    b_data = np.random.rand(b.shape[0], *shape).astype(np.float32)

    helpers.write_ndbuffer_from_numpy(a, a_data.flatten())
    helpers.write_ndbuffer_from_numpy(b, b_data.flatten())

    if type == float3:
        res: NDBuffer = m.add_vectors(a, b)
    else:
        res: NDBuffer = m.add_matrix(a, b)

    assert res.shape == (b.shape[0],)

    res_data = helpers.read_ndbuffer_from_numpy(res).reshape(b.shape[0], *shape)

    for j in range(b.shape[0]):
        a = a_data[0]
        b = b_data[j]
        expected = a + b
        r = res_data[j]
        assert np.allclose(r, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("data_type", [[float3, [3]], [float2x3, [2, 3]]])
def test_add_vectors_matrix_broadcast_from_buffer_2(device_type: DeviceType, data_type: Any):
    m = load_test_module(device_type)

    # Test the output transform, where we take 2 1D buffers with different
    # sizes and braodcast each to a different dimension.

    type = data_type[0]
    shape = data_type[1]

    a = NDBuffer(device=m.device, shape=(1, 5), dtype=type)
    b = NDBuffer(device=m.device, shape=(10, 5), dtype=type)

    a_data = np.random.rand(a.shape[0], a.shape[1], *shape).astype(np.float32)
    b_data = np.random.rand(b.shape[0], b.shape[1], *shape).astype(np.float32)

    helpers.write_ndbuffer_from_numpy(a, a_data.flatten())
    helpers.write_ndbuffer_from_numpy(b, b_data.flatten())

    if type == float3:
        res: NDBuffer = m.add_vectors(a, b)
    else:
        res: NDBuffer = m.add_matrix(a, b)

    assert res.shape == (b.shape[0], b.shape[1])

    res_data = helpers.read_ndbuffer_from_numpy(res).reshape(b.shape[0], b.shape[1], *shape)

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            av = a_data[0][j]
            bv = b_data[i][j]
            expected = av + bv
            r = res_data[i][j]
            assert np.allclose(r, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("data_type", [[float3, [3]], [float2x3, [2, 3]]])
def test_add_vectors_matrix_broadcast_from_diff_buffer(device_type: DeviceType, data_type: Any):
    m = load_test_module(device_type)

    # Test the output transform, where we take 2 1D buffers with different
    # sizes and braodcast each to a different dimension.

    type = data_type[0]
    shape = data_type[1]

    a = Tensor.empty(device=m.device, shape=(1, 5), dtype=type)
    b = NDBuffer(device=m.device, shape=(10, 5), dtype=type)

    a_data = np.random.rand(a.shape[0], a.shape[1], *shape).astype(np.float32)
    b_data = np.random.rand(b.shape[0], b.shape[1], *shape).astype(np.float32)

    helpers.write_ndbuffer_from_numpy(a, a_data.flatten())
    helpers.write_ndbuffer_from_numpy(b, b_data.flatten())

    if type == float3:
        res: NDBuffer = m.add_vectors(a, b)
    else:
        res: NDBuffer = m.add_matrix(a, b)

    assert res.shape == (b.shape[0], b.shape[1])

    res_data = helpers.read_ndbuffer_from_numpy(res).reshape(b.shape[0], b.shape[1], *shape)

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            av = a_data[0][j]
            bv = b_data[i][j]
            expected = av + bv
            r = res_data[i][j]
            assert np.allclose(r, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
