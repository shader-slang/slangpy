# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

import slangpy.tests.helpers as helpers
from slangpy import Module
from slangpy.backend import DeviceType, float3
from slangpy.types import NDBuffer, Tensor


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_transforms.slang"))


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
    a.from_numpy(a_data)

    # Call function, which should copy to output with dimensions flipped
    func = m.copy_values.as_func()
    func = func.map((1, 0))
    func(a, b)

    # Get and verify output
    b_data = b.storage.to_numpy().view(np.float32).reshape(-1, 2)
    for i in range(2):
        for j in range(2):
            a = a_data[j, i]
            b = b_data[i, j]
            assert a == b


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_add_vectors_basic_input_transform(device_type: DeviceType):
    m = load_test_module(device_type)

    # Slightly more complex test involving 2 inputs of float3s,
    # outputing to a result buffer

    a = NDBuffer(device=m.device, shape=(2, 3), dtype=float3)
    b = NDBuffer(device=m.device, shape=(3, 2), dtype=float3)

    a_data = np.random.rand(2, 3, 3).astype(np.float32)
    b_data = np.random.rand(3, 2, 3).astype(np.float32)

    a.from_numpy(a_data)
    b.from_numpy(b_data)

    func = m.add_vectors.map((1, 0))

    res: NDBuffer = func(a, b)

    assert res.shape == (3, 2)

    res_data = res.storage.to_numpy().view(np.float32).reshape(3, 2, 3)

    for i in range(3):
        for j in range(2):
            a = a_data[j, i]
            b = b_data[i, j]
            expected = a + b
            r = res_data[i, j]
            assert np.allclose(r, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_add_vectors_vecindex_inputcontainer_input_transform(device_type: DeviceType):
    m = load_test_module(device_type)

    # Test remapping when one of the inputs is an 3D buffer of floats
    # instead of 2D buffer of float3s. In this case, the remapping
    # involves only the lower 2 dimensions (i.e. those of the container)

    a = NDBuffer(device=m.device, shape=(2, 3, 3), dtype=float)
    b = NDBuffer(device=m.device, shape=(3, 2), dtype=float3)

    a_data = np.random.rand(2, 3, 3).astype(np.float32)
    b_data = np.random.rand(3, 2, 3).astype(np.float32)

    a.from_numpy(a_data)
    b.from_numpy(b_data)

    func = m.add_vectors.map((1, 0))

    res: NDBuffer = func(a, b)

    assert res.shape == (3, 2)

    res_data = res.storage.to_numpy().view(np.float32).reshape(3, 2, 3)

    for i in range(3):
        for j in range(2):
            a = a_data[j, i]
            b = b_data[i, j]
            expected = a + b
            r = res_data[i, j]
            assert np.allclose(r, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_vectors_vecindex_inputcontainer_input_transform(device_type: DeviceType):
    m = load_test_module(device_type)

    # Test remapping when one of the inputs is an 3D buffer of floats
    # instead of 2D buffer of float3s. In this case, the remapping
    # involves only the lower 2 dimensions (i.e. those of the container)

    inn = NDBuffer(device=m.device, shape=(2, 3, 3), dtype=float)
    out = NDBuffer(device=m.device, shape=(3, 2), dtype=float3)

    inn_data = np.random.rand(2, 3, 3).astype(np.float32)
    inn.from_numpy(inn_data)

    func = m.copy_vectors.map((1, 0))

    func(inn, out)

    out_data = out.storage.to_numpy().view(np.float32).reshape(3, 2, 3)

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
    inn.from_numpy(inn_data)

    func = m.copy_vectors.map(None, (1, 0))

    func(inn, out)

    out_data = out.storage.to_numpy().view(np.float32).reshape(3, 2, 3)

    for i in range(2):
        for j in range(3):
            inn = inn_data[i, j]
            out = out_data[j, i]
            assert np.allclose(inn, out)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_add_vectors_basic_output_transform(device_type: DeviceType):
    m = load_test_module(device_type)

    # Test the output transform, where we take 2 1D buffers with different
    # sizes and braodcast each to a different dimension.

    a = NDBuffer(device=m.device, shape=(5,), dtype=float3)
    b = NDBuffer(device=m.device, shape=(10,), dtype=float3)

    a_data = np.random.rand(a.shape[0], 3).astype(np.float32)
    b_data = np.random.rand(b.shape[0], 3).astype(np.float32)

    a.from_numpy(a_data)
    b.from_numpy(b_data)

    func = m.add_vectors.map((0,), (1,))

    res: NDBuffer = func(a, b)

    assert res.shape == (a.shape[0], b.shape[0])

    res_data = res.storage.to_numpy().view(
        np.float32).reshape(res.shape[0], res.shape[1], 3)

    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            a = a_data[i]
            b = b_data[j]
            expected = a + b
            r = res_data[i, j]
            assert np.allclose(r, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_add_vectors_broadcast_from_buffer(device_type: DeviceType):
    m = load_test_module(device_type)

    # Test the output transform, where we take 2 1D buffers with different
    # sizes and braodcast each to a different dimension.

    a = NDBuffer(device=m.device, shape=(1,), dtype=float3)
    b = NDBuffer(device=m.device, shape=(10,), dtype=float3)

    a_data = np.random.rand(a.shape[0], 3).astype(np.float32)
    b_data = np.random.rand(b.shape[0], 3).astype(np.float32)

    a.from_numpy(a_data)
    b.from_numpy(b_data)

    res: NDBuffer = m.add_vectors(a, b)
    assert res.shape == (b.shape[0],)

    res_data = res.storage.to_numpy().view(np.float32).reshape(res.shape[0], 3)

    for j in range(b.shape[0]):
        a = a_data[0]
        b = b_data[j]
        expected = a + b
        r = res_data[j]
        assert np.allclose(r, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_add_vectors_broadcast_from_buffer_2(device_type: DeviceType):
    m = load_test_module(device_type)

    # Test the output transform, where we take 2 1D buffers with different
    # sizes and braodcast each to a different dimension.

    a = NDBuffer(device=m.device, shape=(1, 5), dtype=float3)
    b = NDBuffer(device=m.device, shape=(10, 5), dtype=float3)

    a_data = np.random.rand(a.shape[0], a.shape[1], 3).astype(np.float32)
    b_data = np.random.rand(b.shape[0], b.shape[1], 3).astype(np.float32)

    a.from_numpy(a_data)
    b.from_numpy(b_data)

    res: NDBuffer = m.add_vectors(a, b)
    assert res.shape == (b.shape[0], b.shape[1])

    res_data = res.storage.to_numpy().view(
        np.float32).reshape(res.shape[0], res.shape[1], 3)

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            av = a_data[0][j]
            bv = b_data[i][j]
            expected = av + bv
            r = res_data[i][j]
            assert np.allclose(r, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_add_vectors_broadcast_from_diff_buffer(device_type: DeviceType):
    m = load_test_module(device_type)

    # Test the output transform, where we take 2 1D buffers with different
    # sizes and braodcast each to a different dimension.

    a = Tensor.empty(device=m.device, shape=(1, 5), dtype=float3)
    b = NDBuffer(device=m.device, shape=(10, 5), dtype=float3)

    a_data = np.random.rand(a.shape[0], a.shape[1], 3).astype(np.float32)
    b_data = np.random.rand(b.shape[0], b.shape[1], 3).astype(np.float32)

    a.storage.from_numpy(a_data)
    b.from_numpy(b_data)

    res: NDBuffer = m.add_vectors(a, b)
    assert res.shape == (b.shape[0], b.shape[1])

    res_data = res.storage.to_numpy().view(
        np.float32).reshape(res.shape[0], res.shape[1], 3)

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            av = a_data[0][j]
            bv = b_data[i][j]
            expected = av + bv
            r = res_data[i][j]
            assert np.allclose(r, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
