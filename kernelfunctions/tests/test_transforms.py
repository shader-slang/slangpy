import pytest
from sgl import float3
from kernelfunctions.backend import DeviceType, float2
from kernelfunctions.function import Function
from kernelfunctions.module import Module
import kernelfunctions.tests.helpers as helpers
from kernelfunctions.types.buffer import NDBuffer
import numpy as np


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_transforms.slang"))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_values_basic_input_transform(device_type: DeviceType):
    # Really simple test that just copies values from one buffer to another
    # with a transform involved

    m = load_test_module(device_type)

    # Create input+output buffers
    a = NDBuffer(device=m.device, shape=(2, 2), element_type=float)
    b = NDBuffer(device=m.device, shape=(2, 2), element_type=float)

    # Populate input
    a_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    a.from_numpy(a_data)

    # Call function, which should copy to output with dimensions flipped
    func = m.copy_values.as_func()
    func = func.transform_input({
        'input': (1, 0),
    })
    func(a, b)

    # Get and verify output
    b_data = b.buffer.to_numpy().view(np.float32).reshape(-1, 2)
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

    a = NDBuffer(device=m.device, shape=(2, 3), element_type=float3)
    b = NDBuffer(device=m.device, shape=(3, 2), element_type=float3)

    a_data = np.random.rand(2, 3, 3).astype(np.float32)
    b_data = np.random.rand(3, 2, 3).astype(np.float32)

    a.from_numpy(a_data)
    b.from_numpy(b_data)

    func = m.add_vectors.transform_input({
        'a': (1, 0),
    }).as_func()

    res: NDBuffer = func(a, b)

    assert res.shape == (3, 2)

    res_data = res.buffer.to_numpy().view(np.float32).reshape(3, 2, 3)

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

    a = NDBuffer(device=m.device, shape=(2, 3, 3), element_type=float)
    b = NDBuffer(device=m.device, shape=(3, 2), element_type=float3)

    a_data = np.random.rand(2, 3, 3).astype(np.float32)
    b_data = np.random.rand(3, 2, 3).astype(np.float32)

    a.from_numpy(a_data)
    b.from_numpy(b_data)

    func = m.add_vectors.transform_input({
        'a': (1, 0, 2),
    }).as_func()

    res: NDBuffer = func(a, b)

    assert res.shape == (3, 2)

    res_data = res.buffer.to_numpy().view(np.float32).reshape(3, 2, 3)

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

    inn = NDBuffer(device=m.device, shape=(2, 3, 3), element_type=float)
    out = NDBuffer(device=m.device, shape=(3, 2), element_type=float3)

    inn_data = np.random.rand(2, 3, 3).astype(np.float32)
    inn.from_numpy(inn_data)

    func = m.copy_vectors.transform_input({
        'input': (1, 0, 2),
    }).as_func()

    func(inn, out)

    out_data = out.buffer.to_numpy().view(np.float32).reshape(3, 2, 3)

    for i in range(3):
        for j in range(2):
            inn = inn_data[j, i]
            out = out_data[i, j]
            assert np.allclose(inn, out)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_vectors_vecindex_outputcontainer_input_transform(device_type: DeviceType):
    m = load_test_module(device_type)

    # Test remapping when one of the inputs is an 3D buffer of floats
    # instead of 2D buffer of float3s. In this case, the remapping
    # involves only the lower 2 dimensions (i.e. those of the container)

    inn = NDBuffer(device=m.device, shape=(2, 3, 3), element_type=float)
    out = NDBuffer(device=m.device, shape=(3, 2), element_type=float3)

    inn_data = np.random.rand(2, 3, 3).astype(np.float32)
    inn.from_numpy(inn_data)

    func = m.copy_vectors.transform_input({
        'output': (1, 0),
    }).as_func()

    func(inn, out)

    out_data = out.buffer.to_numpy().view(np.float32).reshape(3, 2, 3)

    for i in range(2):
        for j in range(3):
            inn = inn_data[i, j]
            out = out_data[j, i]
            assert np.allclose(inn, out)


# @pytest.mark.skip("Not yet supporting transforms witin element")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_add_vectors_vecindex_element_input_transform(device_type: DeviceType):
    m = load_test_module(device_type)

    # Test remapping when one of the inputs is an 3D buffer of floats
    # instead of 2D buffer of float3s.

    # The current problem with this case is that we're asking to remap
    # a dimension within A that is at the element level of the inputs
    # to the slang function. i.e. the slang function is:
    # copy_vectors(float3 A, float3 B)

    # A is a (3,2) buffer of floats, so we're effectively asking to swizzle
    # around the load the elements of a single float3 value from A.

    # Currently, the transform code works at the container level, so this
    # is broken!

    m = load_test_module(device_type)

    # Create input+output buffers
    a = NDBuffer(device=m.device, shape=(3, 2), element_type=float)
    b = NDBuffer(device=m.device, shape=(2,), element_type=float3)

    # Populate input
    a_data = np.random.rand(3, 2).astype(np.float32)
    a.from_numpy(a_data)

    # Call function, which should copy to output with dimensions flipped
    func = m.copy_vectors.as_func()
    func = func.transform_input({
        'input': (1, 0),
    })
    func(a, b)

    # Get and verify output
    b_data = b.buffer.to_numpy().view(np.float32).reshape(-1, 3)
    for i in range(1):
        for j in range(3):
            a = a_data[j, i]
            b = b_data[i, j]
            assert a == b


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_add_vectors_basic_output_transform(device_type: DeviceType):
    m = load_test_module(device_type)

    # Test the output transform, where we take 2 1D buffers with different
    # sizes and braodcast each to a different dimension.

    a = NDBuffer(device=m.device, shape=(5,), element_type=float3)
    b = NDBuffer(device=m.device, shape=(10,), element_type=float3)

    a_data = np.random.rand(5, 3).astype(np.float32)
    b_data = np.random.rand(10, 3).astype(np.float32)

    a.from_numpy(a_data)
    b.from_numpy(b_data)

    func = m.add_vectors.transform_output({
        'a': (0,),
        'b': (1,)
    }).as_func()

    res: NDBuffer = func(a, b)

    assert res.shape == (5, 10)

    res_data = res.buffer.to_numpy().view(np.float32).reshape(5, 10, 3)

    for i in range(5):
        for j in range(10):
            a = a_data[i]
            b = b_data[j]
            expected = a + b
            r = res_data[i, j]
            assert np.allclose(r, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
