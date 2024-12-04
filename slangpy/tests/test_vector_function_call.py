import pytest
from slangpy.backend import DeviceType, float2
import numpy as np
from slangpy.types import NDBuffer
import slangpy.tests.helpers as helpers
from slangpy.types.buffer import NDDifferentiableBuffer


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_function_scalar(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(float2 a, float2 b) {
}
""",
    )

    function(float2(1, 2), float2(2, 3))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_function_broadcast(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(float2 a, float2 b, out float2 res) {
    res = a + b;
}
""",
    )

    a_data = np.random.rand(2).astype(np.float32)
    a = float2(a_data[0], a_data[1])

    b_data = np.random.rand(100, 2).astype(np.float32)
    b = NDBuffer(device=device, element_count=100, element_type=float2)
    b.buffer.from_numpy(b_data)

    res = NDBuffer(
        device=device, element_count=100, element_type=float2)

    function(a, b, res)

    expected = a_data + b_data
    actual = res.buffer.to_numpy().view(np.float32).reshape(-1, 2)

    assert np.allclose(expected, actual)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_float_buffer_against_vector(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(float2 a, float2 b, out float2 res) {
    res = a + b;
}
""",
    )

    a_data = np.random.rand(100, 2).astype(np.float32)
    a = NDBuffer(device=device, shape=(100, 2), element_type=float)
    a.buffer.from_numpy(a_data)

    b_data = np.random.rand(100, 2).astype(np.float32)
    b = NDBuffer(device=device, element_count=100, element_type=float2)
    b.buffer.from_numpy(b_data)

    res = NDBuffer(
        device=device, element_count=100, element_type=float2)

    function(a, b, res)

    expected = a_data + b_data
    actual = res.buffer.to_numpy().view(np.float32).reshape(-1, 2)

    assert np.allclose(expected, actual)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_float_buffer_against_vector_readwrite(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers_vecreadwrite",
        r"""
void add_numbers_vecreadwrite(float2 a, float2 b, out float2 res) {
    res = a + b;
}
""",
    )

    a_data = np.random.rand(100, 2).astype(np.float32)
    a = NDBuffer(device=device, shape=(100, 2), element_type=float)
    a.buffer.from_numpy(a_data)

    b_data = np.random.rand(100, 2).astype(np.float32)
    b = NDBuffer(device=device, element_count=100, element_type=float2)
    b.buffer.from_numpy(b_data)

    res = NDBuffer(device=device, shape=(100, 2), element_type=float)

    function(a, b, res)

    expected = a_data + b_data
    actual = res.buffer.to_numpy().view(np.float32).reshape(-1, 2)

    assert np.allclose(expected, actual)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_float_buffer_against_vector_diffpair(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers_diffpair",
        r"""
void add_numbers_diffpair(float2 a, float2 b, out float2 res) {
    res = a + b;
}
""",
    )

    a_data = np.random.rand(100, 2).astype(np.float32)
    a = NDDifferentiableBuffer(device=device, shape=(100, 2), element_type=float)
    a.buffer.from_numpy(a_data)

    b_data = np.random.rand(100, 2).astype(np.float32)
    b = NDBuffer(device=device, element_count=100, element_type=float2)
    b.buffer.from_numpy(b_data)

    res = NDBuffer(device=device, shape=(100, 2), element_type=float)

    function(a, b, res)

    expected = a_data + b_data
    actual = res.buffer.to_numpy().view(np.float32).reshape(-1, 2)

    assert np.allclose(expected, actual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])