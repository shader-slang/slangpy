import pytest
import sgl
import numpy as np
from kernelfunctions.buffer import StructuredBuffer
import kernelfunctions.tests.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_function_scalar(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(float2 a, float2 b) {
}
""",
    )

    function(sgl.float2(1, 2), sgl.float2(2, 3))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_function_broadcast(device_type: sgl.DeviceType):

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
    a = sgl.float2(a_data[0], a_data[1])

    b_data = np.random.rand(100, 2).astype(np.float32)
    b = StructuredBuffer(device=device, element_count=100, element_type=sgl.float2)
    b.buffer.from_numpy(b_data)

    res = StructuredBuffer(device=device, element_count=100, element_type=sgl.float2)

    function(a, b, res)

    expected = a_data + b_data
    actual = res.buffer.to_numpy().view(np.float32).reshape(-1, 2)

    assert np.allclose(expected, actual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
