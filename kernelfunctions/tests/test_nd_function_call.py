import pytest
import sgl
import kernelfunctions as kf
from kernelfunctions.buffer import StructuredBuffer
import kernelfunctions.tests.helpers as helpers
import numpy as np


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_function(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "add_numbers_nd",
        r"""
void add_numbers_nd(float a, float b, out float c) {
    c = a + b;
}
""",
    )

    a = StructuredBuffer(device, element_type=float, shape=(512, 256, 4))
    a.buffer.from_numpy(np.random.rand(a.element_count).astype(np.float32))

    b = StructuredBuffer(device, element_type=float, shape=(512, 256, 4))
    b.buffer.from_numpy(np.random.rand(a.element_count).astype(np.float32))

    c = StructuredBuffer(device, element_type=float, shape=(512, 256, 4))
    c.buffer.from_numpy(np.zeros(a.element_count, dtype=np.float32))

    function(a, b, c)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
