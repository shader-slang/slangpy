# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType
from slangpy.types import NDBuffer


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_basic_3d_call(device_type: DeviceType):

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

    a = NDBuffer(device, dtype=float, shape=(2, 2))
    b = NDBuffer(device, dtype=float, shape=(2, 2))
    c = NDBuffer(device, dtype=float, shape=(2, 2))

    a_data = np.random.rand(*a.shape).astype(np.float32)
    b_data = np.random.rand(*b.shape).astype(np.float32)

    a.storage.from_numpy(a_data)
    b.storage.from_numpy(b_data)

    function(a, b, c)

    c_expected = a_data + b_data
    c_data = c.storage.to_numpy().view(np.float32).reshape(*c.shape)
    assert np.allclose(c_data, c_expected, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
