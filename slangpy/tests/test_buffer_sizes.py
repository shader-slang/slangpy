# SPDX-License-Identifier: Apache-2.0
import pytest

from slangpy.backend import DeviceType
from slangpy.tests import helpers
from slangpy.types import Tensor


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_simple_int_buffer_shape(device_type: DeviceType):

    device = helpers.get_device(device_type)

    buffer = Tensor.empty(device, dtype=float, shape=(10,))
    assert buffer.shape == (10,)
    assert buffer.strides == (1,)
    assert buffer.storage.size == 40


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_2d_int_buffer(device_type: DeviceType):

    device = helpers.get_device(device_type)

    buffer = Tensor.empty(device, dtype=float, shape=(10, 5))
    assert buffer.shape == (10, 5)
    assert buffer.strides == (5, 1)
    assert buffer.storage.size == 200


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_3d_int_buffer(device_type: DeviceType):

    device = helpers.get_device(device_type)

    buffer = Tensor.empty(device, dtype=float, shape=(8, 10, 5))
    assert buffer.shape == (8, 10, 5)
    assert buffer.strides == (50, 5, 1)
    assert buffer.storage.size == 1600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
