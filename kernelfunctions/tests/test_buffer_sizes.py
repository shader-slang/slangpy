import pytest
import sgl
from kernelfunctions.types import NDDifferentiableBuffer
from kernelfunctions.tests import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_simple_int_buffer_elcount(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)

    buffer = NDDifferentiableBuffer(device, element_type=int, element_count=10)
    assert buffer.element_count == 10
    assert buffer.shape == (10,)
    assert buffer.element_type == int
    assert buffer.strides == (1,)
    assert buffer.element_size == 4
    assert buffer.buffer.size == 40


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_simple_int_buffer_shape(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)

    buffer = NDDifferentiableBuffer(device, element_type=int, shape=(10,))
    assert buffer.element_count == 10
    assert buffer.shape == (10,)
    assert buffer.element_type == int
    assert buffer.strides == (1,)
    assert buffer.element_size == 4
    assert buffer.buffer.size == 40


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_2d_int_buffer(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)

    buffer = NDDifferentiableBuffer(device, element_type=int, shape=(10, 5))
    assert buffer.element_count == 50
    assert buffer.shape == (10, 5)
    assert buffer.element_type == int
    assert buffer.strides == (5, 1)
    assert buffer.element_size == 4
    assert buffer.buffer.size == 200


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_3d_int_buffer(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)

    buffer = NDDifferentiableBuffer(device, element_type=int, shape=(8, 10, 5))
    assert buffer.element_count == 400
    assert buffer.shape == (8, 10, 5)
    assert buffer.element_type == int
    assert buffer.strides == (50, 5, 1)
    assert buffer.element_size == 4
    assert buffer.buffer.size == 1600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
