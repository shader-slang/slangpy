# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
from typing import Any

from slangpy import DeviceType, BufferUsage, MemoryType
from slangpy.core.native import NativeNDBuffer
from slangpy.types import NDBuffer
from slangpy.testing import helpers


# ---------------------------------------------------------------------------
# NDBuffer construction and data transfer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_empty(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buf = NDBuffer.empty(device, shape=(10,), dtype=float)
    assert buf.shape == (10,)
    assert buf.is_writable


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_empty_readonly(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buf = NDBuffer.empty(
        device,
        shape=(8,),
        dtype=float,
        usage=BufferUsage.shader_resource,
    )
    assert buf.shape == (8,)
    assert not buf.is_writable


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_zeros(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buf = NDBuffer.zeros(device, shape=(16,), dtype=float)
    data = buf.to_numpy()
    assert data.shape[0] == 16
    assert np.allclose(data, 0.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_from_numpy(device_type: DeviceType):
    device = helpers.get_device(device_type)
    src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf = NDBuffer.from_numpy(device, src)
    out = buf.to_numpy()
    assert np.allclose(out, src)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_from_numpy_2d(device_type: DeviceType):
    device = helpers.get_device(device_type)
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    buf = NDBuffer.from_numpy(device, src)
    assert buf.shape == (3, 4)
    out = buf.to_numpy().reshape(3, 4)
    assert np.allclose(out, src)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_element_count(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buf = NDBuffer(device, dtype=float, element_count=20)
    assert buf.shape == (20,)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_empty_like(device_type: DeviceType):
    device = helpers.get_device(device_type)
    original = NDBuffer.empty(device, shape=(5, 3), dtype=float)
    clone = NDBuffer.empty_like(original)
    assert clone.shape == original.shape
    assert clone.dtype.full_name == original.dtype.full_name


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_zeros_like(device_type: DeviceType):
    device = helpers.get_device(device_type)
    original = NDBuffer.empty(device, shape=(4,), dtype=float)
    clone = NDBuffer.zeros_like(original)
    data = clone.to_numpy()
    assert np.allclose(data, 0.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_clear(device_type: DeviceType):
    device = helpers.get_device(device_type)
    src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    buf = NDBuffer.from_numpy(device, src)
    buf.clear()
    data = buf.to_numpy()
    assert np.allclose(data, 0.0)


# ---------------------------------------------------------------------------
# NDBuffer validation errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_requires_shape_or_count(device_type: DeviceType):
    device = helpers.get_device(device_type)
    with pytest.raises(ValueError, match="Either element_count or shape must be provided"):
        NDBuffer(device, dtype=float)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_rejects_both_shape_and_count(device_type: DeviceType):
    device = helpers.get_device(device_type)
    with pytest.raises(ValueError, match="Only one of element_count or shape"):
        NDBuffer(device, dtype=float, element_count=10, shape=(10,))


# ---------------------------------------------------------------------------
# NDBuffer views and broadcasting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_broadcast(device_type: DeviceType):
    device = helpers.get_device(device_type)
    src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    buf = NDBuffer.from_numpy(device, src)
    broadcasted = buf.broadcast_to((4, 3))
    assert broadcasted.shape == (4, 3)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_view(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buf = NDBuffer.empty(device, shape=(12,), dtype=float)
    v = buf.view(shape=(3, 4))
    assert v.shape == (3, 4)


# ---------------------------------------------------------------------------
# NDBuffer round-trip through Slang functions (exercises NDBufferMarshall)
# ---------------------------------------------------------------------------

ADD_SHADER = r"""
void add_buffers(float a, float b, out float c) {
    c = a + b;
}
"""

SCALE_SHADER = r"""
void scale_buffer(float a, float factor, out float result) {
    result = a * factor;
}
"""

READ_BUFFER_SHADER = r"""
float read_first(Tensor<float, 1> buf) {
    return buf[0];
}
"""

STRUCTURED_BUFFER_SHADER = r"""
float read_structured(StructuredBuffer<float> buf) {
    return buf[0];
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_add_1d(device_type: DeviceType):
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_buffers", ADD_SHADER)

    a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    a = NDBuffer.from_numpy(device, a_data)
    b = NDBuffer.from_numpy(device, b_data)
    c = NDBuffer.zeros(device, shape=(4,), dtype=float)

    func(a, b, c)

    c_out = c.to_numpy()
    assert np.allclose(c_out, a_data + b_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_add_2d(device_type: DeviceType):
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_buffers", ADD_SHADER)

    a_data = np.arange(6, dtype=np.float32).reshape(2, 3)
    b_data = np.ones((2, 3), dtype=np.float32) * 10

    a = NDBuffer.from_numpy(device, a_data)
    b = NDBuffer.from_numpy(device, b_data)
    c = NDBuffer.zeros(device, shape=(2, 3), dtype=float)

    func(a, b, c)

    c_out = c.to_numpy().reshape(2, 3)
    assert np.allclose(c_out, a_data + b_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_with_scalar_broadcast(device_type: DeviceType):
    """NDBuffer + scalar argument: scalar is broadcast across the buffer."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "scale_buffer", SCALE_SHADER)

    a_data = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    a = NDBuffer.from_numpy(device, a_data)
    result = NDBuffer.zeros(device, shape=(3,), dtype=float)

    func(a, 3.0, result)

    out = result.to_numpy()
    assert np.allclose(out, a_data * 3.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_read_only_pass(device_type: DeviceType):
    """Pass a read-only NDBuffer to a function expecting Tensor<float,1>."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "read_first", READ_BUFFER_SHADER)

    src = np.array([42.0, 1.0, 2.0], dtype=np.float32)
    buf = NDBuffer.from_numpy(device, src, usage=BufferUsage.shader_resource)

    result = func(buf)
    assert result == pytest.approx(42.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_as_structured_buffer(device_type: DeviceType):
    """Pass a 1D NDBuffer to a function expecting StructuredBuffer<float>."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "read_structured", STRUCTURED_BUFFER_SHADER)

    src = np.array([99.0, 1.0], dtype=np.float32)
    buf = NDBuffer.from_numpy(device, src, usage=BufferUsage.shader_resource)

    result = func(buf)
    assert result == pytest.approx(99.0)


RETURN_SHADER = r"""
float add_and_return(float a, float b) {
    return a + b;
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_as_return_type(device_type: DeviceType):
    """Use NDBuffer as the return type of a function with a return value."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_and_return", RETURN_SHADER).return_type(
        NDBuffer
    )

    a_data = np.array([1.0, 2.0], dtype=np.float32)
    b_data = np.array([3.0, 4.0], dtype=np.float32)
    a = NDBuffer.from_numpy(device, a_data)
    b = NDBuffer.from_numpy(device, b_data)

    result = func(a, b)
    assert isinstance(result, NativeNDBuffer)
    out = result.to_numpy()
    assert np.allclose(out, a_data + b_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_copy_from_numpy(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buf = NDBuffer.empty(device, shape=(4,), dtype=float)
    src = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    buf.copy_from_numpy(src)
    out = buf.to_numpy()
    assert np.allclose(out, src)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_int_dtype(device_type: DeviceType):
    device = helpers.get_device(device_type)
    src = np.array([1, 2, 3, 4], dtype=np.int32)
    buf = NDBuffer.from_numpy(device, src)
    out = buf.to_numpy()
    assert np.array_equal(out, src)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
