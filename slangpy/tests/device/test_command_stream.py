# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_command_stream(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    stream = device.create_command_stream()
    assert stream is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_stream_copy_buffer(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    data = np.array([1, 2, 3, 4], dtype=np.float32)

    src = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source,
        data=data,
    )
    dst = device.create_buffer(
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.copy_destination,
        size=data.nbytes,
    )

    stream = device.create_command_stream()
    stream.copy_buffer(dst, 0, src, 0, data.nbytes)
    stream.flush()

    result = dst.to_numpy().view(np.float32)
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_stream_context_manager(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    data = np.array([10, 20, 30, 40], dtype=np.float32)

    src = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source,
        data=data,
    )
    dst = device.create_buffer(
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.copy_destination,
        size=data.nbytes,
    )

    with device.create_command_stream() as stream:
        stream.copy_buffer(dst, 0, src, 0, data.nbytes)

    result = dst.to_numpy().view(np.float32)
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_stream_submit_and_wait(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    data = np.array([5, 6, 7, 8], dtype=np.float32)

    src = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source,
        data=data,
    )
    dst = device.create_buffer(
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.copy_destination,
        size=data.nbytes,
    )

    stream = device.create_command_stream()
    stream.copy_buffer(dst, 0, src, 0, data.nbytes)
    submit_id = stream.submit()
    stream.wait(submit_id)

    result = dst.to_numpy().view(np.float32)
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_stream_multiple_submits(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    buf_a = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source,
        data=np.array([1, 2], dtype=np.float32),
    )
    buf_b = device.create_buffer(
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.copy_source
        | spy.BufferUsage.copy_destination,
        size=8,
    )
    buf_c = device.create_buffer(
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.copy_destination,
        size=8,
    )

    stream = device.create_command_stream()

    # First submit: copy a -> b
    stream.copy_buffer(buf_b, 0, buf_a, 0, 8)
    id1 = stream.submit()
    stream.wait(id1)

    # Second submit: copy b -> c
    stream.copy_buffer(buf_c, 0, buf_b, 0, 8)
    stream.flush()

    result = buf_c.to_numpy().view(np.float32)
    np.testing.assert_array_equal(result, np.array([1, 2], dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_stream_inherits_command_recorder(device_type: spy.DeviceType):
    """Verify CommandStream is a CommandRecorder subclass."""
    device = helpers.get_device(device_type)
    stream = device.create_command_stream()
    assert isinstance(stream, spy.CommandRecorder)
    assert isinstance(stream, spy.CommandStream)
