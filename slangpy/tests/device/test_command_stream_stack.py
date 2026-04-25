# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_push_pop_current_command_stream(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    stream = device.create_command_stream()

    spy.push_command_stream(stream)
    assert spy.current_command_stream() is stream
    spy.pop_command_stream()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_stream_context_manager_pushes_and_pops(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    stream = device.create_command_stream()

    with stream:
        assert spy.current_command_stream() is stream

    # Stack should be empty after exiting context manager.
    with pytest.raises(RuntimeError):
        spy.current_command_stream()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_nested_command_streams(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    stream1 = device.create_command_stream()
    stream2 = device.create_command_stream()

    spy.push_command_stream(stream1)
    assert spy.current_command_stream() is stream1

    spy.push_command_stream(stream2)
    assert spy.current_command_stream() is stream2

    spy.pop_command_stream()
    assert spy.current_command_stream() is stream1

    spy.pop_command_stream()


def test_pop_empty_stack_raises():
    with pytest.raises(RuntimeError):
        spy.pop_command_stream()


def test_current_empty_stack_raises():
    with pytest.raises(RuntimeError):
        spy.current_command_stream()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_free_standing_copy_buffer(device_type: spy.DeviceType):
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

    with device.create_command_stream():
        spy.copy_buffer(dst, 0, src, 0, data.nbytes)

    result = dst.to_numpy().view(np.float32)
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_free_standing_clear_buffer(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    buf = device.create_buffer(
        usage=spy.BufferUsage.unordered_access,
        data=np.array([1, 2, 3, 4], dtype=np.float32),
    )

    with device.create_command_stream():
        spy.clear_buffer(buf)

    result = buf.to_numpy().view(np.float32)
    np.testing.assert_array_equal(result, np.zeros(4, dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_free_standing_submit_and_flush(device_type: spy.DeviceType):
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

    stream = device.create_command_stream()
    spy.push_command_stream(stream)
    try:
        spy.copy_buffer(dst, 0, src, 0, data.nbytes)
        submit_id = spy.submit()
        assert submit_id > 0
        stream.wait(submit_id)
    finally:
        spy.pop_command_stream()

    result = dst.to_numpy().view(np.float32)
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_free_standing_flush(device_type: spy.DeviceType):
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
    spy.push_command_stream(stream)
    try:
        spy.copy_buffer(dst, 0, src, 0, data.nbytes)
        spy.flush()
    finally:
        spy.pop_command_stream()

    result = dst.to_numpy().view(np.float32)
    np.testing.assert_array_equal(result, data)
