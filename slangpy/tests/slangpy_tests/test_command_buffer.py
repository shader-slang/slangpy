# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc

import pytest
import numpy as np
from slangpy import Module, Tensor
from slangpy import CommandQueueType, DeviceType, float3
from slangpy.testing import helpers


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_modules.slang"))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("use_arg", [True, False])
def test_command_buffer(device_type: DeviceType, use_arg: bool):
    if device_type == DeviceType.metal:
        pytest.skip(
            "Metal backend can not atomically accumulate float3 types due to sizeof(float3) returning 12"
        )

    m = load_test_module(device_type)
    assert m is not None

    polynomial = m.polynomial.as_func()

    command_encoder = m.device.create_command_encoder()
    assert command_encoder.queue == CommandQueueType.graphics
    assert isinstance(command_encoder.recording_id, int)
    assert command_encoder.recording_id != 0

    discarded_recording_ids: list[int] = []
    second_discarded_recording_ids: list[int] = []
    unregistered_discarded_recording_ids: list[int] = []

    def on_discarded(event: object) -> None:
        assert event.device == m.device
        discarded_recording_ids.append(event.id)
        m.device.unregister_command_recording_discarded_callback(discarded_callback_id)

    def on_second_discarded(event: object) -> None:
        assert event.device == m.device
        second_discarded_recording_ids.append(event.id)

    def on_unregistered_discarded(event: object) -> None:
        unregistered_discarded_recording_ids.append(event.id)

    discarded_callback_id = m.device.register_command_recording_discarded_callback(on_discarded)
    second_discarded_callback_id = m.device.register_command_recording_discarded_callback(
        on_second_discarded
    )
    unregistered_discarded_callback_id = m.device.register_command_recording_discarded_callback(
        on_unregistered_discarded
    )
    assert isinstance(discarded_callback_id, int)
    assert isinstance(second_discarded_callback_id, int)
    assert discarded_callback_id != unregistered_discarded_callback_id
    m.device.unregister_command_recording_discarded_callback(unregistered_discarded_callback_id)

    discarded_command_encoder = m.device.create_command_encoder()
    discarded_recording_id = discarded_command_encoder.recording_id
    del discarded_command_encoder
    gc.collect()
    assert discarded_recording_ids == [discarded_recording_id]
    assert second_discarded_recording_ids == [discarded_recording_id]
    assert unregistered_discarded_recording_ids == []
    m.device.unregister_command_recording_discarded_callback(discarded_callback_id)
    m.device.unregister_command_recording_discarded_callback(second_discarded_callback_id)

    a = Tensor.empty(m.device, (10,), dtype=float3).with_grads()
    b = Tensor.empty(m.device, (10,), dtype=float3).with_grads()
    res = Tensor.empty(m.device, (10,), dtype=float3).with_grads()
    assert a.grad is not None
    assert b.grad is not None
    assert res.grad is not None

    a_data = np.random.rand(10, 3).astype(np.float32)
    b_data = np.random.rand(10, 3).astype(np.float32)
    res_data = np.zeros((10, 3), dtype=np.float32)

    helpers.write_tensor_from_numpy(a, a_data.flatten(), 3)
    helpers.write_tensor_from_numpy(b, b_data.flatten(), 3)
    helpers.write_tensor_from_numpy(res, res_data.flatten(), 3)
    helpers.write_tensor_from_numpy(res.grad, np.ones_like(res_data).flatten(), 3)

    if use_arg:
        polynomial(a, b, _result=res, _append_to=command_encoder)
        polynomial.bwds(a, b, _result=res, _append_to=command_encoder)
    else:
        polynomial.append_to(command_encoder, a, b, _result=res)
        polynomial.bwds.append_to(command_encoder, a, b, _result=res)

    # Nothing should have happened yet if command buffer is not submitted!
    res_data = helpers.read_tensor_from_numpy(res).reshape(-1, 3)
    assert not np.allclose(res_data, a_data * a_data + b_data + 1)

    # Submit the command buffer to execute the operations
    submitted_events: list[tuple[int, int, int]] = []
    second_submitted_events: list[tuple[int, int, int]] = []
    unregistered_submitted_events: list[int] = []

    def on_submitted(event: object) -> None:
        assert event.device == m.device
        submitted_events.append((event.id, event.command_buffer.recording_id, event.submit_id))
        m.device.unregister_command_recording_submitted_callback(submitted_callback_id)

    def on_second_submitted(event: object) -> None:
        assert event.device == m.device
        second_submitted_events.append(
            (event.id, event.command_buffer.recording_id, event.submit_id)
        )

    def on_unregistered_submitted(event: object) -> None:
        unregistered_submitted_events.append(event.id)

    submitted_callback_id = m.device.register_command_recording_submitted_callback(on_submitted)
    second_submitted_callback_id = m.device.register_command_recording_submitted_callback(
        on_second_submitted
    )
    unregistered_submitted_callback_id = m.device.register_command_recording_submitted_callback(
        on_unregistered_submitted
    )
    assert isinstance(submitted_callback_id, int)
    assert isinstance(second_submitted_callback_id, int)
    assert submitted_callback_id != unregistered_submitted_callback_id
    m.device.unregister_command_recording_submitted_callback(unregistered_submitted_callback_id)

    command_buffer = command_encoder.finish()
    assert command_buffer.queue == command_encoder.queue
    assert command_buffer.recording_id == command_encoder.recording_id
    m.device.submit_command_buffer(command_buffer)
    assert len(submitted_events) == 1
    assert len(second_submitted_events) == 1
    submitted_recording_id, submitted_command_buffer_recording_id, submit_id = submitted_events[0]
    (
        second_submitted_recording_id,
        second_submitted_command_buffer_recording_id,
        second_submit_id,
    ) = second_submitted_events[0]
    assert submitted_recording_id == command_buffer.recording_id
    assert submitted_command_buffer_recording_id == command_buffer.recording_id
    assert submit_id != 0
    assert second_submitted_recording_id == command_buffer.recording_id
    assert second_submitted_command_buffer_recording_id == command_buffer.recording_id
    assert second_submit_id == submit_id
    assert unregistered_submitted_events == []
    m.device.unregister_command_recording_submitted_callback(submitted_callback_id)
    m.device.unregister_command_recording_submitted_callback(second_submitted_callback_id)

    # Now the result should be computed
    res_data = helpers.read_tensor_from_numpy(res).reshape(-1, 3)
    assert np.allclose(res_data, a_data * a_data + b_data + 1)

    a_grad = helpers.read_tensor_from_numpy(a.grad).reshape(-1, 3)
    b_grad = helpers.read_tensor_from_numpy(b.grad).reshape(-1, 3)
    assert np.allclose(a_grad, 2 * a_data)
    assert np.allclose(b_grad, np.ones_like(b_data))

    # Also check nothing dies when calling function directly with a None encoder
    polynomial(a, b, _result=res, _append_to=None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reuse_finished_command_encoder(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    polynomial = m.polynomial.as_func()

    a = Tensor.empty(m.device, (10,), dtype=float3)
    b = Tensor.empty(m.device, (10,), dtype=float3)
    res = Tensor.empty(m.device, (10,), dtype=float3)

    command_encoder = m.device.create_command_encoder()
    polynomial.append_to(command_encoder, a, b, _result=res)
    m.device.submit_command_buffer(command_encoder.finish())

    # Reusing a finished command encoder must raise a clean error rather than crash.
    with pytest.raises(Exception, match="Command encoder is finished"):
        polynomial.append_to(command_encoder, a, b, _result=res)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
