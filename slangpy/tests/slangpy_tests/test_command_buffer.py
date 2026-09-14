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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_recording_created_callback(device_type: DeviceType):
    device = helpers.get_device(device_type)

    created_events: list[tuple[int, int]] = []
    second_created_ids: list[int] = []
    unregistered_created_ids: list[int] = []

    def on_created(event: object) -> None:
        assert event.device == device
        assert event.encoder is not None
        # The encoder must be open and recordable inside the callback (the load-bearing
        # profiler use case): appending a balanced debug group must not raise.
        event.encoder.push_debug_group("spy-created", float3(0, 0, 0))
        event.encoder.pop_debug_group()
        created_events.append((event.id, event.encoder.recording_id))
        # Unregistering during notify must be safe (copy-on-write callback list).
        device.unregister_command_recording_created_callback(created_callback_id)

    def on_second_created(event: object) -> None:
        assert event.device == device
        second_created_ids.append(event.id)

    def on_unregistered_created(event: object) -> None:
        unregistered_created_ids.append(event.id)

    created_callback_id = device.register_command_recording_created_callback(on_created)
    second_created_callback_id = device.register_command_recording_created_callback(
        on_second_created
    )
    unregistered_created_callback_id = device.register_command_recording_created_callback(
        on_unregistered_created
    )
    assert isinstance(created_callback_id, int)
    assert created_callback_id != unregistered_created_callback_id
    device.unregister_command_recording_created_callback(unregistered_created_callback_id)

    # created fires synchronously during create_command_encoder, carrying the live encoder + its id.
    command_encoder = device.create_command_encoder()
    assert created_events == [(command_encoder.recording_id, command_encoder.recording_id)]
    assert second_created_ids == [command_encoder.recording_id]
    assert unregistered_created_ids == []

    # on_created unregistered itself during notify; a subsequent create must not re-invoke it.
    second_encoder = device.create_command_encoder()
    assert created_events == [(command_encoder.recording_id, command_encoder.recording_id)]
    assert second_created_ids == [command_encoder.recording_id, second_encoder.recording_id]

    device.unregister_command_recording_created_callback(second_created_callback_id)

    del command_encoder
    del second_encoder
    gc.collect()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_recording_created_covers_helper_encoders(device_type: DeviceType):
    # Encoders created internally by SGL (functional-API dispatch) route through
    # Device::create_command_encoder, so the created hook must fire for them too.
    m = load_test_module(device_type)
    assert m is not None
    polynomial = m.polynomial.as_func()

    a = Tensor.empty(m.device, (10,), dtype=float3)
    b = Tensor.empty(m.device, (10,), dtype=float3)
    res = Tensor.empty(m.device, (10,), dtype=float3)

    created_ids: list[int] = []

    def on_created(event: object) -> None:
        created_ids.append(event.id)

    callback_id = m.device.register_command_recording_created_callback(on_created)
    try:
        polynomial(a, b, _result=res)
    finally:
        m.device.unregister_command_recording_created_callback(callback_id)

    assert len(created_ids) >= 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_recording_before_finish_callback(device_type: DeviceType):
    device = helpers.get_device(device_type)

    sequence: list[str] = []
    before_finish_events: list[tuple[int, int]] = []
    second_before_finish_ids: list[int] = []
    unregistered_before_finish_ids: list[int] = []

    def on_created(event: object) -> None:
        sequence.append(f"created:{event.id}")

    def on_before_finish(event: object) -> None:
        assert event.device == device
        assert event.encoder is not None
        # Still open and recordable right before finish: appending must not raise.
        event.encoder.push_debug_group("spy-before-finish", float3(0, 0, 0))
        event.encoder.pop_debug_group()
        before_finish_events.append((event.id, event.encoder.recording_id))
        sequence.append(f"before_finish:{event.id}")
        device.unregister_command_recording_before_finish_callback(before_finish_callback_id)

    def on_second_before_finish(event: object) -> None:
        second_before_finish_ids.append(event.id)

    def on_unregistered_before_finish(event: object) -> None:
        unregistered_before_finish_ids.append(event.id)

    def on_submitted(event: object) -> None:
        sequence.append(f"submitted:{event.id}")

    created_callback_id = device.register_command_recording_created_callback(on_created)
    before_finish_callback_id = device.register_command_recording_before_finish_callback(
        on_before_finish
    )
    second_before_finish_callback_id = device.register_command_recording_before_finish_callback(
        on_second_before_finish
    )
    unregistered_before_finish_callback_id = (
        device.register_command_recording_before_finish_callback(on_unregistered_before_finish)
    )
    device.unregister_command_recording_before_finish_callback(
        unregistered_before_finish_callback_id
    )
    submitted_callback_id = device.register_command_recording_submitted_callback(on_submitted)

    # Discard path: encoder dropped without finish -> created fires, before_finish does NOT.
    discarded_encoder = device.create_command_encoder()
    discarded_id = discarded_encoder.recording_id
    del discarded_encoder
    gc.collect()
    assert f"created:{discarded_id}" in sequence
    assert before_finish_events == []

    # Finish path: created -> before_finish -> submitted, ids all equal.
    encoder = device.create_command_encoder()
    recording_id = encoder.recording_id
    command_buffer = encoder.finish()
    device.submit_command_buffer(command_buffer)

    assert before_finish_events == [(recording_id, recording_id)]
    assert second_before_finish_ids == [recording_id]
    assert unregistered_before_finish_ids == []
    assert (
        sequence.index(f"created:{recording_id}")
        < sequence.index(f"before_finish:{recording_id}")
        < sequence.index(f"submitted:{recording_id}")
    )

    device.unregister_command_recording_created_callback(created_callback_id)
    device.unregister_command_recording_before_finish_callback(second_before_finish_callback_id)
    device.unregister_command_recording_submitted_callback(submitted_callback_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
