# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy
from slangpy.testing import helpers


@pytest.fixture
def empty_device_stack():
    """Save and restore the device stack, providing an empty stack for the test."""
    saved = []
    while True:
        try:
            saved.append(spy.current_device())
            spy.pop_device()
        except Exception:
            break
    yield
    # Clean up anything the test left on the stack.
    while True:
        try:
            spy.current_device()
            spy.pop_device()
        except Exception:
            break
    # Restore original stack.
    for dev in reversed(saved):
        spy.push_device(dev)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_push_pop_device(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    spy.push_device(device)
    assert spy.current_device() is device
    spy.pop_device()


def test_current_device_throws_when_empty(empty_device_stack: None):
    with pytest.raises(Exception, match="No current device"):
        spy.current_device()


def test_pop_device_throws_when_empty(empty_device_stack: None):
    with pytest.raises(Exception, match="No device to pop"):
        spy.pop_device()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_context_manager(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    with device:
        assert spy.current_device() is device


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_context_manager_pops_on_exit(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    # Record stack state before with block.
    before_device = None
    try:
        before_device = spy.current_device()
    except Exception:
        pass
    with device:
        pass
    # Stack should be restored to pre-with state.
    after_device = None
    try:
        after_device = spy.current_device()
    except Exception:
        pass
    assert after_device is before_device


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_context_manager_pops_on_exception(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    # Record stack state before with block.
    before_device = None
    try:
        before_device = spy.current_device()
    except Exception:
        pass
    with pytest.raises(RuntimeError):
        with device:
            raise RuntimeError("test")
    # Stack should be restored to pre-with state.
    after_device = None
    try:
        after_device = spy.current_device()
    except Exception:
        pass
    assert after_device is before_device


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_nested_context_managers(device_type: spy.DeviceType):
    device1 = helpers.get_device(device_type)
    device2 = helpers.get_device(device_type, use_cache=False)
    with device1:
        assert spy.current_device() is device1
        with device2:
            assert spy.current_device() is device2
        assert spy.current_device() is device1
    device2.close()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_buffer_with_context(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    if device_type == spy.DeviceType.cuda:
        pytest.skip("CUDA does not support create_buffer with size")
    with device:
        buffer = spy.current_device().create_buffer(size=256)
        assert buffer is not None
        assert buffer.size == 256


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_context_manager_returns_device(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    with device as d:
        assert d is device


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_free_create_buffer(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    with device:
        # Test desc overload.
        desc = spy.BufferDesc()
        desc.size = 256
        desc.usage = spy.BufferUsage.shader_resource
        buffer = spy.create_buffer(desc)
        assert buffer is not None
        assert buffer.size == 256

        # Test kwargs overload.
        buffer2 = spy.create_buffer(size=512, usage=spy.BufferUsage.shader_resource)
        assert buffer2 is not None
        assert buffer2.size == 512


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_free_create_texture(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    with device:
        # Test desc overload.
        desc = spy.TextureDesc()
        desc.type = spy.TextureType.texture_2d
        desc.format = spy.Format.rgba8_unorm
        desc.width = 64
        desc.height = 64
        desc.usage = spy.TextureUsage.shader_resource
        texture = spy.create_texture(desc)
        assert texture is not None
        assert texture.width == 64
        assert texture.height == 64

        # Test kwargs overload.
        texture2 = spy.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba8_unorm,
            width=32,
            height=32,
            usage=spy.TextureUsage.shader_resource,
        )
        assert texture2 is not None
        assert texture2.width == 32
        assert texture2.height == 32


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_free_create_sampler(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    with device:
        # Test desc overload.
        sampler = spy.create_sampler(spy.SamplerDesc())
        assert sampler is not None

        # Test kwargs overload.
        sampler2 = spy.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
        )
        assert sampler2 is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_free_create_fence(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    with device:
        # Test desc overload.
        fence = spy.create_fence(spy.FenceDesc())
        assert fence is not None

        # Test kwargs overload.
        fence2 = spy.create_fence(initial_value=0, shared=False)
        assert fence2 is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_free_create_command_encoder(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    with device:
        encoder = spy.create_command_encoder()
        assert encoder is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_free_load_program(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    with device:
        program = spy.load_program(
            module_name="test_device_api",
            entry_point_names=["main"],
        )
        assert program is not None


# ---------------------------------------------------------------------------
# Device auto-push/pop tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_auto_push_on_create(device_type: spy.DeviceType):
    """Device constructor auto-pushes onto thread-local stack (like cuCtxCreate)."""
    device = helpers.get_device(device_type, use_cache=False)
    assert spy.current_device() is device
    device.close()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_auto_push_stacks(device_type: spy.DeviceType):
    """Creating multiple devices stacks them; most recent is current."""
    device_a = helpers.get_device(device_type, use_cache=False)
    device_b = helpers.get_device(device_type, use_cache=False)
    assert spy.current_device() is device_b
    spy.pop_device()
    assert spy.current_device() is device_a
    spy.pop_device()
    device_b.close()
    device_a.close()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_close_pops_if_on_top(device_type: spy.DeviceType, empty_device_stack: None):
    """close() pops the device if it is current (like cuCtxDestroy)."""
    device = helpers.get_device(device_type, use_cache=False)
    assert spy.current_device() is device
    device.close()
    with pytest.raises(Exception, match="No current device"):
        spy.current_device()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_close_does_not_pop_if_not_on_top(device_type: spy.DeviceType):
    """close() does NOT pop if device is not current (like cuCtxDestroy)."""
    device_a = helpers.get_device(device_type, use_cache=False)
    device_b = helpers.get_device(device_type, use_cache=False)
    # Stack = [A, B], current = B.
    device_a.close()
    # A is not on top, so it stays in the stack (stale, matches CUDA).
    assert spy.current_device() is device_b
    spy.pop_device()
    # Now A (closed) is on top - same as CUDA stale context behavior.
    assert spy.current_device() is device_a
    spy.pop_device()
    device_b.close()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_auto_push_with_context_manager(device_type: spy.DeviceType):
    """Auto-push + context manager stack correctly."""
    device = helpers.get_device(device_type, use_cache=False)
    # Stack = [device] from auto-push.
    with device:
        # Stack = [device, device].
        assert spy.current_device() is device
    # Stack = [device] after context manager pop.
    assert spy.current_device() is device
    device.close()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_after_close(device_type: spy.DeviceType):
    """After closing a device, creating a new one makes it current."""
    device_a = helpers.get_device(device_type, use_cache=False)
    assert spy.current_device() is device_a
    device_a.close()
    device_b = helpers.get_device(device_type, use_cache=False)
    assert spy.current_device() is device_b
    device_b.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
