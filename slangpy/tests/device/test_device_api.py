# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_push_pop_device(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    spy.push_device(device)
    assert spy.current_device() is device
    spy.pop_device()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_current_device_throws_when_empty(device_type: spy.DeviceType):
    with pytest.raises(Exception, match="No current device"):
        spy.current_device()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pop_device_throws_when_empty(device_type: spy.DeviceType):
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
    with device:
        pass
    with pytest.raises(Exception, match="No current device"):
        spy.current_device()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_context_manager_pops_on_exception(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    with pytest.raises(RuntimeError):
        with device:
            raise RuntimeError("test")
    with pytest.raises(Exception, match="No current device"):
        spy.current_device()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_nested_context_managers(device_type: spy.DeviceType):
    device1 = helpers.get_device(device_type)
    device2 = helpers.get_device(device_type, use_cache=False)
    with device1:
        assert spy.current_device() is device1
        with device2:
            assert spy.current_device() is device2
        assert spy.current_device() is device1


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
