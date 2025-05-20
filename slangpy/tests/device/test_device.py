# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import pytest
import slangpy as spy
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import sglhelpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_enumerate_adapters(device_type: spy.DeviceType):
    print(spy.Device.enumerate_adapters(type=device_type))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_device(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    assert device.desc.type == device_type
    assert device.desc.enable_debug_layers == True

    assert device.info.type == device_type
    assert device.info.adapter_name != ""
    API_NAMES = {
        spy.DeviceType.d3d12: "D3D12",
        spy.DeviceType.vulkan: "Vulkan",
        spy.DeviceType.metal: "Metal",
        spy.DeviceType.cuda: "CUDA",
    }
    assert device.info.api_name == API_NAMES[device_type]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_device_close_handler(device_type: spy.DeviceType):

    # Create none-cached device.
    device = helpers.get_device(device_type, use_cache=False)

    # Define a callback that increments a counter and captures the closed device.
    count = 0
    closed_device: Optional[spy.Device] = None

    def on_close(cd: spy.Device):
        nonlocal count
        nonlocal device
        nonlocal closed_device
        assert cd == device
        count += 1
        closed_device = cd

    # Register device, then close it.
    device.register_device_close_callback(on_close)
    device.close()

    # Check that the callback was called.
    assert count == 1
    assert closed_device is not None

    # Null the device, but the captured reference should still be
    # valid, so it won't be GC yet.
    device = None

    # Call close on the already closed device. Should be safe as it
    # hasn't been garbage collected, but should have no effect as
    # device is already closed.
    closed_device.close()
    assert count == 1


# Checks fix for alignment issues when creating/accessing a small buffer,
# followed by creating/accessing a texture.
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_global_buffer_alignment(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    # Create a small 16B buffer.
    small_buffer = device.create_buffer(
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
        size=16,
        label="count_buffer",
        data=np.array([0, 0, 0, 0], dtype=np.uint32),
    )

    # Read it, resulting in temporary allocation in the device's read back heap of 16B.
    small_buffer.to_numpy()

    # Data to populate the texture with.
    texture_data = np.random.rand(256, 256, 4).astype(np.float32).flatten()

    # Create an RGBA float texture.
    texture = device.create_texture(
        format=spy.Format.rgba32_float,
        width=256,
        height=256,
        usage=spy.TextureUsage.shader_resource,
        label="render_texture",
        data=texture_data,
    )

    # Read it, resulting in temporary allocation in the device's read back heap of 256*256*4*4B.
    val = texture.to_numpy().astype(np.float32).flatten()
    assert np.allclose(val, texture_data, atol=1e-6)


# Tests the hot reload event callback.
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hot_reload_event(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type, use_cache=False)

    # Load a shader
    program = device.load_program(
        module_name="test_shader_foo.slang",
        entry_point_names=["main_a", "main_b", "main_vs", "main_fs"],
    )

    # Setup a hook that increments a counter on hot reload.
    count = 0

    def inc_count(x: spy.ShaderHotReloadEvent):
        nonlocal count
        count += 1

    device.register_shader_hot_reload_callback(inc_count)

    # Force hot reload.
    device.reload_all_programs()

    # Check count.
    assert count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
