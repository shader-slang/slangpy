# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers

from typing import Optional


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
        entry_point_names=["main_a"],
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


@pytest.mark.parametrize(
    "device_type", [spy.DeviceType.cuda, spy.DeviceType.vulkan, spy.DeviceType.d3d12]
)
def test_device_import(device_type: spy.DeviceType):
    if not device_type in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip(f"Device type {device_type} not supported.")

    # Create new device
    device1 = spy.Device(
        type=device_type,
        enable_debug_layers=True,
        compiler_options={
            "debug_info": spy.SlangDebugInfoLevel.standard,
        },
        label=f"deviceimport-{device_type.name}-1",
    )

    # Create another device sharing the same handles
    device2 = spy.Device(
        type=device_type,
        enable_debug_layers=True,
        compiler_options={
            "debug_info": spy.SlangDebugInfoLevel.standard,
        },
        existing_device_handles=device1.native_handles,
        label=f"deviceimport-{device_type.name}-2",
    )

    # Verify handles match
    d1handles = device1.native_handles
    d2handles = device2.native_handles
    assert d1handles[0].type == d2handles[0].type
    assert d1handles[0].value == d2handles[0].value
    assert d1handles[1].type == d2handles[1].type
    assert d1handles[1].value == d2handles[1].value
    assert d1handles[2].type == d2handles[2].type
    assert d1handles[2].value == d2handles[2].value

    # In theory this little test verifies device2 can use the memory
    # allocated by device1, but it seems to work well even when not
    # sharing handles, presumably due to singletons at the driver level.
    buffer_size = 4 * 1024 * 1024
    data = (np.random.rand(buffer_size) * 255).astype(np.uint8)
    src_buffer = device1.create_buffer(
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
        size=buffer_size,
        data=data,
    )
    dst_buffer = device2.create_buffer(
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource, size=buffer_size
    )
    module = device2.load_module_from_source(
        module_name=f"copy_buffer_{device_type.name}",
        source=r"""
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void copy_kernel(uint1 tid: SV_DispatchThreadID, StructuredBuffer<int> src, RWStructuredBuffer<int> dst) {
            dst[tid.x] = src[tid.x];
        }
        """,
    )
    copy_kernel = device2.create_compute_kernel(
        device2.link_program([module], [module.entry_point("copy_kernel")])
    )
    copy_kernel.dispatch([buffer_size // 4, 1, 1], src=src_buffer, dst=dst_buffer)
    dst_data = dst_buffer.to_numpy()
    assert np.array_equal(data, dst_data)

    # Make sure device2 closes before device 1 is released.
    device2.close()
    device2 = None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_report_heaps(device_type: spy.DeviceType):
    """Test the report_heaps API."""
    device = helpers.get_device(device_type)

    # Call report_heaps - this should not throw an exception
    heap_reports = device.report_heaps()

    # Verify the return type is a list
    assert isinstance(heap_reports, list)

    # Verify each heap report has the expected structure
    for heap_report in heap_reports:
        assert hasattr(heap_report, "label")
        assert hasattr(heap_report, "num_pages")
        assert hasattr(heap_report, "total_allocated")
        assert hasattr(heap_report, "total_mem_usage")
        assert hasattr(heap_report, "num_allocations")
        assert isinstance(heap_report.label, str)

        # Verify the types are correct
        assert isinstance(heap_report.num_pages, int)
        assert isinstance(heap_report.total_allocated, int)
        assert isinstance(heap_report.total_mem_usage, int)
        assert isinstance(heap_report.num_allocations, int)

        # Verify values are non-negative
        assert heap_report.num_pages >= 0
        assert heap_report.total_allocated >= 0
        assert heap_report.total_mem_usage >= 0
        assert heap_report.num_allocations >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
