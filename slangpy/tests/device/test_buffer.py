# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
import slangpy as spy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import sglhelpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_buffer_init_data(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    data = np.random.randint(0, 0xFFFFFFFF, size=1024, dtype=np.uint32)

    # init data must match the size of the buffer
    with pytest.raises(Exception):
        buffer = device.create_buffer(
            size=4 * 1024 - 1,
            usage=spy.BufferUsage.shader_resource,
            data=data,
        )

    # init data must match the size of the buffer
    with pytest.raises(Exception):
        buffer = device.create_buffer(
            size=4 * 1024 + 1,
            usage=spy.BufferUsage.shader_resource,
            data=data,
        )

    buffer = device.create_buffer(
        size=4 * 1024,
        usage=spy.BufferUsage.shader_resource,
        data=data,
    )
    readback = buffer.to_numpy().view(np.uint32)
    assert np.all(data == readback)


# TODO we should also test buffers bound as root descriptors in D3D12 (allow larger buffers)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "type",
    [
        "byte_address_buffer",
        "buffer_uint",
        "structured_buffer_uint",
    ],
)
@pytest.mark.parametrize("size_MB", [128, 1024, 2048, 3072, 4096])
def test_buffer(device_type: spy.DeviceType, type: str, size_MB: int):
    device = helpers.get_device(device_type)

    if device_type == spy.DeviceType.d3d12 and size_MB > 2048:
        pytest.skip("D3D12 does not support buffers > 2048MB if not bound as in a root descriptor")

    if device_type == spy.DeviceType.vulkan and type == "buffer_uint":
        pytest.skip("Test currently failing with Vulkan")
    if device_type == spy.DeviceType.vulkan and type == "buffer_uint" and size_MB > 128:
        pytest.skip("Vulkan does not support large type buffers (storage buffers)")
    if device_type == spy.DeviceType.metal and type == "buffer_uint":
        pytest.skip("Test currently failing with Metal")
    if device_type == spy.DeviceType.metal and size_MB >= 1024:
        pytest.skip("Metal does not support buffers >= 1024MB")
    if (
        device_type == spy.DeviceType.vulkan
        and type == "byte_address_buffer"
        and sys.platform == "darwin"
        and size_MB >= 4000
    ):
        pytest.skip("MoltenVK does not support large byte buffers")
    if device_type == spy.DeviceType.cuda and type == "buffer_uint":
        pytest.skip("CUDA does not support Buffer/RWBuffer resources")
    if device_type == spy.DeviceType.cuda and size_MB > 1024:
        pytest.skip("Large buffers sometimes lead to crashes on CUDA")

    element_size = 4
    size = size_MB * 1024 * 1024

    # Vulkan does not support actual 4GB buffers, but 4GB - 1B
    if device_type == spy.DeviceType.vulkan and size >= 4096 * 1024 * 1024:
        size -= element_size

    # create device local buffer
    device_buffer = device.create_buffer(
        size=size,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    # check we can get usage
    assert (
        device_buffer.desc.usage
        == spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    )

    element_count = device_buffer.size // element_size
    check_count = 1024
    check_offsets = [
        0,
        (element_count * 1) // 4,
        (element_count * 2) // 4,
        (element_count * 3) // 4,
        element_count - check_count,
    ]

    # create upload buffer
    write_buffer = device.create_buffer(
        size=check_count * element_size,
        usage=spy.BufferUsage.shader_resource,
    )

    # create read-back buffer
    read_buffer = device.create_buffer(
        size=check_count * element_size,
        usage=spy.BufferUsage.unordered_access,
    )

    copy_kernel = device.create_compute_kernel(
        device.load_program("test_buffer.slang", ["copy_" + type])
    )

    for offset in check_offsets:
        data = np.random.randint(0, 0xFFFFFFFF, size=check_count, dtype=np.uint32)
        write_buffer.copy_from_numpy(data)
        copy_kernel.dispatch(
            thread_count=[element_count, 1, 1],
            src=write_buffer,
            dst=device_buffer,
            src_offset=0,
            dst_offset=offset,
            count=check_count,
        )
        copy_kernel.dispatch(
            thread_count=[check_count, 1, 1],
            src=device_buffer,
            dst=read_buffer,
            src_offset=offset,
            dst_offset=0,
            count=check_count,
        )
        readback = read_buffer.to_numpy().view(np.uint32)
        assert np.all(data == readback)

    # Set allocated resources to None and have the device wait
    # to ensure resources are cleaned up. Running the tests on devices with lower
    # amounts of available GPU memory can result in failures without clean up.
    device_buffer = None
    write_buffer = None
    read_buffer = None
    copy_kernel = None
    device.wait_for_idle()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_upload_buffer(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    buffer = device.create_buffer(
        size=4 * 1024,
        usage=spy.BufferUsage.unordered_access,
    )

    data = np.random.randint(0, 0xFFFFFFFF, size=1024, dtype=np.uint32)

    buffer.copy_from_numpy(np.zeros_like(data))

    encoder = device.create_command_encoder()
    encoder.upload_buffer_data(buffer, 0, data)
    device.submit_command_buffer(encoder.finish())

    readback = buffer.to_numpy().view(np.uint32)
    assert np.all(data == readback)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_upload_buffer_with_offset(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    buffer = device.create_buffer(
        size=4 * 1024,
        usage=spy.BufferUsage.unordered_access,
    )

    data = np.random.randint(0, 0xFFFFFFFF, size=512, dtype=np.uint32)

    buffer.copy_from_numpy(np.zeros_like(data))

    encoder = device.create_command_encoder()
    encoder.upload_buffer_data(buffer, 2048, data)
    device.submit_command_buffer(encoder.finish())

    readback = buffer.to_numpy().view(np.uint32)

    # readback should be all zeros, except for the last 512 bytes
    expected = np.zeros(1024, dtype=np.uint32)
    expected[512:] = data
    assert np.all(expected == readback)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_upload_buffer_overflow_fail(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    buffer = device.create_buffer(
        size=4 * 1024,
        usage=spy.BufferUsage.unordered_access,
    )
    data = np.random.randint(0, 0xFFFFFFFF, size=1024, dtype=np.uint32)

    with pytest.raises(
        RuntimeError, match=".*upload would exceed the size of the destination buffer.*"
    ):
        encoder = device.create_command_encoder()
        encoder.upload_buffer_data(buffer, 2048, data)
        device.submit_command_buffer(encoder.finish())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
