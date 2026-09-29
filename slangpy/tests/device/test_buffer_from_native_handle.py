# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers


# Native handle type accepted by Device.create_buffer_from_native_handle, per device type.
# Device types missing from this table have no buffer import in slang-rhi.
NATIVE_BUFFER_HANDLE_TYPES = {
    spy.DeviceType.d3d12: spy.NativeHandleType.D3D12Resource,
    spy.DeviceType.vulkan: spy.NativeHandleType.VkBuffer,
    spy.DeviceType.metal: spy.NativeHandleType.MTLBuffer,
    spy.DeviceType.cuda: spy.NativeHandleType.CUdeviceptr,
    spy.DeviceType.wgpu: spy.NativeHandleType.WGPUBuffer,
}

BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_buffer_from_native_handle(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    data = np.random.randint(0, 0xFFFFFFFF, size=1024, dtype=np.uint32)
    buffer = device.create_buffer(usage=BUFFER_USAGE, data=data)
    desc = {"size": buffer.size, "usage": BUFFER_USAGE}
    handle = buffer.native_handle

    if device_type not in NATIVE_BUFFER_HANDLE_TYPES:
        with pytest.raises(RuntimeError, match="not implemented"):
            device.create_buffer_from_native_handle(desc, handle)
        return

    assert handle
    assert handle.type == NATIVE_BUFFER_HANDLE_TYPES[device_type]

    # Wrapping the handle must alias the original allocation, not copy it.
    imported = device.create_buffer_from_native_handle(desc, handle)
    assert imported.size == buffer.size
    assert imported.native_handle.value == handle.value
    assert np.all(imported.to_numpy().view(np.uint32) == data)

    # Writes through the imported buffer are visible through the original one.
    new_data = np.random.randint(0, 0xFFFFFFFF, size=1024, dtype=np.uint32)
    encoder = device.create_command_encoder()
    encoder.upload_buffer_data(imported, 0, new_data)
    device.submit_command_buffer(encoder.finish())
    assert np.all(buffer.to_numpy().view(np.uint32) == new_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_buffer_from_native_handle_invalid(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    if device_type not in NATIVE_BUFFER_HANDLE_TYPES:
        pytest.skip(f"{device_type} cannot import native buffers")

    data = np.random.randint(0, 0xFFFFFFFF, size=1024, dtype=np.uint32)
    buffer = device.create_buffer(usage=BUFFER_USAGE, data=data)
    desc = {"size": buffer.size, "usage": BUFFER_USAGE}

    # A default constructed handle carries no resource.
    with pytest.raises(RuntimeError, match="Invalid native handle"):
        device.create_buffer_from_native_handle(desc, spy.NativeHandle())

    # The device handle is valid, but it is not a buffer.
    with pytest.raises(RuntimeError, match="Expected a native handle of type"):
        device.create_buffer_from_native_handle(desc, device.native_handles[0])


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
def test_native_handle_from_cuda_device_ptr(device_type: spy.DeviceType):
    if device_type not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("CUDA is not available")
    device = helpers.get_device(device_type)

    buffer = device.create_buffer(size=1024, usage=BUFFER_USAGE)

    # On CUDA a buffer handle is nothing but the device pointer, so a handle built from a raw
    # pointer (e.g. torch.Tensor.data_ptr()) is interchangeable with the buffer's own handle.
    handle = spy.NativeHandle.from_cuda_device_ptr(buffer.device_address)
    assert handle.type == spy.NativeHandleType.CUdeviceptr
    assert handle == buffer.native_handle


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
