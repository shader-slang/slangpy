# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import slangpy as spy
import numpy as np
from slangpy import DeviceType
from slangpy.testing import helpers

MODULE = r"""
float read_only(StructuredBuffer<float> buffer) {
    return buffer[0];
}
void write_only(RWStructuredBuffer<float> buffer) {
    buffer[0] = 0.0f;
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_unstructured_buffer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    buffer = device.create_buffer(size=4, usage=spy.BufferUsage.shader_resource)
    buffer.copy_from_numpy(np.full((1,), 5.0, dtype="float32"))

    result = module.read_only(buffer)

    assert result == 5.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_buffer_usage_error(device_type: DeviceType):

    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    ro_buffer = device.create_buffer(size=4, usage=spy.BufferUsage.shader_resource)

    with pytest.raises(
        Exception, match=r"Buffer\[rw=False\] does not match slang type RWStructuredBuffer"
    ):
        module.write_only(ro_buffer)


MODULE_BA = r"""
uint read_byte_buffer(ByteAddressBuffer buffer) {
    return buffer.Load(0);
}
void write_byte_buffer(RWByteAddressBuffer buffer, uint val) {
    buffer.Store(0, val);
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_byte_address_buffer_read(device_type: DeviceType):
    """ByteAddressBuffer read-only path in BufferMarshall gen_calldata."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE_BA)

    buffer = device.create_buffer(size=4, usage=spy.BufferUsage.shader_resource)
    buffer.copy_from_numpy(np.array([42], dtype=np.uint32))

    result = module.read_byte_buffer(buffer)
    assert result == 42


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_byte_address_buffer_write(device_type: DeviceType):
    """RWByteAddressBuffer path in BufferMarshall gen_calldata."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE_BA)

    buffer = device.create_buffer(
        size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    buffer.copy_from_numpy(np.array([0], dtype=np.uint32))

    module.write_byte_buffer(buffer, 99)

    result = buffer.to_numpy().view(np.uint32)
    assert result[0] == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
