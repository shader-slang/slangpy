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

    with pytest.raises(Exception, match=r"Buffers bound to RW"):
        module.write_only(ro_buffer)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
