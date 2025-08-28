# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from slangpy import DeviceType, NDBuffer
from slangpy.testing import helpers

MODULE = r"""
struct Foo {
    int x;
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_buffer_to_string(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buffer = NDBuffer.zeros(device, shape=(10, 20), dtype="float")

    # Test that repr() returns a meaningful string
    repr_str = repr(buffer)
    print(f"NDBuffer: {repr_str}")

    # Verify the repr contains expected information
    assert "NativeNDBuffer" in repr_str
    assert "shape" in repr_str
    assert "strides" in repr_str
    assert "dtype" in repr_str


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slangtype_to_string(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)
    foo_struct = module["Foo"].as_struct()

    # Access the underlying SlangType
    foo_type = foo_struct.struct

    # Test that repr() returns a meaningful string
    repr_str = repr(foo_type)
    print(f"SlangType: {repr_str}")

    # Verify the repr contains expected information
    assert "NativeSlangType" in repr_str
    assert "name" in repr_str
    assert "shape" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
