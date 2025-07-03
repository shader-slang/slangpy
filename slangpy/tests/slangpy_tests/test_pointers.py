# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import sys
import numpy as np
import pytest
from slangpy import grid
from slangpy import DeviceType
from slangpy.types import NDBuffer


sys.path.append(str(Path(__file__).parent))
import helpers

# Filter default device types to only include those that support pointers
POINTER_DEVICE_TYPES = [
    x
    for x in helpers.DEFAULT_DEVICE_TYPES
    if x in [DeviceType.vulkan, DeviceType.cuda, DeviceType.metal]
]


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_return_pointer_value(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "return_pointer",
        r"""
int* return_pointer(int* ptr) {
    return ptr;
}
""",
    )

    res = function(100)

    assert res == 100


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_call_function(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
int add_numbers(int call_id, int* a_buffer, int* b_buffer) {
    int a = a_buffer[call_id];
    int b = b_buffer[call_id];
    return a + b;
}
""",
    )

    a = NDBuffer.from_numpy(device, np.array([1, 2, 3, 4], dtype=np.int32))
    b = NDBuffer.from_numpy(device, np.array([5, 6, 7, 8], dtype=np.int32))

    a_address = a.storage.device_address
    b_address = b.storage.device_address

    res = function(grid(shape=(4,)), a_address, b_address, _result="numpy")

    expected = np.array([6, 8, 10, 12], dtype=np.int32)
    assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
