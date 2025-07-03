# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import sys
import numpy as np
import pytest
from slangpy import grid
from slangpy import DeviceType, BufferUsage
from slangpy.types import NDBuffer


sys.path.append(str(Path(__file__).parent))
import helpers

# Filter default device types to only include those that support pointers
POINTER_DEVICE_TYPES = [
    x
    for x in helpers.DEFAULT_DEVICE_TYPES
    if x in [DeviceType.vulkan, DeviceType.cuda, DeviceType.metal]
]


# Just makes sure the actual value of a pointer (not what it points at) can
# be read and returned correctly.
@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_copy_pointer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_copy_pointer",
        r"""
int* test_copy_pointer(int* ptr) {
    return ptr;
}
""",
    )

    res = function(100)

    assert res == 100


USAGES = [
    BufferUsage.shader_resource,
    BufferUsage.unordered_access,
    BufferUsage.shader_resource | BufferUsage.unordered_access,
    BufferUsage.shader_resource | BufferUsage.unordered_access | BufferUsage.shader_resource,
]


# Sets up a single buffer with 1 entry in, passes it as a pointer
# and returns the value pointed to. Tries a few different usages
# to make sure there aren't some weird memory type issues.
@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
@pytest.mark.parametrize("usage", USAGES)
def test_copy_pointer_value(device_type: DeviceType, usage: BufferUsage):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_copy_pointer_value",
        r"""
int test_copy_pointer_value(int* ptr) {
    return *ptr;
}
""",
    )

    buffer = device.create_buffer(
        size=4,  # Size of int in bytes
        usage=usage,
        data=np.array([42], dtype=np.int32),  # Initialize with a value
    )

    res = function(buffer.device_address)

    assert res == 42, f"Expected 42, got {res}"


# Same as above but uses subscript to access the value
@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_copy_pointer_subscript(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_copy_pointer_subscript",
        r"""
int test_copy_pointer_subscript(int* ptr) {
    return ptr[0];
}
""",
    )

    buffer = device.create_buffer(
        size=4,  # Size of int in bytes
        usage=BufferUsage.shader_resource,
        data=np.array([42], dtype=np.int32),  # Initialize with a value
    )

    res = function(buffer.device_address)

    assert res == 42, f"Expected 42, got {res}"


# Same as above but uses subscript to access the value
@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_copy_pointer_subscript_fullbuffer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_copy_pointer_subscript_fullbuffer",
        r"""
int test_copy_pointer_subscript_fullbuffer(int idx, int* ptr) {
    return ptr[idx];
}
""",
    )

    num_ints = 10000

    rand_ints = np.random.randint(0, 100, size=(num_ints,), dtype=np.int32)

    buffer = device.create_buffer(
        size=4 * 10000,  # Size of int in bytes
        usage=BufferUsage.shader_resource,
        data=rand_ints,
    )

    res = function(grid(shape=(num_ints,)), buffer.device_address, _result="numpy")

    assert np.array_equal(res, rand_ints), f"Expected {rand_ints}, got {res}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_add_numbers(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_add_numbers",
        r"""
int test_add_numbers(int call_id, int* a_buffer, int* b_buffer) {
    int a = a_buffer[call_id];
    int b = b_buffer[call_id];
    return a + b;
}
""",
    )

    num_ints = 10000

    a_data = np.random.randint(0, 100, size=(num_ints,), dtype=np.int32)
    b_data = np.random.randint(0, 100, size=(num_ints,), dtype=np.int32)

    a = NDBuffer.from_numpy(device, a_data)
    b = NDBuffer.from_numpy(device, b_data)

    a_address = a.storage.device_address
    b_address = b.storage.device_address

    res = function(grid(shape=(num_ints,)), a_address, b_address, _result="numpy")

    expected = a_data + b_data
    assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_pass_raw_buffers(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_pass_raw_buffers",
        r"""
int test_pass_raw_buffers(int call_id, int* a_buffer, int* b_buffer) {
    int a = a_buffer[call_id];
    int b = b_buffer[call_id];
    return a + b;
}
""",
    )

    num_ints = 10000
    a_data = np.random.randint(0, 100, size=(num_ints,), dtype=np.int32)
    b_data = np.random.randint(0, 100, size=(num_ints,), dtype=np.int32)
    a = NDBuffer.from_numpy(device, a_data)
    b = NDBuffer.from_numpy(device, b_data)
    res = function(grid(shape=(num_ints,)), a.storage, b.storage, _result="numpy")

    expected = a_data + b_data
    assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


# Same as above but uses subscript to access the value
@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_raw_buffer_of_pointers(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_raw_buffer_of_pointers",
        r"""
int test_raw_buffer_of_pointers(int idx, StructuredBuffer<int*> buffer) {
    return *buffer[idx];
}
""",
    )

    # Setup a data buffer to hold 10000 integers
    num_ints = 10000
    rand_ints = np.random.randint(0, 100, size=(num_ints,), dtype=np.int32)
    data_buffer = device.create_buffer(
        size=4 * 10000,  # Size of int in bytes
        usage=BufferUsage.shader_resource,
        data=rand_ints,
    )

    # Now create another set of poitners to random locations in the data buffer
    num_ptrs = 500
    pointers = (
        np.random.randint(0, num_ints, size=(num_ptrs,), dtype=np.uint64) * 4
        + data_buffer.device_address
    )
    pointers_buffer = device.create_buffer(
        struct_size=8,
        element_count=num_ptrs,
        usage=BufferUsage.shader_resource,
        data=pointers,
    )

    # read values cpu side to get expected data
    indices = (pointers - data_buffer.device_address) // 4
    expected_values = rand_ints[indices]

    # Call the function with the grid shape and the pointers buffer0
    res = function(grid(shape=(num_ptrs,)), pointers_buffer, _result="numpy")

    assert np.array_equal(res, expected_values), f"Expected {expected_values}, got {res}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
