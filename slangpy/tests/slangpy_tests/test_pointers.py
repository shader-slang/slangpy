# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from time import time
import numpy as np

from slangpy import DeviceType, BufferUsage, QueryType, ResourceState, grid, float3
from slangpy.types import NDBuffer, Tensor
from slangpy.testing import helpers

from typing import Any, cast

# Filter default device types to only include those that support pointers
# TODO: Metal does support pointers but the is a slang bug leading to incorrect Metal shader code
# https://github.com/shader-slang/slang/issues/7605
POINTER_DEVICE_TYPES = [
    x for x in helpers.DEFAULT_DEVICE_TYPES if x in [DeviceType.vulkan, DeviceType.cuda]
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


# Creates a buffer of values, and a buffer of pointers into those values,
# and returns the value pointed to by the pointer at the given index.
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


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_vectorize_buffer_of_pointers(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_vectorize_buffer_of_pointers",
        r"""
int test_vectorize_buffer_of_pointers(int* buffer) {
    return *buffer;
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
    pointers_buffer = NDBuffer.from_numpy(device, pointers)

    # read values cpu side to get expected data
    indices = (pointers - data_buffer.device_address) // 4
    expected_values = rand_ints[indices]

    # Call the function with so it vectorizes across the pointers buffer
    res = function(pointers_buffer, _result="numpy")

    assert np.array_equal(res, expected_values), f"Expected {expected_values}, got {res}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_write_raw_buffers(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_write_raw_buffers",
        r"""
void test_write_raw_buffers(int call_id, int* in_buffer, int* out_buffer) {
    out_buffer[call_id] = in_buffer[call_id];
}
""",
    )

    num_ints = 10000
    in_data = np.random.randint(0, 100, size=(num_ints,), dtype=np.int32)
    in_buffer = NDBuffer.from_numpy(device, in_data)
    out_buffer = NDBuffer.zeros_like(in_buffer)
    function(grid(shape=(num_ints,)), in_buffer.storage, out_buffer.storage)
    out_data = out_buffer.to_numpy()
    assert np.array_equal(in_data, out_data), f"Expected {in_data}, got {out_data}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_float_tensor_storage(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_float_tensor_storage",
        r"""
void test_float_tensor_storage(int call_id, float* in_buffer, float* out_buffer) {
    out_buffer[call_id] = in_buffer[call_id];
}
""",
    )

    num_vals = 10000
    in_data = np.random.random(num_vals).astype(np.float32)
    in_buffer = Tensor.from_numpy(device, in_data)
    assert np.array_equal(in_data, in_buffer.to_numpy())

    out_buffer = Tensor.zeros_like(in_buffer)
    function(grid(shape=(num_vals,)), in_buffer.storage, out_buffer.storage)
    out_data = out_buffer.to_numpy()
    assert np.array_equal(in_data, out_data), f"Expected {in_data}, got {out_data}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_copy_whole_structs(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_copy_whole_structs",
        r"""
struct Test {
    int value;
    float value2;
    int8_t value3;
    float3 value4;
}
void test_copy_whole_structs(int call_id, Test* in_buffer, Test* out_buffer) {
    out_buffer[call_id] = in_buffer[call_id];
}
""",
    )

    # Setup a load of data
    count = 100
    data = []
    for i in range(count):
        data.append(
            {
                "value": i,
                "value2": float(i) / 100.0,
                "value3": i % 2,
                "value4": float3(i, i + 1, i + 2),
            }
        )

    # Fill input buffer with it
    in_buffer = NDBuffer.empty(device, (count,), dtype=function.module.Test)
    cursor = in_buffer.cursor()
    for i, item in enumerate(data):
        cursor[i].write(item)
    cursor.apply()

    # Create and copy to output buffer
    out_buffer = NDBuffer.empty_like(in_buffer)
    function(grid(shape=(count,)), in_buffer.storage, out_buffer.storage)

    # Sanity check: clear the input buffer to ensure there isn't some happy
    # bug that means we're just checking the input against itself!
    in_buffer.clear()

    # Read back the output buffer and check values
    out_cursor = out_buffer.cursor()
    for i in range(count):
        in_val = data[i]
        out_val = cast(Any, out_cursor[i].read())
        assert in_val["value"] == out_val["value"], f"Value mismatch at index {i}"
        assert np.isclose(in_val["value2"], out_val["value2"]), f"Value2 mismatch at index {i}"
        assert in_val["value3"] == out_val["value3"], f"Value3 mismatch at index {i}"
        assert np.allclose(in_val["value4"], out_val["value4"]), f"Value4 mismatch at index {i}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_copy_fields_from_structs(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_copy_fields_from_structs",
        r"""
struct Test {
    int value;
    float value2;
    int8_t value3;
    float3 value4;
}
void test_copy_fields_from_structs(int call_id, Test* in_buffer, int* out_value1, float* out_value2, int8_t* out_value3, float3* out_value4) {
    out_value1[call_id] = in_buffer[call_id].value;
    out_value2[call_id] = in_buffer[call_id].value2;
    out_value3[call_id] = in_buffer[call_id].value3;
    out_value4[call_id] = in_buffer[call_id].value4;
}
""",
    )

    # Setup a load of data
    count = 100
    data = []
    for i in range(count):
        data.append(
            {
                "value": i,
                "value2": float(i) / 100.0,
                "value3": i % 2,
                "value4": float3(i, i + 1, i + 2),
            }
        )

    # Fill input buffer with it
    in_buffer = NDBuffer.empty(device, (count,), dtype=function.module.Test)
    cursor = in_buffer.cursor()
    for i, item in enumerate(data):
        cursor[i].write(item)
    cursor.apply()

    # Create and copy to output buffer
    out_buffer_value1 = NDBuffer.empty(device, (count,), dtype="int")
    out_buffer_value2 = NDBuffer.empty(device, (count,), dtype="float")
    out_buffer_value3 = NDBuffer.empty(device, (count,), dtype="int8_t")
    out_buffer_value4 = NDBuffer.empty(device, (count,), dtype="float3")

    function(
        grid(shape=(count,)),
        in_buffer.storage,
        out_buffer_value1.storage,
        out_buffer_value2.storage,
        out_buffer_value3.storage,
        out_buffer_value4.storage,
    )

    # Sanity check: clear the input buffer to ensure there isn't some happy
    # bug that means we're just checking the input against itself!
    in_buffer.clear()

    # Read back the output buffers and check values
    out_value1 = out_buffer_value1.to_numpy()
    out_value2 = out_buffer_value2.to_numpy()
    out_value3 = out_buffer_value3.to_numpy()
    out_value4 = out_buffer_value4.to_numpy()

    for i in range(count):
        in_val = data[i]
        assert in_val["value"] == out_value1[i], f"Value mismatch at index {i}"
        assert np.isclose(in_val["value2"], out_value2[i]), f"Value2 mismatch at index {i}"
        assert in_val["value3"] == out_value3[i], f"Value3 mismatch at index {i}"
        assert np.allclose(in_val["value4"], out_value4[i]), f"Value4 mismatch at index {i}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_copy_fields_by_pointer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_copy_fields_by_pointer",
        r"""
struct Test {
    int value;
    float value2;
    int8_t value3;
    float3 value4;
}
void test_copy_fields_by_pointer(int call_id, Test* in_buffer, int* out_value1, float* out_value2, int8_t* out_value3, float3* out_value4) {
    Test* ptr = in_buffer + call_id;
    out_value1[call_id] = ptr.value;
    out_value2[call_id] = ptr.value2;
    out_value3[call_id] = ptr.value3;
    out_value4[call_id] = ptr.value4;
}
""",
    )

    # Setup a load of data
    count = 100
    data = []
    for i in range(count):
        data.append(
            {
                "value": i,
                "value2": float(i) / 100.0,
                "value3": i % 2,
                "value4": float3(i, i + 1, i + 2),
            }
        )

    # Fill input buffer with it
    in_buffer = NDBuffer.empty(device, (count,), dtype=function.module.Test)
    cursor = in_buffer.cursor()
    for i, item in enumerate(data):
        cursor[i].write(item)
    cursor.apply()

    # Create and copy to output buffer
    out_buffer_value1 = NDBuffer.empty(device, (count,), dtype="int")
    out_buffer_value2 = NDBuffer.empty(device, (count,), dtype="float")
    out_buffer_value3 = NDBuffer.empty(device, (count,), dtype="int8_t")
    out_buffer_value4 = NDBuffer.empty(device, (count,), dtype="float3")

    function(
        grid(shape=(count,)),
        in_buffer.storage,
        out_buffer_value1.storage,
        out_buffer_value2.storage,
        out_buffer_value3.storage,
        out_buffer_value4.storage,
    )

    # Sanity check: clear the input buffer to ensure there isn't some happy
    # bug that means we're just checking the input against itself!
    in_buffer.clear()

    # Read back the output buffers and check values
    out_value1 = out_buffer_value1.to_numpy()
    out_value2 = out_buffer_value2.to_numpy()
    out_value3 = out_buffer_value3.to_numpy()
    out_value4 = out_buffer_value4.to_numpy()

    for i in range(count):
        in_val = data[i]
        assert in_val["value"] == out_value1[i], f"Value mismatch at index {i}"
        assert np.isclose(in_val["value2"], out_value2[i]), f"Value2 mismatch at index {i}"
        assert in_val["value3"] == out_value3[i], f"Value3 mismatch at index {i}"
        assert np.allclose(in_val["value4"], out_value4[i]), f"Value4 mismatch at index {i}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_atomic_float_access(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_atomic_float_access",
        r"""
void test_atomic_float_access(int call_id, Atomic<float>* af) {
    af[call_id % 10].add(1.0f);
}
""",
    )

    num_vals = 10
    in_buffer = Tensor.zeros(device, shape=(num_vals,), dtype="float")

    function(grid(shape=(num_vals * 100,)), in_buffer.storage)

    out_data = in_buffer.to_numpy()
    expected_data = np.ones(num_vals, dtype=np.float32) * 100
    assert np.array_equal(out_data, expected_data), f"Expected {expected_data}, got {out_data}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_atomic_float_ptr_access(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_atomic_float_ptr_access",
        r"""
void test_atomic_float_ptr_access(int call_id, Atomic<float>* af) {
    Atomic<float>* cur = af + (call_id % 10);
    cur.add(1.0f);
}
""",
    )

    num_vals = 10
    in_buffer = Tensor.zeros(device, shape=(num_vals,), dtype="float")

    function(grid(shape=(num_vals * 100,)), in_buffer.storage)

    out_data = in_buffer.to_numpy()
    expected_data = np.ones(num_vals, dtype=np.float32) * 100
    assert np.array_equal(out_data, expected_data), f"Expected {expected_data}, got {out_data}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_atomic_float_in_struct_access(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "test_atomic_float_in_struct_access",
        r"""
struct Test {
    int intvalue;
    float value;
    int intvalue2;
}
void test_atomic_float_in_struct_access(int call_id, Test* buffer) {
    Atomic<float>* cur = (Atomic<float>*) &buffer[call_id%10].value;
    cur.add(1.0f);
}
""",
    )

    # Setup a load of data
    count = 10
    data = []
    for i in range(count):
        data.append(
            {
                "intvalue": i,
                "value": 0.0,
                "intvalue2": i + 1,
            }
        )

    # Fill input buffer with it
    in_buffer = NDBuffer.empty(device, (count,), dtype=function.module.Test)
    cursor = in_buffer.cursor()
    for i, item in enumerate(data):
        cursor[i].write(item)
    cursor.apply()

    function(grid(shape=(count * 100,)), in_buffer.storage)

    # Read back the output buffer and check values
    out_cursor = in_buffer.cursor()
    for i in range(count):
        in_val = data[i]
        out_val = cast(Any, out_cursor[i].read())
        assert in_val["intvalue"] == out_val["intvalue"], f"IntValue mismatch at index {i}"
        assert np.isclose(in_val["value"] + 100, out_val["value"]), f"Value mismatch at index {i}"
        assert in_val["intvalue2"] == out_val["intvalue2"], f"IntValue2 mismatch at index {i}"


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
def test_perf(device_type: DeviceType):
    pytest.skip()

    device = helpers.get_device(device_type)

    # Add 0s to these to see the weird perf behaviours
    DISPATCHES_PER_LOOP = 30
    THREADS_PER_DISPATCH = 10000000

    pointers_function = helpers.create_function_from_module(
        device,
        "pointers_function",
        r"""
void pointers_function(int call_id, int* in_buffer, int* out_buffer) {
    out_buffer[call_id] = in_buffer[call_id];
}
""",
    )

    bindings_function = helpers.create_function_from_module(
        device,
        "bindings_function",
        r"""
void bindings_function(int call_id, StructuredBuffer<int> in_buffer, RWStructuredBuffer<int> out_buffer) {
    out_buffer[call_id] = in_buffer[call_id];
}
""",
    )

    pointer_to_bindings_function = helpers.create_function_from_module(
        device,
        "pointer_to_bindings_function",
        r"""
void pointer_to_bindings_function(int call_id, int* in_buffer, RWStructuredBuffer<int> out_buffer) {
    out_buffer[call_id] = in_buffer[call_id];
}
""",
    )

    bindings_to_pointer_function = helpers.create_function_from_module(
        device,
        "bindings_to_pointer_function",
        r"""
void bindings_to_pointer_function(int call_id, StructuredBuffer<int> in_buffer, int* out_buffer) {
    out_buffer[call_id] = in_buffer[call_id];
}
""",
    )

    # List of tests to run
    tests = [
        ("Pointers->Pointers", pointers_function),
        ("Bindings->Bindings", bindings_function),
        ("Pointers->Bindings", pointer_to_bindings_function),
        ("Bindings->Pointers", bindings_to_pointer_function),
    ]

    qp = device.create_query_pool(QueryType.timestamp, DISPATCHES_PER_LOOP + 1)

    for i in range(0, 1):
        for test_name, func in tests:

            # Setup inputs etc
            in_data = np.random.randint(0, 100, size=(THREADS_PER_DISPATCH,), dtype=np.int32)
            in_buffer = NDBuffer.from_numpy(device, in_data)
            out_buffer = NDBuffer.zeros_like(in_buffer)

            g = grid(shape=(THREADS_PER_DISPATCH,))

            # Do 1 call of pointers function to get it compiled and validate it
            func(g, in_buffer.storage, out_buffer.storage)
            out_data = out_buffer.to_numpy()
            assert np.array_equal(in_data, out_data), f"Expected {in_data}, got {out_data}"
            out_buffer.clear()

            # Do 1 call to ensure warmed up
            command_encoder = device.create_command_encoder()
            func(g, in_buffer.storage, out_buffer.storage, _append_to=command_encoder)
            command_buffer = command_encoder.finish()
            device.submit_command_buffer(command_buffer)
            device.wait_for_idle()

            # Now time 100 calls
            qp.reset()
            command_encoder = device.create_command_encoder()
            command_encoder.write_timestamp(qp, 0)
            for i in range(DISPATCHES_PER_LOOP):
                func(g, in_buffer.storage, out_buffer.storage, _append_to=command_encoder)
                command_encoder.write_timestamp(qp, i + 1)
            command_buffer = command_encoder.finish()

            pointers_start = time()
            device.submit_command_buffer(command_buffer)
            device.wait_for_idle()
            pointers_end = time()

            timers = np.array(qp.get_timestamp_results(0, DISPATCHES_PER_LOOP + 1))
            dispatch_times = timers - timers[0]
            dispatch_times = timers[1:] - timers[0:-1]  # Get the time between each dispatch

            min_time = np.min(dispatch_times)
            max_time = np.max(dispatch_times)
            avg_time = (
                np.sum(dispatch_times) / DISPATCHES_PER_LOOP
            )  # Do 1 big sum/divide to reduce floating point errors

            pointers_duration = pointers_end - pointers_start
            print(
                f"\n{test_name} function took {pointers_duration*1000:.2f}ms for {DISPATCHES_PER_LOOP} calls, "
            )
            print(
                f"Min dispatch time: {min_time*1000:.2f}ms, "
                f"Max dispatch time: {max_time*1000:.2f}ms, "
                f"Avg dispatch time: {avg_time*1000:.2f}ms"
            )
            print(",".join([f"{x*1000:.2f}" for x in dispatch_times]))


@pytest.mark.parametrize("device_type", POINTER_DEVICE_TYPES)
@pytest.mark.parametrize("sync_type", ["none", "global", "resource"])
def test_pointer_barriers(device_type: DeviceType, sync_type: str):
    if sync_type == "none":
        pytest.skip("Skipping non-deterministic race-condition test")

    device = helpers.get_device(device_type)

    ptr_to_ptr = helpers.create_function_from_module(
        device,
        "ptr_to_ptr",
        r"""
void ptr_to_ptr(int idx, int* src, int* dst) {
    dst[idx] = src[idx];
}
""",
    )
    buffer_to_buffer = helpers.create_function_from_module(
        device,
        "buffer_to_buffer",
        r"""
void buffer_to_buffer(int idx, StructuredBuffer<int> src, RWStructuredBuffer<int> dst) {
    dst[idx] = src[idx];
}
""",
    )
    ptr_to_buffer = helpers.create_function_from_module(
        device,
        "ptr_to_buffer",
        r"""
void ptr_to_buffer(int idx, int* src, RWStructuredBuffer<int> dst) {
    dst[idx] = src[idx];
}
""",
    )
    buffer_to_ptr = helpers.create_function_from_module(
        device,
        "buffer_to_ptr",
        r"""
void buffer_to_ptr(int idx, StructuredBuffer<int> src, int* dst) {
    dst[idx] = src[idx];
}
""",
    )

    # Init data + grid args
    num_ints = 1000
    rand_ints = np.random.randint(0, 100, size=(num_ints,), dtype=np.int32)
    gridshape = grid(shape=(num_ints,))

    # Create buffers to transition data through
    # Default state should be unnecesary, but as we're messing with state tracking it's
    # helpful to be explicit
    src = device.create_buffer(
        element_count=num_ints,
        struct_size=4,
        usage=BufferUsage.shader_resource,
        data=rand_ints,
    )
    tmp = device.create_buffer(
        element_count=num_ints,
        struct_size=4,
        usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
    )
    dst = device.create_buffer(
        element_count=num_ints,
        struct_size=4,
        usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
    )

    # Create encoder so we can submit all commands as quick as possible
    encoder = device.create_command_encoder()

    # If using global sync, just add a global barrier to make sure
    # upload is complete.
    # If using resource sync, switch tmp to SR, and dst to UAV
    if sync_type == "global":
        encoder.global_barrier()
    else:
        encoder.set_buffer_state(tmp, ResourceState.unordered_access)

    # Call the function once to transition data from src to tmp
    ptr_to_ptr(gridshape, src, tmp, _append_to=encoder)

    # If using global sync, just add a global barrier
    # If using resource sync, switch tmp to SR, and dst to UAV
    if sync_type == "global":
        encoder.global_barrier()
    elif sync_type == "resource":
        encoder.set_buffer_state(tmp, ResourceState.shader_resource)
        encoder.set_buffer_state(dst, ResourceState.unordered_access)

    # Call the function again to transition data from tmp to dst
    ptr_to_ptr(gridshape, tmp, dst, _append_to=encoder)

    # Final barriers (potentially not needed) for after final copy
    if sync_type == "global":
        encoder.global_barrier()
    else:
        encoder.set_buffer_state(dst, ResourceState.shader_resource)

    # Submit all work at once
    device.submit_command_buffer(encoder.finish())

    res = dst.to_numpy().view(np.int32)

    if sync_type == "none" and device_type in [DeviceType.vulkan, DeviceType.d3d12]:
        assert not np.array_equal(res, rand_ints), f"Expected {rand_ints}, got {res}"
    else:
        assert np.array_equal(res, rand_ints), f"Expected {rand_ints}, got {res}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
