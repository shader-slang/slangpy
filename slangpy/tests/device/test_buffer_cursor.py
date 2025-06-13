# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import random
from typing import Any
import pytest
import slangpy as spy
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import sglhelpers as helpers
from sglhelpers import test_id  # type: ignore (pytest fixture)

TESTS = [
    ("f_bool", "bool", "true", True),
    ("f_bool1", "bool1", "false", spy.bool1(False)),
    ("f_bool2", "bool2", "bool2(true, false)", spy.bool2(True, False)),
    ("f_bool3", "bool3", "bool3(true, false, true)", spy.bool3(True, False, True)),
    (
        "f_bool4",
        "bool4",
        "bool4(true, false, true, false)",
        spy.bool4(True, False, True, False),
    ),
    ("f_int16", "int16_t", "42", 42),
    ("f_uint16", "uint16_t", "312", 312),
    ("f_int32", "int", "-53134", -53134),
    ("f_uint32", "uint", "2123", 2123),
    ("f_int64", "int64_t", "-412", -412),
    ("f_uint64", "uint64_t", "7567", 7567),
    ("f_float", "float", "3.25", 3.25),
    ("f_float1", "float1", "123", spy.float1(123.0)),
    ("f_float2", "float2", "float2(1.0, 2.0)", spy.float2(1.0, 2.0)),
    ("f_float3", "float3", "float3(1.0, 2.0, 3.0)", spy.float3(1.0, 2.0, 3.0)),
    (
        "f_float4",
        "float4",
        "float4(1.0, 2.0, 3.0, 4.0)",
        spy.float4(1.0, 2.0, 3.0, 4.0),
    ),
    ("f_int1", "int1", "123", spy.int1(123)),
    ("f_int2", "int2", "int2(1, 2)", spy.int2(1, 2)),
    ("f_int3", "int3", "int3(1, 2, 3)", spy.int3(1, 2, 3)),
    ("f_int4", "int4", "int4(1, 2, 3, 4)", spy.int4(1, 2, 3, 4)),
    ("f_uint1", "uint1", "123", spy.uint1(123)),
    ("f_uint2", "uint2", "uint2(1, 2)", spy.uint2(1, 2)),
    ("f_uint3", "uint3", "uint3(1, 2, 3)", spy.uint3(1, 2, 3)),
    ("f_uint4", "uint4", "uint4(1, 2, 3, 4)", spy.uint4(1, 2, 3, 4)),
    (
        "f_float2x2",
        "float2x2",
        "float2x2(1.0, 2.0, 3.0, 4.0)",
        spy.float2x2([1.0, 2.0, 3.0, 4.0]),
    ),
    (
        "f_float2x4",
        "float2x4",
        "float2x4(1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0)",
        spy.float2x4([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]),
    ),
    (
        "f_float3x3",
        "float3x3",
        "float3x3(1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 9.0)",
        spy.float3x3([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 9.0]),
    ),
    (
        "f_float3x4",
        "float3x4",
        "float3x4(1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0)",
        spy.float3x4([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0]),
    ),
    (
        "f_float4x3",
        "float4x3",
        "float4x3(1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0)",
        spy.float4x3([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0]),
    ),
    (
        "f_float4x4",
        "float4x4",
        "float4x4(1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0, -5.0, -6.0, -7.0, -8.0)",
        spy.float4x4(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                -5.0,
                -6.0,
                -7.0,
                -8.0,
            ]
        ),
    ),
    (
        "f_int32array",
        "int32_t[5]",
        "{1, 2, 3, 4, 5}",
        [1, 2, 3, 4, 5],
    ),
    (
        "f_int32array_numpy",
        "int32_t[5]",
        "{1, 2, 3, 4, 5}",
        [1, 2, 3, 4, 5],
    ),
    (
        "f_float3_list",
        "float3",
        "float3(1.0, 2.0, 3.0)",
        [1.0, 2.0, 3.0],
        spy.float3(1.0, 2.0, 3.0),
    ),
    (
        "f_float3_tuple",
        "float3",
        "float3(1.0, 2.0, 3.0)",
        (1.0, 2.0, 3.0),
        spy.float3(1.0, 2.0, 3.0),
    ),
    (
        "f_float3_numpy",
        "float3",
        "float3(1.0, 2.0, 3.0)",
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        spy.float3(1.0, 2.0, 3.0),
    ),
    ("f_child", "TestChild", "TestChild(1, 2.0)", {"uintval": 1, "floatval": 2.0}),
    ## Test arrays of bools and their vectors
    ("f_bool_array", "bool[2]", "{true, false}", [True, False]),
    (
        "f_bool2_array",
        "bool2[2]",
        "{ {true, false}, {true, false} }",
        [spy.bool2(True, False), spy.bool2(True, False)],
    ),
    # Fails on: SGL_CHECK(nbarray.ndim() == 1, "numpy array must have 1 dimension.");
    # (
    #     "f_bool2_array_numpy",
    #     "bool2[2]",
    #     "{ {true, false}, {true, false} }",
    #     np.array([[True, False], [True, False]], dtype=bool),
    #     [spy.bool2(True, False), spy.bool2(True, False)],
    # ),
    ## Test nested arrays of scalar and vector (and matrix) types
    ("f_float32_array", "float[2]", "{ 1.0, 2.0 }", [1.0, 2.0]),
    ("f_float32_array2", "float[2][2]", "{ { 1.0, 2.0 }, { 3.0, 4.0 } }", [[1.0, 2.0], [3.0, 4.0]]),
    # Fails on: SGL_CHECK(nbarray.ndim() == 1, "numpy array must have 1 dimension.");
    # (
    #     "f_float_array2_numpy",
    #     "float[2][2]",
    #     "{ { 1.0, 2.0 }, { 3.0, 4.0 } }",
    #     np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    #     [[1.0, 2.0], [3.0, 4.0]],
    # ),
    (
        "f_float2_array2",
        "float2[2][2]",
        "{ { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } }",
        [
            [spy.float2(1.0, 2.0), spy.float2(3.0, 4.0)],
            [spy.float2(5.0, 6.0), spy.float2(7.0, 8.0)],
        ],
    ),
    # Fails on: SGL_CHECK(nbarray.ndim() == 1, "numpy array must have 1 dimension.");
    # (
    #     "f_float2_array2_numpy",
    #     "float2[2][2]",
    #     "{ { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } }",
    #     np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
    #     [
    #         [spy.float2(1.0, 2.0), spy.float2(3.0, 4.0)],
    #         [spy.float2(5.0, 6.0), spy.float2(7.0, 8.0)],
    #     ],
    # ),
    ("f_int32_array", "int32_t[2]", "{ 1, 2 }", [1, 2]),
    ("f_int32_array2", "int32_t[2][2]", "{ { 1, 2 }, { 3, 4 } }", [[1, 2], [3, 4]]),
    # Fails on: SGL_CHECK(nbarray.ndim() == 1, "numpy array must have 1 dimension.");
    # (
    #     "f_int_array2_numpy",
    #     "int32_t[2][2]",
    #     "{ { 1, 2 }, { 3, 4 } }",
    #     np.array([[[1, 2], [3, 4]]], dtype=np.int32),
    #     [[1, 2], [3, 4]],
    # ),
    (
        "f_int2_array2",
        "int2[2][2]",
        "{ { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }",
        [[spy.int2(1, 2), spy.int2(3, 4)], [spy.int2(5, 6), spy.int2(7, 8)]],
    ),
    # Fails on: SGL_CHECK(nbarray.ndim() == 1, "numpy array must have 1 dimension.");
    # (
    #     "f_int2_array2_numpy",
    #     "int2[2][2]",
    #     "{ { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }",
    #     np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
    #     [[spy.int2(1, 2), spy.int2(3, 4)], [spy.int2(5, 6), spy.int2(7, 8)]],
    # ),
    # Fails on: SGL_CHECK(nbarray.ndim() == 1, "numpy array must have 1 dimension.");
    # (
    #     "f_float2x2_array_numpy",
    #     "float2x2[2]",
    #     "{ { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 } }",
    #     np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
    #     [
    #         spy.float2x2(np.array([[1, 2], [3, 4]], dtype=np.float32)),
    #         spy.float2x2(np.array([[5, 6], [7, 8]], dtype=np.float32)),
    #     ],
    # ),
    (
        "f_float2x2_array2",
        "float2x2[2][2]",
        "{ { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 } }, { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 } } }",
        [
            [
                spy.float2x2(np.array([[1, 2], [3, 4]], dtype=np.float32)),
                spy.float2x2(np.array([[5, 6], [7, 8]], dtype=np.float32)),
            ],
            [
                spy.float2x2(np.array([[1, 2], [3, 4]], dtype=np.float32)),
                spy.float2x2(np.array([[5, 6], [7, 8]], dtype=np.float32)),
            ],
        ],
    ),
]


def get_tests(device_type: spy.DeviceType):
    return TESTS


def variable_decls(tests: list[Any]):
    return "".join([f"    {t[1]} {t[0]};\n" for t in tests])


def variable_sets(tests: list[Any]):
    return "".join([f"    buffer[i].{t[0]} = {t[2]};\n" for t in tests])


def gen_fill_in_module(tests: list[Any]):
    return f"""
    struct TestChild {{
        uint uintval;
        float floatval;
    }}
    struct TestType {{
        uint value;
        TestChild child;
    {variable_decls(tests)}
    }};

    [shader("compute")]
    [numthreads(1, 1, 1)]
    void compute_main(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<TestType> buffer) {{
        uint i = tid.x;
        buffer[i].value = i+1;
        buffer[i].child.floatval = i+2.0;
        buffer[i].child.uintval = i+3;
    {variable_sets(tests)}
    }}
    """


def gen_copy_module(tests: list[Any]):
    return f"""
    struct TestChild {{
        uint uintval;
        float floatval;
    }}
    struct TestType {{
        uint value;
        TestChild child;
    {variable_decls(tests)}
    }};

    [shader("compute")]
    [numthreads(1, 1, 1)]
    void compute_main(uint3 tid: SV_DispatchThreadID, StructuredBuffer<TestType> src, RWStructuredBuffer<TestType> dest) {{
        uint i = tid.x;
        dest[i] = src[i];
    }}
    """


# RAND_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
RAND_SEEDS = [1, 2, 3]


def check_match(test: tuple[Any, ...], result: Any):
    if len(test) == 4:
        (name, gpu_type, gpu_val, value) = test
        outvalue = value
    elif len(test) == 5:
        (name, gpu_type, gpu_val, value, outvalue) = test
    assert result == outvalue


def make_fill_in_module(device_type: spy.DeviceType, tests: list[Any]):
    code = gen_fill_in_module(tests)
    # print(code)
    mod_name = "test_buffer_cursor_TestType_" + hashlib.sha256(code.encode()).hexdigest()[0:8]
    device = helpers.get_device(type=device_type)
    module = device.load_module_from_source(mod_name, code)
    prog = device.link_program([module], [module.entry_point("compute_main")])
    buffer_layout = module.layout.get_type_layout(
        module.layout.find_type_by_name("StructuredBuffer<TestType>")
    )
    return (device.create_compute_kernel(prog), buffer_layout)


def make_copy_module(device_type: spy.DeviceType, tests: list[Any]):
    code = gen_copy_module(tests)
    mod_name = "test_buffer_cursor_FillIn_" + hashlib.sha256(code.encode()).hexdigest()[0:8]
    device = helpers.get_device(type=device_type)
    module = device.load_module_from_source(mod_name, code)
    prog = device.link_program([module], [module.entry_point("compute_main")])
    buffer_layout = module.layout.get_type_layout(
        module.layout.find_type_by_name("StructuredBuffer<TestType>")
    )
    return (device.create_compute_kernel(prog), buffer_layout)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("seed", RAND_SEEDS)
def test_cursor_read_write(device_type: spy.DeviceType, seed: int):

    # Randomize the order of the tests
    tests = get_tests(device_type).copy()
    if device_type == spy.DeviceType.cuda:
        tests = [x for x in tests if "bool" not in x[0]]
    random.seed(seed)
    random.shuffle(tests)

    # Create the module and buffer layout
    (kernel, buffer_layout) = make_fill_in_module(device_type, tests)

    # Create a buffer cursor with its own data
    cursor = spy.BufferCursor(buffer_layout.element_type_layout, 1)

    # Populate the first element
    element = cursor[0]
    for test in tests:
        (name, gpu_type, gpu_val, value) = test[0:4]
        element[name] = value

    # Create new cursor by copying the first, and read element
    cursor2 = spy.BufferCursor(buffer_layout.element_type_layout, 1)
    cursor2.copy_from_numpy(cursor.to_numpy())
    element2 = cursor2[0]

    # Verify data matches
    for test in tests:
        name = test[0]
        check_match(test, element2[name].read())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("seed", RAND_SEEDS)
def test_fill_from_kernel(device_type: spy.DeviceType, seed: int):

    # Randomize the order of the tests
    tests = get_tests(device_type).copy()
    random.seed(seed)
    random.shuffle(tests)

    # Create the module and buffer layout
    (kernel, buffer_layout) = make_fill_in_module(device_type, tests)

    # Make a buffer with 128 elements
    count = 128
    buffer = kernel.device.create_buffer(
        element_count=count,
        struct_type=buffer_layout,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    # Dispatch the kernel
    kernel.dispatch([count, 1, 1], buffer=buffer)

    # Create a cursor and read the buffer by copying its data
    cursor = spy.BufferCursor(buffer_layout.element_type_layout, count)
    cursor.copy_from_numpy(buffer.to_numpy())

    # Verify data matches
    for i in range(count):
        element = cursor[i]
        for test in tests:
            name = test[0]
            check_match(test, element[name].read())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("seed", RAND_SEEDS)
def test_wrap_buffer(device_type: spy.DeviceType, seed: int):

    # Randomize the order of the tests
    tests = get_tests(device_type).copy()
    random.seed(seed)
    random.shuffle(tests)

    # Create the module and buffer layout
    (kernel, buffer_layout) = make_fill_in_module(device_type, tests)

    # Make a buffer with 128 elements and a cursor to wrap it
    count = 128
    buffer = kernel.device.create_buffer(
        element_count=count,
        struct_type=buffer_layout,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    cursor = spy.BufferCursor(buffer_layout.element_type_layout, buffer)

    # Cursor shouldn't have read data from buffer yet
    assert not cursor.is_loaded

    # Dispatch the kernel
    kernel.dispatch([count, 1, 1], buffer=buffer)

    # Clear buffer reference to verify lifetime is maintained due to cursor
    buffer = None

    # Load 1 element and verify we still haven't loaded anything
    element = cursor[0]
    assert not cursor.is_loaded

    # Read a value from the element and verify it is now loaded
    element["f_int32"].read()
    assert cursor.is_loaded

    # Verify data matches
    for i in range(count):
        element = cursor[i]
        for test in tests:
            name = test[0]
            check_match(test, element[name].read())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_cursor_lifetime(device_type: spy.DeviceType):

    # Create the module and buffer layout
    (kernel, buffer_layout) = make_fill_in_module(device_type, get_tests(device_type))

    # Create a buffer cursor with its own data
    cursor = spy.BufferCursor(buffer_layout.element_type_layout, 1)

    # Get element
    element = cursor[0]

    # Null the cursor
    cursor = None

    # Ensure we can still write to the element (as it should be holding a reference to the cursor)
    element["f_int32"] = 123


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("seed", RAND_SEEDS)
def test_apply_changes(device_type: spy.DeviceType, seed: int):

    # Randomize the order of the tests
    tests = get_tests(device_type).copy()
    random.seed(seed)
    random.shuffle(tests)

    # Create the module and buffer layout
    (kernel, buffer_layout) = make_copy_module(device_type, tests)

    # Make a buffer with 128 elements and a cursor to wrap it
    count = 128
    src = kernel.device.create_buffer(
        element_count=count,
        struct_type=buffer_layout,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(buffer_layout.element_type_layout.stride * count, dtype=np.uint8),
    )
    dest = kernel.device.create_buffer(
        element_count=count,
        struct_type=buffer_layout,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(buffer_layout.element_type_layout.stride * count, dtype=np.uint8),
    )
    src_cursor = spy.BufferCursor(buffer_layout.element_type_layout, src)
    dest_cursor = spy.BufferCursor(buffer_layout.element_type_layout, dest)

    # Populate source cursor
    for i in range(count):
        element = src_cursor[i]
        for test in tests:
            (name, gpu_type, gpu_val, value) = test[0:4]
            element[name] = value

    # Apply changes to source
    src_cursor.apply()

    # Load the dest cursor - this should end up with it containing 0s as its not been written
    dest_cursor.load()

    # Verify 0s
    for i in range(count):
        element = dest_cursor[i]
        assert element["f_int32"].read() == 0

    # Dispatch the kernel
    kernel.dispatch([count, 1, 1], src=src, dest=dest)

    # Verify still 0s as we've not refreshed the cursor yet!
    for i in range(count):
        element = dest_cursor[i]
        assert element["f_int32"].read() == 0

    # Refresh the buffer
    dest_cursor.load()

    # Verify data in dest buffer matches
    for i in range(count):
        element = dest_cursor[i]
        for test in tests:
            name = test[0]
            check_match(test, element[name].read())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("seed", RAND_SEEDS)
@pytest.mark.parametrize("element_class", [np.array, spy.bool2, tuple, list])
def test_bool_buffers(device_type: spy.DeviceType, seed: int, element_class: Any):
    code = f"""
    [shader("compute")]
    [numthreads(1, 1, 1)]
    void compute_main(uint3 tid: SV_DispatchThreadID, StructuredBuffer<bool2> src, RWStructuredBuffer<bool2> dest) {{
        uint i = tid.x;
        dest[i] = src[i];
    }}
    """
    mod_name = (
        "test_buffer_cursor_TestBoolBuffers_" + hashlib.sha256(code.encode()).hexdigest()[0:8]
    )
    device = helpers.get_device(device_type)
    module = device.load_module_from_source(mod_name, code)
    prog = device.link_program([module], [module.entry_point("compute_main")])
    buffer_layout = module.layout.get_type_layout(
        module.layout.find_type_by_name("StructuredBuffer<bool2>")
    )
    (kernel, buffer_layout) = (device.create_compute_kernel(prog), buffer_layout)

    # Make a buffer with 128 elements and a cursor to wrap it
    count = 128
    src = kernel.device.create_buffer(
        element_count=count,
        struct_type=buffer_layout,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(buffer_layout.element_type_layout.stride * count, dtype=np.uint8),
    )
    dest = kernel.device.create_buffer(
        element_count=count,
        struct_type=buffer_layout,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(buffer_layout.element_type_layout.stride * count, dtype=np.uint8),
    )
    src_cursor = spy.BufferCursor(buffer_layout.element_type_layout, src)
    dest_cursor = spy.BufferCursor(buffer_layout.element_type_layout, dest)

    random.seed(seed)
    list_data = [[random.randint(0, 1) == 1, random.randint(0, 1) == 1] for i in range(count)]
    data = []
    if element_class == np.array:
        data = [element_class(x, dtype=np.bool_) for x in list_data]
    elif element_class == spy.bool2:
        data = [element_class(x) for x in list_data]
    elif element_class == tuple:
        data = [(x[0], x[1]) for x in list_data]
    elif element_class == list:
        data = list_data

    for i in range(count):
        src_cursor[i].write(data[i])

    # Apply changes to source
    src_cursor.apply()

    # Dispatch the kernel
    kernel.dispatch([count, 1, 1], src=src, dest=dest)

    dest_cursor.load()
    for i in range(count):
        result = dest_cursor[i].read()
        data_ref = spy.bool2(list_data[i])
        src_ref = src_cursor[i].read()
        assert result == data_ref
        assert result == src_ref


# test introduced to warn us when issue https://github.com/shader-slang/slang/issues/7441
# has been resolved and the type information or the underlying types have changed.
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_boolX_reflection(device_type: spy.DeviceType):
    code = f"""
    [shader("compute")]
    [numthreads(1, 1, 1)]
    void compute_main(uint3 tid: SV_DispatchThreadID, StructuredBuffer<bool2> src, RWStructuredBuffer<bool2> dest) {{
        uint i = tid.x;
        dest[i] = src[i];
    }}
    """
    mod_name = (
        "test_buffer_cursor_test_boolX_reflection_" + hashlib.sha256(code.encode()).hexdigest()[0:8]
    )
    device = helpers.get_device(device_type)
    module = device.load_module_from_source(mod_name, code)
    prog = device.link_program([module], [module.entry_point("compute_main")])
    sb_bool2_layout = module.layout.get_type_layout(
        module.layout.find_type_by_name("StructuredBuffer<bool2>")
    )
    pb_bool2_layout = module.layout.get_type_layout(
        module.layout.find_type_by_name("ParameterBlock<bool2>")
    )
    u_bool2_layout = module.layout.get_type_layout(module.layout.find_type_by_name("bool2"))

    sb_bool2_element_layout = sb_bool2_layout.element_type_layout
    pb_bool2_element_layout = pb_bool2_layout.element_type_layout

    def make_layout(type_layout: spy.TypeLayoutReflection):
        return {
            "size": type_layout.size,
            "stride": type_layout.size,
            "element_stride": type_layout.element_stride(),
            "element_type_layout.size": type_layout.element_type_layout.size,
            "element_type_layout.stride": type_layout.element_type_layout.stride,
        }

    def make_layout_ref():
        if device_type == spy.DeviceType.d3d12:
            return {
                "size": 8,
                "stride": 8,
                "element_stride": 4,
                "element_type_layout.size": 4,
                "element_type_layout.stride": 4,
            }
        if device_type == spy.DeviceType.vulkan:
            return {
                "size": 8,
                "stride": 8,
                "element_stride": 4,
                "element_type_layout.size": 4,
                "element_type_layout.stride": 4,
            }
        if device_type == spy.DeviceType.metal:
            return {
                "size": 2,
                "stride": 2,
                "element_stride": 1,
                "element_type_layout.size": 1,
                "element_type_layout.stride": 1,
            }
        if device_type == spy.DeviceType.wgpu:
            return {
                "size": 8,
                "stride": 8,
                "element_stride": 4,
                "element_type_layout.size": 4,
                "element_type_layout.stride": 4,
            }
        if device_type == spy.DeviceType.cpu:
            return {
                "size": 2,
                "stride": 2,
                "element_stride": 1,
                "element_type_layout.size": 1,
                "element_type_layout.stride": 1,
            }
        # This is actually reporting wrong, see issue: https://github.com/shader-slang/slang/issues/7441
        # Once that issue has been resolved, this test should trigger and workarounds can be removed
        if device_type == spy.DeviceType.cuda:
            return {
                "size": 8,
                "stride": 8,
                "element_stride": 1,
                "element_type_layout.size": 1,
                "element_type_layout.stride": 1,
            }

    layout_descs = {
        "u_bool2": make_layout(u_bool2_layout),
        "sb_bool2_element": make_layout(sb_bool2_element_layout),
        "pb_bool2_element": make_layout(pb_bool2_element_layout),
    }

    ref_desc = make_layout_ref()

    for k, v in layout_descs.items():
        assert v == ref_desc


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("seed", RAND_SEEDS)
def test_apply_changes_ndarray(device_type: spy.DeviceType, seed: int):

    # Randomize the order of the tests
    tests = [
        ("f_float", "float", "1.0", 1.0),
        ("f_float1", "float1", "2.0", [2.0]),
        ("f_float2", "float2", "float2(2.0, 3.0)", [2.0, 3.0]),
        ("f_float3", "float3", "float3(2.0, 3.0, 4.0)", [2.0, 3.0, 4.0]),
        ("f_float4", "float4", "float4(2.0, 3.0, 4.0, 5.0)", [2.0, 3.0, 4.0, 5.0]),
    ]
    random.seed(seed)
    random.shuffle(tests)

    # Create the module and buffer layout
    (kernel, buffer_layout) = make_copy_module(device_type, tests)

    # Make a buffer with 128 elements and a cursor to wrap it
    count = 128
    src = kernel.device.create_buffer(
        element_count=count,
        struct_type=buffer_layout,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(buffer_layout.element_type_layout.stride * count, dtype=np.uint8),
    )
    dest = kernel.device.create_buffer(
        element_count=count,
        struct_type=buffer_layout,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(buffer_layout.element_type_layout.stride * count, dtype=np.uint8),
    )
    src_cursor = spy.BufferCursor(buffer_layout.element_type_layout, src)
    dest_cursor = spy.BufferCursor(buffer_layout.element_type_layout, dest)

    # Populate source cursor
    source_data = {}
    for test in tests:
        (name, gpu_type, gpu_val, value) = test[0:4]
        my_list = []
        if isinstance(value, list):
            my_list = [[x + i for x in value] for i in range(count)]
        elif isinstance(value, float):
            my_list = [value + i for i in range(count)]
        source_data[name] = np.array(my_list, dtype=np.float32)

    src_cursor.write_from_numpy(source_data)

    # Apply changes to source
    src_cursor.apply()

    # Load the dest cursor - this should end up with it containing 0s as its not been written
    dest_cursor.load()

    # Verify 0s
    for i in range(count):
        element = dest_cursor[i]
        assert element["f_float"].read() == 0

    # Dispatch the kernel
    kernel.dispatch([count, 1, 1], src=src, dest=dest)

    # Verify still 0s as we've not refreshed the cursor yet!
    for i in range(count):
        element = dest_cursor[i]
        assert element["f_float"].read() == 0

    # Refresh the buffer
    dest_cursor.load()

    # Verify data in dest buffer matches
    for i in range(count):
        element = dest_cursor[i]
        for test in tests:
            name = test[0]
            expected_value = test[3]
            if isinstance(expected_value, float):
                expected_value += i
            elif isinstance(expected_value, list):
                expected_value = [x + i for x in expected_value]
            result = element[name].read()
            assert expected_value == result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
