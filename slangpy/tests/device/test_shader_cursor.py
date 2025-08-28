# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
import struct
import numpy as np
from dataclasses import dataclass

import slangpy as spy
from slangpy.testing import helpers

from typing import Literal, Any
from numpy.typing import DTypeLike


INT_MIN = -2147483648
INT_MAX = 2147483647
UINT_MIN = 0
UINT_MAX = 4294967295
FLOAT_MIN = -3.402823466e38
FLOAT_MAX = 3.402823466e38
FLOAT16_MIN = -65504.0
FLOAT16_MAX = 65504.0


@dataclass
class TypeInfo:
    # size in bytes when reading back
    size: int
    # encoding of read back data
    struct: str
    # numpy dtype
    dtype: DTypeLike  # type: ignore


TYPE_INFOS = {
    # The size=4 for bool here is not reflecting on the actual size of bool on the device.
    # The result is a buffer of uint32_t, and each single write to it is at least 4 bytes.
    "bool": TypeInfo(size=4, struct="I", dtype=np.bool_),
    "int": TypeInfo(size=4, struct="i", dtype=np.int32),
    "uint": TypeInfo(size=4, struct="I", dtype=np.uint32),
    "float": TypeInfo(size=4, struct="f", dtype=np.float32),
    "int16_t": TypeInfo(size=4, struct="hxx", dtype=np.int16),
    "uint16_t": TypeInfo(size=4, struct="Hxx", dtype=np.uint16),
    "float16_t": TypeInfo(size=4, struct="exx", dtype=np.float16),
    ## Unused for now
    # "int64_t": TypeInfo(size=8, struct="q", dtype=np.int64),
    # "int64_t": TypeInfo(size=8, struct="Q", dtype=np.uint64),
    # "float64_t": TypeInfo(size=8, struct="d", dtype=np.float64),
}


@dataclass
class Var:
    kind: Literal["scalar", "vector", "matrix", "array"]
    type: str
    value: Any


TEST_VARS = {
    # fmt: off
    # Global uniforms
    # bool
    "u_bool_false": Var(kind="scalar", type="bool", value=False),
    "u_bool_true": Var(kind="scalar", type="bool", value=True),
    # bool2
    "u_bool2": Var(kind="vector", type="bool", value=[False, True]),
    # bool3
    "u_bool3": Var(kind="vector", type="bool", value=[False, True, False]),
    # bool4
    "u_bool4": Var(kind="vector", type="bool", value=[False, True, False, True]),
    # int
    "u_int": Var(kind="scalar", type="int", value=-12345),
    "u_int_min": Var(kind="scalar", type="int", value=INT_MIN),
    "u_int_max": Var(kind="scalar", type="int", value=INT_MAX),
    # int2
    "u_int2": Var(kind="vector", type="int", value=[-12345, 12345]),
    "u_int2_min": Var(kind="vector", type="int", value=[INT_MIN, INT_MIN]),
    "u_int2_max": Var(kind="vector", type="int", value=[INT_MAX, INT_MAX]),
    # int3
    "u_int3": Var(kind="vector", type="int", value=[-12345, 12345, -123456]),
    "u_int3_min": Var(kind="vector", type="int", value=[INT_MIN, INT_MIN, INT_MIN]),
    "u_int3_max": Var(kind="vector", type="int", value=[INT_MAX, INT_MAX, INT_MAX]),
    # int4
    "u_int4": Var(kind="vector", type="int", value=[-12345, 12345, -123456, 123456]),
    "u_int4_min": Var(kind="vector", type="int", value=[INT_MIN, INT_MIN, INT_MIN, INT_MIN]),
    "u_int4_max": Var(kind="vector", type="int", value=[INT_MAX, INT_MAX, INT_MAX, INT_MAX]),
    # uint
    "u_uint": Var(kind="scalar", type="uint", value=12345),
    "u_uint_min": Var(kind="scalar", type="uint", value=UINT_MIN),
    "u_uint_max": Var(kind="scalar", type="uint", value=UINT_MAX),
    # uint2
    "u_uint2": Var(kind="vector", type="uint", value=[12345, 123456]),
    "u_uint2_min": Var(kind="vector", type="uint", value=[UINT_MIN, UINT_MIN]),
    "u_uint2_max": Var(kind="vector", type="uint", value=[UINT_MAX, UINT_MAX]),
    # uint3
    "u_uint3": Var(kind="vector", type="uint", value=[12345, 123456, 1234567]),
    "u_uint3_min": Var(kind="vector", type="uint", value=[UINT_MIN, UINT_MIN, UINT_MIN]),
    "u_uint3_max": Var(kind="vector", type="uint", value=[UINT_MAX, UINT_MAX, UINT_MAX]),
    # uint4
    "u_uint4": Var(kind="vector", type="uint", value=[12345, 123456, 1234567, 12345678]),
    "u_uint4_min": Var(kind="vector", type="uint", value=[UINT_MIN, UINT_MIN, UINT_MIN, UINT_MIN]),
    "u_uint4_max": Var(kind="vector", type="uint", value=[UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX]),
    # float
    "u_float": Var(kind="scalar", type="float", value=1.2345),
    "u_float_min": Var(kind="scalar", type="float", value=FLOAT_MIN),
    "u_float_max": Var(kind="scalar", type="float", value=FLOAT_MAX),
    # float2
    "u_float2": Var(kind="vector", type="float", value=[1.2345, -1.2345]),
    "u_float2_min": Var(kind="vector", type="float", value=[FLOAT_MIN, FLOAT_MIN]),
    "u_float2_max": Var(kind="vector", type="float", value=[FLOAT_MAX, FLOAT_MAX]),
    # float3
    "u_float3": Var(kind="vector", type="float", value=[1.2345, -1.2345, 12.345]),
    "u_float3_min": Var(kind="vector", type="float", value=[FLOAT_MIN, FLOAT_MIN, FLOAT_MIN]),
    "u_float3_max": Var(kind="vector", type="float", value=[FLOAT_MAX, FLOAT_MAX, FLOAT_MAX]),
    # float4
    "u_float4": Var(kind="vector", type="float", value=[1.2345, -1.2345, 12.345, -12.345]),
    "u_float4_min": Var(kind="vector", type="float", value=[FLOAT_MIN, FLOAT_MIN, FLOAT_MIN, FLOAT_MIN]),
    "u_float4_max": Var(kind="vector", type="float", value=[FLOAT_MAX, FLOAT_MAX, FLOAT_MAX, FLOAT_MAX]),
    # floatXxX
    "u_float2x2": Var(kind="matrix", type="float", value=[[0, 1], [2, 3]]),
    "u_float3x3": Var(kind="matrix", type="float", value=[[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
    "u_float2x4": Var(kind="matrix", type="float", value=[[0, 1, 2, 3], [4, 5, 6, 7]]),
    "u_float3x4": Var(kind="matrix", type="float", value=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
    "u_float4x4": Var(kind="matrix", type="float", value=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
    # float16_t
    "u_float16_t": Var(kind="scalar", type="float16_t", value=1.2345),
    "u_float16_t_min": Var(kind="scalar", type="float16_t", value=FLOAT16_MIN),
    "u_float16_t_max": Var(kind="scalar", type="float16_t", value=FLOAT16_MAX),
    # float16_t2
    "u_float16_t2": Var(kind="vector", type="float16_t", value=[1.2345, -1.2345]),
    "u_float16_t2_min": Var(kind="vector", type="float16_t", value=[FLOAT16_MIN, FLOAT16_MIN]),
    "u_float16_t2_max": Var(kind="vector", type="float16_t", value=[FLOAT16_MAX, FLOAT16_MAX]),
    # float16_t3
    "u_float16_t3": Var(kind="vector", type="float16_t", value=[1.2345, -1.2345, 12.345]),
    "u_float16_t3_min": Var(kind="vector", type="float16_t", value=[FLOAT16_MIN, FLOAT16_MIN, FLOAT16_MIN]),
    "u_float16_t3_max": Var(kind="vector", type="float16_t", value=[FLOAT16_MAX, FLOAT16_MAX, FLOAT16_MAX]),
    # float16_t4
    "u_float16_t4": Var(kind="vector", type="float16_t", value=[1.2345, -1.2345, 12.345, -12.345]),
    "u_float16_t4_min": Var(kind="vector", type="float16_t", value=[FLOAT16_MIN, FLOAT16_MIN, FLOAT16_MIN, FLOAT16_MIN]),
    "u_float16_t4_max": Var(kind="vector", type="float16_t", value=[FLOAT16_MAX, FLOAT16_MAX, FLOAT16_MAX, FLOAT16_MAX]),
    # uniform arrays
    "u_bool_array": Var(kind="array", type="bool", value=[False, True, False, True]),
    "u_int_array": Var(kind="array", type="int", value=[-2, -1, 0, 1]),
    "u_uint_array": Var(kind="array", type="uint", value=[10, 20, 30, 40]),
    "u_float_array": Var(kind="array", type="float", value=[1.1, 2.2, 3.3, 4.4]),
    # u_struct
    "u_struct": {
        "f_bool": Var(kind="scalar", type="bool", value=True),
        "f_bool2": Var(kind="vector", type="bool", value=[False, True]),
        "f_bool3": Var(kind="vector", type="bool", value=[False, True, False]),
        "f_bool4": Var(kind="vector", type="bool", value=[False, True, False, True]),
        "f_int": Var(kind="scalar", type="int", value=-123),
        "f_int2": Var(kind="vector", type="int", value=[-123, 123]),
        "f_int3": Var(kind="vector", type="int", value=[-123, 123, -1234]),
        "f_int4": Var(kind="vector", type="int", value=[-123, 123, -1234, 1234]),
        "f_uint": Var(kind="scalar", type="uint", value=12),
        "f_uint2": Var(kind="vector", type="uint", value=[123, 1234]),
        "f_uint3": Var(kind="vector", type="uint", value=[123, 1234, 12345]),
        "f_uint4": Var(kind="vector", type="uint", value=[123, 1235, 123456, 1234567]),
        "f_float": Var(kind="scalar", type="float", value=1.2),
        "f_float2": Var(kind="vector", type="float", value=[1.23, 1.234]),
        "f_float3": Var(kind="vector", type="float", value=[1.23, 1.234, 1.2345]),
        "f_float4": Var(kind="vector", type="float", value=[1.23, 1.235, 1.23456, 1.234567]),
        "f_bool_array": Var(kind="array", type="bool", value=[False]),
        "f_int_array": Var(kind="array", type="int", value=[-10, 10]),
        "f_uint_array": Var(kind="array", type="uint", value=[0, 10, 20]),
        "f_float_array": Var(kind="array", type="float", value=[0.1, 0.2, 0.3, 0.4]),
    },
    # u_int_array_2
    "u_int_array_2": [
        Var(kind="scalar", type="int", value=1),
        Var(kind="scalar", type="int", value=2),
        Var(kind="scalar", type="int", value=3),
        Var(kind="scalar", type="int", value=4),
    ],
    # u_struct_array
    "u_struct_array": [
        {
            "f_int": Var(kind="scalar", type="int", value=-1),
            "f_uint": Var(kind="scalar", type="uint", value=1),
            "f_float": Var(kind="scalar", type="float", value=1.0),
        },
        {
            "f_int": Var(kind="scalar", type="int", value=-2),
            "f_uint": Var(kind="scalar", type="uint", value=2),
            "f_float": Var(kind="scalar", type="float", value=2.0),
        },
        {
            "f_int": Var(kind="scalar", type="int", value=-3),
            "f_uint": Var(kind="scalar", type="uint", value=3),
            "f_float": Var(kind="scalar", type="float", value=3.0),
        },
        {
            "f_int": Var(kind="scalar", type="int", value=-4),
            "f_uint": Var(kind="scalar", type="uint", value=4),
            "f_float": Var(kind="scalar", type="float", value=4.0),
        },
    ]
    # u_test
    # fmt: on
}


def flatten(values: Any):
    if isinstance(values, list):
        if isinstance(values[0], list):
            return [item for sublist in values for item in sublist]
        return values
    else:
        return [values]


def convert_vector(type: str, dim: int, values: Any):
    TABLE = {
        ("float16_t", 2): spy.float16_t2,
        ("float16_t", 3): spy.float16_t3,
        ("float16_t", 4): spy.float16_t4,
    }
    key = (type, dim)
    return TABLE[key](values) if key in TABLE else values


def convert_matrix(type: str, rows: int, cols: int, values: Any):
    TABLE = {
        ("float", 2, 2): spy.float2x2,
        ("float", 3, 3): spy.float3x3,
        ("float", 2, 4): spy.float2x4,
        ("float", 3, 4): spy.float3x4,
        ("float", 4, 4): spy.float4x4,
    }
    key = (type, rows, cols)
    return TABLE[key](flatten(values))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("use_numpy", [False, True])
def test_shader_cursor(device_type: spy.DeviceType, use_numpy: bool):
    if device_type == spy.DeviceType.vulkan and sys.platform == "darwin":
        pytest.skip("Test shader doesn't currently compile on MoltenVK")

    device = helpers.get_device(type=device_type)

    program = device.load_program("test_shader_cursor.slang", ["compute_main"])
    kernel = device.create_compute_kernel(program)

    result_buffer = device.create_buffer(
        size=4096,
        struct_size=4,
        usage=spy.BufferUsage.unordered_access,
    )

    names = []
    sizes = []
    references = []
    types = []

    def write_var(
        device_type: spy.DeviceType,
        cursor: spy.ShaderCursor,
        name_or_index: str | int,
        var: Var,
        name_prefix: str,
    ):
        nonlocal names
        nonlocal sizes
        nonlocal references
        nonlocal types

        type_info = TYPE_INFOS[var.type]

        # Determine descriptive name
        if isinstance(name_or_index, str):
            name = name_prefix + name_or_index
        elif isinstance(name_or_index, int):
            name = name_prefix + f"[{name_or_index}]"
        names.append(name)
        types.append(var.type)

        # Value to write
        value = var.value
        flat_value = flatten(value)

        # Read back size
        size = type_info.size
        struct_pattern = type_info.struct
        element_count = 1

        if var.kind == "vector":
            element_count = len(var.value)
            if use_numpy:
                value = np.array(var.value, dtype=type_info.dtype)
            else:
                value = convert_vector(var.type, element_count, var.value)
        elif var.kind == "matrix":
            rows = len(var.value)
            cols = len(var.value[0])
            element_count = rows * cols
            if use_numpy:
                value = np.array(var.value, dtype=type_info.dtype)
            else:
                value = convert_matrix(var.type, rows, cols, var.value)
        elif var.kind == "array":
            element_count = len(var.value)
            value = np.array(var.value, dtype=type_info.dtype)

        size *= element_count
        struct_pattern *= element_count

        sizes.append(size)
        references.append(struct.pack(struct_pattern, *flat_value).hex())

        cursor[name_or_index] = value

    def write_vars(
        device_type: spy.DeviceType,
        cursor: spy.ShaderCursor,
        vars: dict[str, Any] | list[Any],
        name_prefix: str = "",
    ):
        if isinstance(vars, dict):
            for key, var in vars.items():
                if isinstance(var, dict):
                    write_vars(device_type, cursor[key], var, key + ".")
                elif isinstance(var, list):
                    write_vars(device_type, cursor[key], var, key)
                else:
                    write_var(device_type, cursor, key, var, name_prefix)
        elif isinstance(vars, list):
            for i, var in enumerate(vars):
                if isinstance(var, dict):
                    write_vars(device_type, cursor[i], var, name_prefix + f"[{i}].")
                elif isinstance(var, list):
                    write_vars(device_type, cursor[i], var, name_prefix)
                else:
                    write_var(device_type, cursor, i, var, name_prefix)

    command_encoder = device.create_command_encoder()
    with command_encoder.begin_compute_pass() as pass_encoder:
        shader_object = pass_encoder.bind_pipeline(kernel.pipeline)
        cursor = spy.ShaderCursor(shader_object)
        cursor["results"] = result_buffer
        write_vars(device_type, cursor, TEST_VARS)
        pass_encoder.dispatch(thread_count=[1, 1, 1])
    device.submit_command_buffer(command_encoder.finish())

    data = result_buffer.to_numpy().tobytes()
    results = []
    for size in sizes:
        results.append(data[0:size].hex())
        data = data[size:]

    named_typed_references = list(zip(names, types, references))
    named_typed_results = list(zip(names, types, results))

    for named_typed_result, named_typed_reference in zip(
        named_typed_results, named_typed_references
    ):
        # Vulkan/Metal/CUDA packing rule for certain matrix types are not the same as D3D12's
        if (device_type in [spy.DeviceType.vulkan, spy.DeviceType.metal, spy.DeviceType.cuda]) and (
            named_typed_result[0] == "u_float2x2" or named_typed_result[0] == "u_float3x3"
        ):
            continue

        assert named_typed_result == named_typed_reference


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
