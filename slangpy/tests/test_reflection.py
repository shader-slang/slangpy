# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable

import pytest

from slangpy.backend import TypeReflection

import slangpy.reflection as r
import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType
from slangpy.reflection.reflectiontypes import is_float

MODULE = """
import "slangpy";
float foo(float a) { return a; }
float foo2(float a, float b) { return a+b; }
float foo_v3(float3 a) { return a.x; }
float foo_ol(float a) { return a; }
float foo_ol(float a, float b) { return a+b; }
float foo_generic<T>(T a) { return 0; }

"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vector_types_correct(device_type: DeviceType):
    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo2", MODULE)
    layout = function.module.layout

    for st in TypeReflection.ScalarType:
        if st == TypeReflection.ScalarType.void or st == TypeReflection.ScalarType.none:
            continue
        for i in range(1, 4):
            assert layout.vector_type(st, i).num_elements == i


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_matrix_types_correct(device_type: DeviceType):
    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo2", MODULE)
    layout = function.module.layout

    for st in TypeReflection.ScalarType:
        if st == TypeReflection.ScalarType.void or st == TypeReflection.ScalarType.none:
            continue
        for row in range(1, 4):
            for col in range(1, 4):
                m = layout.matrix_type(st, row, col)
                assert m.rows == row
                assert m.cols == col


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_basic_function_decl(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo2", MODULE)
    layout = function.module.layout

    res = layout.find_function_by_name("foo2")
    assert res is not None
    assert res.name == "foo2"
    assert res.parameters[0].name == "a"
    assert res.parameters[0].type == layout.scalar_type(TypeReflection.ScalarType.float32)
    assert res.parameters[1].name == "b"
    assert res.parameters[1].type == layout.scalar_type(TypeReflection.ScalarType.float32)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_specialization(device_type: DeviceType):
    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo_generic<float>", MODULE)
    assert function is not None

    layout = function.module.layout

    res = layout.find_function_by_name("foo_generic<float>")
    assert res is not None
    assert res.name == "foo_generic"
    assert res.full_name == "foo_generic<float>"
    assert res.parameters[0].name == "a"
    assert res.parameters[0].type == layout.scalar_type(TypeReflection.ScalarType.float32)


def check_texture(type: r.SlangType, resource_shape: TypeReflection.ResourceShape, resource_access: TypeReflection.ResourceAccess, num_dims: int, element_type: str):
    assert isinstance(type, r.TextureType)
    assert type.resource_shape == resource_shape
    assert type.resource_access == resource_access
    assert type.texture_dims == num_dims

    et = type._program.find_type_by_name(element_type)
    assert et is not None
    assert type.num_dims == type.texture_dims + et.num_dims
    assert type.element_type == et


def check_scalar(type: r.SlangType, scalar_type: TypeReflection.ScalarType):
    assert isinstance(type, r.ScalarType)
    assert type.slang_scalar_type == scalar_type
    assert type.differentiable == is_float(scalar_type)


def check_vector(type: r.SlangType, scalar_type: TypeReflection.ScalarType, size: int):
    assert isinstance(type, r.VectorType)
    assert isinstance(type.element_type, r.ScalarType)
    assert type.element_type.slang_scalar_type == scalar_type
    assert type.num_elements == size
    assert type.differentiable == type.element_type.differentiable


def check_matrix(type: r.SlangType, scalar_type: TypeReflection.ScalarType, rows: int, cols: int):
    assert isinstance(type, r.MatrixType)
    assert isinstance(type.element_type, r.VectorType)
    assert type.rows == rows
    assert type.cols == cols
    assert type.differentiable == type.element_type.differentiable


def check_structured_buffer(type: r.SlangType, resource_access: TypeReflection.ResourceAccess, element_type: str):
    assert isinstance(type, r.StructuredBufferType)
    assert type.element_type == type._program.find_type_by_name(element_type)
    assert type.resource_access == resource_access


def check_address_buffer(type: r.SlangType, resource_access: TypeReflection.ResourceAccess):
    assert isinstance(type, r.ByteAddressBufferType)
    assert type.element_type == type._program.find_type_by_name('uint8_t')
    assert type.resource_access == resource_access


def check_array(type: r.SlangType, element_type: str, num_elements: int):
    assert isinstance(type, r.ArrayType)
    assert type.element_type is not None
    assert type.element_type == type._program.find_type_by_name(element_type)
    assert type.num_elements == num_elements
    if num_elements == 0:
        assert type.full_name == f"{type.element_type.full_name}[]"
    else:
        assert type.full_name == f"{type.element_type.full_name}[{num_elements}]"
    assert type.differentiable == type.element_type.differentiable


def check_struct(type: r.SlangType, fields: dict[str, str]):
    assert isinstance(type, r.StructType)

    input_field_types = {n: type._program.find_type_by_name(
        t) for (n, t) in fields.items()}
    struct_field_types = {f.name: f.type for f in type.fields.values()}
    assert input_field_types == struct_field_types


def check_interface(type: r.SlangType):
    assert isinstance(type, r.InterfaceType)


ARG_TYPE_CHECKS = [
    ("float16_t", lambda x: check_scalar(x, TypeReflection.ScalarType.float16)),
    ("float", lambda x: check_scalar(x, TypeReflection.ScalarType.float32)),
    ("int8_t", lambda x: check_scalar(x, TypeReflection.ScalarType.int8)),
    ("int16_t", lambda x: check_scalar(x, TypeReflection.ScalarType.int16)),
    ("int", lambda x: check_scalar(x, TypeReflection.ScalarType.int32)),
    ("int64_t", lambda x: check_scalar(x, TypeReflection.ScalarType.int64)),
    ("uint8_t", lambda x: check_scalar(x, TypeReflection.ScalarType.uint8)),
    ("uint16_t", lambda x: check_scalar(x, TypeReflection.ScalarType.uint16)),
    ("uint", lambda x: check_scalar(x, TypeReflection.ScalarType.uint32)),
    ("uint64_t", lambda x: check_scalar(x, TypeReflection.ScalarType.uint64)),
    ("float3", lambda x: check_vector(x, TypeReflection.ScalarType.float32, 3)),
    ("float4", lambda x: check_vector(x, TypeReflection.ScalarType.float32, 4)),
    ("vector<float,4>", lambda x: check_vector(x, TypeReflection.ScalarType.float32, 4)),
    ("int3", lambda x: check_vector(x, TypeReflection.ScalarType.int32, 3)),
    ("bool2", lambda x: check_vector(x, TypeReflection.ScalarType.bool, 2)),
    ("uint1", lambda x: check_vector(x, TypeReflection.ScalarType.uint32, 1)),
    ("float3x4", lambda x: check_matrix(x, TypeReflection.ScalarType.float32, 3, 4)),
    ("matrix<float,3,4>", lambda x: check_matrix(x, TypeReflection.ScalarType.float32, 3, 4)),
    ("Texture1D<float>", lambda x: check_texture(
        x, TypeReflection.ResourceShape.texture_1d, TypeReflection.ResourceAccess.read, 1, 'float')),
    ("RWTexture1D<float>", lambda x: check_texture(
        x, TypeReflection.ResourceShape.texture_1d, TypeReflection.ResourceAccess.read_write, 1, 'float')),
    ("Texture2D<float3>", lambda x: check_texture(
        x, TypeReflection.ResourceShape.texture_2d, TypeReflection.ResourceAccess.read, 2, 'float3')),
    ("RWTexture2D<float3>", lambda x: check_texture(
        x, TypeReflection.ResourceShape.texture_2d, TypeReflection.ResourceAccess.read_write, 2, 'float3')),
    ("StructuredBuffer<float>", lambda x: check_structured_buffer(
        x, TypeReflection.ResourceAccess.read, 'float')),
    ("RWStructuredBuffer<float4>", lambda x: check_structured_buffer(
        x, TypeReflection.ResourceAccess.read_write, 'float4')),
    ("float[10]", lambda x: check_array(x, 'float', 10)),
    ("float3[]", lambda x: check_array(x, 'float3', 0)),
    ("ByteAddressBuffer", lambda x: check_address_buffer(
        x, TypeReflection.ResourceAccess.read)),
    ("RWByteAddressBuffer", lambda x: check_address_buffer(
        x, TypeReflection.ResourceAccess.read_write)),
    ("TestStruct", lambda x: check_struct(x, {"foo": 'float'})),
    ("ITestInterface", lambda x: check_interface(x)),
]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("arg_type", ARG_TYPE_CHECKS, ids=lambda x: x[0])
def test_arg_types(device_type: DeviceType, arg_type: tuple[str, Callable[[Any], bool]]):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo",
                                                   f"""
import "slangpy";
struct TestStruct {{
    float foo;
}}
interface ITestInterface {{}}

float foo({arg_type[0]} a) {{ return 0; }}
""")
    layout = function.module.layout

    res = layout.find_function_by_name('foo')
    assert res is not None
    assert res.name == "foo"
    assert res.parameters[0].name == "a"

    arg_type[1](res.parameters[0].type)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
