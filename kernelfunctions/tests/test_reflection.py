from typing import Any, Callable
import pytest
from sgl import float3, float4
from kernelfunctions.backend import DeviceType
import kernelfunctions.tests.helpers as helpers
import kernelfunctions.core.reflection as r
from kernelfunctions.types.buffer import NDBuffer

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

    for v in r.VECTOR.values():
        assert v[0] == None
        for i in range(1, 4):
            assert v[i].num_elements == i


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_matrix_types_correct(device_type: DeviceType):

    for v in r.MATRIX.values():
        assert v[0] == None
        for row in range(1, 4):
            assert v[row][0] == None
            for col in range(1, 4):
                assert v[row][col].rows == row
                assert v[row][col].cols == col


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_basic_function_decl(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo2", MODULE)

    res = r.reflect_function(function.reflections[0], None)
    assert res is not None
    assert res.name == "foo2"
    assert res.parameters[0].name == "a"
    assert res.parameters[0].type == r.float32
    assert res.parameters[1].name == "b"
    assert res.parameters[1].type == r.float32


def check_texture(type: r.ScalarType, resource_shape: r.TR.ResourceShape, resource_access: r.TR.ResourceAccess, num_dims: int, element_type: r.SlangType):
    assert isinstance(type, r.TextureType)
    assert type.resource_shape == resource_shape
    assert type.resource_access == resource_access
    assert type.num_dims == num_dims
    assert type.element_type == element_type


def check_scalar(type: r.SlangType, scalar_type: r.TR.ScalarType):
    assert isinstance(type, r.ScalarType)
    assert type.scalar_type == scalar_type
    assert type.differentiable == r.is_float(scalar_type)


def check_vector(type: r.SlangType, scalar_type: r.TR.ScalarType, size: int):
    assert isinstance(type, r.VectorType)
    assert type.element_type.scalar_type == scalar_type
    assert type.num_elements == size
    assert type.differentiable == type.element_type.differentiable


def check_matrix(type: r.SlangType, scalar_type: r.TR.ScalarType, rows: int, cols: int):
    assert isinstance(type, r.MatrixType)
    assert type.rows == rows
    assert type.cols == cols
    assert type.differentiable == type.element_type.differentiable


ARG_TYPE_CHECKS = [
    ("float16_t", lambda x: check_scalar(x, r.TR.ScalarType.float16)),
    ("float", lambda x: check_scalar(x, r.TR.ScalarType.float32)),
    ("int8_t", lambda x: check_scalar(x, r.TR.ScalarType.int8)),
    ("int16_t", lambda x: check_scalar(x, r.TR.ScalarType.int16)),
    ("int", lambda x: check_scalar(x, r.TR.ScalarType.int32)),
    ("int64_t", lambda x: check_scalar(x, r.TR.ScalarType.int64)),
    ("uint8_t", lambda x: check_scalar(x, r.TR.ScalarType.uint8)),
    ("uint16_t", lambda x: check_scalar(x, r.TR.ScalarType.uint16)),
    ("uint", lambda x: check_scalar(x, r.TR.ScalarType.uint32)),
    ("uint64_t", lambda x: check_scalar(x, r.TR.ScalarType.uint64)),
    ("float3", lambda x: check_vector(x, r.TR.ScalarType.float32, 3)),
    ("float4", lambda x: check_vector(x, r.TR.ScalarType.float32, 4)),
    ("vector<float,4>", lambda x: check_vector(x, r.TR.ScalarType.float32, 4)),
    ("int3", lambda x: check_vector(x, r.TR.ScalarType.int32, 3)),
    ("bool2", lambda x: check_vector(x, r.TR.ScalarType.bool, 2)),
    ("uint1", lambda x: check_vector(x, r.TR.ScalarType.uint32, 1)),
    ("float3x4", lambda x: check_matrix(x, r.TR.ScalarType.float32, 3, 4)),
    ("matrix<float,3,4>", lambda x: check_matrix(x, r.TR.ScalarType.float32, 3, 4)),
    ("Texture1D<float>", lambda x: check_texture(
        x, r.TR.ResourceShape.texture_1d, r.TR.ResourceAccess.read, 1, r.float32)),
    ("RWTexture1D<float>", lambda x: check_texture(
        x, r.TR.ResourceShape.texture_1d, r.TR.ResourceAccess.read_write, 1, r.float32)),
    ("Texture2D<float3>", lambda x: check_texture(
        x, r.TR.ResourceShape.texture_2d, r.TR.ResourceAccess.read, 2, r.float3)),
    ("RWTexture2D<float3>", lambda x: check_texture(
        x, r.TR.ResourceShape.texture_2d, r.TR.ResourceAccess.read_write, 2, r.float3)),
]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("arg_type", ARG_TYPE_CHECKS, ids=lambda x: x[0])
def test_arg_types(device_type: DeviceType, arg_type: tuple[str, Callable[[Any], bool]]):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo",
                                                   f"""
import "slangpy";
float foo({arg_type[0]} a) {{ return 0; }}
""")

    res = r.reflect_function(function.reflections[0], None)
    assert res is not None
    assert res.name == "foo"
    assert res.parameters[0].name == "a"

    arg_type[1](res.parameters[0].type)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
