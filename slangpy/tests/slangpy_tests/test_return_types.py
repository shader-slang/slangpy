# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy.bindings.typeregistry as tr
from slangpy import DeviceType, TypeReflection
from slangpy.core.native import CallContext
from slangpy.bindings import ReturnContext
from slangpy.bindings.boundvariableruntime import BoundVariableRuntime
from slangpy.builtin.valueref import ValueRefMarshall
from slangpy.reflection import SlangProgramLayout
from slangpy.types.valueref import ValueRef
from slangpy.testing import helpers

from typing import Any


class Foo:
    def __init__(self, x: int):
        super().__init__()
        self.x = x


class FooType(ValueRefMarshall):
    def __init__(self, layout: SlangProgramLayout):
        super().__init__(layout, layout.scalar_type(TypeReflection.ScalarType.int32))

    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: Any):
        return Foo(super().read_output(context, binding, data))


def create_test_type(layout: SlangProgramLayout, value: Any):
    if isinstance(value, Foo):
        return FooType(layout)
    elif isinstance(value, ReturnContext):
        if value.slang_type.name != "int":
            raise ValueError(f"Expected int, got {value.slang_type.name}")
        if value.bind_context.call_dimensionality != 0:
            raise ValueError(f"Expected scalar, got {value.bind_context.call_dimensionality}")
        return FooType(layout)
    else:
        raise ValueError(f"Unexpected value {value}")


@pytest.fixture(autouse=True)
def _register_foo_type():
    tr.PYTHON_TYPES[Foo] = create_test_type
    yield
    tr.PYTHON_TYPES.pop(Foo, None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_returnvalue(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
int add_numbers(int a, int b) {
    return a+b;
}
""",
    )

    res = function.return_type(Foo).call(4, 5)

    assert isinstance(res, Foo)
    assert res.x == 9


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_struct_as_dict(device_type: DeviceType):

    device = helpers.get_device(device_type)
    make_struct = helpers.create_function_from_module(
        device,
        "make_struct",
        r"""
struct MyStruct {
    int x;
    int y;
};
MyStruct make_struct(int a, int b) {
    return { a,b};
}
""",
    )

    res = make_struct(4, 5)

    assert isinstance(res, dict)
    assert res["x"] == 4
    assert res["y"] == 5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_nested_struct_as_dict(device_type: DeviceType):

    device = helpers.get_device(device_type)
    make_struct = helpers.create_function_from_module(
        device,
        "make_struct",
        r"""
struct MyStruct {
    int x;
    int y;
};
struct MyStruct2 {
    MyStruct a;
    MyStruct b;
}
MyStruct2 make_struct(int a, int b) {
    return { {a,b}, {b,a} };
}
""",
    )

    res = make_struct(4, 5)

    assert isinstance(res, dict)
    assert res["a"]["x"] == 4
    assert res["a"]["y"] == 5
    assert res["b"]["x"] == 5
    assert res["b"]["y"] == 4


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_inout_struct_as_dict(device_type: DeviceType):

    device = helpers.get_device(device_type)
    make_struct = helpers.create_function_from_module(
        device,
        "make_struct",
        r"""
struct MyStruct {
    int x;
    int y;
};
void make_struct(inout MyStruct v) {
    v.x += 1;
    v.y -= 1;
}
""",
    )

    v = ValueRef({"x": 5, "y": 10})

    make_struct(v)

    assert isinstance(v.value, dict)
    assert v.value["x"] == 6
    assert v.value["y"] == 9


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_inout_scalar_valueref(device_type: DeviceType):
    """inout scalar ValueRef exercises the non-struct buffer path in ValueRefMarshall."""
    device = helpers.get_device(device_type)
    increment = helpers.create_function_from_module(
        device,
        "increment",
        r"""
void increment(inout int x) {
    x += 1;
}
""",
    )
    v = ValueRef(5)
    increment(v)
    assert v.value == 6


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_inout_vector_valueref(device_type: DeviceType):
    """inout vector ValueRef exercises slang_value_to_numpy / numpy_to_slang_value for vectors."""
    import slangpy

    device = helpers.get_device(device_type)
    double_vec = helpers.create_function_from_module(
        device,
        "double_vec",
        r"""
void double_vec(inout float3 v) {
    v *= 2.0;
}
""",
    )
    v = ValueRef(slangpy.float3(1.0, 2.0, 3.0))
    double_vec(v)
    assert abs(v.value.x - 2.0) < 1e-5
    assert abs(v.value.y - 4.0) < 1e-5
    assert abs(v.value.z - 6.0) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_inout_matrix_valueref(device_type: DeviceType):
    """inout float3x3 ValueRef exercises the matrix padding/unpadding paths."""
    import numpy as np
    import slangpy

    device = helpers.get_device(device_type)
    modify_mat = helpers.create_function_from_module(
        device,
        "modify_mat",
        r"""
void modify_mat(inout float3x3 m) {
    m[0][0] += 10.0;
}
""",
    )
    orig = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    v = ValueRef(slangpy.float3x3(orig))
    modify_mat(v)
    result = v.value.to_numpy()
    assert abs(result[0][0] - 11.0) < 1e-5
    assert abs(result[1][1] - 5.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
