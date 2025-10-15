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


tr.PYTHON_TYPES[Foo] = create_test_type


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
