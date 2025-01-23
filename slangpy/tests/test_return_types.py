# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pytest

import slangpy.bindings.typeregistry as tr
import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType, TypeReflection
from slangpy.core.native import CallContext
from slangpy.bindings import ReturnContext
from slangpy.bindings.boundvariableruntime import BoundVariableRuntime
from slangpy.builtin.valueref import ValueRefMarshall
from slangpy.reflection import SlangProgramLayout


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
            raise ValueError(
                f"Expected scalar, got {value.bind_context.call_dimensionality}")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
