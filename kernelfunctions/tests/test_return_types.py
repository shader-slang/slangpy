from typing import Any
import pytest
from kernelfunctions.backend import DeviceType, TypeReflection
from kernelfunctions.backend.slangpynativeemulation import CallContext
from kernelfunctions.core.boundvariableruntime import BoundVariableRuntime
from kernelfunctions.core.reflection import SlangProgramLayout
import kernelfunctions.tests.helpers as helpers
import kernelfunctions.typeregistry as tr
from kernelfunctions.bindings.valuereftype import ValueRefType
from kernelfunctions.core import ReturnContext


class Test:
    def __init__(self, x: int):
        super().__init__()
        self.x = x


class TestType(ValueRefType):
    def __init__(self, layout: SlangProgramLayout):
        super().__init__(layout, layout.scalar_type(TypeReflection.ScalarType.int32))

    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: Any):
        return Test(super().read_output(context, binding, data))


def create_test_type(layout: SlangProgramLayout, value: Any):
    if isinstance(value, Test):
        return TestType(layout)
    elif isinstance(value, ReturnContext):
        if value.slang_type.name != "int":
            raise ValueError(f"Expected int, got {value.slang_type.name}")
        if value.bind_context.call_dimensionality != 0:
            raise ValueError(
                f"Expected scalar, got {value.bind_context.call_dimensionality}")
        return TestType(layout)
    else:
        raise ValueError(f"Unexpected value {value}")


tr.PYTHON_TYPES[Test] = create_test_type


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

    res = function.return_type(Test).call(4, 5)

    assert isinstance(res, Test)
    assert res.x == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])