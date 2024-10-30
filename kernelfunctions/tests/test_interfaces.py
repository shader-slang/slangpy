from typing import Any
import pytest
from kernelfunctions.backend import DeviceType, TypeReflection
from kernelfunctions.core.basetype import BindContext
from kernelfunctions.core.boundvariable import BoundVariable
from kernelfunctions.core.codegen import CodeGenBlock
from kernelfunctions.core.slangvariable import SlangFunction
import kernelfunctions.tests.helpers as helpers
import kernelfunctions.typeregistry as tr
from kernelfunctions.bindings.valuetype import ValueType
from kernelfunctions.core import BaseType, BaseTypeImpl, Shape
from kernelfunctions import Module
from kernelfunctions.utils import get_resolved_generic_args


TEST_MODULE = """
import slangpy;

interface ITest<T, let N : int> {
    float sentinel();
}
interface IFoo {}

struct Test2f : ITest<float, 2> {
    float sentinel() { return 42.0f; }
    void load_primal(IContext ctx, out Test2f x) { x = this; }
}
struct Test3i : ITest<int, 3> {
    float sentinel() { return 0.0f; }
}

float bar(IFoo x) {
    return 0.0f;
}
float foo(ITest<float, 2> x) {
    return x.sentinel();
}
"""


class ITest(BaseTypeImpl):
    def __init__(self, reflection: TypeReflection):
        super().__init__()
        self.args = get_resolved_generic_args(reflection)
        self.name = "ITest<float, 2>"

    @property
    def needs_specialization(self):
        return True


tr.SLANG_INTERFACE_TYPES_BY_NAME["ITest"] = ITest


class Test:
    def __init__(self, T: BaseType, N: int):
        super().__init__()
        self.T = T
        self.N = N
        self.slangpy_signature = f"{T.name}{N}"


class TestImpl(ValueType):
    def __init__(self, T: BaseType, N: int):
        super().__init__()
        self.T = T
        self.N = N
        self.name = f"Test{self.N}{self.T.name[0]}"
        self.concrete_shape = Shape()

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        cgb.type_alias(f"_{binding.variable_name}", self.name)

    def specialize_type(self, type: BaseType):
        if not isinstance(type, ITest):
            return None
        assert type.args is not None
        assert isinstance(type.args[0], BaseType)
        if type.args[0].name != self.T.name or type.args[1] != self.N:
            return None
        return self


def create_test_impl(value: Any):
    assert isinstance(value, Test)
    return TestImpl(value.T, value.N)


tr.SLANG_INTERFACE_TYPES_BY_NAME["ITest"] = ITest
tr.PYTHON_TYPES[Test] = create_test_impl


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_interface_resolution(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = Module(device.load_module_from_source(
        "test_interface_resolution", TEST_MODULE))

    with tr.scope(module.device_module):
        function = SlangFunction(module.foo.as_func().reflections[0])

    param = function.parameters[0].primal

    assert isinstance(param, ITest)
    assert param.args is not None
    assert isinstance(param.args[0], BaseType) and param.args[0].name == "float"
    assert isinstance(param.args[1], int) and param.args[1] == 2


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialization(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = Module(device.load_module_from_source("test_specialization", TEST_MODULE))

    float32 = tr.SLANG_SCALAR_TYPES[TypeReflection.ScalarType.float32]
    int32 = tr.SLANG_SCALAR_TYPES[TypeReflection.ScalarType.int32]

    test2f = Test(float32, 2)
    test3i = Test(int32, 3)

    with pytest.raises(ValueError):
        module.bar(test3i)

    with pytest.raises(ValueError):
        module.foo(test3i)

    result = module.foo(test2f)

    assert result == 42.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
