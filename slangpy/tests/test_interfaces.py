# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pytest

from slangpy.core.native import Shape

import slangpy.bindings.typeregistry as tr
import slangpy.tests.helpers as helpers
from slangpy import Module
from slangpy.backend import DeviceType, TypeReflection
from slangpy.bindings.marshall import BindContext, Marshall
from slangpy.bindings.boundvariable import BoundVariable
from slangpy.bindings.codegen import CodeGenBlock
from slangpy.builtin.value import ValueMarshall
from slangpy.reflection import SlangProgramLayout, SlangType

TEST_MODULE = """
import slangpy;

interface ITest<T, let N : int> {
    float sentinel();
}
interface IFoo {}

struct Test2f : ITest<float, 2> {
    float sentinel() { return 42.0f; }
    void load(Context0D ctx, out Test2f x) { x = this; }
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


class Foo:
    def __init__(self, T: SlangType, N: int):
        super().__init__()
        self.T = T
        self.N = N
        self.slangpy_signature = f"{T.full_name}{N}"


class FooImpl(Marshall):
    def __init__(self, layout: SlangProgramLayout, T: SlangType, N: int):
        super().__init__(layout)
        self.T = T
        self.N = N
        st = layout.find_type_by_name(f"Test{N}{T.full_name[0]}")
        assert isinstance(st, SlangType)
        self.slang_type = st
        self.concrete_shape = Shape()

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        cgb.type_alias(f"_t_{binding.variable_name}", self.slang_type.full_name)


def create_test_impl(layout: SlangProgramLayout, value: Any):
    assert isinstance(value, Foo)
    return FooImpl(layout, value.T, value.N)


tr.PYTHON_TYPES[Foo] = create_test_impl


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialization(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = Module(device.load_module_from_source("test_specialization", TEST_MODULE))

    float32 = module.layout.scalar_type(TypeReflection.ScalarType.float32)
    int32 = module.layout.scalar_type(TypeReflection.ScalarType.int32)

    test2f = Foo(float32, 2)
    test3i = Foo(int32, 3)

    with pytest.raises(ValueError):
        module.bar(test3i)

    with pytest.raises(ValueError):
        module.foo(test3i)

    result = module.foo(test2f)

    assert result == 42.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
