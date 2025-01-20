# SPDX-License-Identifier: Apache-2.0


from typing import Any

from slangpy.core.native import AccessType, CallContext, Shape
from slangpy.backend import TypeReflection

from slangpy.bindings import (PYTHON_TYPES, Marshall, BindContext,
                              BoundVariable, BoundVariableRuntime,
                              CodeGenBlock)
from slangpy.reflection import SlangProgramLayout, SlangType


class RangeMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout):
        super().__init__(layout)
        st = layout.find_type_by_name(f"RangeType")
        if st is None:
            raise ValueError(
                f"Could not find RangeType slang type. This usually indicates the slangpy module has not been imported.")
        self.slang_type = st

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: range) -> Any:
        access = binding.access
        if access[0] == AccessType.read:
            return {
                'start': data.start,
                'stop': data.stop,
                'step': data.step
            }

    def get_shape(self, data: range):
        s = ((data.stop-data.start)//data.step,)
        return Shape(s)

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        return context.layout.scalar_type(TypeReflection.ScalarType.int32)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return 1 - len(vector_target_type.shape)


PYTHON_TYPES[range] = lambda l, x: RangeMarshall(l)
