# SPDX-License-Identifier: Apache-2.0
from typing import Any

from slangpy.core.native import Shape

import slangpy.bindings.typeregistry as tr
from slangpy.reflection import SlangType, ArrayType, ScalarType
from slangpy.bindings import Marshall, BindContext, CodeGenBlock, BoundVariable
from slangpy.reflection import SlangProgramLayout


class ArrayMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, element_type: Marshall, element_count: int):
        super().__init__(layout)
        self.slang_type = layout.array_type(element_type.slang_type, element_count)

    def get_shape(self, value: list[Any]) -> Shape:
        slang_shape = self.slang_type.shape
        return Shape((len(value),)+slang_shape.as_tuple()[1:])

    def resolve_type(self, context: BindContext, bound_type: SlangType):
        # If we're dealing with scalars, conform to the target type. Otherwise, passing
        # scalar types becomes quite hard - e.g. python only knows int, and trying to
        # pass to uint/int16 etc. would always throw an error
        if isinstance(bound_type, ArrayType) and isinstance(bound_type.element_type, ScalarType):
            return bound_type

        return self.slang_type

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        name = binding.variable_name
        cgb.type_alias(f"_t_{name}", f"ValueType<{binding.vector_type.full_name}>")

    # An array can be passed directly as a value for raw dispatch
    def create_dispatchdata(self, data: Any) -> Any:
        if isinstance(data, list):
            return data
        else:
            raise ValueError(
                f"Expected list for array type, got {type(data)}")


def python_lookup_array_type(layout: SlangProgramLayout, value: list[Any]) -> Marshall:
    et = tr.get_or_create_type(layout, type(value[0]), value[0])
    return ArrayMarshall(layout, et, len(value))


tr.PYTHON_TYPES[list] = python_lookup_array_type
