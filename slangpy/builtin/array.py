from typing import Any

from slangpy.core.native import Shape

import slangpy.bindings.typeregistry as tr
from slangpy.bindings import BaseType
from slangpy.builtin.valuetype import ValueType
from slangpy.reflection import SlangProgramLayout


class ArrayType(ValueType):
    def __init__(self, layout: SlangProgramLayout, element_type: BaseType, element_count: int):
        super().__init__(layout)
        self.slang_type = layout.array_type(element_type.slang_type, element_count)

    def get_shape(self, value: list[Any]) -> Shape:
        slang_shape = self.slang_type.shape
        return Shape((len(value),)+slang_shape.as_tuple()[1:])

    # An array can be passed directly as a value for raw dispatch
    def create_dispatchdata(self, data: Any) -> Any:
        if isinstance(data, list):
            return data
        else:
            raise ValueError(
                f"Expected list for array type, got {type(data)}")


def python_lookup_array_type(layout: SlangProgramLayout, value: list[Any]) -> BaseType:
    et = tr.get_or_create_type(layout, value[0])
    return ArrayType(layout, et, len(value))


tr.PYTHON_TYPES[list] = python_lookup_array_type
