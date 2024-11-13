from typing import Any

from kernelfunctions.core import BaseType, Shape

from kernelfunctions.core.reflection import SlangProgramLayout
import kernelfunctions.typeregistry as tr

from .valuetype import ValueType


class ArrayType(ValueType):
    def __init__(self, layout: SlangProgramLayout, element_type: BaseType, element_count: int):
        super().__init__(layout)
        self.slang_type = layout.array_type(element_type.slang_type, element_count)

    def get_shape(self, value: list[Any]) -> Shape:
        slang_shape = self.slang_type.shape
        return Shape((len(value),)+slang_shape.as_tuple()[1:])


def python_lookup_array_type(layout: SlangProgramLayout, value: list[Any]) -> BaseType:
    et = tr.get_or_create_type(layout, value[0])
    return ArrayType(layout, et, len(value))


tr.PYTHON_TYPES[list] = python_lookup_array_type
