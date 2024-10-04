from typing import Any, Optional

from kernelfunctions.core import BaseType, Shape

import kernelfunctions.typeregistry as tr

from .valuetype import ValueType


class ArrayType(ValueType):
    def __init__(self, element_type: BaseType, element_count: Optional[int]):
        super().__init__()
        self.element_type = element_type
        self.ec = element_count
        self.name = f"{self.element_type.name}[{self.ec}]"

    def get_byte_size(self, value: Optional[list[Any]] = None) -> int:
        assert self.element_type is not None
        if self.ec is not None:
            return self.ec * self.element_type.get_byte_size()
        elif value is not None:
            return len(value) * self.element_type.get_byte_size()
        else:
            raise ValueError("Array size must be known to compute byte size")

    def get_container_shape(self, value: Optional[list[Any]] = None) -> Shape:
        return Shape(self.ec if self.ec else -1)

    @property
    def differentiable(self):
        return self.element_type.differentiable

    @property
    def derivative(self):
        et = self.element_type.derivative
        if et is not None:
            return ArrayType(et, self.ec)
        else:
            return None

    @property
    def python_return_value_type(self) -> type:
        return list


def slang_lookup_array_type(slang_type: tr.TypeReflection) -> BaseType:
    assert slang_type.kind == tr.TypeReflection.Kind.array
    et = tr.get_or_create_type(slang_type.element_type)
    return ArrayType(et, slang_type.element_count)


tr.SLANG_ARRAY_TYPE = slang_lookup_array_type


def python_lookup_array_type(value: list[Any]) -> BaseType:
    et = tr.get_or_create_type(value[0])
    return ArrayType(et, len(value))


tr.PYTHON_TYPES[list] = python_lookup_array_type
