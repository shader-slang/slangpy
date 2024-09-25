from typing import Any, Optional

from kernelfunctions.core import BaseType

import kernelfunctions.typeregistry as tr

from .valuetype import ValueType


class ArrayType(ValueType):
    def __init__(self, element_type: BaseType, element_count: Optional[int]):
        super().__init__()
        self.et = element_type
        self.ec = element_count

    @property
    def name(self) -> str:
        return f"{self.et.name}[{self.ec}]"

    def byte_size(self, value: Optional[list[Any]] = None) -> int:
        if self.ec is not None:
            return self.ec * self.et.byte_size()
        elif value is not None:
            return len(value) * self.et.byte_size()
        else:
            raise ValueError("Array size must be known to compute byte size")

    def container_shape(self, value: Optional[list[Any]] = None):
        return (self.ec,)

    @property
    def element_type(self):
        return self.et

    @property
    def differentiable(self):
        return self.et.differentiable

    def differentiate(self, value: Optional[list[Any]] = None):
        et = self.et.differentiate(value)
        if et is not None:
            return ArrayType(et, self.ec)
        else:
            return None

    def python_return_value_type(self, value: Optional[list[Any]] = None) -> type:
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
