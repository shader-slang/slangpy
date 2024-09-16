

from typing import Any, Optional

from sgl import TypeReflection
from kernelfunctions.typeregistry import PYTHON_TYPES
import kernelfunctions.typeregistry as tr
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.valuetype import ValueType


class StructType(ValueType):

    def __init__(self, struct_name: str):
        super().__init__()
        self.struct_name = struct_name

    def name(self, value: Any = None) -> str:
        return self.struct_name

    def shape(self, value: Any = None):
        return (1,)

    def differentiable(self, value: Any = None):
        return True

    def differentiate(self, value: Any = None):
        return self

    def has_derivative(self, value: Any = None) -> bool:
        return True

    def is_writable(self, value: Any = None) -> bool:
        return True


PYTHON_TYPES[dict] = StructType(struct_name='dict')


def _get_or_create_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.struct
    return StructType(struct_name=slang_type.name)


tr.SLANG_STRUCT_BASE_TYPE = _get_or_create_slang_type_reflection
