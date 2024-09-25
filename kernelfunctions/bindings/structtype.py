from typing import Any

from kernelfunctions.core import BaseType

from kernelfunctions.backend import TypeReflection
from kernelfunctions.typeregistry import PYTHON_TYPES
import kernelfunctions.typeregistry as tr

from .valuetype import ValueType


class StructType(ValueType):

    def __init__(self, struct_name: str):
        super().__init__()
        self.struct_name = struct_name

    @property
    def name(self) -> str:
        return self.struct_name

    def get_shape(self, value: Any = None):
        return (1,)

    @property
    def differentiable(self):
        return True

    @property
    def derivative(self):
        return self

    @property
    def has_derivative(self) -> bool:
        return True

    @property
    def is_writable(self) -> bool:
        return True


PYTHON_TYPES[dict] = StructType(struct_name='dict')


def _get_or_create_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.struct
    return StructType(struct_name=slang_type.full_name)


tr.SLANG_STRUCT_BASE_TYPE = _get_or_create_slang_type_reflection
