from typing import Any, Sequence, Optional

from kernelfunctions.core import BaseType, BaseTypeImpl, Shape

from kernelfunctions.backend import TypeReflection
import kernelfunctions.typeregistry as tr


class InterfaceType(BaseTypeImpl):

    def __init__(self, interface_name: str):
        super().__init__()
        self.name = interface_name
        self.concrete_shape = Shape()

    @property
    def needs_specialization(self) -> bool:
        return True


def _get_or_create_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.interface
    return InterfaceType(slang_type.full_name)


tr.SLANG_INTERFACE_BASE_TYPE = _get_or_create_slang_type_reflection
