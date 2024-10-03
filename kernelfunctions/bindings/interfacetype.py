from typing import Any, Sequence, Optional

from kernelfunctions.core import BaseType, BaseTypeImpl

from kernelfunctions.backend import TypeReflection
import kernelfunctions.typeregistry as tr


class InterfaceType(BaseTypeImpl):

    def __init__(self, interface_name: str):
        super().__init__()
        self.interface_name = interface_name

    def name(self, value: Any = None) -> str:
        return self.interface_name

    def shape(self, value: Any = None):
        return ()

    def container_shape(self, value: Any = None) -> Sequence[Optional[int]]:
        return ()

    def element_type(self, value: Any = None):
        return self


def _get_or_create_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.interface
    return InterfaceType(slang_type.full_name)


tr.SLANG_INTERFACE_BASE_TYPE = _get_or_create_slang_type_reflection
