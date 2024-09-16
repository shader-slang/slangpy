

from typing import Any, Optional

from sgl import Buffer, TypeReflection
from kernelfunctions.typeregistry import PYTHON_TYPES
import kernelfunctions.typeregistry as tr
from kernelfunctions.core.basetype import BaseType
from kernelfunctions.types.valuetype import ValueType


class StructuredBufferType(ValueType):

    def __init__(self, element_type: BaseType):
        super().__init__()
        self._el_type = element_type

    def name(self, value: Optional[Buffer] = None) -> str:
        return f"StructuredBuffer<{self._el_type.name()}>"

    def container_shape(self, value: Optional[Buffer] = None):
        if value is not None:
            return (int(value.desc.size/value.desc.struct_size),)
        else:
            return (None,)

    def element_type(self, value: Any = None):
        return self._el_type

    def differentiable(self, value: Optional[Buffer] = None):
        return self.element_type(value).differentiable()

    def differentiate(self, value: Optional[Buffer] = None):
        el_diff = self.element_type(value).differentiate()
        if el_diff is not None:
            return StructuredBufferType(el_diff)
        else:
            return None


class RWStructuredBufferType(StructuredBufferType):
    def __init__(self, element_type: BaseType):
        super().__init__(element_type=element_type)

    def is_writable(self, value: Optional[Buffer] = None) -> bool:
        return True

    def differentiate(self, value: Optional[Buffer] = None):
        el_diff = self.element_type(value).differentiate()
        if el_diff is not None:
            return RWStructuredBufferType(el_diff)
        else:
            return None


def _get_or_create_ro_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.resource
    return StructuredBufferType(element_type=tr.get_or_create_type(slang_type.resource_result_type))


def _get_or_create_rw_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.resource
    return RWStructuredBufferType(element_type=tr.get_or_create_type(slang_type.resource_result_type))


tr.SLANG_STRUCT_TYPES_BY_NAME['StructuredBuffer'] = _get_or_create_ro_slang_type_reflection
tr.SLANG_STRUCT_TYPES_BY_NAME['RWStructuredBuffer'] = _get_or_create_rw_slang_type_reflection
