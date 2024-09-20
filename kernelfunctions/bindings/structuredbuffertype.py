

from typing import Any, Optional

from kernelfunctions.core import BaseType, BaseVariable, CodeGenBlock, AccessType

from kernelfunctions.backend import Device, Buffer, TypeReflection
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_STRUCT_TYPES_BY_NAME, get_or_create_type

from .valuetype import ValueType


class StructuredBufferType(ValueType):

    def __init__(self, element_type: Optional[BaseType]):
        super().__init__()
        self._el_type = element_type

    def name(self, value: Optional[Buffer] = None) -> str:
        if self._el_type is not None:
            return f"StructuredBuffer<{self._el_type.name()}>"
        else:
            return "StructuredBuffer<Unknown>"

    def container_shape(self, value: Optional[Buffer] = None):
        if value is not None:
            return (int(value.desc.size/value.desc.struct_size),)
        else:
            return (None,)

    def shape(self, value: Optional[Buffer] = None):
        if self._el_type is not None:
            return super().shape(value)
        elif value is not None:
            return self.container_shape(value) + (None,)
        else:
            return (None, None)

    def element_type(self, value: Any = None):
        return self._el_type

    def differentiable(self, value: Optional[Buffer] = None):
        if self._el_type is not None:
            return self._el_type.differentiable()
        else:
            return False

    def differentiate(self, value: Optional[Buffer] = None):
        if self._el_type is not None:
            el_diff = self._el_type.differentiate()
            if el_diff is not None:
                return StructuredBufferType(el_diff)
            else:
                return None
        else:
            return None

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):

        # As raw structured buffers don't necessary come with a type from the python side, we have to
        # resolve to the type of the slang argument
        el_name = input_value.primal_element_name
        if el_name is None:
            assert input_value.binding is not None
            el_name = input_value.binding.slang.primal_type_name

        # Can now generate
        if access[0] == AccessType.read:
            cgb.type_alias(f"_{name}", f"StructuredBufferType<{el_name}>")
        elif access[0] in (AccessType.write, AccessType.readwrite):
            cgb.type_alias(f"_{name}", f"RWStructuredBufferType<{el_name}>")
        else:
            cgb.type_alias(f"_{name}", f"NoneType")

    # Call data just returns the primal
    def create_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: Any) -> Any:
        if access[0] != AccessType.none:
            return {
                'value': data
            }


class RWStructuredBufferType(StructuredBufferType):
    def __init__(self, element_type: BaseType):
        super().__init__(element_type=element_type)

    def is_writable(self, value: Optional[Buffer] = None) -> bool:
        return True

    def differentiate(self, value: Optional[Buffer] = None):
        if self._el_type is not None:
            el_diff = self._el_type.differentiate()
            if el_diff is not None:
                return StructuredBufferType(el_diff)
            else:
                return None
        else:
            return None


def _get_or_create_ro_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.resource
    return StructuredBufferType(element_type=get_or_create_type(slang_type.resource_result_type))


def _get_or_create_rw_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.resource
    return RWStructuredBufferType(element_type=get_or_create_type(slang_type.resource_result_type))


SLANG_STRUCT_TYPES_BY_NAME['StructuredBuffer'] = _get_or_create_ro_slang_type_reflection
SLANG_STRUCT_TYPES_BY_NAME['RWStructuredBuffer'] = _get_or_create_rw_slang_type_reflection

PYTHON_TYPES[Buffer] = StructuredBufferType(None)
