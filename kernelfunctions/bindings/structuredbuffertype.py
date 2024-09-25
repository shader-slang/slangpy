

from typing import Any, Optional

from sgl import ResourceUsage

from kernelfunctions.core import BaseType, BoundVariable, CodeGenBlock, AccessType

from kernelfunctions.backend import Device, Buffer, TypeReflection
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_STRUCT_TYPES_BY_NAME, get_or_create_type

from .valuetype import ValueType


class StructuredBufferType(ValueType):

    def __init__(self, element_type: Optional[BaseType]):
        super().__init__()
        self._el_type = element_type

    @property
    def name(self) -> str:
        if self._el_type is not None:
            return f"StructuredBuffer<{self._el_type.name}>"
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
        else:
            return None

    @property
    def element_type(self):
        return self._el_type

    @property
    def differentiable(self):
        if self._el_type is not None:
            return self._el_type.differentiable
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
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BoundVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):

        # As raw structured buffers don't necessary come with a type from the python side, we have to
        # resolve to the type of the slang argument
        el_name = input_value.python.primal_element_name
        if el_name is None:
            el_name = input_value.slang.primal_type_name

        # Can now generate
        if access[0] == AccessType.read:
            cgb.type_alias(f"_{name}", f"StructuredBufferType<{el_name}>")
        elif access[0] in (AccessType.write, AccessType.readwrite):
            cgb.type_alias(f"_{name}", f"RWStructuredBufferType<{el_name}>")
        else:
            cgb.type_alias(f"_{name}", f"NoneType")

    # Call data just returns the primal
    def create_calldata(self, device: Device, input_value: 'BoundVariable', access: tuple[AccessType, AccessType], broadcast: list[bool], data: Any) -> Any:
        if access[0] != AccessType.none:
            return {
                'value': data
            }

    def update_from_bound_type(self, bound_type: 'BaseType'):
        while True:
            stshape = bound_type.shape()
            if stshape is None or None in stshape:
                next_type = bound_type.element_type
                if next_type == bound_type:
                    raise ValueError("Cannot resolve shape")
                bound_type = next_type
            else:
                break
        self._el_type = bound_type


class RWStructuredBufferType(StructuredBufferType):
    def __init__(self, element_type: Optional[BaseType]):
        super().__init__(element_type=element_type)

    @property
    def name(self) -> str:
        if self._el_type is not None:
            return f"RWStructuredBuffer<{self._el_type.name}>"
        else:
            return "RWStructuredBuffer<Unknown>"

    @property
    def is_writable(self) -> bool:
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


def _get_or_create_python_type(value: Buffer):
    assert isinstance(value, Buffer)
    # TODO: Read usage correctly when SGL supports it
    usage = ResourceUsage.unordered_access
    if (usage & ResourceUsage.unordered_access.value) != 0:
        return RWStructuredBufferType(None)
    else:
        return StructuredBufferType(None)


PYTHON_TYPES[Buffer] = _get_or_create_python_type
