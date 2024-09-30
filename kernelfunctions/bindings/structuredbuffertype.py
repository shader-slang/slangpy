

from typing import Any, Optional

from sgl import ResourceUsage

from kernelfunctions.core import BaseType, BoundVariable, CodeGenBlock, AccessType, BoundVariableRuntime

from kernelfunctions.backend import Device, Buffer, TypeReflection
from kernelfunctions.shapes import TLooseOrUndefinedShape, TLooseShape
from kernelfunctions.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES, SLANG_STRUCT_TYPES_BY_NAME, get_or_create_type

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

    def get_container_shape(self, value: Optional[Buffer] = None) -> TLooseShape:
        if value is not None:
            return (int(value.desc.size/value.desc.struct_size),)
        else:
            return (None,)

    def get_shape(self, value: Optional[Buffer] = None) -> TLooseOrUndefinedShape:
        if self._el_type is not None:
            return super().get_shape(value)
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

    @property
    def derivative(self):
        if self._el_type is not None:
            el_diff = self._el_type.derivative
            if el_diff is not None:
                return StructuredBufferType(el_diff)
            else:
                return None
        else:
            return None

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name

        # As raw structured buffers don't necessary come with a type from the python side, we have to
        # resolve to the type of the slang argument
        el_name = binding.python.primal_element_name
        if el_name is None:
            el_name = binding.slang.primal_type_name

        # Can now generate
        if access[0] == AccessType.read:
            cgb.type_alias(f"_{name}", f"StructuredBufferType<{el_name}>")
        elif access[0] in (AccessType.write, AccessType.readwrite):
            cgb.type_alias(f"_{name}", f"RWStructuredBufferType<{el_name}>")
        else:
            cgb.type_alias(f"_{name}", f"NoneType")

    # Call data just returns the primal
    def create_calldata(self, device: Device, binding: 'BoundVariableRuntime', broadcast: list[bool], data: Any) -> Any:
        access = binding.access
        if access[0] != AccessType.none:
            return {
                'value': data
            }

    def update_from_bound_type(self, bound_type: 'BaseType'):
        while True:
            stshape = bound_type.get_shape()
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

    @property
    def derivative(self):
        if self._el_type is not None:
            el_diff = self._el_type.derivative
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
    usage = value.desc.usage
    if (usage & ResourceUsage.unordered_access.value) != 0:
        return RWStructuredBufferType(None)
    else:
        return StructuredBufferType(None)


PYTHON_TYPES[Buffer] = _get_or_create_python_type

PYTHON_SIGNATURES[Buffer] = lambda x: f"[{x.desc.usage}]"
