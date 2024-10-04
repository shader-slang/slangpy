

from typing import Any, Optional

from sgl import ResourceUsage

from kernelfunctions.core import BaseType, BoundVariable, CodeGenBlock, AccessType, BoundVariableRuntime, CallContext, Shape

from kernelfunctions.backend import Buffer, TypeReflection
from kernelfunctions.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES, SLANG_STRUCT_TYPES_BY_NAME, get_or_create_type

from .valuetype import ValueType


class StructuredBufferType(ValueType):

    def __init__(self, element_type: Optional[BaseType]):
        super().__init__()
        self.element_type = element_type
        if self.element_type is not None:
            self.name = f"StructuredBuffer<{self.element_type.name}>"
        else:
            self.name = "StructuredBuffer<Unknown>"

    def get_container_shape(self, value: Optional[Buffer] = None) -> Shape:
        if value is not None:
            return Shape(int(value.desc.size/value.desc.struct_size))
        else:
            return Shape(-1)

    def get_shape(self, value: Optional[Buffer] = None) -> Shape:
        if self.element_type is not None:
            return super().get_shape(value)
        else:
            return Shape(None)

    @property
    def differentiable(self):
        if self.element_type is not None:
            return self.element_type.differentiable
        else:
            return False

    @property
    def derivative(self):
        if self.element_type is not None:
            el_diff = self.element_type.derivative
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
    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: Any) -> Any:
        access = binding.access
        if access[0] != AccessType.none:
            return {
                'value': data
            }

    def update_from_bound_type(self, bound_type: 'BaseType'):
        while True:
            stshape = bound_type.get_shape()
            if stshape is None or -1 in stshape:
                next_type = bound_type.element_type
                if next_type == bound_type:
                    raise ValueError("Cannot resolve shape")
                assert isinstance(next_type, BaseType)
                bound_type = next_type
            else:
                break
        self.element_type = bound_type


class RWStructuredBufferType(StructuredBufferType):
    def __init__(self, element_type: Optional[BaseType]):
        super().__init__(element_type=element_type)
        self.name = "RW" + self.name

    @property
    def is_writable(self) -> bool:
        return True

    @property
    def derivative(self):
        if self.element_type is not None:
            el_diff = self.element_type.derivative
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
