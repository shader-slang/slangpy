from types import NoneType
from typing import Optional

from kernelfunctions.backend import FunctionReflection, ModifierID, VariableReflection, TypeReflection
from kernelfunctions.typeregistry import get_or_create_type

from .basevariableimpl import BaseVariableImpl
from .basetype import BaseType
from .enums import IOType

"""
def _reflect_this(reflection: FunctionReflection, this_reflection: TypeReflection) -> 'SlangVariable':
    iot = IOType.inn
    if reflection.has_modifier(ModifierID.mutating):
        iot = IOType.inout
    type = get_or_create_type(this_reflection)
    # TODO: Check for [NoDiffThis]
    return SlangVariable(type, "_this", io_type=iot, no_diff=False, has_default=False)


def _reflect_param(reflection: VariableReflection) -> 'SlangVariable':
    if reflection.has_modifier(ModifierID.inout):
        io_type = IOType.inout
    elif reflection.has_modifier(ModifierID.out):
        io_type = IOType.out
    else:
        io_type = IOType.inn
    no_diff = reflection.has_modifier(ModifierID.nodiff)
    type = get_or_create_type(reflection.type)
    # TODO: Get actual value of has_default from reflection API
    return SlangVariable(type, reflection.name, io_type, no_diff, has_default=False)


def _reflect_return_type(reflection: FunctionReflection) -> 'SlangVariable':
    io_type = IOType.out
    # TODO: Need to check no_diff on function, not [Differentiable]
    no_diff = not reflection.has_modifier(ModifierID.differentiable)
    type = get_or_create_type(reflection.return_type)
    return SlangVariable(type, "_result", io_type, no_diff, has_default=True)


class SlangFunction:
    def __init__(self, reflection: FunctionReflection, this_reflection: Optional[TypeReflection] = None) -> NoneType:
        super().__init__()
        self.reflection = reflection
        self.name = reflection.name

        # Start with empty paramter list
        self.parameters: list[SlangVariable] = []

        # Handle 'this' parameter for class methods UNLESS a static function
        if this_reflection is not None and self.name != "$init" and not reflection.has_modifier(ModifierID.static):
            self.this = _reflect_this(reflection, this_reflection)
        else:
            self.this = None

        # Read function parameters from reflection info
        reflection_parameters = [x for x in reflection.parameters]
        self.parameters += [_reflect_param(x)
                            for i, x in enumerate(reflection_parameters)]

        # Add return value
        if reflection.return_type is not None and reflection.return_type.scalar_type != TypeReflection.ScalarType.void:
            self.return_value = _reflect_return_type(reflection)
        else:
            self.return_value = None

        # Record if function is differentiable
        self.differentiable = reflection.has_modifier(ModifierID.differentiable)


class SlangVariable(BaseVariableImpl):
    def __init__(self,
                 primal: BaseType,
                 name: str,
                 io_type: IOType,
                 no_diff: bool,
                 has_default: bool):
        super().__init__()

        self.primal = primal
        self.derivative = self.primal.derivative
        self.name = name
        self.io_type = io_type
        self.no_diff = no_diff
        self.has_default = has_default

        if primal.fields is not None:
            self.fields = {name: SlangVariable.from_parent(self, type, name)
                           for name, type in primal.fields.items()}
        else:
            self.fields = None

    @staticmethod
    def from_parent(other: 'SlangVariable', primal: BaseType, name: str) -> 'SlangVariable':
        return SlangVariable(primal,
                             name,
                             other.io_type,
                             other.no_diff,
                             other.has_default)
"""
