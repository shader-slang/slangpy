from types import NoneType
from typing import Optional, Union

from kernelfunctions.backend import FunctionReflection, ModifierID, VariableReflection, TypeReflection
from kernelfunctions.typeregistry import get_or_create_type

from .basevariableimpl import BaseVariableImpl
from .enums import IOType


class SlangFunction:
    def __init__(self, reflection: FunctionReflection, this_reflection: Optional[TypeReflection] = None) -> NoneType:
        super().__init__()
        self.name = reflection.name

        # Start with empty paramter list
        self.parameters = []

        # Handle 'this' parameter for class methods UNLESS an init/static function
        if this_reflection is not None and self.name != "$init" and not reflection.has_modifier(ModifierID.static):
            iot = IOType.inn
            if reflection.has_modifier(ModifierID.mutating):
                iot = IOType.inout
            self.this = SlangVariable(this_reflection, index=-1,
                                      name="_this", iotype_override=iot)
            self.parameters.append(self.this)
        else:
            self.this = None

        # Read function parameters from reflection info
        reflection_parameters = [x for x in reflection.parameters]

        # Append function parameters
        self.parameters += [SlangVariable(a, index=i)
                            for i, a in enumerate(reflection_parameters)]

        # Add return value
        if reflection.return_type is not None and reflection.return_type.scalar_type != TypeReflection.ScalarType.void:
            self.return_value = SlangVariable(reflection, index=len(self.parameters))
        else:
            self.return_value = None

        # Record if function is differentiable
        self.differentiable = reflection.has_modifier(ModifierID.differentiable)


class SlangVariable(BaseVariableImpl):
    def __init__(self,
                 reflection: Union[TypeReflection, FunctionReflection, VariableReflection, TypeReflection.ScalarType],
                 index: int,
                 parent: Optional['SlangVariable'] = None,
                 name: Optional[str] = None,
                 iotype_override: Optional[IOType] = None) -> NoneType:
        super().__init__()

        if parent is not None:
            # Child value, assume variable or scalar type + inherit modifiers
            assert isinstance(reflection, (VariableReflection, TypeReflection.ScalarType))
            io_type = parent.io_type
            no_diff = parent.no_diff
            has_default = parent.has_default
            if isinstance(reflection, TypeReflection.ScalarType):
                assert name is not None
                self.name = name
                slang_type = reflection
            else:
                assert name is None
                slang_type = reflection.type
                self.name = reflection.name
        elif isinstance(reflection, VariableReflection):
            # Function argument - check modifiers
            slang_type = reflection.type
            self.name = reflection.name
            if reflection.has_modifier(ModifierID.inout):
                io_type = IOType.inout
            elif reflection.has_modifier(ModifierID.out):
                io_type = IOType.out
            else:
                io_type = IOType.inn
            no_diff = reflection.has_modifier(ModifierID.nodiff)
            has_default = False
        elif isinstance(reflection, FunctionReflection):
            # Just a return value - always out, and only differentiable if function is
            slang_type = reflection.return_type
            self.name = "_result"
            io_type = IOType.out
            no_diff = not reflection.has_modifier(ModifierID.differentiable)
            has_default = True
        elif isinstance(reflection, TypeReflection):
            # Just a type
            slang_type = reflection
            self.name = name if name is not None else ""
            io_type = IOType.inn
            no_diff = False
            has_default = True

        if iotype_override is not None:
            io_type = iotype_override

        self.param_index = index
        self.io_type = io_type
        self.no_diff = no_diff
        self.primal = get_or_create_type(slang_type)
        self.derivative = self.primal.derivative
        self.has_default = has_default

        if isinstance(slang_type, TypeReflection):
            if slang_type.kind == TypeReflection.Kind.struct:
                self.fields = {f.name: SlangVariable(
                    f, index, self) for f in slang_type.fields}
            elif slang_type.kind == TypeReflection.Kind.vector:
                self.fields = {f: SlangVariable(slang_type.scalar_type, index, self, f) for f in [
                    "x", "y", "z", "w"][:slang_type.col_count]}
            else:
                self.fields = None
        else:
            self.fields = None

    def gen_trampoline_argument(self, differentiable: bool):
        arg_def = self.argument_declaration
        if self.io_type == IOType.inout:
            arg_def = f"inout {arg_def}"
        elif self.io_type == IOType.out:
            arg_def = f"out {arg_def}"
        elif self.io_type == IOType.inn:
            arg_def = f"in {arg_def}"
        if self.no_diff or not differentiable:
            arg_def = f"no_diff {arg_def}"
        return arg_def
