from types import NoneType
from typing import Union

from kernelfunctions.backend import TypeReflection
from kernelfunctions.typeregistry import create_slang_type_marshal
from kernelfunctions.types.enums import IOType, PrimType


class SlangValue:
    def __init__(self,
                 name: str,
                 io_type: IOType,
                 no_diff: bool,
                 primal_type: Union[TypeReflection, TypeReflection.ScalarType]) -> NoneType:
        super().__init__()
        self.name = name
        self.type = type
        self.io_type = io_type
        self.no_diff = no_diff
        self.primal_type = primal_type
        self.primal = create_slang_type_marshal(primal_type)
        self.derivative = self.primal.differentiate()

    def load_primal_fields(self):
        if isinstance(self.primal_type, TypeReflection):
            return self.primal.load_fields(self.primal_type)
        else:
            return None

    def get(self, t: PrimType):
        if t == PrimType.primal:
            return self.primal
        else:
            assert self.derivative is not None
            return self.derivative

    @property
    def primal_type_name(self):
        return self.primal.name

    @property
    def derivative_type_name(self):
        return self.derivative.name if self.derivative is not None else None

    @property
    def argument_declaration(self):
        return f"{self.primal_type_name} {self.name}"

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
