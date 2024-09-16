from .invokables import InvokableSlangFunc
from .reflection import EnumType
from .layers import Program, compute_layer

from typing import Any
import enum


def import_function(prog: Any, func_name: str) -> InvokableSlangFunc:
    if not isinstance(prog, Program):
        prog = compute_layer().from_raw_program(prog)
    return InvokableSlangFunc(prog, func_name)


def import_enum(prog: Any, name: str) -> enum.IntEnum | enum.IntFlag:
    if not isinstance(prog, Program):
        prog = compute_layer().from_raw_program(prog)

    enum_type = prog.find_type(name)
    if enum_type is None:
        raise ValueError(f"Could not find enum '{name}'")
    enum_type = enum_type.type()
    if not isinstance(enum_type, EnumType):
        raise ValueError(f"'{name}' is not an enum")

    enum_class = enum.IntFlag if enum_type.is_flags else enum.IntEnum
    return enum_class(enum_type.name, enum_type.cases)
