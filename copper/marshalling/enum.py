from .base import IOFlag, register_wrapper, wrap, register_assignment, assign
from ..reflection import EnumType

import enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from layers import ShaderCursor


def wrap_enum(type: EnumType, value: enum.Enum, flag: IOFlag):
    val_name = value.__class__.__name__
    if val_name != type.name.base:
        raise ValueError(f"Expected a value from enum '{type}', "
                         f"but received a value from enum {val_name}")

    return wrap(type.dtype, value.value, flag)


def assign_enum(type: EnumType, var: ShaderCursor, value: enum.Enum):
    assign(type.dtype, var, value.value)


register_wrapper(enum.Enum, wrap_enum)
register_assignment(EnumType, assign_enum)
