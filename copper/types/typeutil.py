from .interfaces import SlangType
from .base import VoidType, ScalarKind
from .arithmetic import ScalarType
from .indexables import IntIndexableType, StrIndexableType

from typing import Optional


def is_flattenable(type: SlangType) -> bool:
    return flattened_type(type) is not None


def flattened_type(type: SlangType) -> Optional[ScalarKind]:
    if isinstance(type, ScalarType):
        return type.kind
    elif isinstance(type, IntIndexableType):
        return flattened_type(type.element_type())
    return None


def type_to_shape(type: SlangType) -> tuple[int, ...]:
    if isinstance(type, ScalarType):
        return ()
    elif isinstance(type, IntIndexableType):
        return (type.length(),) + type_to_shape(type.element_type())
    raise ValueError(f"Can't map type {type} to a tensor shape")


def is_empty_type(type: SlangType):
    if isinstance(type, VoidType):
        return True
    elif isinstance(type, IntIndexableType) and type.length() == 0:
        return True
    elif isinstance(type, StrIndexableType) and not type.fields():
        return True
    return False
