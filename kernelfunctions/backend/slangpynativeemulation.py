
"""
This file contains python-only emulation for the current native functionality of slangpy embedded in SGL
"""

from enum import Enum
from typing import Any, Callable, Optional


class AccessType(Enum):
    none = 0
    read = 1
    write = 2
    readwrite = 3


class NativeType:
    """
    Native base class for all type marshalls
    """

    def __init__(self):
        super().__init__()


def hash_signature(value_to_id: Callable[[Any], str], *args: Any, **kwargs: Any) -> str:
    """
    Generates a unique hash for a given python signature
    """

    x = []

    x.append("args\n")
    for arg in args:
        x.append(f"N:")
        _get_value_signature(value_to_id, arg, x)
        x.append("\n")

    x.append("kwargs\n")
    for k, v in kwargs.items():
        x.append(f"{k}:")
        _get_value_signature(value_to_id, v, x)
        x.append("\n")

    text = "".join(x)
    return text


def _get_value_signature(value_to_id: Callable[[Any], str], x: Any, out: list[str]):
    """
    Recursively get the signature of x
    """

    out.append(type(x).__name__)

    s = getattr(x, "get_this", None)
    if s is not None:
        _get_value_signature(value_to_id, s(), out)
        return

    s = getattr(x, "slangpy_signature", None)
    if s is not None:
        out.append(s)
        return

    if isinstance(x, dict):
        out.append("\n")
        for k, v in x.items():
            out.append(f"{k}:\n")
            _get_value_signature(value_to_id, v, out)
            return

    s = value_to_id(x)
    if s is not None:
        out.append(s)
        return
