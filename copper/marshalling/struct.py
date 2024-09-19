from __future__ import annotations

from .base import IOFlag, wrap, register_wrapper
from ..reflection import SlangType, StructType
from ..invokables import InvokableSlangMethod
from ..layers import Program, compute_layer

from collections import OrderedDict
from typing import Any, cast


def matches_struct(name_or_cls: str | type):
    struct_name = None

    def wrapper(cls: type):
        def init(self: Any, *args: Any, prog: Any = None, **kwargs: Any):
            if prog is None:
                raise ValueError(
                    f"Can't instantiate {cls.__name__}: Program is None. You "
                    f"likely forgot to pass prog=... to {cls.__name__}'s constructor."
                )

            cls.__init__(self, *args, **kwargs)
            Struct.__init__(self, prog, struct_name)

        return type(cls.__name__, (cls, Struct), {"__init__": init})

    if isinstance(name_or_cls, str):
        struct_name = name_or_cls
        return wrapper
    else:
        struct_name = None
        return wrapper(name_or_cls)


class Struct:
    def __init__(self, prog: Any, name: str | None = None):
        super().__init__()

        if not isinstance(prog, Program):
            prog = compute_layer().from_raw_program(prog)

        if name is None:
            name = cast(str, self.__class__.__name__)

        reflected_type = prog.find_type(name)
        if reflected_type is None:
            raise ValueError(f"Can't find struct '{name}' in slang program")
        type = reflected_type.type()
        if not isinstance(type, StructType):
            raise ValueError(f"'{name}' is not a struct")

        self._type = type

        methods = reflected_type.methods()
        for method in methods:
            self.__dict__[method.name] = InvokableSlangMethod(method, self, prog)

        register_wrapper(self.__class__, wrap_struct)

    def get_type(self) -> SlangType:
        return self._type


def wrap_struct(type: StructType, value: Struct, flag: IOFlag):
    if type != value.get_type():
        raise ValueError(
            f"Expected struct '{type}' but encountered type {value.get_type()}")

    values = OrderedDict()
    for k, v in type.members.items():
        if hasattr(value, k):
            values[k] = getattr(value, k)

    return wrap(type, values, flag)
