from __future__ import annotations

from .interfaces import TypedValue, SlangType
from .indexables import StrIndexableType
from ..invokables import InvokableSlangMethod
from ..layers import Program, compute_layer

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SlangName
    from ..variable import AssignmentCursor


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


class Struct(TypedValue):
    def __init__(self, prog: Any, name: str | None = None):
        super().__init__()

        if not isinstance(prog, Program):
            prog = compute_layer().from_raw_program(prog)

        if name is None:
            name = self.__class__.__name__

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

    def get_type(self) -> SlangType:
        return self._type


class StructType(StrIndexableType):
    def __init__(self, name: SlangName, members: dict[str, SlangType]):
        super().__init__()

        self.name = name
        self.members = members
        self.differential: Optional[StructType] = None

    def to_slang_type(self) -> str:
        return self.name.declared

    def differentiable(self) -> bool:
        return self.differential is not None

    def differentiate(self) -> Optional[SlangType]:
        return self.differential

    def assign_vars(self, var: AssignmentCursor, val: Any):
        if isinstance(val, Struct):
            assert self is val.get_type()
            if self != var.type:
                var.error(
                    f"Expected struct '{self.name.declared}' but encountered "
                    f"type {var.type.to_slang_type()}"
                )

            var.is_leaf = False
            for k in self.members.keys():
                if k not in var and hasattr(val, k):
                    val = getattr(val, k)
                    if val is not None:
                        var[k] = val
        else:
            super().assign_vars(var, val)

    def fields(self):
        return self.members

    def __eq__(self, other: Any):
        if not isinstance(other, StructType):
            return NotImplemented
        if not (self.name == other.name and len(self.members) == len(other.members)):
            return False

        if not all(k in other.members for k in self.members.keys()):
            return False

        return all(v == other.members[k] for k, v in self.members.items())

    def __repr__(self):
        return f'StructType({repr(self.name)} {{ {", ".join(f"{repr(v)} {k}" for k, v in self.members.items())} }})'


class InterfaceType(SlangType):
    pass
