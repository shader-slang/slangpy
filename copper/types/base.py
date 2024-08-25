from .interfaces import SlangType

from typing import Any
import enum


class Modifier(enum.IntFlag):
    Nothing = 0
    In = enum.auto()
    Out = enum.auto()
    InOut = enum.auto()
    NoDiff = enum.auto()
    NoDiffThis = enum.auto()
    ForwardDifferentiable = enum.auto()
    BackwardDifferentiable = enum.auto()
    Mutating = enum.auto()


class ScalarKind(enum.Enum):
    Bool = enum.auto()
    Uint8 = enum.auto()
    Uint16 = enum.auto()
    Uint = enum.auto()
    Uint64 = enum.auto()
    Int8 = enum.auto()
    Int16 = enum.auto()
    Int = enum.auto()
    Int64 = enum.auto()
    Float16 = enum.auto()
    Float = enum.auto()
    Float64 = enum.auto()

    def to_slang_type(self) -> str:
        if self is ScalarKind.Bool:
            return "bool"
        if self is ScalarKind.Uint8:
            return "uint8_t"
        if self is ScalarKind.Uint16:
            return "uint16_t"
        if self is ScalarKind.Uint:
            return "uint"
        if self is ScalarKind.Uint64:
            return "uint64_t"
        if self is ScalarKind.Int8:
            return "int8_t"
        if self is ScalarKind.Int16:
            return "int16_t"
        if self is ScalarKind.Int:
            return "int"
        if self is ScalarKind.Int64:
            return "int64_t"
        if self is ScalarKind.Float16:
            return "half"
        if self is ScalarKind.Float:
            return "float"
        if self is ScalarKind.Float64:
            return "double"
        raise RuntimeError("Invalid enum value")

    def is_float(self) -> bool:
        return (
            self is ScalarKind.Float16
            or self is ScalarKind.Float
            or self is ScalarKind.Float64
        )

    def is_integer(self) -> bool:
        return not self.is_float()


class SlangName:
    # For a type Foo::Bar::Baz<float>,
    # base = "Baz"
    # specialized = "Baz<float>"
    # declared = "Foo::Bar::Baz<float>"
    def __init__(self, base: str, specialized: str, declared: str):
        super().__init__()
        self.base = base
        self.specialized = specialized
        self.declared = declared

    def __eq__(self, other: Any):
        if not isinstance(other, SlangName):
            return NotImplemented
        # To make sure two things talk about the same type, we compare the most specific name
        return self.declared == other.declared


class VoidType(SlangType):
    def to_slang_type(self) -> str:
        return "void"

    def __eq__(self, other: Any):
        if not isinstance(other, VoidType):
            return NotImplemented
        return isinstance(other, VoidType)
