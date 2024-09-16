from __future__ import annotations

from collections import OrderedDict
import enum

from typing import Any, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..layers import ReflectedType


class SlangType:
    opaque_types = {}

    def differentiable(self) -> bool:
        return False

    def differentiate(self) -> SlangType:
        raise RuntimeError("Trying to differentiate non-differentiable type")

    def __str__(self) -> str:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    def __eq__(self, value: object) -> bool:
        raise NotImplementedError


def opaque_type(name: str, parent: Optional[type[SlangType]] = None, args: Optional[list[type]] = None):
    return lambda cls: cls  # TODO
# def opaque_type(*names: str):
#    def register_opaque_type(cls: type):
#        for name in names:
#            SlangType.opaque_types[name] = cls
#        return cls
#
#    return register_opaque_type


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


class ScalarKind(enum.IntEnum):
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

    def __str__(self) -> str:
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

    def __hash__(self) -> int:
        return hash(self.declared)

    def __eq__(self, other: Any):
        if not isinstance(other, SlangName):
            return NotImplemented
        # To make sure two things talk about the same type, we compare the most specific name
        return self.declared == other.declared


class VoidType(SlangType):
    def __str__(self) -> str:
        return "void"

    def __hash__(self) -> int:
        return hash(VoidType)

    def __eq__(self, other: Any):
        if not isinstance(other, VoidType):
            return NotImplemented
        return isinstance(other, VoidType)


class ScalarType(SlangType):
    def __init__(self, kind: ScalarKind):
        super().__init__()
        self.kind = kind

    def differentiable(self) -> bool:
        return self.kind.is_float()

    def differentiate(self) -> SlangType:
        assert self.differentiable()
        return self

    def __str__(self) -> str:
        return str(self.kind)

    def __hash__(self) -> int:
        return hash((ScalarType, self.kind))

    def __eq__(self, other: Any):
        if not isinstance(other, ScalarType):
            return NotImplemented
        return self.kind == other.kind


class IntIndexableType(SlangType):
    def length(self) -> int:
        raise NotImplementedError

    def element_type(self) -> SlangType:
        raise NotImplementedError


class StrIndexableType(SlangType):
    def fields(self) -> dict[str, SlangType]:
        raise NotImplementedError


class VectorType(IntIndexableType):
    def __init__(self, dtype: ScalarType, dim: int):
        super().__init__()

        self.dtype = dtype
        self.dim = dim

    def length(self):
        return self.dim

    def element_type(self) -> SlangType:
        return self.dtype

    def differentiable(self) -> bool:
        return self.dtype.differentiable()

    def differentiate(self) -> SlangType:
        assert self.differentiable()
        return self

    def __str__(self) -> str:
        return f"{self.dtype}{self.dim}"

    def __hash__(self) -> int:
        return hash((VectorType, self.dtype, self.dim))

    def __eq__(self, other: Any):
        if not isinstance(other, VectorType):
            return NotImplemented
        return self.dim == other.dim and self.dtype == other.dtype


class ArrayType(IntIndexableType):
    def __init__(self, dtype: SlangType, dim: int):
        super().__init__()
        self.dtype = dtype
        self.dim = dim

    def differentiable(self) -> bool:
        return self.dtype.differentiable()

    def differentiate(self) -> SlangType:
        assert self.differentiable()
        d_dtype = self.dtype.differentiate()
        return self if d_dtype is self.dtype else ArrayType(d_dtype, self.dim)

    def length(self):
        return self.dim

    def element_type(self) -> SlangType:
        return self.dtype

    def __str__(self) -> str:
        return f"{self.dtype}[{self.dim}]"

    def __hash__(self) -> int:
        return hash((ArrayType, self.dtype, self.dim))

    def __eq__(self, other: Any):
        if not isinstance(other, ArrayType):
            return NotImplemented
        return self.dim == other.dim and self.dtype == other.dtype


class EnumType(SlangType):
    def __init__(
        self,
        name: SlangName,
        dtype: SlangType,
        is_flags: bool,
        cases: list[tuple[str, int]],
    ):
        super().__init__()
        self.name = name
        self.dtype = dtype
        self.is_flags = is_flags
        self.cases = cases

    def __str__(self) -> str:
        return self.name.declared

    def __hash__(self) -> int:
        return hash((EnumType, self.name))

    def __eq__(self, other: Any):
        if not isinstance(other, EnumType):
            return NotImplemented
        return self.name == other.name and self.dtype == other.dtype


class StructType(StrIndexableType):
    def __init__(self, name: SlangName, members: dict[str, SlangType]):
        super().__init__()

        self.name = name
        self.members = members
        self.differential: Optional[StructType] = None

    def differentiable(self) -> bool:
        return self.differential is not None

    def differentiate(self) -> SlangType:
        assert self.differential is not None
        return self.differential

    def fields(self):
        return self.members

    def __str__(self) -> str:
        return self.name.declared

    def __hash__(self) -> int:
        return hash((StructType, self.name))

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


@opaque_type("DifferentialPair")
class DifferentialPairType(StrIndexableType):
    def __init__(self, primal: SlangType):
        assert primal.differentiable()
        super().__init__()
        self.primal = primal
        self.differential = primal.differentiate()

    def differentiable(self) -> bool:
        return True

    def differentiate(self) -> SlangType:
        if self.primal is self.differential:
            return self
        return DifferentialPairType(self.differential)

    def fields(self) -> dict[str, SlangType]:
        return OrderedDict([("p", self.primal), ("d", self.differential)])

    def __str__(self):
        return f"DifferentialPair<{self.primal}>"

    def __hash__(self) -> int:
        return hash((DifferentialPairType, self.primal))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DifferentialPairType):
            return NotImplemented

        return self.primal == other.primal

    @staticmethod
    def from_reflection(name: SlangName, type: ReflectedType) -> DifferentialPairType:
        return DifferentialPairType(type.generic_args()[0])
