from __future__ import annotations

from .interfaces import SlangType
from .indexables import IntIndexableType

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ScalarKind
    from ..variable import AssignmentCursor


class ScalarType(SlangType):
    def __init__(self, kind: ScalarKind):
        super().__init__()
        self.kind = kind

    def to_slang_type(self) -> str:
        return self.kind.to_slang_type()

    def differentiable(self) -> bool:
        return self.kind.is_float()

    def differentiate(self) -> Optional[SlangType]:
        return self if self.differentiable() else None

    def assign_vars(self, var: AssignmentCursor, val: Any):
        if isinstance(val, int) or isinstance(val, float):
            var.value = float(val) if self.kind.is_float() else int(val)
        else:
            var.error(
                f"Expected argument of type {self.to_slang_type()}; "
                f"received {val.__class__.__name__} instead"
            )

    def __eq__(self, other: Any):
        if not isinstance(other, ScalarType):
            return NotImplemented
        return self.kind == other.kind

    def __repr__(self):
        return self.to_slang_type()


class VectorType(IntIndexableType):
    def __init__(self, dtype: ScalarType, dim: int):
        super().__init__()

        self.dtype = dtype
        self.dim = dim

    def length(self):
        return self.dim

    def element_type(self) -> SlangType:
        return self.dtype

    def to_slang_type(self) -> str:
        return f"{self.dtype.to_slang_type()}{self.dim}"

    def differentiable(self) -> bool:
        return self.dtype.differentiable()

    def differentiate(self) -> Optional[SlangType]:
        return self if self.differentiable() else None

    def __eq__(self, other: Any):
        if not isinstance(other, VectorType):
            return NotImplemented
        return self.dim == other.dim and self.dtype == other.dtype

    def __repr__(self):
        return self.to_slang_type()


class ArrayType(IntIndexableType):
    def __init__(self, dtype: SlangType, dim: int):
        super().__init__()
        self.dtype = dtype
        self.dim = dim

    def to_slang_type(self) -> str:
        return f"{self.dtype.to_slang_type()}[{self.dim}]"

    def differentiable(self) -> bool:
        return self.dtype.differentiable()

    def differentiate(self) -> Optional[SlangType]:
        if not self.differentiable():
            return None
        d_dtype = self.dtype.differentiate()
        assert d_dtype is not None
        return self if d_dtype is self.dtype else ArrayType(d_dtype, self.dim)

    def length(self):
        return self.dim

    def element_type(self) -> SlangType:
        return self.dtype

    def __eq__(self, other: Any):
        if not isinstance(other, ArrayType):
            return NotImplemented
        return self.dim == other.dim and self.dtype == other.dtype

    def __repr__(self):
        return self.to_slang_type()
