# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from .TypeLike import TypeLike
from .Auto import Auto, AutoSettable
from .Real import Real
from .CoopVecType import CoopVecType

from slangpy.reflection import SlangType, ScalarType, ArrayType, VectorType
from slangpy.backend import TypeReflection
from typing import Optional
from enum import Enum


class ArrayKind(Enum):
    array = 0
    vector = 1
    coopvec = 2


class RealArray:
    def __init__(self, kind: AutoSettable[ArrayKind], dtype: AutoSettable[Real], length: AutoSettable[int]):
        super().__init__()
        self._kind = kind
        self._dtype = dtype
        self._length = length

    def resolve(self, other: RealArray, must_match: bool = False):
        if self._kind is Auto:
            self._kind = other._kind
        if self._dtype is Auto:
            self._dtype = other._dtype
        if self._length is Auto:
            self._length = other._length

        if not self.is_resolved:
            raise ValueError("Could not fully resolve Auto values for type "
                             f"'{self.name()}' from input type '{other.name()}'")

        if must_match:
            mismatched = \
                (self._kind != other._kind and other._kind != Auto) or \
                (self._dtype != other._dtype and other._dtype != Auto) or \
                (self._length != other._length and other._length != Auto)

            if mismatched:
                raise ValueError(f"Input array type '{other}' does not "
                                 f"match required array type '{self._kind}'")

    @property
    def is_resolved(self) -> bool:
        return self._kind is not Auto and self._dtype is not Auto and self._length is not Auto

    def name(self):
        if self._kind is Auto:
            return f"AutoArrayKind<{self._dtype}, {self._length}>"
        elif self._kind == ArrayKind.array:
            return f"{self._dtype}[{self._length}]"
        elif self._kind == ArrayKind.vector:
            if isinstance(self._length, int) and self._length <= 4:
                return f"{self._dtype}{self._length}"
            else:
                return f"vector<{self._dtype}, {self._length}>"
        else:
            return f"DiffCoopVec<{self._dtype}, {self._length}>"

    def __str__(self):
        return self.name()

    @property
    def kind(self) -> ArrayKind:
        if self._kind is Auto:
            raise ValueError("Trying to access unresolved (i.e. still set "
                             "to Auto) property RealArray.kind")
        assert isinstance(self._kind, ArrayKind)
        return self._kind

    @property
    def dtype(self) -> Real:
        if self._dtype is Auto:
            raise ValueError("Trying to access unresolved (i.e. still set "
                             "to Auto) property RealArray.dtype")
        assert isinstance(self._dtype, Real)
        return self._dtype

    @property
    def length(self) -> int:
        if self._length is Auto:
            raise ValueError("Trying to access unresolved (i.e. still set "
                             "to Auto) property RealArray.length")
        assert isinstance(self._length, int)
        return self._length

    @staticmethod
    def from_slangtype(st: Optional[TypeLike]) -> RealArray:
        assert isinstance(st, SlangType)  # TODO
        if st is None:
            return RealArray(Auto, Auto, Auto)

        kind: Optional[ArrayKind] = None
        if isinstance(st, ArrayType):
            kind = ArrayKind.array
        elif isinstance(st, VectorType):
            kind = ArrayKind.vector
        elif isinstance(st, CoopVecType):
            kind = ArrayKind.coopvec

        shape = st.shape
        if kind is None or len(shape) != 1 or st.element_type is None:
            raise ValueError("Expected a 1D array-like input type (vector, array, coopvec, etc.), "
                             f"received '{st.full_name}' instead")

        dtype: Optional[Real] = None
        if isinstance(st.element_type, ScalarType):
            scalar = st.element_type.slang_scalar_type
            if scalar == TypeReflection.ScalarType.float16:
                dtype = Real.half
            elif scalar == TypeReflection.ScalarType.float32:
                dtype = Real.float
            elif scalar == TypeReflection.ScalarType.float64:
                dtype = Real.double

        if dtype is None:
            raise ValueError("Expected an input with a Real element type (half, float or double). "
                             f"Received '{st.element_type.full_name}' instead")

        return RealArray(kind, dtype, shape[0])
