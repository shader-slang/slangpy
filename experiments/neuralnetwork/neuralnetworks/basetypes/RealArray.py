# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from .Real import Real
from .CoopVecType import CoopVecType

from slangpy.reflection import SlangType, ScalarType, ArrayType, VectorType
from slangpy.backend import TypeReflection
from typing import Optional
from enum import Enum


class ArrayKind(Enum):
    """
    Enum specifying the kind of an array-like type.

    ArrayKind.array maps to plain arrays, e.g. float[10], ArrayKind.vector maps
    to e.g. float2, and ArrayKind.coopvec maps to DiffCoopVec
    """
    array = 0
    vector = 1
    coopvec = 2

    def __str__(self):
        return self._name_


class RealArray:
    """
    Utility class for representing 1D array-like slang types (plain arrays, vectors, or CoopVec) of Reals.

    This may be used to specify array-like types (e.g. for passing to IModel.initialize) or
    inspecting reflected slang types (e.g. inside IModel.model_init).

    RealArrays may be constructed directly, or from a reflected SlangType via RealArray.from_slangtype.
    .name() or __str__() return the equivalent slang type declaration, and may be used to e.g.
    to specify types in IModel.type_name
    """

    def __init__(self, kind: ArrayKind, dtype: Real, length: int):
        super().__init__()
        self.kind = kind
        self.dtype = dtype
        self.length = length

    def name(self):
        """Returns the equivalent definition of the type in slang."""
        if self.kind == ArrayKind.array:
            return f"{self.dtype}[{self.length}]"
        elif self.kind == ArrayKind.vector:
            if self.length <= 4:
                return f"{self.dtype}{self.length}"
            else:
                return f"vector<{self.dtype}, {self.length}>"
        else:
            return f"DiffCoopVec<{self.dtype}, {self.length}>"

    def __str__(self):
        """Alias for .name(). Returns the equivalent definition of the type in slang."""
        return self.name()

    @staticmethod
    def from_slangtype(st: SlangType) -> RealArray:
        """
        Attempts to parse a RealArray from a SlangType.

        Throws if argument is not an array-like type (plain array, vector, CoopVec) of Reals.
        """
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
