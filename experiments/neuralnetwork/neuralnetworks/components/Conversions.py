# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from slangpy import Module

from ..basetypes import IModel, Real, ArrayKind, RealArray, SlangType, Auto, AutoSettable, resolve_auto

from typing import Optional


class ConvertArrayKind(IModel):
    """
    Converts one array-like type to another, e.g. a float[3] to a vector<float, 3>.

    Usually this type will not be used directly, as it is more convenient to use
    one of the shorthand functions in Convert.

    Element count and element type of input and output will be identical.
    """

    def __init__(self, to_kind: ArrayKind, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__()
        self.to_kind = to_kind
        self._width = width
        self._dtype = dtype

    def model_init(self, module: Module, input_type: SlangType):
        input_array = RealArray.from_slangtype(input_type)
        self.width = resolve_auto(self._width, input_array.length)
        self.dtype = resolve_auto(self._dtype, input_array.dtype)

    def resolve_input_type(self, module: Module) -> Optional[SlangType]:
        if self._width is Auto or self._dtype is Auto:
            return None
        return RealArray(self.to_kind, self._dtype, self._width)

    @property
    def type_name(self) -> str:
        if self.to_kind == ArrayKind.array:
            suffix = "Array"
        elif self.to_kind == ArrayKind.vector:
            suffix = "Vector"
        elif self.to_kind == ArrayKind.coopvec:
            suffix = "CoopVec"
        else:
            assert False, "Invalid ArrayKind"
        return f"ConvertTo{suffix}<{self.dtype}, {self.width}>"


class ConvertArrayPrecision(IModel):
    """
    Converts the element type of an array-like type, e.g. float3 to half3.

    Usually this type will not be used directly, as it is more convenient to use
    one of the shorthand functions in Convert.

    Element count and array kind of input and output will be identical.
    """

    def __init__(self, to_dtype: Real, width: AutoSettable[int] = Auto, from_dtype: AutoSettable[Real] = Auto, kind: AutoSettable[ArrayKind] = Auto):
        super().__init__()
        self.to_dtype = to_dtype
        self._width = width
        self._from_dtype = from_dtype
        self._kind = kind

    def model_init(self, module: Module, input_type: SlangType):
        input_array = RealArray.from_slangtype(input_type)
        self.width = resolve_auto(self._width, input_array.length)
        self.from_dtype = resolve_auto(self._from_dtype, input_array.dtype)
        self.kind = resolve_auto(self._kind, input_array.kind)

    def resolve_input_type(self, module: Module) -> Optional[SlangType]:
        if self._kind is Auto or self._width is Auto or self._from_dtype is Auto:
            return None
        return RealArray(self._kind, self._from_dtype, self._width)

    @property
    def type_name(self) -> str:
        return f"ConvertArrayPrecision<{self.from_dtype}, {self.to_dtype}, {self.width}>"


class Convert(IModel):
    """
    Utility class for converting between different array-like types.

    to_array_kind or to_coopvec/to_array/to_vector may be used to convert
    the kind of the array-like type.

    to_precision or to_half/to_float/to_double may be used to convert the
    element types of the array-like type.
    """

    @staticmethod
    def to_array_kind(kind: ArrayKind):
        return ConvertArrayKind(kind)

    @staticmethod
    def to_coopvec():
        return Convert.to_array_kind(ArrayKind.coopvec)

    @staticmethod
    def to_array():
        return Convert.to_array_kind(ArrayKind.array)

    @staticmethod
    def to_vector():
        return Convert.to_array_kind(ArrayKind.vector)

    @staticmethod
    def to_precision(dtype: Real):
        return ConvertArrayPrecision(dtype)

    @staticmethod
    def to_half():
        return Convert.to_precision(Real.half)

    @staticmethod
    def to_float():
        return Convert.to_precision(Real.float)

    @staticmethod
    def to_double():
        return Convert.to_precision(Real.double)
