# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from slangpy.backend import DataType, TypeReflection
from slangpy.reflection import SlangType, ScalarType

from typing import Optional
from enum import Enum
import numpy as np


class Real(Enum):
    """
    Utility class that represents slang types that implement IReal.

    This should be used when specifying the precision or element type
    of a model to unify the network API.
    A Real may be converted to data types in different APIs: .numpy()
    for a numpy.dtype, .slang() for a TypeReflection.ScalarType, and
    .sgl() for an sgl.DataType.
    __str__() will return a string that can be used to declare the type
    in slang, and may be used e.g. in IModel.type_name to specify generic
    parameters.

    Types implementing IReal currently are half, float and double, but this is
    subject to change. Generally, members of IReal will be types that represent
    a differentiable scalar.
    In slang, IReal is currently just an alias for __BuiltinFloatingPointType,
    but this is subject to change.
    """

    half = 1
    float = 2
    double = 3

    def __str__(self):
        """
        Returns a string uniquely identifying the Real.

        May be used to declare the type in slang, e.g. when specifying generic parameters
        in IModel.type_name
        """
        return self._name_

    def numpy(self):
        """Returns the equivalent numpy.dtype"""
        if self is Real.half:
            return np.float16
        elif self is Real.float:
            return np.float32
        elif self is Real.double:
            return np.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def slang(self):
        """Returns the equivalent slang reflection enum, TypeReflection.ScalarType"""
        if self is Real.half:
            return TypeReflection.ScalarType.float16
        elif self is Real.float:
            return TypeReflection.ScalarType.float32
        elif self is Real.double:
            return TypeReflection.ScalarType.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def sgl(self):
        """Returns the equivalent sgl.DataType"""
        if self is Real.half:
            return DataType.float16
        elif self is Real.float:
            return DataType.float32
        elif self is Real.double:
            return DataType.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def size(self):
        """Returns the size (in bytes) of this Real type"""
        if self is Real.half:
            return 2
        elif self is Real.float:
            return 4
        elif self is Real.double:
            return 8
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    @staticmethod
    def from_slangtype(st: Optional[SlangType]) -> Optional[Real]:
        """Tries to convert a slang type reflection to a Real. Returns None on failure"""
        if not isinstance(st, ScalarType):
            return None

        if st.slang_scalar_type == TypeReflection.ScalarType.float16:
            return Real.half
        elif st.slang_scalar_type == TypeReflection.ScalarType.float32:
            return Real.float
        elif st.slang_scalar_type == TypeReflection.ScalarType.float64:
            return Real.double

        return None
