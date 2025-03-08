# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from slangpy.backend import DataType, TypeReflection
from slangpy.reflection import SlangType, ScalarType

from typing import Optional
from enum import Enum
import numpy as np


class Real(Enum):
    half = 1
    float = 2
    double = 3

    def __str__(self):
        return self._name_

    def numpy(self):
        if self is Real.half:
            return np.float16
        elif self is Real.float:
            return np.float32
        elif self is Real.double:
            return np.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def slang(self):
        if self is Real.half:
            return TypeReflection.ScalarType.float16
        elif self is Real.float:
            return TypeReflection.ScalarType.float32
        elif self is Real.double:
            return TypeReflection.ScalarType.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def sgl(self):
        if self is Real.half:
            return DataType.float16
        elif self is Real.float:
            return DataType.float32
        elif self is Real.double:
            return DataType.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def size(self):
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
        if not isinstance(st, ScalarType):
            return None

        if st.slang_scalar_type == TypeReflection.ScalarType.float16:
            return Real.half
        elif st.slang_scalar_type == TypeReflection.ScalarType.float32:
            return Real.float
        elif st.slang_scalar_type == TypeReflection.ScalarType.float64:
            return Real.double

        return None
