# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from enum import Enum

from slangpy.native_refl import IOType


class PrimType(Enum):
    primal = 0
    derivative = 1
