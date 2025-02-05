# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from enum import Enum


class IOType(Enum):
    none = 0
    inn = 1
    out = 2
    inout = 3


class PrimType(Enum):
    primal = 0
    derivative = 1
