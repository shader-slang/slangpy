

from enum import Enum


# Native enums
from kernelfunctions.backend import slangpynative
AccessType = slangpynative.AccessType


class CallMode(Enum):
    prim = 0
    bwds = 1
    fwds = 2


class IOType(Enum):
    none = 0
    inn = 1
    out = 2
    inout = 3


class PrimType(Enum):
    primal = 0
    derivative = 1
