

from enum import Enum


class AccessType(Enum):
    none = 0
    read = 1
    write = 2
    readwrite = 3


class CallMode(Enum):
    prim = 0
    bwds = 1
    fwds = 2


class IOType(Enum):
    none = 0
    inn = 1
    out = 2
    inout = 3
