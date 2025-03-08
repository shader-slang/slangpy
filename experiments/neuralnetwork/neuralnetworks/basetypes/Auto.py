# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Union, TypeVar


class AutoType:
    def __str__(self):
        return "Auto"


Auto = AutoType()
T = TypeVar('T')
AutoSettable = Union[T, AutoType]
