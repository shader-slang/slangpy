# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Union, TypeVar


class AutoType:
    def __str__(self):
        return "Auto"


Auto = AutoType()
T = TypeVar('T')
AutoSettable = Union[T, AutoType]


def resolve_auto(auto_settable: AutoSettable[T], default: T) -> T:
    if auto_settable is Auto:
        return default
    assert not isinstance(auto_settable, AutoType)
    return auto_settable
