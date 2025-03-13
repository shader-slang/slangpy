# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Union, TypeVar


class AutoType:
    """Represents """

    def __str__(self):
        return "Auto"


Auto = AutoType()  # Singleton value representing automatically settable model arguments
T = TypeVar('T')
# AutoSettable[T] is a type hint for an argument that can take T or Auto
AutoSettable = Union[T, AutoType]


def resolve_auto(auto_settable: AutoSettable[T], default: T) -> T:
    """If the first argument is Auto, return default; otherwise, return the first argument unmodified"""
    if auto_settable is Auto:
        return default
    assert not isinstance(auto_settable, AutoType)
    return auto_settable
