# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from slangpy.bindings.marshall import Marshall
from slangpy.core.native import NativeMarshall

if TYPE_CHECKING:
    from slangpy.reflection import SlangProgramLayout

TTypeLookup = Callable[["SlangProgramLayout", Any], Union[Marshall, NativeMarshall]]

#: Dictionary of python types to function that allocates a corresponding type
#: marshall.
PYTHON_TYPES: dict[type, TTypeLookup] = {}

#: Dictionary of python types to custom function that returns a signature
#: Note: preferred mechanism is to provide a slangpy_signature attribute
PYTHON_SIGNATURES: dict[type, Optional[Callable[[Any], str]]] = {}


def get_or_create_type(
    layout: "SlangProgramLayout", python_type: Any, value: Any = None
) -> NativeMarshall:
    """
    Use the type registry to get or create a type marshall for a given python type.
    """
    if isinstance(python_type, type):
        cb = PYTHON_TYPES.get(python_type)
        if cb is None:
            raise ValueError(f"Unsupported type {python_type}")
        res = cb(layout, value)
        if res is None:
            raise ValueError(f"Unsupported type {python_type}")
        return res
    elif isinstance(python_type, Marshall):
        return python_type
    else:
        raise ValueError(f"Unsupported type {python_type}")
