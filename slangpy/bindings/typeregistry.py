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


def _lookup_mro(registry: dict[type, Any], python_type: type) -> tuple[bool, Any]:
    if python_type in registry:
        return True, registry[python_type]

    for base_type in python_type.__mro__[1:]:
        if base_type in registry:
            return True, registry[base_type]

    return False, None


def lookup_type_callback(python_type: type) -> Optional[TTypeLookup]:
    found, callback = _lookup_mro(PYTHON_TYPES, python_type)
    if not found:
        return None
    return callback


def lookup_signature_callback(
    python_type: type,
) -> tuple[bool, Optional[Callable[[Any], str]]]:
    return _lookup_mro(PYTHON_SIGNATURES, python_type)


def has_registered_type_or_signature(value: Any) -> bool:
    python_type = type(value)
    if lookup_type_callback(python_type) is not None:
        return True
    found, _ = lookup_signature_callback(python_type)
    return found


def get_or_create_type(
    layout: "SlangProgramLayout", python_type: Any, value: Any = None
) -> NativeMarshall:
    """
    Use the type registry to get or create a type marshall for a given python type.
    """
    if isinstance(python_type, type):
        cb = lookup_type_callback(python_type)
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
