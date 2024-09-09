from io import StringIO
from typing import Any, Callable, Optional, Union

from kernelfunctions.backend import TypeReflection
from kernelfunctions.types import PythonMarshal, SlangMarshall

# Dictionary of python types to corresponding marshall
PYTHON_TYPE_MARSHAL: dict[type, PythonMarshal] = {}

# Dictionary of python types to corresponding hash functions
PYTHON_SIGNATURE_HASH: dict[type, Optional[Callable[[StringIO, Any], Any]]] = {
    int: None,
    float: None,
    bool: None,
    str: None,
    list: None,
    dict: None,
    tuple: None,
}

# Register a mapping from type to marshall


def register_python_type(
    python_type: type,
    marshall: PythonMarshal,
    hash_fn: Optional[Callable[[StringIO, Any], Any]],
):
    PYTHON_TYPE_MARSHAL[python_type] = marshall
    PYTHON_SIGNATURE_HASH[python_type] = hash_fn


def get_python_type_marshall(value: Any) -> PythonMarshal:
    if isinstance(value, type):
        return PYTHON_TYPE_MARSHAL[value]
    else:
        return PYTHON_TYPE_MARSHAL[type(value)]


# Create slang marshall for reflection type
SLANG_MARSHALS_BY_FULL_NAME: dict[str, type[SlangMarshall]] = {}
SLANG_MARSHALS_BY_NAME: dict[str, type[SlangMarshall]] = {}
SLANG_MARSHALS_BY_KIND: dict[TypeReflection.Kind, type[SlangMarshall]] = {}
SLANG_MARSHALS_BY_SCALAR_TYPE: dict[TypeReflection.ScalarType,
                                    type[SlangMarshall]] = {}


def create_slang_type_marshal(slang_type: Union[TypeReflection, TypeReflection.ScalarType]) -> SlangMarshall:
    """
    Looks up correct marshall for a given slang type using
    first full name search, then base name search, then kind
    """
    if isinstance(slang_type, TypeReflection):
        marshal = SLANG_MARSHALS_BY_FULL_NAME.get(slang_type.full_name, None)
        if marshal is not None:
            return marshal(slang_type)
        marshal = SLANG_MARSHALS_BY_NAME.get(slang_type.name, None)
        if marshal is not None:
            return marshal(slang_type)
        marshal = SLANG_MARSHALS_BY_KIND.get(slang_type.kind, None)
        if marshal is not None:
            return marshal(slang_type)
        raise ValueError(f"Unsupported slang type {slang_type.full_name}")
    else:
        marshal = SLANG_MARSHALS_BY_SCALAR_TYPE.get(slang_type, None)
        if marshal is not None:
            return marshal(slang_type)
        raise ValueError(f"Unsupported slang type {slang_type}")
