from io import StringIO
from typing import Any, Callable, Optional, Union

from sgl import TypeLayoutReflection

from kernelfunctions.backend import TypeReflection
from kernelfunctions.types.basetype import BaseType

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

# Dictionary of python types to corresponding base type
PYTHON_TYPES: dict[type, Union[BaseType, Callable[[Any], BaseType]]] = {}

# Slang types to corresponding base type
SLANG_SCALAR_TYPES: dict[TypeReflection.ScalarType, BaseType] = {}
SLANG_VECTOR_TYPES: dict[TypeReflection.ScalarType, list[BaseType]] = {}
SLANG_MATRIX_TYPES: dict[TypeReflection.ScalarType, list[list[BaseType]]] = {}


def _get_or_create_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    if slang_type.kind == TypeReflection.Kind.scalar:
        return SLANG_SCALAR_TYPES[slang_type.scalar_type]
    elif slang_type.kind == TypeReflection.Kind.vector:
        return SLANG_VECTOR_TYPES[slang_type.scalar_type][slang_type.col_count]
    elif slang_type.kind == TypeReflection.Kind.matrix:
        return SLANG_MATRIX_TYPES[slang_type.scalar_type][slang_type.row_count][slang_type.col_count]
    else:
        raise ValueError(f"Unsupported slang type {slang_type}")


def get_or_create_type(python_or_slang_type: Any, value: Any = None):
    if isinstance(python_or_slang_type, type):
        pt = PYTHON_TYPES[python_or_slang_type]
        if callable(pt):
            return pt(value)
        else:
            assert isinstance(pt, BaseType)
            return pt
    elif isinstance(python_or_slang_type, TypeReflection):
        return _get_or_create_slang_type_reflection(python_or_slang_type)
    elif isinstance(python_or_slang_type, TypeLayoutReflection):
        return _get_or_create_slang_type_reflection(python_or_slang_type.type)
    elif isinstance(python_or_slang_type, TypeReflection.ScalarType):
        return SLANG_SCALAR_TYPES[python_or_slang_type]
    raise ValueError(f"Unsupported type {python_or_slang_type}")
