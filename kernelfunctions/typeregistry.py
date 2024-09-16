from io import StringIO
from typing import Any, Callable, Optional, Union

from kernelfunctions.core import BaseType

from kernelfunctions.backend import TypeReflection, TypeLayoutReflection

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

TTypeLookup = Union[BaseType, Callable[[Any], BaseType]]

# Dictionary of python types to corresponding base type
PYTHON_TYPES: dict[type, TTypeLookup] = {}

# Slang types to corresponding base type
SLANG_SCALAR_TYPES: dict[TypeReflection.ScalarType, BaseType] = {}
SLANG_VECTOR_TYPES: dict[TypeReflection.ScalarType, list[BaseType]] = {}
SLANG_MATRIX_TYPES: dict[TypeReflection.ScalarType, list[list[BaseType]]] = {}
SLANG_STRUCT_TYPES_BY_FULL_NAME: dict[str, TTypeLookup] = {}
SLANG_STRUCT_TYPES_BY_NAME: dict[str, TTypeLookup] = {}
SLANG_STRUCT_BASE_TYPE: TTypeLookup = None  # type: ignore


def _get_or_create_slang_type_reflection(slang_type: TypeReflection) -> TTypeLookup:
    if slang_type.kind == TypeReflection.Kind.scalar:
        res = SLANG_SCALAR_TYPES[slang_type.scalar_type]
    elif slang_type.kind == TypeReflection.Kind.vector:
        res = SLANG_VECTOR_TYPES[slang_type.scalar_type][slang_type.col_count]
    elif slang_type.kind == TypeReflection.Kind.matrix:
        res = SLANG_MATRIX_TYPES[slang_type.scalar_type][slang_type.row_count][slang_type.col_count]
    elif slang_type.kind == TypeReflection.Kind.struct:
        res = SLANG_STRUCT_TYPES_BY_FULL_NAME.get(slang_type.full_name)
        if res is None:
            res = SLANG_STRUCT_TYPES_BY_NAME.get(slang_type.name)
        if res is None:
            res = SLANG_STRUCT_BASE_TYPE
    else:
        res = SLANG_STRUCT_TYPES_BY_FULL_NAME.get(slang_type.full_name)
        if res is None:
            res = SLANG_STRUCT_TYPES_BY_NAME.get(slang_type.name)
    if callable(res):
        res = res(slang_type)
    if res is None:
        raise ValueError(f"Unsupported slang type {slang_type}")
    return res


def get_or_create_type(python_or_slang_type: Any, value: Any = None) -> BaseType:
    res: Optional[TTypeLookup] = None
    if isinstance(python_or_slang_type, type):
        res = PYTHON_TYPES[python_or_slang_type]
    elif isinstance(python_or_slang_type, BaseType):
        res = python_or_slang_type
    elif isinstance(python_or_slang_type, TypeReflection):
        res = _get_or_create_slang_type_reflection(python_or_slang_type)
    elif isinstance(python_or_slang_type, TypeLayoutReflection):
        res = _get_or_create_slang_type_reflection(python_or_slang_type.type)
    elif isinstance(python_or_slang_type, TypeReflection.ScalarType):
        res = SLANG_SCALAR_TYPES[python_or_slang_type]
    if callable(res):
        res = res(value)
    if res is None:
        raise ValueError(f"Unsupported type {python_or_slang_type}")
    assert isinstance(res, BaseType)
    return res
