from types import NoneType
from typing import Any, Callable, Literal, Optional, Type, Union
import sgl

# This file contaains lots of utilities for handling the mappiong between python,
# SGL and slang types. These mappings are used when marshalling data between the
# python api, SGL api and slang kernel generation.

TPythonScalar = Union[int, float, bool]
TSGLV1 = Union[sgl.int1, sgl.uint1, sgl.float1, sgl.bool1]
TSGLV2 = Union[sgl.int2, sgl.uint2, sgl.float2, sgl.bool2]
TSGLV3 = Union[sgl.int3, sgl.uint3, sgl.float3, sgl.bool3]
TSGLV4 = Union[sgl.int4, sgl.uint4, sgl.float4, sgl.bool4]
TSGLVector = Union[TSGLV1, TSGLV2, TSGLV3, TSGLV4]
TSGLScalar = Union[sgl.int1, sgl.uint1, sgl.float1, sgl.bool1]
TSGLOrPythonScalar = Union[TPythonScalar, TSGLScalar]
TSGLOrPython = Union[TPythonScalar, TSGLVector]
TPythonScalarName = Literal["int", "float", "bool"]
TSGLScalarName = Literal["int", "float", "bool", "uint"]

int1type = getattr(sgl, "int1")

# Suppported python types.
PYTHON_INTEGER_TYPES: list[Type[TPythonScalar]] = [int]
PYTHON_FLOAT_TYPES: list[Type[TPythonScalar]] = [float]
PYTHON_BOOL_TYPES: list[Type[TPythonScalar]] = [bool]

# Combiend list of python types.
PYTHON_SCALAR_TYPES = PYTHON_INTEGER_TYPES + PYTHON_FLOAT_TYPES + PYTHON_BOOL_TYPES

# SGL types (eg int1, int2, int3, int4).
SGL_INTEGER_TYPES: list[TSGLVector] = [getattr(sgl, f"int{i}") for i in range(1, 5)]
SGL_UNSIGNED_INTEGER_TYPES: list[TSGLVector] = [
    getattr(sgl, f"uint{i}") for i in range(1, 5)
]
SGL_FLOAT_TYPES: list[TSGLVector] = [getattr(sgl, f"float{i}") for i in range(1, 5)]
SGL_BOOL_TYPES: list[TSGLVector] = [getattr(sgl, f"bool{i}") for i in range(1, 5)]

# SGL types by base name
SGL_TYPES_BY_NAME = {
    "int": SGL_INTEGER_TYPES,
    "uint": SGL_UNSIGNED_INTEGER_TYPES,
    "float": SGL_FLOAT_TYPES,
    "bool": SGL_BOOL_TYPES,
}

# All basic types.
ALL_BASIC_TYPES = set(
    PYTHON_SCALAR_TYPES
    + SGL_INTEGER_TYPES
    + SGL_UNSIGNED_INTEGER_TYPES
    + SGL_FLOAT_TYPES
    + SGL_BOOL_TYPES
)

# SGL types from python type.
SGL_TYPES_FROM_PYTHON_TYPE: dict[Type[TPythonScalar], list[TSGLVector]] = {
    int: SGL_INTEGER_TYPES,
    float: SGL_FLOAT_TYPES,
    bool: SGL_BOOL_TYPES,
}


def is_scalar_type(python_type: type):
    return python_type in [int, float, bool, sgl.int1, sgl.uint1, sgl.float1, sgl.bool1]


# Check if python type maps directly to a slang scalar type
def is_scalar_instance(python_variable: Any):
    return isinstance(
        python_variable, (int, float, bool, sgl.int1, sgl.uint1, sgl.float1, sgl.bool1)
    )


# Get an SGL type by its name (eg int) and dimensionality (1-4).
def get_sgl_type_by_name(type_name: TSGLScalarName, dim: int):
    return SGL_TYPES_BY_NAME[type_name][dim - 1]


# Get an SGL type by python type and dimensionality (eg python int -> sgl int[1-4]).
def python_2_sgl_type(python_type: Type[TPythonScalar], dim: int):
    return SGL_TYPES_FROM_PYTHON_TYPE[python_type][dim - 1]


# List of slang scalar type ids by base type.
SLANG_BOOL_TYPES = [
    sgl.TypeReflection.ScalarType.bool,
]
SLANG_INT_TYPES = [
    sgl.TypeReflection.ScalarType.int8,
    sgl.TypeReflection.ScalarType.int16,
    sgl.TypeReflection.ScalarType.int32,
    sgl.TypeReflection.ScalarType.int64,
]
SLANG_UINT_TYPES = [
    sgl.TypeReflection.ScalarType.uint8,
    sgl.TypeReflection.ScalarType.uint16,
    sgl.TypeReflection.ScalarType.uint32,
    sgl.TypeReflection.ScalarType.uint64,
]
SLANG_FLOAT_TYPES = [
    sgl.TypeReflection.ScalarType.float16,
    sgl.TypeReflection.ScalarType.float32,
    sgl.TypeReflection.ScalarType.float64,
]
SLANG_ALL_INT_TYPES = SLANG_INT_TYPES + SLANG_UINT_TYPES
SLANG_SCALAR_TYPES = (
    SLANG_BOOL_TYPES + SLANG_INT_TYPES + SLANG_UINT_TYPES + SLANG_FLOAT_TYPES
)

# Generate larger list that contains valid conversions for each additional dimension.
VALID_CONVERSIONS: list[dict[sgl.TypeReflection.ScalarType, set[TSGLOrPython]]] = [
    {}
] * 5
for dim in range(1, 5):
    res = {}
    res.update(
        {
            x: set(
                [
                    bool,
                    get_sgl_type_by_name("bool", 1),
                    get_sgl_type_by_name("bool", dim),
                ]
            )
            for x in SLANG_BOOL_TYPES
        }
    )
    res.update(
        {
            x: set(
                [int, get_sgl_type_by_name("int", 1), get_sgl_type_by_name("int", dim)]
            )
            for x in SLANG_INT_TYPES
        }
    )
    res.update(
        {
            x: set(
                [
                    int,
                    get_sgl_type_by_name("uint", 1),
                    get_sgl_type_by_name("uint", dim),
                ]
            )
            for x in SLANG_UINT_TYPES
        }
    )
    res.update(
        {
            x: set(
                [
                    float,
                    get_sgl_type_by_name("float", 1),
                    get_sgl_type_by_name("float", dim),
                ]
            )
            for x in SLANG_FLOAT_TYPES
        }
    )
    VALID_CONVERSIONS[dim] = res


# Helper to check if conversion from a python type to a scalar slang type is valid.
# Note: This is really just checking against the 1D vector types.
def is_valid_scalar_type_conversion(
    slang_scalar_type: sgl.TypeReflection.ScalarType, python_type: Any
):
    valid_conversions = VALID_CONVERSIONS[1][slang_scalar_type]
    return python_type in valid_conversions


# Helper to check if conversion from a python type to a scalar slang type is valid.
def is_valid_vector_type_conversion(
    slang_scalar_type: sgl.TypeReflection.ScalarType, python_type: Any, dim: int
):
    valid_conversions = VALID_CONVERSIONS[dim][slang_scalar_type]
    return python_type in valid_conversions


ELEMENT_TYPE_TO_SLANG_TYPE: dict[type, tuple[sgl.TypeReflection.Kind, int, int]] = {}


def _reg_mapping(sgl_type: type, kind: sgl.TypeReflection.Kind, rows: int, cols: int):
    ELEMENT_TYPE_TO_SLANG_TYPE[sgl_type] = (kind, rows, cols)


def is_match_scalar(slang_type: sgl.TypeReflection, scalar_types: list[sgl.TypeReflection.ScalarType]):
    return slang_type.kind == sgl.TypeReflection.Kind.scalar and slang_type.scalar_type in scalar_types


def is_match_vector(slang_type: sgl.TypeReflection, scalar_types: list[sgl.TypeReflection.ScalarType], dim: int):
    return slang_type.kind == sgl.TypeReflection.Kind.vector and slang_type.col_count == dim and slang_type.scalar_type in scalar_types


def is_match_matrix(slang_type: sgl.TypeReflection, scalar_types: list[sgl.TypeReflection.ScalarType], rows: int, cols: int):
    return slang_type.kind == sgl.TypeReflection.Kind.matrix and slang_type.row_count == rows and slang_type.col_count == cols and slang_type.scalar_type in scalar_types


MATCHERS: dict[type, Callable[[sgl.TypeReflection], bool]] = {}

# Register matchers for scalar types.
MATCHERS[int] = lambda slang_type: is_match_scalar(slang_type, SLANG_ALL_INT_TYPES)
MATCHERS[float] = lambda slang_type: is_match_scalar(slang_type, SLANG_FLOAT_TYPES)
MATCHERS[bool] = lambda slang_type: is_match_scalar(slang_type, SLANG_BOOL_TYPES)

# Register matchers sgl types.
for sgl_pair in zip(["int", "float", "bool", "uint", "float16_t"], [SLANG_INT_TYPES, SLANG_FLOAT_TYPES, SLANG_BOOL_TYPES, SLANG_UINT_TYPES, SLANG_FLOAT_TYPES]):
    # The scalar (i.e. float1) types
    sgl_type: type = getattr(sgl.math, f"{sgl_pair[0]}1", None)  # type: ignore
    if sgl_type is not None:
        MATCHERS[sgl_type] = lambda slang_type, scalar_types=sgl_pair[1], dim=dim: is_match_scalar(
            slang_type, scalar_types)

    # Vector (i.e. float2) types
    for dim in range(2, 5):
        sgl_type: type = getattr(sgl.math, f"{sgl_pair[0]}{dim}", None)  # type: ignore
        if sgl_type is not None:
            MATCHERS[sgl_type] = lambda slang_type, scalar_types=sgl_pair[1], dim=dim: is_match_vector(
                slang_type, scalar_types, dim)

    # Quaternion type
    MATCHERS[sgl.quatf] = lambda slang_type, scalar_types=sgl_pair[1], dim=dim: is_match_vector(
        slang_type, scalar_types, 4)

    # Matrix types (note: currently only floats, search for all in case we add them later)
    for row in range(2, 5):
        for col in range(2, 5):
            sgl_type: type = getattr(
                sgl.math, f"{sgl_pair[0]}{row}x{col}", None)  # type: ignore
            if sgl_type is not None:
                MATCHERS[sgl_type] = lambda slang_type, scalar_types=sgl_pair[1], rows=row, cols=col: is_match_matrix(
                    slang_type, scalar_types, rows, cols)

# Matcher for dict
MATCHERS[dict] = lambda slang_type: slang_type.kind == sgl.TypeReflection.Kind.struct


def are_element_types_compatible(
        element_type: Optional[Union[Type[TSGLVector], Type[TPythonScalar], sgl.TypeReflection, sgl.TypeLayoutReflection]],
        slang_type: sgl.TypeReflection
):
    """
    Checks if a core element type is compatible with a slang type
    """
    if element_type is NoneType:
        return True
    elif isinstance(element_type, sgl.TypeReflection):
        return element_type.full_name == slang_type.full_name
    elif isinstance(element_type, sgl.TypeLayoutReflection):
        return element_type.type.full_name == slang_type.full_name
    elif isinstance(element_type, type):
        matcher = MATCHERS.get(element_type, None)
        if matcher is not None:
            return matcher(slang_type)
    return False
