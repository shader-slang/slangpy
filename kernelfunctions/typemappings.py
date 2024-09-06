from types import NoneType
from typing import Callable, Optional, Type, Union
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


def is_match_scalar(slang_type: sgl.TypeReflection, scalar_types: list[sgl.TypeReflection.ScalarType]):
    return slang_type.kind == sgl.TypeReflection.Kind.scalar and slang_type.scalar_type in scalar_types


def is_match_vector(slang_type: sgl.TypeReflection, scalar_types: list[sgl.TypeReflection.ScalarType], dim: int):
    return slang_type.kind == sgl.TypeReflection.Kind.vector and slang_type.col_count == dim and slang_type.scalar_type in scalar_types


def is_match_matrix(slang_type: sgl.TypeReflection, scalar_types: list[sgl.TypeReflection.ScalarType], rows: int, cols: int):
    return slang_type.kind == sgl.TypeReflection.Kind.matrix and slang_type.row_count == rows and slang_type.col_count == cols and slang_type.scalar_type in scalar_types


MATCHERS: dict[type, Callable[[sgl.TypeReflection], bool]] = {}
SHAPES: dict[type, tuple[int, ...]] = {}

# Register matchers for scalar types.
MATCHERS[int] = lambda slang_type: is_match_scalar(slang_type, SLANG_ALL_INT_TYPES)
MATCHERS[float] = lambda slang_type: is_match_scalar(slang_type, SLANG_FLOAT_TYPES)
MATCHERS[bool] = lambda slang_type: is_match_scalar(slang_type, SLANG_BOOL_TYPES)
SHAPES[int] = (1,)
SHAPES[float] = (1,)
SHAPES[bool] = (1,)

# Register matchers sgl types.
for sgl_pair in zip(["int", "float", "bool", "uint", "float16_t"], [SLANG_INT_TYPES, SLANG_FLOAT_TYPES, SLANG_BOOL_TYPES, SLANG_UINT_TYPES, SLANG_FLOAT_TYPES]):
    # The scalar (i.e. float1) types
    sgl_type: type = getattr(sgl.math, f"{sgl_pair[0]}1", None)  # type: ignore
    if sgl_type is not None:
        MATCHERS[sgl_type] = lambda slang_type, scalar_types=sgl_pair[1], dim=1: is_match_scalar(
            slang_type, scalar_types)
        SHAPES[sgl_type] = (1,)

    # Vector (i.e. float2) types
    for dim in range(2, 5):
        sgl_type: type = getattr(sgl.math, f"{sgl_pair[0]}{dim}", None)  # type: ignore
        if sgl_type is not None:
            MATCHERS[sgl_type] = lambda slang_type, scalar_types=sgl_pair[1], dim=dim: is_match_vector(
                slang_type, scalar_types, dim)
            SHAPES[sgl_type] = (dim,)

    # Quaternion type
    MATCHERS[sgl.quatf] = lambda slang_type, scalar_types=sgl_pair[1], dim=dim: is_match_vector(
        slang_type, scalar_types, 4)
    SHAPES[sgl.quatf] = (4,)

    # Matrix types (note: currently only floats, search for all in case we add them later)
    for row in range(2, 5):
        for col in range(2, 5):
            sgl_type: type = getattr(
                sgl.math, f"{sgl_pair[0]}{row}x{col}", None)  # type: ignore
            if sgl_type is not None:
                MATCHERS[sgl_type] = lambda slang_type, scalar_types=sgl_pair[1], rows=row, cols=col: is_match_matrix(
                    slang_type, scalar_types, rows, cols)
                SHAPES[sgl_type] = (row, col)

# Matcher for dict
MATCHERS[dict] = lambda slang_type: slang_type.kind in [
    sgl.TypeReflection.Kind.struct, sgl.TypeReflection.Kind.vector]


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
