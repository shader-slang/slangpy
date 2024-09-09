from types import NoneType
from typing import Callable, Optional, Type, Union
from kernelfunctions.backend import (
    int1, int2, int3, int4,
    float1, float2, float3, float4,
    uint1, uint2, uint3, uint4,
    bool1, bool2, bool3, bool4,
    quatf,
    TypeReflection, TypeLayoutReflection,
    math
)


# This file contaains lots of utilities for handling the mappiong between python,
# SGL and slang types. These mappings are used when marshalling data between the
# python api, SGL api and slang kernel generation.

TPythonScalar = Union[int, float, bool]
TSGLV1 = Union[int1, uint1, float1, bool1]
TSGLV2 = Union[int2, uint2, float2, bool2]
TSGLV3 = Union[int3, uint3, float3, bool3]
TSGLV4 = Union[int4, uint4, float4, bool4]
TSGLVector = Union[TSGLV1, TSGLV2, TSGLV3, TSGLV4]

# List of slang scalar type ids by base type.
SLANG_BOOL_TYPES = [
    TypeReflection.ScalarType.bool,
]
SLANG_INT_TYPES = [
    TypeReflection.ScalarType.int8,
    TypeReflection.ScalarType.int16,
    TypeReflection.ScalarType.int32,
    TypeReflection.ScalarType.int64,
]
SLANG_UINT_TYPES = [
    TypeReflection.ScalarType.uint8,
    TypeReflection.ScalarType.uint16,
    TypeReflection.ScalarType.uint32,
    TypeReflection.ScalarType.uint64,
]
SLANG_FLOAT_TYPES = [
    TypeReflection.ScalarType.float16,
    TypeReflection.ScalarType.float32,
    TypeReflection.ScalarType.float64,
]
SLANG_ALL_INT_TYPES = SLANG_INT_TYPES + SLANG_UINT_TYPES


def is_match_scalar(slang_type: TypeReflection, scalar_types: list[TypeReflection.ScalarType]):
    return slang_type.kind == TypeReflection.Kind.scalar and slang_type.scalar_type in scalar_types


def is_match_vector(slang_type: TypeReflection, scalar_types: list[TypeReflection.ScalarType], dim: int):
    return slang_type.kind == TypeReflection.Kind.vector and slang_type.col_count == dim and slang_type.scalar_type in scalar_types


def is_match_matrix(slang_type: TypeReflection, scalar_types: list[TypeReflection.ScalarType], rows: int, cols: int):
    return slang_type.kind == TypeReflection.Kind.matrix and slang_type.row_count == rows and slang_type.col_count == cols and slang_type.scalar_type in scalar_types


MATCHERS: dict[type, Callable[[TypeReflection], bool]] = {}
SHAPES: dict[type, tuple[int, ...]] = {}

# Register matchers for scalar types.
MATCHERS[int] = lambda slang_type: is_match_scalar(slang_type, SLANG_ALL_INT_TYPES)
MATCHERS[float] = lambda slang_type: is_match_scalar(slang_type, SLANG_FLOAT_TYPES)
MATCHERS[bool] = lambda slang_type: is_match_scalar(slang_type, SLANG_BOOL_TYPES)
SHAPES[int] = (1,)
SHAPES[float] = (1,)
SHAPES[bool] = (1,)

VEC_TYPES: dict[TypeReflection.ScalarType, list[Optional[Type[TSGLVector]]]] = {
    x: [None]*5 for x in TypeReflection.ScalarType
}


def add_vec_types(scalar_type: list[TypeReflection.ScalarType], dim: int, vec_type: Type[TSGLVector]):
    for st in scalar_type:
        if VEC_TYPES[st][dim] is None:
            VEC_TYPES[st][dim] = vec_type


# Register matchers sgl types.
for sgl_pair in zip(["int", "float", "bool", "uint", "float16_t"], [SLANG_INT_TYPES, SLANG_FLOAT_TYPES, SLANG_BOOL_TYPES, SLANG_UINT_TYPES, SLANG_FLOAT_TYPES]):
    # The scalar (i.e. float1) types
    sgl_type: type = getattr(math, f"{sgl_pair[0]}1", None)  # type: ignore
    if sgl_type is not None:
        MATCHERS[sgl_type] = lambda slang_type, scalar_types=sgl_pair[1], dim=1: is_match_scalar(
            slang_type, scalar_types)
        SHAPES[sgl_type] = (1,)
        add_vec_types(sgl_pair[1], 1, sgl_type)

    # Vector (i.e. float2) types
    for dim in range(2, 5):
        sgl_type: type = getattr(math, f"{sgl_pair[0]}{dim}", None)  # type: ignore
        if sgl_type is not None:
            MATCHERS[sgl_type] = lambda slang_type, scalar_types=sgl_pair[1], dim=dim: is_match_vector(
                slang_type, scalar_types, dim)
            SHAPES[sgl_type] = (dim,)
            add_vec_types(sgl_pair[1], dim, sgl_type)

    # Quaternion type
    MATCHERS[quatf] = lambda slang_type, scalar_types=sgl_pair[1], dim=dim: is_match_vector(
        slang_type, scalar_types, 4)
    SHAPES[quatf] = (4,)

    # Matrix types (note: currently only floats, search for all in case we add them later)
    for row in range(2, 5):
        for col in range(2, 5):
            sgl_type: type = getattr(
                math, f"{sgl_pair[0]}{row}x{col}", None)  # type: ignore
            if sgl_type is not None:
                MATCHERS[sgl_type] = lambda slang_type, scalar_types=sgl_pair[1], rows=row, cols=col: is_match_matrix(
                    slang_type, scalar_types, rows, cols)
                SHAPES[sgl_type] = (row, col)

# Matcher for dict
MATCHERS[dict] = lambda slang_type: slang_type.kind in [
    TypeReflection.Kind.struct, TypeReflection.Kind.vector]


def calc_element_type_size(element_type: Union[Type[TSGLVector], Type[TPythonScalar], TypeReflection, TypeLayoutReflection]) -> int:
    if isinstance(element_type, TypeLayoutReflection):
        return element_type.size
    elif element_type in (int1, uint1, float1, bool1, int, float, bool):
        return 4
    elif element_type in (int2, uint2, float2, bool2):
        return 8
    elif element_type in (int3, uint3, float3, bool3):
        return 12
    elif element_type in (int4, uint4, float4, bool4):
        return 16
    raise ValueError(f"Unsupported type: {element_type}")


def are_element_types_compatible(
        element_type: Optional[Union[Type[TSGLVector], Type[TPythonScalar], TypeReflection, TypeLayoutReflection]],
        slang_type: TypeReflection
):
    """
    Checks if a core element type is compatible with a slang type
    """
    if element_type is NoneType:
        return True
    elif isinstance(element_type, TypeReflection):
        return element_type.full_name == slang_type.full_name
    elif isinstance(element_type, TypeLayoutReflection):
        return element_type.type.full_name == slang_type.full_name
    elif isinstance(element_type, type):
        matcher = MATCHERS.get(element_type, None)
        if matcher is not None:
            return matcher(slang_type)
    return False
