# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from slangpy import TypeReflection
from slangpy import TypeReflection as TR
from slangpy.core.native import Shape
from slangpy.native_refl import (
    ArrayType,
    ByteAddressBufferType,
    DiffTensorViewType,
    DifferentialPairType,
    Field as SlangField,
    Function as SlangFunction,
    InterfaceType,
    Layout as SlangProgramLayout,
    MatrixType,
    Parameter as SlangParameter,
    PointerType,
    RaytracingAccelerationStructureType,
    ResourceType,
    SamplerStateType,
    ScalarType,
    StructuredBufferType,
    StructType,
    TensorType as ITensorType,
    TensorViewType,
    TextureType,
    Type as SlangType,
    TypeLayout as SlangLayout,
    UnhandledType,
    UnknownType,
    VectorType,
    VoidType,
    is_known,
    is_known_or_none,
    is_unknown,
)

TensorType = ITensorType.Kind
TensorAccess = ITensorType.Access

EXPERIMENTAL_VECTORIZATION = False


def set_experimental_vectorization(enabled: bool) -> None:
    global EXPERIMENTAL_VECTORIZATION
    EXPERIMENTAL_VECTORIZATION = enabled


scalar_names = {
    TR.ScalarType.none: "Unknown",
    TR.ScalarType.void: "void",
    TR.ScalarType.bool: "bool",
    TR.ScalarType.int8: "int8_t",
    TR.ScalarType.int16: "int16_t",
    TR.ScalarType.int32: "int",
    TR.ScalarType.int64: "int64_t",
    TR.ScalarType.uint8: "uint8_t",
    TR.ScalarType.uint16: "uint16_t",
    TR.ScalarType.uint32: "uint",
    TR.ScalarType.uint64: "uint64_t",
    TR.ScalarType.float16: "half",
    TR.ScalarType.float32: "float",
    TR.ScalarType.float64: "double",
}

SIGNED_INT_TYPES = {
    TR.ScalarType.int8,
    TR.ScalarType.int16,
    TR.ScalarType.int32,
    TR.ScalarType.int64,
}
UNSIGNED_INT_TYPES = {
    TR.ScalarType.uint8,
    TR.ScalarType.uint16,
    TR.ScalarType.uint32,
    TR.ScalarType.uint64,
}
FLOAT_TYPES = {TR.ScalarType.float16, TR.ScalarType.float32, TR.ScalarType.float64}
BOOL_TYPES = {TR.ScalarType.bool}
INT_TYPES = SIGNED_INT_TYPES | UNSIGNED_INT_TYPES

SCALAR_TYPE_TO_NUMPY_TYPE = {
    TR.ScalarType.int8: np.int8,
    TR.ScalarType.int16: np.int16,
    TR.ScalarType.int32: np.int32,
    TR.ScalarType.int64: np.int64,
    TR.ScalarType.uint8: np.uint8,
    TR.ScalarType.uint16: np.uint16,
    TR.ScalarType.uint32: np.uint32,
    TR.ScalarType.uint64: np.uint64,
    TR.ScalarType.float16: np.float16,
    TR.ScalarType.float32: np.float32,
    TR.ScalarType.float64: np.float64,
    TR.ScalarType.bool: np.int8,
}
NUMPY_TYPE_TO_SCALAR_TYPE = {np.dtype(v): k for k, v in SCALAR_TYPE_TO_NUMPY_TYPE.items()}


def is_matching_array_type(a: SlangType, b: SlangType, allow_generics: bool = True) -> bool:
    if not isinstance(a, ArrayType) or not isinstance(b, ArrayType):
        return False
    if allow_generics:
        if a.any_generic_dims or b.any_generic_dims:
            return True
        if isinstance(a.inner_element_type, UnknownType) or isinstance(
            b.inner_element_type, UnknownType
        ):
            return True
    return (
        a.array_shape == b.array_shape
        and a.inner_element_type.full_name == b.inner_element_type.full_name
    )


def vectorize_type(
    marshall_type: Union[SlangType, str],
    bound_type: Union[SlangType, str],
    program: Optional[SlangProgramLayout] = None,
) -> Optional[SlangType]:
    if not EXPERIMENTAL_VECTORIZATION:
        raise RuntimeError("vectorize_type is an experimental feature and is disabled")

    if program is None:
        if isinstance(marshall_type, SlangType):
            program = marshall_type.program
        elif isinstance(bound_type, SlangType):
            program = bound_type.program
    if isinstance(marshall_type, SlangType):
        marshall_type = marshall_type.full_name
    if isinstance(bound_type, SlangType):
        bound_type = bound_type.vector_type_name
    if program is None:
        raise ValueError("Program must be provided or inferable from from_type or to_type")

    witness = program.find_type_by_name(f"Vectorizer<{marshall_type}, {bound_type}>.VectorType")
    if witness is None or witness.is_generic:
        return None
    return witness
