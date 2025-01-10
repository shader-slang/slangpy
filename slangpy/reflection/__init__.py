# SPDX-License-Identifier: Apache-2.0
# pyright: reportUnusedImport=false
# isort: skip_file

from .reflectiontypes import (
    SlangLayout,
    SlangType,
    VoidType,
    ScalarType,
    VectorType,
    MatrixType,
    ArrayType,
    StructType,
    InterfaceType,
    TextureType,
    StructuredBufferType,
    ByteAddressBufferType,
    DifferentialPairType,
    RaytracingAccelerationStructureType,
    SamplerStateType,
    UnhandledType,
    SlangFunction,
    SlangField,
    SlangParameter,
    SlangProgramLayout,
    TYPE_OVERRIDES,
    is_matching_array_type,
    SCALAR_TYPE_TO_NUMPY_TYPE
)

# Regularly needed for access to scalar type by slang type
from slangpy.backend import TypeReflection
