# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any

from slangpy.bindings.boundvariable import BoundVariable
from slangpy.bindings.codegen import CodeGenBlock
from slangpy.bindings.marshall import BindContext, ReturnContext
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES
from slangpy.core.native import NativeNumpyMarshall
from slangpy.builtin.ndbuffer import (
    ndbuffer_gen_calldata,
    ndbuffer_reduce_type,
    ndbuffer_resolve_dimensionality,
    ndbuffer_resolve_types,
)

import numpy as np
import numpy.typing as npt

from slangpy.reflection.reflectiontypes import (
    NUMPY_TYPE_TO_SCALAR_TYPE,
    SCALAR_TYPE_TO_NUMPY_TYPE,
    SlangProgramLayout,
    ScalarType,
    SlangType,
    VectorType,
    MatrixType,
)


class NumpyMarshall(NativeNumpyMarshall):

    def __init__(
        self,
        layout: SlangProgramLayout,
        dtype: np.dtype[Any],
        dims: int,
        writable: bool,
    ):
        slang_el_type = layout.scalar_type(NUMPY_TYPE_TO_SCALAR_TYPE[dtype])
        assert slang_el_type is not None

        slang_el_layout = slang_el_type.buffer_layout

        slang_buffer_type = layout.find_type_by_name(f"RWTensor<{slang_el_type.full_name},{dims}>")
        assert slang_buffer_type is not None

        super().__init__(dims, slang_buffer_type, slang_el_type, slang_el_layout.reflection, dtype)

    @property
    def has_derivative(self) -> bool:
        return False

    @property
    def is_writable(self) -> bool:
        return True

    def reduce_type(self, context: BindContext, dimensions: int):
        return ndbuffer_reduce_type(self, context, dimensions)

    def resolve_types(self, context: BindContext, bound_type: "SlangType"):
        return ndbuffer_resolve_types(self, context, bound_type)

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "SlangType",
    ):
        return ndbuffer_resolve_dimensionality(self, context, binding, vector_target_type)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: "BoundVariable"):
        return ndbuffer_gen_calldata(self, cgb, context, binding)


def create_vr_type_for_value(layout: SlangProgramLayout, value: Any):
    if isinstance(value, np.ndarray):
        return NumpyMarshall(layout, value.dtype, value.ndim, True)
    elif isinstance(value, ReturnContext):
        if isinstance(value.slang_type, (ScalarType, VectorType, MatrixType)):
            scalar_type = value.slang_type.slang_scalar_type
            dtype = np.dtype(SCALAR_TYPE_TO_NUMPY_TYPE[scalar_type])
            return NumpyMarshall(
                layout,
                dtype,
                value.bind_context.call_dimensionality + value.slang_type.num_dims,
                True,
            )
        else:
            raise ValueError(
                f"Numpy values can only be automatically returned from scalar, vector or matrix types. Got {value.slang_type}"
            )
    else:
        raise ValueError(f"Unexpected type {type(value)} attempting to create numpy marshall")


PYTHON_TYPES[np.ndarray] = create_vr_type_for_value


def hash_numpy(value: npt.NDArray[Any]) -> str:
    return f"numpy.ndarray[{value.dtype},{value.ndim}]"


PYTHON_SIGNATURES[np.ndarray] = hash_numpy
