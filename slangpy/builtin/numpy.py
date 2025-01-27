# SPDX-License-Identifier: Apache-2.0
from typing import Any

from slangpy.bindings.boundvariable import BoundVariable
from slangpy.bindings.codegen import CodeGenBlock
from slangpy.bindings.marshall import BindContext, ReturnContext
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES
from slangpy.core.native import NativeNumpyMarshall
from slangpy.builtin.ndbuffer import ndbuffer_gen_calldata, ndbuffer_reduce_type, ndbuffer_resolve_dimensionality, ndbuffer_resolve_type

import numpy as np
import numpy.typing as npt

from slangpy.reflection.reflectiontypes import NUMPY_TYPE_TO_SCALAR_TYPE, SCALAR_TYPE_TO_NUMPY_TYPE, SlangProgramLayout, ScalarType, SlangType, VectorType, MatrixType


class NumpyMarshall(NativeNumpyMarshall):

    def __init__(self, layout: SlangProgramLayout, dtype: np.dtype[Any], dims: int, writable: bool):
        slang_el_type = layout.scalar_type(NUMPY_TYPE_TO_SCALAR_TYPE[dtype])
        assert slang_el_type is not None

        slang_el_layout = slang_el_type.buffer_layout

        slang_buffer_type = layout.find_type_by_name(
            f"RWNDBuffer<{slang_el_type.full_name},{dims}>")
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

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        return ndbuffer_resolve_type(self, context, bound_type)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return ndbuffer_resolve_dimensionality(self, context, binding, vector_target_type)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        return ndbuffer_gen_calldata(self, cgb, context, binding)


"""
    def get_shape(self, value: Optional[npt.NDArray[Any]] = None) -> Shape:
        if value is not None:
            return Shape(value.shape)+self.slang_element_type.shape
        else:
            return Shape((-1,)*self.dims)+self.slang_element_type.shape

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: npt.NDArray[Any]) -> Any:
        shape = Shape(data.shape)
        vec_shape = binding.vector_type.shape.as_tuple()
        if len(vec_shape) > 0:
            el_shape = shape.as_tuple()[-len(vec_shape):]
            if el_shape != vec_shape:
                raise ValueError(
                    f"{binding.variable_name}: Element shape mismatch: val={el_shape}, expected={vec_shape}")

        buffer = NDBuffer(context.device, dtype=self.slang_element_type, shape=shape)
        buffer.from_numpy(data)
        return super().create_calldata(context, binding, buffer)

    def read_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: npt.NDArray[Any], result: Any) -> None:
        access = binding.access
        if access[0] in [AccessType.write, AccessType.readwrite]:
            assert isinstance(result['buffer'], Buffer)
            data[:] = result['buffer'].to_numpy().view(data.dtype).reshape(data.shape)
            pass

    def create_dispatchdata(self, data: NDBuffer) -> Any:
        raise ValueError("Numpy values do not support direct dispatch")

    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        shape = context.call_shape + binding.vector_type.shape
        return np.empty(shape.as_tuple(), dtype=self.dtype)

    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: npt.NDArray[Any]) -> Any:
        return data
"""


def create_vr_type_for_value(layout: SlangProgramLayout, value: Any):
    if isinstance(value, np.ndarray):
        return NumpyMarshall(layout, value.dtype, value.ndim, True)
    elif isinstance(value, ReturnContext):
        if isinstance(value.slang_type, (ScalarType, VectorType, MatrixType)):
            scalar_type = value.slang_type.slang_scalar_type
            dtype = np.dtype(SCALAR_TYPE_TO_NUMPY_TYPE[scalar_type])
            return NumpyMarshall(layout, dtype, value.bind_context.call_dimensionality + value.slang_type.num_dims, True)
        else:
            raise ValueError(
                f"Numpy values can only be automatically returned from scalar, vector or matrix types. Got {value.slang_type}")
    else:
        raise ValueError(
            f"Unexpected type {type(value)} attempting to create NDBuffer marshall")


PYTHON_TYPES[np.ndarray] = create_vr_type_for_value


def hash_numpy(value: npt.NDArray[Any]) -> str:
    return f"numpy.ndarray[{value.dtype},{value.ndim}]"


PYTHON_SIGNATURES[np.ndarray] = hash_numpy
