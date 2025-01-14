# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

from slangpy.backend import Buffer
from slangpy.backend.slangpynativeemulation import AccessType, CallContext
from slangpy.bindings.boundvariableruntime import BoundVariableRuntime
from slangpy.bindings.marshall import ReturnContext
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES
from slangpy.core.native import Shape
from slangpy.builtin.ndbuffer import NDBufferMarshall

import numpy as np
import numpy.typing as npt

from slangpy.reflection.reflectiontypes import NUMPY_TYPE_TO_SCALAR_TYPE, SCALAR_TYPE_TO_NUMPY_TYPE, SlangProgramLayout, ScalarType, VectorType, MatrixType
from slangpy.types.buffer import NDBuffer


class NumpyMarshall(NDBufferMarshall):
    def __init__(self, layout: SlangProgramLayout, dtype: np.dtype[Any], dims: int, writable: bool):
        scalar_type = layout.scalar_type(NUMPY_TYPE_TO_SCALAR_TYPE[dtype])
        super().__init__(layout, scalar_type, dims, writable)
        self.dtype = dtype

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
