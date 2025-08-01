# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from slangpy import BufferCursor

from slangpy.core.native import AccessType, CallContext

import slangpy.reflection as kfr
from slangpy import Buffer, BufferUsage
from slangpy.bindings import (
    PYTHON_TYPES,
    Marshall,
    BindContext,
    BoundVariable,
    BoundVariableRuntime,
    CodeGenBlock,
    ReturnContext,
    get_or_create_type,
)
from slangpy.builtin.value import slang_type_to_return_type
from slangpy.reflection.reflectiontypes import SlangType
from slangpy.types import ValueRef


# The input value for this function is not aligned, because it's user provided data.
# So we need to take care of the alignment of the matrix here. When the matrix size is aligned,
# we can just return the numpy array. Otherwise, we need to padding each row to the alignment.
# Calculate the number of columns that is aligned to the buffer layout alignment
# slang_type.buffer_layout.alignment is the alignment in bytes of each row of the matrix
def slang_matrix_to_numpy_with_padding(slang_type: kfr.MatrixType, value: Any) -> npt.NDArray[Any]:
    aligned_row_num = (
        slang_type.buffer_layout.stride // slang_type.rows
    ) // slang_type.scalar_type.buffer_layout.stride

    if aligned_row_num == slang_type.cols:
        return value.to_numpy()
    else:
        mat_aligned = np.zeros(
            (slang_type.rows, aligned_row_num),
            dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type],
        )
        mat_aligned[:, : slang_type.cols] = value.to_numpy()
        return mat_aligned


# The input value for this function is aligned, because it's the output of the kernel.
# We need to remove the padding of the matrix here before returning to user code.
def numpy_to_slang_matrix_remove_padding(
    slang_type: kfr.MatrixType, value: npt.NDArray[Any], python_type: Any
) -> Any:
    np_data = value.view(dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type])
    aligned_row_num = (
        slang_type.buffer_layout.stride // slang_type.rows
    ) // slang_type.scalar_type.buffer_layout.stride
    if aligned_row_num == slang_type.cols:
        return python_type(np_data)
    else:
        mat_aligned = np_data.reshape((slang_type.rows, aligned_row_num))
        mat_remove_padding = np.zeros(
            (slang_type.rows, slang_type.cols),
            dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type],
        )
        mat_remove_padding = mat_aligned[:, : slang_type.cols]
        return python_type(mat_remove_padding)


def slang_value_to_numpy(slang_type: kfr.SlangType, value: Any) -> npt.NDArray[Any]:
    if isinstance(slang_type, kfr.ScalarType):
        # value should be a basic python type (int/float/bool)
        return np.array([value], dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type])
    elif isinstance(slang_type, kfr.VectorType):
        # value should be one of the SGL vector types, which are iterable
        data = [value[i] for i in range(slang_type.num_elements)]
        return np.array(data, dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type])
    elif isinstance(slang_type, kfr.MatrixType):
        # value should be an SGL matrix type, which has a to_numpy function
        return slang_matrix_to_numpy_with_padding(slang_type, value)
    elif isinstance(slang_type, kfr.PointerType):
        return np.array(
            [value], dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[kfr.TypeReflection.ScalarType.uint64]
        )
    else:
        raise ValueError(f"Can not convert slang type {slang_type} to numpy array")


def numpy_to_slang_value(slang_type: kfr.SlangType, value: npt.NDArray[Any]) -> Any:
    python_type = slang_type_to_return_type(slang_type)
    if isinstance(slang_type, kfr.ScalarType):
        # convert first element of numpy array to basic python type
        np_data = value.view(dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type])
        return python_type(np_data[0])
    elif isinstance(slang_type, kfr.VectorType):
        # convert to one of the SGL vector types (can be constructed from sequence)
        np_data = value.view(dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type])
        return python_type(*np_data)
    elif isinstance(slang_type, kfr.MatrixType):
        # convert to one of the SGL matrix types (can be constructed from numpy array)
        return numpy_to_slang_matrix_remove_padding(slang_type, value, python_type)
    elif isinstance(slang_type, kfr.PointerType):
        # Pointers are represented as uint64_t in slang, so we return int
        np_data = value.view(dtype=np.uint64)
        return python_type(np_data[0])
    else:
        raise ValueError(f"Can not convert numpy array to slang type {slang_type}")


class ValueRefMarshall(Marshall):

    def __init__(self, layout: kfr.SlangProgramLayout, value_type: kfr.SlangType):
        super().__init__(layout)
        self.value_type = value_type

        st = layout.find_type_by_name(f"ValueRef<{value_type.full_name}>")
        if st is None:
            raise ValueError(
                f"Could not find ValueRef<{value_type.full_name}> slang type. This usually indicates the slangpy module has not been imported."
            )
        self.slang_type = st
        assert value_type.shape.concrete
        self.concrete_shape = value_type.shape

    # Values don't store a derivative - they're just a value
    @property
    def has_derivative(self) -> bool:
        return False

    # Refs can be written to!
    @property
    def is_writable(self) -> bool:
        return True

    def resolve_type(self, context: BindContext, bound_type: "kfr.SlangType"):
        if self.value_type.name != "Unknown":
            return self.value_type
        else:
            return bound_type

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "kfr.SlangType",
    ):
        return len(self.value_type.shape) - len(vector_target_type.shape)

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: "BoundVariable"):
        access = binding.access
        name = binding.variable_name
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        assert binding.vector_type is not None
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", f"ValueRef<{binding.vector_type.full_name}>")
        else:
            cgb.type_alias(f"_t_{name}", f"RWValueRef<{binding.vector_type.full_name}>")

    # Call data just returns the primal
    def create_calldata(
        self, context: CallContext, binding: "BoundVariableRuntime", data: ValueRef
    ) -> Any:
        access = binding.access
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if access[0] == AccessType.read:
            return {"value": data.value}
        else:
            if isinstance(binding.vector_type, kfr.StructType):
                buffer = context.device.create_buffer(
                    element_count=1,
                    struct_size=binding.vector_type.buffer_layout.stride,
                    usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
                )
                cursor = BufferCursor(binding.vector_type.buffer_layout.reflection, buffer, False)
                cursor[0].write(data.value)
                cursor.apply()
                return {"value": buffer}
            else:
                if isinstance(self.value_type, kfr.SlangType):
                    npdata = slang_value_to_numpy(self.value_type, data.value)
                else:
                    npdata = self.value_type.to_numpy(data.value)
                npdata = npdata.view(dtype=np.uint8)
                return {
                    "value": context.device.create_buffer(
                        element_count=1,
                        struct_size=npdata.size,
                        data=npdata,
                        usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
                    )
                }

    # Value ref just passes its value for raw dispatch
    def create_dispatchdata(self, data: Any) -> Any:
        return data

    # Read back from call data
    def read_calldata(
        self,
        context: CallContext,
        binding: "BoundVariableRuntime",
        data: ValueRef,
        result: Any,
    ) -> None:
        access = binding.access
        if access[0] in [AccessType.write, AccessType.readwrite]:
            assert isinstance(result["value"], Buffer)
            if isinstance(binding.vector_type, kfr.StructType):
                cursor = BufferCursor(binding.vector_type.buffer_layout.reflection, result["value"])
                data.value = cursor[0].read()
            else:
                npdata = result["value"].to_numpy()
                if isinstance(self.value_type, kfr.SlangType):
                    data.value = numpy_to_slang_value(self.value_type, npdata)
                else:
                    data.value = self.value_type.copy_from_numpy(npdata)

    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        pt = slang_type_to_return_type(self.value_type)
        if pt is not None:
            return ValueRef(pt())
        else:
            return ValueRef(None)

    def read_output(
        self, context: CallContext, binding: BoundVariableRuntime, data: ValueRef
    ) -> Any:
        return data.value


def create_vr_type_for_value(layout: kfr.SlangProgramLayout, value: Any):
    if isinstance(value, ValueRef):
        return ValueRefMarshall(
            layout,
            cast(
                SlangType,
                get_or_create_type(layout, type(value.value), value.value).slang_type,
            ),
        )
    elif isinstance(value, ReturnContext):
        return ValueRefMarshall(layout, value.slang_type)
    else:
        raise ValueError(f"Unsupported value type {type(value)}")


PYTHON_TYPES[ValueRef] = create_vr_type_for_value
