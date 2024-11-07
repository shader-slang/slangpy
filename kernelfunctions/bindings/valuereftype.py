

from typing import Any, Optional, Union
import numpy as np

from kernelfunctions.core import CodeGenBlock, BindContext, ReturnContext, BaseType, BaseTypeImpl, BoundVariable, AccessType, BoundVariableRuntime, CallContext, Shape

import kernelfunctions.core.reflection as kfr

from kernelfunctions.types import ValueRef

from kernelfunctions.backend import Buffer, ResourceUsage
from kernelfunctions.typeregistry import PYTHON_TYPES, get_or_create_type

import kernelfunctions.backend as kfbackend


def slang_type_to_return_type(slang_type: kfr.SlangType) -> Any:
    if isinstance(slang_type, kfr.ScalarType):
        if slang_type.slang_scalar_type in kfr.FLOAT_TYPES:
            return float
        elif slang_type.slang_scalar_type in kfr.INT_TYPES:
            return int
        elif slang_type.slang_scalar_type in kfr.BOOL_TYPES:
            return bool
    elif isinstance(slang_type, kfr.VectorType):
        if slang_type.slang_scalar_type in kfr.FLOAT_TYPES:
            return getattr(kfbackend, f'float{slang_type.num_elements}')
        elif slang_type.slang_scalar_type in kfr.SIGNED_INT_TYPES:
            return getattr(kfbackend, f'int{slang_type.num_elements}')
        elif slang_type.slang_scalar_type in kfr.UNSIGNED_INT_TYPES:
            return getattr(kfbackend, f'uint{slang_type.num_elements}')
        elif slang_type.slang_scalar_type in kfr.BOOL_TYPES:
            return getattr(kfbackend, f'bool{slang_type.num_elements}')
    elif isinstance(slang_type, kfr.MatrixType):
        if slang_type.slang_scalar_type in kfr.FLOAT_TYPES:
            return getattr(kfbackend, f'float{slang_type.rows}x{slang_type.cols}')
        elif slang_type.slang_scalar_type in kfr.SIGNED_INT_TYPES:
            return getattr(kfbackend, f'int{slang_type.rows}x{slang_type.cols}')
        elif slang_type.slang_scalar_type in kfr.UNSIGNED_INT_TYPES:
            return getattr(kfbackend, f'uint{slang_type.rows}x{slang_type.cols}')
        elif slang_type.slang_scalar_type in kfr.BOOL_TYPES:
            return getattr(kfbackend, f'bool{slang_type.rows}x{slang_type.cols}')
    else:
        raise ValueError(f"Slang type {slang_type} has no associated python value type")


def slang_value_to_numpy(slang_type: kfr.SlangType, value: Any) -> np.ndarray:
    if isinstance(slang_type, kfr.ScalarType):
        # value should be a basic python type (int/float/bool)
        return np.array([value], dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type])
    elif isinstance(slang_type, kfr.VectorType):
        # value should be one of the SGL vector types, which are iterable
        data = [value[i] for i in range(slang_type.num_elements)]
        return np.array(data, dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type])
    elif isinstance(slang_type, kfr.MatrixType):
        # value should be an SGL matrix type, which has a to_numpy function
        return value.to_numpy()
    else:
        raise ValueError(f"Can not convert slang type {slang_type} to numpy array")


def numpy_to_slang_value(slang_type: kfr.SlangType, value: np.ndarray) -> Any:
    python_type = slang_type_to_return_type(slang_type)
    if isinstance(slang_type, kfr.ScalarType):
        # convert first element of numpy array to basic python type
        np_data = value.view(
            dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type])
        return python_type(value[0])
    elif isinstance(slang_type, kfr.VectorType):
        # convert to one of the SGL vector types (can be constructed from sequence)
        np_data = value.view(
            dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type])
        return python_type(*np_data)
    elif isinstance(slang_type, kfr.MatrixType):
        # convert to one of the SGL matrix types (can be constructed from numpy array)
        np_data = value.view(
            dtype=kfr.SCALAR_TYPE_TO_NUMPY_TYPE[slang_type.slang_scalar_type])
        return python_type(np_data)
    else:
        raise ValueError(f"Can not convert numpy array to slang type {slang_type}")


class ValueRefType(BaseTypeImpl):

    def __init__(self, value_type: Union[BaseType, kfr.SlangType]):
        super().__init__()
        self.value_type = value_type
        self.element_type = self.value_type.element_type
        self.name = self.value_type.name

    # Values don't store a derivative - they're just a value
    @property
    def has_derivative(self) -> bool:
        return False

    # Refs can be written to!
    @property
    def is_writable(self) -> bool:
        return True

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", f"ValueRef<{self.value_type.name}>")
        else:
            cgb.type_alias(
                f"_t_{name}", f"RWValueRef<{self.value_type.name}>")

    # Call data just returns the primal

    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: ValueRef) -> Any:
        access = binding.access
        assert access[0] != AccessType.none
        assert access[1] == AccessType.none
        if access[0] == AccessType.read:
            return {'value': data.value}
        else:
            if isinstance(self.value_type, kfr.SlangType):
                npdata = slang_value_to_numpy(self.value_type, data.value)
            else:
                npdata = self.value_type.to_numpy(data.value)
            npdata = npdata.view(dtype=np.uint8)
            return {
                'value': context.device.create_buffer(element_count=1, struct_size=npdata.size, data=npdata, usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)
            }

    # Read back from call data does nothing
    def read_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: ValueRef, result: Any) -> None:
        access = binding.access
        if access[0] in [AccessType.write, AccessType.readwrite]:
            assert isinstance(result['value'], Buffer)
            npdata = result['value'].to_numpy()
            if isinstance(self.value_type, kfr.SlangType):
                data.value = numpy_to_slang_value(self.value_type, npdata)
            else:
                data.value = self.value_type.from_numpy(npdata)

    def get_shape(self, value: Optional[ValueRef] = None) -> Shape:
        if isinstance(self.value_type, kfr.SlangType):
            return self.value_type.shape
        else:
            return self.value_type.get_shape()

    @property
    def differentiable(self):
        return self.value_type.differentiable

    @property
    def derivative(self):
        return self.value_type.derivative

    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        if isinstance(self.value_type, kfr.SlangType):
            pt = slang_type_to_return_type(self.value_type)
        else:
            pt = self.value_type.python_return_value_type
        if pt is not None:
            return ValueRef(pt())
        else:
            return ValueRef(None)

    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: ValueRef) -> Any:
        return data.value


def create_vr_type_for_value(value: Any):
    if isinstance(value, ValueRef):
        return ValueRefType(get_or_create_type(type(value.value)))
    elif isinstance(value, ReturnContext):
        return ValueRefType(value.slang_type)


PYTHON_TYPES[ValueRef] = create_vr_type_for_value
