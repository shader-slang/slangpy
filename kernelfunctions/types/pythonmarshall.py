# Base class for marshalling python types
from typing import Any, Optional, Union
from numpy.typing import ArrayLike

from sgl import Buffer, Device, ResourceUsage, TypeLayoutReflection

from kernelfunctions.shapes import TLooseOrUndefinedShape
from kernelfunctions.typemappings import TPythonScalar, TSGLVector, calc_element_type_size

from .enums import AccessType

import kernelfunctions.codegen as cg


class PythonMarshal:
    def __init__(self, python_type: type):
        super().__init__()
        self.type = python_type

    def get_element_shape(self, value: Any) -> TLooseOrUndefinedShape:
        """
        Returns the shape of the elements of the given value. For a none-container,
        this should just be the shape of the type.
        """
        return ()

    def get_container_shape(self, value: Any) -> TLooseOrUndefinedShape:
        """
        Returns the shape of the container. For a none-container, this should be an empty tuple.
        """
        return ()

    def get_element_type(self, value: Any) -> Optional[Union[type[TSGLVector], type[TPythonScalar], TypeLayoutReflection]]:
        """
        Returns the type of the elements of the given value. For a none-container,
        this should just be the type of the value.
        """
        return type(value)

    def is_writable(self, value: Any) -> bool:
        """
        Whether this value can be written to.
        """
        return False

    def is_differentiable(self, value: Any) -> bool:
        """
        Whether this value is differentiable.
        """
        return False

    def gen_calldata(self, slang_type_name: str, call_data_name: str, shape: TLooseOrUndefinedShape, access: AccessType):
        """
        Declare the call data for this value. By default, read only values are stored as uniforms, and read-write
        values are stored as structured buffers with a single element.
        """
        if access == AccessType.read:
            return cg.declare(slang_type_name, call_data_name)
        else:
            return cg.declare(f"RWStructuredBuffer<{slang_type_name}>", call_data_name)

    def gen_load(self, from_call_data: str, to_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Generate code to load the value from the call data. By default, read only values are simply copied
        from the uniform, and read-write values are copied from the first element of the structured buffer.
        """
        if access == AccessType.read:
            return cg.assign(to_variable, from_call_data)
        else:
            return cg.assign(to_variable, f"{from_call_data}[0]")

    def gen_store(self, to_call_data: str, from_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Generate code to store the value to the call data. By default, this assumes a writable value
        has a single element structured buffer to write to.
        """
        return cg.assign(f"{to_call_data}[0]", from_variable)

    def primal_to_numpy(self, value: Any):
        """
        Convert the primal value to a numpy array. Required for writable values that
        don't override the create_primal and read_primal methods.
        """
        raise NotImplementedError()

    def primal_from_numpy(self, data: ArrayLike) -> None:
        """
        Convert the primal value from a numpy array. Required for writable values that
        don't override the create_primal and read_primal methods.
        """
        raise NotImplementedError()

    def derivative_to_numpy(self, value: Any):
        """
        Convert the derivative value to a numpy array. Required for writable values that
        don't override the create_derivative and read_derivative methods.
        """
        raise NotImplementedError()

    def derivative_from_numpy(self, data: ArrayLike) -> None:
        """
        Convert the derivative value from a numpy array. Required for writable values that
        don't override the create_derivative and read_derivative methods.
        """
        raise NotImplementedError()

    def create_primal_calldata(self, device: Device, value: Any, access: AccessType):
        """
        Return entry in call data for the primal value. Default behaviour is to return
        the value for read-mode, and a buffer filled from numpy array for write-mode.
        """
        if access == AccessType.read:
            return value
        else:
            buffer = device.create_buffer(
                element_count=1,
                struct_size=calc_element_type_size(self.get_element_type(value)),
                usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)
            buffer.from_numpy(self.primal_to_numpy(value))
            return buffer

    def create_derivative_calldata(self, device: Device, value: Any, access: AccessType):
        """
        Return entry in call data for the derivative value. Default behaviour is to return
        the value for read-mode, and a buffer filled from numpy array for write-mode.
        """
        if access == AccessType.read:
            return value
        else:
            buffer = device.create_buffer(
                element_count=1,
                struct_size=calc_element_type_size(self.get_element_type(value)),
                usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)
            buffer.from_numpy(self.derivative_to_numpy(value))
            return buffer

    def read_primal_calldata(self, device: Device, call_data: Any, access: AccessType, value: Any):
        """
        Read back entry in call data for the primal value. Must be implemented for writable
        types.
        """
        raise ValueError("Cannot read back primal value for non-writable type")

    def read_derivative_calldata(self, device: Device, call_data: Any, access: AccessType, value: Any):
        """
        Read back entry in call data for the derivative value. Must be implemented for writable
        differentiable types.
        """
        raise ValueError("Cannot read back derivative value for non-writable type")

    def allocate_return_value(self, device: Device, call_shape: list[int], element_type: Any):
        """
        Allocate a return value for this type. Only required for types that can be directly
        allocated and returned from function calls.
        """
        raise NotImplementedError()

    def as_return_value(self, value: Any):
        """
        Convert the allocated return value into the value returned to the user when calling a kernel
        function. Default behaviour is just to return the value.
        """
        return value

    @property
    def name(self):
        return self.type.__name__

    def __repr__(self):
        return self.type.__name__
