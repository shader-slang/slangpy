# Base class for marshalling python types
from typing import Any, Optional, Union
from numpy.typing import ArrayLike

from sgl import Buffer, Device, ResourceUsage, TypeLayoutReflection

from kernelfunctions.shapes import TLooseOrUndefinedShape
from kernelfunctions.typemappings import TPythonScalar, TSGLVector, calc_element_type_size

from .enums import AccessType


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

    def get_calldata_typename(self, typename: str, shape: TLooseOrUndefinedShape, access: AccessType):
        """
        Get the typename that should be written to call data for this type.    
        Default behaviour is to return the typename for read access and a buffer to
        contain the type for write access.
        """
        if access == AccessType.read:
            return typename
        else:
            return f"RWStructuredBuffer<{typename}>"

    def get_indexer(self, call_transform: list[Optional[int]], access: AccessType):
        """
        Get the index string that should be used to index into the buffer. Default
        behaviour is no index for read access, and access element 0 for write access.
        """
        if access == AccessType.read:
            return ""
        else:
            return "[0]"

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
