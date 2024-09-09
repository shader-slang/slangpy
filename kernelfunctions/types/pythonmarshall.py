# Base class for marshalling python types
from typing import Any, Optional, Union
from numpy.typing import ArrayLike

import kernelfunctions.codegen as cg

from kernelfunctions.backend import Buffer, Device, ResourceUsage, TypeLayoutReflection
from kernelfunctions.shapes import TLooseOrUndefinedShape
from kernelfunctions.typemappings import TPythonScalar, TSGLVector, calc_element_type_size
from kernelfunctions.types.enums import AccessType, PrimType


class PythonDescriptor:
    def __init__(self,
                 python_type: type,
                 element_shape: TLooseOrUndefinedShape,
                 container_shape: TLooseOrUndefinedShape,
                 element_type: Optional[Union[type[TSGLVector], type[TPythonScalar], TypeLayoutReflection]],
                 writable: bool,
                 differentiable: bool):
        super().__init__()
        self.type = python_type
        self.element_shape = element_shape
        self.container_shape = container_shape
        self.element_type = element_type
        self.writable = writable
        self.differentiable = differentiable

        # Calculate combined element and container shape
        python_shape = ()
        if self.container_shape is not None:
            python_shape += self.container_shape
        if self.element_shape is not None:
            python_shape += self.element_shape
        self.shape = python_shape if len(python_shape) > 0 else None


class PythonMarshal:
    def __init__(self, python_type: type):
        super().__init__()
        self.type = python_type

    def get_descriptor(self, value: Any) -> PythonDescriptor:
        """
        Return a descriptor for the given value. This is used to store information about the value
        that is used in code generation. This can normally be left alone, as it is populated
        using the individual accessors below.
        """
        return PythonDescriptor(
            python_type=type(value),
            element_shape=self.get_element_shape(value),
            container_shape=self.get_container_shape(value),
            element_type=self.get_element_type(value),
            writable=self.is_writable(value),
            differentiable=self.is_differentiable(value),
        )

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

    def gen_calldata(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, type_name: str, variable_name: str, access: AccessType):
        """
        Declare the call data for this value. By default, read only values are stored as uniforms, and read-write
        values are stored as structured buffers with a single element.
        """
        if access == AccessType.read:
            cgb.declare(type_name, variable_name)
        else:
            cgb.declare(f"RWStructuredBuffer<{type_name}>", variable_name)

    def gen_load(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, from_call_data: str, to_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Generate code to load the value from the call data. By default, read only values are simply copied
        from the uniform, and read-write values are copied from the first element of the structured buffer.
        """
        if access == AccessType.read:
            cgb.assign(to_variable, from_call_data)
        else:
            cgb.assign(to_variable, f"{from_call_data}[0]")

    def gen_store(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, to_call_data: str, from_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Generate code to store the value to the call data. By default, this assumes a writable value
        has a single element structured buffer to write to.
        """
        cgb.assign(f"{to_call_data}[0]", from_variable)

    def to_numpy(self, value: Any, prim: PrimType) -> ArrayLike:
        """
        Convert value to a numpy array. Required for writable values that
        don't override the create_ and read_ methods.
        """
        raise NotImplementedError()

    def from_numpy(self, data: ArrayLike, prim: PrimType) -> None:
        """
        Convert value from a numpy array. Required for writable values that
        don't override the create_ and read_ methods.
        """
        raise NotImplementedError()

    def to_buffer(self, device: Device, value: Any, prim: PrimType) -> Any:
        """
        Allocate a buffer for the value and use from_numpy to fill it.
        """
        assert prim == PrimType.primal
        buffer = device.create_buffer(
            element_count=1,
            struct_size=calc_element_type_size(
                self.get_element_type(value)),  # type: ignore
            usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)
        buffer.from_numpy(self.to_numpy(value, prim))
        return buffer

    def from_buffer(self, device: Device, buffer: Buffer, prim: PrimType) -> Any:
        assert prim == PrimType.primal
        numpy_value = buffer.to_numpy()
        primal = self.from_numpy(numpy_value, PrimType.primal)
        return primal

    def create_calldata(self, device: Device, value: Any, access: AccessType, prim: PrimType):
        """
        Return entry in call data for the value. Default behaviour is to return
        the value for read-mode, and a buffer filled from numpy array for write-mode.
        """
        if prim == PrimType.primal:
            if access == AccessType.read:
                return value
            else:
                return self.to_buffer(device, value, prim)
        else:
            raise NotImplementedError()

    def read_calldata(self, device: Device, call_data: Any, access: AccessType, prim: PrimType, value: Any):
        """
        Read back entry in call data for the value. Must be implemented for writable
        types.
        """
        raise ValueError("Cannot read back value for non-writable type")

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
