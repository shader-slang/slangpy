from typing import Any, Optional
from sgl import Device, TypeLayoutReflection
from kernelfunctions.buffer import StructuredBuffer
from kernelfunctions.shapes import TConcreteShape
from kernelfunctions.typeregistry import create_slang_type_marshal, get_python_type_marshall, register_python_type
from kernelfunctions.types import AccessType, PythonMarshal
import kernelfunctions.codegen as cg


class BufferMarshall(PythonMarshal):
    """
    Marshall for scalar ref (will be 1 per scalar element type)
    """

    def __init__(self):
        super().__init__(StructuredBuffer)

    def get_element_shape(self, value: StructuredBuffer):
        if isinstance(value.element_type, TypeLayoutReflection):
            slang_marshall = create_slang_type_marshal(value.element_type.type)
            element_shape = slang_marshall.value_shape
        else:
            python_marshall = get_python_type_marshall(value.element_type)
            element_shape = python_marshall.get_element_shape(None)
        assert element_shape
        assert not None in element_shape
        return element_shape

    def get_container_shape(self, value: StructuredBuffer):
        return value.shape

    def get_element_type(self, value: StructuredBuffer):
        return value.element_type

    def is_differentiable(self, value: StructuredBuffer) -> bool:
        return value.is_differentiable

    def is_writable(self, value: StructuredBuffer) -> bool:
        return value.is_writable

    def gen_calldata(self, slang_type_name: str, call_data_name: str, shape: TConcreteShape, access: AccessType):
        """
        Call data either contains a read-only or read-write buffer.
        """
        if access == AccessType.read:
            return cg.declare(f"TensorBuffer<{slang_type_name},{len(shape)}>", call_data_name)
        else:
            return cg.declare(f"RWTensorBuffer<{slang_type_name},{len(shape)}>", call_data_name)

    def _transform_to_subscript(self, transform: list[Optional[int]]):
        """
        Generates the subscript to be passed into the [] operator when loading or storing
        from the buffer.
        """
        vals = ",".join(
            ("0" if x is None else f"context.call_id[{x}]") for x in transform)
        return f"[{{{vals}}}]"

    def gen_load(self, from_call_data: str, to_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Load the value from the buffer into the variable.
        """
        return cg.assign(to_variable, f"{from_call_data}{self._transform_to_subscript(transform)}")

    def gen_store(self, to_call_data: str, from_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Store the value from the variable into the buffer.
        """
        return cg.assign(f"{to_call_data}{self._transform_to_subscript(transform)}", from_variable)

    def create_primal_calldata(self, device: Device, value: StructuredBuffer, access: AccessType):
        return {
            "buffer": value.buffer,
            "strides": list(value.strides),
        }

    def create_derivative_calldata(self, device: Device, value: StructuredBuffer, access: AccessType):
        return {
            "buffer": value.grad_buffer,
            "strides": list(value.strides),
        }

    def read_primal_calldata(self, device: Device, call_data: Any, access: AccessType, value: StructuredBuffer):
        assert call_data['buffer'] == value.buffer
        assert call_data['strides'] == list(value.strides)

    def read_derivative_calldata(self, device: Device, call_data: Any, access: AccessType, value: StructuredBuffer):
        assert call_data['buffer'] == value.grad_buffer
        assert call_data['strides'] == list(value.strides)

    def allocate_return_value(self, device: Device, call_shape: list[int], element_type: Any):
        return StructuredBuffer(
            device=device,
            shape=tuple(call_shape),
            element_type=element_type,
            requires_grad=True)


register_python_type(StructuredBuffer,
                     BufferMarshall(),
                     lambda stream, x: stream.write(type(x.value).__name + "\n"))
