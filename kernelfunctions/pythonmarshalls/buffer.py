from typing import Any, Optional
from sgl import Device, TypeLayoutReflection
from kernelfunctions.buffer import StructuredBuffer
from kernelfunctions.shapes import TConcreteShape
from kernelfunctions.typeregistry import create_slang_type_marshal, get_python_type_marshall, register_python_type
from kernelfunctions.types import AccessType, PythonMarshal


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

    def get_calldata_typename(self, typename: str, shape: TConcreteShape, access: AccessType):
        if access == AccessType.read:
            return f"TensorBuffer<{typename},{len(shape)}>"
        else:
            return f"RWTensorBuffer<{typename},{len(shape)}>"

    def get_indexer(self, transform: list[Optional[int]], access: AccessType):
        vals = ",".join(("0" if x is None else f"call_id[{x}]") for x in transform)
        return f"[{{{vals}}}]"

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


register_python_type(StructuredBuffer,
                     BufferMarshall(),
                     lambda stream, x: stream.write(type(x.value).__name + "\n"))
