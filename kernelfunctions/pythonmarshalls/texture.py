from typing import Any, Optional

from sgl import FormatType, ResourceUsage, Texture, get_format_info

from kernelfunctions.backend import Device, TypeLayoutReflection
from kernelfunctions.typeregistry import create_slang_type_marshal, get_python_type_marshall, register_python_type
from kernelfunctions.types import AccessType, PythonMarshal
import kernelfunctions.codegen as cg
from kernelfunctions.types.enums import PrimType
from kernelfunctions.types.pythonmarshall import PythonDescriptor


class TextureMarshal(PythonMarshal):
    """
    Base class for marshalling buffer types.
    """

    def __init__(self, python_type: type[Texture]):
        super().__init__(python_type)

    def get_element_shape(self, value: Texture):
        fi = get_format_info(value.format)
        return (fi.channel_count,)

    def get_container_shape(self, value: Texture):
        return (value.width, value.height)

    def get_element_type(self, value: Texture):
        fi = get_format_info(value.format)
        return fi.type
        return fi.type

    def is_writable(self, value: Texture) -> bool:
        return (value.desc.usage & ResourceUsage.unordered_access) != 0

    def gen_calldata(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, type_name: str, variable_name: str, access: AccessType):
        """
        Call data either contains a read-only or read-write buffer.
        """
        assert desc.container_shape is not None
        if access == AccessType.read:
            cgb.declare(
                f"Texture<{type_name},{len(desc.container_shape)}>", variable_name)
        else:
            cgb.declare(
                f"RWTexture<{type_name},{len(desc.container_shape)}>", variable_name)

    def _transform_to_subscript(self, transform: list[Optional[int]]):
        """
        Generates the subscript to be passed into the [] operator when loading or storing
        from the buffer.
        """
        vals = ",".join(
            ("0" if x is None else f"context.call_id[{x}]") for x in transform)
        return f"[{{{vals}}}]"

    def gen_load(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, from_call_data: str, to_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Load the value from the buffer into the variable.
        """
        cgb.assign(
            to_variable, f"{from_call_data}{self._transform_to_subscript(transform)}")

    def gen_store(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, to_call_data: str, from_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Store the value from the variable into the buffer.
        """
        cgb.assign(
            f"{to_call_data}{self._transform_to_subscript(transform)}", from_variable)

    def create_calldata(self, device: Device, value: Texture, access: AccessType, prim: PrimType):
        if prim == PrimType.primal:
            return {
                "buffer": value.buffer,
                "strides": list(value.strides),
            }
        else:
            raise NotImplementedError()

    def read_calldata(self, device: Device, call_data: Any, access: AccessType, prim: PrimType, value: Texture):
        if prim == PrimType.primal:
            assert call_data['buffer'] == value.buffer
            assert call_data['strides'] == list(value.strides)
        else:
            raise NotImplementedError()

    def allocate_return_value(self, device: Device, call_shape: list[int], element_type: Any):
        return Texture(
            device=device,
            shape=tuple(call_shape),
            element_type=element_type)


register_python_type(Texture,
                     TextureMarshall(),
                     lambda stream, x: stream.write(type(x.value).__name + "\n"))
