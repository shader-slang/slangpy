from typing import Any, Optional
from slangpy.builtin.commonimports import *


class StructuredBufferType(BaseTypeImpl):

    def __init__(self, layout: kfr.SlangProgramLayout, usage: ResourceUsage):
        super().__init__(layout)
        st = layout.find_type_by_name("StructuredBuffer<Unknown>")
        if st is None:
            raise ValueError(
                f"Could not find StructuredBuffer<Unknown> slang type. This usually indicates the slangpy module has not been imported.")
        self.slang_type = st
        self.usage = usage

    def get_shape(self, value: Optional[Buffer] = None) -> Shape:
        if value is not None:
            return Shape(int(value.desc.size/value.desc.struct_size))
        else:
            return Shape(-1)

    def resolve_type(self, context: BindContext, bound_type: 'BaseType'):
        if isinstance(bound_type, (kfr.StructuredBufferType, kfr.ByteAddressBufferType)):
            return bound_type
        else:
            raise ValueError(
                "Raw buffers can not be vectorized. If you need vectorized buffers, see the NDBuffer slangpy type")

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: BaseType):
        # structured buffer can only ever be taken to another structured buffer,
        if isinstance(vector_target_type, (kfr.StructuredBufferType, kfr.ByteAddressBufferType)):
            return 0
        else:
            raise ValueError(
                "Raw buffers can not be vectorized. If you need vectorized buffers, see the NDBuffer slangpy type")

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access[0]
        name = binding.variable_name
        assert access == AccessType.read

        if isinstance(binding.vector_type, kfr.StructuredBufferType):
            if binding.vector_type.writable:
                cgb.type_alias(
                    f"_t_{name}", f"RWStructuredBufferType<{binding.vector_type.element_type.full_name}>")
            else:
                cgb.type_alias(
                    f"_t_{name}", f"StructuredBufferType<{binding.vector_type.element_type.full_name}>")
        elif isinstance(binding.vector_type, kfr.ByteAddressBufferType):
            if binding.vector_type.writable:
                cgb.type_alias(
                    f"_t_{name}", f"RWByteAddressBufferType")
            else:
                cgb.type_alias(
                    f"_t_{name}", f"ByteAddressBufferType")
        else:
            raise ValueError(
                "Raw buffers can not be vectorized. If you need vectorized buffers, see the NDBuffer slangpy type")

    # Call data just returns the primal

    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: Any) -> Any:
        access = binding.access
        if access[0] != AccessType.none:
            return {
                'value': data
            }

    # Buffers just return themselves for raw dispatch
    def create_dispatchdata(self, data: Any) -> Any:
        return data

    @property
    def is_writable(self) -> bool:
        return (self.usage & ResourceUsage.unordered_access) != 0


def _get_or_create_python_type(layout: kfr.SlangProgramLayout, value: Buffer):
    if isinstance(value, Buffer):
        usage = value.desc.usage
        return StructuredBufferType(layout, usage)
    else:
        # Handle user trying to pass types like torch tensors in to a structured buffer arg
        return StructuredBufferType(layout, ResourceUsage.shader_resource | ResourceUsage.unordered_access)


PYTHON_TYPES[Buffer] = _get_or_create_python_type

PYTHON_SIGNATURES[Buffer] = lambda x: f"[{x.desc.usage}]"
