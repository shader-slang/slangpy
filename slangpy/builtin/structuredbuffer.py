# SPDX-License-Identifier: Apache-2.0

from slangpy.core.native import AccessType, NativeBufferMarshall

from slangpy.reflection import SlangProgramLayout, SlangType, StructuredBufferType, ByteAddressBufferType
from slangpy.backend import Buffer, ResourceUsage
from slangpy.bindings import (PYTHON_SIGNATURES, PYTHON_TYPES,
                              BindContext, BoundVariable,
                              CodeGenBlock)


class BufferMarshall(NativeBufferMarshall):

    def __init__(self, layout: SlangProgramLayout, usage: ResourceUsage):
        st = layout.find_type_by_name("StructuredBuffer<Unknown>")
        if st is None:
            raise ValueError(
                f"Could not find StructuredBuffer<Unknown> slang type. This usually indicates the slangpy module has not been imported.")

        super().__init__(st, usage)
        self.slang_type: SlangType

    def resolve_type(self, context: BindContext, bound_type: SlangType):
        if isinstance(bound_type, (StructuredBufferType, ByteAddressBufferType)):
            return bound_type
        else:
            raise ValueError(
                "Raw buffers can not be vectorized. If you need vectorized buffers, see the NDBuffer slangpy type")

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: SlangType):
        # structured buffer can only ever be taken to another structured buffer,
        if isinstance(vector_target_type, (StructuredBufferType, ByteAddressBufferType)):
            return 0
        else:
            raise ValueError(
                "Raw buffers can not be vectorized. If you need vectorized buffers, see the NDBuffer slangpy type")

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access[0]
        name = binding.variable_name
        assert access == AccessType.read

        if isinstance(binding.vector_type, StructuredBufferType):
            assert binding.vector_type.element_type is not None
            if binding.vector_type.writable:
                cgb.type_alias(
                    f"_t_{name}", f"RWStructuredBufferType<{binding.vector_type.element_type.full_name}>")
            else:
                cgb.type_alias(
                    f"_t_{name}", f"StructuredBufferType<{binding.vector_type.element_type.full_name}>")
        elif isinstance(binding.vector_type, ByteAddressBufferType):
            if binding.vector_type.writable:
                cgb.type_alias(
                    f"_t_{name}", f"RWByteAddressBufferType")
            else:
                cgb.type_alias(
                    f"_t_{name}", f"ByteAddressBufferType")
        else:
            raise ValueError(
                "Raw buffers can not be vectorized. If you need vectorized buffers, see the NDBuffer slangpy type")

    @property
    def is_writable(self) -> bool:
        return (self.usage & ResourceUsage.unordered_access) != 0

    def reduce_type(self, context: BindContext, dimensions: int) -> 'SlangType':
        if dimensions == 0:
            return self.slang_type
        raise ValueError("Cannot reduce dimensions of Buffer")


def _get_or_create_python_type(layout: SlangProgramLayout, value: Buffer):
    if isinstance(value, Buffer):
        usage = value.desc.usage
        return BufferMarshall(layout, usage)
    else:
        # Handle user trying to pass types like torch tensors in to a structured buffer arg
        return BufferMarshall(layout, ResourceUsage.shader_resource | ResourceUsage.unordered_access)


PYTHON_TYPES[Buffer] = _get_or_create_python_type

PYTHON_SIGNATURES[Buffer] = lambda x: f"[{x.desc.usage}]"
