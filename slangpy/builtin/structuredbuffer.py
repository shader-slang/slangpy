# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy.core.native import AccessType, NativeBufferMarshall

from slangpy.reflection import (
    SlangProgramLayout,
    SlangType,
    StructuredBufferType,
    ByteAddressBufferType,
    PointerType,
)
from slangpy import Buffer, BufferUsage
from slangpy.bindings import (
    PYTHON_SIGNATURES,
    PYTHON_TYPES,
    BindContext,
    BoundVariable,
    CodeGenBlock,
)


class BufferMarshall(NativeBufferMarshall):

    def __init__(self, layout: SlangProgramLayout, usage: BufferUsage):
        st = layout.find_type_by_name("StructuredBuffer<Unknown>")
        if st is None:
            raise ValueError(
                f"Could not find StructuredBuffer<Unknown> slang type. This usually indicates the slangpy module has not been imported."
            )

        super().__init__(st, usage)
        self.slang_type: SlangType

    def resolve_type(self, context: BindContext, bound_type: SlangType):
        if isinstance(bound_type, (StructuredBufferType, ByteAddressBufferType, PointerType)):
            return bound_type
        else:
            raise ValueError(
                "Raw buffers can not be vectorized. If you need vectorized buffers, see the NDBuffer slangpy type"
            )

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: SlangType,
    ):
        # structured buffer can only ever be taken to another structured buffer or a pointer
        if isinstance(vector_target_type, PointerType):
            return 0
        elif isinstance(vector_target_type, (StructuredBufferType, ByteAddressBufferType)):
            # For buffers, check for RW / ua mismatch
            has_ua = (self.usage & BufferUsage.unordered_access) != BufferUsage.none
            if vector_target_type.writable and not has_ua:
                raise ValueError(
                    f"Buffers bound to {vector_target_type.name} must have "
                    "the BufferUsage.unordered_access flag"
                )
            return 0
        else:
            raise ValueError(
                "Raw buffers can not be vectorized. If you need vectorized buffers, see the NDBuffer slangpy type"
            )

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: "BoundVariable"):
        access = binding.access[0]
        name = binding.variable_name
        assert access == AccessType.read

        if isinstance(binding.vector_type, StructuredBufferType):
            assert binding.vector_type.element_type is not None
            if binding.vector_type.writable:
                cgb.type_alias(
                    f"_t_{name}",
                    f"RWStructuredBufferType<{binding.vector_type.element_type.full_name}>",
                )
            else:
                cgb.type_alias(
                    f"_t_{name}",
                    f"StructuredBufferType<{binding.vector_type.element_type.full_name}>",
                )
        elif isinstance(binding.vector_type, ByteAddressBufferType):
            if binding.vector_type.writable:
                cgb.type_alias(f"_t_{name}", f"RWByteAddressBufferType")
            else:
                cgb.type_alias(f"_t_{name}", f"ByteAddressBufferType")
        elif isinstance(binding.vector_type, PointerType):
            # To bind as a pointer, use the 'ValueType', which just like the buffer wrappers
            # has a 'value' field that refers to the actual buffer (in this case as a pointer)
            cgb.type_alias(f"_t_{name}", f"ValueType<{binding.vector_type.full_name}>")
        else:
            raise ValueError(
                "Raw buffers can not be vectorized. If you need vectorized buffers, see the NDBuffer slangpy type"
            )

    @property
    def is_writable(self) -> bool:
        return (self.usage & BufferUsage.unordered_access) != 0

    def reduce_type(self, context: BindContext, dimensions: int) -> "SlangType":
        if dimensions == 0:
            return self.slang_type
        raise ValueError("Cannot reduce dimensions of Buffer")


def _get_or_create_python_type(layout: SlangProgramLayout, value: Buffer):
    if isinstance(value, Buffer):
        usage = value.desc.usage
        return BufferMarshall(layout, usage)
    else:
        # Handle user trying to pass types like torch tensors in to a structured buffer arg
        return BufferMarshall(layout, BufferUsage.shader_resource | BufferUsage.unordered_access)


PYTHON_TYPES[Buffer] = _get_or_create_python_type

PYTHON_SIGNATURES[Buffer] = lambda x: f"[{x.desc.usage}]"
