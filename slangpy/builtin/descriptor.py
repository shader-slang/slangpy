# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, cast

from slangpy.core.native import AccessType, NativeDescriptorMarshall, unpack_arg, Shape

import slangpy
from slangpy import TypeReflection, ShaderCursor
import slangpy.reflection as kfr
from slangpy import math
from slangpy.bindings import (
    PYTHON_SIGNATURES,
    PYTHON_TYPES,
    BindContext,
    BoundVariable,
    CodeGenBlock,
)
from slangpy import DescriptorHandle, DescriptorHandleType


class DescriptorMarshall(NativeDescriptorMarshall):
    def __init__(self, layout: kfr.SlangProgramLayout, type: DescriptorHandleType):
        st = layout.find_type_by_name("DescriptorHandle<StructuredBuffer<Unknown>>")
        if st is None:
            raise ValueError(
                f"Could not find DescriptorHandle<StructuredBuffer<Unknown>> slang type. "
                "This usually indicates the slangpy module has not been imported."
            )

        super().__init__(st, type)

        self.concrete_shape = Shape()
        self.slang_type: kfr.SlangType

    def resolve_type(self, context: BindContext, bound_type: kfr.SlangType):
        return bound_type

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: "BoundVariable",
        vector_target_type: kfr.SlangType,
    ):
        return 0

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: "BoundVariable"):
        name = "(None)" if self.slang_type is None else self.slang_type.full_name
        vec_name = "(None)" if binding.vector_type is None else binding.vector_type.full_name
        access = binding.access
        name = binding.variable_name
        if access[0] in [AccessType.read, AccessType.readwrite]:
            assert binding.vector_type is not None
            cgb.type_alias(
                f"_t_{name}",
                binding.vector_type.full_name.replace("DescriptorHandle", "DescriptorType"),
            )
        else:
            cgb.type_alias(f"_t_{name}", f"NoneType")

    def reduce_type(self, context: BindContext, dimensions: int) -> kfr.SlangType:
        if dimensions == 0:
            return self.slang_type
        raise ValueError("Cannot reduce dimensions of Descriptor")

    def build_shader_object(self, context: "BindContext", data: DescriptorHandle) -> "ShaderObject":
        so = context.device.create_shader_object(self.slang_type.uniform_layout.reflection)
        cursor = ShaderCursor(so)
        cursor.write(data)
        return so


PYTHON_TYPES[DescriptorHandle] = lambda layout, handle: DescriptorMarshall(layout, handle.type)

PYTHON_SIGNATURES[DescriptorHandle] = lambda handle: f"[{handle.type}]"
