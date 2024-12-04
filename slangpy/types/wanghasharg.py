# SPDX-License-Identifier: Apache-2.0
from typing import Any

from slangpy.backend import TypeReflection
from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable, BoundVariableRuntime, CallContext,
                              CodeGenBlock, Shape)
from slangpy.reflection import SlangProgramLayout, SlangType


class WangHashArg:
    """
    Request random uints using a wang hash function. eg
    void myfunc(uint3 input) { }
    """

    def __init__(self, dims: int = 3, seed: int = 0):
        super().__init__()
        self.dims = dims
        self.seed = seed

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


class WangHashArgMarshall(Marshall):
    """
    Slangpy type extension for WangHashArg
    """

    def __init__(self, layout: SlangProgramLayout, dims: int):
        super().__init__(layout)
        self.dims = dims
        st = layout.find_type_by_name(f"WangHashArg<{self.dims}>")
        if st is None:
            raise ValueError(
                f"Could not find WangHashArg slang type. This usually indicates the wanghasharg module has not been imported.")
        self.slang_type = st
        self.concrete_shape = Shape(self.dims)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.add_import("wanghasharg")
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: WangHashArg) -> Any:
        access = binding.access
        if access[0] == AccessType.read:
            return {
                'seed': data.seed
            }

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        return context.layout.vector_type(TypeReflection.ScalarType.uint32, self.dims)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: SlangType):
        return 1 - len(vector_target_type.shape)


PYTHON_TYPES[WangHashArg] = lambda l, x: WangHashArgMarshall(l, x.dims)
