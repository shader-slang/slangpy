# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any

from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable, BoundVariableRuntime, CallContext,
                              CodeGenBlock, Shape)
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection
from slangpy.reflection.reflectiontypes import VectorType
from slangpy.types.helpers import resolve_vector_generator_type


class WangHashArg:
    """
    Generates a random int/vector per thread when passed as an argument using a wang
    hash of the thread id.
    """

    def __init__(self, dims: int = -1, seed: int = 0):
        super().__init__()
        self.dims = dims
        self.seed = seed

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


def wang_hash(dim: int = -1, seed: int = 0):
    """
    Create a WangHashArg to pass to a SlangPy function, which generates a
    random int/vector per thread using a wang hash of the thread id.
    Specify dims to enforce a vector size (uint1/2/3). If unspecified this will be
    inferred from the function argument.
    """
    return WangHashArg(dim, seed)


class WangHashArgMarshall(Marshall):

    def __init__(self, layout: SlangProgramLayout, dims: int):
        super().__init__(layout)
        self.dims = dims

        # Find slang type
        st = layout.find_type_by_name(f"WangHashArg")
        if st is None:
            raise ValueError(
                f"Could not find WangHashArg slang type. This usually indicates the wanghasharg module has not been imported.")
        self.slang_type = st
        self.concrete_shape = Shape(dims)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: WangHashArg) -> Any:
        access = binding.access
        if access[0] == AccessType.read:
            return {
                'seed': data.seed
            }

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        # Wang hash arg is valid to pass to vector or scalar integer types.
        return resolve_vector_generator_type(context, bound_type, self.dims, TypeReflection.ScalarType.int32)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: SlangType):
        # Wang hash arg is generated for every thread and has no effect on call shape,
        # so it can just return a dimensionality of 0.
        return 0


PYTHON_TYPES[WangHashArg] = lambda l, x: WangHashArgMarshall(l, x.dims)
