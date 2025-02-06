# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any

from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable, BoundVariableRuntime, CallContext,
                              CodeGenBlock, Shape)
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection
from slangpy.types.helpers import resolve_vector_generator_type


class RandFloatArg:
    """
    Generates a random float/vector per thread when passed as an argument
    to a SlangPy function. The min and max values are inclusive.
    """

    def __init__(self, min: float, max: float, dim: int = -1, seed: int = 0):
        super().__init__()
        self.seed = seed
        self.min = float(min)
        self.max = float(max)
        self.dims = int(dim)

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


def rand_float(min: float = 0, max: float = 1, dim: int = -1, seed: int = 0):
    """
    Create a RandFloatArg to pass to a SlangPy function, which generates a 
    random float/vector per thread. The min and max values are inclusive.
    Dim enforces a requirement for a specific vector size (float1/2/3). If
    unspecified this will be inferred from the function argument.
    """
    return RandFloatArg(min, max, dim, seed)


class RandFloatArgMarshall(Marshall):

    def __init__(self, layout: SlangProgramLayout, dim: int):
        super().__init__(layout)
        self.dims = dim
        st = layout.find_type_by_name(f"RandFloatArg")
        if st is None:
            raise ValueError(
                f"Could not find RandFloatArg slang type. This usually indicates the randfloatarg module has not been imported.")
        self.slang_type = st
        self.concrete_shape = Shape(dim)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            # cgb.add_import("randfloatarg")
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: RandFloatArg) -> Any:
        access = binding.access
        if access[0] == AccessType.read:
            return {
                'seed': data.seed,
                'min': data.min,
                'max': data.max
            }

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        # Wang hash arg is valid to pass to vector or scalar integer types.
        return resolve_vector_generator_type(context, bound_type, self.dims, TypeReflection.ScalarType.float32)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        # Rand float arg is generated for every thread and has no effect on call shape,
        # so it can just return a dimensionality of 0.
        return 0


PYTHON_TYPES[RandFloatArg] = lambda l, x: RandFloatArgMarshall(l, x.dims)
