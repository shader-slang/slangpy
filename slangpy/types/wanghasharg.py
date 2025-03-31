# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any

import numpy as np
import numpy.typing as npt

from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable, BoundVariableRuntime, CallContext,
                              CodeGenBlock, Shape)
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection
from slangpy.types.helpers import resolve_vector_generator_type


def calc_wang_hash_numpy(seed: npt.NDArray[Any]) -> npt.NDArray[Any]:
    seed = (seed ^ 61) ^ (seed >> 16)
    seed *= 9
    seed = seed ^ (seed >> 4)
    seed *= 0x27d4eb2d
    seed = seed ^ (seed >> 15)
    return seed


def calc_wang_hash(seed: int):
    return int(calc_wang_hash_numpy(np.array([seed], dtype=np.uint32))[0])


class WangHashArg:
    """
    Generates a random int/vector per thread when passed as an argument using a wang
    hash of the thread id.
    """

    def __init__(self, dims: int = -1, seed: int = 0, warmup: int = 0, hash_seed: bool = True):
        super().__init__()
        self.dims = dims
        self.seed = seed
        self.warmup = int(warmup)
        self.hash_seed = bool(hash_seed)

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims},{self.warmup}]"


def wang_hash(dim: int = -1, seed: int = 2640457667, warmup: int = 0, hash_seed: bool = True):
    """
    Create a WangHashArg to pass to a SlangPy function, which generates a
    random int/vector per thread using a wang hash of the thread id.
    Specify dims to enforce a vector size (uint1/2/3). If unspecified this will be
    inferred from the function argument.

    Warmup will result in multiple per-thread warmup iterations gpu side, to increase
    quality of random hash at expense of performance.

    Hash seed is a CPU side option to hash the seed value, reducing correlation through
    using sequential seeds (eg the frame number.)
    """
    return WangHashArg(dim, seed, warmup, hash_seed)


class WangHashArgMarshall(Marshall):

    def __init__(self, layout: SlangProgramLayout, dims: int, warmup: int):
        super().__init__(layout)
        self.dims = dims
        self.warmup = warmup

        # Find slang type
        st = layout.find_type_by_name(f"WangHashArg<{self.warmup}>")
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
        seed = data.seed
        if data.hash_seed:
            seed = calc_wang_hash(seed)
        if access[0] == AccessType.read:
            return {
                'seed': seed
            }

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        # Wang hash arg is valid to pass to vector or scalar integer types.
        return resolve_vector_generator_type(context, bound_type, self.dims, TypeReflection.ScalarType.int32, max_dims=3)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: SlangType):
        # Wang hash arg is generated for every thread and has no effect on call shape,
        # so it can just return a dimensionality of 0.
        return 0


PYTHON_TYPES[WangHashArg] = lambda l, x: WangHashArgMarshall(l, x.dims, x.warmup)
