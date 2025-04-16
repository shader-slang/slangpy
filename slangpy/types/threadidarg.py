# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable,
                              CodeGenBlock, Shape)
from slangpy.core.native import NativeObject
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection
from slangpy.types.helpers import resolve_vector_generator_type


class ThreadIdArg(NativeObject):
    """
    Passes the thread id as an argument to a SlangPy function.
    """

    def __init__(self, dims: int = -1):
        super().__init__()
        self.dims = dims
        self.slangpy_signature = f"[{self.dims}]"


def thread_id(dims: int = -1):
    """
    Create a ThreadIdArg to pass to a SlangPy function, which passes the thread id.
    Specify dims to enforce a vector size (uint1/2/3). If unspecified this will be
    inferred from the function argument.
    """
    return ThreadIdArg(dims)


class ThreadIdArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, dims: int):
        super().__init__(layout)
        self.dims = dims
        st = layout.find_type_by_name(f"ThreadIdArg")
        if st is None:
            raise ValueError(
                f"Could not find ThreadIdArg slang type. This usually indicates the threadidarg module has not been imported.")
        self.slang_type = st
        self.concrete_shape = Shape(dims)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        # Thread id arg is valid to pass to vector or scalar integer types.
        return resolve_vector_generator_type(context, bound_type, self.dims, TypeReflection.ScalarType.int32, max_dims=3)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        # Thread id arg is generated for every thread and has no effect on call shape,
        # so it can just return a dimensionality of 0.
        return 0


PYTHON_TYPES[ThreadIdArg] = lambda l, x: ThreadIdArgMarshall(l, x.dims)
