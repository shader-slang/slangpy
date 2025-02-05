# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable,
                              CodeGenBlock, Shape)
from slangpy.core.utils import is_type_castable_on_host
from slangpy.core.native import NativeObject
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection
from slangpy.reflection.reflectiontypes import ScalarType, VectorType


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
        if isinstance(bound_type, VectorType):
            if bound_type.num_elements > 3:
                raise ValueError(
                    f"Thread id argument must be a vector of size 1, 2 or 3. Got {bound_type.num_elements}.")
            if self.dims != -1 and self.dims != bound_type.num_elements:
                raise ValueError(
                    f"Thread id argument must be a vector of size {self.dims}. Got {bound_type.num_elements}.")
            resolved_type = context.layout.vector_type(
                TypeReflection.ScalarType.int32, bound_type.shape[0])
            if not is_type_castable_on_host(resolved_type, bound_type):
                raise ValueError(
                    f"Unable to convert thread id argument of type {resolved_type.full_name} to {bound_type.full_name}.")
            return bound_type
        elif isinstance(bound_type, ScalarType):
            if self.dims != -1 and self.dims != 1:
                raise ValueError(
                    f"Thread id argument must be a scalar or vector of size {self.dims}. Got {bound_type.full_name}.")
            resolved_type = context.layout.scalar_type(TypeReflection.ScalarType.int32)
            if not is_type_castable_on_host(resolved_type, bound_type):
                raise ValueError(
                    f"Unable to convert thread id argument of type {resolved_type.full_name} to {bound_type.full_name}.")
            return bound_type
        else:
            raise ValueError(
                f"Thread id argument must be a scalar or vector type. Got {bound_type.full_name}.")

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        # Thread id arg is generated for every thread and has no effect on call shape,
        # so it can just return a dimensionality of 0.
        return 0


PYTHON_TYPES[ThreadIdArg] = lambda l, x: ThreadIdArgMarshall(l, x.dims)
