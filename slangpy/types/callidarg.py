# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable,
                              CodeGenBlock)
from slangpy.experimental.gridarg import grid
from slangpy.reflection import SlangProgramLayout, SlangType


class CallIdArg:
    """
    Passes the thread id as an argument to a SlangPy function.
    """

    def __init__(self, dims: int = -1):
        super().__init__()
        if isinstance(dims, tuple):
            raise ValueError(
                "Using call id argument with a tuple is deprecated. To specify a shape, use the 'grid' argument type instead of 'call_id'")
        self.dims = dims

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


def call_id(dims: int = -1):
    """
    Create a ThreadIdArg to pass to a SlangPy function, which passes the thread id.
    """
    if isinstance(dims, tuple):
        return grid(shape=dims)
    return CallIdArg(dims)


class CallIdArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, dims: int):
        super().__init__(layout)
        self.dims = dims
        st = layout.find_type_by_name(f"CallIdArg")
        if st is None:
            raise ValueError(
                f"Could not find CallIdArg slang type. This usually indicates the threadidarg module has not been imported.")
        self.slang_type = st
        self.match_call_shape = True

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        # Resolve type using reflection.
        conv_type = bound_type.program.find_type_by_name(
            f"VectorizeCallidArgTo<{bound_type.full_name}, {self.dims}>.VectorType")
        if conv_type is None:
            raise ValueError(
                f"Could not find suitable conversion from CallIdArg<{self.dims}> to {bound_type.full_name}")
        return conv_type

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        # Thread id arg is generated for every thread and has no effect on call shape,
        # so it can just return a dimensionality of 0.
        return -1


PYTHON_TYPES[CallIdArg] = lambda l, x: CallIdArgMarshall(l, x.dims)
